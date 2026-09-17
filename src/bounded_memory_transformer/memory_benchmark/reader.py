"""Train only a reader: admission, updating, eviction and retrieval stay symbolic."""

import random
import time

import torch

from bounded_memory_transformer.tiny_transformer import TinyTransformerLM, TransformerConfig

from .generator import symbol_space
from .operations import Kind, Operation
from .policies import POLICIES, MemoryPolicy
from .views import QueryView, read_visible

ALPHABET = "#0123456789abcdSUDNMCQ=;?"
TOKEN_IDS = {character: index for index, character in enumerate(ALPHABET)}
OP_CHAR = {Kind.SET: "S", Kind.UPDATE: "U", Kind.DELETE: "D", Kind.NOISE: "N"}


def render_operations(operations: tuple[Operation, ...]) -> str:
    return "".join(
        f"{OP_CHAR[o.kind]}{o.entity:02d}{'abcd'[o.attribute]}"
        + ("??" if o.value is None else f"{o.value:02d}")
        + ";"
        for o in operations
    )


def render(view: QueryView) -> str:
    return (
        "M"
        + render_operations(view.memory)
        + "C"
        + render_operations(view.current)
        + f"Q{view.query.entity:02d}{'abcd'[view.query.attribute]}="
    )


def build_batch(examples: list[tuple[str, str]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(prompt) + 1 for prompt, _ in examples)
    inputs = torch.full((len(examples), width), TOKEN_IDS["#"], dtype=torch.long)
    targets = torch.full_like(inputs, -100)
    for index, (prompt, answer) in enumerate(examples):
        if len(answer) != 2:
            raise ValueError("answers must contain exactly two characters")
        ids = torch.tensor([TOKEN_IDS[c] for c in prompt + answer], dtype=torch.long)
        inputs[index, : len(prompt) + 1] = ids[:-1]
        targets[index, len(prompt) - 1 : len(prompt) + 1] = ids[-2:]
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def predict(
    model: TinyTransformerLM, prompts: list[str], *, batch_size: int, device: str
) -> list[str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    answers = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            width = max(map(len, batch)) + 1
            if width > model.config.context_length:
                raise ValueError("visible prompt exceeds context; no silent truncation is allowed")
            inputs = torch.full((len(batch), width), TOKEN_IDS["#"], dtype=torch.long)
            for index, prompt in enumerate(batch):
                inputs[index, : len(prompt)] = torch.tensor([TOKEN_IDS[c] for c in prompt])
            inputs = inputs.to(device)
            lengths = torch.tensor([len(prompt) for prompt in batch], device=device)
            indices = torch.arange(len(batch), device=device)
            first = model(inputs).logits[indices, lengths - 1].argmax(dim=-1)
            inputs[indices, lengths] = first
            second = model(inputs).logits[indices, lengths].argmax(dim=-1)
            answers.extend(
                ALPHABET[a] + ALPHABET[b]
                for a, b in zip(first.tolist(), second.tolist(), strict=True)
            )
    finally:
        model.train(was_training)
    return answers


def make_training_views(split: str, *, seed: int, count: int, capacity: int) -> list[QueryView]:
    """Independent reading microtasks, including conflicts and near-key rejection.

    No labels supervise memory actions. Complete-symbol splits are the same as
    for episodes. Training balances available evidence rather than mostly teaching
    UNKNOWN on uniformly forgotten long histories.
    """
    rng = random.Random(seed)
    symbols = symbol_space(split)
    views = []
    for index in range(count):
        query = Operation(Kind.ASK, rng.choice(symbols.entities), rng.randrange(4))
        peers = [e for e in symbols.entities if e != query.entity]
        policy = MemoryPolicy(POLICIES[index % len(POLICIES)], capacity)
        events = [
            Operation(Kind.SET, rng.choice(peers), rng.randrange(4), rng.choice(symbols.values))
            for _ in range(rng.randrange(capacity + 1))
        ]
        if rng.random() < 0.7:
            events.insert(
                rng.randrange(len(events) + 1),
                Operation(Kind.SET, *query.key, rng.choice(symbols.values)),
            )
            if rng.random() < 0.35:
                kind = rng.choice((Kind.UPDATE, Kind.DELETE))
                value = None if kind == Kind.DELETE else rng.choice(symbols.values)
                events.append(Operation(kind, *query.key, value))
        for operation in events:
            policy.observe(operation)
        current = []
        if rng.random() < 0.5:
            current.append(
                Operation(
                    rng.choice((Kind.SET, Kind.NOISE)),
                    rng.choice(peers),
                    rng.randrange(4),
                    rng.choice(symbols.values),
                )
            )
        kind = rng.choices(
            (None, Kind.SET, Kind.UPDATE, Kind.DELETE, Kind.NOISE),
            weights=(0.4, 0.15, 0.2, 0.15, 0.1),
        )[0]
        if kind is not None:
            current.append(
                Operation(
                    kind, *query.key, None if kind == Kind.DELETE else rng.choice(symbols.values)
                )
            )
        views.append(
            QueryView(
                policy.retrieve(query), tuple(current), query, policy.state.nbytes, policy.records()
            )
        )
    return views


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def train_reader(config: dict, seed: int, device: str) -> tuple[TinyTransformerLM, dict]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model_config = TransformerConfig(
        vocab_size=len(ALPHABET),
        context_length=config["context_length"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=0.0,
    )
    model = TinyTransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    training = make_training_views(
        "train", seed=seed + 1000, count=config["train_examples"], capacity=config["capacity"]
    )
    validation = make_training_views(
        "validation",
        seed=20260917,
        count=config["validation_examples"],
        capacity=config["capacity"],
    )
    examples = [(render(v), read_visible(v.memory, v.current, v.query)) for v in training]
    validation_prompts = [render(v) for v in validation]
    validation_targets = [read_visible(v.memory, v.current, v.query) for v in validation]
    if max(len(p) + 1 for p, _ in examples) > model_config.context_length:
        raise ValueError("training example exceeds context length")
    history = []
    synchronize(device)
    started = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        model.train()
        batch = [rng.choice(examples) for _ in range(config["batch_size"])]
        inputs, targets = build_batch(batch, device)
        loss = model(inputs, targets).loss
        assert loss is not None
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            answers = predict(
                model, validation_prompts, batch_size=config["batch_size"], device=device
            )
            accuracy = sum(a == b for a, b in zip(answers, validation_targets, strict=True)) / len(
                answers
            )
            entry = {"step": step, "train_loss": loss.item(), "validation_accuracy": accuracy}
            history.append(entry)
            print(
                f"seed={seed} step={step:04d} loss={loss.item():.4f} "
                f"validation_accuracy={accuracy:.3%}",
                flush=True,
            )
    synchronize(device)
    return model.eval(), {
        "seed": seed,
        "parameters": model.parameter_count(),
        "training_seconds_including_validation": time.perf_counter() - started,
        "history": history,
        "validation_accuracy": history[-1]["validation_accuracy"],
    }
