"""Train/save four reader arms with genuinely shared selector and generator weights."""

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from bounded_memory_transformer.memory_benchmark.reader import (
    ALPHABET,
    build_batch,
    predict,
    render,
    synchronize,
)
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.tiny_transformer import TinyTransformerLM, TransformerConfig

from .reader_cases import (
    STRATA,
    ReaderTask,
    copy_selected,
    generate_reader_tasks,
    oracle_select,
    project_selected,
)
from .selection import RecordSelector, encode_selection

SYSTEMS = ("character_full", "selected_character", "selected_copy", "oracle_copy")


def _accuracy(answers: list[str], targets: list[str], indices: list[int]) -> float | None:
    return sum(answers[i] == targets[i] for i in indices) / len(indices) if indices else None


def validation_metrics(tasks: list[ReaderTask], result: dict) -> dict:
    targets = [task.target for task in tasks]
    subsets = {
        "overall": list(range(len(tasks))),
        "known": [i for i, t in enumerate(tasks) if t.known],
        "unknown": [i for i, t in enumerate(tasks) if not t.known],
        "full_occupancy": [i for i, t in enumerate(tasks) if t.occupancy == 4],
    }
    metrics = {}
    for name, answers in result["answers"].items():
        metrics[name] = {
            label: _accuracy(answers, targets, indices) for label, indices in subsets.items()
        }
        metrics[name]["strata"] = {
            stratum: _accuracy(
                answers, targets, [i for i, t in enumerate(tasks) if t.stratum == stratum]
            )
            for stratum in STRATA
        }
        metrics[name]["counts"] = {label: len(indices) for label, indices in subsets.items()}
    metrics["selector_index_accuracy"] = sum(
        a == b for a, b in zip(result["selector_indices"], result["oracle_indices"], strict=True)
    ) / len(tasks)
    return metrics


@dataclass
class ReaderBundle:
    selector: RecordSelector
    character: TinyTransformerLM
    config: dict
    metadata: dict
    device: str = "cpu"

    def predict(self, views: list[QueryView]) -> dict:
        batch_size = self.config["batch_size"]
        indices = self.selector.select(views, batch_size)
        oracle_indices = [oracle_select(v) for v in views]
        copied = [copy_selected(v, i) for v, i in zip(views, indices, strict=True)]
        oracle = [copy_selected(v, i) for v, i in zip(views, oracle_indices, strict=True)]
        assert oracle == [read_visible(v.memory, v.current, v.query) for v in views]
        full = predict(
            self.character, [render(v) for v in views], batch_size=batch_size, device=self.device
        )
        selected = predict(
            self.character,
            [render(project_selected(v, i)) for v, i in zip(views, indices, strict=True)],
            batch_size=batch_size,
            device=self.device,
        )
        return {
            "answers": dict(zip(SYSTEMS, (full, selected, copied, oracle), strict=True)),
            "selector_indices": indices,
            "oracle_indices": oracle_indices,
            "metadata": {
                "character_weights_shared": True,
                "selector_decisions_shared": True,
                "oracle_visible_only": True,
            },
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "selector_settings": self.selector.settings,
                "selector_state": self.selector.state_dict(),
                "character_settings": asdict(self.character.config),
                "character_state": self.character.state_dict(),
                "config": self.config,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ReaderBundle":
        saved = torch.load(path, map_location=device, weights_only=True)
        selector = RecordSelector(**saved["selector_settings"]).to(device)
        selector.load_state_dict(saved["selector_state"])
        character = TinyTransformerLM(TransformerConfig(**saved["character_settings"])).to(device)
        character.load_state_dict(saved["character_state"])
        return cls(selector.eval(), character.eval(), saved["config"], saved["metadata"], device)


def train_readers(config: dict, seed: int, device: str) -> ReaderBundle:
    """Train on train symbols only; log validation without selecting a best checkpoint.

    The character model sees equal-probability full and oracle-projected prompts.
    Both training pools also include paired empty-memory views. Selector labels
    are visible oracle indices, and character labels are visible answer tokens.
    """
    config = dict(config)
    for key in ("train_examples", "validation_examples", "batch_size", "log_every"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("selector_steps", "character_steps"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    selector = RecordSelector(
        config.get("selector_d_model", config["d_model"]),
        config["n_heads"],
        config.get("selector_n_layers", 1),
    ).to(device)
    character = TinyTransformerLM(
        TransformerConfig(
            len(ALPHABET),
            config["context_length"],
            config["d_model"],
            config["n_heads"],
            config["n_layers"],
            dropout=0.0,
        )
    ).to(device)
    training = generate_reader_tasks(
        "train", seed=seed + 1000, count=config["train_examples"], capacity=config["capacity"]
    )
    validation = generate_reader_tasks(
        "validation",
        seed=20260920,
        count=config["validation_examples"],
        capacity=config["capacity"],
    )
    views = [view for task in training for view in (task.view, task.empty_view)]
    # Oracle projection is a supervised engineering control, disclosed below.
    examples = []
    for view in views:
        index = oracle_select(view)
        answer = read_visible(view.memory, view.current, view.query)
        examples.append(((render(view), answer), (render(project_selected(view, index)), answer)))
    if max(len(prompt) + 1 for pair in examples for prompt, _ in pair) > config["context_length"]:
        raise ValueError("training evidence exceeds character context length")
    selector_history, character_history = [], []
    metadata = {
        "seed": seed,
        "supervision": "visible oracle indices and visible answer tokens",
        "character_prompt_mix": "50% full, 50% oracle-projected; paired empty views included",
        "selector_features": "joint query/record characters, origin, visible absolute event order",
        "key_equality_features": False,
        "train_split": "train",
        "validation_split": "validation",
        "selector_parameters": sum(p.numel() for p in selector.parameters()),
        "character_parameters": character.parameter_count(),
        "selector_history": selector_history,
        "character_history": character_history,
    }
    bundle = ReaderBundle(selector, character, config, metadata, device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=config["learning_rate"])
    synchronize(device)
    started = time.perf_counter()
    for step in range(1, config["selector_steps"] + 1):
        selector.train()
        sampled = [rng.choice(views) for _ in range(config["batch_size"])]
        batch = encode_selection(sampled, device)
        scores = selector(batch)
        labels = [oracle_select(v) for v in sampled]
        targets = torch.tensor(
            [i if i >= 0 else scores.shape[1] - 1 for i in labels], device=device
        )
        loss = F.cross_entropy(scores, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite selector loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["selector_steps"]:
            val_views = [t.view for t in validation]
            indices = selector.select(val_views, config["batch_size"])
            answers = [copy_selected(v, i) for v, i in zip(val_views, indices, strict=True)]
            val_result = {
                "answers": {"selected_copy": answers},
                "selector_indices": indices,
                "oracle_indices": [t.oracle_index for t in validation],
            }
            entry = {
                "step": step,
                "train_loss": loss.item(),
                "validation": validation_metrics(validation, val_result),
            }
            selector_history.append(entry)
            print(
                f"reader selector seed={seed} step={step} loss={loss.item():.4f} "
                f"val={entry['validation']['selected_copy']['overall']:.4f}",
                flush=True,
            )
    synchronize(device)
    metadata["selector_training_seconds_including_validation"] = time.perf_counter() - started
    optimizer = torch.optim.AdamW(character.parameters(), lr=config["learning_rate"])
    started = time.perf_counter()
    for step in range(1, config["character_steps"] + 1):
        character.train()
        sampled = [rng.choice(examples)[rng.randrange(2)] for _ in range(config["batch_size"])]
        inputs, targets = build_batch(sampled, device)
        loss = character(inputs, targets).loss
        assert loss is not None
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite character loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(character.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["character_steps"]:
            result = bundle.predict([t.view for t in validation])
            entry = {
                "step": step,
                "train_loss": loss.item(),
                "validation": validation_metrics(validation, result),
            }
            character_history.append(entry)
            print(
                f"reader character seed={seed} step={step} loss={loss.item():.4f} "
                f"val={entry['validation']['character_full']['overall']:.4f}",
                flush=True,
            )
    synchronize(device)
    metadata["character_training_seconds_including_validation"] = time.perf_counter() - started
    metadata["validation"] = character_history[-1]["validation"]
    selector.eval()
    character.eval()
    return bundle
