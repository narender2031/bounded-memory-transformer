"""Train/validation-only development CLI. No test split is reachable here."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.reader import synchronize
from bounded_memory_transformer.memory_experiments.metrics import recovery
from bounded_memory_transformer.memory_experiments.reader_cases import (
    copy_selected,
    generate_reader_tasks,
)

from .reader import FactorizedReader, supervised_loss
from .tasks import varied_tasks


def freeze_comparator(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in ("authority.weight", "order_weight"))


def pretrain_comparator(model, steps, learning_rate):
    """Learn the finite shared alphabet's equality relation before composition.

    All symbols come from TRAIN keys. Full entity IDs are never combined here.
    This exhaustive supervised primitive is an explicitly disclosed aid; it is
    not a claim to generalize to unseen characters or learn unsupervised binding.
    """
    symbols = symbol_space("train")
    digits = sorted({digit for e in symbols.entities for digit in (e // 10, e % 10)})
    pairs = [
        (a, b) for alphabet in (digits, list(range(10, 14))) for a in alphabet for b in alphabet
    ]
    device = next(model.parameters()).device
    tokens = torch.tensor([[a, b, 14] for a, b in pairs], device=device)
    target = torch.tensor([a == b for a, b in pairs], dtype=torch.float32, device=device)
    positive = target.bool()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for step in range(1, steps + 1):
        logits = model.pair_logits(tokens)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss = 0.5 * (losses[positive].mean() + losses[~positive].mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == steps:
            with torch.no_grad():
                logits = model.pair_logits(tokens)
                history.append(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        errors=int(((logits > 0) != positive).sum()),
                        minimum_signed_margin=float((logits * (2 * target - 1)).min()),
                    )
                )
            print(json.dumps(dict(comparator=history[-1])), flush=True)
    freeze_comparator(model)
    return dict(
        pairs=len(pairs),
        alphabet=digits + list(range(10, 14)),
        history=history,
        frozen_after_pretraining=True,
    )


def task_metrics(model, tasks, batch_size=128):
    indices = model.select([t.view for t in tasks], batch_size)
    answers = [copy_selected(t.view, i) for t, i in zip(tasks, indices, strict=True)]
    accuracy = sum(a == t.target for a, t in zip(answers, tasks, strict=True)) / len(tasks)
    empty = sum(t.empty_target == t.target for t in tasks) / len(tasks)
    categories = {
        s: [i for i, t in enumerate(tasks) if t.stratum == s]
        for s in sorted({t.stratum for t in tasks})
    }
    categories.update(
        known=[i for i, t in enumerate(tasks) if t.known],
        unknown=[i for i, t in enumerate(tasks) if not t.known],
    )
    categories.update(
        {
            f"occupancy_{n}": [i for i, t in enumerate(tasks) if t.occupancy == n]
            for n in sorted({t.occupancy for t in tasks})
        }
    )
    slices = {
        name: dict(
            count=len(ids), accuracy=sum(answers[i] == tasks[i].target for i in ids) / len(ids)
        )
        for name, ids in categories.items()
        if ids
    }
    relative = recovery(accuracy, 1.0, empty)
    return dict(
        accuracy=accuracy,
        slices=slices,
        recovery=relative,
        absolute_gate=accuracy >= 0.95 and all(v["accuracy"] >= 0.95 for v in slices.values()),
        relative_gate=relative["value"] is not None and relative["value"] >= 0.95,
    )


def train(config, seed, output):
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    rng = random.Random(seed)
    device = config["device"]
    model = FactorizedReader(**config["model"]).to(device)
    training = generate_reader_tasks(
        "train", seed=seed + 10000, count=config["original_examples"], capacity=4
    )
    for capacity in (4, 8):
        for current in config["training_current_counts"]:
            training += varied_tasks(
                "train",
                seed=seed * 1000 + capacity * 100 + current,
                count=config["varied_examples_per_condition"],
                capacity=capacity,
                current_count=current,
            )
    views = [v for t in training for v in (t.view, t.empty_view)]
    validation = {
        f"original_k{k}": generate_reader_tasks(
            "validation", seed=20260922, count=config["validation_examples"], capacity=k
        )
        for k in (4, 8)
    }
    for k in (4, 8):
        for current in config["validation_current_counts"]:
            validation[f"varied_k{k}_c{current}"] = varied_tasks(
                "validation",
                seed=20260922 + k * 100 + current,
                count=config["validation_examples"],
                capacity=k,
                current_count=current,
            )
    history = []
    synchronize(device)
    started = time.perf_counter()
    curriculum = None
    if config.get("comparator_steps", 0):
        curriculum = pretrain_comparator(
            model, config["comparator_steps"], config["comparator_learning_rate"]
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]
    )
    for step in range(1, config["steps"] + 1):
        model.train()
        examples = rng.choices(views, k=config["batch_size"])
        loss, components = supervised_loss(model, examples)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            metrics = {key: task_metrics(model, tasks) for key, tasks in validation.items()}
            entry = dict(
                step=step, loss=float(loss.detach()), components=components, validation=metrics
            )
            history.append(entry)
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        step=step,
                        loss=entry["loss"],
                        accuracy={k: v["accuracy"] for k, v in metrics.items()},
                        all_gates=all(
                            v["absolute_gate"] and v["relative_gate"] for v in metrics.values()
                        ),
                    )
                ),
                flush=True,
            )
    synchronize(device)
    metadata = dict(
        seed=seed,
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started,
        history=history,
        comparator_curriculum=curriculum,
        checkpoint="final, not validation-best",
        config=config,
        training_tasks=len(training),
        training_views=len(views),
        supervision="visible index, per-coordinate equality and operation authority",
        inference="no equality labels, no values, no hidden world, no retained metadata",
    )
    torch.save(
        dict(settings=model.settings, state=model.state_dict(), metadata=metadata),
        output / f"reader-{seed}.pt",
    )
    (output / f"training-{seed}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 43])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("refusing to overwrite nonempty output")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    paths = list(Path("src/bounded_memory_transformer").rglob("*.py"))
    (args.output / "source-hashes.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}, indent=2
        )
        + "\n"
    )
    for seed in args.seeds:
        train(config, seed, args.output)


if __name__ == "__main__":
    main()
