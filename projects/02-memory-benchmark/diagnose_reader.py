"""Frozen-reader factorial diagnostic; no training or held-out test symbols.

Rename entities and values independently in already-retrieved reading microtasks.
Equality, operation order, answer availability, and prompt length stay fixed.
Retrieval is NOT rerun: lexical similarity could change after renaming. This
isolates the reader and is not a new memory-policy benchmark.
"""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import UNKNOWN
from bounded_memory_transformer.memory_benchmark.policies import POLICIES
from bounded_memory_transformer.memory_benchmark.reader import (
    make_training_views,
    predict,
    render,
    synchronize,
)
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.tiny_transformer import TinyTransformerLM, TransformerConfig


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(view: QueryView, entity_split: str, value_split: str, seed: int) -> QueryView:
    """Injective, per-view renaming; each axis is independent of the other."""
    operations = (*view.memory, *view.current, *view.retained, view.query)
    entities = sorted({o.entity for o in operations})
    values = sorted({o.value for o in operations if o.value is not None})
    entity_map = dict(
        zip(
            entities,
            random.Random(seed).sample(symbol_space(entity_split).entities, len(entities)),
            strict=True,
        )
    )
    value_map = dict(
        zip(
            values,
            random.Random(seed + 1_000_003).sample(symbol_space(value_split).values, len(values)),
            strict=True,
        )
    )

    def operation(o):
        return replace(
            o, entity=entity_map[o.entity], value=None if o.value is None else value_map[o.value]
        )

    renamed = QueryView(
        tuple(map(operation, view.memory)),
        tuple(map(operation, view.current)),
        operation(view.query),
        view.state_bytes,
        tuple(map(operation, view.retained)),
    )
    original_answer = read_visible(view.memory, view.current, view.query)
    expected = UNKNOWN if original_answer == UNKNOWN else f"{value_map[int(original_answer)]:02d}"
    assert read_visible(renamed.memory, renamed.current, renamed.query) == expected
    assert len(render(renamed)) == len(render(view))
    assert [o.key == view.query.key for o in operations] == [
        operation(o).key == renamed.query.key for o in operations
    ]
    return renamed


def score(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0, "accuracy": None}
    return {
        "count": len(rows),
        "accuracy": sum(r["answer"] == r["expected"] for r in rows) / len(rows),
    }


def summarize(rows: list[dict]) -> dict:
    known = [r for r in rows if r["expected"] != UNKNOWN]
    unknown = [r for r in rows if r["expected"] == UNKNOWN]
    wrong_known = [r for r in known if r["answer"] != r["expected"]]
    return {
        "overall": score(rows),
        "known_visible": score(known),
        "required_abstention": score(unknown),
        "known_memory_only": score([r for r in known if not r["current_known"]]),
        "known_current": score([r for r in known if r["current_known"]]),
        "by_visible_slots": {
            str(k): score([r for r in rows if r["visible_slots"] == k]) for k in range(5)
        },
        "by_policy": {p: score([r for r in rows if r["policy"] == p]) for p in POLICIES},
        "errors": {
            "known_wrong": len(wrong_known),
            "known_unnecessary_abstention": sum(r["answer"] == UNKNOWN for r in wrong_known),
            "known_wrong_train_value": sum(
                r["answer"] in {f"{v:02d}" for v in symbol_space("train").values}
                for r in wrong_known
            ),
            "unknown_false_answer": sum(r["answer"] != UNKNOWN for r in unknown),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(config["entity_splits"] + config["value_splits"]) - {"train", "validation"}:
        raise ValueError("this diagnostic must not use held-out test symbols")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(config["threads"])
    device = config["device"]
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_directory = Path(config["checkpoint_directory"])
    paths = [Path(__file__), args.config, *Path("src").rglob("*.py")]
    report = {
        "purpose": "Exploratory reader diagnostic, not a held-out memory result",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
        },
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha256(p) for p in paths},
        "checkpoint_sha256": {
            str(seed): sha256(checkpoint_directory / f"seed-{seed}.pt") for seed in config["seeds"]
        },
        "conditions": {},
    }
    base = make_training_views(
        "train",
        seed=config["generator_seed"],
        count=config["examples"],
        capacity=config["capacity"],
    )
    synchronize(device)
    started = time.perf_counter()
    with (args.output / "predictions.jsonl").open("w") as output:
        for entity_split in config["entity_splits"]:
            for value_split in config["value_splits"]:
                condition = f"entities_{entity_split}__values_{value_split}"
                views = [
                    rename(v, entity_split, value_split, config["renaming_seed"] + i)
                    for i, v in enumerate(base)
                ]
                prompts = [render(v) for v in views]
                expected = [read_visible(v.memory, v.current, v.query) for v in views]
                condition_rows, seeds = [], {}
                for seed in config["seeds"]:
                    checkpoint = torch.load(
                        checkpoint_directory / f"seed-{seed}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    assert checkpoint["seed"] == seed
                    model = TinyTransformerLM(TransformerConfig(**checkpoint["model_config"]))
                    model.load_state_dict(checkpoint["state_dict"])
                    model.to(device).eval()
                    answers = predict(
                        model, prompts, batch_size=config["batch_size"], device=device
                    )
                    rows = [
                        {
                            "seed": seed,
                            "condition": condition,
                            "view": i,
                            "prompt": prompts[i],
                            "expected": target,
                            "answer": answer,
                            "policy": POLICIES[i % len(POLICIES)],
                            "visible_slots": len(view.memory),
                            "current_known": read_visible((), view.current, view.query) != UNKNOWN,
                        }
                        for i, (view, target, answer) in enumerate(
                            zip(views, expected, answers, strict=True)
                        )
                    ]
                    seeds[str(seed)] = summarize(rows)
                    condition_rows.extend(rows)
                    output.writelines(json.dumps(row) + "\n" for row in rows)
                    del model
                report["conditions"][condition] = {
                    "pooled_descriptive": summarize(condition_rows),
                    "per_seed": seeds,
                }
                print(condition, json.dumps(summarize(condition_rows)), flush=True)
    synchronize(device)
    report["seconds_including_generation_and_io"] = time.perf_counter() - started
    report["predictions_sha256"] = sha256(args.output / "predictions.jsonl")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
