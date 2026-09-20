"""Frozen reader evaluation on fresh episodes and unchanged reference actors."""

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bounded_memory_transformer.memory_benchmark.operations import UNKNOWN
from bounded_memory_transformer.memory_benchmark.reader import synchronize
from bounded_memory_transformer.memory_benchmark.views import read_visible
from bounded_memory_transformer.memory_experiments import capacity, lifecycle
from bounded_memory_transformer.memory_experiments.metrics import reader_report
from bounded_memory_transformer.memory_experiments.reader_cases import (
    STRATA,
    copy_selected,
    empty_memory,
    generate_reader_tasks,
    oracle_select,
)
from bounded_memory_transformer.memory_experiments.reader_training import ReaderBundle
from bounded_memory_transformer.memory_experiments.run import (
    _bank_view,
    _capacity_views,
    _combined_capacity,
)

from .reader import FactorizedReader
from .tasks import varied_tasks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def evaluate_views(model, views, world, ids, categories, path, bootstrap, old=None):
    started = time.perf_counter()
    device = str(next(model.parameters()).device)
    selected = model.select(views)
    paired = model.select([empty_memory(v) for v in views])
    answers = [copy_selected(v, i) for v, i in zip(views, selected, strict=True)]
    empty_answers = [copy_selected(empty_memory(v), i) for v, i in zip(views, paired, strict=True)]
    visible = [read_visible(v.memory, v.current, v.query) for v in views]
    empty = [read_visible((), v.current, v.query) for v in views]
    oracle = [oracle_select(v) for v in views]
    assert visible == [copy_selected(v, i) for v, i in zip(views, oracle, strict=True)]
    old_indices = old.select(views) if old is not None else None
    synchronize(device)
    seconds = time.perf_counter() - started
    categories = dict(categories)
    known = [answer != UNKNOWN for answer in visible]
    categories.update(visible_known=known, visible_unknown=[not k for k in known])
    present = {key: mask for key, mask in categories.items() if any(mask)}
    report = reader_report(
        predictions=answers,
        visible_targets=visible,
        world_targets=world,
        no_evidence=empty,
        paired_no_memory=empty_answers,
        categories=present,
        episode_ids=ids,
        bootstrap_samples=bootstrap,
    )
    report["absent_categories"] = [key for key, mask in categories.items() if not any(mask)]
    report["selection_index_accuracy"] = sum(
        a == b for a, b in zip(selected, oracle, strict=True)
    ) / len(views)
    report["inference_seconds_with_controls"] = seconds
    evidence_hash = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        for i, v in enumerate(views):
            evidence = dict(
                memory=[asdict(op) for op in v.memory],
                current=[asdict(op) for op in v.current],
                query=asdict(v.query),
                world=world[i],
                state_bytes=v.state_bytes,
                episode_id=ids[i],
            )
            evidence_hash.update(json.dumps(evidence, sort_keys=True).encode())
            row = dict(
                **evidence,
                visible=visible[i],
                current_only=empty[i],
                answer=answers[i],
                no_memory_answer=empty_answers[i],
                selected=selected[i],
                no_memory_selected=paired[i],
                oracle=oracle[i],
                categories=[key for key, mask in categories.items() if mask[i]],
            )
            if old_indices is not None:
                row["v1_selected"] = old_indices[i]
                row["v1_answer"] = copy_selected(v, old_indices[i])
            handle.write(json.dumps(row) + "\n")
    report["evidence_sha256"] = evidence_hash.hexdigest()
    report["rows_sha256"] = digest(path)
    if old_indices is not None:
        report["frozen_v1_copy_accuracy"] = sum(
            copy_selected(v, j) == answer
            for v, j, answer in zip(views, old_indices, visible, strict=True)
        ) / len(views)
    report["passed"] = report["absolute_gate_passed"] and report["relative_gate_passed"] is True
    return report


def task_categories(tasks, slots):
    categories = {s: [t.stratum == s for t in tasks] for s in STRATA}
    categories.update(
        {f"occupancy_{k}": [t.occupancy == k for t in tasks] for k in range(slots + 1)}
    )
    sources = [
        "none"
        if t.oracle_index < 0
        else ("memory" if t.oracle_index < len(t.view.memory) else "current")
        for t in tasks
    ]
    categories.update(
        {f"source_{s}": [origin == s for origin in sources] for s in ("memory", "current", "none")}
    )
    return categories


def run(config, training, references, output, *, smoke=False):
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite nonempty output")
    if config["threshold"] != 0.95 or config["capacities"] != [4, 8]:
        raise ValueError("this protocol fixes the 95% gate and 4/8-slot budgets")
    tracked_scope = [
        "src",
        "projects/04-memory-followup/configs",
        "research/reader-followup-protocol-2026-09-20.md",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked_scope], text=True
    )
    if dirty and not smoke:
        raise ValueError("commit source/config/protocol before held-out evaluation:\n" + dirty)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    paths = list(Path("src").rglob("*.py"))
    paths += list(Path("projects/04-memory-followup/configs").glob("*.json"))
    paths += [Path("research/reader-followup-protocol-2026-09-20.md")]
    for seed in config["seeds"]:
        paths += [training / f"reader-{seed}.pt", training / f"training-{seed}.json"]
        paths += [references / f"{name}-{seed}.pt" for name in ("reader", "lifecycle", "capacity")]
    hashes = {str(p): digest(p) for p in paths}
    summary = dict(
        config=config,
        smoke=smoke,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hashes=hashes,
        seeds={},
        environment=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            platform=platform.platform(),
            device=config["device"],
        ),
    )
    # Load/check every seed BEFORE constructing held-out examples.
    loaded = {}
    train_config = json.loads((training / "config.json").read_text())
    for seed in config["seeds"]:
        saved = torch.load(
            training / f"reader-{seed}.pt", map_location=config["device"], weights_only=True
        )
        if saved["metadata"]["config"] != train_config or saved["metadata"]["seed"] != seed:
            raise ValueError("checkpoint/config/seed mismatch")
        model = FactorizedReader(**saved["settings"]).to(config["device"])
        model.load_state_dict(saved["state"])
        model.eval()
        utility = torch.load(
            references / f"capacity-{seed}.pt", weights_only=True, map_location="cpu"
        )
        scorer = capacity.RetentionScorer()
        scorer.load_state_dict(utility["state_dict"])
        writer = torch.load(
            references / f"lifecycle-{seed}.pt", weights_only=True, map_location="cpu"
        )
        controller = lifecycle.LifecycleModel(writer["config"]["hidden"])
        controller.load_state_dict(writer["state_dict"])
        old = ReaderBundle.load(references / f"reader-{seed}.pt", config["device"]).selector
        loaded[seed] = model, scorer.eval(), controller.eval(), old
    save_json(output / "frozen-manifest.json", summary)
    started = time.perf_counter()
    for seed, (model, scorer, controller, old) in loaded.items():
        conditions = {}

        def evaluate(
            name,
            views,
            world,
            ids,
            categories,
            prior=None,
            model=model,
            seed=seed,
            conditions=conditions,
        ):
            report = evaluate_views(
                model,
                views,
                world,
                ids,
                categories,
                output / f"seed-{seed}" / f"{name}.jsonl.gz",
                config["bootstrap_samples"],
                prior,
            )
            conditions[name] = report
            print(
                json.dumps(
                    dict(
                        seed=seed,
                        condition=name,
                        visible=report["visible_accuracy"],
                        world=report["world_accuracy"],
                        recovery=report["reader_recovery"]["value"],
                        harm=report["paired"]["harmful_rate"],
                        passed=report["passed"],
                    )
                ),
                flush=True,
            )

        for slots in config["capacities"]:
            tasks = generate_reader_tasks(
                config["split"],
                seed=config["original_seed"],
                count=config["original_count"],
                capacity=slots,
            )
            evaluate(
                f"original-k{slots}",
                [t.view for t in tasks],
                [t.target for t in tasks],
                [t.episode_id for t in tasks],
                task_categories(tasks, slots),
                old,
            )
            for current in config["current_counts"]:
                tasks = varied_tasks(
                    config["split"],
                    seed=config["varied_seed"] + slots * 100 + current,
                    count=config["varied_count"],
                    capacity=slots,
                    current_count=current,
                )
                evaluate(
                    f"varied-k{slots}-c{current}",
                    [t.view for t in tasks],
                    [t.target for t in tasks],
                    [t.episode_id for t in tasks],
                    task_categories(tasks, slots),
                )
        episodes = lifecycle.generate_lifecycle(
            config["split"], config["lifecycle_seed"], config["lifecycle_count"]
        )
        result = lifecycle.evaluate_lifecycle(controller, episodes)
        save_json(output / f"lifecycle-{seed}.json", result)
        rows = result["queries"]
        views = [
            _bank_view(np.array(r["bank"], dtype=np.int32), r["entity"], r["attribute"])
            for r in rows
        ]
        categories = dict(
            update=[r["case"] == "update" and not r["control"] for r in rows],
            delete=[r["case"] == "delete" and not r["control"] for r in rows],
            control=[r["control"] for r in rows],
        )
        evaluate(
            "lifecycle",
            views,
            [r["expected"] for r in rows],
            [r["episode_id"] for r in rows],
            categories,
        )
        for workload in ("A", "B"):
            episodes = capacity.generate_capacity(
                config["split"],
                seed=config["capacity_seed"],
                count=config["capacity_count"],
                workload=workload,
            )
            for slots in config["capacities"]:
                for policy in config["capacity_policies"]:
                    result = capacity.evaluate_capacity(
                        episodes, policy, slots, model=scorer, seed=seed
                    )
                    name = f"capacity-{workload}-k{slots}-{policy}"
                    save_json(output / f"{name}-{seed}.json", result)
                    views, targets, ids = _capacity_views(result, policy, slots, scorer, seed)
                    evaluate(name, views, targets, ids, {})
                combined, writes = _combined_capacity(controller, scorer, episodes, slots)
                name = f"combined-{workload}-k{slots}"
                save_json(output / f"{name}-{seed}.json", dict(result=combined, writes=writes))
                views, targets, ids = _capacity_views(combined, "learned", slots, scorer, seed)
                evaluate(name, views, targets, ids, {})
        summary["seeds"][str(seed)] = dict(
            conditions=conditions,
            passed=all(r["passed"] for r in conditions.values()),
            failed=[key for key, row in conditions.items() if not row["passed"]],
        )
        save_json(output / "summary.json", summary)
    if hashes != {str(p): digest(p) for p in paths}:
        raise RuntimeError("frozen source/config/checkpoint changed during evaluation")
    summary["seconds"] = time.perf_counter() - started
    summary["passed"] = all(r["passed"] for r in summary["seeds"].values())
    summary["artifact_hashes"] = {
        str(p.relative_to(output)): digest(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "summary.json"
    }
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.smoke:
        config.update(
            split="validation",
            seeds=[7],
            original_count=50,
            varied_count=100,
            lifecycle_count=8,
            capacity_count=2,
            bootstrap_samples=50,
            device="cpu",
        )
    run(config, args.training, args.references, args.output, smoke=args.smoke)


if __name__ == "__main__":
    main()
