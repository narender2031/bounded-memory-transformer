"""Reproducible five-case experiment and explicitly labelled combined diagnostics."""

import argparse
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

from ..cli.train_tiny import resolve_device
from ..memory_benchmark.operations import Kind, Operation
from ..memory_benchmark.reader import render, synchronize
from ..memory_benchmark.views import QueryView, read_visible
from . import capacity, lifecycle
from .metrics import paired_accuracy, reader_report, selection_diagnostics
from .reader_cases import STRATA, empty_memory, generate_reader_tasks
from .reader_training import ReaderBundle, train_readers


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict:
    root = Path(__file__).resolve().parents[3]
    files = sorted((root / "src").rglob("*.py"))
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
    status = subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True)
    return {
        "git_head": revision.strip(),
        "git_status": status.splitlines(),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in files},
    }


def _evaluate_views(
    bundle: ReaderBundle,
    views: list[QueryView],
    targets: list[str],
    ids: list[int],
    categories: dict[str, list[bool]],
    labels: list[str],
    bootstrap_samples: int,
) -> tuple[dict, list[dict], dict]:
    synchronize(bundle.device)
    started = time.perf_counter()
    result = bundle.predict(views)
    empty_views = [empty_memory(view) for view in views]
    paired = bundle.predict(empty_views)
    synchronize(bundle.device)
    seconds = time.perf_counter() - started
    visible = result["answers"]["oracle_copy"]
    empty_exact = paired["answers"]["oracle_copy"]
    reports = {
        system: reader_report(
            predictions=answers,
            visible_targets=visible,
            world_targets=targets,
            no_evidence=empty_exact,
            paired_no_memory=paired["answers"][system],
            categories=categories,
            episode_ids=ids,
            bootstrap_samples=bootstrap_samples,
        )
        for system, answers in result["answers"].items()
    }
    rows = []
    for index, view in enumerate(views):
        rows.append(
            {
                "episode_id": ids[index],
                "stratum": labels[index],
                "entity": view.query.entity,
                "attribute": view.query.attribute,
                "prompt": render(view),
                "empty_prompt": render(empty_views[index]),
                "target": visible[index],
                "world_target": targets[index],
                "no_evidence": empty_exact[index],
                "state_bytes": view.state_bytes,
                "answers": {name: answers[index] for name, answers in result["answers"].items()},
                "paired_no_memory": {
                    name: answers[index] for name, answers in paired["answers"].items()
                },
                "selector_index": result["selector_indices"][index],
                "oracle_index": result["oracle_indices"][index],
            }
        )
    diagnostics = selection_diagnostics(
        result["selector_indices"],
        result["oracle_indices"],
        result["answers"]["selected_copy"],
        result["answers"]["selected_character"],
        visible,
    )
    diagnostics["four_arms_and_paired_controls_seconds"] = seconds
    diagnostics["mean_prompt_tokens"] = sum(len(render(v)) for v in views) / len(views)
    return reports, rows, diagnostics


def _bank_view(bank: np.ndarray, entity: int, attribute: int, *, policy=None) -> QueryView:
    query = Operation(Kind.ASK, entity, attribute)
    records = capacity.bank_records(bank)
    if policy is not None:
        policy.bank = bank.copy()
        visible = policy.retrieve(query)
    else:
        visible = records
    return QueryView(visible, (), query, bank.nbytes, records)


def _capacity_views(result: dict, policy_name: str, slots: int, model, seed: int):
    policy = capacity.make_capacity_policy(policy_name, slots, model=model, seed=seed)
    views, targets, ids = [], [], []
    for episode in result["episodes"]:
        bank = np.array(episode["final_bank"], dtype=np.int32).reshape(-1, 4)
        for row in episode["records"]:
            views.append(_bank_view(bank, row["entity"], row["attribute"], policy=policy))
            targets.append(row["truth"])
            ids.append(episode["episode_id"])
    return views, targets, ids


def _visible_categories(views: list[QueryView]) -> dict[str, list[bool]]:
    known = [read_visible(v.memory, v.current, v.query) != "??" for v in views]
    categories = {"visible_known": known, "visible_unknown": [not x for x in known]}
    # These are descriptive episode slices. Absent slices are not competence gates.
    return {key: mask for key, mask in categories.items() if any(mask)}


def _combined_capacity(controller, scorer, episodes, slots: int) -> tuple[dict, list[dict]]:
    results, writes = [], []
    for episode in episodes:
        bank = lifecycle.empty_bank(slots)
        policy = capacity.make_capacity_policy("learned", slots, model=scorer)
        policy.bank = bank
        for session_index, session in enumerate(episode.sessions):
            for operation_index, (operation, cue) in enumerate(session):
                decision = controller.decide(bank, operation)
                allocation = None
                if decision.action == lifecycle.Action.STORE and decision.target == len(bank):
                    allocation = policy.select_allocation(bank, operation, cue)
                before = bank.tolist()
                bank = lifecycle.apply_decision(bank, operation, decision, cue, allocation)
                policy.bank = bank
                writes.append(
                    {
                        "episode_id": episode.episode_id,
                        "session": session_index,
                        "operation_index": operation_index,
                        "operation": asdict(operation),
                        "cue": cue,
                        "action": decision.action.name,
                        "target": decision.target,
                        "allocation": allocation,
                        "bank_before": before,
                        "bank_after": bank.tolist(),
                    }
                )
            bank = np.frombuffer(bank.tobytes(), dtype=np.int32).copy().reshape(slots, 4)
            policy.bank = bank
        score = capacity.score_bank(episode, bank)
        oracle = capacity.oracle_retention(episode, slots)
        results.append(
            {
                "episode_id": episode.episode_id,
                "workload": episode.workload,
                "final_bank": bank.tolist(),
                **score,
                "oracle_utility": oracle["utility"],
                "oracle_hits": oracle["hits"],
                "oracle_regret": oracle["utility"] - score["utility"],
            }
        )
    count = sum(row["queries"] for row in results)
    utility = sum(row["hits"] for row in results) / count
    oracle_utility = sum(row["oracle_hits"] for row in results) / count
    return {
        "episodes": results,
        "summary": {
            "utility": utility,
            "oracle_utility": oracle_utility,
            "oracle_regret": oracle_utility - utility,
            "queries": count,
            "state_bytes": slots * 16,
        },
    }, writes


def episode_reader_gate(result: dict) -> dict:
    """Require recovery on each declared episode control, separately from microtasks."""
    controls = {}
    for workload, capacities in result["capacity"].items():
        for slots, policies in capacities.items():
            for policy, row in policies.items():
                if policy in ("no_memory", "paired_comparisons"):
                    continue
                controls[f"capacity/{workload}/{slots}/{policy}"] = row["neural"]
    controls["lifecycle"] = result["end_to_end"]["lifecycle"]["reader"]
    for workload, capacities in result["end_to_end"]["combined_capacity"].items():
        for slots, row in capacities.items():
            controls[f"combined/{workload}/{slots}"] = row["neural"]
    values = {
        key: row["selected_copy"]["reader_recovery"]["value"] for key, row in controls.items()
    }
    return {
        "passed": bool(values) and all(v is not None and v >= 0.95 for v in values.values()),
        "required_recovery": values,
        "unavailable": [key for key, value in values.items() if value is None],
        "failed": [key for key, value in values.items() if value is not None and value < 0.95],
    }


def _validate_config(config: dict) -> None:
    required = {
        "seeds",
        "device",
        "threads",
        "reader",
        "lifecycle",
        "capacity_training",
        "evaluation",
    }
    if set(config) != required:
        raise ValueError(f"configuration keys must be {sorted(required)}")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if config["threads"] < 1 or config["reader"]["capacity"] != 4:
        raise ValueError("positive threads and four-slot reader training required")
    if config["lifecycle"]["slots"] != 4 or config["capacity_training"]["capacity"] != 4:
        raise ValueError("train all controllers at four slots")
    evaluation = config["evaluation"]
    if evaluation["split"] not in ("validation", "test"):
        raise ValueError("evaluation split must be validation or test")
    if evaluation["reader_tasks"] <= 0 or evaluation["reader_tasks"] % len(STRATA):
        raise ValueError("reader task count must balance the five strata")
    for name in (
        "lifecycle_episodes",
        "capacity_episodes",
        "capacity_queries",
        "bootstrap_samples",
    ):
        if evaluation[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if evaluation["capacity_slots"] != [4, 8]:
        raise ValueError("this protocol evaluates four/eight slots in that order")


def run_experiment(config: dict, output: Path) -> dict:
    _validate_config(config)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.json").open("x") as handle:
        json.dump(config, handle, indent=2, sort_keys=True, allow_nan=False)
    save_json(output / "source-manifest.json", source_manifest())
    torch.set_num_threads(config["threads"])
    device = resolve_device(config["device"]).type
    summary = {
        "config": config,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "reader_device": device,
            "controller_device": "cpu",
            "threads": config["threads"],
        },
        "contract": {
            "raw_history_tokens_reprocessed": 0,
            "primary_slots": 4,
            "primary_state_bytes": 64,
            "slot_fields": ["entity", "attribute", "value", "cue"],
            "test_opened_after_all_training": True,
            "oracle_retention": "evaluation-only feasible top-K future probe counts",
            "capacity_training": "delayed reward; no future action or frequency labels",
            "eight_slots": "same model weights trained at four slots",
        },
        "training": {},
        "seeds": {},
    }
    trained = {}
    # Freeze every seed's models before constructing held-out examples.
    for seed in config["seeds"]:
        print(f"Training all components for seed {seed}", flush=True)
        started = time.perf_counter()
        reader = train_readers(config["reader"], seed, device)
        reader.save(output / f"reader-{seed}.pt")
        writer_started = time.perf_counter()
        writer = lifecycle.train_lifecycle(config["lifecycle"], seed, "cpu")
        writer_seconds = time.perf_counter() - writer_started
        torch.save(
            {"state_dict": writer.model.state_dict(), "config": config["lifecycle"]},
            output / f"lifecycle-{seed}.pt",
        )
        utility_started = time.perf_counter()
        utility = capacity.train_capacity(config["capacity_training"], seed, "cpu")
        utility_seconds = time.perf_counter() - utility_started
        torch.save(
            {"state_dict": utility.model.state_dict(), "config": config["capacity_training"]},
            output / f"capacity-{seed}.pt",
        )
        trained[seed] = (reader, writer, utility)
        summary["training"][str(seed)] = {
            "reader": reader.metadata,
            "lifecycle": {
                **writer.metadata,
                "curve": writer.learning_curve,
                "training_seconds": writer_seconds,
            },
            "capacity": {
                **utility.metadata,
                "curve": utility.curve,
                "training_seconds": utility_seconds,
            },
            "total_seconds": time.perf_counter() - started,
        }
        save_json(output / f"training-{seed}.json", summary["training"][str(seed)])

    evaluation = config["evaluation"]
    split, bootstrap = evaluation["split"], evaluation["bootstrap_samples"]
    tasks = generate_reader_tasks(
        split, seed=evaluation["reader_seed"], count=evaluation["reader_tasks"], capacity=4
    )
    lifecycle_episodes = lifecycle.generate_lifecycle(
        split, evaluation["lifecycle_seed"], evaluation["lifecycle_episodes"]
    )
    workloads = {
        name: capacity.generate_capacity(
            split,
            evaluation["capacity_seed"],
            evaluation["capacity_episodes"],
            workload=name,
            queries=evaluation["capacity_queries"],
        )
        for name in ("A", "B")
    }
    save_json(output / "reader-data.json", [asdict(task) for task in tasks])
    save_json(output / "lifecycle-data.json", [asdict(ep) for ep in lifecycle_episodes])
    for name, episodes in workloads.items():
        save_json(output / f"capacity-data-{name}.json", [asdict(ep) for ep in episodes])
    for seed, (reader, writer, utility) in trained.items():
        print(f"Evaluating all five cases for seed {seed} on {split}", flush=True)
        categories = {name: [t.stratum == name for t in tasks] for name in STRATA}
        categories.update(
            {
                "known": [t.known for t in tasks],
                "unknown": [not t.known for t in tasks],
                "full_occupancy": [t.occupancy == 4 for t in tasks],
            }
        )
        reports, rows, diagnostics = _evaluate_views(
            reader,
            [t.view for t in tasks],
            [t.target for t in tasks],
            [t.episode_id for t in tasks],
            categories,
            [t.stratum for t in tasks],
            bootstrap,
        )
        save_rows(output / f"reader-{seed}.jsonl", rows)
        seed_result = {
            "reader": reports,
            "reader_diagnostics": diagnostics,
            "lifecycle": {},
            "capacity": {},
            "end_to_end": {"combined_capacity": {}},
        }
        learned_lifecycle = None
        for name, controller in (("exact", lifecycle.exact_decision), ("learned", writer)):
            result = lifecycle.evaluate_lifecycle(controller, lifecycle_episodes)
            seed_result["lifecycle"][name] = result["summary"]
            save_json(output / f"lifecycle-{seed}-{name}.json", result)
            if name == "learned":
                learned_lifecycle = result
        assert learned_lifecycle is not None
        life_rows = learned_lifecycle["queries"]
        views = [
            _bank_view(np.asarray(r["bank"], dtype=np.int32), r["entity"], r["attribute"])
            for r in life_rows
        ]
        e2e, rows, diagnostics = _evaluate_views(
            reader,
            views,
            [r["expected"] for r in life_rows],
            [r["episode_id"] for r in life_rows],
            {
                "update": [r["case"] == "update" for r in life_rows],
                "delete": [r["case"] == "delete" for r in life_rows],
                "control": [r["control"] for r in life_rows],
            },
            [r["case"] for r in life_rows],
            bootstrap,
        )
        seed_result["end_to_end"]["lifecycle"] = {"reader": e2e, "diagnostics": diagnostics}
        save_rows(output / f"end-to-end-lifecycle-{seed}.jsonl", rows)
        for workload, episodes in workloads.items():
            seed_result["capacity"][workload] = {}
            seed_result["end_to_end"]["combined_capacity"][workload] = {}
            for slots in evaluation["capacity_slots"]:
                condition = {}
                exact_predictions = {}
                for policy_name in capacity.POLICIES:
                    print(
                        f"seed={seed} capacity={slots} workload={workload} policy={policy_name}",
                        flush=True,
                    )
                    result = capacity.evaluate_capacity(
                        episodes, policy_name, slots, model=utility.model, seed=seed
                    )
                    stem = f"capacity-{seed}-{workload}-{slots}-{policy_name}"
                    save_json(output / f"{stem}-banks.json", result)
                    views, targets, ids = _capacity_views(
                        result, policy_name, slots, utility.model, seed
                    )
                    exact_predictions[policy_name] = [
                        r["answer"] for e in result["episodes"] for r in e["records"]
                    ]
                    neural, rows, diagnostics = _evaluate_views(
                        reader,
                        views,
                        targets,
                        ids,
                        _visible_categories(views),
                        [workload] * len(views),
                        bootstrap,
                    )
                    save_rows(output / f"{stem}.jsonl", rows)
                    condition[policy_name] = {
                        "exact": result["summary"],
                        "neural": neural,
                        "diagnostics": diagnostics,
                    }
                condition["paired_comparisons"] = {
                    baseline: paired_accuracy(
                        exact_predictions["learned"],
                        exact_predictions[baseline],
                        targets,
                        episode_ids=ids,
                        bootstrap_samples=bootstrap,
                    )
                    for baseline in ("fifo", "recency", "random", "cue_priority")
                }
                seed_result["capacity"][workload][str(slots)] = condition
                combined, writes = _combined_capacity(writer, utility.model, episodes, slots)
                stem = f"combined-{seed}-{workload}-{slots}"
                save_json(output / f"{stem}-banks.json", combined)
                save_rows(output / f"{stem}-writes.jsonl", writes)
                views, targets, ids = _capacity_views(
                    combined, "learned", slots, utility.model, seed
                )
                neural, rows, diagnostics = _evaluate_views(
                    reader,
                    views,
                    targets,
                    ids,
                    _visible_categories(views),
                    [workload] * len(views),
                    bootstrap,
                )
                save_rows(output / f"{stem}.jsonl", rows)
                seed_result["end_to_end"]["combined_capacity"][workload][str(slots)] = {
                    "exact": combined["summary"],
                    "neural": neural,
                    "diagnostics": diagnostics,
                }
        chosen = reports["selected_copy"]
        life = seed_result["lifecycle"]["learned"]
        seed_result["episode_reader_gate"] = episode_reader_gate(seed_result)
        seed_result["gates"] = {
            "reader_absolute": chosen["absolute_gate_passed"],
            "reader_relative": chosen["relative_gate_passed"],
            "reader_episode_relative": seed_result["episode_reader_gate"]["passed"],
            "lifecycle": all(
                life[key] is not None and life[key] >= 0.95
                for key in (
                    "semantic_accuracy",
                    "update_accuracy",
                    "delete_accuracy",
                    "control_preservation",
                )
            ),
        }
        seed_result["end_to_end"]["interpretation"] = (
            "component gates passed; assess utility and harm separately"
            if all(seed_result["gates"].values())
            else "diagnostic only: a component gate failed"
        )
        summary["seeds"][str(seed)] = seed_result
        save_json(output / f"evaluation-{seed}.json", seed_result)
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["artifact_sha256"] = {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()
    }
    save_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = run_experiment(json.loads(args.config.read_text()), args.output)
    print(
        json.dumps({seed: result["gates"] for seed, result in summary["seeds"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
