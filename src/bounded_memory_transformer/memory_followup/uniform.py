"""Preregistered uniform-query replication using frozen Project 03 components.

Episode draws are the sampling units. Shared checkpoint behavior is evaluated
once; it is never counted as additional independent evidence.
"""

import argparse
import gzip
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import torch

from ..memory_benchmark.operations import UNKNOWN, Kind, Operation
from ..memory_experiments.capacity import (
    CapacityEpisode,
    RetentionScorer,
    evaluate_capacity,
    generate_capacity,
    make_capacity_policy,
)


def paired_interval(values: Any, *, family_size: int, alpha: float,
                    margin: float) -> dict[str, Any]:
    """Large-sample episode interval; Bonferroni does not make it exact."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("need at least two finite paired episode differences")
    if family_size < 1 or not 0 < alpha < 1 or not 0 < margin <= 1:
        raise ValueError("invalid family size, alpha, or equivalence margin")
    mean = float(data.mean())
    deviation = float(data.std(ddof=1))
    standard_error = deviation / math.sqrt(len(data))
    critical = NormalDist().inv_cdf(1 - alpha / (2 * family_size))
    interval = [mean - critical * standard_error, mean + critical * standard_error]
    ordinary = NormalDist().inv_cdf(.975) * standard_error
    return {"episodes": len(data), "mean": mean, "standard_deviation": deviation,
            "standard_error": standard_error, "ci95": [mean - ordinary, mean + ordinary],
            "simultaneous_ci": interval, "family_size": family_size, "alpha": alpha,
            "equivalence_margin": margin,
            "equivalent": bool(-margin < interval[0] and interval[1] < margin),
            "positive": bool(interval[0] > 0), "negative": bool(interval[1] < 0)}


def analyze_contrasts(by_seed: dict[int, dict[str, list[float]]],
                      inference: dict[str, Any]) -> dict[str, Any]:
    """Pair whole episodes; bootstrap within dataset seeds with shared indices."""
    if not by_seed:
        raise ValueError("no dataset replicates")
    seeds = sorted(by_seed)
    names = sorted(by_seed[seeds[0]])
    if not names or any(sorted(by_seed[seed]) != names for seed in seeds):
        raise ValueError("all dataset seeds must have the same contrasts")
    lengths = {len(by_seed[seed][name]) for seed in seeds for name in names}
    if len(lengths) != 1 or min(lengths) < 2:
        raise ValueError("all contrasts require the same fixed episode count, at least two")
    cube = np.asarray([[by_seed[seed][name] for name in names] for seed in seeds],
                      dtype=np.float64).transpose(0, 2, 1)
    if not np.isfinite(cube).all():
        raise ValueError("episode differences must be finite")
    alpha = float(inference["familywise_alpha"])
    margin = float(inference["equivalence_margin"])
    samples = int(inference["bootstrap_samples"])
    if samples < 2:
        raise ValueError("need at least two bootstrap samples")
    family = len(names)
    pooled = {name: paired_interval(cube[:, :, index].ravel(), family_size=family,
                                    alpha=alpha, margin=margin)
              for index, name in enumerate(names)}
    # Per-dataset intervals are descriptive. Do not add hypothesis-test flags.
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {}
        for name in names:
            ordinary = paired_interval(by_seed[seed][name], family_size=1,
                                       alpha=.05, margin=margin)
            per_seed[str(seed)][name] = {key: ordinary[key] for key in (
                "episodes", "mean", "standard_deviation", "standard_error", "ci95")}
    rng = np.random.default_rng(int(inference["bootstrap_seed"]))
    bootstrap = np.empty((samples, family), dtype=np.float64)
    seed_count, episode_count, _ = cube.shape
    for start in range(0, samples, 64):
        batch = min(64, samples - start)
        indices = rng.integers(0, episode_count, size=(batch, seed_count, episode_count))
        selected = cube[np.arange(seed_count)[None, :, None], indices]
        bootstrap[start:start + batch] = selected.mean(axis=(1, 2))
    tail = alpha / (2 * family)
    intervals = np.quantile(bootstrap, [tail, 1 - tail], axis=0)
    for index, name in enumerate(names):
        interval = intervals[:, index].tolist()
        pooled[name]["bootstrap_ci"] = interval
        pooled[name]["bootstrap_equivalent"] = bool(-margin < interval[0] and interval[1] < margin)
        pooled[name]["bootstrap_positive"] = bool(interval[0] > 0)
        pooled[name]["bootstrap_negative"] = bool(interval[1] < 0)
    return {
        "dataset_seeds": seeds, "family_size": family,
        "independent_unit": "episode; fixed models and policy seeds are not extra draws",
        "interval_method": "large-sample normal; six-contrast Bonferroni in the full protocol",
        "per_seed_interpretation": "descriptive unadjusted 95% intervals; no selection or stopping",
        "bootstrap": {"samples": samples, "seed": int(inference["bootstrap_seed"]),
                      "method": "paired episode percentile bootstrap within dataset seed",
                      "quantiles": [tail, 1 - tail]},
        "per_seed": per_seed, "pooled": pooled,
        "all_contrasts_practically_equivalent": all(
            row["equivalent"] and row["bootstrap_equivalent"] for row in pooled.values()),
    }


def verify_model_equivalence(models: dict[int, RetentionScorer],
                             capacities: list[int]) -> dict[str, Any]:
    """Exhaust the scorer's finite input alphabet before any dataset is opened.

    Frozen observe() fills empty slots and supersedes existing keys without
    consulting scores. The only learned choice is allocation at a full bank.
    Identical target choices therefore induce identical trajectories by induction.
    """
    if not models or not capacities or any(not 1 <= size <= 16 for size in capacities):
        raise ValueError("need models and positive, exhaustively checkable capacities")
    representative = min(models)
    model_details = {}
    for seed, model in sorted(models.items()):
        checked = 0
        for size in capacities:
            learned = make_capacity_policy("learned", size, model=model)
            heuristic = make_capacity_policy("cue_priority", size)
            for cues in product((0, 1), repeat=size + 1):
                bank = np.asarray([[entity, 0, 20, cues[entity]] for entity in range(size)],
                                  dtype=np.int32)
                incoming = Operation(Kind.SET, 99, 3, 21)
                expected = heuristic.select_allocation(bank, incoming, cues[-1])
                observed = learned.select_allocation(bank, incoming, cues[-1])
                if observed != expected:
                    raise ValueError(f"checkpoint {seed} does not match cue-priority at K={size}")
                checked += 1
        with torch.no_grad():
            device = next(model.parameters()).device
            scores = model(torch.tensor([0, 1], device=device)).cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError(f"checkpoint {seed} has nonfinite scores")
        model_details[str(seed)] = {"binary_cue_scores": scores,
                                    "parameters": sum(p.numel() for p in model.parameters()),
                                    "allocations_checked": checked}
    return {
        "representative_model_seed": representative,
        "allocations_checked_per_model": sum(2 ** (size + 1) for size in capacities),
        "models": model_details,
        "prediction_aliases": {**{f"learned:{seed}": "learned" for seed in sorted(models)},
                               "cue_priority": "learned"},
        "latency_note": "only the representative learned actor is timed; aliases add no runs",
    }


def validate_result(episodes: list[CapacityEpisode], result: dict[str, Any], *,
                    capacity: int, keys: int) -> None:
    """Independently check K current keys and raw predictions after evaluation.

    This evaluator-only world is never provided to CapacityPolicy. Invalid banks
    abort the run rather than silently changing the K/keys null expectation.
    """
    rows = result["episodes"]
    if len(rows) != len(episodes) or not episodes:
        raise ValueError("result episode count differs from dataset")
    total_hits = total_queries = 0
    for episode, row in zip(episodes, rows, strict=True):
        if episode.workload != "A" or row["episode_id"] != episode.episode_id:
            raise ValueError("wrong workload or unpaired episode")
        world = {}
        for session in episode.sessions:
            for operation, cue in session:
                if operation.kind not in (Kind.SET, Kind.UPDATE):
                    raise ValueError("uniform replication requires SET/UPDATE writes only")
                world[operation.key] = (operation.value, cue)
        if len(world) != keys:
            raise ValueError("unexpected number of current world keys")
        bank = row["final_bank"]
        if len(bank) != capacity or any(len(record) != 4 for record in bank):
            raise ValueError("bank shape differs from the fixed capacity")
        bank_keys = [(record[0], record[1]) for record in bank]
        if len(set(bank_keys)) != capacity:
            raise ValueError("bank contains duplicate keys")
        retained = {}
        for record, key in zip(bank, bank_keys, strict=True):
            if key not in world or (record[2], record[3]) != world[key]:
                raise ValueError("bank contains an absent key or stale value/cue")
            retained[key] = record[2]
        if len(row["records"]) != len(episode.queries) or row["queries"] != len(episode.queries):
            raise ValueError("incorrect episode query denominator")
        hits = 0
        for index, (query, record) in enumerate(zip(episode.queries, row["records"], strict=True)):
            if query.key not in world:
                raise ValueError("uniform query is not historically answerable")
            truth = f"{world[query.key][0]:02d}"
            prediction = f"{retained[query.key]:02d}" if query.key in retained else UNKNOWN
            expected = {"query_index": index, "entity": query.entity,
                        "attribute": query.attribute, "truth": truth, "answer": prediction,
                        "correct": prediction == truth, "historical_answerable": True}
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("raw prediction does not agree with bank and world")
            hits += prediction == truth
        counts = Counter(query.key for query in episode.queries)
        oracle_hits = sum(sorted(counts.values(), reverse=True)[:capacity])
        count = len(episode.queries)
        if (row["hits"] != hits or row["utility"] != hits / count
                or row["oracle_hits"] != oracle_hits
                or row["oracle_utility"] != oracle_hits / count
                or row["oracle_regret"] != oracle_hits / count - hits / count):
            raise ValueError("episode score does not agree with raw predictions")
        total_hits += hits
        total_queries += count
    summary = result["summary"]
    if (summary["episodes"] != len(episodes) or summary["queries"] != total_queries
            or summary["utility"] != total_hits / total_queries
            or summary["capacity"] != capacity
            or summary["persistent_state_bytes"] != capacity * 16
            or summary["historical_tokens_reprocessed"] != 0):
        raise ValueError("summary does not agree with fixed-capacity raw results")


def evaluate_uniform_dataset(episodes: list[CapacityEpisode], model: RetentionScorer,
                             capacities: list[int], random_policy_seeds: list[int], *,
                             keys: int) -> dict[str, Any]:
    """Use the original causal policies and exact evaluator without modification."""
    results = {}
    actors = [(name, name, 0) for name in ("learned", "fifo", "recency")]
    actors += [(f"random:{seed}", "random", seed) for seed in random_policy_seeds]
    for capacity in capacities:
        for label, policy, seed in actors:
            result = evaluate_capacity(episodes, policy, capacity,
                                       model=model if policy == "learned" else None, seed=seed)
            validate_result(episodes, result, capacity=capacity, keys=keys)
            results[f"K{capacity}:{label}"] = result
    return results


def build_contrasts(results: dict[str, Any], capacities: list[int],
                    random_policy_seeds: list[int]) -> dict[str, list[float]]:
    """Average fixed random-policy seeds before statistical resampling."""
    if not random_policy_seeds or len(set(random_policy_seeds)) != len(random_policy_seeds):
        raise ValueError("random policy seeds must be nonempty and distinct")
    contrasts = {}
    expected_pairing = None
    for capacity in capacities:
        prefix = f"K{capacity}:"
        rates = {}
        names = ["learned", "fifo", "recency", *[f"random:{s}" for s in random_policy_seeds]]
        for name in names:
            rows = results[prefix + name]["episodes"]
            pairing = [(row["episode_id"], row["queries"]) for row in rows]
            if (not rows or any(count <= 0 for _, count in pairing)
                    or len({identifier for identifier, _ in pairing}) != len(rows)):
                raise ValueError("need distinct paired episodes and positive query counts")
            if expected_pairing is None:
                expected_pairing = pairing
            elif pairing != expected_pairing:
                raise ValueError("policy results are not paired by episode and query count")
            rates[name] = np.asarray([row["hits"] / row["queries"] for row in rows])
        rates["random"] = np.mean([rates[f"random:{seed}"] for seed in random_policy_seeds], axis=0)
        for baseline in ("fifo", "recency", "random"):
            difference = rates["learned"] - rates[baseline]
            contrasts[prefix + f"learned-{baseline}"] = difference.tolist()
    return contrasts


def verify_freeze(repo_root: Path, files: list[Path], commit: str) -> str:
    """Require committed bytes, not merely a user-supplied revision label."""
    root = repo_root.resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", f"{commit}^{{commit}}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        for path in sorted(set(file.resolve() for file in files)):
            relative = path.relative_to(root).as_posix()
            frozen = subprocess.run(["git", "-C", str(root), "show", f"{revision}:{relative}"],
                                    check=True, capture_output=True).stdout
            if frozen != path.read_bytes():
                raise ValueError(f"{relative} differs from the frozen commit {revision}")
    except (subprocess.CalledProcessError, OSError) as error:
        raise ValueError("cannot verify all files against the frozen commit") from error
    return revision


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_json(path: Path, value: Any) -> None:
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(value, handle, sort_keys=True, allow_nan=False, separators=(",", ":"))
    else:
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n")


def _effective_settings(config: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in ("smoke", "held-out"):
        raise ValueError("mode must be smoke or held-out")
    evaluation = dict(config["smoke"] if mode == "smoke" else config["evaluation"])
    required_split = "train" if mode == "smoke" else "test"
    if evaluation["split"] != required_split:
        raise ValueError(f"{mode} requires the {required_split} split")
    seeds = evaluation["dataset_seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or evaluation["episodes_per_seed"] < 2:
        raise ValueError("require distinct dataset seeds and at least two episodes each")
    if set(config["smoke"]["dataset_seeds"]) & set(config["evaluation"]["dataset_seeds"]):
        raise ValueError("smoke and held-out dataset seeds must be disjoint")
    generator = config["generator"]
    capacities = config["capacities"]
    if (generator["workload"] != "A" or not capacities
            or len(set(capacities)) != len(capacities)
            or any(not 1 <= size <= min(generator["keys"], 16) for size in capacities)):
        raise ValueError("require uniform workload and distinct feasible capacities")
    policy_seeds = config["random_policy_seeds"]
    if not policy_seeds or len(set(policy_seeds)) != len(policy_seeds):
        raise ValueError("require distinct random policy seeds")
    model_seeds = [item["model_seed"] for item in config["checkpoints"]]
    if not model_seeds or len(set(model_seeds)) != len(model_seeds):
        raise ValueError("require distinct model seeds")
    inference = dict(config["inference"])
    if mode == "smoke":
        inference["bootstrap_samples"] = evaluation.pop("bootstrap_samples")
    if int(config["threads"]) < 1 or int(inference["bootstrap_samples"]) < 2:
        raise ValueError("invalid thread or bootstrap count")
    # Validate inference options before data generation.
    paired_interval([0, 0], family_size=3 * len(capacities),
                    alpha=float(inference["familywise_alpha"]),
                    margin=float(inference["equivalence_margin"]))
    return evaluation, inference


def run_replication(config_path: Path, output: Path, *, mode: str,
                    freeze_commit: str | None = None,
                    repo_root: Path | None = None) -> dict[str, Any]:
    """Run a fixed protocol, with train-only smoke and an explicit held-out gate."""
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    source_root = Path(__file__).resolve().parents[3]
    root = repo_root.resolve() if repo_root is not None else source_root
    config_path = (root / config_path).resolve()
    output = (root / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if mode == "held-out" and not freeze_commit:
        raise ValueError("held-out evaluation requires an explicit reviewed freeze commit")
    config = json.loads(config_path.read_text())
    evaluation, inference = _effective_settings(config, mode)
    protocol_path = root / config["protocol"]
    # Include imported local dependencies and package initializers in the freeze.
    runtime_relative = [
        "src/bounded_memory_transformer/__init__.py",
        "src/bounded_memory_transformer/memory_followup/__init__.py",
        "src/bounded_memory_transformer/memory_followup/uniform.py",
        "src/bounded_memory_transformer/memory_experiments/__init__.py",
        "src/bounded_memory_transformer/memory_experiments/capacity.py",
        "src/bounded_memory_transformer/memory_benchmark/__init__.py",
        "src/bounded_memory_transformer/memory_benchmark/generator.py",
        "src/bounded_memory_transformer/memory_benchmark/operations.py",
        "src/bounded_memory_transformer/memory_benchmark/policies.py",
        "src/bounded_memory_transformer/memory_benchmark/state_machine.py",
        "src/bounded_memory_transformer/memory_benchmark/views.py",
    ]
    runtime_files = [source_root / relative for relative in runtime_relative]
    # Namespace-package __init__ files are optional; every actual runtime file is hashed.
    runtime_files = [path for path in runtime_files if path.exists()]
    frozen_files = [root / relative for relative in config["frozen_sources"]]
    freeze_files = [*runtime_files, *frozen_files, config_path, protocol_path,
                    source_root / "tests/test_uniform_replication.py"]
    revision = verify_freeze(root, freeze_files, freeze_commit) if mode == "held-out" else None
    for relative, expected in config["frozen_sources"].items():
        if _sha256(root / relative) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")
    torch.set_num_threads(int(config["threads"]))
    models = {}
    checkpoint_details = []
    for item in config["checkpoints"]:
        path = root / item["path"]
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {item['path']}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = RetentionScorer().eval().requires_grad_(False)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[item["model_seed"]] = model
        checkpoint_details.append({**item, "training_config": payload["config"]})
    proof = verify_model_equivalence(models, config["capacities"])
    representative = models[proof["representative_model_seed"]]
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "incomplete", "mode": mode, "started_at": started_at,
        "freeze_commit": revision, "original_frozen_revision": config["original_frozen_revision"],
        "held_out_generated": False, "effective_evaluation": evaluation,
        "config_sha256": _sha256(config_path), "protocol_sha256": _sha256(protocol_path),
        "runtime_source_hashes": {str(path): _sha256(path) for path in runtime_files},
        "frozen_source_hashes": config["frozen_sources"], "checkpoints": checkpoint_details,
        "equivalence_proof": proof, "artifact_hashes": {},
        "environment": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "device": "cpu", "threads": torch.get_num_threads()},
        "completed_dataset_seeds": [],
    }

    def save(name: str, value: Any) -> None:
        path = output / name
        _save_json(path, value)
        manifest["artifact_hashes"][name] = _sha256(path)

    save("config.json", config)
    (output / "protocol.md").write_bytes(protocol_path.read_bytes())
    manifest["artifact_hashes"]["protocol.md"] = _sha256(output / "protocol.md")
    _save_json(output / "manifest.json", manifest)
    contrasts_by_seed = {}
    seed_summaries = {}
    generator = config["generator"]
    for seed in evaluation["dataset_seeds"]:
        print(f"Uniform replication: mode={mode} dataset_seed={seed}", flush=True)
        manifest["held_out_generated"] = mode == "held-out"
        _save_json(output / "manifest.json", manifest)
        episodes = generate_capacity(evaluation["split"], seed,
                                     evaluation["episodes_per_seed"], **generator)
        save(f"dataset-{seed}.json.gz", [asdict(episode) for episode in episodes])
        results = evaluate_uniform_dataset(episodes, representative, config["capacities"],
                                           config["random_policy_seeds"], keys=generator["keys"])
        contrasts_by_seed[seed] = build_contrasts(results, config["capacities"],
                                                 config["random_policy_seeds"])
        policies = {}
        for name, result in results.items():
            filename = f"dataset-{seed}-{name.replace(':', '-')}.json.gz"
            save(filename, result)
            policies[name] = {**result["summary"], "artifact": filename,
                              "expected_uniform_recall": result["summary"]["capacity"]
                              / generator["keys"]}
        for capacity in config["capacities"]:
            random = [policies[f"K{capacity}:random:{s}"] for s in config["random_policy_seeds"]]
            policies[f"K{capacity}:random_mean"] = {
                "utility": float(np.mean([row["utility"] for row in random])),
                "episodes": len(episodes), "queries": len(episodes) * generator["queries"],
                "capacity": capacity, "persistent_state_bytes_per_actor": capacity * 16,
                "expected_uniform_recall": capacity / generator["keys"],
                "interpretation": "mean of separate actors; no extra independent episodes",
                "write_seconds": sum(row["write_seconds"] for row in random),
                "read_seconds": sum(row["read_seconds"] for row in random),
            }
        seed_summaries[str(seed)] = policies
        manifest["completed_dataset_seeds"].append(seed)
        _save_json(output / "manifest.json", manifest)
    analysis = analyze_contrasts(contrasts_by_seed, inference)
    pooled_policies = {}
    for name in next(iter(seed_summaries.values())):
        rows = [policies[name] for policies in seed_summaries.values()]
        pooled_policies[name] = {
            "utility": float(np.mean([row["utility"] for row in rows])),
            "dataset_seed_standard_deviation": (float(np.std([row["utility"] for row in rows],
                                                              ddof=1)) if len(rows) > 1 else None),
            "episodes": sum(row["episodes"] for row in rows),
            "queries": sum(row["queries"] for row in rows),
            "expected_uniform_recall": rows[0]["expected_uniform_recall"],
            "write_seconds": sum(row["write_seconds"] for row in rows),
            "read_seconds": sum(row["read_seconds"] for row in rows),
        }
    summary = {
        "mode": mode, "freeze_commit": revision, "effective_evaluation": evaluation,
        "interpretation": ("fixed held-out replication" if mode == "held-out"
                           else "training-only pipeline smoke; not held-out evidence"),
        "equivalence_proof": proof, "per_seed_policies": seed_summaries,
        "pooled_policies": pooled_policies, "analysis": analysis,
        "wall_seconds": time.perf_counter() - started,
    }
    save("summary.json", summary)
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                    wall_seconds=time.perf_counter() - started)
    _save_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("projects/04-memory-followup/configs/uniform-replication.json"))
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="training-only pipeline check")
    mode.add_argument("--held-out", action="store_true",
                      help="open the preregistered test episodes")
    parser.add_argument("--freeze-commit", help="reviewed commit required for held-out mode")
    args = parser.parse_args()
    result = run_replication(args.config, args.output, mode="smoke" if args.smoke else "held-out",
                             freeze_commit=args.freeze_commit)
    print(f"Complete: {args.output} ({result['wall_seconds']:.2f} seconds)", flush=True)


if __name__ == "__main__":
    main()
