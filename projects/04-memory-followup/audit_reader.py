"""Audit saved reader results without importing models, generators, or metrics.

Only NumPy is used to recompute the paired episode bootstrap. PyTorch is loaded
inside checkpoint verification solely to inspect saved metadata, never inference.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UNKNOWN = "??"
STRATA = ("select", "unsupported", "contradicted", "irrelevant", "deleted")
EVIDENCE_FIELDS = ("memory", "current", "query", "world", "state_bytes", "episode_id")
POLICIES = ("fifo", "recency", "random", "similarity", "cue_priority", "learned")


class AuditError(ValueError):
    """A saved artifact disagrees with its independent reconstruction."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def equal(expected, observed, location):
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"{location}: expected a mapping")
        require(set(expected) <= set(observed), f"{location}: missing fields")
        for name, value in expected.items():
            equal(value, observed[name], f"{location}/{name}")
    elif isinstance(expected, list):
        require(
            isinstance(observed, list) and len(expected) == len(observed),
            f"{location}: list length mismatch",
        )
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            equal(left, right, f"{location}/{index}")
    elif isinstance(expected, bool) or expected is None:
        require(expected is observed, f"{location}: {observed!r} != {expected!r}")
    elif isinstance(expected, float):
        require(
            isinstance(observed, (float, int))
            and not isinstance(observed, bool)
            and math.isfinite(observed)
            and math.isclose(expected, observed, rel_tol=1e-11, abs_tol=1e-12),
            f"{location}: {observed!r} != {expected!r}",
        )
    else:
        require(
            type(expected) is type(observed) and expected == observed,
            f"{location}: {observed!r} != {expected!r}",
        )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def key(operation):
    return operation["entity"], operation["attribute"]


def validate_operation(operation, *, query=False):
    require(set(operation) == {"kind", "entity", "attribute", "value"}, "invalid operation fields")
    allowed = (5,) if query else (1, 2, 3, 4)
    require(
        type(operation["kind"]) is int and operation["kind"] in allowed,
        "invalid query/evidence operation kind",
    )
    require(
        type(operation["entity"]) is int and 0 <= operation["entity"] < 100, "invalid entity symbol"
    )
    require(
        type(operation["attribute"]) is int and 0 <= operation["attribute"] < 4,
        "invalid attribute symbol",
    )
    value = operation["value"]
    require(
        value is None if operation["kind"] in (3, 5) else type(value) is int and 0 <= value < 100,
        "invalid operation value",
    )


def authoritative(operations, query):
    """Last matching SET/UPDATE/DELETE; matching NOISE has no authority."""
    answer, selected = UNKNOWN, -1
    for index, operation in enumerate(operations):
        if key(operation) == key(query) and operation["kind"] in (1, 2, 3):
            selected = index
            answer = UNKNOWN if operation["kind"] == 3 else f"{operation['value']:02d}"
    return answer, selected


def copied(operations, index):
    """Copy exactly the selected record, including wrong keys and NOISE."""
    require(type(index) is int and -1 <= index < len(operations), "selection index out of bounds")
    if index == -1 or operations[index]["value"] is None:
        return UNKNOWN
    return f"{operations[index]['value']:02d}"


def paired_statistics(rows, bootstrap_samples):
    require(bootstrap_samples > 0, "bootstrap count must be positive")
    clusters = {}
    correct = baseline = harmful = beneficial = 0
    for row in rows:
        good = row["answer"] == row["world"]
        base = row["no_memory_answer"] == row["world"]
        correct += good
        baseline += base
        harmful += base and not good
        beneficial += good and not base
        total, count = clusters.get(row["episode_id"], (0, 0))
        clusters[row["episode_id"]] = (total + int(good) - int(base), count + 1)
    totals, counts = np.asarray(list(clusters.values()), dtype=np.int64).T
    rng = np.random.default_rng(8128)
    samples = np.empty(bootstrap_samples)
    # Independent implementation: bounded batches of whole episode identities.
    for start in range(0, bootstrap_samples, 64):
        draws = rng.integers(
            len(clusters), size=(min(64, bootstrap_samples - start), len(clusters))
        )
        samples[start : start + len(draws)] = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    count = len(rows)
    return {
        "queries": count,
        "unique_episodes": len(clusters),
        "accuracy": correct / count,
        "baseline_accuracy": baseline / count,
        "delta": (beneficial - harmful) / count,
        "harmful_count": harmful,
        "beneficial_count": beneficial,
        "harmful_rate": harmful / count,
        "beneficial_rate": beneficial / count,
        "delta_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def category_universe(condition, present=None):
    labels = []
    micro = re.fullmatch(r"(original|varied)-k([48])(?:-c(\d+))?", condition or "")
    if micro:
        slots = int(micro[2])
        labels = [
            *STRATA,
            *[f"occupancy_{k}" for k in range(slots + 1)],
            "source_memory",
            "source_current",
            "source_none",
        ]
    elif re.fullmatch(r"authority-k([48])", condition or ""):
        slots = int(condition[-1])
        transitions = [
            f"transition_{first}_{last}"
            for first in ("SET", "UPDATE", "DELETE")
            for last in ("SET", "UPDATE", "DELETE")
        ]
        if present is not None:
            transitions = [label for label in transitions if label in present]
        labels = [
            *STRATA,
            *transitions,
            *[f"occupancy_{k}" for k in range(slots + 1)],
            "source_memory",
            "source_current",
            "source_none",
        ]
    elif condition == "lifecycle":
        labels = ["update", "delete", "control"]
    return [*labels, "visible_known", "visible_unknown"]


def micro_categories(row, condition, position):
    authority = re.fullmatch(r"authority-k([48])", condition or "")
    if authority:
        slots = int(authority[1])
        equal(position, row["episode_id"], "authority episode order")
        equal(slots, len(row["memory"]), "authority occupancy")
        equal(slots * 16, row["state_bytes"], "authority capacity")
        in_current = (position // 9) % 2 == 1
        equal(32 if in_current else 0, len(row["current"]), "authority current-record count")
        equal(row["visible"], row["world"], "authority world target")
        matching = [
            operation
            for operation in row["memory"] + row["current"]
            if key(operation) == key(row["query"]) and operation["kind"] in (1, 2, 3)
        ]
        require(len(matching) == 2, "authority challenge requires exactly two matching operations")
        first, last = [operation["kind"] for operation in matching]
        equal(
            [(position % 9) // 3 + 1, position % 3 + 1],
            [first, last],
            "predeclared authoritative transition",
        )
        names = {1: "SET", 2: "UPDATE", 3: "DELETE"}
        origin = "current" if row["oracle"] >= len(row["memory"]) else "memory"
        equal("current" if in_current else "memory", origin, "authority evidence origin")
        return [
            f"transition_{names[first]}_{names[last]}",
            f"occupancy_{slots}",
            f"source_{origin}",
        ]
    micro = re.fullmatch(r"(original|varied)-k([48])(?:-c(\d+))?", condition or "")
    if micro is None:
        return None
    original, slots = micro[1] == "original", int(micro[2])
    require((micro[3] is None) == original, "invalid microtask condition name")
    equal(position, row["episode_id"], "microtask episode order")
    occupancy = slots if original else (position // 5) % (slots + 1)
    current = 6 if original else int(micro[3])
    equal(occupancy, len(row["memory"]), "microtask occupancy")
    equal(current, len(row["current"]), "microtask current-record count")
    equal(slots * 16, row["state_bytes"], "microtask capacity")
    equal(row["visible"], row["world"], "microtask world target")
    stratum = STRATA[position % 5]
    if not original:
        if occupancy + current == 0:
            stratum = "unsupported"
        elif stratum == "irrelevant" and current == 0:
            stratum = "select"
        elif stratum == "contradicted" and occupancy + current < 2:
            stratum = "select"
    origin = "none" if row["oracle"] < 0 else "memory" if row["oracle"] < occupancy else "current"
    return [stratum, f"occupancy_{occupancy}", f"source_{origin}"]


def recompute_rows(rows, *, bootstrap_samples=2000, condition=None, category_overrides=None):
    require(bool(rows), "empty reader condition")
    digest = hashlib.sha256()
    visible_correct, world_correct, oracle_correct, empty_correct = [], [], [], []
    categories = {}
    v1_flags = {("v1_selected" in row, "v1_answer" in row) for row in rows}
    require(v1_flags in ({(True, True)}, {(False, False)}), "incomplete frozen-v1 records")
    has_v1 = (True, True) in v1_flags
    if condition is not None and condition.startswith("original-"):
        require(has_v1, "original tasks require the frozen-v1 copy control")
    for index, row in enumerate(rows):
        records = row["memory"] + row["current"]
        validate_operation(row["query"], query=True)
        for operation in records:
            validate_operation(operation)
        require(type(row["episode_id"]) is int and row["episode_id"] >= 0, "invalid episode id")
        require(
            type(row["state_bytes"]) is int
            and row["state_bytes"] % 16 == 0
            and row["state_bytes"] >= 16 * len(row["memory"]),
            "invalid memory budget",
        )
        require(
            isinstance(row["world"], str) and re.fullmatch(r"\?\?|\d{2}", row["world"]),
            "invalid world target",
        )
        visible, oracle = authoritative(records, row["query"])
        current = authoritative(row["current"], row["query"])[0]
        reconstructed = {
            "visible": visible,
            "current_only": current,
            "oracle": oracle,
            "answer": copied(records, row["selected"]),
            "no_memory_answer": copied(row["current"], row["no_memory_selected"]),
        }
        if has_v1:
            reconstructed["v1_answer"] = copied(records, row["v1_selected"])
        equal(reconstructed, row, f"row/{index}")
        require(len(set(row["categories"])) == len(row["categories"]), "duplicate category labels")
        labels = set(row["categories"])
        expected_known = "visible_unknown" if visible == UNKNOWN else "visible_known"
        equal(
            [expected_known],
            sorted(labels & {"visible_known", "visible_unknown"}),
            f"row/{index}/known-category",
        )
        expected_categories = micro_categories(row, condition, index)
        if category_overrides is not None:
            expected_categories = category_overrides[index]
        if expected_categories is not None:
            equal(
                sorted(set(expected_categories) | {expected_known}),
                sorted(labels),
                f"row/{index}/categories",
            )
        if condition is not None:
            require(labels <= set(category_universe(condition)), "unexpected categories")
        visible_correct.append(row["answer"] == visible)
        world_correct.append(row["answer"] == row["world"])
        oracle_correct.append(visible == row["world"])
        empty_correct.append(current == row["world"])
        for label in row["categories"]:
            categories.setdefault(label, []).append(index)
        digest.update(
            json.dumps({field: row[field] for field in EVIDENCE_FIELDS}, sort_keys=True).encode()
        )
    count = len(rows)
    neural, oracle, empty = (
        sum(values) / count for values in (world_correct, oracle_correct, empty_correct)
    )
    gain = oracle - empty
    value = (neural - empty) / gain if gain > 0 else None
    by_category = {
        label: {
            "count": len(indices),
            "accuracy": sum(visible_correct[i] for i in indices) / len(indices),
        }
        for label, indices in categories.items()
    }
    absolute = sum(visible_correct) / count >= 0.95 and all(
        category["accuracy"] >= 0.95 for category in by_category.values()
    )
    relative = value >= 0.95 if value is not None else None
    unknown = sum(row["world"] == UNKNOWN for row in rows)
    abstentions = sum(row["answer"] == UNKNOWN for row in rows)
    true_abstentions = sum(row["answer"] == row["world"] == UNKNOWN for row in rows)
    universe = category_universe(condition, categories)
    result = {
        "queries": count,
        "visible_accuracy": sum(visible_correct) / count,
        "world_accuracy": neural,
        "oracle_accuracy": oracle,
        "reader_recovery": {
            "value": value,
            "reason": None if gain > 0 else "no_positive_oracle_gain",
            "neural_accuracy": neural,
            "oracle_accuracy": oracle,
            "no_evidence_accuracy": empty,
            "oracle_gain": gain,
        },
        "by_category": by_category,
        "absolute_gate_passed": absolute,
        "relative_gate_passed": relative,
        "passed": absolute and relative is True,
        "oracle_correct_neural_wrong": sum(
            o and not n for o, n in zip(oracle_correct, world_correct, strict=True)
        ),
        "oracle_wrong_neural_correct": sum(
            n and not o for o, n in zip(oracle_correct, world_correct, strict=True)
        ),
        "unknown_queries": unknown,
        "abstentions": abstentions,
        "abstention_precision": true_abstentions / abstentions if abstentions else None,
        "abstention_recall": true_abstentions / unknown if unknown else None,
        "abstention_f1": 2 * true_abstentions / (unknown + abstentions)
        if unknown + abstentions
        else None,
        "paired": paired_statistics(rows, bootstrap_samples),
        "selection_index_accuracy": sum(row["selected"] == row["oracle"] for row in rows) / count,
        "evidence_sha256": digest.hexdigest(),
        "absent_categories": [label for label in universe if label not in categories],
    }
    if has_v1:
        result["frozen_v1_copy_accuracy"] = (
            sum(row["v1_answer"] == row["visible"] for row in rows) / count
        )
    return result


def audit_condition(
    path, reported, *, bootstrap_samples=2000, condition=None, category_overrides=None
):
    equal(reported["rows_sha256"], sha256(path), "row artifact hash")
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    fresh = recompute_rows(
        rows,
        bootstrap_samples=bootstrap_samples,
        condition=condition,
        category_overrides=category_overrides,
    )
    require(
        len(set(reported["absent_categories"])) == len(reported["absent_categories"]),
        "duplicate absent categories",
    )
    equal(
        {**fresh, "absent_categories": sorted(fresh["absent_categories"])},
        {**reported, "absent_categories": sorted(reported["absent_categories"])},
        str(path),
    )
    equal(
        set(fresh) | {"rows_sha256", "inference_seconds_with_controls"},
        set(reported),
        "report fields",
    )
    require(
        math.isfinite(reported["inference_seconds_with_controls"])
        and reported["inference_seconds_with_controls"] >= 0,
        "invalid recorded latency",
    )
    return fresh


def expected_conditions(config):
    equal(0.95, config["threshold"], "fixed threshold")
    equal([4, 8], config["capacities"], "fixed capacities")
    equal([0, 1, 6, 32], config["current_counts"], "fixed current counts")
    equal(list(POLICIES), config["capacity_policies"], "fixed capacity policies")
    conditions = ["lifecycle"]
    for slots in (4, 8):
        if "authority_count" in config:
            require(config["authority_count"] > 0, "authority challenge count must be positive")
            conditions.append(f"authority-k{slots}")
        conditions.append(f"original-k{slots}")
        conditions.extend(f"varied-k{slots}-c{count}" for count in (0, 1, 6, 32))
        for workload in ("A", "B"):
            conditions.append(f"combined-{workload}-k{slots}")
            conditions.extend(f"capacity-{workload}-k{slots}-{policy}" for policy in POLICIES)
    expected = 41 if "authority_count" in config else 39
    require(len(conditions) == expected, "protocol condition count mismatch")
    return conditions


def verify_file_hashes(hashes, root):
    require(bool(hashes), "missing hash inventory")
    for name, expected in hashes.items():
        path = root / name
        require(path.is_file(), f"missing hashed file: {name}")
        equal(expected, sha256(path), f"file hash/{name}")
    return len(hashes)


def verify_checkpoints(hashes, seeds, root):
    import torch

    verified = {}
    paths = {(root / name).resolve() for name in hashes}
    for seed in seeds:
        logs = [path for path in paths if path.name == f"training-{seed}.json"]
        require(len(logs) == 1, f"need one hashed training log for seed {seed}")
        log_path = logs[0]
        checkpoint = log_path.with_name(f"reader-{seed}.pt")
        require(checkpoint in paths, "trained reader checkpoint is missing from hash inventory")
        readers = [path for path in paths if path.name == checkpoint.name and path != checkpoint]
        require(len(readers) == 1, f"need one frozen-v1 reader checkpoint for seed {seed}")
        for component in ("lifecycle", "capacity"):
            require(
                readers[0].with_name(f"{component}-{seed}.pt") in paths,
                f"missing frozen {component} checkpoint for seed {seed}",
            )
        saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
        metadata = saved["metadata"]
        training = load_json(log_path)
        equal(set(metadata), set(training), "checkpoint/training metadata fields")
        equal(metadata, training, "checkpoint/training metadata")
        equal(seed, metadata["seed"], "checkpoint seed")
        config = load_json(log_path.with_name("config.json"))
        equal(metadata["config"], config, "checkpoint/training config")
        equal(set(metadata["config"]), set(config), "checkpoint/training config fields")
        equal(config["model"], saved["settings"], "checkpoint model settings")
        count = sum(tensor.numel() for tensor in saved["state"].values())
        equal(metadata["parameters"], count, "checkpoint parameter count")
        verified[str(seed)] = {
            "reader": str(checkpoint),
            "frozen_reader": str(readers[0]),
            "parameters": count,
            "settings": saved["settings"],
            "training_config_sha256": sha256(log_path.with_name("config.json")),
        }
    return verified


def bank_records(bank, slots):
    equal(slots, len(bank), "bank slot count")
    records = []
    for row in bank:
        require(
            len(row) == 4 and all(type(value) is int for value in row), "invalid int32 bank row"
        )
        if row[0] < 0:
            equal([-1] * 4, row, "empty bank row")
            continue
        entity, attribute, value, cue = row
        require(cue in (0, 1), "invalid observed cue")
        record = {
            "kind": 3 if value == -1 else 1,
            "entity": entity,
            "attribute": attribute,
            "value": None if value == -1 else value,
        }
        validate_operation(record)
        records.append(record)
    return records


def write_prefixes(writes):
    prefixes = defaultdict(list)
    last_position = {}
    for row in writes:
        identity = row["episode_id"]
        position = row["session"], row["operation_index"]
        require(position > last_position.get(identity, (-1, -1)), "unordered/duplicate write trace")
        validate_operation(row["operation"])
        prefixes[identity].append(row["operation"])
        last_position[identity] = position
    return dict(prefixes)


def verify_capacity_evidence(raw, rows, *, slots, policy, prefixes):
    flattened = [(episode, record) for episode in raw["episodes"] for record in episode["records"]]
    equal(len(flattened), len(rows), "capacity reader row count")
    total_hits = total_oracle = total_queries = 0
    for episode in raw["episodes"]:
        identity = episode["episode_id"]
        require(identity in prefixes, "capacity episode missing write prefix")
        records = bank_records(episode["final_bank"], slots)
        hits = 0
        counts = Counter()
        for index, record in enumerate(episode["records"]):
            query = {
                "kind": 5,
                "entity": record["entity"],
                "attribute": record["attribute"],
                "value": None,
            }
            validate_operation(query, query=True)
            world = authoritative(prefixes[identity], query)[0]
            answer = authoritative(records, query)[0]
            equal(
                {
                    "query_index": index,
                    "truth": world,
                    "answer": answer,
                    "correct": answer == world,
                    "historical_answerable": world != UNKNOWN,
                },
                record,
                "capacity raw query/world truth",
            )
            hits += answer == world
            counts[key(query)] += 1
        count = len(episode["records"])
        require(count > 0, "capacity episode has no probes")
        oracle_hits = sum(sorted(counts.values(), reverse=True)[:slots])
        equal(
            {
                "queries": count,
                "hits": hits,
                "utility": hits / count,
                "oracle_hits": oracle_hits,
                "oracle_utility": oracle_hits / count,
                "oracle_regret": (oracle_hits - hits) / count,
            },
            episode,
            "capacity raw episode",
        )
        total_hits += hits
        total_oracle += oracle_hits
        total_queries += count
    for (episode, record), row in zip(flattened, rows, strict=True):
        query = {
            "kind": 5,
            "entity": record["entity"],
            "attribute": record["attribute"],
            "value": None,
        }
        visible = bank_records(episode["final_bank"], slots)
        if policy == "similarity" and visible:
            requested = f"{query['entity']:02d}{query['attribute']}"
            scores = []
            for position, item in enumerate(visible):
                candidate = f"{item['entity']:02d}{item['attribute']}"
                overlap = sum(a == b for a, b in zip(requested, candidate, strict=True))
                scores.append((overlap, position))
            visible = [visible[max(scores)[1]]]
        equal(
            {
                "memory": visible,
                "current": [],
                "query": query,
                "world": record["truth"],
                "state_bytes": slots * 16,
                "episode_id": episode["episode_id"],
            },
            row,
            "capacity reader evidence/world",
        )
    require(total_queries > 0, "capacity condition has no queries")
    equal(
        {
            "queries": total_queries,
            "utility": total_hits / total_queries,
            "oracle_utility": total_oracle / total_queries,
            "oracle_regret": (total_oracle - total_hits) / total_queries,
        },
        raw["summary"],
        "capacity raw summary",
    )
    byte_field = (
        "persistent_state_bytes" if "persistent_state_bytes" in raw["summary"] else "state_bytes"
    )
    equal(slots * 16, raw["summary"][byte_field], "capacity raw bank bytes")


def verify_lifecycle_evidence(raw, rows):
    equal(len(raw["queries"]), len(rows), "lifecycle reader row count")
    operations = defaultdict(list)
    kinds = {"SET": 1, "UPDATE": 2, "DELETE": 3, "NOISE": 4}
    for record in raw["operations"]:
        operation = {
            "kind": kinds[record["input_kind"]],
            "entity": record["entity"],
            "attribute": record["attribute"],
            "value": record["value"],
        }
        validate_operation(operation)
        operations[record["episode_id"]].append(
            (record["session"], record["operation_index"], operation, record["bank_after"])
        )
    overrides = []
    for query, row in zip(raw["queries"], rows, strict=True):
        identity = query["episode_id"]
        position = query["session"], query["operation_index"]
        preceding = [record for record in operations[identity] if record[:2] < position]
        require(bool(preceding), "lifecycle query has no preceding writes")
        ask = {"kind": 5, "entity": query["entity"], "attribute": query["attribute"], "value": None}
        truth = authoritative([record[2] for record in preceding], ask)[0]
        bank = bank_records(query["bank"], 4)
        exact = authoritative(bank, ask)[0]
        control = query["operation_index"] == 1
        case = "delete" if identity % 2 else "update"
        equal(
            {
                "expected": truth,
                "answer": exact,
                "correct": exact == truth,
                "case": case,
                "control": control,
                "bank": preceding[-1][3],
            },
            query,
            "lifecycle raw query",
        )
        equal(
            {
                "memory": bank,
                "current": [],
                "query": ask,
                "world": truth,
                "state_bytes": 64,
                "episode_id": identity,
            },
            row,
            "lifecycle reader evidence/world",
        )
        overrides.append(["control" if control else case])
    return overrides


def load_rows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_run(run, *, source_root=None):
    run = Path(run).resolve()
    root = Path(source_root).resolve() if source_root else Path(__file__).resolve().parents[2]
    require((run / "summary.json").is_file(), "incomplete run: summary.json missing")
    summary = load_json(run / "summary.json")
    require(
        {"seconds", "passed", "artifact_hashes", "hashes", "config", "seeds"} <= set(summary),
        "incomplete run: completion fields missing",
    )
    config = summary["config"]
    conditions = expected_conditions(config)
    seeds = config["seeds"]
    require(seeds and len(set(seeds)) == len(seeds), "invalid seed inventory")
    equal({str(seed) for seed in seeds}, set(summary["seeds"]), "completed seeds")
    expected_artifacts = {"frozen-manifest.json"}
    for seed in seeds:
        equal(
            set(conditions),
            set(summary["seeds"][str(seed)]["conditions"]),
            f"seed/{seed}/complete conditions",
        )
        expected_artifacts.add(f"lifecycle-{seed}.json")
        for condition in conditions:
            expected_artifacts.add(f"seed-{seed}/{condition}.jsonl.gz")
            if condition.startswith(("capacity-", "combined-")):
                expected_artifacts.add(f"{condition}-{seed}.json")
    equal(expected_artifacts, set(summary["artifact_hashes"]), "complete artifact hash inventory")
    artifact_count = verify_file_hashes(summary["artifact_hashes"], run)
    frozen = load_json(run / "frozen-manifest.json")
    equal({}, frozen["seeds"], "manifest was frozen before any seed result")
    for field in ("config", "hashes", "smoke", "frozen_utc", "git_commit", "environment"):
        equal(frozen[field], summary[field], f"frozen manifest/{field}")
        if isinstance(frozen[field], dict):
            equal(set(frozen[field]), set(summary[field]), f"frozen manifest/{field}/fields")
    source_count = verify_file_hashes(summary["hashes"], root)
    source_names = set(summary["hashes"])
    required_sources = [
        "memory_followup/reader.py",
        "memory_followup/tasks.py",
        "memory_followup/evaluate_reader.py",
        "memory_experiments/metrics.py",
        "memory_experiments/reader_cases.py",
        "memory_experiments/capacity.py",
        "memory_experiments/lifecycle.py",
        "memory_experiments/run.py",
        "memory_benchmark/operations.py",
        "memory_benchmark/views.py",
    ]
    for suffix in required_sources:
        require(
            any(name.endswith("src/bounded_memory_transformer/" + suffix) for name in source_names),
            f"runtime source missing from hash inventory: {suffix}",
        )
    evaluation_paths = [
        root / name for name in source_names if name.endswith("reader-evaluation.json")
    ]
    require(len(evaluation_paths) == 1, "expected one hashed reader-evaluation configuration")
    declared_config = load_json(evaluation_paths[0])
    equal(set(declared_config), set(config), "evaluation configuration fields")
    if summary["smoke"]:
        require(config["split"] in ("train", "validation"), "smoke opened held-out symbols")
        overrides = {
            "split",
            "device",
            "seeds",
            "original_count",
            "varied_count",
            "authority_count",
            "lifecycle_count",
            "capacity_count",
            "bootstrap_samples",
        }
        equal(
            {k: v for k, v in declared_config.items() if k not in overrides},
            config,
            "declared smoke configuration",
        )
        require(set(seeds) <= set(declared_config["seeds"]), "undeclared smoke model seed")
    else:
        equal(declared_config, config, "declared held-out configuration")
        equal("test", config["split"], "held-out split")
        equal(2000, config["bootstrap_samples"], "predeclared bootstrap count")
    checkpoints = verify_checkpoints(summary["hashes"], seeds, root)
    recomputed, seed_passes = {}, {}
    query_count = 0
    shared_micro = {}
    for seed in seeds:
        prefixes = {}
        for workload in ("A", "B"):
            shared = None
            for slots in (4, 8):
                raw = load_json(run / f"combined-{workload}-k{slots}-{seed}.json")
                current = write_prefixes(raw["writes"])
                equal(
                    set(range(config["capacity_count"])),
                    set(current),
                    "capacity episode identities",
                )
                require(
                    all(len(prefix) == 30 for prefix in current.values()),
                    "capacity prefix must contain the predeclared 30 writes",
                )
                if shared is not None:
                    equal(shared, current, "shared capacity write prefixes across slot counts")
                shared = current
            prefixes[workload] = shared
        equal(prefixes["A"], prefixes["B"], "shared A/B write prefixes")
        seed_reports = {}
        shared_queries = {}
        for condition in conditions:
            path = run / f"seed-{seed}/{condition}.jsonl.gz"
            rows = load_rows(path)
            require(bool(rows), "empty condition file")
            overrides = None
            if condition.startswith(("original-", "varied-", "authority-")):
                family = condition.split("-")[0]
                equal(config[f"{family}_count"], len(rows), f"{condition}/sample count")
            elif condition == "lifecycle":
                equal(2 * config["lifecycle_count"], len(rows), "lifecycle sample count")
                raw = load_json(run / f"lifecycle-{seed}.json")
                overrides = verify_lifecycle_evidence(raw, rows)
            else:
                match = re.fullmatch(r"(capacity|combined)-([AB])-k([48])(?:-(\w+))?", condition)
                require(match is not None, "invalid capacity condition name")
                family, workload, slots, policy = match.groups()
                slots = int(slots)
                equal(32 * config["capacity_count"], len(rows), "capacity sample count")
                raw = load_json(run / f"{condition}-{seed}.json")
                if family == "combined":
                    raw = raw["result"]
                verify_capacity_evidence(
                    raw, rows, slots=slots, policy=policy, prefixes=prefixes[workload]
                )
                queries = [
                    {"episode_id": row["episode_id"], "query": row["query"], "world": row["world"]}
                    for row in rows
                ]
                if workload in shared_queries:
                    equal(shared_queries[workload], queries, "paired capacity queries/world")
                shared_queries[workload] = queries
            report = audit_condition(
                path,
                summary["seeds"][str(seed)]["conditions"][condition],
                bootstrap_samples=config["bootstrap_samples"],
                condition=condition,
                category_overrides=overrides,
            )
            if condition.startswith(("original-", "varied-", "authority-")):
                if condition in shared_micro:
                    equal(
                        shared_micro[condition],
                        report["evidence_sha256"],
                        "shared microtask evidence",
                    )
                shared_micro[condition] = report["evidence_sha256"]
            seed_reports[condition] = report
            query_count += len(rows)
        passed = all(report["passed"] for report in seed_reports.values())
        failed = sorted(name for name, report in seed_reports.items() if not report["passed"])
        equal(passed, summary["seeds"][str(seed)]["passed"], "per-seed gate conjunction")
        equal(
            failed, sorted(summary["seeds"][str(seed)]["failed"]), "per-seed failed condition list"
        )
        seed_passes[str(seed)] = passed
        recomputed[str(seed)] = seed_reports
    gate = all(seed_passes.values())
    equal(gate, summary["passed"], "overall gate conjunction")
    return {
        "passed": True,
        "reader_gates_passed": gate,
        "seed_gates": seed_passes,
        "run": str(run),
        "source_revision": summary["git_commit"],
        "smoke": summary["smoke"],
        "auditor_sha256": sha256(Path(__file__)),
        "summary_sha256": sha256(run / "summary.json"),
        "counts": {
            "artifact_hashes": artifact_count,
            "source_and_checkpoint_hashes": source_count,
            "checkpoints": 4 * len(seeds),
            "conditions": len(conditions) * len(seeds),
            "reader_rows": query_count,
        },
        "checkpoints": checkpoints,
        "recomputed": recomputed,
        "limitations": [
            "Hashes check the recorded files against current source/checkpoint bytes.",
            "Neural indices are verified by copying, not by rerunning model inference.",
            "Timing and training causality require source/run review.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = audit_run(args.run, source_root=args.source_root)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result = {"passed": False, "failures": [str(error)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("recomputed", "checkpoints")}, indent=2
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
