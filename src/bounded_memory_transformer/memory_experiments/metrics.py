"""Denominator-explicit reader controls; no changes to historical pilot metrics."""

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def recovery(neural: float, oracle: float, no_evidence: float) -> dict:
    values = (neural, oracle, no_evidence)
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("accuracies must be finite values in [0, 1]")
    gain = oracle - no_evidence
    return {
        "value": (neural - no_evidence) / gain if gain > 0 else None,
        "reason": None if gain > 0 else "no_positive_oracle_gain",
        "neural_accuracy": neural,
        "oracle_accuracy": oracle,
        "no_evidence_accuracy": no_evidence,
        "oracle_gain": gain,
    }


def _aligned(*columns: Sequence) -> int:
    lengths = {len(column) for column in columns}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("columns must be nonempty and have equal length")
    return len(columns[0])


def paired_accuracy(
    predictions: Sequence[str],
    baseline: Sequence[str],
    targets: Sequence[str],
    *,
    episode_ids: Sequence[int] | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 8128,
) -> dict:
    count = _aligned(predictions, baseline, targets)
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    ids = list(range(count)) if episode_ids is None else episode_ids
    _aligned(predictions, ids)
    correct = np.array([p == t for p, t in zip(predictions, targets, strict=True)])
    base = np.array([p == t for p, t in zip(baseline, targets, strict=True)])
    delta = correct.astype(float) - base.astype(float)
    groups: dict[int, list[float]] = defaultdict(list)
    for key, value in zip(ids, delta, strict=True):
        groups[key].append(float(value))
    # Resample whole episodes, retaining every query/seed belonging to an episode.
    sums = np.array([sum(group) for group in groups.values()])
    sizes = np.array([len(group) for group in groups.values()])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, bootstrap_samples, 100):
        indices = rng.integers(
            0, len(groups), size=(min(100, bootstrap_samples - start), len(groups))
        )
        samples.extend((sums[indices].sum(1) / sizes[indices].sum(1)).tolist())
    return {
        "queries": count,
        "unique_episodes": len(groups),
        "accuracy": float(correct.mean()),
        "baseline_accuracy": float(base.mean()),
        "delta": float(delta.mean()),
        "harmful_count": int((base & ~correct).sum()),
        "beneficial_count": int((~base & correct).sum()),
        "harmful_rate": float((base & ~correct).mean()),
        "beneficial_rate": float((~base & correct).mean()),
        "delta_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
    }


def reader_report(
    *,
    predictions: Sequence[str],
    visible_targets: Sequence[str],
    world_targets: Sequence[str],
    no_evidence: Sequence[str],
    paired_no_memory: Sequence[str],
    categories: dict[str, Sequence[bool]],
    episode_ids: Sequence[int] | None = None,
    bootstrap_samples: int = 2000,
    threshold: float = 0.95,
) -> dict:
    count = _aligned(predictions, visible_targets, world_targets, no_evidence, paired_no_memory)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    for mask in categories.values():
        _aligned(predictions, mask)
    visible = np.array([p == t for p, t in zip(predictions, visible_targets, strict=True)])
    world = np.array([p == t for p, t in zip(predictions, world_targets, strict=True)])
    oracle = np.array([p == t for p, t in zip(visible_targets, world_targets, strict=True)])
    empty = np.array([p == t for p, t in zip(no_evidence, world_targets, strict=True)])
    by_category = {}
    for name, mask in categories.items():
        mask_array = np.asarray(mask, dtype=bool)
        by_category[name] = {
            "count": int(mask_array.sum()),
            "accuracy": float(visible[mask_array].mean()) if mask_array.any() else None,
        }
    relative = recovery(float(world.mean()), float(oracle.mean()), float(empty.mean()))
    absolute_passed = bool(
        visible.mean() >= threshold
        and all(
            row["accuracy"] is not None and row["accuracy"] >= threshold
            for row in by_category.values()
        )
    )
    unknown = np.array([answer == "??" for answer in world_targets])
    abstained = np.array([answer == "??" for answer in predictions])
    true_abstained = int((unknown & abstained).sum())
    unknown_count, abstained_count = int(unknown.sum()), int(abstained.sum())
    return {
        "queries": count,
        "visible_accuracy": float(visible.mean()),
        "world_accuracy": float(world.mean()),
        "oracle_accuracy": float(oracle.mean()),
        "reader_recovery": relative,
        "by_category": by_category,
        "absolute_gate_passed": absolute_passed,
        "relative_gate_passed": relative["value"] >= threshold
        if relative["value"] is not None
        else None,
        "oracle_correct_neural_wrong": int((oracle & ~world).sum()),
        "oracle_wrong_neural_correct": int((~oracle & world).sum()),
        "unknown_queries": unknown_count,
        "abstentions": abstained_count,
        "abstention_precision": true_abstained / abstained_count if abstained_count else None,
        "abstention_recall": true_abstained / unknown_count if unknown_count else None,
        "abstention_f1": 2 * true_abstained / (unknown_count + abstained_count)
        if unknown_count + abstained_count
        else None,
        "paired": paired_accuracy(
            predictions,
            paired_no_memory,
            world_targets,
            episode_ids=episode_ids,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def selection_diagnostics(
    selected: Sequence[int],
    oracle: Sequence[int],
    copied: Sequence[str],
    generated: Sequence[str],
    targets: Sequence[str],
) -> dict:
    count = _aligned(selected, oracle, copied, generated, targets)
    right_index = np.array([i == j for i, j in zip(selected, oracle, strict=True)])
    right_copy = np.array([p == t for p, t in zip(copied, targets, strict=True)])
    right_generation = np.array([p == t for p, t in zip(generated, targets, strict=True)])
    return {
        "queries": count,
        "selector_index_accuracy": float(right_index.mean()),
        "copy_accuracy": float(right_copy.mean()),
        "generation_accuracy": float(right_generation.mean()),
        "both_correct": int((right_copy & right_generation).sum()),
        "both_wrong": int((~right_copy & ~right_generation).sum()),
        "copy_correct_generation_wrong": int((right_copy & ~right_generation).sum()),
        "copy_wrong_generation_correct": int((~right_copy & right_generation).sum()),
        "wrong_index_correct_copy": int((~right_index & right_copy).sum()),
    }
