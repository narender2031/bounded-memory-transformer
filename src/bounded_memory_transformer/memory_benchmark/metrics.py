"""Paired, denominator-explicit metrics independent of the model and policies."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .operations import UNKNOWN, QueryTruth


@dataclass(frozen=True)
class Result:
    episode_id: int
    model_seed: int
    case: str
    truth: QueryTruth
    prediction: str
    no_memory_prediction: str
    useful_retained: bool = False


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: Sequence[Result], *, bootstrap_samples: int = 2000, seed: int = 8128) -> dict:
    if not rows or bootstrap_samples <= 0:
        raise ValueError("need predictions and a positive bootstrap sample count")
    correct = np.array([r.prediction == r.truth.answer for r in rows], dtype=float)
    baseline = np.array([r.no_memory_prediction == r.truth.answer for r in rows], dtype=float)
    unknown = sum(r.truth.answer == UNKNOWN for r in rows)
    abstained = sum(r.prediction == UNKNOWN for r in rows)
    true_abstained = sum(r.prediction == r.truth.answer == UNKNOWN for r in rows)
    updated = [r for r in rows if r.truth.overwrites > 0 and r.truth.answer != UNKNOWN]
    deleted = [r for r in rows if r.truth.deleted]
    historical = [
        r
        for r in rows
        if r.truth.answer != UNKNOWN
        and r.truth.evidence_gap is not None
        and r.truth.evidence_gap > 0
    ]
    stale = sum(r.prediction in r.truth.stale_values for r in updated)

    # All model seeds for the same episode remain together. The interval reflects
    # episode sampling uncertainty, CONDITIONAL on the trained seeds, not new models.
    clusters: dict[int, list[float]] = defaultdict(list)
    seeds: dict[int, list[bool]] = defaultdict(list)
    for row, delta in zip(rows, correct - baseline, strict=True):
        clusters[row.episode_id].append(float(delta))
        seeds[row.model_seed].append(row.prediction == row.truth.answer)
    cluster_deltas = np.array([np.mean(deltas) for deltas in clusters.values()])
    rng = np.random.default_rng(seed)
    # Small batches keep bootstrap memory bounded for larger evaluation sets.
    samples = []
    for start in range(0, bootstrap_samples, 100):
        indices = rng.integers(
            0, len(cluster_deltas), size=(min(100, bootstrap_samples - start), len(cluster_deltas))
        )
        samples.extend(cluster_deltas[indices].mean(axis=1).tolist())

    def slices(field: str) -> dict:
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            value = row.case if field == "case" else getattr(row.truth, field)
            groups[str(value)].append(row.prediction == row.truth.answer)
        return {
            key: {"count": len(items), "accuracy": float(np.mean(items))}
            for key, items in sorted(groups.items())
        }

    seed_means = {str(key): float(np.mean(values)) for key, values in seeds.items()}
    return {
        "queries": len(rows),
        "unique_episodes": len(clusters),
        "accuracy": float(correct.mean()),
        "no_memory_accuracy": float(baseline.mean()),
        "accuracy_delta": float((correct - baseline).mean()),
        "accuracy_delta_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "harmful_rate": float(((baseline == 1) & (correct == 0)).mean()),
        "beneficial_rate": float(((baseline == 0) & (correct == 1)).mean()),
        "updated_queries": len(updated),
        "stale_answers": stale,
        "stale_answer_rate": ratio(stale, len(updated)),
        "deleted_queries": len(deleted),
        "deleted_fact_leakage": ratio(sum(r.prediction != UNKNOWN for r in deleted), len(deleted)),
        "unknown_queries": unknown,
        "abstention_precision": ratio(true_abstained, abstained),
        "abstention_recall": ratio(true_abstained, unknown),
        "abstention_f1": ratio(2 * true_abstained, abstained + unknown),
        "historical_answerable_queries": len(historical),
        "useful_fact_retention": ratio(sum(r.useful_retained for r in historical), len(historical)),
        "by_case": slices("case"),
        "by_gap": slices("evidence_gap"),
        "by_overwrites": slices("overwrites"),
        "accuracy_by_seed": seed_means,
        "seed_accuracy_std": float(np.std(list(seed_means.values()), ddof=1))
        if len(seed_means) > 1
        else 0.0,
    }
