"""Make reviewable tables and a compact summary from immutable run artifacts."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean

ARMS = ("character_full", "selected_character", "selected_copy", "oracle_copy")
POLICIES = ("no_memory", "fifo", "recency", "random", "similarity", "cue_priority", "learned")


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def points(value):
    return "N/A" if value is None else f"{100 * value:.2f}"


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
                *("| " + " | ".join(str(x) for x in row) + " |" for row in rows),
            ]
        )
        + "\n"
    )


def summarize(run: Path, output: Path):
    raw = (run / "summary.json").read_bytes()
    summary = json.loads(raw)
    compact = copy.deepcopy(summary)
    for training in compact["training"].values():
        for component in training.values():
            if isinstance(component, dict):
                for name in ("curve", "selector_history", "character_history"):
                    component.pop(name, None)
    compact["original_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    compact["source_manifest"] = json.loads((run / "source-manifest.json").read_text())
    compact["raw_run_directory"] = str(run)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    seeds = {str(seed): summary["seeds"][str(seed)] for seed in summary["config"]["seeds"]}
    sections = [
        "# Five-case measured tables\n",
        "Generated from the saved run, without model inference. Percentages use the\n"
        "denominators in the machine-readable summary. Capacity rows average the same\n"
        "episodes across model seeds; this does not triple the independent sample size.\n",
        "## Reader arms: held-out visible accuracy\n",
        table(
            ["Seed", *ARMS, "Copy recovery"],
            [
                [
                    seed,
                    *(percent(row["reader"][arm]["visible_accuracy"]) for arm in ARMS),
                    percent(row["reader"]["selected_copy"]["reader_recovery"]["value"]),
                ]
                for seed, row in seeds.items()
            ],
        ),
        "## Reader category means\n",
        table(
            ["Category", *ARMS],
            [
                [
                    category,
                    *(
                        percent(
                            mean(
                                row["reader"][arm]["by_category"][category]["accuracy"]
                                for row in seeds.values()
                            )
                        )
                        for arm in ARMS
                    ),
                ]
                for category in (
                    "select",
                    "unsupported",
                    "contradicted",
                    "irrelevant",
                    "deleted",
                    "known",
                    "unknown",
                    "full_occupancy",
                )
            ],
        ),
        "## Reader paired harm and benefit\n",
        table(
            ["Seed", "Arm", "Harm / all queries", "Benefit / all queries", "Δ accuracy (pp)"],
            [
                [
                    seed,
                    arm,
                    percent(row["reader"][arm]["paired"]["harmful_rate"]),
                    percent(row["reader"][arm]["paired"]["beneficial_rate"]),
                    points(row["reader"][arm]["paired"]["delta"]),
                ]
                for seed, row in seeds.items()
                for arm in ARMS
            ],
        ),
        "## Lifecycle through the exact reader\n",
    ]
    metrics = (
        "update_accuracy",
        "delete_accuracy",
        "control_preservation",
        "action_macro_f1",
        "target_accuracy",
        "transition_accuracy",
        "stale_answer_rate",
        "deletion_nonabstention_rate",
        "deleted_value_repetition_rate",
    )
    sections.append(
        table(
            ["Metric", *seeds],
            [
                [metric, *(percent(row["lifecycle"]["learned"][metric]) for row in seeds.values())]
                for metric in metrics
            ],
        )
    )
    sections.append("## Capacity: exact-reader useful recall\n")
    conditions = [(w, k) for w in ("A", "B") for k in ("4", "8")]
    rows = [
        [
            policy,
            *(
                percent(
                    mean(
                        row["capacity"][w][k][policy]["exact"]["utility"] for row in seeds.values()
                    )
                )
                for w, k in conditions
            ),
        ]
        for policy in POLICIES
    ]
    rows.append(
        [
            "clairvoyant oracle",
            *(
                percent(
                    mean(
                        row["capacity"][w][k]["learned"]["exact"]["oracle_utility"]
                        for row in seeds.values()
                    )
                )
                for w, k in conditions
            ),
        ]
    )
    sections.append(table(["Policy", *(f"{w}, K={k}" for w, k in conditions)], rows))
    sections.append("## Learned capacity versus FIFO: paired episode bootstrap\n")
    rows = []
    for seed, row in seeds.items():
        for w, k in conditions:
            comparison = row["capacity"][w][k]["paired_comparisons"]["fifo"]
            low, high = comparison["delta_ci95"]
            rows.append(
                [seed, w, k, points(comparison["delta"]), f"[{points(low)}, {points(high)}]"]
            )
    sections.append(
        table(["Seed", "Workload", "Slots", "Gain (percentage points)", "95% paired CI (pp)"], rows)
    )
    sections.append("## Retention regret (oracle minus policy utility, percentage points)\n")
    sections.append(
        table(
            ["Policy", *(f"{w}, K={k}" for w, k in conditions)],
            [
                [
                    policy,
                    *(
                        points(
                            mean(
                                row["capacity"][w][k][policy]["exact"]["oracle_regret"]
                                for row in seeds.values()
                            )
                        )
                        for w, k in conditions
                    ),
                ]
                for policy in POLICIES
            ],
        )
    )
    sections.append("## Combined learned writer, retention, and selector/copy\n")
    rows = []
    for w, k in conditions:
        values = [row["end_to_end"]["combined_capacity"][w][k] for row in seeds.values()]
        recoveries = [v["neural"]["selected_copy"]["reader_recovery"]["value"] for v in values]
        rows.append(
            [
                w,
                k,
                percent(mean(v["exact"]["utility"] for v in values)),
                percent(mean(v["neural"]["selected_copy"]["world_accuracy"] for v in values)),
                percent(mean(recoveries)) if all(x is not None for x in recoveries) else "N/A",
            ]
        )
    sections.append(
        table(["Workload", "Slots", "Exact reading", "Learned reading", "Recovery"], rows)
    )
    sections.append("## Predeclared gates\n")
    gates = list(next(iter(seeds.values()))["gates"])
    sections.append(
        table(
            ["Seed", *gates],
            [[seed, *(str(row["gates"][gate]) for gate in gates)] for seed, row in seeds.items()],
        )
    )
    sections.append("## Training cost (seconds; reader times include validation)\n")
    sections.append(
        table(
            ["Seed", "Selector", "Character", "Lifecycle", "Capacity", "Total"],
            [
                [
                    seed,
                    f"{t['reader']['selector_training_seconds_including_validation']:.2f}",
                    f"{t['reader']['character_training_seconds_including_validation']:.2f}",
                    f"{t['lifecycle']['training_seconds']:.2f}",
                    f"{t['capacity']['training_seconds']:.2f}",
                    f"{t['total_seconds']:.2f}",
                ]
                for seed, t in summary["training"].items()
            ],
        )
    )
    sections.append("## Four-slot B timing by policy\n")
    sections.append(
        "Joint reader time includes all four arms and their paired no-memory controls;\n"
        "it is not a per-arm latency estimate. Exact-policy times exclude neural reading.\n"
    )
    rows = []
    for policy in POLICIES:
        values = [row["capacity"]["B"]["4"][policy] for row in seeds.values()]
        write_ms = mean(v["exact"]["write_seconds"] / v["exact"]["episodes"] * 1000 for v in values)
        read_us = mean(v["exact"]["read_seconds"] / v["exact"]["queries"] * 1e6 for v in values)
        joint_seconds = mean(
            v["diagnostics"]["four_arms_and_paired_controls_seconds"] for v in values
        )
        rows.append(
            [
                policy,
                f"{write_ms:.3f}",
                f"{read_us:.3f}",
                f"{joint_seconds:.3f}",
            ]
        )
    sections.append(
        table(["Policy", "Write ms/episode", "Exact read µs/query", "Joint readers s"], rows)
    )
    (output / "tables.md").write_text("\n".join(sections))
    # Deterministic first failures, for inspection rather than error-rate estimates.
    first_seed = str(summary["config"]["seeds"][0])
    examples = {}
    reader_rows = [
        json.loads(line) for line in (run / f"reader-{first_seed}.jsonl").read_text().splitlines()
    ]
    for row in reader_rows:
        if row["answers"]["selected_copy"] != row["target"]:
            examples.setdefault(f"selection_error/{row['stratum']}", row)
        if (
            row["answers"]["selected_copy"] == row["target"]
            and row["answers"]["selected_character"] != row["target"]
        ):
            examples.setdefault("generation_error_after_correct_copy", row)
    (output / "examples.json").write_text(
        json.dumps(
            {
                "selection_rule": f"first matching failure in saved seed {first_seed} row order",
                "examples": examples,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.run, args.output)
