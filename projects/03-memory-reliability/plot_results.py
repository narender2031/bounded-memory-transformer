"""Render standalone figures from the saved five-case summary; no model inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

ARMS = ("character_full", "selected_character", "selected_copy", "oracle_copy")
ARM_LABELS = ("Character", "Select + generate", "Select + copy", "Oracle + copy")
COLORS = ("#9b6eaa", "#d78d45", "#147d92", "#303846")
POLICIES = ("no_memory", "fifo", "recency", "random", "similarity", "cue_priority", "learned")
POLICY_LABELS = ("None", "FIFO", "Recency", "Random", "Similarity", "Cue rule", "Learned")


def finish(fig, output: Path, stem: str):
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, 1.065)
    ax.grid(axis="y", alpha=0.16)
    ax.set_axisbelow(True)


def plot(summary: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(summary["seeds"].values())
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    names = ("select", "unsupported", "contradicted", "irrelevant", "deleted")
    x = np.arange(len(names))
    width = 0.18
    for index, (arm, label, color) in enumerate(zip(ARMS, ARM_LABELS, COLORS, strict=True)):
        scores = np.array(
            [
                [seed["reader"][arm]["by_category"][name]["accuracy"] for name in names]
                for seed in seeds
            ]
        )
        positions = x + (index - 1.5) * width
        axes[0].bar(positions, scores.mean(0), width, label=label, color=color)
        for score in scores:
            axes[0].scatter(positions, score, color="black", s=8, alpha=0.65, zorder=3)
    axes[0].axhline(0.95, ls="--", color="#ac3c46", lw=1, label="95% gate")
    axes[0].set_xticks(
        x, ["Select (Case 1)", "Unsupported", "Contradicted", "Irrelevant", "Deleted"]
    )
    axes[0].set_title("Reader: same evidence, four ways to answer (Cases 1–2)", loc="left")
    axes[0].set_ylabel("Visible-evidence accuracy")
    metrics = (
        "update_accuracy",
        "delete_accuracy",
        "control_preservation",
        "action_macro_f1",
        "target_accuracy",
        "transition_accuracy",
    )
    scores = np.array([[seed["lifecycle"]["learned"][m] for m in metrics] for seed in seeds])
    axes[1].bar(np.arange(len(metrics)), scores.mean(0), color="#147d92", width=0.6)
    for score in scores:
        axes[1].scatter(np.arange(len(metrics)), score, color="black", s=10, zorder=3)
    axes[1].set_xticks(
        np.arange(len(metrics)),
        ["Update", "Delete", "Preserve control", "Action F1", "Target", "Transition"],
    )
    axes[1].axhline(0.95, ls="--", color="#ac3c46", lw=1)
    axes[1].set_title(
        "Writer: lifecycle predictions scored with exact reading (Cases 3–4)", loc="left"
    )
    axes[1].set_ylabel("Accuracy / macro-F1")
    for ax in axes:
        style(ax)
    fig.suptitle("Separate reading failures from writing failures", fontsize=17, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=5,
        loc="outside lower center",
        fontsize=9,
        title=f"Bars: mean of {len(seeds)} seeds · dots: each seed · 4 slots / 64 bytes",
    )
    finish(fig, output, "reader-lifecycle")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for row, slots in enumerate(("4", "8")):
        for col, workload in enumerate(("A", "B")):
            ax = axes[row, col]
            conditions = [seed["capacity"][workload][slots] for seed in seeds]
            exact = np.array([[c[p]["exact"]["utility"] for p in POLICIES] for c in conditions])
            neural = np.array(
                [
                    [c[p]["neural"]["selected_copy"]["world_accuracy"] for p in POLICIES]
                    for c in conditions
                ]
            )
            oracle = np.mean([c["learned"]["exact"]["oracle_utility"] for c in conditions])
            x = np.arange(len(POLICIES))
            ax.bar(x - 0.18, exact.mean(0), 0.36, label="Exact reader", color="#147d92")
            ax.bar(x + 0.18, neural.mean(0), 0.36, label="Learned select + copy", color="#dda45c")
            for values in exact:
                ax.scatter(x - 0.18, values, s=8, color="black", zorder=3)
            ax.axhline(oracle, color="#303846", ls="--", label="Clairvoyant retention bound")
            label = "A · unpredictable queries" if workload == "A" else "B · cue predicts utility"
            ax.set_title(f"{label} · {slots} slots / {int(slots) * 16} bytes", loc="left")
            ax.set_xticks(x, POLICY_LABELS, rotation=25, ha="right")
            ax.set_ylabel("Useful recall on historical probes")
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    evaluation = summary["config"]["evaluation"]
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="outside lower center",
        title=(
            f"Case 5 · {evaluation['capacity_episodes']} episodes × "
            f"{evaluation['capacity_queries']} probes · {len(seeds)} seeds · hard resets"
        ),
    )
    fig.suptitle(
        "Can learned retention find future value within a hard budget?", fontsize=16, weight="bold"
    )
    finish(fig, output, "capacity-utility")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, workload in zip(axes, ("A", "B"), strict=True):
        for policy, color in zip(
            ("fifo", "recency", "cue_priority", "learned"), COLORS, strict=True
        ):
            scores = np.array(
                [
                    [
                        seed["capacity"][workload][k][policy]["exact"]["oracle_regret"]
                        for k in ("4", "8")
                    ]
                    for seed in seeds
                ]
            )
            ax.plot(
                [4, 8],
                scores.mean(0),
                marker="o",
                label=policy,
                color=color,
                ls="--" if policy == "learned" else "-",
            )
        ax.set_title(f"Workload {workload}", loc="left")
        ax.set_xticks([4, 8])
        ax.set_xlabel("Memory slots (same model weights)")
        style(ax)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("Oracle utility − policy utility")
    axes[1].legend()
    fig.suptitle("Retention regret: remaining gap to future-aware storage", fontsize=15)
    finish(fig, output, "retention-regret")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.summary.read_text()), args.output)
