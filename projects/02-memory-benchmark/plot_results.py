"""Export a research figure from recorded results, without rerunning experiments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot(summary_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text())
    config = summary["effective_config"]
    main_scenario = config["scenarios"]["main"]
    policies = ("no_memory", "fifo", "recency", "similarity")
    labels = ("No memory", "FIFO", "Recency", "Similarity¹")
    neural = summary["neural"]["main"]
    symbolic = summary["symbolic"]["main"]
    for metrics in neural.values():
        assert (
            abs(metrics["accuracy_delta"] - metrics["beneficial_rate"] + metrics["harmful_rate"])
            < 1e-10
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": "#252a31",
            "axes.labelcolor": "#252a31",
            "axes.edgecolor": "#b4b8bc",
            "svg.fonttype": "none",
            "svg.hashsalt": "bounded-memory-transformer-experiment-01",
        }
    )
    figure, (absolute, paired) = plt.subplots(1, 2, figsize=(12, 5.8), width_ratios=(1.2, 1))
    y = np.arange(len(policies))
    absolute.barh(
        y - 0.17,
        [100 * neural[p]["accuracy"] for p in policies],
        height=0.30,
        color="#35658d",
        label="Tiny Transformer",
    )
    absolute.barh(
        y + 0.17,
        [100 * symbolic[p]["accuracy"] for p in policies],
        height=0.30,
        color="#e6bd66",
        label="Exact reader (same evidence)",
    )
    for index, policy in enumerate(policies):
        for offset, metrics in ((-0.17, neural[policy]), (0.17, symbolic[policy])):
            value = 100 * metrics["accuracy"]
            absolute.text(value + 1, index + offset, f"{value:.1f}%", va="center", fontsize=10)
    absolute.set(yticks=y, yticklabels=labels, xlim=(0, 100), xlabel="Exact-match accuracy (%)")
    absolute.invert_yaxis()
    absolute.set_title(
        f"Accuracy at a {config['capacity']}-slot budget", loc="left", pad=16, fontsize=12
    )
    handles, legend_labels = absolute.get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.10, 0.228),
        frameon=False,
        fontsize=10,
        borderaxespad=0,
    )
    for index, policy in enumerate(policies):
        mean = 100 * neural[policy]["accuracy_delta"]
        low, high = [100 * value for value in neural[policy]["accuracy_delta_ci95"]]
        paired.errorbar(
            mean,
            index,
            xerr=[[mean - low], [high - mean]],
            color="#35658d",
            fmt="o",
            markersize=6,
            capsize=4,
            linewidth=1.6,
        )
        paired.annotate(
            f"{mean:+.1f} pp",
            (mean, index),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    paired.axvline(0, color="#777b80", linewidth=1, linestyle="--")
    lows = [100 * neural[p]["accuracy_delta_ci95"][0] for p in policies]
    highs = [100 * neural[p]["accuracy_delta_ci95"][1] for p in policies]
    paired.set(
        yticks=y,
        yticklabels=labels,
        ylim=(-0.6, 3.6),
        xlim=(min(-5, min(lows) - 4), max(5, max(highs) + 4)),
        xlabel="Accuracy change vs no memory (percentage points)",
    )
    paired.invert_yaxis()
    paired.set_title("Paired effect of adding memory", loc="left", pad=16, fontsize=12)
    for axis in (absolute, paired):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#e9e9e9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    validation = [t["validation_accuracy"] for t in summary["training"]]
    figure.suptitle(
        "Experiment 01 · When memory hurts", x=0.025, ha="left", fontsize=19, fontweight="bold"
    )
    figure.text(
        0.025,
        0.895,
        f"{config['test_episodes']:,} unseen-symbol episodes × {len(config['seeds'])} model seeds"
        f"  ·  {main_scenario['candidates']} candidate operations"
        f"  ·  {main_scenario['sessions']} hard-reset sessions  ·  "
        f"{summary['training'][0]['parameters']:,} parameters",
        fontsize=10,
    )
    limitation = (
        "Reader competence gate passed. "
        if summary["reader_competence_passed"]
        else "Pilot: reader competence gate FAILED. "
    )
    figure.text(
        0.025,
        0.115,
        limitation + f"Validation: {min(validation):.1%}–{max(validation):.1%} "
        "(required ≥95% for every seed).",
        fontsize=10,
        fontweight="bold",
    )
    figure.text(
        0.025,
        0.065,
        "95% intervals resample paired episodes, conditional on these trained seeds. "
        "¹ Lexical top-1 retrieval from a bounded FIFO bank.\n"
        "Exact-reader controls separate evidence retention from neural reading failures. "
        "Source: " + summary_path.name,
        fontsize=9,
        color="#565c65",
    )
    figure.subplots_adjust(left=0.10, right=0.985, top=0.79, bottom=0.32, wspace=0.40)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=170, facecolor="white")
    svg_path = output.with_suffix(".svg")
    figure.savefig(svg_path, facecolor="white", metadata={"Date": None})
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
