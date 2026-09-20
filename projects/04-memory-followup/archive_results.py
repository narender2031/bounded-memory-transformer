"""Archive audited summaries and render a source-backed standalone figure."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--reader-audit", type=Path, required=True)
    parser.add_argument("--uniform-audit", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing archive")
    reader, uniform = read(args.reader / "summary.json"), read(args.uniform / "summary.json")
    reader_audit, uniform_audit = read(args.reader_audit), read(args.uniform_audit)
    if not reader_audit["passed"] or uniform_audit["status"] != "passed":
        raise ValueError("both independent audits must pass before archiving")
    args.output.mkdir(parents=True)
    copied = {
        "reader-summary.json": args.reader / "summary.json",
        "uniform-summary.json": args.uniform / "summary.json",
        "reader-audit.json": args.reader_audit,
        "uniform-audit.json": args.uniform_audit,
        "uniform-manifest.json": args.uniform / "manifest.json",
        "reader-manifest.json": args.reader / "frozen-manifest.json",
    }
    for destination, source in copied.items():
        shutil.copyfile(source, args.output / destination)
    development = {}
    for candidate in (1, 2, 3, 4):
        directory = args.development / f"candidate-{candidate}"
        development[str(candidate)] = {
            p.stem: read(p) for p in sorted(directory.glob("training-*.json"))
        }
    for path in args.development.glob("*.json"):
        development[path.stem] = read(path)
    (args.output / "development.json").write_text(json.dumps(development, indent=2) + "\n")
    snapshot = args.development / "candidate-1/tasks-at-training.py"
    if snapshot.exists():
        shutil.copyfile(snapshot, args.output / "candidate-1-generator.py.txt")

    strata = ("select", "unsupported", "contradicted", "irrelevant", "deleted")
    counts = {s: np.zeros(3) for s in strata}
    for seed in reader["config"]["seeds"]:
        with gzip.open(args.reader / f"seed-{seed}/original-k4.jsonl.gz", "rt") as handle:
            for line in handle:
                row = json.loads(line)
                for category in strata:
                    if category in row["categories"]:
                        counts[category] += [
                            row["v1_answer"] == row["visible"],
                            row["answer"] == row["visible"],
                            1,
                        ]
    figure_data = dict(
        reader={
            key: dict(v1=value[0] / value[2], final=value[1] / value[2], evaluations=int(value[2]))
            for key, value in counts.items()
        },
        uniform=uniform["analysis"]["pooled"],
    )
    (args.output / "figure-data.json").write_text(json.dumps(figure_data, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    locations = np.arange(len(strata))
    old = [figure_data["reader"][s]["v1"] * 100 for s in strata]
    new = [figure_data["reader"][s]["final"] * 100 for s in strata]
    left.bar(locations - 0.18, old, 0.36, color="#99a2b1", label="Frozen v1 selector + copy")
    left.bar(locations + 0.18, new, 0.36, color="#2765c5", label="Neural classifier + pointer")
    left.axhline(95, color="#b36b00", linestyle="--", linewidth=1, label="95% absolute gate")
    left.set(
        xticks=locations,
        xticklabels=["Select", "Absent", "Conflict", "Irrelevant", "Deleted"],
        ylim=(0, 107),
        ylabel="Visible-evidence accuracy (%)",
        title="Reader: matched fresh four-slot evidence",
    )
    left.tick_params(axis="x", labelrotation=15)
    left.legend(loc="lower left", fontsize=8)
    for i, value in enumerate(old):
        left.text(i - 0.18, value + 1.1, f"{value:.1f}", ha="center", fontsize=8)
    names = [f"K{k}:learned-{p}" for k in (4, 8) for p in ("fifo", "recency", "random")]
    positions = np.arange(len(names))
    means = np.array([figure_data["uniform"][name]["mean"] * 100 for name in names])
    intervals = np.array([figure_data["uniform"][name]["simultaneous_ci"] for name in names]) * 100
    right.axvspan(-1, 1, color="#e7f2e8")
    right.axvline(0, color="#56616e", linewidth=1)
    right.errorbar(
        means,
        positions,
        xerr=np.array([means - intervals[:, 0], intervals[:, 1] - means]),
        fmt="o",
        color="#2765c5",
        capsize=3,
    )
    right.set(
        yticks=positions,
        yticklabels=[n.replace(":learned-", " vs ") for n in names],
        xlim=(-1.15, 1.15),
        xlabel="Learned advantage (percentage points)",
        title="Uniform control: 10,240 independent episodes\nGreen: predeclared ±1 point margin",
    )
    right.invert_yaxis()
    fig.suptitle("Reliable evidence reading; no uniform-query retention advantage", fontsize=14)
    fig.savefig(args.output / "reader-and-uniform.png", dpi=180)
    fig.savefig(args.output / "reader-and-uniform.svg")
    plt.close(fig)
    svg = args.output / "reader-and-uniform.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    provenance = {
        name: dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in copied.items()
    }
    (args.output / "archive-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
