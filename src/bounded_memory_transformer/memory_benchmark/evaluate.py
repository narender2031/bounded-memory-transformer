"""Run Experiment 01 locally; freeze configuration before inspecting held-out results."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from bounded_memory_transformer.cli.train_tiny import resolve_device

from .generator import EpisodeConfig, generate_episodes
from .metrics import Result, summarize
from .operations import UNKNOWN, Episode
from .policies import POLICIES
from .reader import predict, render, render_operations, synchronize, train_reader
from .views import QueryView, read_visible, stream_views

DEFAULTS = {
    "seeds": [7, 19, 43],
    "device": "auto",
    "threads": 4,
    "steps": 1200,
    "batch_size": 64,
    "train_examples": 16384,
    "validation_examples": 512,
    "test_episodes": 800,
    "d_model": 96,
    "n_layers": 2,
    "n_heads": 4,
    "context_length": 128,
    "capacity": 4,
    "learning_rate": 0.002,
    "log_every": 200,
    "bootstrap_samples": 2000,
    "test_seed": 424242,
    "scenarios": {
        "main": {"candidates": 30, "sessions": 8},
        "longer": {"candidates": 60, "sessions": 16},
        "more_updates": {"candidates": 30, "sessions": 8, "updates": 4},
    },
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_manifest() -> dict:
    sources = sorted(Path(__file__).resolve().parent.glob("*.py"))
    sources += sorted((Path(__file__).resolve().parents[1] / "tiny_transformer").glob("*.py"))
    hashes = {
        str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sources
    }
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "git_head": head.stdout.strip(),
        "git_status": status.stdout.splitlines(),
        "source_sha256": hashes,
    }


def useful_retained(episode: Episode, view: QueryView) -> bool:
    return (
        episode.truth.answer != UNKNOWN
        and read_visible(view.retained, (), view.query) == episode.truth.answer
    )


def result_rows(
    episodes: list[Episode],
    views: list[QueryView],
    answers: list[str],
    baseline: list[str],
    seed: int,
) -> list[Result]:
    return [
        Result(e.episode_id, seed, e.case, e.truth, answer, no_memory, useful_retained(e, v))
        for e, v, answer, no_memory in zip(episodes, views, answers, baseline, strict=True)
    ]


def validate_config(config: dict) -> None:
    if set(config) - set(DEFAULTS):
        raise ValueError(f"unknown configuration fields: {set(config) - set(DEFAULTS)}")
    for name in (
        "threads",
        "steps",
        "batch_size",
        "train_examples",
        "validation_examples",
        "test_episodes",
        "capacity",
        "log_every",
        "bootstrap_samples",
    ):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    if config["learning_rate"] <= 0 or not config["scenarios"]:
        raise ValueError("need a positive learning rate and at least one scenario")
    for scenario in config["scenarios"].values():
        EpisodeConfig(**scenario)


def markdown_report(summary: dict) -> str:
    lines = [
        "# Experiment 01 — When Memory Hurts",
        "",
        "Measured local results. Each neural row uses the same reader weights and episodes",
        "as its no-memory control. Intervals bootstrap paired episodes while keeping model",
        "seeds together; they are conditional on these trained seeds.",
        "",
        f"Device: {summary['environment']['device']}. "
        f"Parameters: {summary['training'][0]['parameters']:,}. "
        f"Slots: {summary['effective_config']['capacity']}.",
        "",
        "## Reader competence",
        "",
    ]
    for training in summary["training"]:
        lines.append(
            f"- Seed {training['seed']}: visible-evidence validation accuracy "
            f"{training['validation_accuracy']:.2%}; "
            f"training + validation {training['training_seconds_including_validation']:.1f}s."
        )
    if not summary["reader_competence_passed"]:
        lines += [
            "",
            "**Limitation:** at least one reader is below the predeclared 95% validation",
            "threshold. Neural errors cannot be attributed solely to memory management.",
        ]
    for scenario, policies in summary["neural"].items():
        lines += [
            "",
            f"## {scenario}",
            "",
            "| Policy | Accuracy | Δ vs no memory (pp, 95% CI) | "
            "Harmful | Beneficial | Stale¹ | Deleted leakage² |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, metrics in policies.items():
            low, high = metrics["accuracy_delta_ci95"]
            stale = metrics["stale_answer_rate"]
            deleted = metrics["deleted_fact_leakage"]
            lines.append(
                f"| {name} | {metrics['accuracy']:.2%} | "
                f"{100 * metrics['accuracy_delta']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}] | "
                f"{metrics['harmful_rate']:.2%} | {metrics['beneficial_rate']:.2%} | "
                f"{stale:.2%} | {deleted:.2%} |"
                if stale is not None and deleted is not None
                else f"| {name} | {metrics['accuracy']:.2%} | n/a | n/a | n/a | n/a | n/a |"
            )
        lines += [
            "",
            "Exact-key symbolic reader accuracy on the same visible evidence: "
            + ", ".join(
                f"{name} {m['accuracy']:.2%}" for name, m in summary["symbolic"][scenario].items()
            )
            + ".",
        ]
    lines += [
        "",
        "¹ Stale answer / updated, answerable queries. ² Non-abstention / deleted queries.",
        "",
        "Harmful = no memory correct and memory wrong; beneficial = the reverse. Both use",
        "all queries as denominator. Their difference equals the overall accuracy change.",
        "",
        "Detailed case, gap, overwrite, abstention, seed, cost and retention metrics are in",
        "`summary.json`; every neural prediction and its bounded prompt is in `predictions.jsonl`.",
        "Logical payload is 16 bytes/slot, excluding interpreter overhead. Raw-history replay",
        "is zero; reconstructed bounded-memory tokens and read/write time are reported separately.",
        "",
    ]
    return "\n".join(lines)


def run_experiment(raw_config: dict, output: Path) -> dict:
    config = DEFAULTS | raw_config
    validate_config(config)
    output.mkdir(parents=True, exist_ok=True)
    # Prevent accidental replacement of an earlier run, including a failed run.
    with (output / "config.json").open("x") as handle:
        json.dump(raw_config, handle, indent=2, sort_keys=True)
    manifest = source_manifest()
    save_json(output / "source-manifest.json", manifest)
    torch.set_num_threads(config["threads"])
    device = resolve_device(config["device"]).type
    if device == "cpu":
        torch.use_deterministic_algorithms(True)
    summary = {
        "config": raw_config,
        "effective_config": config,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": device,
            "threads": config["threads"],
        },
        "source_manifest": manifest,
        "contract": {
            "raw_history_tokens_reprocessed": 0,
            "bounded_payload_bytes": config["capacity"] * 16,
            "slot_fields": ["operation", "entity", "attribute", "value"],
            "slot_dtype": "int32",
            "kv_cache": "not implemented; stateless forwards",
            "similarity": "lexical key overlap, bounded FIFO bank, top-1, newest tie",
            "full_history_reference": "truth scorer only; outside strict budget",
            "confidence_interval": "paired episode bootstrap, conditional on model seeds",
        },
        "training": [],
        "neural": {},
        "symbolic": {},
        "costs": {},
        "dataset_sha256": {},
    }
    # Train all readers before opening the held-out episode set. No test selection.
    for seed in config["seeds"]:
        model, training = train_reader(config, seed, device)
        summary["training"].append(training)
        torch.save(
            {
                "model_config": asdict(model.config),
                "seed": seed,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            },
            output / f"seed-{seed}.pt",
        )
        save_json(output / f"training-{seed}.json", training)
        del model
    summary["reader_competence_passed"] = all(
        t["validation_accuracy"] >= 0.95 for t in summary["training"]
    )
    predictions_file = output / "predictions.jsonl"
    with predictions_file.open("x") as predictions_handle:
        for scenario_name, scenario in config["scenarios"].items():
            episodes = generate_episodes(
                EpisodeConfig(**scenario),
                split="test",
                seed=config["test_seed"],
                count=config["test_episodes"],
            )
            payload = "\n".join(json.dumps(asdict(e), sort_keys=True) for e in episodes) + "\n"
            (output / f"episodes-{scenario_name}.jsonl").write_text(payload)
            summary["dataset_sha256"][scenario_name] = hashlib.sha256(payload.encode()).hexdigest()
            views, prompts, references, costs = {}, {}, {}, {}
            for policy in POLICIES:
                started = time.perf_counter()
                views[policy] = [
                    next(stream_views(iter(e.sessions), policy, config["capacity"]))
                    for e in episodes
                ]
                elapsed = time.perf_counter() - started
                prompts[policy] = [render(v) for v in views[policy]]
                references[policy] = [
                    read_visible(v.memory, v.current, v.query) for v in views[policy]
                ]
                costs[policy] = {
                    "policy_write_and_retrieve_seconds": elapsed,
                    "policy_ms_per_episode": 1000 * elapsed / len(episodes),
                    "payload_bytes": views[policy][0].state_bytes,
                    "mean_memory_tokens_read": sum(
                        len(render_operations(v.memory)) for v in views[policy]
                    )
                    / len(episodes),
                    "mean_prompt_tokens": sum(map(len, prompts[policy])) / len(episodes),
                    "max_prompt_tokens": max(map(len, prompts[policy])),
                    "raw_history_tokens_reprocessed": 0,
                    "reader_ms_per_query_by_seed": {},
                }
            summary["symbolic"][scenario_name] = {
                policy: summarize(
                    result_rows(
                        episodes, views[policy], references[policy], references["no_memory"], -1
                    ),
                    bootstrap_samples=config["bootstrap_samples"],
                )
                for policy in POLICIES
            }
            all_rows: dict[str, list[Result]] = {p: [] for p in POLICIES}
            visible_agreement: dict[str, list[bool]] = {p: [] for p in POLICIES}
            for seed in config["seeds"]:
                from bounded_memory_transformer.tiny_transformer import (
                    TinyTransformerLM,
                    TransformerConfig,
                )

                checkpoint = torch.load(
                    output / f"seed-{seed}.pt", weights_only=True, map_location="cpu"
                )
                model = TinyTransformerLM(TransformerConfig(**checkpoint["model_config"]))
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device).eval()
                answers = {}
                for policy in POLICIES:
                    synchronize(device)
                    started = time.perf_counter()
                    answers[policy] = predict(
                        model, prompts[policy], batch_size=config["batch_size"], device=device
                    )
                    synchronize(device)
                    costs[policy]["reader_ms_per_query_by_seed"][str(seed)] = (
                        1000 * (time.perf_counter() - started) / len(episodes)
                    )
                    rows = result_rows(
                        episodes, views[policy], answers[policy], answers["no_memory"], seed
                    )
                    all_rows[policy].extend(rows)
                    for index, row in enumerate(rows):
                        reference = references[policy][index]
                        visible_agreement[policy].append(row.prediction == reference)
                        record = asdict(row) | {
                            "scenario": scenario_name,
                            "policy": policy,
                            "prompt": prompts[policy][index],
                            "visible_reference": reference,
                        }
                        predictions_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    accuracy = sum(r.prediction == r.truth.answer for r in rows) / len(rows)
                    print(
                        f"scenario={scenario_name} seed={seed} policy={policy} "
                        f"accuracy={accuracy:.3%}",
                        flush=True,
                    )
                del model
            summary["neural"][scenario_name] = {
                policy: summarize(all_rows[policy], bootstrap_samples=config["bootstrap_samples"])
                | {
                    "visible_reader_agreement": sum(visible_agreement[policy])
                    / len(visible_agreement[policy])
                }
                for policy in POLICIES
            }
            summary["costs"][scenario_name] = costs
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "summary.json", summary)
    (output / "report.md").write_text(markdown_report(summary))
    print(f"Results: {output / 'report.md'}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("projects/02-memory-benchmark/configs/baseline-small.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / (
        "when-memory-hurts-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_experiment(json.loads(args.config.read_text()), output)


if __name__ == "__main__":
    main()
