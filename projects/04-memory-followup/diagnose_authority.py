"""Validation-only challenge for every ordered SET/UPDATE/DELETE pair."""

import argparse
import json
from pathlib import Path

import torch

from bounded_memory_transformer.memory_experiments.reader_cases import generate_reader_tasks
from bounded_memory_transformer.memory_followup.reader import FactorizedReader
from bounded_memory_transformer.memory_followup.tasks import authority_tasks, varied_tasks
from bounded_memory_transformer.memory_followup.train_reader import task_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--all-reader-cases", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = {}
    for seed in (7, 19, 43):
        saved = torch.load(
            args.training / f"reader-{seed}.pt", weights_only=True, map_location="cpu"
        )
        model = FactorizedReader(**saved["settings"])
        model.load_state_dict(saved["state"])
        if args.binary:
            model.readout_mode = "binary"
        result[str(seed)] = {}
        for slots in (4, 8):
            tasks = authority_tasks("validation", seed=20260923 + slots, count=900, capacity=slots)
            result[str(seed)][str(slots)] = task_metrics(model, tasks)
            print(seed, slots, result[str(seed)][str(slots)]["accuracy"], flush=True)
            if args.all_reader_cases:
                tasks = generate_reader_tasks(
                    "validation", seed=20260922, count=1000, capacity=slots
                )
                result[str(seed)][f"original_k{slots}"] = task_metrics(model, tasks)
                for current in (0, 1, 6, 32):
                    tasks = varied_tasks(
                        "validation",
                        seed=20260922 + slots * 100 + current,
                        count=1000,
                        capacity=slots,
                        current_count=current,
                    )
                    result[str(seed)][f"varied_k{slots}_c{current}"] = task_metrics(model, tasks)
        print(
            seed,
            "all_gates",
            all(v["absolute_gate"] and v["relative_gate"] for v in result[str(seed)].values()),
            flush=True,
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
