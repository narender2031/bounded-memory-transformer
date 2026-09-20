"""Train/validation-only diagnosis of the frozen v1 selector."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from bounded_memory_transformer.memory_benchmark.operations import Kind
from bounded_memory_transformer.memory_experiments.reader_cases import (
    copy_selected,
    generate_reader_tasks,
    oracle_select,
)
from bounded_memory_transformer.memory_experiments.reader_training import ReaderBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = []
    for seed in (7, 19, 43):
        bundle = ReaderBundle.load(args.run / f"reader-{seed}.pt")
        for split in ("train", "validation"):
            for capacity in (4, 8):
                tasks = generate_reader_tasks(split, seed=20260921, count=2500, capacity=capacity)
                for mode in ("full", "memory_only"):
                    views = [t.view for t in tasks]
                    if mode == "memory_only":
                        views = [type(v)(v.memory, (), v.query, v.state_bytes, ()) for v in views]
                    indices = bundle.selector.select(views)
                    errors = Counter()
                    strata = {name: [0, 0] for name in {t.stratum for t in tasks}}
                    for task, view, selected in zip(tasks, views, indices, strict=True):
                        oracle = oracle_select(view)
                        correct = copy_selected(view, selected) == copy_selected(view, oracle)
                        strata[task.stratum][0] += int(correct)
                        strata[task.stratum][1] += 1
                        if correct:
                            continue
                        records = (*view.memory, *view.current)
                        if selected == -1:
                            errors["false_abstention"] += 1
                        elif records[selected].key != view.query.key:
                            same_entity = records[selected].entity == view.query.entity
                            same_attribute = records[selected].attribute == view.query.attribute
                            errors[
                                f"wrong_key_entity_{same_entity}_attribute_{same_attribute}"
                            ] += 1
                        elif records[selected].kind == Kind.NOISE:
                            errors["matching_noise"] += 1
                        else:
                            errors["precedence"] += 1
                    results.append(
                        dict(
                            seed=seed,
                            split=split,
                            capacity=capacity,
                            mode=mode,
                            accuracy=1 - sum(errors.values()) / len(tasks),
                            errors=dict(errors),
                            strata=strata,
                        )
                    )
                    print(json.dumps(results[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dict(protocol="validation-diagnosis-v1", rows=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
