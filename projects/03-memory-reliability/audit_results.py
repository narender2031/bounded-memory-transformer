"""Independently audit completed five-case artifacts, without experiment imports.

Usage: python audit_results.py --run RUN_DIRECTORY --output /tmp/five-case-audit.json
Only completed runs (summary.json present) are read. Hashes authenticate recorded
files; they do not prove that training inputs were causal or timing was correct.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ("character_full", "selected_character", "selected_copy", "oracle_copy")
ACTIONS = ("STORE", "UPDATE", "DELETE", "IGNORE")
UNKNOWN = "??"


def load(path):
    return json.loads(path.read_text())


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(items):
    return sum(items) / len(items) if items else None


def key(operation):
    return operation["entity"], operation["attribute"]


def format_value(value):
    return UNKNOWN if value is None or value == -1 else f"{value:02d}"


def read_operations(operations, query):
    answer, index = UNKNOWN, -1
    for i, operation in enumerate(operations):
        if key(operation) == key(query) and operation["kind"] in (1, 2, 3):
            answer = UNKNOWN if operation["kind"] == 3 else format_value(operation["value"])
            index = i
    return answer, index


def read_bank(bank, query):
    answer = UNKNOWN
    for entity, attribute, value, _ in bank:
        if (entity, attribute) == key(query):
            answer = format_value(value)
    return answer


def parse_prompt(prompt):
    """Parse the saved symbolic prompt independently of the project's renderer."""
    match = re.fullmatch(r"M(.*?)C(.*?)Q(\d{2})([a-d])=", prompt)
    if not match:
        raise ValueError(f"unrecognized prompt {prompt!r}")
    query = {"entity": int(match[3]), "attribute": ord(match[4]) - ord("a")}
    groups = []
    for body in (match[1], match[2]):
        operations = []
        for token in body.split(";"):
            if not token:
                continue
            operation = re.fullmatch(r"([SUDN])(\d{2})([a-d])(\d{2}|\?\?)", token)
            if not operation:
                raise ValueError(f"unrecognized operation {token!r}")
            operations.append(
                {
                    "kind": {"S": 1, "U": 2, "D": 3, "N": 4}[operation[1]],
                    "entity": int(operation[2]),
                    "attribute": ord(operation[3]) - ord("a"),
                    "value": None if operation[4] == UNKNOWN else int(operation[4]),
                }
            )
        groups.append(operations)
    return groups[0], groups[1], query


def paired(predictions, baseline, truths, ids, bootstrap_samples):
    correct = [p == t for p, t in zip(predictions, truths, strict=True)]
    base = [p == t for p, t in zip(baseline, truths, strict=True)]
    harm = sum(b and not p for p, b in zip(correct, base, strict=True))
    benefit = sum(p and not b for p, b in zip(correct, base, strict=True))
    clusters = defaultdict(list)
    for identity, p, b in zip(ids, correct, base, strict=True):
        clusters[identity].append(int(p) - int(b))
    totals = np.asarray([sum(v) for v in clusters.values()])
    sizes = np.asarray([len(v) for v in clusters.values()])
    # Draw whole episode identities, then retain all queries in each drawn cluster.
    draws = np.random.default_rng(8128).integers(
        len(clusters), size=(bootstrap_samples, len(clusters))
    )
    ratios = totals[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    n = len(predictions)
    return {
        "queries": n,
        "unique_episodes": len(clusters),
        "accuracy": mean(correct),
        "baseline_accuracy": mean(base),
        "delta": (benefit - harm) / n,
        "harmful_count": harm,
        "beneficial_count": benefit,
        "harmful_rate": harm / n,
        "beneficial_rate": benefit / n,
        "delta_ci95": np.quantile(ratios, [0.025, 0.975]).tolist(),
    }


def report_reader(rows, arm, categories, bootstrap_samples):
    predictions = [r["answers"][arm] for r in rows]
    truth = [r["world_target"] for r in rows]
    visible = [r["target"] for r in rows]
    correctness = [p == t for p, t in zip(predictions, truth, strict=True)]
    visible_correct = [p == t for p, t in zip(predictions, visible, strict=True)]
    oracle_correct = [v == t for v, t in zip(visible, truth, strict=True)]
    neural, oracle = mean(correctness), mean(oracle_correct)
    empty = mean([r["no_evidence"] == r["world_target"] for r in rows])
    gain = oracle - empty
    value = (neural - empty) / gain if gain > 0 else None
    slices = {
        name: {"count": len(indices), "accuracy": mean([visible_correct[i] for i in indices])}
        for name, indices in categories.items()
    }
    unknown = sum(t == UNKNOWN for t in truth)
    abstentions = sum(p == UNKNOWN for p in predictions)
    true_abstentions = sum(p == t == UNKNOWN for p, t in zip(predictions, truth, strict=True))
    return {
        "queries": len(rows),
        "visible_accuracy": mean(visible_correct),
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
        "by_category": slices,
        "absolute_gate_passed": mean(visible_correct) >= 0.95
        and all(s["accuracy"] is not None and s["accuracy"] >= 0.95 for s in slices.values()),
        "relative_gate_passed": value >= 0.95 if value is not None else None,
        "oracle_correct_neural_wrong": sum(
            o and not n for o, n in zip(oracle_correct, correctness, strict=True)
        ),
        "oracle_wrong_neural_correct": sum(
            n and not o for o, n in zip(oracle_correct, correctness, strict=True)
        ),
        "unknown_queries": unknown,
        "abstentions": abstentions,
        "abstention_precision": true_abstentions / abstentions if abstentions else None,
        "abstention_recall": true_abstentions / unknown if unknown else None,
        "abstention_f1": 2 * true_abstentions / (unknown + abstentions)
        if unknown + abstentions
        else None,
        "paired": paired(
            predictions,
            [r["paired_no_memory"][arm] for r in rows],
            truth,
            [r["episode_id"] for r in rows],
            bootstrap_samples,
        ),
    }


def diagnostics(rows):
    copied = [r["answers"]["selected_copy"] == r["target"] for r in rows]
    generated = [r["answers"]["selected_character"] == r["target"] for r in rows]
    selected = [r["selector_index"] == r["oracle_index"] for r in rows]
    return {
        "queries": len(rows),
        "selector_index_accuracy": mean(selected),
        "copy_accuracy": mean(copied),
        "generation_accuracy": mean(generated),
        "both_correct": sum(c and g for c, g in zip(copied, generated, strict=True)),
        "both_wrong": sum(not c and not g for c, g in zip(copied, generated, strict=True)),
        "copy_correct_generation_wrong": sum(
            c and not g for c, g in zip(copied, generated, strict=True)
        ),
        "copy_wrong_generation_correct": sum(
            not c and g for c, g in zip(copied, generated, strict=True)
        ),
        "wrong_index_correct_copy": sum(not s and c for s, c in zip(selected, copied, strict=True)),
        "mean_prompt_tokens": mean([len(r["prompt"]) for r in rows]),
    }


def exact_action(bank, operation):
    matches = [i for i, row in enumerate(bank) if tuple(row[:2]) == key(operation)]
    kind = operation["kind"]
    if kind in (4, 5):
        return "IGNORE", -1
    if kind == 3:
        return ("DELETE", matches[0]) if matches else ("IGNORE", -1)
    if matches:
        return "UPDATE", matches[0]
    available = [i for i, row in enumerate(bank) if row[0] == -1]
    return "STORE", available[0] if available else len(bank)


def execute(bank, operation, action, target, cue=0, allocation=None):
    result = [row[:] for row in bank]
    if action == "IGNORE":
        return result
    if action == "STORE" and target == len(bank):
        if allocation is None:
            return result
        target = allocation
    if 0 <= target < len(bank):
        result[target] = (
            [-1] * 4
            if action == "DELETE"
            else [*key(operation), -1 if operation["value"] is None else operation["value"], cue]
        )
    return result


class Audit:
    def __init__(self, run, source_root):
        self.run, self.source_root = run, source_root
        self.errors = []
        self.checks = 0
        self.recomputed = {}
        self.counts = Counter()

    def check(self, label, expected, actual):
        self.checks += 1
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                self.errors.append(f"{label}: expected mapping; got {actual!r}")
                return
            for name, value in expected.items():
                if name not in actual:
                    self.errors.append(f"{label}/{name}: missing")
                else:
                    self.check(f"{label}/{name}", value, actual[name])
            return
        if isinstance(expected, (tuple, list)):
            if not isinstance(actual, (tuple, list)) or len(expected) != len(actual):
                self.errors.append(f"{label}: expected list length {len(expected)}; got {actual!r}")
                return
            for i, (left, right) in enumerate(zip(expected, actual, strict=True)):
                self.check(f"{label}/{i}", left, right)
            return
        same = expected == actual
        if isinstance(expected, float) and isinstance(actual, (float, int)):
            same = math.isfinite(actual) and math.isclose(
                expected, actual, abs_tol=1e-10, rel_tol=1e-10
            )
        if not same:
            self.errors.append(f"{label}: recomputed {expected!r}; recorded {actual!r}")

    def reader(self, stem, reports, diagnostic, categories, bootstrap):
        rows = load_rows(self.run / f"{stem}.jsonl")
        self.counts["reader_query_rows"] += len(rows)
        self.counts["reader_conditions"] += 1
        for i, row in enumerate(rows):
            memory, current, query = parse_prompt(row["prompt"])
            visible, oracle_index = read_operations(memory + current, query)
            empty_memory, empty_current, empty_query = parse_prompt(row["empty_prompt"])
            self.check(f"{stem}/{i}/empty-memory", [], empty_memory)
            self.check(f"{stem}/{i}/empty-current", current, empty_current)
            self.check(f"{stem}/{i}/empty-query", query, empty_query)
            self.check(f"{stem}/{i}/query", key(query), key(row))
            self.check(f"{stem}/{i}/visible", visible, row["target"])
            self.check(f"{stem}/{i}/oracle-copy", visible, row["answers"]["oracle_copy"])
            self.check(f"{stem}/{i}/oracle-index", oracle_index, row["oracle_index"])
            self.check(
                f"{stem}/{i}/empty-exact", read_operations(current, query)[0], row["no_evidence"]
            )
            selected = row["selector_index"]
            if selected < -1 or selected >= len(memory + current):
                self.errors.append(f"{stem}/{i}/selector-index: out of visible bounds {selected}")
            else:
                copied = (
                    UNKNOWN
                    if selected == -1
                    else format_value((memory + current)[selected]["value"])
                )
                self.check(f"{stem}/{i}/selected-copy", copied, row["answers"]["selected_copy"])
        if categories is None:
            categories = {
                name: [i for i, r in enumerate(rows) if (r["target"] != UNKNOWN) == known]
                for name, known in (("visible_known", True), ("visible_unknown", False))
            }
            categories = {name: indices for name, indices in categories.items() if indices}
        fresh = {arm: report_reader(rows, arm, categories, bootstrap) for arm in ARMS}
        self.check(stem + "/reports", fresh, reports)
        self.check(stem + "/diagnostics", diagnostics(rows), diagnostic)
        self.recomputed[stem] = fresh
        return rows, fresh

    def lifecycle(self, seed, name, data, reported):
        stem = f"lifecycle-{seed}-{name}"
        saved = load(self.run / f"{stem}.json")
        self.counts["lifecycle_operation_rows"] += len(saved["operations"])
        self.counts["lifecycle_query_rows"] += len(saved["queries"])
        indexed = {episode["episode_id"]: episode for episode in data}
        action_correct, targets, transitions = [], [], []
        gold_actions, predicted_actions = [], []
        previous = {}
        for i, row in enumerate(saved["operations"]):
            episode = indexed[row["episode_id"]]
            operation = episode["sessions"][row["session"]][row["operation_index"]]
            bank = row["bank_before"]
            self.check(
                f"{stem}/operation/{i}/chain",
                previous.get(row["episode_id"], [[-1] * 4 for _ in bank]),
                bank,
            )
            gold_action, gold_target = exact_action(bank, operation)
            gold = execute(bank, operation, gold_action, gold_target)
            result = execute(bank, operation, row["action"], row["target"])
            previous[row["episode_id"]] = result
            flags = {
                "gold_action": gold_action,
                "gold_target": gold_target,
                "action_correct": row["action"] == gold_action,
                "target_applicable": gold_action != "IGNORE",
                "target_correct": row["target"] == gold_target if gold_action != "IGNORE" else None,
                "transition_correct": result == gold,
                "bank_after": result,
            }
            self.check(f"{stem}/operation/{i}", flags, row)
            action_correct.append(flags["action_correct"])
            transitions.append(flags["transition_correct"])
            if flags["target_applicable"]:
                targets.append(flags["target_correct"])
            gold_actions.append(gold_action)
            predicted_actions.append(row["action"])
        query_rows = []
        for i, row in enumerate(saved["queries"]):
            episode = indexed[row["episode_id"]]
            stream = [operation for session in episode["sessions"] for operation in session]
            query = episode["sessions"][row["session"]][row["operation_index"]]
            expected = read_operations(stream, query)[0]
            answer = read_bank(row["bank"], query)
            control = key(query) == tuple(episode["control_key"])
            flags = {
                "expected": expected,
                "answer": answer,
                "correct": answer == expected,
                "control": control,
                "stale": not control and answer == episode["stale_value"],
                "deletion_nonabstention": not control
                and episode["case"] == "delete"
                and answer != UNKNOWN,
                "deleted_value_repetition": not control and answer == episode["deleted_value"],
            }
            self.check(f"{stem}/query/{i}", flags, row)
            self.check(f"{stem}/query/{i}/bank", previous[row["episode_id"]], row["bank"])
            query_rows.append({**row, **flags})
        controls = [r for r in query_rows if r["control"]]
        updates = [r for r in query_rows if not r["control"] and r["case"] == "update"]
        deletions = [r for r in query_rows if not r["control"] and r["case"] == "delete"]
        f1 = []
        for action in ACTIONS:
            tp = sum(g == p == action for g, p in zip(gold_actions, predicted_actions, strict=True))
            count = gold_actions.count(action) + predicted_actions.count(action)
            f1.append(2 * tp / count if count else 0)
        fresh = {
            "semantic_accuracy": mean([r["correct"] for r in query_rows]),
            "action_accuracy": mean(action_correct),
            "action_macro_f1": mean(f1),
            "target_accuracy": mean(targets),
            "transition_accuracy": mean(transitions),
            "control_preservation": mean([r["correct"] for r in controls]),
            "update_accuracy": mean([r["correct"] for r in updates]),
            "delete_accuracy": mean([r["correct"] for r in deletions]),
            "stale_answer_rate": mean([r["stale"] for r in updates]),
            "deletion_nonabstention_rate": mean([r["deletion_nonabstention"] for r in deletions]),
            "deleted_value_repetition_rate": mean(
                [r["deleted_value_repetition"] for r in deletions]
            ),
            "operation_count": len(action_correct),
            "applicable_target_count": len(targets),
            "query_count": len(query_rows),
            "control_count": len(controls),
            "update_query_count": len(updates),
            "delete_query_count": len(deletions),
            "state_bytes": 64,
            "historical_tokens_replayed": 0,
        }
        fresh["competence_gate"] = fresh["semantic_accuracy"] >= 0.95
        self.check(stem + "/summary", fresh, saved["summary"])
        self.check(stem + "/reported", fresh, reported)
        self.recomputed[stem] = fresh
        return query_rows

    def capacity(self, stem, data, slots, reported):
        saved = load(self.run / f"{stem}-banks.json")
        indexed = {episode["episode_id"]: episode for episode in data}
        self.check(stem + "/episode-count", len(data), len(saved["episodes"]))
        hits = oracle_hits = queries = 0
        answers, truths, identities = [], [], []
        for item in saved["episodes"]:
            episode = indexed[item["episode_id"]]
            self.counts["capacity_episode_banks"] += 1
            self.counts["capacity_exact_query_rows"] += len(item["records"])
            expected_slots = 0 if stem.endswith("no_memory") else slots
            self.check(
                f"{stem}/{item['episode_id']}/bank-slots", expected_slots, len(item["final_bank"])
            )
            for i, row in enumerate(item["final_bank"]):
                self.check(f"{stem}/{item['episode_id']}/bank-width/{i}", 4, len(row))
            writes = [operation for session in episode["sessions"] for operation, _ in session]
            requests = Counter(key(q) for q in episode["queries"])
            counts = sorted(requests.values(), reverse=True)
            oracle = sum(counts[:slots])
            matched = 0
            self.check(
                f"{stem}/{item['episode_id']}/record-count",
                len(episode["queries"]),
                len(item["records"]),
            )
            for i, (query, record) in enumerate(
                zip(episode["queries"], item["records"], strict=True)
            ):
                truth = read_operations(writes, query)[0]
                answer = read_bank(item["final_bank"], query)
                self.check(
                    f"{stem}/{item['episode_id']}/query/{i}",
                    {
                        "entity": query["entity"],
                        "attribute": query["attribute"],
                        "query_index": i,
                        "answer": answer,
                        "truth": truth,
                        "correct": answer == truth,
                        "historical_answerable": truth != UNKNOWN,
                    },
                    record,
                )
                matched += answer == truth
                answers.append(answer)
                truths.append(truth)
                identities.append(item["episode_id"])
            n = len(episode["queries"])
            self.check(
                f"{stem}/{item['episode_id']}/scores",
                {
                    "hits": matched,
                    "queries": n,
                    "utility": matched / n,
                    "oracle_hits": oracle,
                    "oracle_utility": oracle / n,
                    "oracle_regret": (oracle - matched) / n,
                },
                item,
            )
            hits += matched
            oracle_hits += oracle
            queries += n
        fresh = {
            "queries": queries,
            "utility": hits / queries,
            "oracle_utility": oracle_hits / queries,
            "oracle_regret": (oracle_hits - hits) / queries,
        }
        byte_count = 0 if stem.endswith("no_memory") else slots * 16
        fresh["state_bytes" if stem.startswith("combined") else "persistent_state_bytes"] = (
            byte_count
        )
        self.check(stem + "/summary", fresh, saved["summary"])
        self.check(stem + "/reported", fresh, reported)
        self.recomputed[stem + "/exact"] = fresh
        if stem.startswith("combined"):
            last = {}
            for i, row in enumerate(load_rows(self.run / f"{stem}-writes.jsonl")):
                self.counts["combined_operation_rows"] += 1
                identity = row["episode_id"]
                operation, cue = indexed[identity]["sessions"][row["session"]][
                    row["operation_index"]
                ]
                self.check(f"{stem}/write/{i}/input-operation", operation, row["operation"])
                self.check(f"{stem}/write/{i}/input-cue", cue, row["cue"])
                before = last.get(identity, [[-1] * 4 for _ in range(slots)])
                self.check(f"{stem}/write/{i}/before", before, row["bank_before"])
                last[identity] = execute(
                    before,
                    row["operation"],
                    row["action"],
                    row["target"],
                    row["cue"],
                    row["allocation"],
                )
                self.check(f"{stem}/write/{i}/after", last[identity], row["bank_after"])
            for item in saved["episodes"]:
                self.check(
                    f"{stem}/{item['episode_id']}/final-bank",
                    last[item["episode_id"]],
                    item["final_bank"],
                )
        return answers, truths, identities

    def run_all(self):
        summary_path = self.run / "summary.json"
        if not summary_path.is_file():
            raise ValueError("summary.json absent: refusing to inspect an incomplete run")
        summary = load(summary_path)
        config = load(self.run / "config.json")
        self.check("config", config, summary["config"])
        for name, expected in summary["artifact_sha256"].items():
            self.counts["artifact_hashes"] += 1
            path = self.run / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
            self.check(f"artifact-sha256/{name}", expected, actual)
        manifest = load(self.run / "source-manifest.json")
        for name, expected in manifest["source_sha256"].items():
            self.counts["source_hashes"] += 1
            path = self.source_root / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
            self.check(f"source-sha256/{name}", expected, actual)
        reader_data = load(self.run / "reader-data.json")
        lifecycle_data = load(self.run / "lifecycle-data.json")
        capacity_data = {name: load(self.run / f"capacity-data-{name}.json") for name in ("A", "B")}
        self.check(
            "capacity/paired-write-prefixes",
            [r["sessions"] for r in capacity_data["A"]],
            [r["sessions"] for r in capacity_data["B"]],
        )
        categories = {
            name: [i for i, r in enumerate(reader_data) if r["stratum"] == name]
            for name in ("select", "unsupported", "contradicted", "irrelevant", "deleted")
        }
        categories.update(
            {
                "known": [i for i, r in enumerate(reader_data) if r["known"]],
                "unknown": [i for i, r in enumerate(reader_data) if not r["known"]],
                "full_occupancy": [i for i, r in enumerate(reader_data) if r["occupancy"] == 4],
            }
        )
        bootstrap = config["evaluation"]["bootstrap_samples"]
        for seed in config["seeds"]:
            seed_result = summary["seeds"][str(seed)]
            self.check(
                f"evaluation-{seed}", load(self.run / f"evaluation-{seed}.json"), seed_result
            )
            self.check(
                f"training-{seed}",
                load(self.run / f"training-{seed}.json"),
                summary["training"][str(seed)],
            )
            reader_rows, reader = self.reader(
                f"reader-{seed}",
                seed_result["reader"],
                seed_result["reader_diagnostics"],
                categories,
                bootstrap,
            )
            self.check(f"reader-{seed}/dataset-count", len(reader_data), len(reader_rows))
            for i, (task, row) in enumerate(zip(reader_data, reader_rows, strict=True)):
                view = task["view"]
                target, index = read_operations(view["memory"] + view["current"], view["query"])
                self.check(f"reader-data/{i}/target", target, task["target"])
                self.check(f"reader-data/{i}/oracle-index", index, task["oracle_index"])
                self.check(
                    f"reader-{seed}/{i}/dataset",
                    {
                        "episode_id": task["episode_id"],
                        "stratum": task["stratum"],
                        "world_target": target,
                    },
                    row,
                )
            for name in ("exact", "learned"):
                lifecycle_rows = self.lifecycle(
                    seed, name, lifecycle_data, seed_result["lifecycle"][name]
                )
            life_categories = {
                "update": [i for i, r in enumerate(lifecycle_rows) if r["case"] == "update"],
                "delete": [i for i, r in enumerate(lifecycle_rows) if r["case"] == "delete"],
                "control": [i for i, r in enumerate(lifecycle_rows) if r["control"]],
            }
            e2e = seed_result["end_to_end"]["lifecycle"]
            _, life_reader = self.reader(
                f"end-to-end-lifecycle-{seed}",
                e2e["reader"],
                e2e["diagnostics"],
                life_categories,
                bootstrap,
            )
            recoveries = {"lifecycle": life_reader["selected_copy"]["reader_recovery"]["value"]}
            for workload, data in capacity_data.items():
                for slots in config["evaluation"]["capacity_slots"]:
                    condition = seed_result["capacity"][workload][str(slots)]
                    exact = {}
                    for policy, reported in condition.items():
                        if policy == "paired_comparisons":
                            continue
                        stem = f"capacity-{seed}-{workload}-{slots}-{policy}"
                        exact[policy], truths, identities = self.capacity(
                            stem, data, slots, reported["exact"]
                        )
                        rows, neural = self.reader(
                            stem, reported["neural"], reported["diagnostics"], None, bootstrap
                        )
                        self.check(
                            stem + "/neural-world-truth", truths, [r["world_target"] for r in rows]
                        )
                        self.check(
                            stem + "/neural-visible-evidence",
                            exact[policy],
                            [r["target"] for r in rows],
                        )
                        if policy != "no_memory":
                            recoveries[f"capacity/{workload}/{slots}/{policy}"] = neural[
                                "selected_copy"
                            ]["reader_recovery"]["value"]
                    for baseline, reported in condition["paired_comparisons"].items():
                        self.check(
                            f"capacity-{seed}-{workload}-{slots}/paired/{baseline}",
                            paired(
                                exact["learned"], exact[baseline], truths, identities, bootstrap
                            ),
                            reported,
                        )
                    combined = seed_result["end_to_end"]["combined_capacity"][workload][str(slots)]
                    stem = f"combined-{seed}-{workload}-{slots}"
                    answers, truths, _ = self.capacity(stem, data, slots, combined["exact"])
                    rows, neural = self.reader(
                        stem, combined["neural"], combined["diagnostics"], None, bootstrap
                    )
                    self.check(
                        stem + "/neural-world-truth", truths, [r["world_target"] for r in rows]
                    )
                    self.check(
                        stem + "/neural-visible-evidence", answers, [r["target"] for r in rows]
                    )
                    recoveries[f"combined/{workload}/{slots}"] = neural["selected_copy"][
                        "reader_recovery"
                    ]["value"]
            episode_gate = {
                "passed": all(v is not None and v >= 0.95 for v in recoveries.values()),
                "required_recovery": recoveries,
                "unavailable": sorted(k for k, v in recoveries.items() if v is None),
                "failed": sorted(k for k, v in recoveries.items() if v is not None and v < 0.95),
            }
            recorded_gate = {
                **seed_result["episode_reader_gate"],
                "unavailable": sorted(seed_result["episode_reader_gate"]["unavailable"]),
                "failed": sorted(seed_result["episode_reader_gate"]["failed"]),
            }
            self.check(f"seed/{seed}/episode-gate", episode_gate, recorded_gate)
            learned = self.recomputed[f"lifecycle-{seed}-learned"]
            gates = {
                "reader_absolute": reader["selected_copy"]["absolute_gate_passed"],
                "reader_relative": reader["selected_copy"]["relative_gate_passed"],
                "reader_episode_relative": episode_gate["passed"],
                "lifecycle": all(
                    learned[k] is not None and learned[k] >= 0.95
                    for k in (
                        "semantic_accuracy",
                        "update_accuracy",
                        "delete_accuracy",
                        "control_preservation",
                    )
                ),
            }
            self.check(f"seed/{seed}/gates", gates, seed_result["gates"])
        return {
            "passed": not self.errors,
            "checks": self.checks,
            "failures": self.errors,
            "counts": dict(self.counts),
            "run": str(self.run.resolve()),
            "source_revision": manifest["git_head"],
            "recomputed": self.recomputed,
            "limitations": [
                "Source hashes compare the current source-root contents.",
                "Timing and causal training claims require source/run review.",
                "No checkpoint inference or held-out model tuning is performed.",
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", nargs="?", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run = args.run or args.run_directory
    if run is None:
        parser.error("--run RUN_DIRECTORY is required")
    try:
        result = Audit(run, args.source_root).run_all()
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        result = {"passed": False, "failures": [f"audit could not finish: {error}"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "recomputed"}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
