"""Literal expectations and tamper tests for the standalone reader auditor."""

import gzip
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch


def auditor():
    path = Path(__file__).resolve().parents[1] / "projects/04-memory-followup/audit_reader.py"
    spec = importlib.util.spec_from_file_location("followup_reader_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operation(kind, entity, value=None, attribute=0):
    return {"kind": kind, "entity": entity, "attribute": attribute, "value": value}


def literal_rows():
    query = operation(5, 1)
    return [
        {
            "memory": [operation(1, 1, 10), operation(2, 1, 20), operation(4, 1, 99)],
            "current": [operation(1, 9, 20)],
            "query": query,
            "world": "20",
            "state_bytes": 64,
            "episode_id": 0,
            "visible": "20",
            "current_only": "??",
            "answer": "20",
            "no_memory_answer": "20",
            "selected": 3,
            "no_memory_selected": 0,
            "oracle": 1,
            "categories": ["visible_known"],
            "v1_selected": 1,
            "v1_answer": "20",
        },
        {
            "memory": [operation(1, 1, 10)],
            "current": [operation(3, 1)],
            "query": query,
            "world": "??",
            "state_bytes": 64,
            "episode_id": 0,
            "visible": "??",
            "current_only": "??",
            "answer": "10",
            "no_memory_answer": "??",
            "selected": 0,
            "no_memory_selected": 0,
            "oracle": 1,
            "categories": ["visible_unknown"],
            "v1_selected": 1,
            "v1_answer": "??",
        },
        {
            "memory": [operation(1, 2, 30)],
            "current": [],
            "query": query,
            "world": "30",
            "state_bytes": 64,
            "episode_id": 1,
            "visible": "??",
            "current_only": "??",
            "answer": "30",
            "no_memory_answer": "??",
            "selected": 0,
            "no_memory_selected": -1,
            "oracle": -1,
            "categories": ["visible_unknown"],
            "v1_selected": -1,
            "v1_answer": "??",
        },
        {
            "memory": [operation(1, 1, 40)],
            "current": [operation(2, 1, 50), operation(4, 1, 60)],
            "query": query,
            "world": "50",
            "state_bytes": 64,
            "episode_id": 2,
            "visible": "50",
            "current_only": "50",
            "answer": "60",
            "no_memory_answer": "50",
            "selected": 2,
            "no_memory_selected": 0,
            "oracle": 1,
            "categories": ["visible_known"],
            "v1_selected": 1,
            "v1_answer": "50",
        },
    ]


def test_independent_report_preserves_wrong_key_copy_and_world_denominators():
    module = auditor()
    result = module.recompute_rows(literal_rows(), bootstrap_samples=2000)
    assert result["queries"] == 4
    assert result["visible_accuracy"] == 0.25
    assert result["world_accuracy"] == 0.5
    assert result["oracle_accuracy"] == 0.75
    assert result["reader_recovery"] == {
        "value": 0,
        "reason": None,
        "neural_accuracy": 0.5,
        "oracle_accuracy": 0.75,
        "no_evidence_accuracy": 0.5,
        "oracle_gain": 0.25,
    }
    assert result["paired"] == {
        "queries": 4,
        "unique_episodes": 3,
        "accuracy": 0.5,
        "baseline_accuracy": 0.75,
        "delta": -0.25,
        "harmful_count": 2,
        "beneficial_count": 1,
        "harmful_rate": 0.5,
        "beneficial_rate": 0.25,
        "delta_ci95": [-1, 1],
    }
    assert result["by_category"] == {
        "visible_known": {"count": 2, "accuracy": 0.5},
        "visible_unknown": {"count": 2, "accuracy": 0},
    }
    assert result["unknown_queries"] == 1
    assert result["abstentions"] == 0
    assert result["abstention_precision"] is None
    assert result["abstention_recall"] == result["abstention_f1"] == 0
    assert result["oracle_correct_neural_wrong"] == 2
    assert result["oracle_wrong_neural_correct"] == 1
    assert result["selection_index_accuracy"] == 0
    assert result["frozen_v1_copy_accuracy"] == 1
    assert not result["passed"]


def test_nonpositive_recovery_headroom_remains_unavailable():
    module = auditor()
    row = literal_rows()[2]
    row.update(world="??", selected=-1, answer="??")
    result = module.recompute_rows([row], bootstrap_samples=20)
    assert result["visible_accuracy"] == 1
    assert result["absolute_gate_passed"] is True
    assert result["reader_recovery"]["value"] is None
    assert result["reader_recovery"]["reason"] == "no_positive_oracle_gain"
    assert result["relative_gate_passed"] is None
    assert result["passed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("visible", "99"),
        ("current_only", "20"),
        ("answer", "??"),
        ("no_memory_answer", "??"),
        ("oracle", 2),
        ("selected", 99),
        ("v1_answer", "99"),
    ],
)
def test_changed_saved_semantics_are_rejected(field, value):
    module = auditor()
    rows = literal_rows()
    rows[0][field] = value
    with pytest.raises(module.AuditError):
        module.recompute_rows(rows, bootstrap_samples=20)


def test_report_tampering_and_row_hash_tampering_are_both_rejected(tmp_path):
    module = auditor()
    path = tmp_path / "rows.jsonl.gz"
    rows = literal_rows()
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = module.recompute_rows(rows, bootstrap_samples=20)
    report.update(rows_sha256=module.sha256(path), inference_seconds_with_controls=0.01)
    assert module.audit_condition(path, report, bootstrap_samples=20)["visible_accuracy"] == 0.25
    bad = deepcopy(report)
    bad["visible_accuracy"] = 1
    with pytest.raises(module.AuditError, match="visible_accuracy"):
        module.audit_condition(path, bad, bootstrap_samples=20)
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(module.AuditError, match="hash"):
        module.audit_condition(path, report, bootstrap_samples=20)


def original_row():
    return {
        "memory": [
            operation(1, 1, 10),
            operation(1, 2, 20),
            operation(1, 3, 30),
            operation(1, 4, 40),
        ],
        "current": [operation(1, 2, 20)] * 6,
        "query": operation(5, 1),
        "world": "10",
        "state_bytes": 64,
        "episode_id": 0,
        "visible": "10",
        "current_only": "??",
        "answer": "10",
        "no_memory_answer": "??",
        "selected": 0,
        "no_memory_selected": -1,
        "oracle": 0,
        "categories": ["select", "occupancy_4", "source_memory", "visible_known"],
        "v1_selected": 0,
        "v1_answer": "10",
    }


def test_microtask_sources_occupancy_and_absent_categories_are_derived_from_evidence():
    module = auditor()
    row = original_row()
    result = module.recompute_rows([row], condition="original-k4", bootstrap_samples=20)
    assert result["passed"] is True
    assert result["absent_categories"] == [
        "unsupported",
        "contradicted",
        "irrelevant",
        "deleted",
        "occupancy_0",
        "occupancy_1",
        "occupancy_2",
        "occupancy_3",
        "source_current",
        "source_none",
        "visible_unknown",
    ]
    row["categories"][2] = "source_none"
    with pytest.raises(module.AuditError, match="categories"):
        module.recompute_rows([row], condition="original-k4", bootstrap_samples=20)


def test_a_failed_small_category_prevents_absolute_gate_at_95_percent_overall():
    module = auditor()
    rows = [deepcopy(original_row()) for _ in range(20)]
    for index, row in enumerate(rows):
        row["episode_id"] = index
        row["categories"] = ["visible_known"]
    rows[-1].update(selected=1, answer="20", categories=["visible_known", "rare"])
    result = module.recompute_rows(rows, bootstrap_samples=20)
    assert result["visible_accuracy"] == 0.95
    assert result["by_category"]["rare"] == {"count": 1, "accuracy": 0}
    assert result["absolute_gate_passed"] is False
    assert result["passed"] is False


def test_expected_conditions_require_all_39_predeclared_conditions():
    module = auditor()
    config = {
        "capacities": [4, 8],
        "current_counts": [0, 1, 6, 32],
        "capacity_policies": ["fifo", "recency", "random", "similarity", "cue_priority", "learned"],
        "threshold": 0.95,
    }
    conditions = module.expected_conditions(config)
    assert len(conditions) == 39
    assert "lifecycle" in conditions and "combined-B-k8" in conditions
    assert "varied-k8-c32" in conditions and "capacity-A-k4-similarity" in conditions
    config["authority_count"] = 20
    amended = module.expected_conditions(config)
    assert len(amended) == 41 and {"authority-k4", "authority-k8"} <= set(amended)
    config["capacity_policies"].remove("similarity")
    with pytest.raises(module.AuditError, match="policies"):
        module.expected_conditions(config)


def test_external_file_hashes_reject_changed_source_and_missing_files(tmp_path):
    module = auditor()
    path = tmp_path / "source.py"
    path.write_text("frozen\n")
    hashes = {"source.py": module.sha256(path)}
    assert module.verify_file_hashes(hashes, tmp_path) == 1
    path.write_text("changed\n")
    with pytest.raises(module.AuditError, match="hash"):
        module.verify_file_hashes(hashes, tmp_path)
    path.unlink()
    with pytest.raises(module.AuditError, match="missing"):
        module.verify_file_hashes(hashes, tmp_path)


def test_checkpoint_metadata_is_checked_against_training_config_and_log(tmp_path):
    module = auditor()
    train, reference = tmp_path / "train", tmp_path / "reference"
    train.mkdir()
    reference.mkdir()
    config = {"model": {"d_model": 2, "n_heads": 1}, "steps": 4}
    metadata = {"seed": 7, "parameters": 1, "config": config}
    (train / "config.json").write_text(json.dumps(config))
    (train / "training-7.json").write_text(json.dumps(metadata))
    torch.save(
        {
            "settings": config["model"],
            "state": {"weight": torch.tensor([1.0])},
            "metadata": metadata,
        },
        train / "reader-7.pt",
    )
    for name in ("reader", "lifecycle", "capacity"):
        (reference / f"{name}-7.pt").write_bytes(b"frozen reference fixture")
    paths = [train / "reader-7.pt", train / "training-7.json", *reference.glob("*.pt")]
    hashes = {str(path.relative_to(tmp_path)): module.sha256(path) for path in paths}
    result = module.verify_checkpoints(hashes, [7], tmp_path)
    assert result["7"]["parameters"] == 1
    config["steps"] = 5
    (train / "config.json").write_text(json.dumps(config))
    with pytest.raises(module.AuditError, match="config"):
        module.verify_checkpoints(hashes, [7], tmp_path)


def test_capacity_world_and_visible_records_are_checked_against_saved_write_prefix():
    module = auditor()
    prefixes = {0: [operation(1, 1, 10), operation(2, 1, 20), operation(1, 2, 30)]}
    bank = [[1, 0, 20, 1]]
    records = [
        {
            "query_index": 0,
            "entity": 1,
            "attribute": 0,
            "truth": "20",
            "answer": "20",
            "correct": True,
            "historical_answerable": True,
        },
        {
            "query_index": 1,
            "entity": 2,
            "attribute": 0,
            "truth": "30",
            "answer": "??",
            "correct": False,
            "historical_answerable": True,
        },
    ]
    raw = {
        "episodes": [
            {
                "episode_id": 0,
                "final_bank": bank,
                "records": records,
                "hits": 1,
                "queries": 2,
                "utility": 0.5,
                "oracle_hits": 1,
                "oracle_utility": 0.5,
                "oracle_regret": 0.0,
            }
        ],
        "summary": {
            "utility": 0.5,
            "oracle_utility": 0.5,
            "oracle_regret": 0.0,
            "queries": 2,
            "persistent_state_bytes": 16,
        },
    }
    rows = [
        {
            "memory": [operation(1, 1, 20)],
            "current": [],
            "state_bytes": 16,
            "episode_id": 0,
            "world": truth,
            "query": operation(5, entity),
        }
        for entity, truth in ((1, "20"), (2, "30"))
    ]
    module.verify_capacity_evidence(raw, rows, slots=1, policy="fifo", prefixes=prefixes)
    rows[0]["world"] = "99"
    records[0]["truth"] = "99"
    with pytest.raises(module.AuditError, match="world|truth"):
        module.verify_capacity_evidence(raw, rows, slots=1, policy="fifo", prefixes=prefixes)


def test_a_run_missing_completion_fields_is_rejected_before_any_metric_claim(tmp_path):
    module = auditor()
    (tmp_path / "summary.json").write_text(json.dumps({"seeds": {}}))
    with pytest.raises(module.AuditError, match="incomplete"):
        module.audit_run(tmp_path, source_root=tmp_path)


def authority_rows():
    names = {1: "SET", 2: "UPDATE", 3: "DELETE"}
    pairs = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    rows = []
    for index, (first, last) in enumerate(pairs * 2):
        current = index >= 9
        pair = [
            operation(first, 1, None if first == 3 else 40),
            operation(last, 1, None if last == 3 else 70),
        ]
        memory = [operation(1, 2, 20)] * 4 if current else [operation(1, 2, 20)] * 2 + pair
        evidence = [operation(4, 1, 99)] * 30 + pair if current else []
        target = "??" if last == 3 else "70"
        selected = 35 if current else 3
        rows.append(
            {
                "memory": memory,
                "current": evidence,
                "query": operation(5, 1),
                "world": target,
                "state_bytes": 64,
                "episode_id": index,
                "visible": target,
                "current_only": target if current else "??",
                "answer": target,
                "no_memory_answer": target if current else "??",
                "selected": selected,
                "no_memory_selected": 31 if current else -1,
                "oracle": selected,
                "categories": [
                    f"transition_{names[first]}_{names[last]}",
                    "occupancy_4",
                    "source_current" if current else "source_memory",
                    "visible_unknown" if last == 3 else "visible_known",
                ],
            }
        )
    return rows


def test_authority_transition_labels_come_from_matching_operations():
    module = auditor()
    rows = authority_rows()
    result = module.recompute_rows(rows, condition="authority-k4", bootstrap_samples=20)
    assert result["passed"] is True
    assert result["by_category"]["transition_DELETE_SET"] == {"count": 2, "accuracy": 1}
    rows[3]["categories"][0] = "transition_SET_UPDATE"
    with pytest.raises(module.AuditError, match="categories"):
        module.recompute_rows(rows, condition="authority-k4", bootstrap_samples=20)


def test_validly_recorded_stale_selection_is_a_reader_failure_not_a_corrupt_artifact():
    module = auditor()
    rows = authority_rows()
    rows[3].update(selected=2, answer="40")
    result = module.recompute_rows(rows, condition="authority-k4", bootstrap_samples=20)
    assert result["visible_accuracy"] == 17 / 18
    assert result["by_category"]["transition_UPDATE_SET"]["accuracy"] == 0.5
    assert result["passed"] is False


def test_lifecycle_targets_use_write_history_instead_of_saved_answer_labels():
    module = auditor()
    initial = [[1, 0, 10, 0], [2, 0, 30, 0], [-1] * 4, [-1] * 4]
    updated = [[1, 0, 20, 0], [2, 0, 30, 0], [-1] * 4, [-1] * 4]
    operations = [
        {
            "episode_id": 0,
            "session": 0,
            "operation_index": 0,
            "input_kind": "SET",
            "entity": 1,
            "attribute": 0,
            "value": 10,
            "bank_after": initial,
        },
        {
            "episode_id": 0,
            "session": 0,
            "operation_index": 1,
            "input_kind": "SET",
            "entity": 2,
            "attribute": 0,
            "value": 30,
            "bank_after": initial,
        },
        {
            "episode_id": 0,
            "session": 1,
            "operation_index": 0,
            "input_kind": "UPDATE",
            "entity": 1,
            "attribute": 0,
            "value": 20,
            "bank_after": updated,
        },
    ]
    queries = [
        {
            "episode_id": 0,
            "session": 2,
            "operation_index": index,
            "entity": entity,
            "attribute": 0,
            "case": "update",
            "control": bool(index),
            "answer": truth,
            "expected": truth,
            "correct": True,
            "bank": updated,
        }
        for index, entity, truth in ((0, 1, "20"), (1, 2, "30"))
    ]
    rows = [
        {
            "episode_id": 0,
            "memory": [operation(1, 1, 20), operation(1, 2, 30)],
            "current": [],
            "query": operation(5, entity),
            "world": truth,
            "state_bytes": 64,
        }
        for entity, truth in ((1, "20"), (2, "30"))
    ]
    raw = {"operations": operations, "queries": queries}
    assert module.verify_lifecycle_evidence(raw, rows) == [["update"], ["control"]]
    queries[0]["expected"] = rows[0]["world"] = "10"
    with pytest.raises(module.AuditError, match="expected"):
        module.verify_lifecycle_evidence(raw, rows)


def test_complete_seed_inventory_cannot_omit_an_unfavorable_condition(tmp_path):
    module = auditor()
    config = {
        "capacities": [4, 8],
        "current_counts": [0, 1, 6, 32],
        "threshold": 0.95,
        "authority_count": 1800,
        "seeds": [7],
        "capacity_policies": ["fifo", "recency", "random", "similarity", "cue_priority", "learned"],
    }
    summary = {
        "seconds": 1,
        "passed": True,
        "artifact_hashes": {},
        "hashes": {},
        "config": config,
        "seeds": {"7": {"conditions": {}}},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(module.AuditError, match="complete conditions"):
        module.audit_run(tmp_path, source_root=tmp_path)
