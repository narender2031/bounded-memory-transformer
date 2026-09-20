"""Behavioral checks of causal lifecycle routing and its honest executor."""

import numpy as np
import pytest
import torch

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_experiments.lifecycle import (
    Action,
    Decision,
    LifecycleConfig,
    LifecycleModel,
    apply_decision,
    empty_bank,
    evaluate_lifecycle,
    exact_decision,
    generate_lifecycle,
    read_bank,
    train_lifecycle,
)


def populated():
    bank = empty_bank()
    bank[:3] = [[12, 0, 23, 0], [12, 1, 71, 1], [13, 0, 72, 0]]
    return bank


def test_update_binds_both_key_coordinates_and_preserves_controls():
    bank = populated()
    op = Operation(Kind.UPDATE, 12, 0, 45)
    assert exact_decision(bank, op) == Decision(Action.UPDATE, 0)
    changed = apply_decision(bank, op, Decision(Action.UPDATE, 0))
    assert changed.tolist() == [[12, 0, 45, 0], [12, 1, 71, 1], [13, 0, 72, 0], [-1] * 4]
    assert bank[0, 2] == 23
    assert changed.nbytes == 64


def test_delete_removes_only_target_and_missing_delete_is_ignored():
    bank = populated()
    op = Operation(Kind.DELETE, 12, 0)
    changed = apply_decision(bank, op, exact_decision(bank, op))
    assert read_bank(changed, Operation(Kind.ASK, 12, 0)) == "??"
    assert read_bank(changed, Operation(Kind.ASK, 12, 1)) == "71"
    assert read_bank(changed, Operation(Kind.ASK, 13, 0)) == "72"
    assert exact_decision(changed, op) == Decision(Action.IGNORE, -1)


def test_wrong_prediction_really_damages_control_and_is_not_gold_corrected():
    bank = populated()
    op = Operation(Kind.DELETE, 12, 0)
    changed = apply_decision(bank, op, Decision(Action.DELETE, 1))
    assert read_bank(changed, Operation(Kind.ASK, 12, 0)) == "23"
    assert read_bank(changed, Operation(Kind.ASK, 12, 1)) == "??"
    wrong_update = apply_decision(
        bank, Operation(Kind.UPDATE, 12, 0, 45), Decision(Action.UPDATE, 1)
    )
    assert read_bank(wrong_update, Operation(Kind.ASK, 12, 1)) == "??"


def test_ignore_and_existing_set_and_absent_update_semantics():
    bank = populated()
    for op in [Operation(Kind.NOISE, 12, 0, 44), Operation(Kind.ASK, 12, 0)]:
        assert exact_decision(bank, op) == Decision(Action.IGNORE, -1)
        np.testing.assert_array_equal(apply_decision(bank, op, exact_decision(bank, op)), bank)
    assert exact_decision(bank, Operation(Kind.SET, 12, 0, 44)) == Decision(Action.UPDATE, 0)
    assert exact_decision(bank, Operation(Kind.UPDATE, 14, 0, 44)) == Decision(Action.STORE, 3)


def test_full_allocation_is_explicit_and_capacity_decision_is_separate():
    bank = populated()
    bank[3] = [14, 0, 74, 1]
    op = Operation(Kind.SET, 15, 0, 75)
    decision = exact_decision(bank, op)
    assert decision == Decision(Action.STORE, 4)
    np.testing.assert_array_equal(apply_decision(bank, op, decision), bank)
    result = apply_decision(bank, op, decision, cue=1, allocation=2)
    assert result[2].tolist() == [15, 0, 75, 1]
    assert result.nbytes == 64
    with pytest.raises(ValueError):
        apply_decision(bank, op, decision, allocation=4)


def test_generator_deterministic_disjoint_and_exact_lifecycle_perfect():
    episodes = generate_lifecycle("validation", 7, 24)
    assert episodes == generate_lifecycle("validation", 7, 24)
    symbols = symbol_space("validation")
    for episode in episodes:
        for session in episode.sessions:
            for op in session:
                assert op.entity in symbols.entities
                assert op.value is None or op.value in symbols.values
    result = evaluate_lifecycle(exact_decision, episodes)
    assert result["summary"]["semantic_accuracy"] == 1
    assert result["summary"]["transition_accuracy"] == 1
    assert result["summary"]["action_macro_f1"] == 1
    assert result["summary"]["control_preservation"] == 1
    assert result["summary"]["stale_answer_rate"] == 0
    assert result["summary"]["deletion_nonabstention_rate"] == 0
    assert len(result["queries"]) == 48
    assert {r["case"] for r in result["queries"]} == {"update", "delete"}
    assert all(r["state_bytes"] == 64 for r in result["banks"])


def test_small_real_training_generalizes_features_and_has_no_episodic_state(tmp_path):
    config = LifecycleConfig(steps=120, batch_size=128, training_examples=1024)
    bundle = train_lifecycle(config, seed=7, device="cpu")
    assert bundle.learning_curve[-1]["loss"] < bundle.learning_curve[0]["loss"]
    episodes = generate_lifecycle("validation", 17, 24)
    result = evaluate_lifecycle(bundle, episodes)
    assert result["summary"]["semantic_accuracy"] >= 0.95
    before = bundle.decide(populated(), Operation(Kind.DELETE, 12, 0))
    bundle.decide(empty_bank(), Operation(Kind.SET, 99, 0, 33))
    assert bundle.decide(populated(), Operation(Kind.DELETE, 12, 0)) == before
    assert bundle.metadata["symbol_split"] == "train"
    checkpoint = tmp_path / "lifecycle.pt"
    torch.save(bundle.model.state_dict(), checkpoint)
    restored = LifecycleModel(config.hidden)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    expanded = empty_bank(8)
    expanded[:4] = populated()
    for bank in (populated(), expanded):
        operation = Operation(Kind.UPDATE, 12, 0, 45)
        assert restored.decide(bank, operation) == bundle.decide(bank, operation)
    assert expanded.nbytes == 128
    assert evaluate_lifecycle(restored, episodes, slots=8)["summary"]["semantic_accuracy"] >= 0.95


def test_failed_lifecycle_metrics_count_stale_and_deleted_answers_separately():
    def ignore_changes(bank, operation):
        decision = exact_decision(bank, operation)
        return (
            Decision(Action.IGNORE, -1)
            if decision.action in (Action.UPDATE, Action.DELETE)
            else decision
        )

    result = evaluate_lifecycle(ignore_changes, generate_lifecycle("validation", 3, 12))
    summary = result["summary"]
    assert summary["semantic_accuracy"] == 0.5
    assert summary["control_preservation"] == 1
    assert summary["stale_answer_rate"] == 1
    assert summary["deletion_nonabstention_rate"] == 1
    assert summary["deleted_value_repetition_rate"] == 1
    assert summary["transition_accuracy"] == pytest.approx(6 / 7)
    assert not summary["competence_gate"]
    assert summary["update_query_count"] == summary["delete_query_count"] == 6


def test_wrong_action_is_applied_without_input_kind_correction():
    bank = populated()
    operation = Operation(Kind.NOISE, 14, 3, 88)
    changed = apply_decision(bank, operation, Decision(Action.STORE, 0), cue=1)
    assert changed[0].tolist() == [14, 3, 88, 1]
    assert read_bank(changed, Operation(Kind.ASK, 12, 0)) == "??"
    changed = apply_decision(bank, Operation(Kind.DELETE, 12, 0), Decision(Action.UPDATE, 1))
    assert changed[1].tolist() == [12, 0, -1, 0]
    assert read_bank(changed, Operation(Kind.ASK, 12, 0)) == "??"


def test_invalid_bank_shape_and_dtype_rejected():
    operation = Operation(Kind.DELETE, 12, 0)
    for bank in [np.zeros((4, 4), dtype=np.int64), np.zeros((4, 3), dtype=np.int32)]:
        with pytest.raises(ValueError):
            exact_decision(bank, operation)
