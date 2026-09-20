"""Independent expectations for bounded admission and delayed-reward learning."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_experiments.capacity import (
    CapacityEpisode,
    RetentionScorer,
    bank_records,
    evaluate_capacity,
    generate_capacity,
    make_capacity_policy,
    oracle_retention,
    read_bank,
    score_bank,
    train_capacity,
)


def fixture_episode():
    writes = tuple((Operation(Kind.SET, e, 0, e + 20), e % 2) for e in range(4))
    queries = tuple(Operation(Kind.ASK, e, 0) for e in [0] * 5 + [1] * 3 + [2] * 2)
    return CapacityEpisode(0, "A", (writes,), queries)


def test_hand_counted_oracle_and_retention_regret():
    episode = fixture_episode()
    oracle = oracle_retention(episode, 2)
    assert oracle["utility"] == pytest.approx(.8)
    assert oracle["bank"].tolist() == [[0, 0, 20, 0], [1, 0, 21, 1]]
    chosen = np.array([[1, 0, 21, 1], [2, 0, 22, 0]], dtype=np.int32)
    assert score_bank(episode, chosen)["utility"] == pytest.approx(.5)
    assert oracle["utility"] - score_bank(episode, chosen)["utility"] == pytest.approx(.3)


def test_generator_paired_prefix_deterministic_symbols_and_workload_counts():
    a = generate_capacity("train", 7, 3, workload="A")
    b = generate_capacity("train", 7, 3, workload="B")
    assert a == generate_capacity("train", 7, 3, workload="A")
    assert all(x.sessions == y.sessions for x, y in zip(a, b, strict=True))
    assert any(x.queries != y.queries for x, y in zip(a, b, strict=True))
    for episode in a:
        writes = [item for session in episode.sessions for item in session]
        assert len(episode.sessions) == 8
        assert len(writes) == 30
        assert len(episode.queries) == 32
        cues = {op.key: cue for op, cue in writes}
        assert len(cues) == 24
        assert sum(cues.values()) == 12
        assert sum(op.kind == Kind.UPDATE for op, _ in writes) == 6
        assert all(op.entity in symbol_space("train").entities for op, _ in writes)
        assert all(op.value in symbol_space("train").values for op, _ in writes)
    val = generate_capacity("validation", 7, 1)[0]
    assert all(op.entity in symbol_space("validation").entities
               for session in val.sessions for op, _ in session)


@pytest.mark.parametrize("name", ["fifo", "recency", "random", "similarity", "cue_priority",
                                  "learned", "no_memory"])
def test_online_trajectory_has_no_suffix_state_and_queries_never_reload(name):
    model = RetentionScorer()
    episode = generate_capacity("train", 7, 1)[0]
    changed = replace(episode, queries=tuple(reversed(episode.queries)))
    trajectories = []
    for item in (episode, changed):
        policy = make_capacity_policy(name, 4, model=model, seed=91)
        trajectory = []
        for session in item.sessions:
            for operation, cue in session:
                policy.observe(operation, cue)
                trajectory.append(policy.bank.copy())
            before = policy.bank.copy()
            policy.reset_session()
            np.testing.assert_array_equal(policy.bank, before)
        final = policy.bank.copy()
        for query in item.queries:
            policy.answer(query)
        np.testing.assert_array_equal(policy.bank, final)
        assert policy.bank.nbytes == (0 if name == "no_memory" else 64)
        assert policy.bank.dtype == np.int32
        trajectories.append(trajectory)
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_fifo_update_keeps_age_recency_refreshes_and_both_preserve_latest():
    for name, survivors in [("fifo", [13, 14]), ("recency", [12, 14])]:
        policy = make_capacity_policy(name, 2)
        policy.observe(Operation(Kind.SET, 12, 0, 23), 1)
        policy.observe(Operation(Kind.SET, 13, 0, 71), 0)
        policy.observe(Operation(Kind.UPDATE, 12, 0, 45), 1)
        assert policy.answer(Operation(Kind.ASK, 12, 0)) == "45"
        policy.observe(Operation(Kind.SET, 14, 0, 50), 0)
        assert policy.bank[:, 0].tolist() == survivors


def test_cue_priority_rejects_low_cue_and_handles_delete_noise_and_full_key():
    policy = make_capacity_policy("cue_priority", 2)
    policy.observe(Operation(Kind.SET, 12, 0, 23), 1)
    policy.observe(Operation(Kind.SET, 12, 1, 71), 1)
    policy.observe(Operation(Kind.SET, 13, 0, 10), 0)
    policy.observe(Operation(Kind.NOISE, 12, 0, 80), 1)
    assert policy.answer(Operation(Kind.ASK, 13, 0)) == "??"
    policy.observe(Operation(Kind.DELETE, 12, 0), 0)
    assert policy.answer(Operation(Kind.ASK, 12, 0)) == "??"
    assert policy.answer(Operation(Kind.ASK, 12, 1)) == "71"


def test_evaluator_matches_hand_calculation_and_exposes_query_records():
    result = evaluate_capacity([fixture_episode()], "fifo", 2)
    assert result["summary"]["utility"] == pytest.approx(.2)
    assert result["summary"]["oracle_utility"] == pytest.approx(.8)
    assert result["summary"]["oracle_regret"] == pytest.approx(.6)
    assert len(result["episodes"][0]["records"]) == 10
    assert result["summary"]["historical_tokens_reprocessed"] == 0


def test_tiny_delayed_reward_training_is_reproducible_and_saveable(tmp_path):
    config = {"episodes": 4, "capacity": 4, "learning_rate": .01}
    left = train_capacity(config, seed=3, device="cpu")
    right = train_capacity(config, seed=3, device="cpu")
    assert [row["workload"] for row in left.curve] == ["A", "B", "A", "B"]
    assert left.curve == right.curve
    assert all(np.isfinite(row["loss"]) for row in left.curve)
    assert left.metadata["training_signal"] == "delayed_answer_reward_reinforce"
    path = tmp_path / "capacity.pt"
    torch.save(left.model.state_dict(), path)
    restored = RetentionScorer()
    restored.load_state_dict(torch.load(path, weights_only=True))
    episode = generate_capacity("validation", 31, 1)[0]
    a = evaluate_capacity([episode], "learned", 4, model=left.model)
    b = evaluate_capacity([episode], "learned", 4, model=restored)
    assert a["episodes"][0]["final_bank"] == b["episodes"][0]["final_bank"]


def test_capacity_rejects_interleaved_queries_outside_oracle_contract():
    with pytest.raises(ValueError, match="write"):
        CapacityEpisode(0, "A", (((Operation(Kind.ASK, 1, 0), 0),),),
                        (Operation(Kind.ASK, 1, 0),))


def test_capacity_rejects_non_query_probe_and_nonbinary_stored_cue():
    with pytest.raises(ValueError, match="ASK"):
        CapacityEpisode(0, "A", fixture_episode().sessions, (Operation(Kind.SET, 1, 0, 2),))
    with pytest.raises(ValueError, match="cue"):
        CapacityEpisode(0, "A", (((Operation(Kind.SET, 1, 0, 2), 3),),),
                        (Operation(Kind.ASK, 1, 0),))


def test_oracle_selects_latest_value_and_is_feasible_without_reload():
    base = fixture_episode()
    episode = replace(base, sessions=(base.sessions[0],
                      ((Operation(Kind.UPDATE, 0, 0, 90), 0),)))
    oracle = oracle_retention(episode, 2)
    assert oracle["bank"].tolist() == [[0, 0, 90, 0], [1, 0, 21, 1]]
    clairvoyant = make_capacity_policy("fifo", 2)
    for session in episode.sessions:
        for operation, cue in session:
            if operation.key in {(0, 0), (1, 0)}:
                clairvoyant.observe(operation, cue)
        clairvoyant.reset_session()
    np.testing.assert_array_equal(clairvoyant.bank, oracle["bank"])


def test_learned_capacity_uses_same_weights_and_literal_observed_cue_at_eight_slots():
    scorer = RetentionScorer()
    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.fill_(.1)
    for capacity in (4, 8):
        policy = make_capacity_policy("learned", capacity, model=scorer)
        for entity in range(capacity):
            policy.observe(Operation(Kind.SET, entity, 0, 20), 0)
        policy.observe(Operation(Kind.SET, 20, 0, 45), 1)
        assert policy.answer(Operation(Kind.ASK, 20, 0)) == "45"
        assert policy.bank.nbytes == capacity * 16
        before = policy.bank.copy()
        policy.observe(Operation(Kind.SET, 21, 0, 70), 0)
        np.testing.assert_array_equal(policy.bank, before)


def test_similarity_retrieves_one_near_miss_that_exact_reader_rejects():
    policy = make_capacity_policy("similarity", 4)
    policy.observe(Operation(Kind.SET, 12, 0, 23), 0)
    policy.observe(Operation(Kind.SET, 13, 1, 71), 0)
    query = Operation(Kind.ASK, 12, 2)
    assert policy.retrieve(query) == (Operation(Kind.SET, 12, 0, 23),)
    assert policy.answer(query) == "??"
    exact = Operation(Kind.ASK, 13, 1)
    assert policy.retrieve(exact) == (Operation(Kind.SET, 13, 1, 71),)
    assert policy.answer(exact) == "71"
    fifo = make_capacity_policy("fifo", 4)
    fifo.bank[:] = policy.bank
    assert len(fifo.retrieve(query)) == 2


def test_no_memory_uses_zero_bytes_but_reports_requested_comparison_capacity():
    policy = make_capacity_policy("no_memory", 4)
    assert policy.bank.shape == (0, 4)
    assert policy.bank.nbytes == 0
    op = Operation(Kind.SET, 12, 0, 23)
    assert policy.select_allocation(policy.bank, op, 1) is None
    policy.observe(op, 1)
    assert policy.retrieve(Operation(Kind.ASK, 12, 0)) == ()
    assert policy.answer(Operation(Kind.ASK, 12, 0)) == "??"
    result = evaluate_capacity([fixture_episode()], "no_memory", 4)
    assert result["summary"]["capacity"] == 4
    assert result["summary"]["persistent_state_bytes"] == 0
    assert result["summary"]["utility"] == 0


@pytest.mark.parametrize("name", ["no_memory", "fifo", "learned"])
def test_external_allocation_rejects_invalid_observed_cue(name):
    policy = make_capacity_policy(name, 4, model=RetentionScorer())
    with pytest.raises(ValueError, match="cue"):
        policy.select_allocation(policy.bank, Operation(Kind.SET, 12, 0, 23), 2)


def test_wrong_writer_negative_value_preserves_key_as_delete_evidence():
    bank = np.array([[12, 0, 23, 0], [13, 1, -1, 1]], dtype=np.int32)
    assert bank_records(bank) == (Operation(Kind.SET, 12, 0, 23),
                                  Operation(Kind.DELETE, 13, 1))
    assert read_bank(bank, Operation(Kind.ASK, 13, 1)) == "??"
    assert read_bank(bank, Operation(Kind.ASK, 12, 0)) == "23"
    np.testing.assert_array_equal(bank, [[12, 0, 23, 0], [13, 1, -1, 1]])
    # An erroneous duplicate target remains observable; the last row wins.
    duplicate = np.array([[12, 0, 23, 0], [12, 0, -1, 0]], dtype=np.int32)
    assert len(bank_records(duplicate)) == 2
    assert read_bank(duplicate, Operation(Kind.ASK, 12, 0)) == "??"


@pytest.mark.parametrize("invalid_value", [-2, 100])
def test_only_minus_one_is_a_valid_negative_writer_value(invalid_value):
    bank = np.array([[12, 0, invalid_value, 0]], dtype=np.int32)
    with pytest.raises(ValueError):
        bank_records(bank)
    with pytest.raises(ValueError):
        read_bank(bank, Operation(Kind.ASK, 12, 0))
