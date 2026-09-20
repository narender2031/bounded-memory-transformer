"""Behavioral contracts for the structured neural pointer follow-up."""

import pytest
import torch

from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.memory_experiments.reader_cases import ReaderTask, copy_selected
from bounded_memory_transformer.memory_followup.evaluate_reader import (
    task_categories,
    validate_protocol_configs,
)
from bounded_memory_transformer.memory_followup.reader import (
    FactorizedReader,
    encode_views,
    supervised_loss,
)
from bounded_memory_transformer.memory_followup.tasks import authority_tasks, varied_tasks
from bounded_memory_transformer.memory_followup.train_reader import freeze_comparator


def view(records=(), current=(), entity=13, attribute=0):
    return QueryView(tuple(records), tuple(current), Operation(Kind.ASK, entity, attribute), 64, ())


def test_encoder_preserves_both_entity_digits_and_attribute_but_excludes_values():
    a = view([Operation(Kind.SET, 13, 0, 25)])
    b = view([Operation(Kind.SET, 13, 0, 91)])
    c = view([Operation(Kind.SET, 23, 0, 25)])
    d = view([Operation(Kind.SET, 14, 0, 25)])
    e = view([Operation(Kind.SET, 13, 1, 25)])
    enc = encode_views([a, b, c, d, e])
    assert torch.equal(enc.pairs[0], enc.pairs[1])
    assert not torch.equal(enc.pairs[0, 0, 0], enc.pairs[2, 0, 0])
    assert not torch.equal(enc.pairs[0, 0, 1], enc.pairs[3, 0, 1])
    assert not torch.equal(enc.pairs[0, 0, 2], enc.pairs[4, 0, 2])


def test_padding_and_unrelated_batch_members_cannot_change_a_read():
    torch.manual_seed(5)
    reader = FactorizedReader().eval()
    short = view([Operation(Kind.SET, 13, 0, 25)])
    long = view([Operation(Kind.UPDATE, 23, 1, 45)] * 20)
    single = reader(encode_views([short]))[0]
    mixed = reader(encode_views([short, long]))[0]
    torch.testing.assert_close(single[0, 0], mixed[0, 0])
    assert torch.isneginf(mixed[0, 1:-1]).all()
    assert reader.select([short])[0] == reader.select([long, short])[1]


def test_empty_evidence_has_only_abstain_and_no_cross_query_state():
    reader = FactorizedReader().eval()
    empty = view()
    before = {name: value.clone() for name, value in reader.state_dict().items()}
    assert reader.select([empty]) == [-1]
    reader.select([view([Operation(Kind.SET, 13, 0, 25)])])
    assert reader.select([empty]) == [-1]
    for name, value in reader.state_dict().items():
        assert torch.equal(value, before[name])


def test_gradient_reaches_key_comparator_authority_and_chronology():
    reader = FactorizedReader()
    v = view(
        [Operation(Kind.SET, 13, 0, 25), Operation(Kind.UPDATE, 13, 0, 45)],
        [Operation(Kind.NOISE, 13, 0, 91), Operation(Kind.SET, 23, 0, 62)],
    )
    loss, _ = supervised_loss(reader, [v, view()])
    loss.backward()
    for parameter in (reader.embedding.weight, reader.authority.weight, reader.order_weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_frozen_comparator_cannot_drift_during_record_ranking_training():
    model = FactorizedReader()
    freeze_comparator(model)
    v = view([Operation(Kind.SET, 13, 0, 25), Operation(Kind.UPDATE, 13, 0, 45)])
    batch = encode_views([v])
    before = model(batch)[1].detach().clone()
    order = model.order_weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    for _ in range(3):
        loss, _ = supervised_loss(model, [v])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.equal(before, model(batch)[1].detach())
    assert not torch.equal(order, model.order_weight.detach())


def test_validity_confidence_cannot_override_later_authoritative_evidence():
    model = FactorizedReader(saturate=True, readout_mode="binary")
    with torch.no_grad():
        model.match.weight.zero_()
        model.match.bias.fill_(10.0)
        model.authority.weight.copy_(torch.tensor([[0.0], [0.02], [0.2], [6.0], [-3.0]]))
        model.order_weight.fill_(2.0)
    for first in (Kind.SET, Kind.UPDATE, Kind.DELETE):
        for last in (Kind.SET, Kind.UPDATE, Kind.DELETE):
            current = [Operation(Kind.NOISE, 13, 0, 91)] * 32
            current += [
                Operation(first, 13, 0, None if first == Kind.DELETE else 25),
                Operation(last, 13, 0, None if last == Kind.DELETE else 45),
            ]
            assert model.select([view(current=current)]) == [33]


def test_authority_challenge_covers_all_transitions_in_memory_and_current():
    tasks = authority_tasks("train", seed=61, count=180, capacity=4)
    combinations = {(t.stratum, len(t.view.current)) for t in tasks}
    assert len(combinations) == 18
    for task in tasks:
        records = (*task.view.memory, *task.view.current)
        assert task.oracle_index == len(records) - 1
        assert task.target == read_visible(task.view.memory, task.view.current, task.view.query)


def test_copy_does_not_repair_a_wrong_neural_key_selection():
    v = view([Operation(Kind.SET, 23, 0, 25)], [Operation(Kind.UPDATE, 13, 0, 45)])
    assert copy_selected(v, 0) == "25"
    assert copy_selected(v, 1) == "45"
    assert copy_selected(v, -1) == "??"


def test_binary_adapter_does_not_secretly_apply_symbolic_key_matching():
    model = FactorizedReader(readout_mode="binary")
    with torch.no_grad():
        model.match.weight.zero_()
        model.match.bias.fill_(10.0)
        model.authority.weight.fill_(10.0)
    wrong = view([Operation(Kind.SET, 23, 0, 25)])
    assert model.select([wrong]) == [0]
    assert copy_selected(wrong, 0) == "25"


def test_each_question_category_spans_each_occupancy_when_all_are_feasible():
    tasks = varied_tasks("train", seed=17, count=100, capacity=4, current_count=6)
    for category in ("select", "unsupported", "contradicted", "irrelevant", "deleted"):
        assert {t.occupancy for t in tasks if t.stratum == category} == {0, 1, 2, 3, 4}


def test_source_slice_uses_selected_evidence_not_legacy_override_label():
    v = view([Operation(Kind.SET, 13, 0, 25)])
    t = ReaderTask(0, "select", v, view(), "25", "??", 0, 1, True, "none")
    categories = task_categories([t], 4)
    assert categories["source_memory"] == [True]
    assert categories["source_current"] == [False]
    assert categories["source_none"] == [False]


def test_heldout_preflight_rejects_wrong_training_candidate_and_changed_evaluation():
    import json
    from pathlib import Path

    configs = Path("projects/04-memory-followup/configs")
    evaluation = json.loads((configs / "reader-evaluation.json").read_text())
    selected = json.loads((configs / "reader-candidate-4.json").read_text())
    earlier = json.loads((configs / "reader-candidate-2.json").read_text())
    validate_protocol_configs(evaluation, selected, smoke=False)
    with pytest.raises(ValueError, match="candidate"):
        validate_protocol_configs(evaluation, earlier, smoke=False)
    with pytest.raises(ValueError, match="evaluation"):
        validate_protocol_configs({**evaluation, "original_seed": 17}, selected, smoke=False)


@pytest.mark.parametrize("capacity,current", [(4, 0), (4, 1), (8, 12), (8, 32)])
def test_varied_tasks_obey_capacity_chronology_and_independent_visible_truth(capacity, current):
    tasks = varied_tasks("train", seed=67, count=100, capacity=capacity, current_count=current)
    assert tasks == varied_tasks(
        "train", seed=67, count=100, capacity=capacity, current_count=current
    )
    assert {t.occupancy for t in tasks} == set(range(capacity + 1))
    assert any(t.target == "??" for t in tasks)
    assert any(t.target != "??" for t in tasks)
    for task in tasks:
        v = task.view
        assert len(v.memory) <= capacity
        assert len(v.current) == current
        assert task.target == read_visible(v.memory, v.current, v.query)
        assert copy_selected(v, task.oracle_index) == task.target
