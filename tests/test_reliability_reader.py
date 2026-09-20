from collections import Counter
from dataclasses import replace

import pytest

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.memory_experiments.reader_cases import (
    copy_selected,
    generate_reader_tasks,
    oracle_select,
    project_selected,
)


def fixture():
    return QueryView(
        (Operation(Kind.SET, 12, 0, 23),),
        (Operation(Kind.UPDATE, 12, 0, 45),),
        Operation(Kind.ASK, 12, 0),
        64,
        (),
    )


def test_literal_oracle_copy_and_no_gold_correction():
    view = fixture()
    assert oracle_select(view) == 1
    assert copy_selected(view, 1) == "45"
    assert copy_selected(view, 0) == "23"
    assert copy_selected(view, -1) == "??"
    assert copy_selected(replace(view, current=(Operation(Kind.DELETE, 12, 0),)), 1) == "??"
    assert oracle_select(replace(view, query=Operation(Kind.ASK, 12, 1))) == -1
    assert oracle_select(replace(view, memory=(), current=())) == -1
    with pytest.raises(ValueError):
        copy_selected(view, 2)
    with pytest.raises(ValueError):
        copy_selected(view, -2)


def test_retained_archive_is_not_evidence_and_projection_keeps_origin():
    view = fixture()
    forged = replace(view, retained=(Operation(Kind.UPDATE, 12, 0, 99),))
    assert oracle_select(forged) == 1
    assert project_selected(forged, 0).memory == view.memory
    assert project_selected(forged, 0).current == ()
    assert project_selected(forged, 1).current == view.current
    assert project_selected(forged, 1).memory == ()
    assert project_selected(forged, -1).retained == ()
    assert oracle_select(replace(forged, memory=(), current=())) == -1


def test_balanced_deterministic_visible_tasks_and_empty_pairs():
    tasks = generate_reader_tasks("train", seed=19, count=100)
    assert tasks == generate_reader_tasks("train", seed=19, count=100)
    assert Counter(t.stratum for t in tasks) == dict.fromkeys(
        ["select", "unsupported", "contradicted", "irrelevant", "deleted"], 20
    )
    symbols = symbol_space("train")
    for task in tasks:
        assert task.occupancy == len(task.view.memory) == 4
        assert task.target == read_visible(task.view.memory, task.view.current, task.view.query)
        assert task.target == copy_selected(task.view, task.oracle_index)
        assert task.empty_view.memory == ()
        assert task.empty_view.current == task.view.current
        assert task.empty_view.query == task.view.query
        assert task.empty_target == read_visible((), task.view.current, task.view.query)
        for operation in (*task.view.memory, *task.view.current, task.view.query):
            assert operation.entity in symbols.entities
            assert operation.value is None or operation.value in symbols.values
        if task.stratum in ("unsupported", "deleted"):
            assert task.target == "??"
        else:
            assert task.target != "??"
    for stratum in ("contradicted", "deleted"):
        assert {t.override_origin for t in tasks if t.stratum == stratum} == {"memory", "current"}
    # Every query faces both partial-key distractor types.
    for task in tasks:
        query = task.view.query
        records = (*task.view.memory, *task.view.current)
        assert any(o.entity == query.entity and o.attribute != query.attribute for o in records)
        assert any(o.entity != query.entity and o.attribute == query.attribute for o in records)


def test_within_bank_targets_are_not_fixed_to_last_slot():
    tasks = generate_reader_tasks("train", seed=19, count=100)
    indices = {t.oracle_index for t in tasks if t.override_origin == "memory"}
    assert len(indices) > 1
    for task in tasks:
        if task.override_origin == "memory":
            matching = [o for o in task.view.memory if o.key == task.view.query.key]
            assert matching[0].kind == Kind.SET
            assert matching[-1].kind in (Kind.UPDATE, Kind.DELETE)


def test_query_blind_operation_rules_fail_each_rejection_stratum():
    tasks = generate_reader_tasks("train", seed=9, count=500)

    def blind(view, take_last):
        records = (*view.memory, *view.current)
        updates = [o for o in records if o.kind == Kind.UPDATE]
        if updates:
            return f"{updates[-1 if take_last else 0].value:02d}"
        if any(o.kind == Kind.DELETE for o in records):
            return "??"
        current_sets = [o for o in view.current if o.kind == Kind.SET]
        return f"{current_sets[-1].value:02d}" if current_sets else "??"

    for stratum in ("unsupported", "contradicted", "irrelevant", "deleted"):
        subset = [t for t in tasks if t.stratum == stratum]
        for take_last in (False, True):
            errors = sum(blind(t.view, take_last) != t.target for t in subset)
            assert errors / len(subset) >= 0.1, (stratum, take_last, errors)
    assert {tuple(sorted(int(o.kind) for o in t.view.current)) for t in tasks} == {
        (1, 1, 2, 2, 3, 4)
    }
    for task in tasks:
        assert task.view.current[-1].kind == Kind.SET
        assert task.view.current[-1].key != task.view.query.key
        # Nonmatching kinds must appear in current evidence, too.
        assert any(
            o.kind == Kind.UPDATE and o.key != task.view.query.key for o in task.view.current
        )
        assert task.target == read_visible(task.view.memory, task.view.current, task.view.query)
        assert task.target == copy_selected(task.view, task.oracle_index)


def test_eight_slot_tasks_fit_character_context_and_preserve_literal_override():
    from bounded_memory_transformer.memory_benchmark.reader import render

    tasks = generate_reader_tasks("validation", seed=11, count=100, capacity=8)
    assert all(t.occupancy == 8 for t in tasks)
    assert max(len(render(t.view)) + 1 for t in tasks) <= 128
    assert max(len(t.view.current) for t in tasks) <= 6
    assert all(
        copy_selected(t.view, t.oracle_index)
        == read_visible(t.view.memory, t.view.current, t.view.query)
        for t in tasks
    )
