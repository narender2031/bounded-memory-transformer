import numpy as np
import pytest

from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_benchmark.policies import MemoryPolicy
from bounded_memory_transformer.memory_benchmark.views import read_visible, stream_views


def test_fifo_eviction_and_recency_supersession_are_distinct():
    operations = [
        Operation(Kind.SET, 12, 0, 10),
        Operation(Kind.SET, 23, 0, 20),
        Operation(Kind.UPDATE, 12, 0, 30),
        Operation(Kind.SET, 34, 0, 40),
    ]
    fifo, recency = MemoryPolicy("fifo", 3), MemoryPolicy("recency", 3)
    for op in operations:
        fifo.observe(op)
        recency.observe(op)
    assert [op.entity for op in fifo.records()] == [23, 12, 34]
    assert [op.entity for op in recency.records()] == [23, 12, 34]
    fifo.observe(Operation(Kind.SET, 45, 0, 50))
    recency.observe(Operation(Kind.SET, 45, 0, 50))
    assert len(recency.records()) == 3
    # Earlier duplicate consumed a FIFO slot but not a recency slot.
    fifo, recency = MemoryPolicy("fifo", 3), MemoryPolicy("recency", 3)
    for op in operations[:3]:
        fifo.observe(op)
        recency.observe(op)
    assert len(fifo.records()) == 3
    assert len(recency.records()) == 2


@pytest.mark.parametrize("name", ["fifo", "recency", "similarity"])
def test_fixed_capacity_noise_filter_and_deletion_tombstones(name):
    policy = MemoryPolicy(name, 4)
    for value in range(90):
        policy.observe(Operation(Kind.SET, 12, 0, value))
        assert policy.state.shape == (4, 4)
        assert policy.state.dtype == np.int32
        assert policy.state.nbytes == 64
    before = policy.state.copy()
    policy.observe(Operation(Kind.NOISE, 12, 0, 99))
    np.testing.assert_array_equal(policy.state, before)
    policy.observe(Operation(Kind.DELETE, 12, 0))
    query = Operation(Kind.ASK, 12, 0)
    assert read_visible(policy.retrieve(query), (), query) == "??"


def test_similarity_prefers_exact_key_then_newest_without_query_at_write_time():
    policy = MemoryPolicy("similarity", 4)
    for op in [
        Operation(Kind.SET, 12, 0, 10),
        Operation(Kind.SET, 13, 0, 20),
        Operation(Kind.UPDATE, 12, 0, 30),
        Operation(Kind.SET, 99, 3, 40),
    ]:
        policy.observe(op)
    selected = policy.retrieve(Operation(Kind.ASK, 12, 0))
    assert len(selected) == 1
    assert selected[0].value == 30


def test_hard_reset_exposes_only_slots_and_current_session():
    sessions = (
        (Operation(Kind.SET, 12, 0, 23), Operation(Kind.ASK, 12, 0)),
        (Operation(Kind.SET, 34, 0, 45),),
        (Operation(Kind.ASK, 12, 0),),
    )
    views = list(stream_views(iter(sessions), "fifo", 1))
    assert read_visible(views[0].memory, views[0].current, views[0].query) == "23"
    assert views[1].current == ()
    assert [op.entity for op in views[1].memory] == [34]
    assert read_visible(views[1].memory, views[1].current, views[1].query) == "??"
    # A previously emitted view must not alias later writes, nor the next episode.
    assert views[0].memory == ()
    fresh = list(stream_views([(Operation(Kind.ASK, 12, 0),)], "fifo", 1))
    assert fresh[0].memory == ()


def test_no_memory_has_no_persistent_state_but_can_read_current_evidence():
    policy = MemoryPolicy("no_memory", 4)
    policy.observe(Operation(Kind.SET, 12, 0, 23))
    assert policy.state.nbytes == 0
    query = Operation(Kind.ASK, 12, 0)
    view = list(stream_views([(Operation(Kind.SET, 12, 0, 23),), (query,)], "no_memory", 4))
    assert read_visible(view[0].memory, view[0].current, query) == "??"
    local = list(stream_views([(Operation(Kind.SET, 12, 0, 23), query)], "no_memory", 4))
    assert read_visible(local[0].memory, local[0].current, query) == "23"


def test_visible_reader_uses_current_update_and_rejects_near_key():
    query = Operation(Kind.ASK, 12, 0)
    memory = (Operation(Kind.SET, 12, 0, 23),)
    assert read_visible(memory, (Operation(Kind.UPDATE, 12, 0, 45),), query) == "45"
    assert read_visible((Operation(Kind.SET, 13, 0, 23),), (), query) == "??"
