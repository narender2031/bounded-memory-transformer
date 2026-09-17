import pytest

from bounded_memory_transformer.memory_benchmark.generator import (
    CASES,
    EpisodeConfig,
    generate_episodes,
    symbol_space,
)
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_benchmark.state_machine import StateMachine


def test_reference_state_replaces_conflicts_ignores_noise_and_deletes():
    state = StateMachine()
    query = Operation(Kind.ASK, 12, 0)
    assert state.apply(query) == "??"
    state.apply(Operation(Kind.SET, 12, 0, 23))
    state.apply(Operation(Kind.NOISE, 12, 0, 44))
    assert state.apply(query) == "23"
    state.apply(Operation(Kind.UPDATE, 12, 0, 45))
    assert state.apply(query) == "45"
    state.apply(Operation(Kind.SET, 12, 0, 67))
    assert state.apply(query) == "67"
    state.apply(Operation(Kind.DELETE, 12, 0))
    assert state.apply(query) == "??"
    state.apply(Operation(Kind.DELETE, 12, 0))
    assert state.apply(query) == "??"


@pytest.mark.parametrize(
    "args",
    [(Kind.SET, 1, 0, None), (Kind.DELETE, 1, 0, 2), (Kind.ASK, 100, 0, None), (Kind.SET, 1, 4, 2)],
)
def test_invalid_operations_are_rejected(args):
    with pytest.raises(ValueError):
        Operation(*args)


def test_seeded_generator_and_splits_and_independent_truth():
    config = EpisodeConfig()
    episodes = generate_episodes(config, split="test", seed=91, count=80)
    assert episodes == generate_episodes(config, split="test", seed=91, count=80)
    assert episodes != generate_episodes(config, split="test", seed=92, count=80)
    assert {episode.case for episode in episodes} == set(CASES)
    for episode in episodes:
        assert len(episode.sessions) == 8
        assert sum(op.kind != Kind.ASK for s in episode.sessions for op in s) == 30
        values = {}
        for session in episode.sessions:
            for op in session:
                if op.kind in (Kind.SET, Kind.UPDATE):
                    values[op.key] = f"{op.value:02d}"
                elif op.kind == Kind.DELETE:
                    values.pop(op.key, None)
                elif op.kind == Kind.ASK:
                    assert episode.truth.answer == values.get(op.key, "??")
                assert op.entity in symbol_space("test").entities
                if op.value is not None:
                    assert op.value in symbol_space("test").values
        assert episode.sessions[-1][-1].kind == Kind.ASK
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(symbol_space(first).entities) & set(symbol_space(second).entities)
        assert not set(symbol_space(first).values) & set(symbol_space(second).values)


def test_more_updates_and_longer_episodes_preserve_semantics():
    episodes = generate_episodes(
        EpisodeConfig(candidates=60, sessions=16, updates=4), split="test", seed=1, count=80
    )
    for episode in episodes:
        assert len(episode.sessions) == 16
        if "update" in episode.case:
            assert episode.truth.overwrites == 4
            assert len(episode.truth.stale_values) == 4
            assert episode.truth.answer not in episode.truth.stale_values


@pytest.mark.parametrize(
    "kwargs", [{"candidates": 0}, {"sessions": 1}, {"noise_rate": 1.2}, {"updates": 0}]
)
def test_impossible_episode_configs_fail_early(kwargs):
    with pytest.raises(ValueError):
        EpisodeConfig(**kwargs)
