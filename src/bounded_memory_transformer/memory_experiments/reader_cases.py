"""Visible-only reader controls and balanced, reproducible reading tasks."""

import random
from dataclasses import dataclass

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import UNKNOWN, Kind, Operation
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible

STRATA = ("select", "unsupported", "contradicted", "irrelevant", "deleted")


def oracle_select(view: QueryView) -> int:
    """Latest visible authoritative matching operation, or UNKNOWN (-1)."""
    selected = -1
    for index, operation in enumerate((*view.memory, *view.current)):
        if operation.key == view.query.key and operation.kind in (
            Kind.SET,
            Kind.UPDATE,
            Kind.DELETE,
        ):
            selected = index
    return selected


def copy_selected(view: QueryView, index: int) -> str:
    """Copy the chosen value, without checking or correcting the chosen key."""
    records = (*view.memory, *view.current)
    if index < -1 or index >= len(records):
        raise ValueError("selection index outside visible candidates")
    if index == -1:
        return UNKNOWN
    operation = records[index]
    return UNKNOWN if operation.value is None else f"{operation.value:02d}"


def project_selected(view: QueryView, index: int) -> QueryView:
    copy_selected(view, index)  # Validate even when this helper is used independently.
    memory, current = (), ()
    if 0 <= index < len(view.memory):
        memory = (view.memory[index],)
    elif index >= len(view.memory):
        current = (view.current[index - len(view.memory)],)
    return QueryView(memory, current, view.query, view.state_bytes, ())


def empty_memory(view: QueryView) -> QueryView:
    return QueryView((), view.current, view.query, 0, ())


@dataclass(frozen=True)
class ReaderTask:
    episode_id: int
    stratum: str
    view: QueryView
    empty_view: QueryView
    target: str
    empty_target: str
    oracle_index: int
    occupancy: int
    known: bool
    override_origin: str


def generate_reader_tasks(
    split: str, *, seed: int, count: int, capacity: int = 4
) -> list[ReaderTask]:
    if count <= 0 or capacity < 4:
        raise ValueError("positive count and at least four slots are required")
    symbols = symbol_space(split)
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        stratum = STRATA[index % len(STRATA)]
        cycle = index // len(STRATA)
        query = Operation(Kind.ASK, rng.choice(symbols.entities), rng.randrange(4))
        peer = rng.choice([e for e in symbols.entities if e != query.entity])
        other_attribute = (query.attribute + rng.randrange(1, 4)) % 4
        old, new = rng.sample(symbols.values, 2)
        peer_keys = (
            (query.entity, other_attribute),
            (peer, query.attribute),
            (peer, other_attribute),
        )
        distractor_values = [value for value in symbols.values if value not in (old, new)]

        def distractor(kind=None, key=None, keys=peer_keys, values=distractor_values):
            kind = kind or rng.choice((Kind.SET, Kind.UPDATE, Kind.DELETE, Kind.NOISE))
            key = key or rng.choice(keys)
            value = None if kind == Kind.DELETE else rng.choice(values)
            return Operation(kind, *key, value)

        # Partial-key distractors vary in operation kind, not merely in value.
        memory = [distractor(key=peer_keys[0]), distractor(key=peer_keys[1])]
        current_evidence = None
        override_origin = "none"
        if stratum == "select":
            evidence = Operation(Kind.SET, *query.key, new)
            if cycle % 2:
                current_evidence = evidence
            else:
                memory.append(evidence)
        elif stratum in ("contradicted", "deleted"):
            memory.append(Operation(Kind.SET, *query.key, old))
            override_origin = "current" if cycle % 2 else "memory"
            evidence = Operation(
                Kind.UPDATE if stratum == "contradicted" else Kind.DELETE,
                *query.key,
                new if stratum == "contradicted" else None,
            )
            if override_origin == "current":
                current_evidence = evidence
            else:
                memory.append(evidence)
        elif stratum == "irrelevant":
            current_evidence = Operation(Kind.SET, *query.key, new)
        else:
            current_evidence = Operation(Kind.NOISE, *query.key, new)
        while len(memory) < capacity:
            memory.append(distractor())
        # Every stratum has the same current-session kind multiset and length.
        # Two UPDATE candidates prevent a unique current UPDATE from identifying
        # the target. A later nonmatching SET defeats a last-current-value rule.
        current = [
            distractor(kind)
            for kind in (Kind.UPDATE, Kind.UPDATE, Kind.DELETE, Kind.SET, Kind.NOISE)
        ]
        if current_evidence is not None:
            locations = [i for i, op in enumerate(current) if op.kind == current_evidence.kind]
            current[rng.choice(locations)] = current_evidence
        if current_evidence is None or current_evidence.kind != Kind.NOISE:
            if rng.random() < 0.5:
                current[-1] = Operation(Kind.NOISE, *query.key, rng.choice(distractor_values))
        rng.shuffle(current)
        current.append(distractor(Kind.SET))
        # Randomize all locations, preserving only the chronology of matching events.
        matching = [o for o in memory if o.key == query.key]
        others = [o for o in memory if o.key != query.key]
        rng.shuffle(others)
        matching_positions = set(rng.sample(range(capacity), len(matching)))
        match_iter, other_iter = iter(matching), iter(others)
        memory = [
            next(match_iter) if slot in matching_positions else next(other_iter)
            for slot in range(capacity)
        ]
        view = QueryView(tuple(memory), tuple(current), query, capacity * 16, ())
        empty = empty_memory(view)
        selected = oracle_select(view)
        target = copy_selected(view, selected)
        # Independent reference implementation: disagreement is a generator bug.
        assert target == read_visible(view.memory, view.current, query)
        tasks.append(
            ReaderTask(
                index,
                stratum,
                view,
                empty,
                target,
                read_visible((), empty.current, query),
                selected,
                len(memory),
                target != UNKNOWN,
                override_origin,
            )
        )
    return tasks
