"""Variable-occupancy reader stress tasks, separate from the frozen v1 generator."""

import random

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import UNKNOWN, Kind, Operation
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.memory_experiments.reader_cases import (
    STRATA,
    ReaderTask,
    empty_memory,
    oracle_select,
)


def varied_tasks(
    split: str, *, seed: int, count: int, capacity: int, current_count: int
) -> list[ReaderTask]:
    if count < 1 or capacity < 1 or current_count < 0:
        raise ValueError("invalid task dimensions")
    rng = random.Random(seed)
    symbols = symbol_space(split)
    tasks = []
    for index in range(count):
        # Cross the five question types with every occupancy, rather than
        # accidentally aliasing the two periods when capacity is four.
        occupancy = (index // len(STRATA)) % (capacity + 1)
        query = Operation(Kind.ASK, rng.choice(symbols.entities), rng.randrange(4))
        peers = [e for e in symbols.entities if e != query.entity]
        near = [e for e in peers if e // 10 == query.entity // 10 or e % 10 == query.entity % 10]
        other = (query.attribute + rng.randrange(1, 4)) % 4
        length = occupancy + current_count
        records = []
        for _ in range(length):
            key = rng.choice(
                (
                    (rng.choice(near or peers), query.attribute),
                    (query.entity, other),
                    (rng.choice(peers), rng.randrange(4)),
                )
            )
            kind = rng.choice((Kind.SET, Kind.UPDATE, Kind.DELETE, Kind.NOISE))
            if kind == Kind.NOISE and rng.random() < 0.5:
                key = query.key
            records.append(
                Operation(kind, *key, None if kind == Kind.DELETE else rng.choice(symbols.values))
            )
        stratum = STRATA[index % 5]
        if length == 0:
            stratum = "unsupported"
        elif stratum == "irrelevant" and not current_count:
            stratum = "select"
        elif stratum == "contradicted" and length < 2:
            stratum = "select"
        if length and stratum != "unsupported":
            if stratum == "irrelevant":
                positions = [rng.randrange(occupancy, length)]
            elif stratum in ("contradicted", "deleted") and length >= 2:
                positions = sorted(rng.sample(range(length), min(length, rng.randint(2, 5))))
            else:
                positions = [rng.randrange(length)]
            values = rng.sample(symbols.values, len(positions))
            for j, (position, value) in enumerate(zip(positions, values, strict=True)):
                kind = Kind.SET if j == 0 else Kind.UPDATE
                if stratum == "deleted" and j == len(positions) - 1:
                    kind, value = Kind.DELETE, None
                records[position] = Operation(kind, *query.key, value)
        view = QueryView(
            tuple(records[:occupancy]), tuple(records[occupancy:]), query, capacity * 16, ()
        )
        target = read_visible(view.memory, view.current, query)
        selected = oracle_select(view)
        origin = "none" if selected == -1 else ("memory" if selected < occupancy else "current")
        tasks.append(
            ReaderTask(
                index,
                stratum,
                view,
                empty_memory(view),
                target,
                read_visible((), view.current, query),
                selected,
                occupancy,
                target != UNKNOWN,
                origin,
            )
        )
    return tasks


def authority_tasks(
    split: str, *, seed: int, count: int, capacity: int, current_count: int = 32
) -> list[ReaderTask]:
    """Cross all nine ordered authoritative-kind pairs with memory/current origin."""
    if count < 1 or capacity < 2 or current_count < 2:
        raise ValueError("positive count and two slots required")
    rng = random.Random(seed)
    symbols = symbol_space(split)
    kinds = (Kind.SET, Kind.UPDATE, Kind.DELETE)
    tasks = []
    for index in range(count):
        first, last = kinds[(index % 9) // 3], kinds[index % 3]
        in_current = (index // 9) % 2 == 1
        entity, other = rng.sample(symbols.entities, 2)
        attribute = rng.randrange(4)
        old, new = rng.sample(symbols.values, 2)
        query = Operation(Kind.ASK, entity, attribute)
        memory = [
            Operation(Kind.SET, other, rng.randrange(4), rng.choice(symbols.values))
            for _ in range(capacity)
        ]
        current = (
            [
                Operation(Kind.NOISE, entity, attribute, rng.choice(symbols.values))
                for _ in range(current_count)
            ]
            if in_current
            else []
        )
        pair = [
            Operation(first, entity, attribute, None if first == Kind.DELETE else old),
            Operation(last, entity, attribute, None if last == Kind.DELETE else new),
        ]
        destination = current if in_current else memory
        destination[-2:] = pair
        view = QueryView(tuple(memory), tuple(current), query, capacity * 16, ())
        target = read_visible(view.memory, view.current, query)
        tasks.append(
            ReaderTask(
                index,
                f"transition_{first.name}_{last.name}",
                view,
                empty_memory(view),
                target,
                read_visible((), view.current, query),
                oracle_select(view),
                capacity,
                target != UNKNOWN,
                "current" if in_current else "memory",
            )
        )
    return tasks
