"""Seeded episodes independent of models and memory policies."""

import random
from dataclasses import dataclass
from functools import lru_cache

from .operations import Episode, Kind, Operation
from .state_machine import annotate_query

CASES = (
    "history_set",
    "history_update",
    "history_delete",
    "unknown",
    "current_set",
    "current_update",
    "current_delete",
    "near_miss",
)


@dataclass(frozen=True)
class SymbolSpace:
    entities: tuple[int, ...]
    values: tuple[int, ...]


@lru_cache(maxsize=3)
def symbol_space(split: str) -> SymbolSpace:
    partitions = {"train": (0, 60), "validation": (60, 80), "test": (80, 100)}
    if split not in partitions:
        raise ValueError(f"unknown split: {split}")
    start, end = partitions[split]
    entities, values = list(range(100)), list(range(100))
    random.Random(314159).shuffle(entities)
    random.Random(271828).shuffle(values)
    return SymbolSpace(tuple(entities[start:end]), tuple(values[start:end]))


@dataclass(frozen=True)
class EpisodeConfig:
    candidates: int = 30
    sessions: int = 8
    updates: int = 1
    noise_rate: float = 0.25

    def __post_init__(self) -> None:
        if self.sessions < 2 or self.candidates < self.sessions:
            raise ValueError("need at least two sessions and one candidate per session")
        if not 1 <= self.updates <= 16 or self.candidates < self.updates + 4:
            raise ValueError("updates must be in [1, 16] with enough candidate positions")
        if not 0 <= self.noise_rate <= 1:
            raise ValueError("noise_rate must lie in [0, 1]")


def generate_episodes(config: EpisodeConfig, *, split: str, seed: int, count: int) -> list[Episode]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    symbols = symbol_space(split)
    episodes = []
    for index in range(count):
        case = CASES[index % len(CASES)]
        entity, attribute = rng.choice(symbols.entities), rng.randrange(4)
        key = entity, attribute
        values = rng.sample(symbols.values, config.updates + 2)
        other_keys = [(e, a) for e in symbols.entities for a in range(4) if (e, a) != key]

        def distractor(keys: list[tuple[int, int]]) -> Operation:
            other_entity, other_attribute = rng.choice(keys)
            kind = Kind.NOISE if rng.random() < config.noise_rate else Kind.SET
            return Operation(kind, other_entity, other_attribute, rng.choice(symbols.values))

        history_target: list[Operation] = []
        local: list[Operation] = []
        if case not in ("unknown", "near_miss"):
            history_target.append(Operation(Kind.SET, *key, values[0]))
        if "update" in case:
            changes = [Operation(Kind.UPDATE, *key, v) for v in values[1 : config.updates + 1]]
            if case == "current_update":
                history_target.extend(changes[:-1])
                local.append(changes[-1])
            else:
                history_target.extend(changes)
        elif case == "current_set":
            local.append(Operation(Kind.SET, *key, values[1]))
        elif "delete" in case:
            destination = local if case == "current_delete" else history_target
            destination.append(Operation(Kind.DELETE, *key))
        if rng.random() < 0.5:
            local.insert(0, distractor(other_keys))
        history_count = config.candidates - len(local)
        history = [distractor(other_keys) for _ in range(history_count)]
        positions = sorted(rng.sample(range(history_count), len(history_target)))
        for position, operation in zip(positions, history_target, strict=True):
            history[position] = operation
        if case == "near_miss":
            # Same attribute; a nearby entity, never the exact queried key.
            peers = [e for e in symbols.entities if e != entity]
            rng.shuffle(peers)
            peer = max(
                peers,
                key=lambda e: sum(a == b for a, b in zip(f"{e:02d}", f"{entity:02d}", strict=True)),
            )
            history[-1] = Operation(Kind.SET, peer, attribute, values[0])
        sessions = tuple(
            tuple(
                history[
                    i * history_count // (config.sessions - 1) : (i + 1)
                    * history_count
                    // (config.sessions - 1)
                ]
            )
            for i in range(config.sessions - 1)
        ) + (tuple(local) + (Operation(Kind.ASK, *key),),)
        episodes.append(Episode(index, case, sessions, annotate_query(sessions)))
    return episodes
