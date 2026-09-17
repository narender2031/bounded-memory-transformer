"""Shared immutable symbols; no model or policy dependencies."""

from dataclasses import dataclass
from enum import IntEnum

UNKNOWN = "??"


class Kind(IntEnum):
    SET = 1
    UPDATE = 2
    DELETE = 3
    NOISE = 4
    ASK = 5


@dataclass(frozen=True)
class Operation:
    kind: Kind
    entity: int
    attribute: int
    value: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, Kind):
            raise ValueError("kind must be a Kind")
        if not 0 <= self.entity < 100 or not 0 <= self.attribute < 4:
            raise ValueError("entity must be in [0, 100); attribute in [0, 4)")
        requires_value = self.kind in (Kind.SET, Kind.UPDATE, Kind.NOISE)
        if requires_value != (self.value is not None):
            raise ValueError("SET/UPDATE/NOISE require values; DELETE/ASK forbid them")
        if self.value is not None and not 0 <= self.value < 100:
            raise ValueError("value must be in [0, 100)")

    @property
    def key(self) -> tuple[int, int]:
        return self.entity, self.attribute


@dataclass(frozen=True)
class QueryTruth:
    answer: str
    stale_values: tuple[str, ...] = ()
    deleted: bool = False
    overwrites: int = 0
    evidence_gap: int | None = None


@dataclass(frozen=True)
class Episode:
    episode_id: int
    case: str
    sessions: tuple[tuple[Operation, ...], ...]
    truth: QueryTruth
