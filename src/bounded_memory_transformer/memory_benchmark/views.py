"""The strict boundary: bounded memory + current session, never raw history."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .operations import UNKNOWN, Kind, Operation
from .policies import MemoryPolicy


@dataclass(frozen=True)
class QueryView:
    memory: tuple[Operation, ...]
    current: tuple[Operation, ...]
    query: Operation
    state_bytes: int
    retained: tuple[Operation, ...]


def stream_views(
    sessions: Iterable[tuple[Operation, ...]], policy_name: str, capacity: int
) -> Iterator[QueryView]:
    policy = MemoryPolicy(policy_name, capacity)
    for session in sessions:
        current: list[Operation] = []
        for operation in session:
            if operation.kind == Kind.ASK:
                yield QueryView(
                    policy.retrieve(operation),
                    tuple(current),
                    operation,
                    policy.state.nbytes,
                    policy.records(),
                )
            else:
                current.append(operation)
        for operation in current:
            policy.observe(operation)
        # Nothing except the fixed array is carried into the next session.
        current.clear()


def read_visible(
    memory: tuple[Operation, ...], current: tuple[Operation, ...], query: Operation
) -> str:
    """Independent exact-key reference reader; never sees full episode truth."""
    answer = UNKNOWN
    for operation in (*memory, *current):
        if operation.key != query.key:
            continue
        if operation.kind in (Kind.SET, Kind.UPDATE):
            answer = f"{operation.value:02d}"
        elif operation.kind == Kind.DELETE:
            answer = UNKNOWN
    return answer
