"""Unbounded truth used ONLY by the generator/scorer, never by memory policies."""

from collections.abc import Iterable

from .operations import UNKNOWN, Kind, Operation, QueryTruth


class StateMachine:
    def __init__(self) -> None:
        self.facts: dict[tuple[int, int], int] = {}

    def apply(self, operation: Operation) -> str | None:
        if operation.kind in (Kind.SET, Kind.UPDATE):
            # UPDATE is an authoritative assignment, including on partial views.
            assert operation.value is not None
            self.facts[operation.key] = operation.value
        elif operation.kind == Kind.DELETE:
            self.facts.pop(operation.key, None)
        elif operation.kind == Kind.ASK:
            value = self.facts.get(operation.key)
            return UNKNOWN if value is None else f"{value:02d}"
        return None


def annotate_query(sessions: Iterable[tuple[Operation, ...]]) -> QueryTruth:
    """Annotate the final ASK from complete ground truth, not a policy output."""
    state = StateMachine()
    assignments: dict[tuple[int, int], list[str]] = {}
    last_evidence: dict[tuple[int, int], tuple[int, Kind]] = {}
    truth = None
    for session_index, session in enumerate(sessions):
        for operation in session:
            answer = state.apply(operation)
            if operation.kind in (Kind.SET, Kind.UPDATE):
                assignments.setdefault(operation.key, []).append(f"{operation.value:02d}")
            if operation.kind in (Kind.SET, Kind.UPDATE, Kind.DELETE):
                last_evidence[operation.key] = session_index, operation.kind
            if answer is not None:
                values = assignments.get(operation.key, [])
                last = last_evidence.get(operation.key)
                truth = QueryTruth(
                    answer=answer,
                    stale_values=tuple(sorted(set(values) - {answer})),
                    deleted=last is not None and last[1] == Kind.DELETE,
                    overwrites=max(0, len(values) - 1),
                    evidence_gap=None if last is None else session_index - last[0],
                )
    if truth is None:
        raise ValueError("episode has no ASK")
    return truth
