"""Fixed-capacity symbolic policies. No model, truth, or future-query access."""

import numpy as np

from .operations import Kind, Operation

POLICIES = ("no_memory", "fifo", "recency", "similarity")


class MemoryPolicy:
    __slots__ = ("name", "capacity", "state")

    def __init__(self, name: str, capacity: int) -> None:
        if name not in POLICIES or capacity <= 0:
            raise ValueError("unknown policy or nonpositive capacity")
        self.name = name
        self.capacity = 0 if name == "no_memory" else capacity
        self.state = np.full((self.capacity, 4), -1, dtype=np.int32)

    def records(self) -> tuple[Operation, ...]:
        return tuple(
            Operation(Kind(int(k)), int(e), int(a), None if v == -1 else int(v))
            for k, e, a, v in self.state
            if k != -1
        )

    def observe(self, operation: Operation) -> None:
        if self.capacity == 0 or operation.kind in (Kind.NOISE, Kind.ASK):
            return
        records = list(self.records())
        if self.name == "recency":
            records = [record for record in records if record.key != operation.key]
        records.append(operation)
        self.state.fill(-1)
        for index, record in enumerate(records[-self.capacity :]):
            self.state[index] = (
                record.kind,
                record.entity,
                record.attribute,
                -1 if record.value is None else record.value,
            )

    def retrieve(self, query: Operation) -> tuple[Operation, ...]:
        if query.kind != Kind.ASK:
            raise ValueError("retrieve expects ASK")
        records = self.records()
        if self.name != "similarity" or not records:
            return records
        query_key = f"{query.entity:02d}{query.attribute}"

        def score(item: tuple[int, Operation]) -> tuple[int, int]:
            index, record = item
            key = f"{record.entity:02d}{record.attribute}"
            return sum(a == b for a, b in zip(key, query_key, strict=True)), index

        return (max(enumerate(records), key=score)[1],)
