"""Case 5: causal admission into a canonical, bounded int32 fact bank.

Unlike Phase 1's event bank, records are [entity, attribute, value, observed cue].
Physical row order supplies FIFO/recency ordering without hidden timestamps.
The episode and its future queries are evaluator/trainer data, never actor input.
"""

import hashlib
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..memory_benchmark.generator import symbol_space
from ..memory_benchmark.operations import UNKNOWN, Kind, Operation
from ..memory_benchmark.views import read_visible

POLICIES = ("no_memory", "fifo", "recency", "random", "similarity", "cue_priority", "learned")


@dataclass(frozen=True)
class CapacityEpisode:
    episode_id: int
    workload: str
    sessions: tuple[tuple[tuple[Operation, int], ...], ...]
    queries: tuple[Operation, ...]

    def __post_init__(self) -> None:
        if self.workload not in ("A", "B"):
            raise ValueError("workload must be A or B")
        if not self.sessions or any(not session for session in self.sessions):
            raise ValueError("write sessions must be nonempty")
        if any(operation.kind == Kind.ASK for session in self.sessions
               for operation, _ in session):
            raise ValueError("write-then-query oracle forbids queries in write sessions")
        if any(cue not in (0, 1) for session in self.sessions for _, cue in session):
            raise ValueError("cue must be binary")
        if not self.queries or any(query.kind != Kind.ASK for query in self.queries):
            raise ValueError("probe phase requires nonempty ASK operations")


def generate_capacity(
    split: str, seed: int, count: int, *, workload: str = "A", keys: int = 24,
    updates: int = 6, sessions: int = 8, queries: int = 32,
) -> list[CapacityEpisode]:
    """A/B share the full write prefix; independent RNG streams sample probes."""
    symbols = symbol_space(split)
    if workload not in ("A", "B"):
        raise ValueError("workload must be A or B")
    if count <= 0 or keys < 2 or keys % 2 or keys > len(symbols.entities) * 4:
        raise ValueError("need positive count and an even feasible number of keys")
    if updates < 0 or not 1 <= sessions <= keys + updates or queries <= 0:
        raise ValueError("invalid updates, sessions, or queries")
    episodes = []
    for index in range(count):
        # String seeds are stable in Python's version-2 random seeding protocol.
        prefix = random.Random(f"capacity-prefix:{seed}:{index}")
        suffix = random.Random(f"capacity-probes:{seed}:{index}")
        selected = prefix.sample([(e, a) for e in symbols.entities for a in range(4)], keys)
        cues = [0] * (keys // 2) + [1] * (keys // 2)
        prefix.shuffle(cues)
        cue_by_key = dict(zip(selected, cues, strict=True))
        writes = []
        latest: dict[tuple[int, int], int] = {}
        # Interleave repeat writes after their initial observation; every repeat
        # differs from the previous value. Exactly `updates` repeats are generated.
        update_after = Counter(prefix.choices(range(keys), k=updates))
        for position, key in enumerate(selected):
            value = prefix.choice(symbols.values)
            writes.append((Operation(Kind.SET, *key, value), cue_by_key[key]))
            latest[key] = value
            for _ in range(update_after[position]):
                repeated = prefix.choice(selected[:position + 1])
                value = prefix.choice([v for v in symbols.values if v != latest[repeated]])
                writes.append((Operation(Kind.UPDATE, *repeated, value), cue_by_key[repeated]))
                latest[repeated] = value
        count_writes = len(writes)
        write_sessions = tuple(
            tuple(writes[i * count_writes // sessions:(i + 1) * count_writes // sessions])
            for i in range(sessions)
        )
        weights = [5 if workload == "B" and cue else 1 for cue in cues]
        probes = tuple(Operation(Kind.ASK, *key)
                       for key in suffix.choices(selected, weights=weights, k=queries))
        episodes.append(CapacityEpisode(index, workload, write_sessions, probes))
    return episodes


def _check_bank(bank: np.ndarray) -> None:
    if bank.dtype != np.int32 or bank.ndim != 2 or bank.shape[1] != 4:
        raise ValueError("bank must be K x 4 int32")


def _match(bank: np.ndarray, operation: Operation) -> np.ndarray:
    return np.flatnonzero((bank[:, 0] == operation.entity) &
                          (bank[:, 1] == operation.attribute))


def bank_records(bank: np.ndarray) -> tuple[Operation, ...]:
    """Project bounded facts without repairing an erroneous writer's output.

    A wrong STORE/UPDATE on an incoming DELETE can leave an occupied key with
    value -1. Preserve that key as DELETE evidence, including duplicate keys and
    order; this is an honest projection, not a correction of action or target.
    """
    _check_bank(bank)
    return tuple(
        Operation(Kind.DELETE, int(e), int(a)) if v == -1
        else Operation(Kind.SET, int(e), int(a), int(v))
        for e, a, v, _ in bank if e >= 0
    )


def read_bank(bank: np.ndarray, query: Operation) -> str:
    return read_visible(bank_records(bank), (), query)


class RetentionScorer(nn.Module):
    """Shared per-record scorer: observed binary cue only; no identities/history.

    Twenty-five parameters, independent of slot count. It can learn the planted
    signal but has no extra information over the cue-priority heuristic.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, cues: torch.Tensor) -> torch.Tensor:
        return self.network(cues.float().unsqueeze(-1)).squeeze(-1)


class CapacityPolicy:
    """Online policy. Only `bank` changes during evaluation; model/seed are fixed."""

    def __init__(self, name: str, capacity: int, *, model: RetentionScorer | None = None,
                 seed: int = 0) -> None:
        if name not in POLICIES or capacity < 1:
            raise ValueError("unknown policy or nonpositive capacity")
        if name == "learned" and model is None:
            raise ValueError("learned policy requires a scorer")
        self.name, self.model, self.seed = name, model, seed
        self.bank = np.full((0 if name == "no_memory" else capacity, 4), -1, dtype=np.int32)

    def select_allocation(self, bank: np.ndarray, operation: Operation, cue: int) -> int | None:
        """Select a free row/victim, or reject; sees only current bounded input."""
        _check_bank(bank)
        if cue not in (0, 1):
            raise ValueError("cue must be binary")
        if self.name == "no_memory":
            return None
        empty = np.flatnonzero(bank[:, 0] < 0)
        if len(empty):
            return int(empty[0])
        if self.name in ("fifo", "recency", "similarity"):
            return 0
        if self.name == "cue_priority":
            victim = int(np.argmin(bank[:, 3]))
            return victim if cue > int(bank[victim, 3]) else None
        if self.name == "random":
            # Stable random priorities on current facts and incoming key. This
            # is bottom-K hash sampling, without episodic RNG/counter state.
            def priority(entity: int, attribute: int) -> bytes:
                return hashlib.sha256(f"{self.seed}:{entity}:{attribute}".encode()).digest()
            priorities = [priority(int(e), int(a)) for e, a, _, _ in bank]
            victim = max(range(len(bank)), key=priorities.__getitem__)
            return victim if priority(*operation.key) < priorities[victim] else None
        assert self.model is not None
        device = next(self.model.parameters()).device
        with torch.no_grad():
            scores = self.model(torch.tensor([*bank[:, 3], cue], device=device))
        # Reject on an equal minimum, so unnecessary rewrites are avoided.
        if scores[-1] <= scores[:-1].min():
            return None
        return int(scores[:-1].argmin().item())

    def write_selected(self, bank: np.ndarray, operation: Operation, cue: int,
                       target: int) -> None:
        """Apply allocation to this explicit bank; FIFO age is encoded by order."""
        _check_bank(bank)
        if not 0 <= target < len(bank) or operation.value is None:
            raise ValueError("invalid target or write without value")
        row = np.array([operation.entity, operation.attribute, operation.value, cue],
                       dtype=np.int32)
        if self.name in ("fifo", "recency", "similarity"):
            remaining = np.delete(bank, target, axis=0)
            remaining = remaining[remaining[:, 0] >= 0]
            bank[:] = -1
            bank[:len(remaining)] = remaining
            bank[len(remaining)] = row
        else:
            bank[target] = row

    def observe(self, operation: Operation, cue: int = 0) -> None:
        if cue not in (0, 1):
            raise ValueError("cue must be binary")
        if operation.kind in (Kind.NOISE, Kind.ASK) or self.name == "no_memory":
            return
        matches = _match(self.bank, operation)
        if operation.kind == Kind.DELETE:
            if len(matches):
                self.bank[matches] = -1
                if self.name in ("fifo", "recency", "similarity"):
                    occupied = self.bank[self.bank[:, 0] >= 0].copy()
                    self.bank[:] = -1
                    self.bank[:len(occupied)] = occupied
            return
        if len(matches):
            target = int(matches[0])
            if self.name == "recency":
                self.write_selected(self.bank, operation, cue, target)
            else:
                self.bank[target] = [operation.entity, operation.attribute, operation.value, cue]
            return
        target = self.select_allocation(self.bank, operation, cue)
        if target is not None:
            self.write_selected(self.bank, operation, cue, target)

    def reset_session(self) -> None:
        """There is no context/KV cache: the bank is already the entire state."""

    def retrieve(self, query: Operation) -> tuple[Operation, ...]:
        """Retrieve bounded evidence; similarity exposes only its best record.

        Character overlap of the fixed-width entity/attribute key ranks rows,
        with newest FIFO position breaking ties. Near misses remain observable
        to a neural reader; the independent exact reader rejects wrong keys.
        """
        if query.kind != Kind.ASK:
            raise ValueError("retrieve expects ASK")
        records = bank_records(self.bank)
        if self.name != "similarity" or not records:
            return records
        query_key = f"{query.entity:02d}{query.attribute}"

        def score(item: tuple[int, Operation]) -> tuple[int, int]:
            index, record = item
            key = f"{record.entity:02d}{record.attribute}"
            return sum(a == b for a, b in zip(key, query_key, strict=True)), index

        return (max(enumerate(records), key=score)[1],)

    def answer(self, query: Operation) -> str:
        return read_visible(self.retrieve(query), (), query)


def make_capacity_policy(name: str, capacity: int, *, model: RetentionScorer | None = None,
                         seed: int = 0) -> CapacityPolicy:
    return CapacityPolicy(name, capacity, model=model, seed=seed)


def _final_world(episode: CapacityEpisode) -> dict[tuple[int, int], tuple[int, int]]:
    # This reference world exists only in the evaluator/reward calculation.
    world = {}
    for session in episode.sessions:
        for operation, cue in session:
            if operation.kind in (Kind.SET, Kind.UPDATE):
                world[operation.key] = (int(operation.value), cue)
            elif operation.kind == Kind.DELETE:
                world.pop(operation.key, None)
    return world


def score_bank(episode: CapacityEpisode, bank: np.ndarray) -> dict[str, Any]:
    world = _final_world(episode)
    records = []
    for index, query in enumerate(episode.queries):
        truth = f"{world[query.key][0]:02d}" if query.key in world else UNKNOWN
        answer = read_bank(bank, query)
        records.append({"query_index": index, "entity": query.entity,
                        "attribute": query.attribute, "truth": truth, "answer": answer,
                        "correct": answer == truth, "historical_answerable": truth != UNKNOWN})
    eligible = [row for row in records if row["historical_answerable"]]
    hits = sum(row["correct"] for row in eligible)
    return {"utility": hits / len(eligible) if eligible else None,
            "hits": hits, "queries": len(eligible), "records": records}


def oracle_retention(episode: CapacityEpisode, capacity: int) -> dict[str, Any]:
    """Evaluation-only exact top-K bound for writes followed by probes.

    Final keys are all observed during writes. A clairvoyant actor can preserve
    its selected keys on arrival and apply their later updates without reload.
    This oracle is deliberately invalid for interleaved write/query workloads.
    """
    if capacity < 1:
        raise ValueError("capacity must be positive")
    world = _final_world(episode)
    counts = Counter(query.key for query in episode.queries)
    selected = sorted(world, key=lambda key: (-counts[key], key))[:capacity]
    bank = np.full((capacity, 4), -1, dtype=np.int32)
    for index, key in enumerate(selected):
        bank[index] = [*key, *world[key]]
    return {"bank": bank, **score_bank(episode, bank)}


def _sync(model: RetentionScorer | None) -> None:
    if model is not None and next(model.parameters()).device.type == "mps":
        torch.mps.synchronize()
    elif model is not None and next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_capacity(episodes: list[CapacityEpisode], policy_name: str, capacity: int, *,
                      model: RetentionScorer | None = None, seed: int = 0) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes must not be empty")
    rows = []
    for episode in episodes:
        policy = make_capacity_policy(policy_name, capacity, model=model, seed=seed)
        _sync(model)
        start = time.perf_counter()
        for session in episode.sessions:
            for operation, cue in session:
                policy.observe(operation, cue)
            policy.reset_session()
        _sync(model)
        write_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = [policy.answer(query) for query in episode.queries]
        read_seconds = time.perf_counter() - start
        result = score_bank(episode, policy.bank)
        assert predictions == [record["answer"] for record in result["records"]]
        oracle = oracle_retention(episode, capacity)
        rows.append({"episode_id": episode.episode_id, "workload": episode.workload,
                     "final_bank": policy.bank.tolist(), **result,
                     "oracle_utility": oracle["utility"], "oracle_hits": oracle["hits"],
                     "oracle_regret": oracle["utility"] - result["utility"],
                     "write_seconds": write_seconds, "read_seconds": read_seconds})
    denominator = sum(row["queries"] for row in rows)
    utility = sum(row["hits"] for row in rows) / denominator
    oracle_utility = sum(row["oracle_hits"] for row in rows) / denominator
    summary = {"policy": policy_name, "capacity": capacity, "episodes": len(rows),
               "queries": denominator, "utility": utility, "oracle_utility": oracle_utility,
               "oracle_regret": oracle_utility - utility,
               "persistent_state_bytes": 0 if policy_name == "no_memory" else capacity * 16,
               "historical_tokens_reprocessed": 0,
               "write_seconds": sum(row["write_seconds"] for row in rows),
               "read_seconds": sum(row["read_seconds"] for row in rows)}
    return {"summary": summary, "episodes": rows}


@dataclass
class CapacityTraining:
    model: RetentionScorer
    curve: list[dict[str, Any]]
    metadata: dict[str, Any]


def train_capacity(config: dict[str, Any], seed: int, device: str = "cpu") -> CapacityTraining:
    """Episodic REINFORCE with delayed answer rewards, never oracle labels.

    Training-only RNG, action log probabilities, and running reward baseline are
    optimizer machinery; none survive into the evaluation actor's episode state.
    """
    episodes_count = int(config.get("episodes", 256))
    capacity = int(config.get("capacity", 4))
    if episodes_count <= 0 or episodes_count % 2 or capacity < 1:
        raise ValueError("training needs positive even episodes and positive capacity")
    torch.manual_seed(seed)
    model = RetentionScorer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", .03)))
    generator = torch.Generator(device="cpu").manual_seed(seed + 104729)
    baseline = 0.0
    curve = []
    for index in range(episodes_count):
        workload = "A" if index % 2 == 0 else "B"
        episode = generate_capacity("train", seed + index // 2, 1, workload=workload)[0]
        policy = make_capacity_policy("learned", capacity, model=model)
        log_probabilities = []
        for session in episode.sessions:
            for operation, cue in session:
                if len(_match(policy.bank, operation)) or np.any(policy.bank[:, 0] < 0):
                    policy.observe(operation, cue)
                    continue
                cues = torch.tensor([*policy.bank[:, 3], cue], device=device)
                logits = -model(cues)
                distribution = torch.distributions.Categorical(logits=logits)
                drop = int(torch.multinomial(distribution.probs.detach().cpu(), 1,
                                             generator=generator).item())
                log_probabilities.append(distribution.log_prob(torch.tensor(drop, device=device)))
                if drop < capacity:
                    policy.write_selected(policy.bank, operation, cue, drop)
            policy.reset_session()
        # Reward is computed only after the complete probe phase. No counts,
        # oracle eviction choices, or future identities become actor features.
        reward = float(score_bank(episode, policy.bank)["utility"])
        advantage = reward - baseline
        if log_probabilities:
            loss = -advantage * torch.stack(log_probabilities).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
        else:
            loss_value = 0.0
        curve.append({"episode": index, "workload": workload, "reward": reward,
                      "baseline": baseline, "loss": loss_value})
        baseline = .95 * baseline + .05 * reward
    model.eval()
    return CapacityTraining(model, curve, {
        "training_signal": "delayed_answer_reward_reinforce", "seed": seed,
        "episodes": episodes_count, "capacity": capacity,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "features": ["observed_binary_cue"], "baseline_decay": .95,
        "evaluation_decision": "greedy_minimum_score_rejection_on_ties",
        "cross_capacity_weights": "shared_per_record_scorer",
    })
