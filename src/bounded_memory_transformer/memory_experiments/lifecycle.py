"""Causal, explicitly feature-assisted lifecycle learning (Cases 3–4).

The model receives operation, occupancy and symbolic key-equality features. This
is an engineering control, not a representation-learning or novelty claim. Its
weights are shared; only the K x 4 int32 fact bank is episodic persistent state.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..memory_benchmark.generator import symbol_space
from ..memory_benchmark.operations import UNKNOWN, Kind, Operation


class Action(IntEnum):
    STORE = 0
    UPDATE = 1
    DELETE = 2
    IGNORE = 3


@dataclass(frozen=True)
class Decision:
    action: Action
    target: int


def empty_bank(slots: int = 4) -> np.ndarray:
    if slots < 1:
        raise ValueError("slots must be positive")
    return np.full((slots, 4), -1, dtype=np.int32)


def _validate_bank(bank: np.ndarray) -> None:
    if bank.dtype != np.int32 or bank.ndim != 2 or bank.shape[1] != 4 or not len(bank):
        raise ValueError("bank must be a nonempty K x 4 int32 array")


def exact_decision(bank: np.ndarray, operation: Operation) -> Decision:
    """Causal supervised label, derived solely from current bank and input."""
    _validate_bank(bank)
    matching = np.flatnonzero(
        (bank[:, 0] == operation.entity) & (bank[:, 1] == operation.attribute)
    )
    if operation.kind in (Kind.NOISE, Kind.ASK):
        return Decision(Action.IGNORE, -1)
    if operation.kind == Kind.DELETE:
        return (
            Decision(Action.DELETE, int(matching[0]))
            if len(matching)
            else Decision(Action.IGNORE, -1)
        )
    if len(matching):
        return Decision(Action.UPDATE, int(matching[0]))
    empty = np.flatnonzero(bank[:, 0] == -1)
    return Decision(Action.STORE, int(empty[0]) if len(empty) else len(bank))


def apply_decision(
    bank: np.ndarray,
    operation: Operation,
    decision: Decision,
    cue: int = 0,
    allocation: int | None = None,
) -> np.ndarray:
    """Execute exactly the predicted action/slot, returning a fresh bank.

    Target K explicitly requests capacity allocation; allocation=None rejects it.
    Invalid predicted routes (-1 or K for UPDATE/DELETE) are no-ops, never rerouted.
    A wrong write action on a valueless operation writes value -1, read as UNKNOWN.
    No key matching or input-kind correction occurs here.
    """
    _validate_bank(bank)
    result = bank.copy()
    action, target = Action(decision.action), decision.target
    if allocation is not None and not 0 <= allocation < len(bank):
        raise ValueError("allocation must name a bank slot")
    if not -1 <= target <= len(bank):
        raise ValueError("predicted target must be -1, a slot, or allocation sentinel K")
    if action == Action.IGNORE:
        return result
    if action == Action.STORE and target == len(bank):
        if allocation is None:
            return result
        target = allocation
    if not 0 <= target < len(bank):
        return result
    if action == Action.DELETE:
        result[target] = -1
    else:
        value = operation.value if operation.value is not None else -1
        result[target] = [operation.entity, operation.attribute, value, cue]
    return result


def read_bank(bank: np.ndarray, query: Operation) -> str:
    """Exact key reader. Duplicates use last physical slot, without state repair."""
    _validate_bank(bank)
    matching = np.flatnonzero((bank[:, 0] == query.entity) & (bank[:, 1] == query.attribute))
    if not len(matching) or bank[matching[-1], 2] < 0:
        return UNKNOWN
    return f"{bank[matching[-1], 2]:02d}"


def _features(bank: np.ndarray, operation: Operation) -> tuple[np.ndarray, np.ndarray]:
    _validate_bank(bank)
    occupied = bank[:, 0] != -1
    entity = (bank[:, 0] == operation.entity) & occupied
    attribute = (bank[:, 1] == operation.attribute) & occupied
    match = entity & attribute
    op = np.eye(5, dtype=np.float32)[int(operation.kind) - 1]
    global_features = np.concatenate((op, [match.any(), (~occupied).any()])).astype(np.float32)
    slots = np.zeros((len(bank) + 1, 6 + len(global_features)), dtype=np.float32)
    slots[:-1, :4] = np.stack((occupied, entity, attribute, match), axis=1)
    slots[:-1, 4] = np.arange(len(bank)) / len(bank)
    slots[-1, 5] = 1  # explicit full-bank allocation candidate
    slots[:, 6:] = global_features
    return global_features, slots


class LifecycleModel(nn.Module):
    """Small feedforward action head and shared per-slot target scorer."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.action_head = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 4))
        self.target_head = nn.Sequential(nn.Linear(13, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(
        self, global_features: torch.Tensor, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert global_features.ndim == 2 and global_features.shape[-1] == 7
        assert slots.ndim == 3 and slots.shape[0] == global_features.shape[0]
        assert slots.shape[-1] == 13
        return self.action_head(global_features), self.target_head(slots).squeeze(-1)

    @torch.no_grad()
    def decide(self, bank: np.ndarray, operation: Operation) -> Decision:
        global_features, slots = _features(bank, operation)
        device = next(self.parameters()).device
        actions, targets = self(
            torch.tensor(global_features, device=device)[None],
            torch.tensor(slots, device=device)[None],
        )
        action = Action(int(actions.argmax(-1).item()))
        target = -1 if action == Action.IGNORE else int(targets.argmax(-1).item())
        return Decision(action, target)


@dataclass(frozen=True)
class LifecycleConfig:
    steps: int = 300
    batch_size: int = 128
    learning_rate: float = 0.01
    hidden: int = 32
    slots: int = 4
    training_examples: int = 4096

    def __post_init__(self) -> None:
        if min(self.steps, self.batch_size, self.hidden, self.slots, self.training_examples) < 1:
            raise ValueError("training sizes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")


@dataclass
class LifecycleBundle:
    model: LifecycleModel
    learning_curve: list[dict]
    metadata: dict

    def decide(self, bank: np.ndarray, operation: Operation) -> Decision:
        return self.model.decide(bank, operation)


def _training_examples(config: LifecycleConfig, seed: int):
    """Independent randomized current states: no episode suffix or query access."""
    rng = random.Random(seed)
    symbols = symbol_space("train")
    all_keys = [(entity, attr) for entity in symbols.entities for attr in range(4)]
    features, slot_features, actions, targets = [], [], [], []
    for index in range(config.training_examples):
        bank = empty_bank(config.slots)
        occupied = rng.randrange(config.slots + 1)
        chosen_slots = rng.sample(range(config.slots), occupied)
        keys = rng.sample(all_keys, occupied)
        for slot, key in zip(chosen_slots, keys, strict=True):
            bank[slot] = [*key, rng.choice(symbols.values), rng.randrange(2)]
        kind = (Kind.SET, Kind.UPDATE, Kind.DELETE, Kind.NOISE, Kind.ASK)[index % 5]
        if keys and rng.random() < 0.65:
            key = rng.choice(keys)
        else:
            key = rng.choice([key for key in all_keys if key not in keys])
        value = rng.choice(symbols.values) if kind in (Kind.SET, Kind.UPDATE, Kind.NOISE) else None
        operation = Operation(kind, *key, value)
        decision = exact_decision(bank, operation)
        global_input, slot_input = _features(bank, operation)
        features.append(global_input)
        slot_features.append(slot_input)
        actions.append(int(decision.action))
        targets.append(decision.target)
    return np.stack(features), np.stack(slot_features), np.array(actions), np.array(targets)


def train_lifecycle(
    config: LifecycleConfig | dict, seed: int, device: str | torch.device = "cpu"
) -> LifecycleBundle:
    if isinstance(config, dict):
        config = LifecycleConfig(**config)
    torch.manual_seed(seed)
    model = LifecycleModel(config.hidden).to(device)
    tensors = [torch.as_tensor(array, device=device) for array in _training_examples(config, seed)]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(seed + 1009)
    curve = []
    for step in range(config.steps):
        indices = torch.tensor(
            rng.integers(config.training_examples, size=config.batch_size), device=device
        )
        features, slots, gold_actions, gold_targets = [array[indices] for array in tensors]
        actions, targets = model(features, slots)
        action_loss = F.cross_entropy(actions, gold_actions)
        applicable = gold_targets >= 0
        target_loss = (
            F.cross_entropy(targets[applicable], gold_targets[applicable])
            if (applicable.any())
            else targets.sum() * 0
        )
        loss = action_loss + target_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == config.steps:
            curve.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu()),
                    "action_loss": float(action_loss.detach().cpu()),
                    "target_loss": float(target_loss.detach().cpu()),
                }
            )
    model.eval()
    return LifecycleBundle(
        model,
        curve,
        {
            "symbol_split": "train",
            "seed": seed,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_examples": config.training_examples,
            "supervision": "current-bank/input causal action and target; no future information",
            "features": [
                "operation one-hot",
                "occupancy",
                "entity equality",
                "attribute equality",
                "full-key equality",
                "slot position",
                "allocation flag",
                "any match/empty",
            ],
            "episodic_state": "only K x 4 int32 bank; no controller cache",
        },
    )


@dataclass(frozen=True)
class LifecycleEpisode:
    episode_id: int
    case: str
    sessions: tuple[tuple[Operation, ...], ...]
    answers: tuple[str, str]
    stale_value: str | None
    deleted_value: str | None
    control_key: tuple[int, int]


def generate_lifecycle(split: str, seed: int, count: int) -> list[LifecycleEpisode]:
    """Controlled reset episodes with both one-coordinate distractor families."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    symbols = symbol_space(split)
    episodes = []
    for index in range(count):
        entity, other, missing = rng.sample(symbols.entities, 3)
        attribute = rng.randrange(4)
        other_attribute = (attribute + rng.randrange(1, 4)) % 4
        a = entity, attribute
        b = (entity, other_attribute) if (index // 2) % 2 == 0 else (other, attribute)
        c = (other, attribute) if b[0] == entity else (entity, other_attribute)
        old, new, keep, distractor, noise = rng.sample(symbols.values, 5)
        initial = [
            Operation(Kind.SET, *a, old),
            Operation(Kind.SET, *b, keep),
            Operation(Kind.SET, *c, distractor),
        ]
        rng.shuffle(initial)
        case = "update" if index % 2 == 0 else "delete"
        change = (
            Operation(Kind.UPDATE if index % 4 == 0 else Kind.SET, *a, new)
            if (case == "update")
            else Operation(Kind.DELETE, *a)
        )
        middle = (
            Operation(Kind.NOISE, *a, noise),
            change,
            Operation(Kind.NOISE, *b, noise),
            Operation(Kind.DELETE, missing, attribute),
        )
        queries = (Operation(Kind.ASK, *a), Operation(Kind.ASK, *b))
        answers = (f"{new:02d}" if case == "update" else UNKNOWN, f"{keep:02d}")
        episodes.append(
            LifecycleEpisode(
                index,
                case,
                (tuple(initial), middle, queries),
                answers,
                f"{old:02d}" if case == "update" else None,
                f"{old:02d}" if case == "delete" else None,
                b,
            )
        )
    return episodes


def _macro_f1(rows: list[dict]) -> float:
    scores = []
    for action in Action:
        tp = sum(r["gold_action"] == r["action"] == action.name for r in rows)
        fp = sum(r["gold_action"] != action.name and r["action"] == action.name for r in rows)
        fn = sum(r["gold_action"] == action.name and r["action"] != action.name for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores)


def evaluate_lifecycle(
    controller: LifecycleBundle | Callable, episodes: list[LifecycleEpisode], slots: int = 4
) -> dict:
    """Operation labels use the encountered bank; query truth remains independent.

    All records are evaluator output, never passed back to the controller. A reset
    creates a fresh array containing only bank bytes; no session text is retained.
    """
    if not episodes:
        raise ValueError("episodes must be nonempty")
    decide = controller.decide if hasattr(controller, "decide") else controller
    operations, queries, banks = [], [], []
    for episode in episodes:
        bank = empty_bank(slots)
        query_index = 0
        for session_index, session in enumerate(episode.sessions):
            bank = np.frombuffer(bank.tobytes(), dtype=np.int32).copy().reshape(slots, 4)
            for operation_index, operation in enumerate(session):
                common = {
                    "episode_id": episode.episode_id,
                    "case": episode.case,
                    "session": session_index,
                    "operation_index": operation_index,
                }
                if operation.kind == Kind.ASK:
                    answer = read_bank(bank, operation)
                    expected = episode.answers[query_index]
                    control = query_index == 1
                    queries.append(
                        {
                            **common,
                            "entity": operation.entity,
                            "attribute": operation.attribute,
                            "answer": answer,
                            "expected": expected,
                            "correct": answer == expected,
                            "control": control,
                            "stale": not control and answer == episode.stale_value,
                            "deletion_nonabstention": not control
                            and episode.case == "delete"
                            and answer != UNKNOWN,
                            "deleted_value_repetition": not control
                            and answer == episode.deleted_value,
                            "bank": bank.tolist(),
                        }
                    )
                    query_index += 1
                    continue
                gold = exact_decision(bank, operation)
                decision = decide(bank, operation)
                reference = apply_decision(bank, operation, gold)
                before = bank.tolist()
                bank = apply_decision(bank, operation, decision)
                applicable = gold.action != Action.IGNORE
                operations.append(
                    {
                        **common,
                        "input_kind": operation.kind.name,
                        "entity": operation.entity,
                        "attribute": operation.attribute,
                        "value": operation.value,
                        "action": decision.action.name,
                        "target": decision.target,
                        "gold_action": gold.action.name,
                        "gold_target": gold.target,
                        "action_correct": decision.action == gold.action,
                        "target_applicable": applicable,
                        "target_correct": decision.target == gold.target if applicable else None,
                        "transition_correct": bool(np.array_equal(bank, reference)),
                        "bank_before": before,
                        "bank_after": bank.tolist(),
                    }
                )
            banks.append(
                {
                    "episode_id": episode.episode_id,
                    "case": episode.case,
                    "session": session_index,
                    "bank": bank.tolist(),
                    "state_bytes": bank.nbytes,
                }
            )
    targets = [r for r in operations if r["target_applicable"]]
    controls = [r for r in queries if r["control"]]
    updates = [r for r in queries if not r["control"] and r["case"] == "update"]
    deletions = [r for r in queries if not r["control"] and r["case"] == "delete"]

    def rate(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else None

    summary = {
        "semantic_accuracy": rate(queries, "correct"),
        "action_accuracy": rate(operations, "action_correct"),
        "action_macro_f1": _macro_f1(operations),
        "target_accuracy": rate(targets, "target_correct"),
        "transition_accuracy": rate(operations, "transition_correct"),
        "control_preservation": rate(controls, "correct"),
        "update_accuracy": rate(updates, "correct"),
        "delete_accuracy": rate(deletions, "correct"),
        "stale_answer_rate": rate(updates, "stale"),
        "deletion_nonabstention_rate": rate(deletions, "deletion_nonabstention"),
        "deleted_value_repetition_rate": rate(deletions, "deleted_value_repetition"),
        "operation_count": len(operations),
        "applicable_target_count": len(targets),
        "query_count": len(queries),
        "control_count": len(controls),
        "update_query_count": len(updates),
        "delete_query_count": len(deletions),
        "state_bytes": slots * 16,
        "historical_tokens_replayed": 0,
    }
    summary["competence_gate"] = summary["semantic_accuracy"] >= 0.95
    return {"summary": summary, "operations": operations, "queries": queries, "banks": banks}
