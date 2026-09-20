"""Shared-weight, query/record Transformer scorer; no exact-key input features."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from bounded_memory_transformer.memory_benchmark.reader import (
    ALPHABET,
    TOKEN_IDS,
    render_operations,
)
from bounded_memory_transformer.memory_benchmark.views import QueryView
from bounded_memory_transformer.tiny_transformer.config import TransformerConfig
from bounded_memory_transformer.tiny_transformer.model import DecoderBlock

# Each real candidate is Qddx + origin + operation (7 characters) + '='.
PAIR_LENGTH = 13


@dataclass
class SelectionBatch:
    tokens: Tensor
    valid: Tensor
    positions: Tensor


def encode_selection(views: list[QueryView], device: str = "cpu") -> SelectionBatch:
    if not views:
        raise ValueError("cannot encode an empty batch")
    candidates = max(len(v.memory) + len(v.current) for v in views)
    tokens = torch.full((len(views), candidates + 1, PAIR_LENGTH), TOKEN_IDS["#"], dtype=torch.long)
    valid = torch.zeros((len(views), candidates + 1), dtype=torch.bool)
    positions = torch.zeros((len(views), candidates + 1, 1))
    for row, view in enumerate(views):
        query = f"Q{view.query.entity:02d}{'abcd'[view.query.attribute]}"
        for index, operation in enumerate((*view.memory, *view.current)):
            origin = "M" if index < len(view.memory) else "C"
            text = query + origin + render_operations((operation,)) + "="
            assert len(text) == PAIR_LENGTH
            tokens[row, index] = torch.tensor([TOKEN_IDS[c] for c in text])
            valid[row, index] = True
            # Absolute event order is visible information, not a key match feature.
            positions[row, index, 0] = index / 8.0
        unknown = query + "?" * (PAIR_LENGTH - len(query) - 1) + "="
        tokens[row, -1] = torch.tensor([TOKEN_IDS[c] for c in unknown])
        valid[row, -1] = True
    return SelectionBatch(tokens.to(device), valid.to(device), positions.to(device))


class RecordSelector(nn.Module):
    """Score joint query/candidate strings with the existing manual decoder blocks.

    UNKNOWN is the final logit. Padded records are masked and cannot affect any
    score. There are no recurrent activations, cached KV tensors or key-equality
    features. A digit vocabulary is shared across disjoint full-symbol splits.
    """

    def __init__(self, d_model: int = 48, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.settings = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        config = TransformerConfig(
            len(ALPHABET), PAIR_LENGTH, d_model, n_heads, n_layers, dropout=0.0
        )
        self.embedding = nn.Embedding(len(ALPHABET), d_model)
        self.position_embedding = nn.Embedding(PAIR_LENGTH, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.score = nn.Linear(d_model, 1)
        self.event_order = nn.Linear(1, 1, bias=False)

    def forward(self, batch: SelectionBatch) -> Tensor:
        if batch.tokens.ndim != 3 or batch.tokens.shape[-1] != PAIR_LENGTH:
            raise ValueError("tokens must be B x candidates x pair-length")
        rows, candidates, length = batch.tokens.shape
        if batch.valid.shape != (rows, candidates):
            raise ValueError("candidate mask has wrong shape")
        if batch.positions.shape != (rows, candidates, 1):
            raise ValueError("event order has wrong shape")
        tokens = batch.tokens.reshape(rows * candidates, length)
        hidden = self.embedding(tokens) + self.position_embedding(
            torch.arange(length, device=tokens.device)
        )
        for block in self.blocks:
            hidden, _ = block(hidden)
        scores = self.score(self.norm(hidden[:, -1])).reshape(rows, candidates)
        scores = scores + self.event_order(batch.positions).squeeze(-1)
        return scores.masked_fill(~batch.valid, float("-inf"))

    @torch.no_grad()
    def select(self, views: list[QueryView], batch_size: int = 128) -> list[int]:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        device = str(next(self.parameters()).device)
        was_training = self.training
        self.eval()
        selected = []
        try:
            for start in range(0, len(views), batch_size):
                batch = encode_selection(views[start : start + batch_size], device)
                indices = self(batch).argmax(-1).tolist()
                selected.extend(-1 if i == batch.valid.shape[1] - 1 else i for i in indices)
        finally:
            self.train(was_training)
        return selected
