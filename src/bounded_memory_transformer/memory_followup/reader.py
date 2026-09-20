"""Factorized neural pointer: learned symbol matching, authority and chronology.

The three key coordinates are supplied separately. A shared manual Transformer
compares character pairs. A minimum combines the three match logits and a learned
operation-authority logit (an explicit conjunction inductive bias). Bounded
chronology breaks relevance ties; its sign/weight is learned. UNKNOWN has score
zero. Values are excluded from selection and copied without key repair.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bounded_memory_transformer.memory_benchmark.operations import Kind
from bounded_memory_transformer.memory_benchmark.views import QueryView
from bounded_memory_transformer.memory_experiments.reader_cases import oracle_select
from bounded_memory_transformer.tiny_transformer.config import TransformerConfig
from bounded_memory_transformer.tiny_transformer.model import DecoderBlock


@dataclass
class EncodedViews:
    pairs: Tensor  # B x candidates x 3 key coordinates x [query, record, CLS]
    kinds: Tensor
    valid: Tensor
    order: Tensor


def encode_views(views: list[QueryView], device: str = "cpu") -> EncodedViews:
    if not views:
        raise ValueError("cannot encode an empty batch")
    width = max(1, max(len(v.memory) + len(v.current) for v in views))
    pairs = torch.full((len(views), width, 3, 3), 14, dtype=torch.long)
    kinds = torch.zeros((len(views), width), dtype=torch.long)
    valid = torch.zeros((len(views), width), dtype=torch.bool)
    order = torch.zeros((len(views), width))
    for row, v in enumerate(views):
        q = (v.query.entity // 10, v.query.entity % 10, v.query.attribute + 10)
        records = (*v.memory, *v.current)
        for i, op in enumerate(records):
            if op.kind == Kind.ASK:
                raise ValueError("candidate evidence cannot contain ASK")
            r = (op.entity // 10, op.entity % 10, op.attribute + 10)
            pairs[row, i] = torch.tensor([[a, b, 14] for a, b in zip(q, r, strict=True)])
            kinds[row, i] = int(op.kind)
            valid[row, i] = True
            order[row, i] = i / max(1, len(records) - 1)
    return EncodedViews(*(x.to(device) for x in (pairs, kinds, valid, order)))


class FactorizedReader(nn.Module):
    def __init__(self, d_model: int = 24, n_heads: int = 2):
        super().__init__()
        self.settings = dict(d_model=d_model, n_heads=n_heads)
        config = TransformerConfig(15, 3, d_model, n_heads, 1, dropout=0.0)
        self.embedding = nn.Embedding(15, d_model)
        self.position = nn.Embedding(3, d_model)
        self.block = DecoderBlock(config)
        self.norm = nn.LayerNorm(d_model)
        self.match = nn.Linear(d_model, 1)
        self.authority = nn.Embedding(5, 1)
        self.order_weight = nn.Parameter(torch.tensor(0.0))

    def pair_logits(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 3:
            raise ValueError("pair tokens must have shape N x 3")
        hidden = self.embedding(tokens) + self.position(torch.arange(3, device=tokens.device))
        hidden, _ = self.block(hidden)
        return self.match(self.norm(hidden[:, -1])).flatten()

    def forward(self, batch: EncodedViews) -> tuple[Tensor, Tensor, Tensor]:
        if batch.pairs.ndim != 4 or batch.pairs.shape[-2:] != (3, 3):
            raise ValueError("expected B x candidates x 3 x 3 pairs")
        rows, candidates = batch.pairs.shape[:2]
        if any(x.shape != (rows, candidates) for x in (batch.kinds, batch.valid, batch.order)):
            raise ValueError("candidate metadata shape mismatch")
        tokens = batch.pairs.reshape(-1, 3)
        matches = self.pair_logits(tokens).reshape(rows, candidates, 3)
        authority = self.authority(batch.kinds).squeeze(-1)
        eligible = torch.cat((matches, authority.unsqueeze(-1)), dim=-1).amin(dim=-1)
        scores = 12.0 * (2.0 * eligible.sigmoid() - 1.0)
        scores = scores + 2.0 * self.order_weight.tanh() * batch.order
        scores = scores.masked_fill(~batch.valid, float("-inf"))
        unknown = torch.zeros((rows, 1), device=scores.device)
        return torch.cat((scores, unknown), dim=-1), matches, authority

    @torch.no_grad()
    def select(self, views: list[QueryView], batch_size: int = 128) -> list[int]:
        if batch_size < 1:
            raise ValueError("positive batch size required")
        training = self.training
        self.eval()
        selected = []
        device = str(next(self.parameters()).device)
        try:
            for start in range(0, len(views), batch_size):
                scores, _, _ = self(encode_views(views[start : start + batch_size], device))
                selected.extend(
                    -1 if i == scores.shape[1] - 1 else i for i in scores.argmax(-1).tolist()
                )
        finally:
            self.train(training)
        return selected


def supervised_loss(model: FactorizedReader, views: list[QueryView]) -> tuple[Tensor, dict]:
    """Auxiliary labels are derived only during training, never supplied to forward."""
    device = str(next(model.parameters()).device)
    batch = encode_views(views, device)
    scores, matches, authority = model(batch)
    targets = torch.tensor([oracle_select(v) for v in views], device=device)
    targets = torch.where(targets < 0, scores.shape[1] - 1, targets)
    selection = F.cross_entropy(scores, targets)
    valid = batch.valid
    if valid.any():
        equal = (batch.pairs[..., 0] == batch.pairs[..., 1]).float()
        matching = F.binary_cross_entropy_with_logits(matches[valid], equal[valid])
        authoritative = ((batch.kinds >= 1) & (batch.kinds <= 3)).float()
        operation = F.binary_cross_entropy_with_logits(authority[valid], authoritative[valid])
    else:
        matching, operation = selection * 0, selection * 0
    loss = selection + matching + operation
    return loss, dict(
        selection=float(selection.detach()),
        matching=float(matching.detach()),
        authority=float(operation.detach()),
    )
