"""Next-token training batches for a one-dimensional token stream."""

from __future__ import annotations

import torch
from torch import Tensor


class SequenceBatcher:
    """Sample deterministic contiguous sequences and their one-token-shifted targets."""

    def __init__(
        self,
        token_ids: Tensor,
        *,
        context_length: int,
        batch_size: int,
        seed: int,
        device: str | torch.device,
    ) -> None:
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must use torch.long")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if token_ids.numel() <= context_length:
            raise ValueError("token stream must be longer than context_length")

        self.token_ids = token_ids.cpu()
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def next_batch(self) -> tuple[Tensor, Tensor]:
        maximum_start = self.token_ids.numel() - self.context_length
        starts = torch.randint(
            maximum_start,
            (self.batch_size,),
            generator=self.generator,
        )
        inputs = torch.stack(
            [self.token_ids[start : start + self.context_length] for start in starts.tolist()]
        )
        targets = torch.stack(
            [
                self.token_ids[start + 1 : start + self.context_length + 1]
                for start in starts.tolist()
            ]
        )
        return inputs.to(self.device), targets.to(self.device)
