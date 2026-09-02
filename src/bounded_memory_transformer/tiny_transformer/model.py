"""An inspectable decoder-only Transformer language model.

This module deliberately does not use ``nn.Transformer`` or
``nn.MultiheadAttention``. The important tensor operations are visible so the model
can later be modified with persistent memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bounded_memory_transformer.tiny_transformer.config import TransformerConfig


@dataclass
class ModelOutput:
    """Outputs returned by :class:`TinyTransformerLM`."""

    logits: Tensor
    loss: Tensor | None = None
    attentions: tuple[Tensor, ...] | None = None


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask.

    Input and output have shape ``[batch, time, d_model]``. When requested,
    attention probabilities have shape ``[batch, heads, time, time]``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.qkv_projection = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "causal_mask",
            causal_mask.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        batch_size, sequence_length, width = inputs.shape
        if width != self.d_model:
            raise ValueError(f"expected input width {self.d_model}, received {width}")
        if sequence_length > self.causal_mask.size(-1):
            raise ValueError("sequence is longer than the configured context length")

        qkv = self.qkv_projection(inputs)
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.n_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        scores = queries @ keys.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        probabilities = F.softmax(scores, dim=-1)
        attended_values = self.attention_dropout(probabilities) @ values
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, sequence_length, self.d_model)
        output = self.residual_dropout(self.output_projection(attended_values))

        return output, probabilities if return_attention else None


class FeedForward(nn.Module):
    """The position-wise MLP inside a Transformer block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_width = 4 * config.d_model
        self.network = nn.Sequential(
            nn.Linear(config.d_model, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class DecoderBlock(nn.Module):
    """A pre-normalization Transformer decoder block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attention_output, probabilities = self.attention(
            self.attention_norm(inputs),
            return_attention=return_attention,
        )
        hidden = inputs + attention_output
        hidden = hidden + self.feed_forward(self.feed_forward_norm(hidden))
        return hidden, probabilities


class TinyTransformerLM(nn.Module):
    """A GPT-style decoder-only language model for learning and experiments."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        *,
        return_attentions: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")

        batch_size, sequence_length = input_ids.shape
        if sequence_length == 0:
            raise ValueError("input sequence cannot be empty")
        if sequence_length > self.config.context_length:
            raise ValueError("input sequence is longer than the configured context length")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)

        collected_attentions: list[Tensor] = []
        for block in self.blocks:
            hidden, probabilities = block(hidden, return_attention=return_attentions)
            if probabilities is not None:
                collected_attentions.append(probabilities)

        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.config.vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return ModelOutput(
            logits=logits,
            loss=loss,
            attentions=tuple(collected_attentions) if return_attentions else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Autoregressively sample tokens while respecting the context limit."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.size(1) == 0:
            raise ValueError("input sequence cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")

        was_training = self.training
        self.eval()
        generated = input_ids
        try:
            for _ in range(max_new_tokens):
                current_context = generated[:, -self.config.context_length :]
                next_logits = self(current_context).logits[:, -1, :] / temperature

                if top_k is not None:
                    kept = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, kept).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)

        return generated

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())
