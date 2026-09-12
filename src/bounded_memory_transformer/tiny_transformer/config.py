"""Configuration for the tiny Transformer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters that define a decoder-only Transformer.

    The defaults intentionally produce a small model that is easy to run and inspect.
    """

    vocab_size: int
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        """Width of each attention head."""

        return self.d_model // self.n_heads
