import pytest

from bounded_memory_transformer import TransformerConfig


def test_config_computes_head_dimension() -> None:
    config = TransformerConfig(vocab_size=32, d_model=48, n_heads=6)

    assert config.head_dim == 8


def test_config_rejects_incompatible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(vocab_size=32, d_model=50, n_heads=6)
