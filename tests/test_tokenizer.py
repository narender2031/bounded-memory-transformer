import pytest
import torch

from bounded_memory_transformer import CharTokenizer


def test_character_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("memory matters")
    encoded = tokenizer.encode("memory")

    assert encoded.dtype == torch.long
    assert tokenizer.decode(encoded) == "memory"


def test_character_tokenizer_rejects_unknown_character() -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="not in the vocabulary"):
        tokenizer.encode("abd")


@pytest.mark.parametrize("token_id", [-1, 3])
def test_character_tokenizer_rejects_out_of_range_token_id(token_id: int) -> None:
    tokenizer = CharTokenizer.from_text("abc")

    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.decode([token_id])
