"""A transparent character-level tokenizer for the learning project."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class CharTokenizer:
    """Map characters to integer token IDs and back without hidden preprocessing."""

    def __init__(self, vocabulary: Iterable[str]) -> None:
        characters = tuple(dict.fromkeys(vocabulary))
        if not characters:
            raise ValueError("vocabulary cannot be empty")
        if any(len(character) != 1 for character in characters):
            raise ValueError("every vocabulary entry must be exactly one character")

        self._id_to_character = characters
        self._character_to_id = {
            character: index for index, character in enumerate(self._id_to_character)
        }

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_character)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return self._id_to_character

    def encode(self, text: str, *, device: str | torch.device | None = None) -> Tensor:
        try:
            token_ids = [self._character_to_id[character] for character in text]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is not in the vocabulary") from error
        return torch.tensor(token_ids, dtype=torch.long, device=device)

    def decode(self, token_ids: Tensor | Iterable[int]) -> str:
        if isinstance(token_ids, Tensor):
            ids = token_ids.detach().cpu().tolist()
        else:
            ids = list(token_ids)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("token ID is outside the vocabulary")
        return "".join(self._id_to_character[token_id] for token_id in ids)
