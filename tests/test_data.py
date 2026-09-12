import torch

from bounded_memory_transformer.tiny_transformer.data import SequenceBatcher


def test_sequence_batch_targets_are_shifted_by_one_token() -> None:
    tokens = torch.arange(20, dtype=torch.long)
    batcher = SequenceBatcher(
        tokens,
        context_length=5,
        batch_size=4,
        seed=7,
        device="cpu",
    )

    inputs, targets = batcher.next_batch()

    assert inputs.shape == (4, 5)
    assert targets.shape == (4, 5)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
