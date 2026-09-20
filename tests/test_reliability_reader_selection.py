from dataclasses import replace

import pytest
import torch

from bounded_memory_transformer.memory_experiments.reader_cases import generate_reader_tasks
from bounded_memory_transformer.memory_experiments.selection import RecordSelector, encode_selection


def test_scores_padding_invariance_forged_retained_and_no_persistent_state():
    torch.manual_seed(3)
    model = RecordSelector(d_model=16, n_heads=2, n_layers=1).eval()
    view = generate_reader_tasks("train", seed=3, count=5)[0].view
    small = replace(view, memory=view.memory[:1], current=())
    batch = encode_selection([small, view])
    scores = model(batch)
    assert scores.shape == (2, len(view.memory) + len(view.current) + 1)
    assert torch.isneginf(scores[0, 1:-1]).all()
    alone = model(encode_selection([small]))
    assert torch.allclose(scores[0, [0, -1]], alone[0], atol=1e-6)
    forged = replace(small, retained=view.memory)
    assert torch.equal(model(encode_selection([forged])), alone)
    model(encode_selection([view]))
    assert torch.equal(model(encode_selection([small])), alone)
    assert model.select([small, view])[0] in (-1, 0)
    assert model.select([]) == []


def test_padding_values_cannot_influence_any_candidate_scores():
    model = RecordSelector(d_model=16, n_heads=2, n_layers=1).eval()
    view = generate_reader_tasks("train", seed=4, count=1)[0].view
    batch = encode_selection([replace(view, memory=(), current=()), view])
    before = model(batch)
    batch.tokens[0, :-1] = 1
    assert torch.equal(before, model(batch))
    assert model.select([replace(view, memory=(), current=())]) == [-1]


def test_key_coordinates_are_input_and_loss_reaches_attention():
    model = RecordSelector(d_model=16, n_heads=2, n_layers=1)
    tasks = generate_reader_tasks("train", seed=5, count=10)
    batch = encode_selection([t.view for t in tasks])
    scores = model(batch)
    targets = torch.tensor(
        [t.oracle_index if t.oracle_index >= 0 else scores.shape[1] - 1 for t in tasks]
    )
    loss = torch.nn.functional.cross_entropy(scores, targets)
    loss.backward()
    assert model.blocks[0].attention.qkv_projection.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        encode_selection([])


def test_eight_slot_variable_candidates_preserve_padding_mask_and_scores():
    model = RecordSelector(d_model=16, n_heads=2, n_layers=1).eval()
    small = generate_reader_tasks("validation", seed=4, count=1, capacity=4)[0].view
    large = generate_reader_tasks("validation", seed=4, count=1, capacity=8)[0].view
    batch = encode_selection([small, large])
    scores = model(batch)
    count = len(small.memory) + len(small.current)
    assert scores.shape == (2, len(large.memory) + len(large.current) + 1)
    assert torch.isneginf(scores[0, count:-1]).all()
    alone = model(encode_selection([small]))
    assert torch.allclose(scores[0, :count], alone[0, :-1], atol=1e-6)
    assert torch.allclose(scores[0, -1], alone[0, -1], atol=1e-6)
