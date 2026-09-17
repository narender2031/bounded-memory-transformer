import json

import torch
import torch.nn.functional as F

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_benchmark.reader import (
    ALPHABET,
    build_batch,
    make_training_views,
    predict,
    render,
)
from bounded_memory_transformer.memory_benchmark.views import QueryView, read_visible
from bounded_memory_transformer.tiny_transformer import TinyTransformerLM, TransformerConfig


def test_prompt_contains_only_bounded_records_and_current_session():
    view = QueryView(
        (Operation(Kind.SET, 12, 0, 23),),
        (Operation(Kind.UPDATE, 12, 0, 45),),
        Operation(Kind.ASK, 12, 0),
        64,
        (),
    )
    assert render(view) == "MS12a23;CU12a45;Q12a="


def test_answer_loss_alignment_and_padding_have_exactly_two_supervised_tokens():
    inputs, targets = build_batch([("MCQ12a=", "23"), ("MS45b67;CQ45b=", "??")], "cpu")
    assert (targets != -100).sum().item() == 4
    assert inputs[0, 7].item() == ALPHABET.index("2")
    assert targets[0, 6].item() == ALPHABET.index("2")
    assert targets[0, 7].item() == ALPHABET.index("3")
    torch.manual_seed(4)
    model = TinyTransformerLM(TransformerConfig(len(ALPHABET), 64, 16, 2, 1, 0))
    output = model(inputs, targets)
    valid = targets != -100
    expected = F.cross_entropy(output.logits[valid], targets[valid])
    torch.testing.assert_close(output.loss, expected)


def test_batched_greedy_reader_matches_individual_calls_without_hidden_state():
    torch.manual_seed(3)
    model = TinyTransformerLM(TransformerConfig(len(ALPHABET), 64, 16, 2, 1, 0)).eval()
    prompts = ["MCQ12a=", "MS45b67;CQ45b=", "MCQ99d="]
    together = predict(model, prompts, batch_size=3, device="cpu")
    separately = [predict(model, [prompt], batch_size=1, device="cpu")[0] for prompt in prompts]
    assert together == separately
    assert predict(model, [prompts[0]], batch_size=1, device="cpu")[0] == together[0]
    assert not any("cache" in name for name in model.state_dict())


def test_training_views_are_reproducible_and_never_use_test_symbols():
    first = make_training_views("train", seed=5, count=128, capacity=4)
    assert first == make_training_views("train", seed=5, count=128, capacity=4)
    assert {read_visible(v.memory, v.current, v.query) == "??" for v in first} == {True, False}
    for view in first:
        for op in (*view.memory, *view.current, view.query):
            assert op.entity in symbol_space("train").entities
            if op.value is not None:
                assert op.value in symbol_space("train").values
        assert len(view.memory) <= 4


def test_small_real_experiment_saves_replayable_checkpoint_and_all_baselines(tmp_path):
    from bounded_memory_transformer.memory_benchmark.evaluate import run_experiment

    config = {
        "seeds": [5],
        "device": "cpu",
        "threads": 1,
        "steps": 2,
        "batch_size": 4,
        "train_examples": 32,
        "validation_examples": 8,
        "test_episodes": 8,
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "context_length": 64,
        "capacity": 4,
        "learning_rate": 0.002,
        "log_every": 1,
        "bootstrap_samples": 20,
        "scenarios": {"main": {"candidates": 30, "sessions": 8}},
    }
    summary = run_experiment(config, tmp_path)
    assert set(summary["neural"]["main"]) == {"no_memory", "fifo", "recency", "similarity"}
    assert summary["symbolic"]["main"]["no_memory"]["accuracy"] == 0.75
    assert summary["contract"]["raw_history_tokens_reprocessed"] == 0
    assert json.loads((tmp_path / "summary.json").read_text())["config"] == config
    checkpoint = torch.load(tmp_path / "seed-5.pt", weights_only=True)
    model = TinyTransformerLM(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    row = json.loads((tmp_path / "predictions.jsonl").read_text().splitlines()[0])
    assert predict(model, [row["prompt"]], batch_size=1, device="cpu")[0] == row["prediction"]
