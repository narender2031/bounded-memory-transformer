import torch

from bounded_memory_transformer.memory_benchmark.reader import predict, render
from bounded_memory_transformer.memory_experiments.reader_cases import (
    copy_selected,
    generate_reader_tasks,
    project_selected,
)
from bounded_memory_transformer.memory_experiments.reader_training import (
    ReaderBundle,
    train_readers,
)


def test_real_tiny_training_reloads_and_reuses_weights_and_choices(tmp_path):
    torch.set_num_threads(1)
    config = dict(
        capacity=4,
        train_examples=20,
        validation_examples=10,
        selector_steps=2,
        character_steps=2,
        batch_size=4,
        learning_rate=0.002,
        d_model=16,
        n_heads=2,
        n_layers=1,
        selector_d_model=16,
        selector_n_layers=1,
        context_length=96,
        log_every=2,
    )
    bundle = train_readers(config, seed=7, device="cpu")
    views = [t.view for t in generate_reader_tasks("validation", seed=22, count=10)]
    result = bundle.predict(views)
    assert set(result["answers"]) == {
        "character_full",
        "selected_character",
        "selected_copy",
        "oracle_copy",
    }
    selected = result["selector_indices"]
    assert result["answers"]["selected_copy"] == [
        copy_selected(v, i) for v, i in zip(views, selected, strict=True)
    ]
    prompts = [render(project_selected(v, i)) for v, i in zip(views, selected, strict=True)]
    assert result["answers"]["selected_character"] == predict(
        bundle.character, prompts, batch_size=4, device="cpu"
    )
    checkpoint = tmp_path / "reader.pt"
    bundle.save(checkpoint)
    restored = ReaderBundle.load(checkpoint)
    assert restored.predict(views)["answers"] == result["answers"]
    assert restored.predict(views)["selector_indices"] == selected
    assert len(bundle.metadata["selector_history"]) == 2
    assert len(bundle.metadata["character_history"]) == 2
    assert bundle.metadata["selector_parameters"] > 0
    assert bundle.metadata["character_parameters"] > 0
    assert bundle.predict([])["selector_indices"] == []
    for answers in result["answers"].values():
        assert len(answers) == len(views)
    assert bundle.metadata["validation"]["oracle_copy"]["overall"] == 1.0
    assert set(bundle.metadata["validation"]["selected_copy"]["strata"]) == {
        "select",
        "unsupported",
        "contradicted",
        "irrelevant",
        "deleted",
    }
