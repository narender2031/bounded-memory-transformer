import json

import pytest

from bounded_memory_transformer.memory_experiments.reader_training import ReaderBundle
from bounded_memory_transformer.memory_experiments.run import run_experiment


def test_all_five_cases_save_predictions_checkpoints_and_failed_gates(tmp_path):
    config = {
        "seeds": [7],
        "device": "cpu",
        "threads": 1,
        "reader": {
            "capacity": 4,
            "train_examples": 25,
            "validation_examples": 25,
            "selector_steps": 2,
            "character_steps": 2,
            "batch_size": 8,
            "learning_rate": 0.002,
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "context_length": 128,
            "selector_d_model": 16,
            "selector_n_layers": 1,
            "log_every": 2,
        },
        "lifecycle": {
            "steps": 2,
            "batch_size": 8,
            "learning_rate": 0.01,
            "hidden": 8,
            "slots": 4,
            "training_examples": 32,
        },
        "capacity_training": {"episodes": 2, "capacity": 4, "learning_rate": 0.03},
        "evaluation": {
            "split": "validation",
            "reader_tasks": 25,
            "lifecycle_episodes": 4,
            "capacity_episodes": 2,
            "capacity_queries": 8,
            "capacity_slots": [4, 8],
            "reader_seed": 2121,
            "lifecycle_seed": 2122,
            "capacity_seed": 2123,
            "bootstrap_samples": 20,
        },
    }
    summary = run_experiment(config, tmp_path)
    seed = summary["seeds"]["7"]
    assert seed["reader"]["oracle_copy"]["visible_accuracy"] == 1
    assert not seed["gates"]["reader_absolute"]
    assert seed["lifecycle"]["exact"]["semantic_accuracy"] == 1
    assert set(seed["capacity"]) == {"A", "B"}
    assert set(seed["capacity"]["A"]) == {"4", "8"}
    assert "combined_capacity" in seed["end_to_end"]
    assert summary["contract"]["raw_history_tokens_reprocessed"] == 0
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    rows = [json.loads(line) for line in (tmp_path / "reader-7.jsonl").read_text().splitlines()]
    assert len(rows) == 25
    assert sum(row["answers"]["oracle_copy"] == row["target"] for row in rows) == 25
    bundle = ReaderBundle.load(tmp_path / "reader-7.pt")
    assert bundle.metadata["seed"] == 7
    assert summary["artifact_sha256"]["reader-7.jsonl"]
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path)
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileExistsError):
        run_experiment(config, tmp_path)


def test_episode_gate_cannot_be_hidden_by_microtask_success():
    from bounded_memory_transformer.memory_experiments.run import episode_reader_gate

    def report(value):
        return {"selected_copy": {"reader_recovery": {"value": value}}}

    result = {
        "capacity": {
            "A": {
                "4": {
                    "fifo": {"neural": report(0.94)},
                    "no_memory": {"neural": report(None)},
                    "paired_comparisons": {},
                }
            }
        },
        "end_to_end": {
            "lifecycle": {"reader": report(1.0)},
            "combined_capacity": {"A": {"4": {"neural": report(0.98)}}},
        },
    }
    assert not episode_reader_gate(result)["passed"]
    result["capacity"]["A"]["4"]["fifo"]["neural"] = report(0.95)
    assert episode_reader_gate(result)["passed"]
    result["end_to_end"]["lifecycle"]["reader"] = report(None)
    gate = episode_reader_gate(result)
    assert gate["passed"] is False
    assert "lifecycle" in gate["unavailable"]


@pytest.mark.parametrize("slots", [4, 8])
def test_composed_exact_lifecycle_matches_capacity_policy(slots):
    from bounded_memory_transformer.memory_experiments import capacity, lifecycle
    from bounded_memory_transformer.memory_experiments.run import _combined_capacity

    class ExactController:
        decide = staticmethod(lifecycle.exact_decision)

    scorer = capacity.RetentionScorer()
    episodes = capacity.generate_capacity("validation", 774, 3)
    actual, writes = _combined_capacity(ExactController(), scorer, episodes, slots)
    expected = capacity.evaluate_capacity(episodes, "learned", slots, model=scorer)
    for first, second in zip(actual["episodes"], expected["episodes"], strict=True):
        assert first["final_bank"] == second["final_bank"]
        assert first["records"] == second["records"]
    policy = capacity.make_capacity_policy("learned", slots, model=scorer)
    trajectory = iter(writes)
    for episode in episodes:
        policy.bank[:] = -1
        for session in episode.sessions:
            for operation, cue in session:
                policy.observe(operation, cue)
                assert next(trajectory)["bank_after"] == policy.bank.tolist()
            policy.reset_session()
