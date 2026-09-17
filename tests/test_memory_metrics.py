import pytest

from bounded_memory_transformer.memory_benchmark.metrics import Result, summarize
from bounded_memory_transformer.memory_benchmark.operations import QueryTruth


def test_metrics_use_hand_calculated_denominators_and_paired_harm():
    rows = [
        Result(0, 7, "history_update", QueryTruth("45", ("23",), False, 1, 2), "23", "??"),
        Result(1, 7, "history_set", QueryTruth("67", (), False, 0, 3), "67", "??", True),
        Result(2, 7, "history_delete", QueryTruth("??", ("23",), True, 0, 2), "23", "??"),
        Result(3, 7, "unknown", QueryTruth("??"), "??", "??"),
    ]
    result = summarize(rows, bootstrap_samples=100, seed=1)
    assert result["accuracy"] == 0.5
    assert result["no_memory_accuracy"] == 0.5
    assert result["accuracy_delta"] == 0
    assert result["harmful_rate"] == 0.25
    assert result["beneficial_rate"] == 0.25
    assert result["stale_answer_rate"] == 1
    assert result["updated_queries"] == 1
    assert result["deleted_fact_leakage"] == 1
    assert result["abstention_precision"] == 1
    assert result["abstention_recall"] == 0.5
    assert result["abstention_f1"] == pytest.approx(2 / 3)
    assert result["useful_fact_retention"] == 0.5
    assert result["by_case"]["history_update"]["accuracy"] == 0
    assert result["by_gap"]["3"]["accuracy"] == 1
    assert result["by_overwrites"]["1"]["accuracy"] == 0


def test_invalid_output_is_wrong_not_abstention_and_empty_denominators_are_null():
    result = summarize([Result(0, 1, "unknown", QueryTruth("??"), "S;", "??")])
    assert result["accuracy"] == 0
    assert result["abstention_recall"] == 0
    assert result["abstention_precision"] is None
    assert result["stale_answer_rate"] is None
    assert result["useful_fact_retention"] is None
    assert result["accuracy_delta_ci95"] == [-1, -1]


def test_bootstrap_clusters_same_episode_across_model_seeds():
    rows = [
        Result(episode_id, seed, "history_set", QueryTruth("23"), prediction, baseline)
        for seed in (1, 2, 3)
        for episode_id, prediction, baseline in ((0, "23", "??"), (1, "??", "23"))
    ]
    result = summarize(rows)
    assert result["unique_episodes"] == 2
    # Row-level resampling falsely treats the six outcomes as independent and
    # narrows this interval. There are only two independent paired episodes.
    assert result["accuracy_delta_ci95"] == [-1, 1]
    assert result["seed_accuracy_std"] == 0


def test_report_retains_paired_metrics_when_stale_denominator_is_absent():
    from bounded_memory_transformer.memory_benchmark.evaluate import markdown_report

    metrics = summarize([Result(0, 1, "unknown", QueryTruth("??"), "23", "??")])
    summary = {
        "environment": {"device": "cpu"},
        "effective_config": {"capacity": 4},
        "training": [
            {
                "seed": 1,
                "parameters": 100,
                "validation_accuracy": 1,
                "training_seconds_including_validation": 1,
            }
        ],
        "reader_competence_passed": True,
        "neural": {"main": {"fifo": metrics}},
        "symbolic": {"main": {"fifo": {"accuracy": 1}}},
    }
    row = next(line for line in markdown_report(summary).splitlines() if line.startswith("| fifo"))
    assert row.split("|")[3].strip() == "-100.00 [-100.00, -100.00]"
    assert row.split("|")[4].strip() == "100.00%"
    assert row.split("|")[6].strip() == "n/a"
