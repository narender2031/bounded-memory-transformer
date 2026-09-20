import pytest

from bounded_memory_transformer.memory_experiments.metrics import (
    paired_accuracy,
    reader_report,
    recovery,
    selection_diagnostics,
)


def test_recovery_uses_oracle_gain_and_does_not_clamp():
    assert recovery(0.7965, 0.82, 0.35)["value"] == pytest.approx(0.95)
    assert recovery(0.9, 0.8, 0.5)["value"] == pytest.approx(4 / 3)
    for oracle in (0.4, 0.5):
        result = recovery(0.5, oracle, 0.5)
        assert result["value"] is None
        assert result["reason"] == "no_positive_oracle_gain"
    with pytest.raises(ValueError):
        recovery(float("nan"), 0.8, 0.5)


def test_paired_harm_benefit_and_episode_clusters_have_literal_denominators():
    report = paired_accuracy(
        ["a", "x", "b", "??"],
        ["x", "a", "x", "??"],
        ["a", "a", "b", "??"],
        episode_ids=[0, 0, 1, 1],
        bootstrap_samples=100,
    )
    assert report["accuracy"] == 0.75
    assert report["baseline_accuracy"] == 0.5
    assert report["harmful_rate"] == 0.25
    assert report["beneficial_rate"] == 0.5
    assert report["unique_episodes"] == 2
    assert report["delta"] == 0.25
    with pytest.raises(ValueError):
        paired_accuracy(["a"], [], ["a"])


def test_discarded_information_does_not_count_as_reader_incompetence():
    # The last two world answers are gone. Exact visible reading correctly abstains.
    report = reader_report(
        predictions=["23", "45", "??", "??"],
        visible_targets=["23", "45", "??", "??"],
        world_targets=["23", "45", "67", "89"],
        no_evidence=["??"] * 4,
        paired_no_memory=["??"] * 4,
        categories={"known": [True, True, False, False], "unknown": [False, False, True, True]},
        episode_ids=[0, 1, 2, 3],
        bootstrap_samples=100,
    )
    assert report["world_accuracy"] == 0.5
    assert report["visible_accuracy"] == 1
    assert report["reader_recovery"]["value"] == 1
    assert report["absolute_gate_passed"]
    assert report["relative_gate_passed"]


def test_bad_stratum_cannot_hide_behind_overall_accuracy_and_missing_category_is_not_pass():
    report = reader_report(
        predictions=["23"] * 19 + ["wrong"],
        visible_targets=["23"] * 20,
        world_targets=["23"] * 20,
        no_evidence=["??"] * 20,
        paired_no_memory=["??"] * 20,
        categories={
            "select": [True] * 19 + [False],
            "contradicted": [False] * 19 + [True],
            "deleted": [False] * 20,
        },
        bootstrap_samples=100,
    )
    assert report["visible_accuracy"] == 0.95
    assert report["by_category"]["contradicted"]["accuracy"] == 0
    assert report["by_category"]["deleted"]["accuracy"] is None
    assert not report["absolute_gate_passed"]


def test_selection_and_generation_joint_errors_are_not_added():
    result = selection_diagnostics(
        [0, 1, -1, 0],
        [0, 0, -1, 1],
        ["23", "71", "??", "45"],
        ["23", "23", "71", "71"],
        ["23", "23", "??", "45"],
    )
    assert result["selector_index_accuracy"] == 0.5
    assert result["copy_accuracy"] == 0.75
    assert result["generation_accuracy"] == 0.5
    assert result["copy_correct_generation_wrong"] == 2
    assert result["copy_wrong_generation_correct"] == 1
