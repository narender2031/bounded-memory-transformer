"""Hand-derived checks for the predeclared, episode-paired replication."""

import gzip
import hashlib
import json
import math
import subprocess
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from bounded_memory_transformer.memory_benchmark.generator import symbol_space
from bounded_memory_transformer.memory_benchmark.operations import Kind, Operation
from bounded_memory_transformer.memory_experiments.capacity import CapacityEpisode, RetentionScorer


def test_paired_intervals_use_episode_se_and_six_contrast_multiplicity():
    from bounded_memory_transformer.memory_followup.uniform import paired_interval

    result = paired_interval([-.5, 0, .5], family_size=6, alpha=.05, margin=.01)
    assert result["episodes"] == 3
    assert result["mean"] == 0
    assert result["standard_deviation"] == pytest.approx(.5)
    assert result["standard_error"] == pytest.approx(.5 / math.sqrt(3))
    assert result["ci95"] == pytest.approx([-1.959963984540054 * .5 / math.sqrt(3),
                                         1.959963984540054 * .5 / math.sqrt(3)])
    assert result["simultaneous_ci"] == pytest.approx(
        [-2.638257273476752 * .5 / math.sqrt(3), 2.638257273476752 * .5 / math.sqrt(3)]
    )
    assert not result["equivalent"]
    assert not result["positive"]


def test_equivalence_is_distinct_from_significance_and_uses_strict_margin():
    from bounded_memory_transformer.memory_followup.uniform import paired_interval

    tiny = paired_interval([.0001] * 20, family_size=6, alpha=.05, margin=.01)
    assert tiny["equivalent"] and tiny["positive"]
    boundary = paired_interval([.01] * 20, family_size=6, alpha=.05, margin=.01)
    assert not boundary["equivalent"]
    negative = paired_interval([-.2] * 20, family_size=6, alpha=.05, margin=.01)
    assert negative["negative"] and not negative["equivalent"]


@pytest.mark.parametrize("values", [[], [0], [0, np.nan], [0, np.inf]])
def test_interval_rejects_unusable_episode_samples(values):
    from bounded_memory_transformer.memory_followup.uniform import paired_interval

    with pytest.raises(ValueError):
        paired_interval(values, family_size=6, alpha=.05, margin=.01)


def test_analysis_counts_episodes_once_and_bootstraps_within_dataset_seed():
    from bounded_memory_transformer.memory_followup.uniform import analyze_contrasts

    contrasts = {101: {"K4:learned-fifo": [.1, .1]},
                 103: {"K4:learned-fifo": [-.1, -.1]}}
    options = {"familywise_alpha": .05, "equivalence_margin": .01,
               "bootstrap_samples": 80, "bootstrap_seed": 71}
    result = analyze_contrasts(contrasts, options)
    pooled = result["pooled"]["K4:learned-fifo"]
    assert result["dataset_seeds"] == [101, 103]
    assert pooled["episodes"] == 4
    assert pooled["mean"] == 0
    assert pooled["standard_error"] == pytest.approx(math.sqrt(.04 / 3) / 2)
    assert pooled["bootstrap_ci"] == [0, 0]
    assert not result["all_contrasts_practically_equivalent"]
    assert result["per_seed"]["101"]["K4:learned-fifo"]["mean"] == .1
    assert result == analyze_contrasts(contrasts, options)


def test_analysis_refuses_unmatched_dataset_contrasts():
    from bounded_memory_transformer.memory_followup.uniform import analyze_contrasts

    with pytest.raises(ValueError, match="same contrasts"):
        analyze_contrasts({101: {"a": [0, 0]}, 103: {"b": [0, 0]}},
                          {"familywise_alpha": .05, "equivalence_margin": .01,
                           "bootstrap_samples": 80, "bootstrap_seed": 71})


def increasing_scorer():
    model = RetentionScorer()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.1)
    return model.eval()


def uniform_fixture():
    writes = tuple((Operation(Kind.SET, entity, 0, 20 + entity), entity % 2)
                   for entity in range(4))
    update = ((Operation(Kind.UPDATE, 0, 0, 70), 0),)
    return CapacityEpisode(0, "A", (writes, update),
                           tuple(Operation(Kind.ASK, entity, 0) for entity in range(4)))


def test_alias_proof_checks_all_binary_cue_allocations_and_rejects_reversed_scorer():
    from bounded_memory_transformer.memory_followup.uniform import verify_model_equivalence

    model = increasing_scorer()
    proof = verify_model_equivalence({7: model, 19: increasing_scorer()}, [1, 2])
    assert proof["representative_model_seed"] == 7
    assert proof["allocations_checked_per_model"] == 12
    assert proof["prediction_aliases"] == {"learned:7": "learned", "learned:19": "learned",
                                            "cue_priority": "learned"}
    with torch.no_grad():
        model.network[-1].weight.neg_()
    with pytest.raises(ValueError, match="cue-priority"):
        verify_model_equivalence({7: model}, [1, 2])


def test_uniform_enumerated_queries_recover_capacity_fraction_with_current_values():
    from bounded_memory_transformer.memory_followup.uniform import (
        build_contrasts,
        evaluate_uniform_dataset,
    )

    results = evaluate_uniform_dataset([uniform_fixture()], increasing_scorer(), [1, 2],
                                      [7, 19, 43], keys=4)
    assert len(results) == 12
    for name, result in results.items():
        slots = int(name[1])
        assert result["summary"]["utility"] == slots / 4
        assert result["summary"]["persistent_state_bytes"] == slots * 16
        assert result["summary"]["historical_tokens_reprocessed"] == 0
    differences = build_contrasts(results, [1, 2], [7, 19, 43])
    assert len(differences) == 6
    assert all(values == [0] for values in differences.values())


def test_changed_future_queries_cannot_change_any_executed_policy_bank():
    from bounded_memory_transformer.memory_followup.uniform import evaluate_uniform_dataset

    original = uniform_fixture()
    alternative = replace(original, queries=(Operation(Kind.ASK, 0, 0),) * 4)
    first = evaluate_uniform_dataset([original], increasing_scorer(), [2], [7, 19, 43], keys=4)
    second = evaluate_uniform_dataset([alternative], increasing_scorer(), [2], [7, 19, 43],
                                      keys=4)
    for name in first:
        assert first[name]["episodes"][0]["final_bank"] == second[name]["episodes"][0]["final_bank"]
    assert any(first[name]["summary"]["utility"] != second[name]["summary"]["utility"]
               for name in first)


@pytest.mark.parametrize("corruption", ["stale", "duplicate", "prediction", "denominator"])
def test_independent_result_validation_rejects_corrupted_bank_or_metric(corruption):
    from bounded_memory_transformer.memory_followup.uniform import (
        evaluate_uniform_dataset,
        validate_result,
    )

    episode = uniform_fixture()
    result = evaluate_uniform_dataset([episode], increasing_scorer(), [2], [7], keys=4)["K2:fifo"]
    bad = deepcopy(result)
    if corruption == "stale":
        bad["episodes"][0]["final_bank"][0][2] = 99
    elif corruption == "duplicate":
        bad["episodes"][0]["final_bank"][1] = bad["episodes"][0]["final_bank"][0].copy()
    elif corruption == "prediction":
        bad["episodes"][0]["records"][0]["answer"] = "99"
    else:
        bad["summary"]["queries"] = 12
    with pytest.raises(ValueError):
        validate_result([episode], bad, capacity=2, keys=4)


def test_random_contrast_averages_policy_seeds_within_episode_and_checks_pairing():
    from bounded_memory_transformer.memory_followup.uniform import build_contrasts

    def rows(counts):
        return {"episodes": [{"episode_id": index, "hits": hits, "queries": 4}
                             for index, hits in enumerate(counts)]}

    results = {"K4:learned": rows([3, 1]), "K4:fifo": rows([1, 1]),
               "K4:recency": rows([2, 2]), "K4:random:7": rows([0, 2]),
               "K4:random:19": rows([1, 1]), "K4:random:43": rows([2, 0])}
    contrasts = build_contrasts(results, [4], [7, 19, 43])
    assert contrasts == {"K4:learned-fifo": [.5, 0], "K4:learned-recency": [.25, -.25],
                         "K4:learned-random": [.5, 0]}
    results["K4:fifo"]["episodes"].reverse()
    with pytest.raises(ValueError, match="paired"):
        build_contrasts(results, [4], [7, 19, 43])


def smoke_configuration(tmp_path):
    checkpoint = tmp_path / "capacity-7.pt"
    torch.save({"state_dict": increasing_scorer().state_dict(), "config": {}}, checkpoint)
    source = tmp_path / "frozen-marker.txt"
    source.write_text("frozen test fixture\n")
    (tmp_path / "protocol.md").write_text("Predeclared training-only test fixture.\n")
    configuration = {
        "protocol": "protocol.md", "original_frozen_revision": "fixture", "threads": 1,
        "checkpoints": [{"model_seed": 7, "path": checkpoint.name,
                         "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}],
        "frozen_sources": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        "generator": {"workload": "A", "keys": 4, "updates": 2, "sessions": 2, "queries": 4},
        "capacities": [1, 2], "random_policy_seeds": [7],
        "evaluation": {"split": "test", "dataset_seeds": [501, 503], "episodes_per_seed": 4},
        "inference": {"familywise_alpha": .05, "equivalence_margin": .01,
                      "bootstrap_samples": 40, "bootstrap_seed": 919},
        "smoke": {"split": "train", "dataset_seeds": [101, 103],
                  "episodes_per_seed": 2, "bootstrap_samples": 20},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    return path


def test_smoke_saves_recomputable_training_data_hashes_and_refuses_overwrite(tmp_path):
    from bounded_memory_transformer.memory_followup.uniform import run_replication

    config = smoke_configuration(tmp_path)
    output = tmp_path / "smoke"
    summary = run_replication(config, output, mode="smoke", repo_root=tmp_path)
    assert summary["effective_evaluation"]["split"] == "train"
    assert summary["analysis"]["dataset_seeds"] == [101, 103]
    assert all(row["episodes"] == 4 for row in summary["analysis"]["pooled"].values())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "smoke"
    assert manifest["held_out_generated"] is False
    for name, digest in manifest["artifact_hashes"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    with gzip.open(output / "dataset-101.json.gz", "rt") as handle:
        episodes = json.load(handle)
    assert len(episodes) == 2
    assert all(operation["entity"] in symbol_space("train").entities
               for episode in episodes for session in episode["sessions"]
               for operation, _ in session)
    with gzip.open(output / "dataset-101-K1-fifo.json.gz", "rt") as handle:
        result = json.load(handle)
    assert result["summary"]["episodes"] == 2
    assert sum(len(row["records"]) for row in result["episodes"]) == 8
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_replication(config, output, mode="smoke", repo_root=tmp_path)
    assert (output / "manifest.json").read_bytes() == before


def test_held_out_requires_freeze_commit_before_creating_artifacts(tmp_path):
    from bounded_memory_transformer.memory_followup.uniform import run_replication

    config = smoke_configuration(tmp_path)
    output = tmp_path / "held-out"
    with pytest.raises(ValueError, match="freeze"):
        run_replication(config, output, mode="held-out", repo_root=tmp_path)
    assert not output.exists()


def test_smoke_cannot_be_redirected_to_test_split(tmp_path):
    from bounded_memory_transformer.memory_followup.uniform import run_replication

    config = smoke_configuration(tmp_path)
    changed = json.loads(config.read_text())
    changed["smoke"]["split"] = "test"
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="train"):
        run_replication(config, tmp_path / "smoke", mode="smoke", repo_root=tmp_path)


@pytest.mark.parametrize("name", ["capacity-7.pt", "frozen-marker.txt"])
def test_changed_checkpoint_or_source_aborts_before_generation(tmp_path, name):
    from bounded_memory_transformer.memory_followup.uniform import run_replication

    config = smoke_configuration(tmp_path)
    with (tmp_path / name).open("ab") as handle:
        handle.write(b"changed")
    output = tmp_path / "smoke"
    with pytest.raises(ValueError, match="hash"):
        run_replication(config, output, mode="smoke", repo_root=tmp_path)
    assert not output.exists()


def test_freeze_verification_checks_committed_bytes_not_only_a_revision_label(tmp_path):
    from bounded_memory_transformer.memory_followup.uniform import verify_freeze

    def git(*arguments):
        return subprocess.run(["git", "-C", str(tmp_path), *arguments], check=True,
                              text=True, capture_output=True).stdout.strip()

    git("init", "-q")
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    git("add", "source.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Freeze test fixture")
    commit = git("rev-parse", "HEAD")
    assert verify_freeze(tmp_path, [source], commit) == commit
    source.write_text("x = 2\n")
    with pytest.raises(ValueError, match="frozen commit"):
        verify_freeze(tmp_path, [source], commit)
