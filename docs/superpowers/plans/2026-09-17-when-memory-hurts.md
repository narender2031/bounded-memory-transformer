# When Memory Hurts Implementation Plan

> Execute inline, task by task, using test-first development and verification.

**Goal:** Build and run the user's Phase 1 experiment on this Mac.

**Architecture:** Independent truth generator, fixed-array hand-written memory
policies, shared existing Transformer reader, and paired metrics. All model
inputs cross a boundary carrying only bounded records and current operations.

**Tech stack:** Python, NumPy, PyTorch, pytest, Ruff; JSON configurations.

**Spec:** `docs/superpowers/specs/2026-09-17-when-memory-hurts-design.md`.

## Constraints

- Four slots, 30 candidate operations, eight sessions; no historical replay.
- Three model seeds, frozen generator/metrics before held-out evaluation.
- Preserve existing uncommitted research notes and Project 01 implementation.
- Shared reader weights; no learned controller; retain negative results.

## Tasks

- [ ] 1. Typed operations, reference state machine, deterministic episode generator.
  Create `memory_benchmark/{operations,state_machine,generator}.py` and
  `tests/test_episode_generator.py`. Test authoritative replacement, deletion,
  noise, unknown answers, independent replay, exact operation counts, and
  disjoint complete entity/value symbols before implementation.
- [ ] 2. Fixed-array policies and reset boundary.
  Create `memory_benchmark/{policies,views}.py` and
  `tests/test_memory_policies.py`. Test oldest eviction, unique-key refresh,
  tombstones, latest similarity tie, fixed byte count, old view immutability,
  and absence of discarded source evidence after reset.
- [ ] 3. Metrics with literal hand-calculated predictions.
  Create `memory_benchmark/metrics.py` and `tests/test_memory_metrics.py`.
  Verify stale/deleted denominators, abstention, paired harm/benefit and intervals.
- [ ] 4. Shared tiny-Transformer reader and reproducible CLI.
  Create `memory_benchmark/{reader,evaluate}.py`, CLI entry point, config, and
  `tests/test_memory_reader.py`. Test answer-only loss alignment, variable-length
  greedy decoding, stateless calls, checkpoint replay, and a tiny local run.
- [ ] 5. Freeze inputs, run the three-seed local experiment, inspect diagnostics.
  Run `.venv/bin/python -m bounded_memory_transformer.memory_benchmark.evaluate
  --config projects/02-memory-benchmark/configs/baseline-small.json`.
  Preserve all results without tuning on test data. Generate a results table
  and standalone figure from the recorded summary.
- [ ] 6. Verify and document.
  Run `.venv/bin/ruff check .`, `.venv/bin/pytest`, `git diff --check`.
  Update Project 02 README, root README, CODEX, research log/decisions/papers and
  experimental ladder. Review code and measured claims before completion.
