# Five-case bounded memory implementation plan

> **For agentic workers:** Execute the independent domains under the
> superpowers:dispatching-parallel-agents workflow, with test-driven development
> and integration review. Checkboxes track completed work.

**Goal:** Implement and locally test all five approved cases and a combined
diagnostic, preserving unsuccessful results as well as successes.

**Architecture:** Add `memory_experiments` beside the frozen Phase 1 package.
Reader controls consume existing QueryView objects. Lifecycle and capacity share
a 64-byte four-slot canonical fact representation. An orchestration CLI combines
independently tested modules and writes reproducible artifacts.

**Tech stack:** Existing Python/PyTorch/NumPy, manual tiny Transformer blocks,
pytest, Ruff; no new runtime dependency.

**Spec:** [Five-case protocol](../specs/2026-09-20-five-case-memory.md).

## Constraints and ownership

- Worktree: `/Users/narendersingh/transformer/bounded-memory-transformer-five-cases`.
- Python: `PYTHONPATH=src /Users/narendersingh/transformer/bounded-memory-transformer/.venv/bin/python`.
- Four slots, int32 `[entity, attribute, value, cue]`; -1 entity marks an empty slot.
- Only arrays and current input cross module interfaces; no hidden history.
- Reader owner: `reader_cases.py`, `selection.py`, `reader_training.py`, reader tests.
- Lifecycle owner: `lifecycle.py`, lifecycle tests.
- Capacity owner: `capacity.py`, capacity tests.
- Root: common `__init__.py`, metrics, runner, config, reports, integration, docs.
- Agents write only their owned files, no commits, no full training runs, and
  report exact public interfaces and tests. Root integrates and commits.

## Task 1 — Reader oracles, strata, selector, training

- [x] Write tests with literal outputs: old A=23/current A=45 ->45; absent A ->??;
  deleted A ->??; attribute mismatch ->??. Verify failure before implementation.
- [x] Implement visible-only oracle selection, exact copy, and selected QueryView
  projection preserving memory/current origin. Neither oracle nor model may read
  `view.retained`; include a forged-retained-field regression test.
- [x] Generate balanced `select`, `unsupported`, `contradicted`, `irrelevant`,
  `deleted` task strata with occupancy and known/unknown tags. Use `symbol_space`.
  Include old/current and within-bank overrides, both key-coordinate distractors,
  and shared empty-memory controls. Assert oracle copy equals `read_visible`.
- [x] Implement a tiny manual-Transformer record scorer and UNKNOWN candidate.
  Query and each record are encoded jointly; record order/current origin may be
  explicit features. Learn full-key discrimination; do not hard-code key equality
  in the reader. Return B×(N+1) scores with padding masked. Test padding invariance,
  reset/no cross-batch state, invalid indices, and masked candidates.
- [x] Train selector with visible oracle indices and character reader with answer
  tokens. Reuse character weights across full and selected input; reuse selector
  choices across generation and copying. Record supervision and compute.
- [x] Public entrypoint: `train_readers(config, seed, device)` returns a bundle
  with `.predict(views)` mapping four system names to answer lists, selector
  indices, metadata, and saveable state dictionaries. Expose task generation and
  oracle helpers separately for the root runner. Return actual tested interfaces.

Example behavioral fixture (reuse existing Operation/QueryView types):

```python
v = QueryView((Operation(Kind.SET, 12, 0, 23),),
              (Operation(Kind.UPDATE, 12, 0, 45),),
              Operation(Kind.ASK, 12, 0), 64, ())
assert copy_selected(v, oracle_select(v)) == "45"
assert copy_selected(v, 0) == "23"  # no gold correction of a wrong choice
```

## Task 2 — Learned lifecycle and honest executor

- [x] Write tests for update/delete of one key preserving another, both key
  coordinates, IGNORE, missing-target deletion, full-bank allocation, and bytes.
- [x] Implement action/target model using explicit key-equality/occupancy features;
  train with causal state-transition labels. Record that these features simplify
  binding. No future data or capacity-oracle labels enter training.
- [x] Executor accepts predicted `(action, target)` and current operation/cue.
  `STORE` allocation uses a provided victim/empty slot; UPDATE/DELETE act only on
  the predicted slot. Wrong decisions must stay wrong in measured outcomes.
- [x] Train and evaluate controlled two-key update/delete episodes, with three
  seeds, exact reading, action/target/transition metrics and collateral checks.
- [x] Expose model training plus `decide(bank, operation)` and
  `apply_decision(bank, operation, decision, cue=0, allocation=None)`. Bank is an
  int32 array; absent/full allocation is explicit. Provide held-out task generator
  and per-task records, not just aggregate accuracy, for end-to-end integration.

Hand-derived semantic fixture:

```python
bank = np.array([[12, 0, 23, 0], [13, 0, 71, 0], [-1]*4, [-1]*4], dtype=np.int32)
# Correct UPDATE targets slot 0; correct DELETE then clears only slot 0.
# A deliberately wrong target 1 must damage B and be observable by the test.
```

## Task 3 — Capacity workloads, causal policies, retention bound

- [x] Write independent fixtures: four known request counts [5,3,2,0] and K=2
  imply oracle utility .8; a bank retaining counts [3,2] implies utility .5 and
  regret .3. Ensure policy state trajectories are invariant to changing queries.
- [x] Implement 24-key/30-write/32-query generators A/B, with paired identical
  prefixes and separate suffix RNG, cues assigned before probes, and unseen split
  symbols. Provide immutable episodes and streamed `(Operation, cue)` writes.
- [x] Implement exact-lifecycle FIFO facts, write recency, bounded similarity,
  stateless random, cue-priority, and learned retention at equal K×4 bytes.
  No policy may consume an Episode or its future queries at observe time.
- [x] Implement closed-form top-K request-count oracle for this write-then-query
  workload. Keep it outside policies and trainers; verify feasibility/latest values.
- [x] Train a small retention scorer from delayed query rewards (REINFORCE with a
  training-only running baseline), equal A/B mix. Save weights, training seeds,
  learning curve, and feature definition; greedy evaluation has no hidden RNG.
- [x] Expose `generate_capacity(split, seed, count, ...)`, `train_capacity(...)`,
  and `evaluate_capacity(...)` plus an online policy factory. Return final bank
  and query predictions for root neural-reader integration. Include capacity 4/8.
- [x] Test no query reload, no future suffix leakage, fixed bytes, oracle bound,
  unchanged source episodes, and cue-feature availability. Do not run the final
  held-out comparison independently; root freezes all configuration first.

## Task 4 — Gate and paired metric implementation

- [x] Write hand-calculated tests before implementation for recovery and joint
  correctness. Use `A_N=.7965,A_O=.82,A_0=.35` -> recovery .95. Nonpositive
  oracle gain returns null plus reason; do not hide it with an epsilon.
- [x] Implement visible absolute categories, relative gate, copied/generation joint
  errors, abstention, paired harm/benefit, and episode-clustered confidence intervals.
- [x] Keep new metrics outside old `memory_benchmark/metrics.py` and report exact
  denominators, per-seed values, and raw accuracies for each gate decision.

```python
assert recovery(.7965, .82, .35)["value"] == pytest.approx(.95)
assert recovery(.5, .5, .5)["value"] is None
assert recovery(.9, .8, .5)["value"] == pytest.approx(4/3)
```

## Task 5 — Reproducible runner and end-to-end integration

- [x] Create `projects/03-memory-reliability/configs/five-case-small.json`, a
  dedicated evaluation CLI, and smoke config. Fix seeds/budgets before test data.
- [x] Train all components before reading test symbols. Save validation curves,
  component gates and failed gates. Validation-only fixes get separate saved runs.
- [x] Emit per-example reader predictions and selector indices; per-operation
  lifecycle predictions and final banks; per-capacity query outputs, oracle utility,
  regret, all baselines, and combined-reader outputs. Include exact substitutions.
- [x] If gates fail, still complete a labelled diagnostic run; do not silently
  skip cases or call their hypotheses successful. No additional held-out tuning.
- [x] Record environment, parameter counts, state bytes, source/data/checkpoint
  SHA256 hashes, config and timings. Disallow overwriting an existing output dir.
- [x] Add real tiny CPU train/save/reload integration test and JSON reconciliation
  checks. Validate any composed writer adapter against each component in isolation.

## Task 6 — Execute, audit, and report

- [x] Run all tests and Ruff. Run a small CPU smoke experiment before the recorded
  MacBook MPS run. Tests must include masking, persistence/reset, and no-future access.
- [x] Freeze and commit source/config, then execute the recorded three-seed run.
- [x] Independently recalculate all aggregate scores and gate outcomes from saved
  rows; inspect representative false selections, stale/deleted errors, and regrets.
- [x] Generate standalone figures for reader strata, lifecycle integrity, A/B
  utility and regret at 4/8 slots. Use checked-in summaries as chart input.
- [x] Save a factual report, including negative results, compute differences,
  limits of planted utility cues, and next falsifiable question. Update CODEX,
  paper-linked brief, decisions, experiment plan, research log, and README.
- [x] Commit/push the isolated branch and open a reviewable PR stacked on #2;
  verify its CI without merging either base PR.

Completion: [draft PR #3](https://github.com/narender2031/bounded-memory-transformer/pull/3)
is stacked on #2. The [hosted CPU lint/test run](https://github.com/narender2031/bounded-memory-transformer/actions/runs/35499085042)
passed at `116c1f2`. All cases are implemented and tested; failed reader research
gates are retained in the results. Neither base PR was merged.
