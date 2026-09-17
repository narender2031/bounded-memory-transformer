# Research Log

## 2026-08-27 — Project initialization and prior-art check

### Question

Can a small Transformer learn what to store, retrieve, update, and forget across context resets without rereading its history?

### Findings

1. The broad problem is an active research area spanning recurrent sequence models, test-time learning, KV-cache selection, latent memory, and agent memory.
2. RMT already shows that learned memory tokens can carry information between sequence segments.
3. MemoryLLM already supports a fixed latent memory pool that self-updates from new text.
4. Titans and MIRAS treat memory as an online learning system with explicit retention and forgetting design choices.
5. Meta FAIR's SP-KV directly studies learning when to write by predicting future KV utility. Future-utility selection alone is therefore not a novel claim.
6. GradMem tests compact learned memory after the original context is removed, making it a close architectural baseline.
7. Supersede shows that maintaining the current value of changing facts remains difficult under bounded self-managed memory.
8. LiveMem explicitly requires memory state to survive context turnover and affect behaviour after evidence leaves the context, but its measured state contribution is often modest.
9. LongMemEval supplies the most directly relevant natural evaluation categories: updates and abstention in addition to recall and temporal reasoning.

### Inference

The most defensible initial gap is the joint problem of future-utility selection and reliable semantic supersession in a tiny latent state under hard resets and without raw-history fallback.

This remains a provisional inference, not a verified novelty claim.

### Next research actions

1. Read and annotate RMT.
2. Read MemoryLLM's exact update and forgetting mechanism.
3. Compare Titans/MIRAS memory objectives with SP-KV utility gates.
4. Reproduce the Supersede task structure in a small symbolic generator.
5. Define state-matched RMT, FIFO, LRU, and oracle baselines.

## 2026-09-02 — Project 01 implementation

### Work completed

1. Added a reusable character-level tokenizer and deterministic next-token batcher.
2. Implemented causal multi-head self-attention directly from query, key, and value projections.
3. Implemented pre-normalized decoder blocks with residual connections and feed-forward networks.
4. Implemented a small decoder-only language model, next-token loss, and autoregressive generation.
5. Added a reproducible CLI and a tiny original corpus for smoke testing.
6. Added tests for configuration, tokenization, shifted targets, tensor shapes, causal masking, causal isolation, parameter updates, and generation.
7. Added continuous integration for linting and tests.

### Verification evidence

1. `ruff check .` completed with no findings.
2. `pytest` passed all 11 tests in 3.19 seconds on CPU.
3. A 27,520-parameter model completed a 50-step CPU smoke run without numerical failures.
4. Validation loss decreased from 3.5936 at step 1 to 2.7913 at step 50.
5. Autoregressive generation completed from the trained checkpoint. The sample remained mostly incoherent, which is expected from 50 steps on the intentionally tiny corpus.

This satisfies the Project 01 acceptance condition: the implementation is tested and demonstrates finite, decreasing loss end to end.

### Research impact

The resulting model becomes the no-persistent-memory baseline. Persistent state, context-reset episodes, and learned memory controllers remain intentionally excluded until the baseline is trusted.

## 2026-09-17 — Experiment 01: local hard-reset memory pilots

### Authorized milestone and implementation

The user requested Phase 1 locally: use the tiny Transformer, reset context
between sessions, and compare no memory, FIFO, recency, and similarity at four
slots before inventing a learned architecture. They subsequently asked whether
the cases could run on the MacBook; both recorded runs used its Apple MPS GPU.

Added typed operations, an independent full-truth state machine, deterministic
eight-case episodes with disjoint full entity/value symbols, fixed-array memory
policies, a shared neural reader, paired metrics, checkpoint/prediction recording,
a CLI, tests, configurations, a results README, and a source-backed PNG/SVG figure.
The existing `TinyTransformerLM` implementation is unchanged. Symbolic banks
store exactly four int32 fields per slot: 64 logical payload bytes at four slots.
The exact-key reader sees the same evidence as the model and is scored separately.

Branch: `feat/synthetic-memory-benchmark`, based on the existing Project 01
source. PR #1 was still open when checked; it was not merged. Existing local
research notes and Project 01 diagnostic artifacts were preserved.

### Frozen protocol

- 30 candidate operations across eight sessions; four slots. Candidate count
  includes noise and repeated assignments, not 30 distinct live facts.
- Eight equal cases: historical SET/UPDATE/DELETE, unknown, current
  SET/UPDATE/DELETE, and near-miss key. This has 50% unknown answers and 25%
  explicit UPDATE cases; current SET supplies another 12.5% overwrite case.
  It replaces the handoff's tentative 20-candidate/50%-update recipe explicitly.
- Main, longer (60 candidates/16 sessions), and more-updates (four overwrites)
  scenarios, 800 episodes each, identical across policies and model seeds.
- Seeds 7/19/43; width 96, two decoder layers, four heads, 128-token context,
  zero dropout, batch 64, AdamW learning rate 0.002. **237,792 parameters**.
- Reader-only supervision on available evidence; admission and retrieval are
  hand-written. Shared frozen weights and greedy decoding across all policies.
- 95% visible-evidence validation accuracy in every seed is the competence gate.
- Paired bootstrap resamples episodes while keeping model seeds together;
  intervals are conditional on those trained seeds. Seed variability is separate.

### Evidence: original pilot and coverage failure

Command:

```bash
.venv/bin/python -u -m bounded_memory_transformer.memory_benchmark.evaluate \
  --config projects/02-memory-benchmark/configs/baseline-small.json \
  --output runs/when-memory-hurts-2026-09-17-v1
```

The original source is preserved in commit `7fc51fd`; all recorded source hashes
match that commit. The initial run preceded the commit, so its manifest's original
HEAD points to the baseline; `results/provenance.json` records the verified source
revision. Training used 1,200 steps and 16,384 microtasks. Visible-validation
accuracy was 68.95%, 68.36%, and 59.18%, respectively. Main neural accuracy was
63.33% no memory, 51.71% FIFO, 51.54% recency, and 43.58% similarity.

Independent review found a coverage defect: reading microtask distractors
excluded the queried entity, making attribute matching unnecessary. An
attribute-blind reader passed all 512 validation examples, yet failed on separate
validation-symbol episode views. This diagnosis did not use neural held-out
predictions. A new regression test failed before the fix (0/512 counterexamples)
and passed afterward (97/512 attribute-blind failures). The original pilot,
configuration, predictions, checkpoint, and source revision remain available.

### Evidence: corrected fixed-budget replication

Before inspecting the second run, committed the corrected reader microtasks and
configuration as `280d88c`. Half their distractor keys now share an entity but
have another attribute. Increased the fixed budget to 6,000 steps and 65,536
microtasks; used fresh held-out episode seed 1729. Kept the episode generator,
metrics, policies, symbol partitions, model, and learning rate unchanged. Fresh
episodes reuse the same unseen-symbol partition, not an independent symbol split.

```bash
.venv/bin/python -u -m bounded_memory_transformer.memory_benchmark.evaluate \
  --config projects/02-memory-benchmark/configs/reader-extended.json \
  --output runs/when-memory-hurts-2026-09-17-v2
```

Environment: Python 3.13.7, PyTorch 2.9.1, NumPy 2.5.3, macOS arm64, MPS,
four configured CPU threads. The run took 169.67 seconds between recorded start
and finish, excluding Python startup. Per-seed training including validation
took 50.98, 51.12, and 51.19 seconds. These are observations, not stable speed claims.

| Main condition | Neural accuracy | Delta vs no memory, pp (95% episode CI) | Harmful / all queries | Exact-reader accuracy |
|---|---:|---:|---:|---:|
| No memory | 68.71% | 0 | 0% | 75.00% |
| FIFO | 48.50% | −20.21 [−22.63, −17.75] | 23.21% | 82.50% |
| Recency | 48.58% | −20.13 [−22.50, −17.63] | 23.17% | 82.50% |
| Similarity | 48.79% | −19.92 [−23.33, −16.46] | 28.12% | 82.50% |

The decline appears in every seed. On longer episodes neural accuracy was 67.71%
without memory versus 46.46%/46.38%/44.42%; with more updates it was 68.67%
versus 48.17%/48.63%/48.25%, in FIFO/recency/similarity order.

In the main run, stale-value matches on updated answerable queries were 0.11%
without memory and 0.33%/0.33%/3.22% with memory. Deleted-query non-abstention
was 5.83% versus 26.83%/26.83%/28.83%. The exact-key reader has zero stale/deleted
errors with these policies. Raw historical replay is zero; bounded memory reads
serialize 28 characters for FIFO/recency and seven for similarity. The complete
summary reports retention, abstention, case/gap/overwrite slices, per-seed scores,
policy time, synchronized batched read time, and logical payload bytes.

**Negative result:** corrected visible-validation accuracy was only 78.52%,
74.61%, and 73.83%. All seeds failed the 95% competence gate despite the larger
budget. No additional tuning followed the second held-out evaluation.

### Inferences and hypotheses

- **Evidence:** Adding bounded memory lowers this weak neural reader's accuracy
  relative to its paired no-memory condition on these episodes.
- **Inference:** A major bottleneck is how the model reads retained evidence:
  the deterministic reader benefits from the same banks. This does not prove
  that stale storage or inadequate admission causes the neural decline.
- **Inference:** The stronger claim that learned memory management is needed is
  unsupported. A competent reader and strong structured baselines must come first.
- **Hypothesis:** Better complete-key matching, compositional copying of unseen
  values, and validation at full memory occupancy could resolve much of this
  failure. These explanations require separate train/validation diagnostics.

### Verification and artifacts

- `.venv/bin/ruff check .`: clean.
- `.venv/bin/pytest`: **40 passed**, including actual tiny training, save/reload,
  reset/persistence, hand-counted metrics, full-key coverage, and a bootstrap
  regression that rejects treating repeated model seeds as independent episodes.
- `git diff --check`: clean. Installed `bmt-memory-benchmark --help` works.
- Independent code review found no truth/raw-history leakage into the model.
  Both minor review findings were fixed: clustered-bootstrap test strength and
  report formatting when a metric denominator is absent.
- Verified every recorded source hash against `7fc51fd` (v1) and `280d88c` (v2).
  Compact results and prediction hashes are checked in under
  `projects/02-memory-benchmark/results/`; complete predictions and checkpoints
  remain in the two ignored local run directories. The original model source is unchanged.
- Matplotlib 3.11.2 is an optional plotting dependency. The standalone figure
  was rendered and visually checked; it prominently displays the failed gate.
- Fresh literature check (2026-09-17): primary STALE and Memora abstracts read
  and recorded in `papers.md`; no novelty claim or benchmark replication claimed.

**Next action:** Diagnose and improve the shared reader using training and
validation only until every seed clears the competence gate. Predeclare the next
test; then repeat the memory comparison before implementing a learned controller.
