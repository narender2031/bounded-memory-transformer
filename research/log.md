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

Review artifact: [draft PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2),
stacked on Project 01's still-open branch. Only this session's research additions
were committed; earlier local research notes and diagnostic files remain intact
and unstaged. The branch is preserved for the next reader milestone.

## 2026-09-17 — Reader explanation, alternative mechanisms, and fundamentals

**Question:** Why was neural reader validation only 74–79% when the exact-rule
memory system reached 82.5%, and which changes should we test next? The user also
explicitly requested one parallel agent to teach the implemented Transformer
end to end. Root investigated memory while that agent prepared the separate
`research/transformer-fundamentals.md` guide; no production model was changed.

### Evidence: these are different evaluations

The 73.83–78.52% figures score 512 short visible-evidence reading tasks, across
three trained seeds. The 82.50% figure scores 800 complete main episodes against
historical truth using the deterministic `read_visible()` control. The direct
same-episode comparison is 68.71% neural / 75.00% rule without memory, and 48.50%
neural / 82.50% rule with FIFO. A policy plus the rule reader is a memory-based
system; the rule reader is Python exact-key logic, not another trained model or
an oracle with raw-history access. It cannot recover evicted evidence.

### Evidence: frozen-reader factorial diagnostic

Before running it, wrote `projects/02-memory-benchmark/configs/reader-diagnostic.json`:
2,048 generated reading structures, generator seed 91317, renaming seed 73019,
four slots, seeds 7/19/43 from the preserved v2 checkpoints. The four conditions
independently map complete entity and value symbols into train or validation
vocabularies. Per-view mappings are injective and preserve key equality,
operation order, input length, and evidence availability. Renaming occurs after
retrieval, so lexical similarity cannot change which evidence is being read.
No held-out test symbols or new training were used.

```bash
.venv/bin/python -u projects/02-memory-benchmark/diagnose_reader.py \
  --config projects/02-memory-benchmark/configs/reader-diagnostic.json \
  --output runs/reader-diagnostic-2026-09-17
```

| Entities / values | Overall | Known visible answer | Required abstention |
|---|---:|---:|---:|
| Seen / seen | 78.91% | 77.07% | 81.08% |
| Unseen / seen | 78.66% | 76.68% | 81.01% |
| Seen / unseen | 75.49% | 69.73% | 82.29% |
| Unseen / unseen | 76.06% | 70.63% | 82.46% |

Means use the three frozen models on 2,048 matched structures, not 6,144
independent examples. Both-unseen overall scores per seed were 77.88%, 75.83%,
and 74.46%. Known answers present only in memory scored 48.51% (426 structures
per seed); current-session known answers scored 84.43% (683); four visible slots
scored 63.14% (359, mixing known and unknown). These slices have different task
mixtures and are not causal interventions on answer location or slot occupancy.
Changing value vocabulary affects both input and output, not just the decoder.

Local environment: Python 3.13.7, PyTorch 2.9.1, MPS, four configured CPU threads.
The timed region took 3.39 seconds, excluding startup and initial base generation.
It includes renaming, inference, and prediction I/O; no throughput claim follows.
All 24,576 predictions remain in the ignored local run directory. The checked-in
summary is `projects/02-memory-benchmark/results/2026-09-17-reader-diagnostic.json`.
It records configuration, source/checkpoint/prediction hashes, and per-seed slices.

### Inferences, hypotheses, and research implications

- **Evidence:** the readers fail substantially even on seen-symbol microtasks.
  Unfamiliar values further reduce known-answer accuracy; entity renaming alone
  has a small effect in this diagnostic.
- **Inference:** a pure unfamiliar-entity explanation is inadequate. Selection,
  value reproduction, and missing-evidence rejection need separate controls.
  The current-versus-memory accuracy difference does not isolate its cause.
- **Hypothesis:** learned record selection with exact value copying and an UNKNOWN
  option is a useful next reader ablation. Include current-session candidates;
  copying cannot fix wrong selection or discarded evidence. No improved reader
  was implemented or trained during this session.
- **Decision D011:** use these separate reading diagnostics, preserve the 95%
  per-seed competence gate, and declare slice gates before the next training run.
  Keep historical pilots frozen and disclose any direct record supervision.
- **Parallel research:** use the exact-rule reader to investigate symbolic policy
  retention independently. Access-frequency policies require repeated queries;
  expiration needs declared lifetime semantics; all metadata counts as state.

### Fresh primary-source research, 2026-09-17

Added Key-Value Memory Networks (separate addressing and returned content),
Pointer-Generator Networks (copying), Sufficient Context (evidence sufficiency
and abstention), TinyLFU (admission from recent access frequency), and Gated
DeltaNet (gating plus targeted updates) to the map. Rechecked SP-KV, EXPIRE-SPAN,
RMT, and STALE. Read depths and methodological limits are recorded in `papers.md`
and `reading-notes/2026-09-17-memory-improvements.md`. No novelty or paper-result
replication claim. Recency already performs full-key supersession in our code.

### Verification and continuity

- `.venv/bin/ruff check .`: clean; `git diff --check`: clean at verification.
- `.venv/bin/pytest`: **40 passed** in 1.78 seconds.
- Verified diagnostic source and checkpoint hashes; independently recomputed all
  four overall scores from the 24,576 saved predictions and checked their hash.
- Runtime assertions checked every renamed task's answer mapping, full-key
  relationships, and prompt length before neural prediction.
- Reviewed the fundamentals guide against the source: 25-character vocabulary,
  96-wide/two-layer/four-head reader, tied output embeddings, 237,792 parameters,
  answer-only teacher forcing, stateless inference, and 64-byte fact payload.
- The requested teaching agent executed all five guide exercises successfully,
  including shape/mask checks, exact answer alignment, a five-step CPU training
  demonstration, 25 relevant tests, and loading the original seed-7 checkpoint.
  It also checked all 20 relative links. The guide contains 19 sections.
- Preserved prior uncommitted research notes and Project 01 diagnostic artifacts.
  Review continues on [draft PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2).

**Next action:** specify and implement the record-selection/copying reader ablation
on training/validation evidence, with its training budget and slice gates declared
before the run. The remaining weakness is model competence, not missing local
hardware or an already-proven need for a new memory architecture.

## 2026-09-20 — Results synthesis and six-paper primary-source review

### Request, scope, and evidence checked

The user requested a detailed summary of the measured results and then supplied
six new paper links for project-specific review. Created
[`results-summary-2026-09-20.md`](results-summary-2026-09-20.md) and
[`reading-notes/2026-09-20-six-paper-review.md`](reading-notes/2026-09-20-six-paper-review.md).
No model was trained and no new policy experiment was run. Historical generators,
metrics, configurations, and results were not changed.

Independently recalculated all 12 scenario-policy neural accuracies, paired
harmful/beneficial rates, case slices, and exact-reader scores from all 28,800
stored v2 predictions. They agree with the checked-in JSON. Checked the frozen
diagnostic prediction hash against its summary. The new report rounds stored
numbers using half-up rounding; the full-precision JSON remains authoritative.

**Evidence:** main neural accuracy is 68.71% without memory and 48.50–48.79%
with memory. The exact reader improves from 75.00% to 82.50%. Its aggregate
decomposes as 75% + 25% × 30%: only 30% of historical-answerable queries retain
their required fact. On similar-but-wrong keys, similarity neural accuracy falls
to 9.67%, versus 88% without memory. All three reader seeds fail the 95% visible
validation gate. Familiar-symbol reading also fails substantially in the frozen
diagnostic. These are distinct storage and reading limitations.

**Inference:** the pilots support memory harm for these readers and workloads,
not a successful learned controller or a universal defect in FIFO/retrieval.
Retention still has room to improve despite the exact reader's aggregate benefit.
The detailed report distinguishes stale-value matching from any non-abstention
on deleted queries, and visible-evidence validation from world-truth scoring.

### Fresh literature review, 2026-09-20

Read the primary abstracts and targeted full-text methods, results, and limitations
of SeDeM v2 (2608.00311), Metis v2 (2607.26760), TARL v2 (2608.03699), MemOps v1
(2607.12893), dependency-guided rollback v1 (2608.10502), and MetaKV v1
(2609.07966). The linked review records source URLs and table-specific scores.
This was a targeted review of supplied papers, not an exhaustive novelty search
or reproduction of their experiments.

- **Evidence/correction:** Metis reports a trained native-memory prototype. The
  map's previous vision/taxonomy classification was wrong. Strong internal
  operation results coexist with weak external scores and memory interference.
- **Evidence/correction:** TARL uses append/noop/revise/reject_conflict/defer_verify.
  A controller with add/replace/delete/defer/ignore would be our adaptation.
  Its Table 2 and Table 3 next-state scores differ; the review keeps their
  contexts explicit rather than silently combining them.
- **Evidence:** SeDeM stores a bank that grows with segments; top-k limits reading,
  not persistent storage. Its reported TTFT includes online compression,
  selection, and decoder prefill. Matched full-bank/raw-retrieval controls show
  that selective reading is not always the highest-quality condition.
- **Inference:** MemOps-style state-transition and collateral-forgetting probes
  can improve our diagnoses with exact synthetic labels. We already report
  stale-value and deletion-leakage metrics; those are not new additions.
- **Inference:** rollback motivates a later derived-fact workload. Diagnosed faults,
  dependency metadata, and replay cannot be assumed to be free. MetaKV stays
  outside the cross-reset semantic-memory comparison.

### Decision and remaining hypotheses

D012 preserves the strict state budget and reader competence milestone while
allowing an independently evaluated writer through the exact-rule reader.
Updated the paper map, experiment plan, handoff, and README links accordingly.
Pending/rejected items, source/version identifiers, and dependencies count as
persistent state. Keep explicit deletion; distinguish writer deferral from
answer abstention. New uncertainty/derived-state workloads must be declared
separately. Four symbolic slots are not expected to equal a 20–50-record world.

**Hypotheses, not measured improvements:** learned record selection plus copying
and UNKNOWN improves reading; action-and-target learning improves bounded writing;
reconstruction/pollution losses help a later latent representation. Test mechanisms
separately. LRU, a learned writer, latent decompression, and the capacity sweep
remain unimplemented or unevaluated here.

### Verification and continuity

- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/pytest`: **40 passed** in 3.79 seconds.
- `git diff --check`: clean.
- All 52 relative links across the two new documents, handoff, plan, and README
  entry points resolve locally.
- This session changes documentation only. Prior uncommitted research notes and
  Project 01 diagnostic artifacts are preserved separately from these changes.
- Review continues on [draft PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2),
  stacked on the still-open Project 01 PR.

**Next action:** predeclare and implement the record-selection/copying reader
comparison with UNKNOWN and paired irrelevant-memory controls on train/validation
data. Specify writer transition metrics separately, using the exact reader to
avoid attributing reading failures to storage decisions.

## 2026-09-20 — Brief proposal for user review before implementation

The user requested a brief evidence-to-experiment document, with a choice of
four, five, or six cases, to analyze before implementation. Created
[`memory-improvements-review-brief.md`](memory-improvements-review-brief.md).
This synthesizes the six primary-paper versions already reviewed today; no fresh
literature search, training, or new experiment was performed in this session.

**Evidence:** the existing pilots distinguish weak reading from 30% historical
useful-fact retention. The paper findings and limitations are linked in the brief;
none is a measured gain for our implementation.

**Inference/recommendation:** five focused cases cover correct recall, rejection
of wrong memory, updates, selective deletion, and capacity pressure. Four cases
omit the capacity study; a sixth would add derived-fact repair and its provenance
requirements. These are experimental questions, not a replacement for the
eight-case historical generator or a five-action architecture.

**Hypotheses awaiting review:** selection/copying and evidence rejection improve
reading; operation-and-target learning maintains lifecycle semantics; admission
and eviction improve usefulness at equal capacity. The proposed additional 95%
reader slice gates are engineering targets for review, not adopted thresholds.
The brief distinguishes an exact-reader writer study from a combined neural result
and notes that uniformly unpredictable future queries may limit admission gains.

Updated the handoff and experiment plan to make the user's review the next step.
D012 remains the standing methodological decision; the proposed scope is not an
approved architecture or training plan. Historical code, metrics, configurations,
and results were not modified. Prior uncommitted work is preserved.

Verification: `.venv/bin/ruff check .` passed; `.venv/bin/pytest` reported **40
passed** in 1.78 seconds; whitespace checks were clean. All 20 relative links
across the brief and its two entry points resolve. Checked that the document
contains all six versioned primary sources and five numbered proposed cases;
reviewed examples, source attribution, control comparisons, and scope consistency.
The document is part of [draft PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2).

**Next action:** the user reviews the brief and selects the scope. After that
review, resolve requested changes and write the detailed implementation plan
before beginning model or benchmark implementation.

## 2026-09-20 — Approved five-case implementation, before held-out evaluation

**Authorization:** the user approved the five-case proposal with oracle controls,
rejection strata, dual reader gates, predictable/unpredictable capacity workloads,
and retention regret, then requested implementation and testing of every case.
Created an isolated `feat/five-case-memory` worktree; the original checkout's
uncommitted research notes and Project 01 artifacts were left untouched.

**Implementation:** separate reader selection/copy/generation arms, a learned
feature-assisted lifecycle controller, causal delayed-reward capacity training,
exact controls, fixed-state baselines, metrics, saved predictions/checkpoints,
and a combined diagnostic. See D013, the locked protocol, and execution plan.
No additional literature search was needed; paper interpretations are unchanged.

**Evidence from development:** an independent review caught a generator shortcut:
operation-kind tokens alone initially identified the relevant current fact.
Added mixed unrelated current/memory operations and query-blind regression rules;
those shortcuts now fail 55–100% in the four rejection strata. This was fixed on
train/validation before the recorded experiment. An erroneous writer can produce
an occupied key with value -1; its reader projection now preserves it as DELETE
evidence rather than crashing or repairing the decision. Review also caught a
microtask-only relative gate; episode controls now have their own required gate.

**Inference:** tests establish instrumentation behavior, not successful learned
memory. The cue-only learned policy has no information advantage over the
cue-priority control. **Hypothesis:** it may learn that signal and improve useful
retention in B without systematic gains in A. Full held-out results are pending.

**Next actions:** run the checked-in CPU smoke, commit the protocol/source/config,
train seeds 7/19/43 before creating any held-out data, finish all evaluations, and
reconcile saved predictions before reporting results. Preserve failed gates.

Pre-freeze verification: **95 tests passed** in 2.43 seconds; Ruff and whitespace
checks passed. The checked-in five-step CPU smoke completed every case, all
policies and both capacities. Its learned gates failed as expected for untrained
models; the oracle controls were exact. This is pipeline verification only.

## 2026-09-20 — All five cases executed and independently audited

**Evidence:** frozen source/config commit `4765f2f8295ce1d225638a38030ab6e85a985a2f`
ran seeds 7/19/43 on an Apple M2 Pro, 16 GiB, Python 3.13.7, PyTorch 2.9.1, with
MPS readers and CPU controllers. Wall time: 736.83 seconds. Full artifacts are
preserved at `runs/five-case-2026-09-20-v1` in the five-case worktree. All training
completed before held-out generation. No test-based model/generator/metric
changes followed.

Mean reader accuracy: character 39.13%, selected generation 85.03%, selected copy
92.83%, oracle copy 100%. Every seed fails reader gates; seed 43 exceeds 95%
overall but fails required categories and recovery. Controlled writer update,
deletion, control, action, target, and transition metrics are 100% in every seed,
with zero stale/deleted leakage. Features include exact key equality. B learned
retention is 27.91%/57.18% at four/eight slots versus FIFO 15.41%/31.98%, exactly
matching cue-priority. Combined neural B recall of 25.63%/51.35% remains diagnostic.
The full report explains denominators, failed gates, abstention, regret, per-seed
variation, traces, and compute.

**Unexpected evidence:** uniform A8 learned/FIFO difference is +3.08 points with
95% paired interval [0.76, 5.64]. A4's interval crosses zero. Preserve both; the
observed A8 contrast is not evidence that a cue-independent future becomes
predictable. Potential sample variation or generator structure requires an
independent preregistered episode replication. All seeds share evaluation episodes.

**Verification:** the independent auditor, without experiment metric imports,
passes 237 artifact hashes and 25 source hashes and recalculates scores from
405,696 reader rows, 33,600 lifecycle operation rows, 12,288 capacity banks, and
46,080 combined writes. Those are repeated control/policy/seed records, not
independent examples. Tamper checks reject altered metrics/checkpoint hashes
and incomplete runs. Source review verified all episode recovery gates, overwrite
refusal, and four/eight-slot adapter trajectories. The suite has 95 passing tests.
Figures and compact results are checked in; raw checkpoints/predictions remain
local (~362 MiB). No fresh literature search; paper interpretations are unchanged.

**Inference:** learned retention recovers the planted cue rule. Controlled writer
competence and reader failure are now experimentally separable. A perfect exact
reader over good banks still exceeds neural answers. This does not establish
superiority over the strongest heuristic, representation learning, calibrated
source trust, or the original latent-memory hypothesis. Capacity paired harm of
zero has no safety meaning because its no-memory control has 0% accuracy.

**Next action:** follow D014: train/validation reader diagnosis and a separately
predeclared uniform-workload replication. Preserve this run and original-checkout
uncommitted notes. See the [five-case report](five-case-results-2026-09-20.md) and
[reproduction commands](../projects/03-memory-reliability/README.md).

Final verification: `pytest` passed all 95 tests; `ruff check .` and
`git diff --check` passed. All 69 checked local document links resolve. The three
standalone figures were rendered and visually checked. An independent read-only
review reconciled the report/README numbers and scientific caveats with saved
results and found no remaining corrections. Frozen experiment source, tests, and
configuration are unchanged from `4765f2f`.

Review handoff: [draft PR #3](https://github.com/narender2031/bounded-memory-transformer/pull/3)
is open against `feat/synthetic-memory-benchmark`; no base PR was merged. Hosted
CPU lint and tests [passed at `116c1f2`](https://github.com/narender2031/bounded-memory-transformer/actions/runs/35499085042).
Generated SVG whitespace was normalized in the exporter and saved figures after
the staged-file check caught it; the complete branch whitespace check now passes.
The original checkout has its original dirty status; the isolated worktree holds
the completed implementation and results.

## 2026-09-20 — Reader repair, uniform replication and emergent-utility design

**Authorization and isolation:** the user requested a reader fix toward both 95%
gates, replication of the uniform negative control, and design of Case 5B emergent
utility. Continued in the isolated five-case worktree on new branch
`feat/reader-reliability-followup`. Original-checkout dirty work and both historical
experiment packages remain unchanged. Independent agents handled the uniform
replication, prior-art/design track, and read-only method/result audits.

**Development evidence and failures:** a frozen-reader train/validation diagnosis
identified wrong-entity/same-attribute selection as the dominant error (506 of
533 errors in the three-seed fresh four-slot validation sample). A factorized
reader compares field characters through a small manually implemented Transformer.
Candidate 1's perfect seed-7 validation was invalid as coverage evidence because
the new generator correlated occupancy with query category. Corrected this before
test and added cross-occupancy checks. Candidate 2 reached 98.5% aggregate for seed
7 but failed its unsupported category; other seeds passed. Candidate 3 pretrained
and froze the comparator, passing the earlier validation suite, but an additional
nine-transition authority probe scored only 66.67%. Soft operation confidence
could outrank a later authoritative value. Candidate 4 added saturation/transition
training yet still scored only 66.67–77.78% on that probe. Its unchanged weights
with a fixed binary classifier/chronology adapter passed all 12 validation
conditions per seed. No inference exact-key repair or threshold search was added.

The initial three-candidate budget was explicitly amended before reader test
opening for a fourth training variant and a fifth readout variation. The final
adapter separates eligibility from chronological ranking using deterministic
composition of learned classifier outputs. It differs from the soft training
path; auxiliary pair supervision covers the training alphabet. Preserve every
failure and training history. [Protocol](reader-followup-protocol-2026-09-20.md),
[development archive](../projects/04-memory-followup/results/2026-09-20-v1/development.json).

**Reader held-out evidence:** source/config/protocol freeze `1845235` preceded
generation, and all three final models had already trained. Seeds 7/19/43 each
score 100% visible-evidence accuracy and raw oracle-relative recovery on all 41
conditions (123 evaluations), with no unavailable full-run headroom. The 29,600
shared reader microtasks cover original, varied occupancy/current counts, and
all nine authoritative transitions; each seed additionally has 29 episode controls.
All 437,664 saved rows match the exact visible oracle. There are zero paired
harmful answers on this suite. Matched original four-slot old-reader scores are
92.38/90.66/95.18%, averaging 92.74%; all final seeds score 100%.

**Reader limits:** 437,664 is a repeated model/policy row count, not independent N.
Fresh seeds reuse the existing held-out full-symbol partition and familiar
character alphabet. The structured 7,639-parameter classifier, supervision,
composition and copy aids are part of the method. Its success is instrumentation,
not a free-form reader or learned trust contribution. Fresh planted-cue B world
recall remains 28.61%/57.69% with exact lifecycle/learned retention, despite perfect
reading of retained evidence. A discarded live fact can correctly elicit visible
UNKNOWN while losing world recall. An unavailable superseding event cannot be
detected from the old fact alone. Capacity-only zero paired harm has no safety
meaning when current-only world accuracy is zero.

**Independent uniform evidence:** preregistration freeze `569c387` preceded the
unchanged historical generator's 20 dataset seeds, 512 episodes each, 24 keys and
32 uniform queries per episode. Frozen learned checkpoints have identical greedy
cue decisions over 544 exhaustive allocations/model. Learned recall is
16.6956%/33.3240% for K4/K8, versus FIFO 16.7047%/33.3630%. All six learned-minus-
FIFO/recency/random intervals contain zero and lie inside ±1 percentage point,
under both multiplicity-corrected analytic and 10,000-replicate paired bootstrap
methods. The predeclared practical-equivalence criterion is met. The prior
+3.08-point A8 result did not replicate; its original positive interval is preserved.
**Inference:** this supports sampling variation rather than an ability to predict
uniformly unpredictable demand. It is not a test of emergent access-based utility.
See [uniform protocol](uniform-replication-protocol-2026-09-20.md) and D016.

**Compute and independent verification:** M2 Pro, 16 GiB, Python 3.13.7, PyTorch
2.9.1. Final reader training across three seeds took 246.93 s; reader evaluation
115.73 s; CPU uniform replication 189.96 s. Earlier development/audit costs are
additional, and concurrent activity makes these observational timings. Uniform
learned write totals 7.77/6.71 s versus FIFO 2.75/2.60 s at equal 64/128 bytes.
Reader audit passes 211 artifact hashes, 54 source/checkpoint/config hashes,
12 checkpoints and 437,664 prediction rows; it independently reconstructs
semantics/metrics/gates without rerunning all neural forwards. Uniform audit
passes 263 artifact hashes, replaying 122,880 banks and checking 3,932,160 query
records and all bootstrap replicates without project imports. Its exact audit
program is embedded in the saved audit. Later auditor checks/tests strengthened
source/config binding without changing frozen experiment code or outcomes.

Commands and immutable run locations are in
[Project 04](../projects/04-memory-followup/README.md). Compact summaries, audits,
source manifests, development failures and source-backed PNG/SVG are checked in.
The revised two-panel figure was visually inspected; its numbers equal saved
prediction rows and uniform summaries. Local `pytest`: **156 passed in 2.97 s**;
`ruff check .`: passed. Final document-link/whitespace checks and PR follow below.

**Literature evidence, search/read date 2026-09-20:** primary ARC (FAST 2003),
TinyLFU (v2), LRB (**NSDI** 2020), and LeCaR (HotStorage 2018) already establish
access-based utility and recency/frequency adaptation. Rechecked SP-KV/Supersede
abstracts only. Read details in the [prior-art note](reading-notes/2026-09-20-emergent-utility-prior-art.md).
Conventional refill, ghost identities, sketches and delayed-feedback histories
change the information/state contract. This was a targeted check, not a novelty
proof. Updated the literature map and decisions D015–D017.

**Design and next hypothesis:** [EU1](case-5b-emergent-utility-design.md) proposes
interleaved accesses with no ASK refill, causal bounded summaries, strong independent
heuristics (including a charged TinyLFU-style sketch), unseen temporal mechanisms,
five model seeds, and joint superiority/harm gates. It uses exact lifecycle guards
to isolate retention. A feasible small-panel oracle replaces the invalid top-K
future-count bound for interleaving. **No EU1 implementation or results exist.**
Review this design, then implement its generator/serializer/baselines on development
data. The hypothesis that learned retention beats strong heuristics under shift
remains open. No completed test is to be retuned or overwritten.

Final report review independently reconciled all material numbers with raw and
compact artifacts and found no scientific overclaims. Corrected two reproduction
wording details: only retention checkpoints are pinned by the uniform config,
and audit JSON outputs need fresh paths because the auditor can replace them.
All **89 local document links** resolve; six compact raw-artifact copies match
their source bytes/hashes; whitespace checks pass. The final auditor SHA is
`aaef204d727343a7565edb93defbea325bb817ad7f9a2ac7eb62db06f6ca50c1`.
Frozen reader source/config/protocol are unchanged from `1845235`; both historical
experiment packages are unchanged from `4765f2f`. The original checkout retains
its pre-session dirty status. The standalone figure was visually checked after
moving its equivalence-margin legend out of the plotted intervals.
