# Codex Project Handoff

Last updated: 2026-09-20

This document transfers the working context from the original ChatGPT research conversation into the repository. A new Codex session should begin here, then follow the linked source-of-truth documents.

## Mission

Build and evaluate a small Transformer that learns what information to store, retrieve, update, replace, forget, and ignore across hard context resets without rereading its raw history.

The narrow research question is:

> Can learned memory admission plus semantic supersession outperform hand-designed bounded-memory policies under a strict fixed capacity?

This is not a project to solve every limitation of Transformers. It focuses on reliable, editable persistent state under severe capacity pressure.

## Non-negotiable experimental contract

- Use a tiny fixed memory: initially 4 slots, later sweep 1, 2, 4, 8, and 16.
- Present more candidate facts than fit in memory: initially 20 facts across 8 sessions.
- Delete the token context and normal attention KV cache between sessions.
- Preserve only the explicitly measured bounded-memory state.
- Do not retrieve or replay raw historical text in the strict condition.
- Train and test on disjoint entity and value symbols for the main generalisation split.
- Include updates, deletions, distractors, and intentionally unanswerable queries.
- Compare policies at equal slot count and representation width.
- Record seeds, configurations, state size, latency, and historical tokens reprocessed.
- Treat evidence, inference, and hypotheses separately.
- Report negative results and do not tune the held-out generator after inspecting test results.

## Current scientific position

The broad area is active and already contains recurrent memory, compact latent memory, test-time learning, learned KV retention, and external memory agents. We must not claim that bounded memory or learned storage is itself novel.

The provisional research gap is the combination of:

1. learned future-utility selection;
2. reliable semantic supersession and deletion;
3. a very small latent state;
4. hard context resets;
5. no raw-history fallback; and
6. generalisation to unseen symbols and longer episodes.

This remains a hypothesis to test, not a novelty claim.

## Prior art that constrains the design

Read the complete curated map in [research/papers.md](research/papers.md). Start with:

1. Recurrent Memory Transformer — learned memory tokens across segments.
2. MemoryLLM — self-updating fixed latent memory.
3. Titans and MIRAS — test-time neural memory and its design space.
4. SP-KV — future-utility prediction for KV retention.
5. GradMem — compact learned memory tested after source context removal.
6. Supersede — bounded memory under changing facts.
7. LiveMem — state that survives context turnover and affects later behaviour.
8. LongMemEval — updates, temporal reasoning, and abstention evaluation.
9. Metis — trained native memory with forward-only online updates and interference diagnostics.
10. TARL — executable action/target decisions over memory lifecycle transitions.

Compare mechanisms directly: SP-KV for utility-based admission, Supersede for
stale-fact replacement, TARL for executable lifecycle control, and Metis for
native-state maintenance. The [six-paper review](research/reading-notes/2026-09-20-six-paper-review.md)
also covers SeDeM reader selection, MemOps diagnostics, dependency repair, and
MetaKV. Metis is an empirical prototype; its earlier vision-only classification
was incorrect. None of these papers establishes novelty for our proposed system.

## Work completed

### Phase 0 — research specification

The repository contains:

- [research/problem-statement.md](research/problem-statement.md)
- [research/papers.md](research/papers.md)
- [research/experiment-plan.md](research/experiment-plan.md)
- [research/decisions.md](research/decisions.md)
- [research/log.md](research/log.md)
- [research/reading-notes/template.md](research/reading-notes/template.md)

### Project 01 — trusted Transformer baseline

PR [#1](https://github.com/narender2031/bounded-memory-transformer/pull/1) contains a decoder-only Transformer built from inspectable PyTorch components without `nn.Transformer` or `nn.MultiheadAttention`.

Implemented:

- character tokenizer;
- deterministic next-token batcher;
- token and positional embeddings;
- manual multi-head Q/K/V causal attention;
- pre-normalised residual decoder blocks;
- feed-forward network;
- next-token loss;
- training CLI and autoregressive generation;
- shape, masking, causality, optimizer, tokenizer, data, and generation tests;
- GitHub Actions using CPU-only PyTorch.

Verification:

- 11 tests pass;
- lint is clean;
- the 27,520-parameter CPU smoke model reduced validation loss from 3.5936 to 2.7913 over 50 steps;
- hosted GitHub Actions passed.

Project 01 intentionally contains no persistent memory. It is the no-memory control model.

## Project 04 — reader gates and independent uniform replication completed

The user authorized reader repair, independent uniform replication, and design
of Case 5B emergent utility. All three tasks are complete. See the
[detailed results](research/reader-followup-results-2026-09-20.md),
[reproduction commands](projects/04-memory-followup/README.md), and
[Case 5B design](research/case-5b-emergent-utility-design.md).

- Reader source/config/protocol freeze: `1845235`. Seeds 7/19/43 all pass both
  95% gates in all 41 conditions: 100% visible accuracy and 100% raw oracle-relative
  recovery, matching the oracle on all 437,664 saved rows. Repeated model/policy
  rows are not independent samples. Matched old four-slot accuracy is 92.74%.
- This 7,639-parameter reader learns field matching and authority, with auxiliary
  equality labels over the training alphabet, a frozen comparator, deterministic
  conjunction/ranking composition and exact copying. Final binary readout uses
  candidate-4 weights but differs from the soft training path. No exact-key repair.
  Preserve all four training variants, failed authority probes, and the pre-test
  readout amendment. No held-out retuning occurred.
- Zero observed paired harm applies to authoritative symbolic inputs; discarded
  facts still reduce world recall. Plausible stale facts cannot be detected when
  their newer evidence is unavailable. General trust/uncertainty is unproved.
- Uniform preregistration freeze: `569c387`; 20 fresh dataset seeds × 512 episodes.
  All six learned-minus-FIFO/recency/random contrasts satisfy the predeclared
  ±1-point equivalence criterion with corrected analytic and bootstrap intervals.
  The eight-slot learned/FIFO difference is −0.0391 points [−0.3985, +0.3204].
  The earlier +3.08-point result did not replicate and remains in the old report.
- Independent audits pass: reader 211 artifact hashes and 437,664 rows; uniform
  263 artifact hashes, 122,880 bank replays and 3,932,160 saved query records.
- `case_5b_emergent_utility_v1` is **design only**, distinct from planted-cue B.
  It specifies no-refill interleaved access, fully charged state, strong caching
  baselines, held-out mechanisms, and a demanding multi-seed superiority gate.
  There is no emergent-utility model, result, or novelty claim yet.

Worktree: `/Users/narendersingh/transformer/bounded-memory-transformer-five-cases`,
branch `feat/reader-reliability-followup`. Immutable local runs:
`runs/reader-followup-2026-09-20-v1`, `runs/uniform-replication-2026-09-20-v1`;
development: `runs/reader-followup-development`. Compact checked-in evidence:
`projects/04-memory-followup/results/2026-09-20-v1`.
Historical experiment packages and the original checkout's dirty work are unchanged.

Review: [draft PR #4](https://github.com/narender2031/bounded-memory-transformer/pull/4),
stacked on #3. Local verification: 156 tests, Ruff, whitespace and 89 document links
pass. The PR exposes the current hosted CPU-check status. No base PR is merged.

## Project 03 — five cases implemented and tested

The user approved all five cases and authorized implementation and local testing.
Source/config were frozen at `4765f2f`; the three-seed M2 Pro run completed in
736.83 seconds. See [measured results](research/five-case-results-2026-09-20.md)
and [Project 03 reproduction](projects/03-memory-reliability/README.md).

- Reader means: character 39.13%, select/generate 85.03%, select/copy 92.83%,
  oracle/copy 100%. **Every seed fails the full absolute and relative gates.**
- Controlled four-slot lifecycle: all action/target/transition/update/delete/control
  outcomes 100%; stale/deleted leakage zero. Equality features are provided.
- Predictable B retention: learned 27.91%/57.18% at four/eight slots versus FIFO
  15.41%/31.98%, exactly matching the planted cue-priority heuristic.
- Uniform A8 also has a positive observed learned/FIFO paired interval. Do not
  claim that this run confirms the uniform null or demonstrates future prediction.
- Combined B learned reading: 25.63%/51.35%; **diagnostic and unvalidated**.
- 95 tests pass. Independent audit passes 237 artifact and 25 source hashes and
  recomputes all reported metrics from saved query/write/bank records.

Review: [draft PR #3](https://github.com/narender2031/bounded-memory-transformer/pull/3),
stacked on #2. Hosted CPU lint/tests passed.
Full artifacts: `runs/five-case-2026-09-20-v1` in the isolated five-case worktree.
No held-out retuning. Original-checkout uncommitted work remains untouched.

## Project 02 — preserved pilots and original specification

**Current status:** The four-baseline Phase 1 implementation and two local runs
are available in [draft PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2)
on `feat/synthetic-memory-benchmark`; see
[`projects/02-memory-benchmark/README.md`](projects/02-memory-benchmark/README.md).
The corrected second run uses source revision `280d88c`, three seeds, 30
candidate operations, eight sessions, and four symbolic slots (64 logical bytes).
Mean neural accuracy: no memory 68.71%, FIFO 48.50%, recency 48.58%, lexical
similarity 48.79%. The exact-key reader of the same evidence scores 75.00%
without memory and 82.50% with memory. Neural validation is only 73.83–78.52%,
so the 95% competence gate fails. These are weak-reader pilot results, not a
validated learned-memory contribution. Preserve both runs and their limitations.

The subsequent frozen-reader diagnostic and primary-source comparison are in
[`research/reading-notes/2026-09-17-memory-improvements.md`](research/reading-notes/2026-09-17-memory-improvements.md).
On 2,048 matched reading structures, mean accuracy was 78.91% with seen entity/value
symbols and 76.06% with both unseen. This is exploratory train/validation analysis,
not a new held-out policy result. It motivates a record-selection/copying reader
ablation with an UNKNOWN option, now measured separately in Project 03 above.
The [Transformer fundamentals guide](research/transformer-fundamentals.md) explains
the actual architecture, training, memory boundary, and reader terminology.

The [detailed results summary](research/results-summary-2026-09-20.md) independently
checks the saved predictions and explains the denominators. Main useful-fact
retention is only 30% on historical-answerable queries: the exact reader's 82.5%
overall accuracy is 75% + 25% × 30%. Both retention and neural reading need work.
The 2026-09-20 review added no new training or controller results. D012 allows
an independent writer study using the exact-rule reader, while retaining the
competence gate for a combined neural writer/reader claim. Pending evidence,
provenance, and dependency metadata must consume the same total state budget.

The remaining text below preserves the broader Project 02 specification;
it predates Projects 03–04. Interleaved LRU remains future work; Project 04 now
establishes the declared symbolic reader competence with disclosed structural aids.

Build the deterministic synthetic memory benchmark before adding a learned memory controller.

### Symbolic operations

```text
SET entity attribute value
UPDATE entity attribute value
DELETE entity attribute
NOISE entity attribute value
ASK entity attribute
```

Semantics:

- `SET`: create a current fact.
- `UPDATE`: replace the current value for an existing entity/attribute key.
- `DELETE`: invalidate the current fact.
- `NOISE`: provide information that should not affect the queried state.
- `ASK`: query the latest valid value; return an explicit unknown token when absent or deleted.

### Initial episode configuration

- 8 sessions per episode.
- 1–4 operations per session.
- Hard reset after every session.
- 20 candidate facts.
- 4 persistent slots.
- At least one update in 50% of episodes.
- At least one unknown or deleted query in 25% of episodes.
- Distractor rate varied independently of episode length.

### Baselines to implement first

| Baseline | Purpose |
|---|---|
| No memory | Prove that the hard reset removes required evidence. |
| Full history | Accuracy reference with unbounded replay; not a strict-condition competitor. |
| FIFO | Test insertion-order eviction. |
| LRU | Test access-recency eviction. |
| Recency | Retain the newest facts. |
| Similarity | Retain or retrieve by key/query similarity. |
| Oracle utility | Upper bound with knowledge of the future query. |
| Oracle semantic update | Upper bound with correct routing of updates and deletes. |

RMT-style memory tokens, learned admission, and learned admission plus supersession come only after the generator, semantics, baselines, and metrics are independently validated.

### Required metrics

- latest-value accuracy;
- accuracy split by SET, UPDATE, DELETE, and unknown cases;
- stale-answer rate after updates;
- deleted-fact leakage rate;
- abstention precision, recall, and F1;
- useful-fact retention;
- accuracy versus sessions since evidence;
- accuracy versus overwrite count;
- persistent-state bytes;
- historical tokens reprocessed;
- training and inference latency.

Average accuracy alone is not acceptable because it can hide stale answers and deletion failures.

### Generalisation tests

- unseen entity symbols;
- unseen value symbols;
- longer episodes;
- more updates per key;
- more candidate facts at fixed capacity;
- later: paraphrased natural-language rendering.

## Project 02 definition of done

Project 02 is complete when:

1. episode generation is deterministic for a recorded seed;
2. a reference state machine proves the correct answer for every ASK;
3. train, validation, and test symbol spaces are disjoint where required;
4. all bounded baselines see identical episodes and equal capacity;
5. hard reset and no-history-access are asserted in tests;
6. update, deletion, distractor, and abstention cases have targeted tests;
7. the required metrics have tests with hand-calculated examples;
8. a checked-in configuration reproduces the first baseline table;
9. results and failures are appended to `research/log.md`;
10. any durable methodological choice is recorded in `research/decisions.md`.

## Recommended repository shape for Project 02

```text
projects/02-memory-benchmark/
├── README.md
└── configs/
    └── baseline-small.toml

src/bounded_memory_transformer/memory_benchmark/
├── __init__.py
├── operations.py
├── generator.py
├── state_machine.py
├── policies.py
├── metrics.py
└── evaluate.py

tests/
├── test_episode_generator.py
├── test_state_machine.py
├── test_memory_policies.py
└── test_memory_metrics.py
```

Keep the generator independent from the models and policies. A bug in the generator must not be able to make one policy look better than another.

## Codex working protocol

At the beginning of a session:

1. Read this file.
2. Read [AGENTS.md](AGENTS.md).
3. Inspect the latest branch and pull-request state.
4. Read the latest entries in `research/log.md` and `research/decisions.md`.
5. State the specific hypothesis or engineering milestone for the session.

During work:

- use small, inspectable PyTorch components;
- do not use high-level Transformer wrappers during the learning phase;
- add shape and semantic assertions;
- keep hand-designed baselines independent of learned code;
- validate locally before publishing;
- place changes on a feature branch and open or update a reviewable PR.

At the end of every session:

1. run lint and tests;
2. record commands, configurations, results, failures, and next actions in `research/log.md`;
3. update `research/decisions.md` when a choice affects later experiments;
4. link the PR or commit;
5. leave one unambiguous next action.

## Immediate next action

The authorized reader repair, uniform replication, and Case 5B design are complete.
Review the [EU1 design](research/case-5b-emergent-utility-design.md). Its first
implementation milestone is the deterministic generator, canonical byte serializer,
and independent FIFO/recency/LRU/LFU/decayed-LFU/hybrid/TinyLFU-style baselines with
development-only causal and capacity tests. The current request authorized design
of this new workload; no EU1 training or held-out testing has occurred.

Do not retune the completed reader or uniform study after their test opening.
Preserve historical failed gates, original positive A8 interval, all development
variants, and raw runs. Utility learning needs a separate executable freeze and
the declared distribution-shift and harm evaluation. Do not automatically add
latent compression or dependency rollback. Keep the original checkout's
uncommitted work and PR #1/#2/#3 intact.

## External research cadence

A scheduled research task runs every Monday at 9:00 PM Asia/Kolkata to find new bounded-memory papers, avoid duplicates, compare them with the existing map, and identify concrete implications for the next experiment. Relevant findings must still be reviewed before being added to `research/papers.md`.
