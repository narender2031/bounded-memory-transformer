# Codex Project Handoff

Last updated: 2026-09-12

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

The closest overlap with our proposed contribution is SP-KV for utility-based admission and Supersede for stale-fact replacement. Any future claim must compare directly with both mechanisms.

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

## Next implementation: Project 02

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

1. Review and merge PR #1.
2. Create `feat/synthetic-memory-benchmark` from the updated main branch.
3. Implement the typed operation schema and the reference state machine.
4. Add unit tests for SET, UPDATE, DELETE, NOISE, ASK, stale-value prevention, deletion, and unknown answers.
5. Only then implement the deterministic episode generator and hand-designed baselines.

## External research cadence

A scheduled research task runs every Monday at 9:00 PM Asia/Kolkata to find new bounded-memory papers, avoid duplicates, compare them with the existing map, and identify concrete implications for the next experiment. Relevant findings must still be reviewed before being added to `research/papers.md`.
