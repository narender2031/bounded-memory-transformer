# Research Exploration Map

Last updated: 2026-09-12

This file is the navigation layer for exploring the bounded-memory Transformer problem. It explains what has been investigated, what is already established, what remains uncertain, and where the detailed evidence lives.

## Core question

Can a small Transformer learn what information to store, retrieve, update, replace, forget, and ignore across hard context resets without rereading its raw history?

The target is not generic long context. The target is a tiny, editable persistent state that remains useful when more candidate facts arrive than the memory can hold.

## Problem decomposition

| Capability | Required behaviour | Main failure to measure |
|---|---|---|
| Admission | Store information likely to matter later and ignore noise. | Useful facts are never written or noise consumes capacity. |
| Retrieval | Read the correct slot for the current question. | A relevant stored fact is missed. |
| Supersession | Route an update to the existing semantic key and expose the newest value. | A stale value is returned. |
| Deletion | Invalidate a fact without leaking its former value. | Deleted-fact leakage. |
| Eviction | Remove the least useful item when all slots are occupied. | A future-useful fact is evicted. |
| Abstention | Return unknown when a fact is missing or invalid. | Hallucinated answers or excessive abstention. |
| Persistence | Carry only bounded state across a reset. | Hidden replay or KV-cache leakage creates false success. |
| Generalisation | Preserve the learned policy for unseen symbols and longer episodes. | Memorising surface patterns instead of learning memory control. |

## Hard research boundary

Main results must use:

- 4 persistent slots initially, with later sweeps over 1, 2, 4, 8, and 16;
- 20 candidate facts initially;
- 8 sessions per episode;
- deletion of the token context and standard KV cache between sessions;
- no retrieval or replay of raw historical text;
- equal capacity for comparable policies;
- unseen entity and value symbols in the main generalisation evaluation;
- updates, deletions, distractors, and unanswerable questions.

These rules prevent long context, hidden replay, or a larger memory budget from being mistaken for learned persistent memory.

## Research sources

| File | Purpose |
|---|---|
| [CODEX.md](CODEX.md) | Complete conversation handoff, project state, constraints, and immediate next task. |
| [research/problem-statement.md](research/problem-statement.md) | Formal research question, scope, hypotheses, controls, and meaningful-result criteria. |
| [research/papers.md](research/papers.md) | Curated prior-art map and reading order. |
| [research/experiment-plan.md](research/experiment-plan.md) | Dataset, baselines, architecture, curriculum, metrics, and falsification plan. |
| [research/decisions.md](research/decisions.md) | Durable methodological decisions that future work must preserve. |
| [research/log.md](research/log.md) | Dated evidence, inferences, implementations, failures, and next actions. |
| [research/reading-notes/template.md](research/reading-notes/template.md) | Required structure for detailed paper notes. |
| [PR1.md](PR1.md) | Pull request #1 contents, validation evidence, and reproducibility commands. |

## What prior work already establishes

The following ideas are not novel by themselves:

- carrying learned memory tokens between segments;
- maintaining a fixed latent memory pool;
- updating neural memory during inference;
- using surprise, attention, recency, or predicted utility for retention;
- retrieving historical key/value activations;
- managing textual memory through external CRUD actions;
- evaluating multi-session recall, temporal reasoning, updates, and abstention.

Relevant families include Transformer-XL, Compressive Transformer, RMT, MemoryLLM, Titans, MIRAS, SP-KV, GradMem, Supersede, LiveMem, Memorizing Transformers, Infini-attention, EXPIRE-SPAN, H2O, and LongMemEval.

See [research/papers.md](research/papers.md) for paper links and method-level comparisons.

## Provisional research gap

The defensible hypothesis is that a joint learned policy for future-utility admission and semantic supersession may outperform FIFO, LRU, recency, and similarity policies when:

- capacity is extremely small;
- source context is deleted;
- facts change or are deleted;
- raw-history fallback is prohibited; and
- the policy is evaluated on unseen symbols and longer episodes.

This is a hypothesis, not a novelty claim. It is weakened if a simple equal-capacity baseline matches the learned controller.

## Experimental ladder

1. Build and verify an inspectable no-memory Transformer.
2. Build a deterministic symbolic state-tracking benchmark.
3. Validate the reference state machine and memory-aware metrics.
4. Implement no-memory, full-history, FIFO, LRU, recency, similarity, and oracle baselines.
5. Add RMT-style recurrent memory tokens.
6. Add learned admission only.
7. Add learned admission plus semantic supersession and deletion.
8. Run capacity, distractor, overwrite, distance, and generalisation sweeps.
9. Perform component and loss ablations.
10. Adapt successful mechanisms to paraphrased natural language and later LongMemEval-style tasks.

Project 01 is complete on this branch. Project 02—the deterministic benchmark—is next.

## Minimum reporting table

Every experiment must report:

- overall latest-value accuracy;
- SET, UPDATE, DELETE, and unknown accuracy;
- stale-answer rate;
- deleted-fact leakage;
- abstention precision, recall, and F1;
- useful-fact retention;
- accuracy by evidence distance and overwrite count;
- memory-state bytes;
- historical tokens reprocessed;
- training and inference latency;
- seeds and uncertainty.

## Research discipline

- Prefer primary papers and official author code.
- Compare new ideas with the closest mechanism, not only older famous baselines.
- Do not infer novelty from an incomplete search.
- Keep the generator independent from model and policy implementations.
- Do not change held-out data generation after inspecting results.
- Record negative results.
- Append every research session to `research/log.md`.
- Record durable choices in `research/decisions.md`.
