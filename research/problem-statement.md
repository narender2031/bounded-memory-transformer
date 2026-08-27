# Problem Statement

Last updated: 2026-08-27

## Target problem

A standard Transformer produces an output from its current context:

\[
y_t = f_\theta(x_t)
\]

When the context and KV cache are discarded, no episode-specific state remains. We want to study a stateful model:

\[
(y_t, M_{t+1}) = f_\theta(x_t, M_t)
\]

where:

- \(x_t\) is the current session or context chunk.
- \(M_t\) is a fixed-size persistent memory state.
- \(y_t\) is the model output.
- \(M_{t+1}\) is the updated memory.
- \(\theta\) remains fixed during ordinary inference.

After every session, token activations and the normal KV cache are deleted. Only \(M_{t+1}\) survives.

## Required memory operations

The model must learn five distinct behaviours:

1. **Admit:** decide whether incoming information deserves storage.
2. **Retrieve:** locate memory relevant to the current query or update.
3. **Supersede:** replace or invalidate an older value for the same semantic key.
4. **Evict:** select what to remove when capacity is exhausted.
5. **Abstain:** recognise when the answer is unavailable.

## Minimal example

```text
Session 1: SET Ava.office Room_12
<reset>
Session 2: NOISE wall colour_white
<reset>
Session 3: UPDATE Ava.office Room_27
<reset>
Session 4: ASK Ava.office
Expected: Room_27
```

An answer of `Room_12` is specifically a **stale-memory error**, not a generic retrieval error.

## Research gap being tested

Prior work has demonstrated persistent recurrent states, latent memory pools, learned retention, test-time memory updates, and external memory agents. Recent work also directly studies future-utility prediction and fact supersession.

The narrower unresolved combination we will test is:

> Learned future-utility selection plus reliable semantic supersession in a very small latent memory, across hard context resets, without a raw-history fallback.

This is a provisional gap based on the literature reviewed in `papers.md`. It must be rechecked before any novelty claim.

## Hypotheses

### H1: Learned admission

A learned admission gate trained through downstream answer utility will retain more useful facts than FIFO, LRU, recency, or accumulated-attention policies under an equal slot budget.

### H2: Supersession-aware writes

Separating “match an existing semantic key” from “allocate a new slot” will reduce stale-answer errors compared with a single undifferentiated write gate.

### H3: Explicit validity state

Adding validity/version metadata to each memory slot will improve `UPDATE` and `DELETE` handling compared with unconstrained vector blending.

### H4: Generalisation

The policy can generalise to unseen entity and value symbols if it learns operations rather than memorising surface forms.

## Strict evaluation contract

Every main result must satisfy all of the following:

- Fixed memory capacity during an episode.
- Identical capacity for comparable baselines.
- Hard context and KV-cache reset between sessions.
- No retrieval from raw historical text in the strict condition.
- Disjoint entity and value vocabularies for the main generalisation split.
- Updates, deletions, distractors, and intentionally unanswerable queries.
- Multiple seeds with uncertainty reported.

## Primary metrics

- Latest-value accuracy.
- Stale-answer rate.
- Deleted-fact leakage rate.
- Abstention precision, recall, and F1.
- Useful-fact retention at each memory capacity.
- Accuracy as a function of sessions since last evidence.
- Accuracy as a function of update count.
- Peak memory state size and historical tokens reprocessed.
- Training and inference latency.

Average question-answer accuracy alone is insufficient because it hides stale and hallucinated answers.

## What would count as a meaningful result?

A learned method should:

- Beat hand-designed policies at the same capacity.
- Preserve gains on unseen entities and longer episodes.
- Reduce stale answers rather than merely increasing overall recall.
- Demonstrate that evicted source tokens influence later behaviour through the persistent state.
- Show which architectural component caused the gain through ablation.

A complex model beating an intentionally weak baseline would not be meaningful.
