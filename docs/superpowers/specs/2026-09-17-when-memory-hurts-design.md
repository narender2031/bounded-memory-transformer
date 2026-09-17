# Experiment 01 — When Memory Hurts

The user approved implementing and running Phase 1 locally. This is a new
benchmark subsystem, following `CODEX.md` and the user's three-phase outline.
No learned memory controller is part of this experiment.

## Question and interpretation

At four persistent slots, does adding memory lower the existing tiny
Transformer's accuracy compared with the same weights and no memory?
Measure paired harmful answers (no memory correct, memory incorrect), beneficial
answers, overall accuracy differences, and stale/deleted answers. A harmful
subset is not evidence that memory reduces overall accuracy. A negative result
is valid. A rule-based reader of exactly the same visible evidence separates
policy information loss from neural reading errors.

## Environment

Use typed SET/UPDATE/DELETE/NOISE/ASK operations. SET is an authoritative upsert;
UPDATE replaces a fact, DELETE removes it, NOISE has no effect, and ASK returns
the latest value or UNKNOWN. The independent scorer has complete ground truth;
policies and neural readers never receive it or future queries during writes.

Main episodes have 30 candidate operations in 8 sessions, followed by one query
in the final session. The predeclared equal-weight cases are historical SET,
historical UPDATE, historical DELETE, unseen key, current SET, current UPDATE,
current DELETE, and a similar-but-wrong key. Distractors include irrelevant
authoritative facts and explicitly marked NOISE. Updated values differ from old
values. Placement varies by seed, including long evidence gaps and conflicting
authoritative observations. Longer-episode evaluation uses 60 candidates and
16 sessions; high-update evaluation uses four overwrites. Final local evidence
permits a meaningful no-memory comparison.

Entity and value identifiers are two decimal characters. Separate fixed,
shuffled partitions supply disjoint train/validation/test *complete symbols*,
while sharing characters so unseen symbols remain representable. Episode seeds
and model seeds are recorded; generator and metrics are frozen before test runs.

## Policies and reset

All bounded policies allocate exactly K x 4 int32 fields: operation, entity,
attribute, value (64 logical bytes at K=4). Empty slots use a sentinel. Order in
the array supplies recency without an uncounted index or timestamp. Policies
ignore explicitly labelled NOISE.

- No memory: no persistent payload.
- FIFO: latest K factual events, including updates and deletion tombstones.
- Recency: latest K distinct keys; supersede earlier records for the same key.
- Similarity: the same bounded FIFO admission, retrieving one record by character
  overlap in the key; latest event breaks equal scores. The query is used only
  at read time. This is lexical similarity, not embedding retrieval.

FIFO and recency expose all their retained slots. Similarity exposes one slot;
the different read compute is reported. A perfect exact-key reader should not
be harmed by these policies; it is a strong control, not the headline neural
result. Full-history reference is scorer-only and explicitly outside the strict
budget, not a neural competitor. No hidden archive or query-aware write oracle.

At a session boundary, construct the next session from only the fixed array.
No past token sequence, activation, KV cache, or per-episode dictionary is passed
to the model. The existing Transformer is stateless and never implements a KV
cache. Serializing the bounded symbolic state is allowed; raw historical replay
is not. Count memory tokens serialized separately from raw-history tokens (zero).
Logical payload bytes are not Python object overhead or total process RAM.

## Reader and local run

Reuse `TinyTransformerLM` unchanged with character-level symbolic prompts and
two-character answers (digits or `??`). Train one shared reader per seed on
visible-evidence answers on independent short reading microtasks, mixing
no-memory/FIFO/recency/similarity views. Microtasks balance present and absent
evidence and include duplicates, deletes, and local overrides; this avoids
training primarily on UNKNOWN from long histories whose facts were evicted.
They use the same symbol partitions and prompt grammar as the episode test.
This learns reading and precedence, not admission or eviction. All policy comparisons
within a seed use identical weights, episodes, and greedy decoding.

Use a checked-in JSON configuration: width 96, two layers, four heads, zero
dropout, 128-token context, batch 64, AdamW at 0.002, 1,200 steps, seeds 7/19/43.
Use CPU or Apple MPS locally. Log validation independently; no held-out tuning.
If visible-evidence validation accuracy is below 95%, label reader competence
as a limitation instead of interpreting failures as an architectural result.

## Outputs and verification

Report exact-match accuracy, case slices, stale-answer rate on updated queries,
deletion leakage, abstention precision/recall/F1, useful retained evidence,
evidence-gap/update-count slices, paired harm/benefit, paired bootstrap intervals,
seed variability, state bytes, prompt/memory tokens, and policy/read/train time.
Save configuration, environment, hashes, per-query predictions, checkpoints, a
machine-readable summary, and a Markdown table. Check in compact results and a
standalone plot; large local artifacts live under ignored `runs/`.

Unit tests use hand-derived truth and metric counts. Integration tests cover
determinism, disjoint symbols, fixed capacity, deletion/updates, near keys, fresh
session/episode isolation, padding, answer-loss alignment, save/reload, and a
small actual training/evaluation run. Research notes distinguish evidence,
inference, and next hypotheses. Broader architecture design waits for evidence.
