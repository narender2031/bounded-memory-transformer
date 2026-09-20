# Research Decisions

This file records decisions that future sessions should preserve unless new evidence justifies a change.

## 2026-08-27 — D001: Project name

**Decision:** Use `bounded-memory-transformer`.

**Reason:** The name states the defining experimental constraint instead of implying a novel general-purpose memory model.

## 2026-08-27 — D002: Narrow research target

**Decision:** Focus on learned memory selection plus semantic supersession under a strict fixed capacity.

**Reason:** Persistent memory, recurrent memory tokens, test-time neural memory, and future-utility prediction are established areas. The narrower combination remains experimentally unresolved.

## 2026-08-27 — D003: Hard reset contract

**Decision:** In the strict-memory condition, delete token context and the normal KV cache between sessions. Preserve only the bounded memory state.

**Reason:** This prevents long context or hidden replay from being mistaken for persistent memory.

## 2026-08-27 — D004: Synthetic benchmark first

**Decision:** Begin with symbolic `SET`, `UPDATE`, `DELETE`, `NOISE`, and `ASK` episodes before natural language.

**Reason:** Controlled data isolates memory operations, supports unlimited examples, and enables disjoint entity/value splits.

## 2026-08-27 — D005: No broad novelty claim

**Decision:** Do not claim that learned memory, bounded memory, future-utility prediction, or supersession training is individually novel.

**Reason:** RMT, MemoryLLM, Titans/MIRAS, SP-KV, GradMem, Learning to Remember, Supersede, and LiveMem cover substantial portions of this space.

## 2026-08-27 — D006: Memory-aware reporting

**Decision:** Always report stale-answer rate, deletion leakage, and abstention metrics alongside overall accuracy.

**Reason:** Average accuracy can hide the exact failure this project aims to fix.

## 2026-09-02 — D007: Establish a trusted Transformer baseline first

**Decision:** Implement Project 01 as a small decoder-only Transformer without persistent memory and without high-level Transformer wrappers.

**Reason:** The attention, causal masking, residual, normalization, MLP, training, and generation paths must be understandable and tested before memory changes the architecture. The same components will become the no-memory control model.

## 2026-09-17 — D009: Phase 1 isolates policy retention from neural reading

**Decision:** Experiment 01 uses four fixed symbolic slots, an independent
reference state machine, four hand-written policies, and the existing
`TinyTransformerLM` as a shared reader. One reader per seed is reused across
no-memory, FIFO, recency, and similarity views. Do not implement the learned
controller until the basic reader/evaluation controls are trustworthy.

**Definitions:** FIFO retains the newest factual events including tombstones;
recency retains the newest distinct keys with correct supersession; similarity
does top-1 lexical key retrieval from the same bounded FIFO bank with newest
ties. Each bounded bank is a K×4 int32 array (64 logical payload bytes at K=4).
No extra archive, query-aware write oracle, hidden clock, or raw-text replay.
Serialized bounded-memory tokens are counted separately from raw-history tokens.
This is structured symbolic memory, not the eventual learned latent state.

**Reason:** Reliable policies may not be harmed by explicit updates/deletions.
A perfect reader of the same visible state supplies an essential control.
An average decline or paired harm in a weak neural reader cannot establish a
need for learned memory management. Similarity here is not embedding retrieval.

## 2026-09-17 — D010: Freeze comparisons and retain failed reader controls

**Decision:** Save configurations, source revisions/hashes, dataset hashes,
checkpoints, per-query predictions, full metrics, and timing. Use disjoint full
entity/value symbols with shared characters. Compare identical episodes and
weights; report three model seeds, paired harmful/beneficial fractions and
episode-bootstrap intervals conditional on those seeds. Also report variability
across model seeds. A zero denominator is unavailable, not zero error.

**Competence gate:** Require at least 95% exact-match visible-evidence validation
accuracy in every seed before treating neural errors as a strong memory result.
Train on independent reading microtasks, with explicit full-key discrimination,
duplicates, deletes, and current-session precedence. Controller actions are not
learned or supervised in Phase 1.

**Failure protocol:** Preserve the 1,200-step pilot and its deficient microtask
coverage. The independent review found that an attribute-blind reader passed all
512 original validation examples. Correct training/validation coverage, add a
regression test, and predeclare one 6,000-step, 65,536-example replication with
fresh episode seed 1729. Keep the held-out generator, metrics, policies, symbol
splits, architecture, and learning rate unchanged. The new episodes reuse the
test symbol partition; do not describe them as an independent symbol split.

## 2026-09-17 — D011: Diagnose reading before changing memory management

**Decision:** Separate visible-evidence reading competence from episode accuracy.
Use frozen-checkpoint train/validation diagnostics to distinguish symbol
generalization, record selection, value reproduction, and evidence rejection.
The next reader candidate is learned record selection with exact copying and an
UNKNOWN option, compared with character generation and the exact-rule control.
This is an ablation direction, not an adopted final memory architecture.

**Evidence:** an injectively renamed, matched 2,048-task diagnostic averaged
78.91% with seen entity/value vocabularies and 76.06% with both unseen. The latter
had 48.51% accuracy on known answers available only in memory, versus 84.43% on
known answers supplied by the current session. These location slices contain
different task mixtures and do not isolate location as a causal factor.

**Constraints:** keep the 95% per-seed visible-evidence competence gate; predeclare
additional slice thresholds and the next training/evaluation configuration.
Include current-session evidence in any pointer candidate set. Disclose direct
record supervision. Freeze historical pilots and use fresh episodes after model
development; a fresh seed alone does not create a new unseen-symbol partition.
Do not claim gains from copying until measured. A repeated-query workload is
required to meaningfully compare access-frequency/recency policies, and all
metadata must count toward persistent-state capacity.

## 2026-09-20 — D012: Adapt mechanisms without changing the memory contract

**Decision:** integrate the reviewed papers as separate candidate mechanisms and
evaluation controls. Keep the reader competence milestone. A bounded writer may
be evaluated independently through the exact-rule reader before combining it
with neural reading. No all-in-one architecture is selected by this review.

**Evidence:** primary Metis v2 is a trained prototype, correcting its old map
classification. SeDeM separates read selection from a larger stored bank. TARL's
actions are append/noop/revise/reject_conflict/defer_verify, not the proposed
add/replace/delete/defer/ignore set. MemOps supplies finer operation diagnostics;
rollback assumes diagnosed faults, provenance, and replay. MetaKV addresses
inference cache configuration rather than semantic memory after reset.

**Constraints:** top-k read count is not stored capacity. Pending/rejected items,
provenance, versions, and dependency links count in the total persistent budget.
Keep DELETE explicit and distinguish writer deferral from answer abstention.
Introduce uncertainty, derived facts, and dependency repair only as separately
declared workloads. Full raw-history replay remains outside strict comparisons.
Current metrics and historical pilots remain unchanged. Retaining the entire
unbounded world state is not a valid success target for four symbolic slots.

## 2026-09-20 — D013: Execute five decomposed cases with causal capacity learning

**Decision:** the user approved and authorized all five cases. Freeze the
[protocol](../docs/superpowers/specs/2026-09-20-five-case-memory.md) and
[configuration](../projects/03-memory-reliability/configs/five-case-small.json)
before held-out evaluation. Four reader arms share inputs/weights/selector
choices as applicable. Use the exact visible oracle as a 100% implementation
control, not as an oracle for discarded facts. Separately report unsupported,
contradicted, irrelevant, and deleted queries.

**Gates:** ≥95% per-seed visible accuracy overall and on every required reader
slice. Recovery is `(A_neural-A_current_only)/(A_exact_visible-A_current_only)`;
nonpositive headroom is unavailable, not a pass. Require ≥95% recovery on the
reader microtasks and each declared nonempty-memory episode control, including
combined diagnostics. Require ≥95% lifecycle semantic, update, deletion, and
control preservation. Failed gates label the combined system unvalidated; they
do not authorize changing held-out tasks or skipping the remaining cases.

**Capacity:** canonical K×4 int32 `[entity, attribute, value, observed cue]`,
64 bytes at four slots. Evaluate identical weights at eight slots. Workload A
has uniform future queries; B has a planted binary cue with 5:1 query weighting.
The cue is observed before independent query sampling and charged to all banks.
Train the 25-parameter cue scorer with delayed query rewards, never future action
or frequency labels. Compare FIFO, recency, bounded similarity, stateless random,
and a cue-priority heuristic. A clairvoyant top-K future-count oracle is a feasible
evaluation-only bound for this write-then-query workload; report retention regret.

**Interpretation:** this tests whether simple causal learning recovers observable
utility, not whether it predicts arbitrary future questions or beats a matched
heuristic. Writer symbolic equality features and selector provenance supervision
are disclosed engineering aids. ABSTAIN stays in the reader; DEFER, latent slots,
dependency repair, and interleaved-access LRU are deferred. Prior pilots keep their
original event-bank semantics and cannot be directly compared to the new scores.

## 2026-09-20 — D014: Preserve failed reader gates and narrow the capacity claim

**Evidence:** the frozen five-case run completed all cases and combined controls.
Every reader seed fails the full absolute/relative gates despite a 92.83% mean
selection/copy score. Controlled four-slot lifecycle is perfect with supplied
key-equality features. Learned capacity exactly matches cue-priority; B gains
12.50/25.20 percentage points over FIFO at four/eight slots. A8 also has a positive
observed interval (0.76–5.64 points), while its uniform population definition has
no cue advantage. See [full results](five-case-results-2026-09-20.md).

**Decision:** retain the complete run unchanged and label the combined neural
system unvalidated. Do not interpret three identical greedy capacity policies on
shared episodes as three independent dataset replications. Do not conceal the A8
contrast, claim a confirmed null, or assume all clairvoyant regret is learnable.
No future-query information or test-based retuning is authorized by these results.

**Next hypotheses:** use train/validation ablations of key matching, precedence,
UNKNOWN thresholds, and occupancy/current-record distribution. Predeclare a new
held-out protocol after development, and separately replicate uniform A on fresh
episodes. More realistic future utility needs an interleaved, budgeted-access
workload and stronger access-based baselines; latent compression stays deferred.
