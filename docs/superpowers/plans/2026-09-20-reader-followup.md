# Reader reliability and emergent utility follow-up

The user has authorized reader repair, a uniform-control replication, and design
of emergent utility. Existing runs, generators, metrics and checkpoints remain
unchanged. This plan is implementation authorization, not a claim of success.

## Work streams

- [x] Diagnose the frozen selector on fresh train/validation tasks: full key,
  operation authority, precedence, false abstention, 4/8 slots and current counts.
- [x] Implement a small inspectable reader ablation with meaningful tests. Use
  train/validation only for development; explicitly disclose structured features
  and any deterministic read logic. Preserve the exact-copy/oracle controls.
- [x] Freeze the final architecture, training configuration and new evaluation
  protocol before constructing held-out examples. Keep the absolute 95% slice
  gate and raw oracle-relative recovery >=95%; unavailable headroom is not a pass.
- [x] Test three training seeds, current-record/occupancy changes and longer
  evidence; pair every reading condition with the same reader without memory.
- [x] Independently replicate uniform A with frozen retention checkpoints and
  fresh episode seeds. Preregister sample size, comparisons, equivalence margin
  and intervals. Publish all replications, including failures.
- [x] Design Case 5B-v2 using observed past access and lifecycle events, budgeted
  metadata, strong access heuristics, causal learning, and held-out mechanisms.
  No Case 5B-v2 implementation or success claim is implied by this design task.
- [x] Audit saved predictions, run lint/tests, document evidence/inferences and
  failures, update continuity files, and open a stacked reviewable draft PR.

## Boundaries

All changes live in new `memory_followup` modules and Project 04 artifacts;
Project 03 source remains frozen. Initially budget three reader configurations
using validation. A pre-test amendment permits one fourth version after an
additional validation probe exposed reverse-operation/recreation failures despite
passing the earlier slices. A final pre-test readout amendment fixes binary
eligibility/chronology composition using unchanged candidate-4 weights. Both are
recorded in the frozen protocol. Retain every development outcome; if gates fail, report the
failure and a next hypothesis rather than retuning after a held-out result.
The final architecture need not be a new scientific contribution: it first
establishes trustworthy experimental instrumentation for the capacity study.

Full held-out execution depends on passing behavioral checks and freezing the
protocol/configuration/source. Independent uniform replication and Case 5B
design can proceed alongside reader development without sharing model state.

Completed: [results](../../../research/reader-followup-results-2026-09-20.md) and
[draft PR #4](https://github.com/narender2031/bounded-memory-transformer/pull/4).
Emergent utility remains a design; its implementation and evaluation are not
completion criteria for this reader/replication/design task.
