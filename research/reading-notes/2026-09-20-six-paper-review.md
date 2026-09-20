# Six-paper review against our measured memory failures

Reviewed 2026-09-20 from the primary arXiv abstracts and targeted full-text
methods, results, and limitations. These are author-reported results, not local
reproductions. Versions: SeDeM v2, Metis v2, TARL v2, MemOps v1, Rollback v1,
MetaKV v1. The user's supplied scan prompted this review. Our own experiment
results remain those measured on 2026-09-17; see the
[results summary](../results-summary-2026-09-20.md).

## What this changes

The most useful immediate additions concern evidence selection, rejection of
irrelevant memory, and evaluation of state changes. They reinforce the need to
separate the reader from the writer. They do not make our weak neural reader a
trustworthy evaluator of a new controller.

Our existing map incorrectly described Metis as a vision/taxonomy paper. Its
current paper describes a trained prototype and empirical evaluations. That
classification is corrected. The other four core additions are SeDeM, TARL,
MemOps, and dependency-guided rollback; MetaKV belongs in deployment context.

## Verified evidence and scope

| Paper and primary full text | Checked result | Scope relevant to interpretation |
|---|---|---|
| [SeDeM](https://arxiv.org/html/2608.00311v2), Table 1 and sections 6–7 | Llama-3.2-3B: 67.25 F1 on 2WikiMHQA; 58.30 on HotpotQA-Distractor. Reported mean TTFT improvements versus ICAE are 1.74×/2.46× for 1B/3B. | Supervised evidence selection and hidden-state compression. Its bank grows with context segments; selected-block count is a read budget. |
| [Metis](https://arxiv.org/html/2607.26760v2), Table 5 and sections 5–6 | Metis-27B: 73.77 on its internal operation set versus 16.87 for its context-free backbone; 24.76 on MemOps (Gold) versus 87.90 for the full-context backbone. | Distinct datasets and information access; these are not directly comparable to our exact-match synthetic scores. |
| [TARL](https://arxiv.org/html/2608.03699v2), Tables 2–3 | Table 2: action macro-F1 0.8286 and next-state accuracy 0.6621, versus 0.7887/0.6354 for its LongMemEval-style comparison. Gold-binary executors in Table 3 obtain 0.2860/0.4539 next-state accuracy. | Executable structured ledgers, target and reliability metadata; not fixed-capacity latent storage. Table 3 separately reports TARL at 0.6521; preserve table context rather than conflating it with Table 2. |
| [MemOps](https://arxiv.org/html/2607.12893v1), Tables 2–3 | 2,006 unique questions, each evaluated in adjacent/long settings: 4,012 instances. Session RAG 0.845 versus turn RAG 0.618 accuracy. | Both RAG variants use BM25 and GPT-4.1-mini. This is an evaluated granularity difference, not proof that larger records always win at fixed bytes. |
| [Dependency-guided rollback](https://arxiv.org/html/2608.10502v1), abstract and evaluation | 85.3% versus 77.3% recovery on 150 controlled cases; 68% versus 54% on a selected 50-case adapted stress set. | Faulty sources are already diagnosed. Runtime lineage and selective replay are available; neither is free under our contract. |
| [MetaKV](https://arxiv.org/html/2609.07966v1), abstract and method | About +0.07 average, up to +0.135 constrained success rate versus best static configuration. | Chooses among inference cache configurations under latency/memory constraints, not semantic memory across resets. |

## SeDeM: use the separation and controls before adding decompression

**Evidence:** it compresses segments, scores blocks against a query, expands
selected blocks, and injects their states into a decoder. Section 3 defines a
bank containing memories from all segments. Section 6.3 and Appendix G count
online compression, selection, and decoder prefill in TTFT. Appendix R reports
quality trade-offs: matched raw-text RAG and full-bank variants can outperform
selective configurations. Top-k selection is not an unconditional quality win.

**Inference for us:** read capacity and persistent capacity must be reported
separately. Selecting two of many archived blocks would violate our four-slot
contract if the larger archive survives resets. Our symbolic records already
preserve their values exactly, so they do not yet require decompression to
recover those values.

**Proposed ablation:** keep identical four-slot banks and compare all-record
reading, learned top-1/top-2 reading with UNKNOWN, and record selection plus exact
copying. Only then add a learned expansion module if testing compressed latent
slots. Expansion should be temporary read computation, not an extra persistent
archive. Match selector supervision and retained inputs; count expansion compute.
This adapts a design principle rather than reproducing SeDeM.

## Metis: correct the map and borrow interference controls

**Evidence:** Metis trains a native-memory prototype with forward-only online
updates and frozen inference weights; training itself uses gradients. It trains
on remember/update/forget/reflect operations plus auxiliary binding and memory
pollution examples. It reports eight H100s for training. Sections 6.6–6.7 expose
interference across repeated writes and degradation on unrelated tasks after
irrelevant memory is activated. Its external forgetting results remain weak.

**Inference for us:** this is closer architectural prior art than our old map
acknowledged. Neither native state nor forward-only memory maintenance is a new
claim for this project. Its pollution problem is conceptually relevant to our
near-miss and unknown-query failures; the underlying mechanisms are not identical.

**Proposed ablation:** pair the same locally answerable or unknown query with an
empty bank and with unrelated records. Train evidence rejection and measure
paired harm. Separately test multi-entity binding and selective deletion that
preserves an unrelated control fact. Begin with answer/selection supervision;
add reconstruction or pollution losses one at a time and report their ablations.
This does not require reproducing the billion-parameter architecture locally.

## TARL: operation plus target, not a magic number of actions

**Evidence:** the exact actions are `append`, `noop`, `revise`, `reject_conflict`,
and `defer_verify`. They update accepted, pending, and rejected-history ledgers.
The paper supervises action, slot target, reliability, and counterfactual
next-state quality; hypothetical execution is a training mechanism. A binary
write/hold label loses distinctions needed by this executor.

**Correction:** the proposed `add/replace/delete/defer/ignore` set is an adaptation,
not TARL's exact five actions. Explicit deletion is missing from TARL's action
list. `ABSTAIN` concerns answering; `DEFER` concerns handling an incoming fact.
Our roadmap already includes multiple memory operations rather than a committed
binary-only writer. LRU remains planned, not implemented.

**Proposed adaptation:** separate operation choice from slot targeting and
capacity eviction. Keep DELETE explicit. Introduce DEFER only with a declared
uncertain-evidence workload and resolution semantics. The current authoritative
SET/UPDATE/DELETE syntax does not justify learning source reliability from absent
signals. Pending facts and rejected evidence must consume the same fixed total
budget; three independently growing ledgers would break the experiment.

## MemOps: improve diagnoses with exact synthetic labels

**Evidence:** events carry triggers, targets, scopes, transitions, and provenance;
probes distinguish operation detection, target binding, state transition, and
trajectory recovery. Operation diagnostics use a GPT-4o judge, whose instability
the paper acknowledges. It also distinguishes deletion leakage from damaging
unrelated retained facts.

**Inference for us:** final-answer accuracy is insufficient for writer evaluation,
just as our exact-reader control already separates retention from neural reading.
Our synthetic setting can calculate discrete correctness without an LLM judge.

**Proposed metrics:** action macro-F1 with explicit labels and denominators;
slot-target accuracy on applicable actions; correct latest-value transitions;
leakage after deletion; retention of unrelated control facts; source-support
accuracy when source IDs are available; and query-level trajectory correctness
after repeated writes. Our current stale-rate and deletion-leakage metrics already
exist and should not be presented as new. Do not require a four-slot bank to
equal the entire unbounded world state. Score semantic transition integrity and
capacity-constrained usefulness separately. Gold traces may remain with the
scorer; anything accessible to the model after reset counts as persistent state.

## Rollback: useful later, but provenance and replay have costs

**Evidence:** the method traces downstream consequences of diagnosed faulty
memories, retains independently supported claims, deactivates unsupported state,
and replays affected computations. Its stress evaluation is an adapted subset,
not the complete LongMemEval-V2 benchmark.

**Inference for us:** removing a fact can leave derived conclusions invalid.
Our present task contains assignments and queries, not derived stored claims or
tool-action chains, so this is not an explanation already tested by our pilot.

**Proposed future task:** store a source and a conclusion derived from it, then
correct/delete the source and query both dependent and unrelated conclusions.
Source/version identifiers can help, but one source ID is not a full dependency
graph when conclusions have several parents or independent support. Prevent
slot reuse from silently redirecting old dependencies. Charge all such metadata
to capacity. An evaluator-only lineage log is acceptable; model-accessible raw
history or execution replay belongs in a separately named relaxed reference.

## MetaKV: keep it outside the core comparison

**Evidence:** it predicts configuration quality and cost across KVQuant, H2O,
RocketKV, and an uncompressed alternative. Its objective includes satisfying
both latency and memory constraints.

**Decision for this project:** retain it as deployment context. Our evaluator
already exposes memory capacity in configuration; the missing work is the
capacity sweep. Selecting a KV-cache configuration would not establish semantic
state preservation across hard resets.

## Updated order of work

1. Finish the competent-reader ablation on the existing four-slot task: complete
   keys, record selection, exact copying, and UNKNOWN. Retain the current model
   and exact-rule reader as controls. Add paired irrelevant-memory validation.
2. Specify operation/transition/control-fact metrics and new workloads before
   evaluating them. Existing pilots and denominators remain frozen.
3. Evaluate a bounded action-and-target writer through the exact-rule reader to
   isolate writing. Direct action supervision is an engineering control; a later
   answer-only objective needs a separate comparison. A new writer does not have
   to wait for neural reading to be interpretable if its reader is exact.
4. Combine writer and competent neural reader, then test learned expansion or
   latent compression with equal byte budgets and measured read/write costs.
5. Add uncertain evidence, bounded deferral, derived facts, and repair as separate
   task extensions. Sweep capacity; do not merge all mechanisms into one trial.

The useful question is which facts a bounded system should retain, update, and
trust while processing 20–50 candidate facts. Four symbolic slots cannot
simultaneously store 20–50 independent records. Missing evidence must be
distinguished from wrong reading, and abstention coverage must accompany accuracy.
None of these papers establishes a measured improvement in our implementation.
