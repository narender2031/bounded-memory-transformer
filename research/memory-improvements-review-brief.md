# Memory improvements: approved five-case scope

Date: 2026-09-20. **Status: five cases approved; implementation and local testing authorized.**
The user approved the scope with the amendments below. Paper results remain
evidence for experimental choices, not measured improvements to our system.
The [locked protocol](../docs/superpowers/specs/2026-09-20-five-case-memory.md)
and [execution plan](../docs/superpowers/plans/2026-09-20-five-case-memory.md)
govern the new experiment; historical pilots stay frozen.

## Approved amendments

- Case 1 has four explicit arms: character reader; learned selector → character
  generation; the same selector → exact copy; oracle selector → exact copy.
  The oracle reads only visible evidence and must agree with the exact reader 100%.
- Case 2 reports unsupported, contradicted/stale, irrelevant, and deleted
  separately. Every task includes unrelated SET/UPDATE/DELETE/NOISE distractors,
  so an operation-label shortcut cannot replace full-key matching.
- Require ≥95% visible-evidence accuracy per seed and required category, plus
  ≥95% oracle-relative recovery: `(A_neural - A_current_only) /
  (A_exact_visible - A_current_only)`. Nonpositive headroom is unavailable.
  Report and gate recovery on episode controls as well as reader microtasks.
- Case 5 separates uniform unpredictable queries (A) from queries correlated
  with an observable, budgeted durability cue (B). Future suffixes never enter
  the online policy or eviction labels. Use delayed answer rewards for learning.
- Report evaluation-only clairvoyant top-K retention utility and regret. Add
  stateless random and cue-priority controls; the latter exposes whether learning
  merely recovers the planted heuristic.
- Keep ABSTAIN in the reader, four symbolic slots, and eight-slot transfer.
  Defer DEFER, Case 6, latent compression, and LRU until their workloads exist.

The original alternatives and paper-to-case rationale below remain for context;
the approved amendments and locked protocol take precedence.

## Recommendation: five cases

Start with **correct recall, wrong-memory rejection, updates, selective deletion,
and capacity pressure**. Here, a “case” means an experimental question, not one
paper, one controller action, or a replacement for our existing eight-case dataset.
Several papers inform the same case.

| Scope | Included cases | Trade-off |
|---|---|---|
| **Five — recommended** | Cases 1–5 below | Tests reading, lifecycle correctness, and the central limited-capacity problem. |
| Four — smaller first step | Cases 1–4 | Faster diagnosis; does not establish better admission or eviction under pressure. |
| Six — later extension | Cases 1–5 plus dependent-fact repair | Adds provenance and derived facts, requiring new semantics and extra memory accounting. |

## Evidence behind the proposal

These are author-reported results from the versions reviewed on 2026-09-20.
Different tasks and information budgets make their scores incomparable with ours.

| Paper | Evidence | What we propose to borrow; limitation |
|---|---|---|
| [SeDeM v2](https://arxiv.org/html/2608.00311v2) | Selected compressed memories support 67.25 F1 on 2Wiki and 58.30 on HotpotQA-Distractor with Llama-3.2-3B. | Separate selection from answering in case 1. Its stored bank grows with context; top-k reading is not fixed-capacity storage. |
| [Metis v2](https://arxiv.org/html/2607.26760v2) | Internal operation score 73.77; external MemOps score 24.76. Repeated writes interfere, and activated irrelevant memory hurts unrelated tasks. | Paired irrelevant-memory tests in case 2 and preservation checks in cases 3–5. Its large trained system is not a local reproduction target. |
| [TARL v2](https://arxiv.org/html/2608.03699v2) | Table 2 reports action macro-F1 0.8286 and next-state accuracy 0.6621, versus 0.7887/0.6354 for its LongMemEval-style baseline. | Separate operation choice from target selection in cases 3–5. Its structured ledgers are not bounded latent slots. |
| [MemOps v1](https://arxiv.org/html/2607.12893v1) | 2,006 questions evaluated in two settings; session RAG accuracy 0.845 versus turn RAG 0.618. It diagnoses operation trajectories. | Score updates, deletion, target binding, and damage to other facts in cases 3–5. Use exact synthetic labels instead of an LLM judge. |
| [Dependency-guided rollback v1](https://arxiv.org/html/2608.10502v1) | Recovery 85.3% versus 77.3% on 150 controlled cases; 68% versus 54% on a selected 50-case stress set. | Optional case 6: invalidate unsupported derived claims. The paper assumes diagnosed faults, lineage, and replay. |
| [MetaKV v1](https://arxiv.org/html/2609.07966v1) | Adaptive KV configuration raises constrained success rate by about 0.07 on average, up to 0.135. | Deployment context only. It does not test semantic memory across hard resets. |

Our own evidence motivates the order: neural accuracy falls from **68.71% without
memory to 48.50–48.79% with memory**, while the exact reader improves from 75% to
82.5%. Historical useful-fact retention is only **30%**. Both writing and reading
need investigation; a weak neural reader cannot reliably judge a new writer.
See the [measured-results summary](results-summary-2026-09-20.md).

## The five proposed cases

### 1. Read the right fact and reproduce its value

- **Example:** retain `(12, a) = 23` alongside similar keys; reset; ask `(12, a)`.
  Expected answer: `23`. Also test a newer value supplied in the current session.
- **Change:** a learned selector considers memory and current-session records,
  selects the supporting record, and copies its value; include an UNKNOWN option.
- **Comparison:** current character reader, selector feeding the character reader,
  and selector with exact copying, starting from the same available records.
  Hold selector decisions fixed when comparing the latter two. Keep the exact-rule
  control; disclose selector supervision and additional training/read cost.
- **Measure:** visible-answer accuracy, full-key matching, unfamiliar values,
  and four-slot occupancy. Selection and copying may fail independently.

**Hypothesis:** selection plus copying reduces reading errors. SeDeM motivates
separating selection and conditioning; exact copying is our adaptation, also
motivated by our earlier reader diagnostic. Latent decompression comes later.

### 2. Reject similar-but-wrong and irrelevant memory

- **Example:** retain `(12, a) = 23`; ask never-observed `(13, a)`.
  Expected answer: UNKNOWN. Separately, add irrelevant records to a question
  already answerable from the current session; its answer should stay correct.
- **Change:** explicitly test and train evidence rejection with paired empty-bank
  and irrelevant-bank examples, while preserving the question and answer.
- **Measure:** paired harmful-memory rate, abstention precision/recall, and
  accuracy on answerable questions. Always abstaining must not count as success.

**Hypothesis:** targeted rejection reduces memory-induced mistakes without losing
useful recall. Metis motivates the interference control; our near-miss accuracy
of 9.67% with similarity versus 88% without memory makes it a local priority.

### 3. Update the correct fact without damaging another

- **Example:** store `A = old` and `B = keep`; reset; update `A = new`; reset;
  query both. Expected: `new` and `keep`.
- **Change:** test a small learned operation-and-target writer through the exact
  reader first. Start with enough room for both facts, isolating updates from eviction.
- **Measure:** action macro-F1 across lifecycle cases, target accuracy, latest-value
  accuracy, stale-value rate, and transition correctness. Compare with recency, which performs
  correct same-key replacement in our implementation.

**Hypothesis:** learned routing can maintain update semantics; improvement over
the structured baseline is unproven. TARL and MemOps motivate the decomposition.
Parsing an explicit UPDATE label alone is an engineering check, not a research gain.

### 4. Delete one fact while preserving unrelated facts

- **Example:** store `A = secret` and `B = keep`; reset; delete A; reset; query both.
  Expected: UNKNOWN for A and `keep` for B. Keep this control within capacity.
- **Change:** make deletion explicit in the writer and test the unaffected fact.
- **Measure:** existing deleted-query non-abstention, separately identified
  repetition of the deleted value, and collateral forgetting of B. Clearing all
  memory fails the preservation check. Use the exact reader to isolate writing.

**Hypothesis:** targeted deletion preserves other retained information. MemOps and
Metis motivate the paired checks. TARL informs operation/target separation, but
its actual five actions do not include explicit DELETE.

### 5. Choose useful facts when the bank fills

- **Example:** stream 30 candidate operations over eight sessions, exceeding four
  slots. Compare four and eight slots on matched episodes; query after resets.
- **Change:** evaluate admission and eviction separately from correct updating,
  using the exact reader and then the validated neural reader.
- **Measure:** historical useful-fact retention, latest-value accuracy, harmful
  answers, state bytes, and write/read cost. Compare no memory, FIFO, recency,
  and similarity; bounded policies get equal bytes within each capacity condition.

**Hypothesis:** learned choices improve usefulness under pressure. None of these
papers proves this for our task. Uniformly unpredictable future queries may offer
no learnable admission advantage. No future query is visible during writing;
supervised eviction labels must disclose their source. Four symbolic slots are
not expected to retain 30 independent records. Expand to 16/32 slots later.

## What would change first

1. Add the case-specific controls and metrics without altering recorded pilots.
2. Compare the reader variants on cases 1–2 and visible update/delete tasks.
3. Test operation/target writing on cases 3–4 using the exact reader, then case 5
   for capacity. Combine the learned reader and writer only after separate checks.

Proposed writer semantics are STORE, UPDATE, DELETE, and IGNORE, with an eviction
target when full. The exact head design remains an implementation-plan choice.
ABSTAIN is a reader decision. DEFER needs a later uncertain-evidence workload;
our current SET/UPDATE/DELETE events are authoritative. TARL's actual actions are
append/noop/revise/reject_conflict/defer_verify, so this is an adaptation.

## Shared criteria and deferred work

- Preserve hard resets and zero raw-history access. All model-accessible versions,
  pending records, and provenance count toward persistent capacity.
- **Existing reader gate:** at least 95% visible-evidence accuracy in every seed.
  **Proposed additional gates:** at least 95% separately on known answers, unknown
  answers, updates, deletions, and full occupancy. These are engineering targets,
  not results or paper-derived thresholds.
- Report retention and lifecycle errors separately. Claim a writer benefit only
  if paired comparisons at equal bytes show gains beyond sampling uncertainty
  without concealing worse deletion, abstention, or control-fact preservation.
- Predeclare seeds, datasets, training budgets, metric denominators, and tests
  in the implementation plan before training. Preserve unseen-symbol evaluation
  and report seed variation; a fresh episode seed alone is not a new symbol split.

Optional **case 6** would store a conclusion derived from a source, correct/delete
that source, and check dependent versus independently supported conclusions.
It needs explicit dependency semantics and bounded provenance first. Defer it,
latent decompression, reconstruction/pollution auxiliary losses, and uncertain
evidence until the five-case measurements justify a particular addition. MetaKV
remains outside the core comparison. LRU is planned, not an existing baseline.

The user selected **five cases, implemented in stages**, starting with cases 1–2,
and then authorized building and testing all five. The amendments above replace
the earlier review checkpoint.
The [full paper review](reading-notes/2026-09-20-six-paper-review.md) retains the
methodological details behind this brief.
