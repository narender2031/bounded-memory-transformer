# Improving memory: first separate storage, reading, and abstention

Date: 2026-09-17. Targeted primary-source research plus a frozen-checkpoint
diagnostic. This note proposes experiments; it reports no improved architecture.
For the underlying Transformer, see the [fundamentals guide](../transformer-fundamentals.md).

## What our two readers actually do

The memory policy chooses what survives. The reader answers using the surviving
records, current-session operations, and question. These are separate components.

```text
earlier sessions -> FIFO / recency / similarity -> four stored records
                                                        |
current session + question ------------------------------+
                                                        v
                                  neural OR exact-rule reader -> answer
```

The neural reader is our existing decoder-only Transformer. It receives a
character serialization and predicts two answer characters. The exact-rule
reader is `read_visible()` in `memory_benchmark/views.py`: scan the same visible
records in chronological order, match both entity and attribute, apply assignments
and deletions, and return the value or `??`. It sees no full episode or privileged
truth. Memory plus this Python reader is a complete, hand-designed memory system.

For example, memory `(12, a) = 34` followed by current-session `UPDATE (12, a) = 56`
must answer `56` for `ASK (12, a)`. An observation about `(12, b)` is a different
key. The exact reader implements this directly; the Transformer must learn it.

The reported percentages were different evaluations:

| Measurement | Examples and target | Result |
|---|---|---:|
| Neural reader validation | 512 short reading tasks using validation-only entity/value symbols; answer according to visible evidence | 73.83–78.52% across three trained models |
| Neural no-memory episode accuracy | 800 main episodes; answer compared with full historical truth | 68.71% mean |
| Neural FIFO episode accuracy | Those same 800 episodes and weights | 48.50% mean |
| Exact-rule no-memory episode accuracy | Same episodes, current evidence only | 75.00% |
| Exact-rule FIFO episode accuracy | Same episodes, four-slot FIFO plus current evidence | 82.50% |

The last number is **82.5%, not 85%**. It is below 100% because the exact reader
cannot recover a current fact that the policy discarded. It is a reference
reader, not a mathematical upper bound on lucky guesses. Our chosen case mixture
also matters: 50% of world-truth answers are unknown and 25% have a current-session
assignment, so an exact no-memory reader already scores 75%.

**Evidence:** the retained evidence helps the exact reader while harming these
neural readers. **Inference:** reading is a substantial bottleneck. This does not
establish that FIFO or recency is the cause of the neural failure, or that a
learned write controller would solve it. Main-run obsolete-value error rates were
much smaller than deletion leakage and other incorrect answers.

## New diagnostic: unfamiliar names versus unfamiliar values

The checked-in [configuration](../../projects/02-memory-benchmark/configs/reader-diagnostic.json)
was written before this diagnostic ran. It generates 2,048 reading structures
using seed 91317 and performs independent injective renamings of their entities
and values (seed 73019 plus example index). Four conditions use training or
validation symbols for each axis. All use the three saved v2 readers, seeds
7/19/43, without training or test-symbol access.

Renaming preserves equality relationships, operation order, prompt length,
visible slot count, and answer availability; these properties are asserted.
It happens **after retrieval**. Rerunning lexical retrieval after changing names
could select different records and would confound this reader diagnostic.

| Entity symbols | Value symbols | Overall accuracy | Known visible answer | Required abstention |
|---|---|---:|---:|---:|
| Seen vocabulary | Seen vocabulary | 78.91% | 77.07% | 81.08% |
| Unseen vocabulary | Seen vocabulary | 78.66% | 76.68% | 81.01% |
| Seen vocabulary | Unseen vocabulary | 75.49% | 69.73% | 82.29% |
| Unseen vocabulary | Unseen vocabulary | 76.06% | 70.63% | 82.46% |

These are means across three seeds on 2,048 matched structures, **not 6,144
independent examples**. "Seen" means symbols available in training, not that
every prompt is an exact training example. Both-unseen overall accuracy is
77.88%, 75.83%, and 74.46% for seeds 7, 19, and 43 respectively.

Both-unseen diagnostic slices:

| Slice | Structures per seed | Accuracy across seeds |
|---|---:|---:|
| Known answer supplied by current session | 683 | 84.43% |
| Known answer available only in memory | 426 | 48.51% |
| Must abstain on visible evidence | 939 | 82.46% |
| Four visible memory records, any target | 359 | 63.14% |

**Evidence:** substantial reading failure exists even with training-vocabulary
symbols. Changing entity vocabulary has a small effect in this diagnostic;
changing value vocabulary is more damaging. **Inference:** unfamiliar-value
handling deserves an ablation, but it cannot explain the whole failure.
The model needs better record discrimination and evidence rejection too.

**Limits:** one generated structural sample and one renaming protocol, with
descriptive means and per-seed results. The current-versus-memory and occupancy
slices contain different task mixtures; they are not paired interventions on
location or slot count. Renaming values changes input and output symbols, so it
does not isolate a decoder-only copying defect. No causal claim about attention
heads or the training optimizer follows from these scores.

The local run took 3.39 seconds inside the timed region, excluding Python startup
and initial base-view generation, on MPS. The output field also includes renamed
view generation and prediction I/O. This is a single timing observation.
Reproduce from the repository root with the original local v2 checkpoints:

```bash
.venv/bin/python projects/02-memory-benchmark/diagnose_reader.py \
  --config projects/02-memory-benchmark/configs/reader-diagnostic.json \
  --output runs/reader-diagnostic-reproduction
```

[Full summary](../../projects/02-memory-benchmark/results/2026-09-17-reader-diagnostic.json)
includes per-seed results, source/configuration/checkpoint hashes, and environment.
All 24,576 predictions remain in `runs/reader-diagnostic-2026-09-17/`. No original
pilot artifact, held-out generator, model weight, or production policy changed.

## Alternatives worth testing, in order

### 1. Separate record selection from value copying

[Key-Value Memory Networks](https://aclanthology.org/D16-1147/) separate the
representation used to find a memory from the content returned by it.
[Pointer-generator networks](https://arxiv.org/abs/1704.04368) demonstrate source
copying as a way to reproduce factual details. Their original tasks and budgets
differ from ours; the proposed reader is an adaptation, not a reproduction.

**Hypothesis:** score visible records against the full query key, select a record
or UNKNOWN, and copy the selected record's two-character value deterministically.
The candidate set must include current-session records as well as memory.
The model still needs to learn correct matching, latest-operation precedence,
deletion handling, and rejection. Exact copying eliminates spelling errors only
conditional on selecting the right record; it cannot restore evicted evidence.

Keep an explicit comparison with the current character-generating reader and the
exact-rule reader. Also diagnose selection and copying separately with labelled
reading microtasks. Any privileged correct-record control is a diagnostic only.
Direct record supervision must be disclosed; it is reader supervision and does
not establish learned memory admission.

### 2. Retrieve nothing when nothing matches

The present similarity baseline always returns a nearest record from a nonempty
bank, even if the key is wrong. A useful inexpensive control is bounded exact-key
retrieval, returning no record when the key is absent. For these explicit symbols,
the full key is available without a learned semantic matcher or future knowledge.
It remains a hand-designed baseline, and only the same four slots are searchable.

Policy-only experiments can also use the exact-rule reader now to isolate retained
evidence from neural reading errors. They must be reported as structured-policy
results. The neural competence gate applies to interpreting the neural comparison;
it does not prevent investigating symbolic admission controls in parallel.

A learned reader should separately estimate whether evidence supports an answer.
[Sufficient Context](https://arxiv.org/abs/2411.06037) separates insufficient
retrieved evidence from failures to use sufficient evidence and investigates
guided abstention. We can label sufficiency exactly in our synthetic setting.
Report answered-query accuracy alongside coverage, abstention precision/recall,
and known-answer accuracy so always abstaining cannot appear successful.

### 3. Admission by future usefulness, with meaningful controls

FIFO admits every factual event; recency retains the latest distinct keys. Neither
asks whether a newly arrived, valid fact is more useful than a stored one.
[TinyLFU](https://arxiv.org/abs/1512.00727) estimates recent access frequencies
to decide whether to admit a candidate over an eviction candidate. LRU and LFU
are also useful controls when queries recur. Frequency is not factual validity.

Our current episodes have only a final query, so repeated-access policies lack
the intended signal. Predeclare a separate repeated-query workload with changing
interests before judging them. Count every frequency sketch, counter, and age
field in the persistent-state budget. A predictor cannot reliably anticipate an
independent uniform future query without observable predictive information.

[SP-KV](https://arxiv.org/abs/2605.14037) is closer to learned admission: it trains
future-utility prediction jointly with the language model. Its retained local
window and dynamic cache sparsity differ from our hard-reset fixed-capacity
contract. Borrow the question of utility, not its reported results or capacity
claims. This becomes informative after the reader passes its controls.

### 4. Validity, expiry, and semantic supersession

The existing recency baseline already merges same-key writes and retains deletion
tombstones; proposing this again would duplicate a baseline. More challenging
variants concern implicit invalidation, uncertain evidence, and temporary facts.
[EXPIRE-SPAN](https://arxiv.org/abs/2105.06548) learns memory lifetimes.
[STALE](https://arxiv.org/abs/2605.06527) motivates testing implicit invalidation.

Expiration is a hypothesis for a new workload with defined lifetimes, not a
correctness improvement for permanent facts that simply happen to be old.
Version, validity, and confidence metadata consume capacity. A confidence score
must be calibrated on validation examples and does not establish truth by itself.
Do not silently add such signals to the already-evaluated generator.

### 5. Learned compact state after the symbolic controls work

[RMT](https://arxiv.org/abs/2207.06881) passes learned memory tokens between
segments. [Gated DeltaNet](https://arxiv.org/abs/2412.06464) combines gating for
forgetting with targeted delta updates. These offer different representation and
update mechanisms for later experiments, not fixes already validated here.

Compare persistent **bytes** as well as slots and all write/read compute. Four
96-dimensional float32 vectors consume 1,536 payload bytes; today's four records
consume 64. Equal slot counts alone would make that comparison misleading.

## Concrete next experiment: competent reading at the same capacity

1. Freeze the four policies and original two pilot results. Preserve their source
   revisions and all negative outcomes.
2. Develop a structured learned record selector with exact value copying and an
   UNKNOWN option using training/validation only. Keep the current character
   reader and exact-rule reader as controls. Predeclare architecture, supervision,
   optimization budget, validation slices, and seeds before training it.
3. Retain the existing 95% overall visible-evidence gate per seed. Also inspect
   known answers, correct rejection, same-entity/different-attribute cases,
   updates/deletes, current-session precedence, and four-slot occupancy; report
   denominators and seed variation. Declare numerical slice gates before the
   next run rather than choosing them after seeing scores. The 95% number is our
   engineering criterion, not a universal research threshold.
4. If the reader passes, preregister fresh episode evaluation and rerun the paired
   memory comparison. The original 800 episodes are already inspected. A new
   episode seed using the same test symbols is not an independent symbol split.
5. Then compare bounded exact-key retrieval and admission alternatives, followed
   by learned admission/supersession on separately declared harder workloads.

**Immediate next action:** specify and implement the record-selection/copying
reader ablation. Its benefit remains a hypothesis; this research session has
diagnosed the failure and narrowed the next experiment, not fixed the model.
