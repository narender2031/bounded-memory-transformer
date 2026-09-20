# Results summary: when bounded memory helps and hurts

Prepared 2026-09-20 from experiments run on 2026-09-17. The recorded predictions
were rechecked for this summary; no new model was trained. The primary result is
the corrected v2 pilot, with the original v1 failure and subsequent diagnostic
preserved. Percentages are rounded from the JSON using ordinary half-up rounding;
the source artifacts retain full precision.

**Main finding:** adding memory reduced our current Transformer's accuracy by
about 20 percentage points, while an exact-rule reader improved on the same
stored evidence. There are two substantial limitations: capacity loses useful
facts, and the neural reader often misuses the evidence that remains. We have
measured the phenomenon, built controls that expose it, and narrowed the next
experiment. We have not yet demonstrated a better learned memory controller.

## 1. What was implemented and tested

Project 01 supplies an inspectable decoder-only Transformer with manual causal
attention, embeddings, normalization, residual connections, a feed-forward
network, training, and generation. Its original 27,520-parameter CPU smoke run
reduced validation loss from 3.5936 to 2.7913 over 50 steps. This established
working optimization on the bundled toy corpus, not broad language competence.

Project 02 reuses the architecture as a reader alongside independent memory
policies. The actual memory experiment configuration was:

| Component | Setting |
|---|---|
| Reader parameters | 237,792, trained from scratch |
| Architecture | Width 96, two decoder blocks, four heads, 128-token maximum context |
| Input/output | 25-character vocabulary; two-character value or `??` |
| Training | 6,000 steps, batch 64, AdamW learning rate 0.002, zero dropout |
| Training data | 65,536 reading microtasks per seed; sampled with replacement |
| Model seeds | 7, 19, 43 |
| Reading validation | 512 microtasks shared across seeds |
| Main workload | 30 candidate operations across eight sessions |
| Memory | Four symbolic records, 64 logical persistent bytes |
| Episode evaluation | 800 episodes per scenario; three scenarios |
| Evaluation seed | 1729, with each scenario's declared configuration |
| Symbol split | 60 train / 20 validation / 20 test identifiers, independently for entities and values |
| Hardware used | Local M2 Pro MacBook, 16 GiB, Apple MPS |

The 30 candidates include repeated assignments and labelled noise. They are not
30 distinct simultaneously valid facts. Complete identifiers are disjoint across
splits; their individual digit characters are shared.

Within each episode, only the explicit bank carries historical information
between sessions. The reader has stateless forward passes and no implemented KV
cache. Model weights remain frozen during evaluation. Raw historical replay is
zero; reading serialized bounded records is counted separately.

Each record contains operation, entity, attribute, and value as four int32 fields.
Four records therefore consume 64 payload bytes. This is not total application
RAM: model weights, activations, and Python/framework overhead are separate.

## 2. The four policies and two readers

| Policy | Stored evidence | Evidence returned to reader |
|---|---|---|
| No memory | None from earlier sessions | Current session only |
| FIFO | Latest four factual events, including deletion markers | All retained events in order |
| Recency | Latest event for up to four distinct entity/attribute keys | All retained keys |
| Similarity | The same four-event FIFO bank | One record with greatest positional character overlap with the query key |

All policies ignore explicitly labelled NOISE. Recency already handles same-key
replacement and deletion markers. It is write recency, not access-based LRU.
Similarity uses lexical matching, not pretrained embeddings or an archive.
LRU, learned admission, and learned latent memory are not implemented baselines.

The **neural reader** learns to predict an answer from serialized visible evidence.
The **exact-rule reader** scans those same records and current operations, matches
the complete entity/attribute key, applies assignments/deletions, and returns a
value or unknown. It cannot access discarded facts. A separate full-history
state machine supplies scoring truth and is never an input to either reader.

One trained reader per seed is reused unchanged across all policies. Thus the
policy comparison does not give each baseline a differently trained network.

## 3. Main result: memory hurts the current neural reader

| Policy | Neural episode accuracy | Change versus no memory | Exact-rule episode accuracy |
|---|---:|---:|---:|
| No memory | 68.71% | — | 75.00% |
| FIFO | 48.50% | −20.21 pp | 82.50% |
| Recency | 48.58% | −20.13 pp | 82.50% |
| Similarity | 48.79% | −19.92 pp | 82.50% |

The neural scores average three models evaluated on the same 800 episodes. These
are 2,400 predictions per policy, not 2,400 independent episode draws. The rule
reader is deterministic and is evaluated once per episode.

**Evidence:** the direction of harm holds in every trained seed:

| Seed | No memory | FIFO | Recency | Similarity |
|---|---:|---:|---:|---:|
| 7 | 68.25% | 53.38% | 53.50% | 52.75% |
| 19 | 71.38% | 50.50% | 50.63% | 50.75% |
| 43 | 66.50% | 41.63% | 41.63% | 42.88% |

The original proposed illustrative 10M-parameter/79%/82% learned-memory table
has not been realized. The table above contains our actual measured results.

![Recorded pilot accuracy and paired memory effect](../projects/02-memory-benchmark/results/when-memory-hurts.png)

## 4. Why 82.5% can coexist with only 30% useful-fact retention

The main set has eight equally represented cases, 100 episodes each. Four have
an unknown world-truth answer: never-known key, similar wrong key, historical
deletion, and current deletion. Two provide a current-session assignment. Only
the remaining historical SET and UPDATE cases require a retained earlier value.

An exact no-memory reader therefore solves six of eight case categories:

\[
50\%\ \text{unknown cases} + 25\%\ \text{current assignments} = 75\%.
\]

Among the 200 historical-answerable episodes, the four-slot banks retain the
needed current value in 60 cases: **30% retention**. Exact reading adds these
60 successes to the no-memory reader's 600:

\[
\frac{600+60}{800}=82.5\%
\quad\text{or}\quad 75\%+25\%\times30\%=82.5\%.
\]

The remaining 140 errors are historical facts unavailable to the exact reader.
Thus 82.5% is not an 82.5% success rate at remembering old facts. It is overall
accuracy under this deliberately chosen mixture. A workload with more questions
requiring past evidence would have a different overall score.

**Inference:** storage selection can still improve even though the exact reader
benefits from today's policies. The neural reader is an additional limitation,
not the only limitation. The exact reader is a reference, not an upper bound on
an arbitrary system that might guess unavailable values.

## 5. Harm and benefit measured on paired questions

Memory is harmful on a question when no memory is correct and memory is wrong.
It is beneficial when no memory is wrong and memory is correct. Both rates use
all queries as denominator, including cases where the two systems agree.

| Policy | Harmful | Beneficial | 95% interval for accuracy change, pp |
|---|---:|---:|---:|
| FIFO | 23.21% | 3.00% | [−22.63, −17.75] |
| Recency | 23.17% | 3.04% | [−22.50, −17.63] |
| Similarity | 28.13% | 8.21% | [−23.33, −16.46] |

For example, FIFO's 3.00% benefit minus 23.21% harm yields its approximately
20.21-point net decline. Similarity helps more questions, but harms more too.
Its slightly higher overall mean is not evidence of a robust policy ranking.

Intervals bootstrap paired episodes while keeping all three model seeds for an
episode together. They describe episode sampling uncertainty conditional on
these models; they do not characterize all possible training seeds or tasks.

## 6. The case breakdown explains much of the decline

| Main case | No-memory neural | FIFO neural | Similarity neural | FIFO exact-rule |
|---|---:|---:|---:|---:|
| Historical SET | 0.67% | 2.00% | 21.67% | 27.00% |
| Historical UPDATE | 1.67% | 10.33% | 30.33% | 33.00% |
| Historical DELETE | 88.33% | 46.33% | 42.33% | 100.00% |
| Never-known key | 88.00% | 37.00% | 20.00% | 100.00% |
| Current SET | 83.67% | 67.33% | 74.67% | 100.00% |
| Current UPDATE | 99.33% | 91.33% | 91.67% | 100.00% |
| Current DELETE | 100.00% | 100.00% | 100.00% | 100.00% |
| Similar-but-wrong key | 88.00% | 33.67% | 9.67% | 100.00% |

Each row contains 100 episode structures, answered by three neural models.
The full JSON includes recency, whose aggregate and case behavior is very close
to FIFO in this main workload.

Three observations matter. First, similarity substantially improves historical
SET/UPDATE answers compared with no memory. Second, it performs very poorly when
the closest record is about the wrong key. Third, memory can lower accuracy even
when the correct updated value is explicitly available in the current session.

**Inference:** evidence discrimination and abstention deserve immediate work.
The retrieval procedure always returns a record from a nonempty similarity bank;
an absent exact key does not make it return nothing. The reader must reject that
record. These observations do not identify a specific defective attention head,
prove that all retrieval is harmful, or establish that storage is stale.

## 7. Stale values, deletion leakage, and abstention are different failures

| Main metric | No memory | FIFO | Recency | Similarity |
|---|---:|---:|---:|---:|
| Stale-value answer rate | 0.11% | 0.33% | 0.33% | 3.22% |
| Deleted-query non-abstention | 5.83% | 26.83% | 26.83% | 28.83% |
| Abstention recall | 91.08% | 54.25% | 54.58% | 43.00% |
| Abstention precision | 66.69% | 76.32% | 76.34% | 83.36% |
| Abstention F1 | 77.00% | 63.42% | 63.65% | 56.73% |

Definitions and denominators:

- **Stale-value rate:** an output matches an obsolete value on a currently
  answerable, overwritten query. There are 900 such predictions per policy
  across the seeds; stale counts are 1, 3, 3, and 29. This measures a value match,
  not proof that the model causally retrieved an old record.
- **Deleted-query non-abstention:** any output other than `??` after a deletion,
  over 600 predictions per policy. It includes arbitrary wrong answers, not only
  verbatim repetition of the deleted value.
- **Abstention recall:** how often the reader says unknown on the 1,200
  world-truth-unknown predictions. Similarity's 43% means it fails to abstain on
  57% of those cases.
- **Abstention precision:** how often an abstention corresponds to an unknown
  world truth. It may rise while recall falls. Appropriately abstaining because
  an answerable fact was evicted is still incorrect under episode truth; visible
  reading competence therefore needs its own evaluation.

The exact-rule readers have zero stale-value errors and zero deletion leakage
in these runs. The major observed neural decline is not mostly captured by the
narrow stale-value metric. Low stale rate alone would conceal substantial harm.

## 8. Reader validation is a separate test, and every seed failed the gate

| Model seed | Short-task visible-evidence validation | Required threshold |
|---|---:|---:|
| 7 | 78.52% | 95% |
| 19 | 74.61% | 95% |
| 43 | 73.83% | 95% |

Validation asks what the provided records support. Episode accuracy asks for the
current fact according to full historical truth. The 74–79% validation scores
and the 82.5% exact-reader episode score have different examples and targets.
They cannot be subtracted as a direct comparison of reading ability.

The 95% gate is our predeclared engineering criterion, not a universal theorem.
Because all models failed it, neural memory harm alone does not justify claiming
a learned write controller is needed. The next reader must also be checked on
known answers, rejection, full occupancy, key discrimination, and updates/deletes.

On the actual main episode prompts, agreement with the visible exact reader is
89.96% without memory, 54.42% with FIFO, 54.50% with recency, and 51.33% with
similarity. This further shows that short-task validation is not sufficient
evidence of competence on the full episode prompt distribution.

## 9. Longer histories and more updates

| Scenario | No memory | FIFO | Recency | Similarity |
|---|---:|---:|---:|---:|
| Main: 30 operations, 8 sessions | 68.71% | 48.50% | 48.58% | 48.79% |
| Longer: 60 operations, 16 sessions | 67.71% | 46.46% | 46.38% | 44.42% |
| More updates: four overwrites in update cases | 68.67% | 48.17% | 48.63% | 48.25% |

All retain the same four slots. The exact reader scores 79.25% with memory on
longer episodes; historical useful-fact retention falls from 30% to 17%.
On the more-updates set, exact FIFO/similarity score 84.38% and recency 84.63%,
with historical retention 37.5% and 38.5% respectively.

**Evidence:** neural harm persists across the three workloads. **Limit:** these
are generated scenario sets, not a paired intervention holding every other event
constant. The higher exact-reader score with more updates does not prove that
updates inherently improve memory; timing and sampled evidence differ. This is
not the planned comprehensive adversarial suite or a memory-capacity sweep.

## 10. Frozen-reader diagnostic: unfamiliar entities versus values

We subsequently generated 2,048 reading structures and independently renamed
entities and values into training or validation vocabularies. Equality, answer
availability, operation order, prompt length, and visible occupancy were fixed.
Renaming occurred after retrieval to isolate the reader. There was no retraining
and no use of the test-symbol partition.

| Entity vocabulary | Value vocabulary | Overall | Known visible answer | Required abstention |
|---|---|---:|---:|---:|
| Familiar | Familiar | 78.91% | 77.07% | 81.08% |
| Unfamiliar | Familiar | 78.66% | 76.68% | 81.01% |
| Familiar | Unfamiliar | 75.49% | 69.73% | 82.29% |
| Unfamiliar | Unfamiliar | 76.06% | 70.63% | 82.46% |

These means combine the three saved readers on matched structures. "Familiar"
means training-vocabulary symbols, not necessarily exact previously trained
examples. The four conditions produce 24,576 predictions in total.

**Evidence:** failure persists with familiar symbols. Entity renaming alone has
a small effect here; unfamiliar values are more damaging. **Inference:** we need
to investigate value handling as well as selection and rejection. This experiment
does not isolate output copying alone, because values change in both input and
output.

With both vocabularies unfamiliar, accuracy is 84.43% when a known answer is
supplied by the current session, versus 48.51% when available only in memory.
Four-visible-record tasks score 63.14%. These subsets contain different task
mixtures; they are descriptive diagnostics, not causal estimates of position or
occupancy effects. The diagnostic used one structural sample and renaming scheme.

## 11. What went wrong in the first pilot

The v1 pilot trained for 1,200 steps on 16,384 microtasks. Its distractors never
shared the queried entity, allowing a reader to ignore the attribute completely.
An attribute-blind rule scored 512/512 on its validation set. That exposed a
coverage defect in the competence test.

The correction added same-entity/different-attribute distractors. A regression
check now finds 97 counterexamples among 512 validation tasks for the blind rule.
Before the v2 test, the corrected generator and larger budget were recorded.
The held-out episode generator and metrics were unchanged; episode seed changed
from 424242 to 1729, using the same test-symbol partition.

V1 validation was 68.95%, 68.36%, and 59.18%. V2 improved those scores but still
failed the gate. Since data coverage, training budget, and episode draw changed,
the two runs do not isolate the effect of longer training. They neither prove
that more training cannot help nor that one architecture change will succeed.

Both pilots remain available. Exact historical source revisions are `7fc51fd`
for v1 and `280d88c` for v2; current corrected code does not reproduce the flawed
v1 merely by choosing its shorter configuration.

## 12. Local feasibility and verification

The corrected run took **169.67 seconds** from recorded start to finish,
excluding interpreter startup. Each model's training plus validation took about
51 seconds. The later frozen-reader diagnostic took 3.39 seconds in its measured
region. These are observed local timings, not hardware limits or throughput
guarantees for larger future models.

Equal stored capacity did not imply equal reading cost. FIFO and recency
serialized 28 memory characters per full-bank read; similarity serialized seven.
Main mean prompt lengths were 13.23/41.23/41.23/20.23 for no memory/FIFO/recency/
similarity. Full timing fields are retained in the JSON; tiny batched per-query
averages should not be interpreted as single-request service latency.

The repository has 40 passing tests covering model behavior, semantics, reset
boundaries, policies, metrics, and small training/save/reload paths. Such tests
establish implementation properties, not research success. For this summary,
all 12 scenario-policy accuracies, paired harm/benefit rates, case slices, and
exact-reader scores were independently recalculated from 28,800 saved v2
predictions. Diagnostic prediction hashes were also checked.

## 13. What the new paper review contributes

The [six-paper review](reading-notes/2026-09-20-six-paper-review.md) checks the
user-supplied SeDeM, Metis, TARL, MemOps, rollback, and MetaKV sources. Its changes
to our interpretation and proposed work are:

- Correct Metis's old classification: it is an empirical native-memory prototype.
  Its interference and unrelated-memory controls are relevant to our failure modes.
- Use SeDeM's separation of stored representation and reader conditioning, with
  matched input/supervision controls. Top-k retrieval is not a persistent-state
  bound, and symbolic exact values do not yet require latent decompression.
- Use TARL to clarify operation and slot-target semantics. Its five actions do
  not include explicit deletion; our adapted controller must. Pending/rejected
  records cannot become free external storage.
- Adapt MemOps-style operation and trajectory diagnoses with exact synthetic
  scoring. Keep existing stale/deletion metrics and add unrelated-fact retention
  when a declared task supports it.
- Defer rollback until the benchmark includes derived claims and bounded
  dependencies. Model-accessible history replay would violate our strict mode.
- Keep MetaKV as deployment context, outside semantic memory baselines.

These are research directions, not results obtained by our implementation.
Published scores from different tasks/models are not comparable with our table.

## 14. What is established and what is still open

**Established locally:** the experiment runs on the MacBook; hard-reset bounded
controls are implemented; the current neural reader is harmed by adding memory
across all three seeds and workloads; the exact reader benefits from retained
evidence; unfamiliar symbols alone do not explain neural failure.

**Reasonable inference:** both retention and reliable use matter. The current
neural results are strongly affected by reading and evidence-rejection errors.
An improved reader is needed to interpret a combined neural writer/reader result.

**Still hypotheses:** learned admission beats strong hand-written policies;
latent compression is advantageous; a copy-based reader clears the gate; a
five-action writer, reconstruction loss, confidence field, or decompression
module improves our task. No novelty, natural-language generalization, or
4/8/16/32-slot scaling result has been demonstrated.

The next neural experiment should compare the current character reader with
learned record selection, exact value copying, and UNKNOWN at the same four-slot
budget. Include current-session records, predeclare training and validation
criteria, and disclose selector supervision. The exact-rule reader can support
parallel investigation of bounded writing without conflating writer and reader
errors. Freeze the original pilots and evaluate future changes on a separately
declared test.

## Source artifacts

- [V2 configuration](../projects/02-memory-benchmark/configs/reader-extended.json),
  [full results](../projects/02-memory-benchmark/results/2026-09-17-v2.json), and
  [generated report](../projects/02-memory-benchmark/results/2026-09-17-v2.md).
- [V1 results](../projects/02-memory-benchmark/results/2026-09-17-v1.json) and
  [source/prediction provenance](../projects/02-memory-benchmark/results/provenance.json).
- [Diagnostic configuration](../projects/02-memory-benchmark/configs/reader-diagnostic.json)
  and [results](../projects/02-memory-benchmark/results/2026-09-17-reader-diagnostic.json).
- [Earlier interpretation and alternatives](reading-notes/2026-09-17-memory-improvements.md),
  [fundamentals guide](transformer-fundamentals.md), and [decisions](decisions.md).
- Local complete predictions/checkpoints: `runs/when-memory-hurts-2026-09-17-v1/`,
  `runs/when-memory-hurts-2026-09-17-v2/`, and `runs/reader-diagnostic-2026-09-17/`.
- [Review PR #2](https://github.com/narender2031/bounded-memory-transformer/pull/2).
