# Case 5B: emergent utility from bounded access history

Date: 2026-09-20. Protocol identifier: `case_5b_emergent_utility_v1`.
Status: **design only; no implementation, training, or results**. This is a
proposed preregistration for a new experiment. Freeze its executable generator,
configuration, source hashes, and analysis before opening held-out outcomes.

## Question and scope

Can a fixed-capacity policy infer which information will be useful from previous
accesses, transfer that policy to different workload mechanisms, and preserve
correct updates, deletions, and abstention?

The completed Case 5B in the historical five-case run used a planted binary cue
with 5:1 query weighting. Its learned policy exactly matched `cue_priority`.
Preserve that result and call it **planted-cue B**. The new identifier above does
not rename or replace it. Beating FIFO on planted-cue B is not evidence for the
new hypothesis. Neither this design nor the literature check establishes novelty.

Three approaches were considered: repeat the planted-cue task with more seeds;
use a conventional cache that refills missing values on reads; or infer demand
from interleaved accesses without refilling. Use the third for the strict study.
The first addresses replication, and the second changes the information contract.

Separate claims throughout:

- **Writer claim:** learned admission/eviction beats strong bounded policies with
  an exact visible-evidence reader. D012 permits this independently of neural
  reader competence.
- **Combined claim:** those gains survive a frozen neural reader that passes the
  existing absolute and oracle-relative 95% gates, plus the harm checks below.
- **Not tested here:** learning the semantic transition rules, latent compression,
  natural-language reasoning, or arbitrary future prediction. Shared exact
  full-key routing protects updates/deletes; that protection is an engineering
  control, not learned safety.

## Information and transition contract

Each main-workload session contains one operation, followed by destruction of
token context, activations, and the ordinary KV cache. Only serialized bounded
state survives. Policy weights are fixed during evaluation. The actor receives
only the current operation and its own bank. It cannot access the episode object,
world state, hidden workload identity, generation seed, time index, future events,
answer labels, oracle actions, or training returns.

| Operation | Required semantics |
|---|---|
| `SET e a v` | Authoritative current value. If resident, overwrite in place; otherwise it is an admission candidate. A later SET may recreate a deleted key. |
| `UPDATE e a v` | Authoritative replacement even when the old value was evicted. It supplies the complete new value; absence from the bank does not make it invalid. Resident keys must be overwritten; nonresident keys become admission candidates. |
| `DELETE e a` | Remove the matching resident value and its resident metadata immediately. An absent target is a no-op. No stale value or tombstone archive survives elsewhere. |
| `NOISE e a v` | Explicitly non-authoritative; do not store or mutate any factual value. Time-based metadata may advance. This is an easy semantic control; valid low-demand facts supply the harder pollution test. |
| `ASK e a` | Return the current value if supported by visible evidence, otherwise `UNKNOWN`. Observe this request as access history, but never receive its answer value or a refill. |

A missed ASK must not create a value record. Missing a live world fact is a
world-recall error even when `UNKNOWN` is the correct visible-evidence response.
Deleted/never-set world facts correctly yield `UNKNOWN`. Report these cases
separately. Repeating an ASK cannot recover an evicted value; a later explicit
SET/UPDATE can. Any future read-through/refill experiment needs a different
identifier and identical refill access for every policy.

After a nonresident authoritative write, admit to an empty slot if available;
when full, choose `IGNORE` or replace exactly one resident slot. Resident writes,
DELETE, and NOISE have the mandatory transitions above. ASK changes access
metadata only. These rules eliminate stale alternatives before utility selection.
The evaluator keeps an unlimited reference world solely to generate and score
events; neither actor nor shared feature code can call it.

## Exact persistent budget

Use four and eight symbolic value slots: **64 and 128 logical bytes**. Each
policy has `12*K` bytes for K triples of signed int32 `(entity, attribute, value)`
and `4*K` bytes for all metadata. Reserve an entity sentinel for an empty slot.
There are no extra recurrent vectors, global counters, clocks, pending items,
key maps, replay buffers, ghost lists, sketches, or cached reader activations.
Slot ordering is part of the serialized bank, not an external list.

The first learned controller stores four uint8 fields per slot:

| Field | Online update |
|---|---|
| `write_age` | Increment on each operation, saturating at 255; zero on authoritative write to that slot. |
| `ask_age` | Increment on each operation, saturating at 255; zero after a resident ASK. Initialize to 255. |
| `ask_count` | Increment on resident ASK, saturating at 255; initialize to zero. |
| `last_gap` | At a resident ASK after the first, copy the incremented `ask_age` before zeroing it. Initialize to 255. |

Overwrites retain a key's access statistics; deletion/eviction clears them.
New admissions start with `(0,255,0,255)`. `ask_count == 0` distinguishes never
accessed from an old access. Quantization and saturation are deliberate limits,
including on long episodes; never silently substitute unbounded Python integers.
No nonresident history exists in this controller. It cannot infer a forgotten
key's individual demand until its value is supplied again.

Alternative policies may spend their same metadata allowance differently. All
see the same raw operation stream; policy-specific compression is part of the
comparison. Immutable model parameters and temporary computation are reported
separately from episode state. Report allocated bank bytes, serialized bytes,
actual process/tensor peak memory, parameter bytes, latency, and throughput.
Historical tokens reprocessed at inference must be zero.

## Generator and family separation

Use opaque disjoint train, validation, and test entity/value vocabularies with
shared characters. Randomly rename keys/values per episode; no identity, numeric
ID, value, attribute class, insertion order, or explicit flag reveals popularity.
The actor uses identity only for exact full-key equality. Different attributes
of the same entity must occur. Distinct values for successive versions prevent
accidental stale-answer credit.

Start with 16 live keys drawn from a universe of 32 full keys, written in random
order. Reserve additional never-set keys and a deleted key created/deleted in the
preamble so every unknown-query stratum is available. The body interleaves all
five operations. Choose write/update/delete targets independently of hidden
popularity. Query mixtures are 75% live, 12.5% deleted, 12.5% never-set; live
requests follow the family process, renormalized over currently live keys.
If an operation has no legal world target, emit its declared fallback: SET to
UPDATE, UPDATE to SET, DELETE to NOISE, live ASK to never-set ASK. Record fallback
counts. A missing metric denominator is unavailable, never silently zero or a
reason to regenerate an episode.

Half of test episodes use each operation mixture below, balanced within each
generator block. Training uses the first mixture. Validation contains both.

| Mixture | SET | UPDATE | DELETE | NOISE | ASK |
|---|---:|---:|---:|---:|---:|
| Base | .10 | .15 | .10 | .15 | .50 |
| More lifecycle changes | .15 | .20 | .15 | .10 | .40 |

Use separate random streams for operation kinds, values, write targets, latent
demand, and query sampling. A family may control demand but not inspect policy
actions. Unknown/deleted queries do not advance the live-request process.

| Partition | Workload mechanisms and fixed parameter choices |
|---|---|
| Train | Equal mixture of stationary skew (random rank permutation; Zipf exponent .7 or 1.1) and geometric bursts (repeat previous live key with probability .6 or .8, otherwise draw uniformly from live keys). |
| Development validation | Fresh in-family examples plus a separate smooth-drift family: interpolate between two independent Zipf-1 rank permutations with weight `(1-cos(2*pi*q/64))/2`, where `q` is the evaluator's live-request ordinal. This family is never used for gradient updates. |
| Held-out T1: abrupt drift | Hidden hot sets of size 4 or 6 carry .8 demand; remaining live keys share .2. At each phase boundary draw a new hot set and independently choose the next phase length from {24,48} live requests. No boundary cue is emitted. |
| Held-out T2: recurrence | With probability .8, take the next currently live key from a hidden cyclic permutation of 10, 14, or 18 universe keys; with .2, draw uniformly from live keys. If the cycle has no live key, use the uniform draw. This is not a geometric burst. |
| Held-out T3: scan pollution | Alternate 24 hot-set requests (.8 mass on 4 or 6 hidden hot keys) and a scan of 8, 16, or 24 distinct live keys in random order, truncated when fewer live keys exist. Scan requests are for ordinary valid facts, not labelled NOISE. |

These are different temporal mechanisms, not just fresh RNG seeds. Hot-set mass
on an empty live subset goes to the other live subset; if the complement is
empty, all mass goes to the live hot subset. A burst whose previous key was
deleted falls back to uniform live demand. Test episode lengths
are 256 and 512 body operations, versus 128 in training. Report separate results
for each of the **12 family × capacity × length cells**, plus operation-mixture,
time-since-evidence, overwrite-count, and post-shift slices. Expose no phase clock
or family label to any policy. Do not tune T1/T2/T3 after viewing their outcomes.

Also preregister a new uniform interleaved control `EU1-U`: each live ASK is an
independent uniform draw over the live world. It complements, and does not replace,
the separate replication of historical uniform A. Conditional on a valid bank,
its expected live-query hit probability is `resident_live_keys / world_live_keys`.
Audit hit-minus-expectation residuals and paired policy differences. A 95%
interval entirely within ±1 percentage point supports equivalence; a wider
interval is inconclusive. Apparent systematic access-history prediction here is
a failure to investigate, not permission to tune the test generator.

## Required baselines and ablations

Use independent policy implementations and identical transition guards. Baseline
parameters are selected once on validation at K=4; carry them unchanged to K=8.

| Policy | Definition and state accounting |
|---|---|
| No memory | Current operation only; establishes recall/abstention denominators. |
| FIFO | Evict earliest admitted resident key; updating a resident does not refresh insertion order. |
| Write recency | Evict least recently authoritatively written key; ASK does not refresh it. |
| LRU | Evict least recently accessed key, where an authoritative write or resident ASK is an access. Store exact order by rearranging bank rows, without external timestamps. |
| Resident LFU | Always admit, evict smallest resident ASK count, with LRU ties. Use the four metadata bytes as a saturating uint32 count. Evicted counts disappear. |
| Decayed LFU | Always admit; on every ASK decay each resident Q16.16 counter by `2**(-1/H)`, then add one for a hit. Compare H in {8,32,128} on validation. Four bytes per resident; LRU ties. |
| Stateless hash random | Always admit and choose a victim by a fixed hash of current operation, serialized bank, and an immutable policy seed. No hidden RNG/clock state; describe this deterministic seeded variant accurately. |
| Recency-frequency hybrid | Use the learned controller's four fields; score `a*log1p(count)/log(256) + (1-a)*2**(-min(write_age,ask_age)/H)`. Choose a in {0,.25,.5,.75,1} and H in {8,32,128} on validation; score the incoming candidate too, breaking admission ties in its favor. |
| TinyLFU-style admission + LRU | Preserve K value triples. Spend the `4*K` metadata bytes on a uint16 sample counter and the remaining bytes on packed 4-bit frequency counters, two fixed hashes, conservative increment, saturation at 15, and halving every W in {32,64,128} ASK samples; reset the sample counter to zero after halving. Count all ASK keys, including misses; never refill. Compare candidate/victim estimated demand only on writes, admitting on ties. Exact LRU order uses row order. No Bloom doorkeeper. This is a small declared adaptation, not a reproduction of TinyLFU. |
| Learned, no access history | Same architecture, optimization budget, and capacities as learned; hold ASK-related fields at their initial constants during both training and evaluation. Keep write age. |

All grids and tie rules must be frozen before tests. Report the strongest member
of this entire fixed baseline set in each test cell; never select only a favorable
competitor. Include each baseline's own paired comparison.
The shared sketch may retain demand counts after a deletion; those charged,
collision-prone statistics contain no value and cannot certify factual validity.
Deletion still removes the value immediately.

ARC and LeCaR motivate adaptive competitors, but their ghost histories and adaptive
state do not fit for free. A full ARC/LeCaR implementation must either trade value
slots for metadata at the same total bytes or be a separate larger-budget frontier.
A truncated-ghost adaptation needs its own exact layout and name. Do not label a
ghost-free heuristic ARC or LeCaR. The required bounded TinyLFU-style baseline
already tests whether recording nonresident demand beats resident-only learning.
See the [primary-source note](reading-notes/2026-09-20-emergent-utility-prior-art.md).

## First learned candidate and causal feedback

Use a shared two-hidden-layer, width-32 MLP to score each resident and the incoming
candidate. Inputs are normalized bounded fields, a candidate flag, occupancy,
operation kind, and mean/max summaries of resident fields recomputed from the
bank. Never input raw numeric key/value IDs. The selected action removes the
lowest-score item from the K+1 choices; removing the candidate means IGNORE.
Empty-slot admission and semantic guards remain shared. A linear scorer ablation
tests whether the result requires more than a learned recency/frequency blend.
This writer study does not require a new Transformer architecture.

Train stochastic decisions with return-to-go policy gradients. At the time each
ASK actually occurs, the evaluator supplies only a training reward: +1 for a
correct world answer, 0 for UNKNOWN on a live missing fact, and -1 for an incorrect
non-UNKNOWN answer. Rewards never become actor inputs. Future requests can affect
earlier gradient credit only after being observed in the rollout; do not give
future-frequency, next-use, hidden-family, or oracle-action labels. The exact
reader and guards make the last penalty an integrity check in the writer stage.
At evaluation, use frozen weights and greedy decisions. Adaptation then comes
from changing bounded access summaries, not from test-time gradient updates.

Candidate executable configuration to implement and freeze:

```yaml
protocol: case_5b_emergent_utility_v1
state: {train_slots: 4, evaluation_slots: [4, 8], total_bytes_per_slot: 16}
reader_for_writer: exact_visible
refill_on_ask: false
body_operations: {train: 128, validation: 192, test: [256, 512]}
model_seeds: [17, 29, 43, 71, 101]
test_generator_blocks: [701, 709, 719, 727]
episodes_per_test_family_length_block: 128
training: {episodes_per_seed: 8192, batch_episodes: 32, hidden_width: 32}
optimizer: {name: Adam, learning_rate: 0.003, gradient_clip: 1.0}
policy_gradient: {discount: 1.0, entropy_weight: 0.01, baseline: leave_one_out_batch}
checkpoint_selection: validation_macro_world_recall_at_four_slots
bootstrap: {replicates: 10000, seed: 92026}
minimum_mean_gain: 0.02
```

Train all seeds separately. Share held-out episodes across policies and model
seeds for paired comparisons, while retaining four independent generator blocks.
Do not count repeated evaluation of an identical greedy policy as new dataset
replications. Validation has 256 episodes per declared family/mixture. Separate
training/validation RNG namespaces from every test block. The same learned
weights are used at eight slots; no capacity-specific tuning.

## Feasible oracle and honest headroom

An offline top-K count is invalid for interleaved writes, reads, updates, and
deletes. Instead, use exact dynamic programming on an explicitly smaller oracle
panel: 12 universe keys, 8 initially live keys, 64 body operations, 16 episodes
per held-out family, both capacities, and independent seeds 809/811/821 for
T1/T2/T3 respectively. Use hot sets of size 2/4, T1 phases of 8/16 live requests,
T2 cycles of 6/8/10 keys, and T3 scans of 4/8/12 keys. These panel-only size
adaptations are fixed in advance. Start the oracle at the empty bank and include
the entire preamble and initial writes. Enumerate reachable bank key
sets and only the legal transitions above. Values at each time come from the
reference world, but an absent value may enter only on its SET/UPDATE. ASK never
adds a key. Resident updates and deletions are mandatory. Maximize the sum of
correct live-query answers over the known suffix, and save the complete optimal
action path for replay through the ordinary transition validator.

The oracle uses the same value capacity and allocates the same metadata budget;
future access is its explicitly disclosed informational advantage. Verify the DP
against exhaustive action enumeration on smaller traces. Measure panel results
separately. Do not extrapolate its gap to the long main suite. A full-suite oracle
is unavailable unless a solver certifies optimality and its path passes replay;
a best-so-far feasible trajectory is a lower bound on the optimum, not an upper
bound for other policies. A numerical relaxation bound must be labelled separately.

Report oracle regret only where this feasible optimum is certified. Random future
choices and abrupt unannounced changes can create irreducible oracle advantage.
No experiment may call all oracle headroom learnable or require its elimination.

## Metrics, statistical criterion, and harm gates

The primary writer metric is each episode's fraction of live ASK answers that
match the current world value, macro-averaged over episodes within each test cell.
With one operation per reset, these live ASK queries require historical evidence.
Also report query-micro averages, total world accuracy, retained-evidence reading
accuracy, abstention by world/evidence status, updated-key recall, stale answers,
deleted leakage, wrong-key copying, and all resource costs. An independent auditor
must reconstruct these from recorded operations, before/after banks, actions, and
answers, without importing experiment metric functions.
Define stale error as returning a superseded value of the queried key while a
different value is current. Deleted leakage is any non-UNKNOWN answer for a
currently deleted key; also count exact disclosure of its old value separately.
Wrong-key copying returns a value supported only by another full key. A live
miss followed by UNKNOWN is an evidence-correct abstention but a world-recall
failure. Report the denominator for each measure.

For a **robust writer result**, require all of the following, declared in advance:

1. In every one of the 12 test cells, learned mean recall exceeds every required
   heuristic and the no-access ablation by at least 2 percentage points.
2. All those paired differences have positive simultaneous one-sided 95% lower
   confidence bounds. Use a joint max-t bootstrap over the fixed comparisons,
   resampling model seeds and generator blocks as crossed factors, then episodes
   within blocks. Preserve policy, capacity, length, and shared-prefix pairing.
   If a contrast has zero estimated variance, use an exact paired check or mark
   its interval unavailable; do not divide by zero or assign significance.
3. At least four of five individual model seeds have positive mean differences
   against the strongest required heuristic in every test cell. Report all five
   and the minimum; do not hide a failed seed behind episode counts.
4. Contract/semantic acceptance tests pass and the uniform control supplies no
   unresolved evidence of leakage or systematic predictable future advantage.

This is intentionally stricter than a positive pooled mean. If only some cells
pass, report a workload-specific result. Confidence is limited to these declared
families, budgets, and seeds; it is not a guarantee for arbitrary distribution
shift. The linear scorer is an explanatory ablation, not a weak comparator used
to replace the required baselines.

For a **combined result**, reuse a frozen competent reader and measure its exact
visible-evidence counterpart on precisely the same banks. Every seed must reach
95% overall and on every declared nonempty slice, including full-key mismatch,
precedence, unsupported, contradicted, irrelevant, deleted, occupancy, and unseen
symbols. Require oracle-relative recovery
`(A_neural - A_current_only) / (A_exact_visible - A_current_only) >= .95`
on reader microtasks and every main test cell with positive headroom. Nonpositive
headroom or empty slices are unavailable, never a pass. Apply the superiority
criterion again to combined answers; writer superiority alone is insufficient.

Predeclare additional independent, one-query harm challenges, 1,024 episodes per
stratum and seed: stale value after UPDATE, deleted fact, unrelated memory,
current-session contradiction/precedence, and unsupported key. These challenge
sessions may contain `SET old -> UPDATE new -> ASK` or `SET old -> DELETE -> ASK`;
the earlier value remains in the visible current prefix so rejection is not
trivial merely because the bounded writer erased its bank copy. Score current
evidence before any reset. Current-only and memory-enabled readers see identical
current input and weights. Report paired memory harm (current-only correct,
memory wrong), benefit, and abstention coverage. Require zero observed stale-value
answers and deleted-value disclosures, with the one-sided 95% binomial upper
bound below 1% on their independent challenge episodes; require paired memory
harm's upper bound below 1% in each challenge stratum. Report all errors and
denominators. Use Bonferroni-adjusted one-sided bounds across the 35 seed/endpoint
checks (five paired-harm strata and two zero-disclosure checks per seed), so the
collection has at least 95% simultaneous coverage. This gate is stronger than
the 95% reader gate and still does not
prove zero real-world harm. Long-episode harms remain reported separately with
episode-cluster uncertainty. Shared guards passing cannot establish learned
supersession or replace these reader checks.

## Stages and independent acceptance tests

Keep the M2 Pro/16 GiB scope small: vectorize short rollouts, use CPU for the
small writer, run seeds sequentially, and stream compressed audit records. Do not
launch a large hyperparameter search or latent-memory model as part of EU1.

1. **Semantics first:** generator, byte serializer, exact reader, and independent
   baselines; short deterministic traces, no learned result.
2. **Development only:** validate the access signal on SET/ASK traces, then train
   on the full operation mixture. Compare learned/no-access/linear candidates
   and baseline grids using training/validation data only. Record failed ideas.
3. **Freeze:** lock code/configuration, vocabularies, seeds, complete baseline set,
   harm challenges, and analysis. Any subsequent scientific change needs a new
   protocol version and fresh held-out streams; do not overwrite EU1.
4. **Evaluation:** all five seeds and twelve cells, EU1-U, feasible oracle panel,
   independent audit, and compute report. Test outcomes cannot select checkpoints.
5. **Combination:** only claim a combined system if its reader, superiority, and
   harm gates pass. Preserve an independent writer finding if they fail.

Acceptance tests must be written independently of policy/metric implementations:

- Serialize/restore after **every** operation into a fresh policy object; require
  identical subsequent actions and answers. Assert exactly 64/128 state bytes,
  no other mutable episode fields, empty KV/context, and zero historical replay.
- Run two different unseen suffixes after an identical prefix: pre-branch actor
  observations, actions, and bank bytes must be identical. Changing future queries,
  oracle labels, or world evaluator objects must not affect the prefix.
- `SET -> evict -> ASK -> ASK` remains UNKNOWN; `UPDATE(new value) -> ASK` can
  recover only if that supplied value is admitted. ASK must have no value field.
- Resident UPDATE removes the previous value; evicted-key UPDATE remains an
  eligible authoritative write; DELETE invalidates only the target; repeated
  DELETE is idempotent; SET after DELETE recreates; NOISE preserves values.
- Hand-traced FIFO/LRU/LFU/decay/hybrid/sketch examples distinguish insertion,
  write recency, and query recency; exercise collisions, counter saturation,
  halving, deletions, and metadata reset. Test byte costs of every variant.
- Non-hash policies are invariant to injective key/value renaming; the scorer is
  invariant to slot permutation except declared tie-breaking. Hash policies must
  remain deterministic and be statistically exchangeable over independent
  renamings, not assumed pointwise identical.
- Check disjoint symbol sets, family manifests, seed namespaces, workload
  fallbacks, live/deleted/never-set denominators, and independence of uniform
  queries from prior access summaries.
- Exhaustively verify small oracle traces, reject illegal reacquisition, and
  replay each certified oracle trajectory through the independent transition
  validator. No oracle object may enter actor construction or feature extraction.
- Recompute harm/recovery/bootstrap inputs from raw records; flag missing cells,
  unavailable denominators, duplicate episode identities, and identical policies
  evaluated under nominally different seeds.

**Evidence:** planted-cue learning currently matches its strongest cue heuristic.
The old neural reader failed its full gates; the [structured follow-up reader](reader-followup-results-2026-09-20.md)
now passes its declared suite, and the independent uniform replication meets its
equivalence criterion. These do not test EU1. **Inference:** access-conditioned
selection and reader competence require separate tests. **Hypothesis:** a learned
rule over a small causal access summary can transfer better than strong cache
heuristics while an independently tested reader avoids harmful memory use.
EU1 has not yet supplied evidence for that hypothesis.
