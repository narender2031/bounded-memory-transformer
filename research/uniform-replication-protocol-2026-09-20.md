# Uniform-query replication protocol — 2026-09-20

**Status:** declared before generating any new held-out episodes. Commit this
protocol, configuration, implementation, and tests after review, then pass that
commit explicitly to the held-out runner. Training-only smoke checks may precede
the commit. This document declares the analysis; it contains no new test result.

## Motivation and hypotheses

**Existing evidence:** Project 03's frozen uniform A8 comparison was learned
minus FIFO +3.08 percentage points, with a positive paired 95% interval on 128
episodes. The three trained cue scorers produced the same greedy policy on shared
episodes. They are not independent dataset replications. Preserve that evidence,
all original files, and the failed neural-reader gates.

**Mathematical expectation:** conditional on a write prefix, a query-independent
bank containing K distinct, current keys among 24 keys answers a uniform probe
with probability K/24. For 32 independent uniform probes, expected episode recall
is K/24. Consequently every valid full K-key bank has the same expected recall,
regardless of its correlation with cues or write order. Expected paired policy
differences are zero. This follows by summing the K retained-key probabilities;
it does not require the policies to retain the same keys. The evaluator will
assert occupancy, distinct keys, and current values, since invalid or stale banks
would break the premise. The frozen prefix/probe streams use distinct seed strings.

**Hypothesis:** the prior positive interval reflects finite sampling, not a
reliable cue-based advantage under uniform probes. This replication estimates
the differences and assesses a declared practical-null margin. A nonsignificant
test alone will not be described as evidence of equivalence. Fresh episodes use
the existing held-out symbol partition; they are not a new vocabulary split.

## Fixed experiment

Use the unmodified Project 03 `generate_capacity`, `CapacityPolicy`, and
`evaluate_capacity`, and the original frozen `capacity-{7,19,43}.pt` checkpoints.
The [configuration](../projects/04-memory-followup/configs/uniform-replication.json)
pins every checkpoint and direct frozen dependency by SHA-256. No retraining,
parameter updates, learned readers, threshold calibration, or generator changes.

- Workload A only: 24 uniformly queried keys, 24 initial SET operations, six
  UPDATE operations, eight write sessions, then 32 probes with replacement.
- Twenty declared dataset seeds `2026092101` through `2026092120`, exactly 512
  episodes per seed: 10,240 independent episode draws and 327,680 probe draws.
  Seeds are fixed here, not selected by observed results. The original dataset
  seed `2026092013` is excluded.
- Four and eight canonical int32 slots: 64 and 128 persistent bytes. Every slot
  includes entity, attribute, current value, and observed binary cue. The same
  learned weights serve both capacities. No persistent counters or raw-history
  access are added. The bank is the only mutable episode state; hard session
  reset retains it and discards any context. The actor receives only each current
  write and its cue; future probes and the evaluator's world state are excluded.
- Compare learned, FIFO, recency, and stateless hash-random retention. Random
  policies keep original policy seeds 7, 19, and 43 and run as separate actors.
  Do not equate their three banks with a larger actor bank.
- Exact bank reader only. All uniform probes are historical-answerable, so
  useful-fact recall and latest-value accuracy have the same denominator. The
  generator has no deletions, unanswerable probes, or interleaved access. Existing
  lifecycle results cover those distinct conditions; this run adds no such claim.

Before dataset generation, enumerate every full-bank binary-cue configuration
and incoming cue at both capacities for every checkpoint. Require that frozen
learned allocation exactly matches cue-priority, including ties. Empty-row fill,
existing-key supersession, deletion, and reset use the same frozen shared code,
independently of learned scores. Because the scorer observes only the binary cue,
this establishes policy equivalence for the declared inputs. If equivalence
fails, abort before generating data and record the failure; do not silently
change the design.

Evaluate model seed 7 once per dataset/capacity and link seeds 19/43 and
cue-priority to those same predictions. Evaluate FIFO and recency once each, and
all three random policies separately. This is an execution optimization, not
additional replication. Preserve representative learned latency; do not invent
latency observations for aliases. Frozen `evaluate_capacity` also computes its
clairvoyant top-K oracle; retain its utility/regret only as evaluation diagnostics.
Oracle counts never enter an actor. Positive oracle regret does not imply a
learnable causal advantage.

## Predeclared inference

The six primary contrasts are learned minus FIFO, recency, and random at each
capacity. The random contrast is the mean of the three policy-seed differences
within each episode. Reusing a learned prediction or averaging random policy
seeds does not increase the number of independent observations. All policies and
capacities are paired on the same episodes.

For episode differences d_i, report mean(d), sample standard deviation s, and
SE = s/sqrt(n), using n = 10,240, not the number of policy/model/query rows.
Each two-sided primary interval is mean(d) ± z * SE, where
z = NormalQuantile(1 - 0.05/(2*6)). These are **large-sample normal intervals**
with a Bonferroni family of six; Bonferroni does not make the normal approximation
finite-sample exact. Also report ordinary 95% intervals. Differences and margins
are stored as fractions; multiply by 100 when describing percentage points.

The practical-null margin is **±0.01 absolute recall (±1 percentage point)**.
A primary contrast meets the equivalence criterion only if its entire corrected
interval lies strictly inside that margin. Directional evidence uses whether the
corrected interval excludes zero; tiny directional differences can coexist with
practical equivalence. Report those flags separately.

As a predeclared sensitivity, perform 10,000 paired nonparametric bootstrap
replicates, seed `2026092099`, resampling whole episodes within each dataset seed
and using the same resampled indices for every contrast. Use percentile endpoints
0.05/(2*6) and 1 - 0.05/(2*6). This also has approximate, not exact, simultaneous
coverage. A broad practical-null conclusion requires **all six** contrasts to
meet the margin under **both** the analytic and bootstrap intervals. Report any
disagreement without replacing the primary analysis.

Publish all 20 dataset-seed estimates, standard errors, and ordinary 95% intervals
for every contrast. These per-seed intervals are descriptive, unadjusted, and are
not used for significance counting, equivalence claims, selection, or stopping.
Also publish every policy's per-seed scores, all three random-policy scores, each
frozen model's explicit shared-prediction mapping, pooled policy means, expected
K/24 recall, and latency totals. No post hoc exclusions, extra seeds, selective
restarts, or seek-until-nonsignificant stopping. An engineering failure after
preflight leaves an incomplete manifest. Preflight hash/freeze/checkpoint failures
abort before output creation and are reported on stderr. Any amendment must be
documented before a new held-out run.

## Artifacts and validation

The runner refuses an existing output directory. It records the preregistration
commit, source/config/protocol/checkpoint hashes, Python/NumPy/PyTorch/platform
versions, CPU/thread settings, mode, and start/finish times. Held-out mode requires
the relevant source, config, protocol, and test files to match the supplied
commit. The manifest is incomplete until all predeclared runs and analysis finish.

For every dataset seed, save generated sessions/writes/cues/queries and raw
episode results (final banks, exact per-query predictions/truths, oracle
diagnostics, and measured times). Save all six actual policy executions at both
capacities; alias files are unnecessary because the mapping is explicit. Each
artifact gets a SHA-256 entry. Keep a separate uncompressed summary for independent
recalculation and reporting. Timing is observational and need not be deterministic.

Tests must catch incorrect episode pairing, inflated sample counts, wrong
interval scaling or multiplicity, equivalence misclassification, invalid/stale
banks, query-dependent bank changes, unmatched checkpoint behavior, hash/commit
mismatch, accidental test generation in smoke mode, and overwrite attempts.
Use deterministic hand-counted fixtures and training-only integration examples.

Implementation sequence: write the protocol/config; add failing tests; implement
the new runner and inference; run targeted tests, lint, and training-only smoke;
have the root agent review and commit the complete preregistration; only then run
the fixed held-out command. Keep source edits within the new uniform follow-up
module/test/config/protocol; the original experiment remains frozen.

```sh
env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python -m bounded_memory_transformer.memory_followup.uniform \
  --config projects/04-memory-followup/configs/uniform-replication.json \
  --output runs/uniform-replication-2026-09-20-smoke --smoke

env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python -m bounded_memory_transformer.memory_followup.uniform \
  --config projects/04-memory-followup/configs/uniform-replication.json \
  --output runs/uniform-replication-2026-09-20-v1 --held-out \
  --freeze-commit <reviewed-preregistration-commit>
```
