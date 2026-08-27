# Initial Experiment Plan

Last updated: 2026-08-27

## Goal

Determine whether learned memory admission and supersession improve latest-fact recall under a strict fixed capacity, without raw-history access.

## Stage 1: Synthetic state-tracking environment

Start with a symbolic language so pretrained linguistic knowledge cannot conceal memory failures.

### Operations

```text
SET entity attribute value
UPDATE entity attribute value
DELETE entity attribute
NOISE entity attribute value
ASK entity attribute
```

### Initial episode configuration

- 8 sessions per episode.
- 1–4 operations per session.
- Hard context reset after each session.
- 20 candidate facts.
- 4 persistent memory slots.
- At least one update in 50% of episodes.
- At least one unanswerable or deleted query in 25% of episodes.
- Distractor rate varied independently of episode length.

These are starting values, not fixed conclusions.

### Generalisation splits

1. Unseen entity symbols.
2. Unseen value symbols.
3. Longer episode lengths.
4. More updates per key.
5. More candidate facts at the same memory capacity.
6. Paraphrased natural-language rendering of the symbolic operations.

## Stage 2: Baselines

All bounded baselines receive the same number and width of memory slots.

| Baseline | Purpose |
|---|---|
| No memory | Confirms hard reset removes required evidence. |
| Full history | Accuracy upper reference with unbounded historical replay. |
| FIFO slots | Tests recency-only replacement. |
| LRU slots | Tests query/access-based recency. |
| Oracle utility | Upper bound when the system knows which future query will be asked. |
| Oracle semantic update | Upper bound when updates are routed to the correct existing key. |
| RMT-style memory tokens | Learned recurrent compression baseline. |
| Learned admission only | Tests selection without explicit supersession. |
| Learned admission + supersession | Target method. |

H2O-style accumulated-attention retention and EXPIRE-SPAN-style learned lifetimes should be added after the simple policies are validated.

## Stage 3: Candidate target architecture

Each slot contains a latent representation plus explicit controller state:

```text
slot = content + key + validity + age/version + confidence
```

The exact representation is an experimental choice. Metadata may be continuous and learned rather than manually interpreted.

The controller predicts:

- `admit_gate`: store or ignore the incoming candidate.
- `match_scores`: update an existing slot or allocate a new one.
- `erase_gate`: invalidate or overwrite old content.
- `eviction_scores`: choose a slot when all are occupied.
- `read_scores`: retrieve relevant slots for the current output.

## Stage 4: Training curriculum

### Debugging curriculum

Use direct supervision for `STORE`, `IGNORE`, `UPDATE`, and `DELETE` to verify the architecture and dataset.

### Research curriculum

Remove memory-operation labels. Train from downstream answer loss plus explicit capacity and calibration terms.

For short episodes, keep the memory path differentiable across resets so later answer loss can train earlier writes. Compare against truncated backpropagation as episode length grows.

## Losses to evaluate

\[
\mathcal{L} =
\mathcal{L}_{answer}
+ \lambda_s\mathcal{L}_{stale}
+ \lambda_a\mathcal{L}_{abstain}
+ \lambda_b\mathcal{L}_{budget}
+ \lambda_c\mathcal{L}_{consistency}
\]

Each auxiliary term needs an ablation. If answer loss alone works, prefer the simpler objective.

## Metrics

Report at minimum:

- Overall latest-value accuracy.
- Accuracy separately for `SET`, `UPDATE`, `DELETE`, and unknown cases.
- Stale-answer rate after updates.
- Deleted-fact leakage.
- Abstention precision, recall, and F1.
- Useful-fact retention under capacity 1, 2, 4, 8, and 16.
- Accuracy versus sessions since evidence.
- Accuracy versus number of overwrites.
- Historical tokens reprocessed at inference.
- Persistent-state bytes and latency.

## Falsification conditions

The main hypothesis is weakened if:

- FIFO/LRU matches the learned controller at equal capacity.
- Gains disappear for unseen entities or longer episodes.
- The model succeeds only with direct operation labels.
- The target method reduces stale answers by abstaining excessively.
- Performance depends on accidental surface patterns in the generator.
- The model cannot demonstrate that evicted evidence affected later output.

Negative results will be recorded in `research/log.md`.
