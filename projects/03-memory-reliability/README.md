# Five cases: reliable reading, lifecycle, and bounded retention

This experiment implements the five cases approved on 2026-09-20. It separates
reading from writing before composing a system. Historical Project 02 pilot code
and results are unchanged. The [locked protocol](../../docs/superpowers/specs/2026-09-20-five-case-memory.md)
defines the hypotheses, gates, controls, and frozen budgets.

| Case | Question | Control |
|---|---|---|
| 1 — select | Can the reader find and reproduce the right value? | Four reader arms; visible oracle must be exact. |
| 2 — reject | Can it handle unsupported, stale, irrelevant, and deleted evidence? | Separate strata; matched empty-memory queries. |
| 3 — update | Does the writer update the right key? | Exact reading and unrelated-fact preservation. |
| 4 — delete | Does it remove one fact without clearing others? | Leakage, control preservation, operation/target/transition metrics. |
| 5 — capacity | Does learning retain information with observable future value? | Uniform A versus cue-predictable B; heuristic and clairvoyant controls. |

## Recorded result

The [2026-09-20 report](../../research/five-case-results-2026-09-20.md) covers the
three-seed MacBook run (12.28 minutes). Mean reader accuracy was 39.13% with full
character generation, 85.03% after learned selection, 92.83% with selection/copy,
and 100% with oracle/copy. Every seed failed the full reader gates. Controlled
four-slot updates/deletions and unrelated-fact preservation reached 100%.

In predictable workload B, learned retention scored 27.91% versus FIFO's 15.41%
at four slots and 57.18% versus 31.98% at eight, exactly matching the cue-priority
heuristic. The combined neural system remains unvalidated. The uniform A8 control
also has a positive observed learned/FIFO contrast; see the report before making
claims about the uniform workload.

![Capacity study](results/2026-09-20-v1/capacity-utility.png)

[All measured tables](results/2026-09-20-v1/tables.md) ·
[Machine-readable summary](results/2026-09-20-v1/summary.json) ·
[Independent audit](results/2026-09-20-v1/audit.json) ·
[Retention regret figure](results/2026-09-20-v1/retention-regret.png)

## What is a reader?

The writer decides what survives a session. The reader receives the surviving
records, any current evidence, and the question, then produces the answer. The
**exact reader** is a small rule: match both key coordinates, apply visible
updates/deletions in order, and copy the latest value or return `??`. It measures
what the available evidence permits. It cannot recover evicted information.

The four arms are the character Transformer on all evidence, learned selection
followed by character generation, the same selection followed by exact copying,
and visible-oracle selection followed by copying. The two character arms share
weights; the two learned-selection arms share decisions. Direct selector labels
and oracle-projected character training examples are disclosed supervision.

Reader validation checks whether it can interpret evidence that is present.
World-truth accuracy also depends on what the writer retained. The experiment
reports both, plus normalized recovery of the exact reader's gain over reading
only current evidence. No positive oracle gain means recovery is unavailable.

## Run on a MacBook

From the repository root, with the existing environment or a new virtualenv:

```bash
python -m pip install -e ".[dev,plots]"
pytest
ruff check .

# Fast CPU pipeline check; intentionally too little training for competence.
python -m bounded_memory_transformer.memory_experiments.run \
  --config projects/03-memory-reliability/configs/smoke.json \
  --output runs/five-case-smoke

# Frozen three-seed experiment. Uses MPS when available, otherwise CPU.
python -m bounded_memory_transformer.memory_experiments.run \
  --config projects/03-memory-reliability/configs/five-case-small.json \
  --output runs/five-case-main

python projects/03-memory-reliability/audit_results.py \
  --run runs/five-case-main --output runs/five-case-main-audit.json
python projects/03-memory-reliability/summarize_results.py \
  --run runs/five-case-main --output runs/five-case-main-tables
python projects/03-memory-reliability/plot_results.py \
  --summary runs/five-case-main/summary.json --output runs/five-case-main-figures
```

Choose an empty output directory: the runner refuses to overwrite any existing
artifacts. It trains every seed's components before generating the held-out
episodes. Saved artifacts include checkpoints, per-query answers, writer actions,
bank snapshots, dataset/config/source hashes, validation curves, timing, gates,
and all failed diagnostics. The smoke configuration uses validation symbols.

Training costs differ between methods and are recorded. Character reading uses
MPS when available; the much smaller lifecycle and utility models train on CPU.
Four- and eight-slot evaluations reuse exactly the same trained weights. The
runner records hardware/software identity; exact floating-point replication
across devices is not guaranteed.

## Capacity contract and interpretation

Each persistent slot is four int32 fields: entity, attribute, value, observed
utility cue. Four slots cost **64 payload bytes**, eight cost 128; there is no
hidden episodic archive, external per-key counter, cached token history, or KV
cache. Immutable shared model weights and transient computation are separate
from the episodic state budget. Current-session tokens are allowed. The primary
memory is symbolic and inspectable; learned latent compression is deferred.

Workloads A/B share 24 keys and 30 writes across eight sessions. In A, each key
has equal probability of appearing in the 32 future probes. In B, cue-1 keys
have five times the probability of cue-0 keys. The cue is sampled and observed
before any query suffix. Its field counts toward every policy's budget.

The learned 25-parameter retention scorer sees only that cue. It learns from
delayed answer rewards; it receives no future eviction or frequency labels.
The cue-priority heuristic has the same information and is a necessary strong
baseline. Matching it can demonstrate acquisition of a planted signal, but
cannot establish a general memory-management advantage. Random is a stateless
bottom-K hash policy. Similarity performs top-1 lexical retrieval from bounded
FIFO facts; it is not embedding retrieval. LRU awaits an interleaved-access task.

The clairvoyant evaluator counts future probes and retains the K most-requested
keys with their latest values. This is a feasible upper bound for this specific
write-then-query workload, not an online competitor. Regret is its utility minus
the policy's utility. Deletions are isolated in Case 4; Case 5 currently has no
deletion or uncertain-source events. `ABSTAIN` belongs to reading, `DEFER` is
deferred, and dependency repair is outside these five cases.

The lifecycle model is also deliberately small and uses explicit key-equality
and occupancy features. Its predicted actions and target slots drive the actual
executor; mistakes are not repaired with gold routing. Passing it demonstrates
feature-assisted lifecycle control, not learned semantic representations.

The causal expected-utility reference is also simple: in A, any complete K-key
bank has expected utility K/24. In B, a bank of cue-1 keys has expected utility
5K/72: 27.78% at four slots and 55.56% at eight. The future-aware oracle can exceed
these because it sees which requests actually occur. Its regret gap is therefore
not all learnable from the available past observations.
