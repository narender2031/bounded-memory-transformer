# Pull Request 1: Tiny Transformer Baseline

Last updated: 2026-09-12

GitHub pull request: [#1 — Add Project 01: tiny Transformer from first principles](https://github.com/narender2031/bounded-memory-transformer/pull/1)

Branch: `feat/tiny-transformer`

Base branch: `main`

Status at the time of this snapshot: open, mergeable, and passing GitHub Actions.

This file keeps the complete PR #1 handoff inside the branch so a clone does not depend on the GitHub conversation to understand what was built or why.

## Purpose

Establish a trusted, inspectable no-persistent-memory control model before modifying the architecture for bounded memory.

The implementation intentionally avoids `nn.Transformer` and `nn.MultiheadAttention`. Each important Transformer operation remains visible and testable.

## Code delivered

```text
src/bounded_memory_transformer/
├── __init__.py
├── cli/
│   ├── __init__.py
│   └── train_tiny.py
└── tiny_transformer/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── model.py
    └── tokenizer.py
```

Implemented components:

- validated model configuration;
- character-level tokenizer;
- deterministic contiguous next-token batches;
- token and learned positional embeddings;
- manual query, key, and value projections;
- scaled dot-product causal self-attention;
- multiple attention heads;
- pre-normalised residual decoder blocks;
- feed-forward network;
- tied language-model output projection;
- next-token cross-entropy loss;
- autoregressive generation;
- reproducible command-line training.

## Learning project

`projects/01-tiny-transformer/` contains:

- a component-by-component explanation;
- tensor-shape reference;
- setup, test, and training commands;
- a five-token attention inspection exercise;
- an original small corpus for smoke testing;
- an explicit list of memory features intentionally excluded from Project 01.

## Tests delivered

The branch includes tests for:

- invalid configurations;
- tokenizer round trips and invalid token IDs;
- deterministic next-token target shifting;
- logits, loss, and attention tensor shapes;
- zero attention probability above the causal diagonal;
- causal isolation when a future token changes;
- optimizer updates;
- generation prefix preservation.

Verification result: all 11 tests passed.

## Training evidence

The initial CPU smoke test used:

- 27,520 parameters;
- 50 optimization steps;
- context length 32;
- model width 32;
- 4 attention heads;
- 2 decoder layers;
- batch size 4.

Observed validation loss:

| Step | Validation loss |
|---:|---:|
| 1 | 3.5936 |
| 10 | 3.1846 |
| 20 | 3.0130 |
| 30 | 2.9421 |
| 40 | 2.8738 |
| 50 | 2.7913 |

The generated sample remained mostly incoherent, as expected from 50 steps on the intentionally tiny corpus. The purpose was to verify the complete learning path, not language quality.

## Reproduce locally

```bash
git clone https://github.com/narender2031/bounded-memory-transformer.git
cd bounded-memory-transformer
git switch feat/tiny-transformer

python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

ruff check .
pytest -q

bmt-train-tiny \
  --steps 50 \
  --batch-size 4 \
  --eval-batches 3 \
  --log-every 10 \
  --context-length 32 \
  --d-model 32 \
  --heads 4 \
  --layers 2 \
  --generate-tokens 40 \
  --device cpu
```

## CI

`.github/workflows/tests.yml` runs lint and tests on pull requests and pushes to `main`. It installs CPU-only PyTorch and cancels superseded runs on the same ref.

The latest hosted run associated with the handoff changes passed.

## Research records included

- [CODEX.md](CODEX.md)
- [EXPLORE.md](EXPLORE.md)
- [AGENTS.md](AGENTS.md)
- [research/problem-statement.md](research/problem-statement.md)
- [research/papers.md](research/papers.md)
- [research/experiment-plan.md](research/experiment-plan.md)
- [research/decisions.md](research/decisions.md)
- [research/log.md](research/log.md)

Together these files preserve the original conversation’s research question, prior-art findings, non-negotiable constraints, decisions, experiment design, verification evidence, and immediate next step.

## Explicit non-goals of PR #1

PR #1 does not implement:

- persistent memory slots;
- hard-reset episodes during training;
- memory admission, retrieval, replacement, or eviction;
- the SET/UPDATE/DELETE/NOISE/ASK benchmark;
- learned semantic supersession;
- a generation KV cache.

These exclusions are deliberate. Adding memory before verifying the base Transformer would make failures harder to localise.

## Next action after merge

Create `feat/synthetic-memory-benchmark` from the merged `main` branch and implement:

1. typed SET, UPDATE, DELETE, NOISE, and ASK operations;
2. a reference state machine;
3. targeted semantic tests;
4. a deterministic episode generator;
5. equal-capacity FIFO, LRU, recency, similarity, and oracle policies;
6. memory-aware metrics.

Do not add the learned memory controller until the benchmark and simple baselines are independently trusted.
