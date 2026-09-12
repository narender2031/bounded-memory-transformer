# Project 01: Tiny Transformer from First Principles

This project builds the decoder-only Transformer that will become the control model for our bounded-memory experiments.

The implementation uses PyTorch tensors and basic neural-network layers, but intentionally avoids `nn.Transformer` and `nn.MultiheadAttention` so every important operation remains visible.

## What this project teaches

1. How characters become token IDs.
2. How token and positional embeddings are combined.
3. How linear projections produce queries, keys, and values.
4. Why attention scores are divided by \(\sqrt{d_{head}}\).
5. How a causal mask prevents access to future tokens.
6. How multiple attention heads are split and recombined.
7. How residual connections, layer normalization, and the MLP form a decoder block.
8. How next-token cross-entropy trains the complete model.
9. How autoregressive generation repeatedly samples the next token.

## Data flow and tensor shapes

For batch size \(B\), sequence length \(T\), model width \(D\), and \(H\) heads:

| Stage | Shape |
|---|---|
| Token IDs | `[B, T]` |
| Token + position embeddings | `[B, T, D]` |
| Queries, keys, values | `[B, H, T, D/H]` each |
| Attention scores | `[B, H, T, T]` |
| Recombined attention output | `[B, T, D]` |
| Vocabulary logits | `[B, T, vocab_size]` |

The causal mask makes every attention probability above the diagonal zero. Therefore, the representation at token position \(t\) can depend only on positions \(0\) through \(t\).

## Code map

```text
src/bounded_memory_transformer/tiny_transformer/
├── config.py       # validated model hyperparameters
├── tokenizer.py    # transparent character tokenizer
├── data.py         # deterministic next-token batches
└── model.py        # attention, MLP, decoder block, LM, generation

src/bounded_memory_transformer/cli/train_tiny.py
└── reproducible training and sampling command
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

The tests verify more than tensor shapes: one test changes a future token and confirms that earlier logits remain identical, directly testing causal isolation.

## Train the model

Quick smoke test:

```bash
bmt-train-tiny --steps 5 --batch-size 4 --eval-batches 2
```

Small learning run:

```bash
bmt-train-tiny --steps 300
```

You can train on another UTF-8 text file:

```bash
bmt-train-tiny --text path/to/corpus.txt --steps 1000
```

The bundled corpus is intentionally tiny. Its purpose is to validate the implementation and make loss reduction visible, not to create a useful general-purpose language model.

## Verified baseline

The initial CPU smoke run used a 27,520-parameter model for 50 steps. All 11 tests passed, and validation loss decreased from `3.5936` at step 1 to `2.7913` at step 50. This verifies the end-to-end learning path; it is not a language-quality benchmark.

## What this project does not contain yet

- Persistent memory slots.
- Context resets during training.
- Memory admission, retrieval, replacement, or eviction.
- The `SET/UPDATE/DELETE/NOISE/ASK` benchmark.
- A KV cache for faster generation.

Those are later milestones. Keeping them out of Project 01 gives us a trusted baseline before memory changes the architecture.

## Suggested learning exercise

Open `model.py`, place a breakpoint immediately after the QKV permutation, and run a batch with `B=1`, `T=5`, `D=8`, and `H=2`. Inspect:

1. The shape of each query, key, and value tensor.
2. The raw \(5 \times 5\) score matrix.
3. The upper-triangular causal mask.
4. The probabilities after softmax.
5. The two head outputs before they are joined.

This five-token trace is the best bridge between the attention equations and the code.
