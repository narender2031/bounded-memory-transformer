# Bounded Memory Transformer

Open research on whether a small Transformer can learn to maintain useful, current information in a tiny persistent memory without rereading its history.

## Research question

> Can a small Transformer learn what to store, retrieve, update, replace, forget, and ignore across context resets under a strict fixed memory budget?

The central challenge is not merely recall. It is the combination of:

- **Selection:** preserve information with likely future utility.
- **Supersession:** replace or invalidate facts when the world changes.
- **Interference control:** avoid corrupting unrelated memories.
- **Abstention:** do not invent an answer when the required fact was never stored.
- **Bounded operation:** keep memory and inference cost independent of total history length.

## Scope

The first system will use:

- A small decoder-only Transformer implemented from understandable components.
- Explicit context resets between sessions.
- A fixed bank of latent memory slots that alone survives each reset.
- Learned read, write, update, and eviction controls.
- Synthetic tasks with `SET`, `UPDATE`, `DELETE`, `NOISE`, and `ASK` operations.
- Evaluation on unseen entities, values, episode lengths, and update patterns.

At answer time, the model must not receive the raw historical sessions.

## Non-goals

This project is not initially attempting to:

- Build a larger context window.
- Add a vector database or conventional RAG pipeline.
- Compress only the standard Transformer KV cache.
- Store unlimited facts perfectly in fixed-size memory.
- Claim that neural or persistent memory itself is novel.

## Working hypothesis

A memory controller trained for both future utility and semantic supersession can outperform FIFO, LRU, recency, and attention-only retention policies at the same memory and compute budget.

This is a hypothesis, not a result.

## Research status

The project has completed **Phase 0: literature mapping and experimental specification** and has started **Phase 1: build and verify the Transformer baseline**.

See:

- [`research/problem-statement.md`](research/problem-statement.md) — precise problem and success criteria.
- [`research/papers.md`](research/papers.md) — curated paper map and reading order.
- [`research/experiment-plan.md`](research/experiment-plan.md) — proposed baselines, tasks, and metrics.
- [`research/decisions.md`](research/decisions.md) — durable research decisions.
- [`research/log.md`](research/log.md) — chronological research record.

## Project 01: Tiny Transformer

[`projects/01-tiny-transformer`](projects/01-tiny-transformer) contains the first runnable learning project: a decoder-only Transformer implemented without `nn.Transformer` or `nn.MultiheadAttention`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
bmt-train-tiny --steps 300
```

It includes causal multi-head attention, decoder blocks, next-token training, autoregressive generation, an example corpus, and tests for causal isolation.

## Planned experimental ladder

1. Implement and verify a tiny decoder-only Transformer.
2. Build a deterministic multi-session memory benchmark.
3. Establish no-memory and full-history bounds.
4. Implement equal-capacity FIFO, LRU, and recurrent-memory baselines.
5. Reproduce a small Recurrent Memory Transformer-style model.
6. Add learned admission and eviction.
7. Add explicit supersession and deletion handling.
8. Run ablations and evaluate on natural conversational memory tasks.

## Research principles

- Separate evidence, inference, and hypothesis.
- Compare methods at equal memory capacity and comparable compute.
- Keep raw history unavailable in the strict-memory condition.
- Report stale-answer and abstention failures, not only average accuracy.
- Record failed experiments and negative results.
- Do not make novelty claims without a fresh prior-art check.

## License

[MIT](LICENSE)
