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
- A fixed symbolic bank that alone survives each reset; latent representations are deferred.
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

The five-case study is implemented and has completed a three-seed local run.
Selection/copying improves reader accuracy to 92.83%, but every seed still fails
the full reader gates. Controlled updates/deletions score 100%. On predictable
future utility, learned retention reaches 27.91% versus FIFO's 15.41% with four
slots, matching the cue-priority heuristic. The combined system remains unvalidated.

Start with [Project 03 and reproduction commands](projects/03-memory-reliability/README.md)
and the [detailed five-case report](research/five-case-results-2026-09-20.md).
The run took 12.28 minutes on an M2 Pro MacBook with 16 GiB RAM. All 95 tests pass;
independent auditing reconciles the saved predictions, metrics, and hashes.
Historical **Experiment 01 — When Memory Hurts** remains a weak-reader pilot.

See:

- [Detailed results summary](research/results-summary-2026-09-20.md) — measured results, reader versus storage failures, and remaining hypotheses.
- [Six-paper review](research/reading-notes/2026-09-20-six-paper-review.md) — verified findings and bounded-memory adaptations from the latest supplied papers.
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

The [end-to-end fundamentals guide](research/transformer-fundamentals.md) teaches
the exact implementation with worked attention arithmetic, tensor shapes,
training and generation, parameter counts, memory boundaries, and runnable
exercises. The [memory improvement note](research/reading-notes/2026-09-17-memory-improvements.md)
explains the reader results, new frozen-model diagnostics, and researched next
experiments.

## Planned experimental ladder

See [`projects/02-memory-benchmark`](projects/02-memory-benchmark/README.md) for
the measured three-seed results, limitations, figure, and reproduction commands.

```bash
python -m bounded_memory_transformer.memory_benchmark.evaluate \
  --config projects/02-memory-benchmark/configs/reader-extended.json
```

The run uses the existing 237,792-parameter Transformer, 30 candidate operations,
eight sessions, and four slots, with MPS or CPU. Checkpoints and full predictions
are saved locally under `runs/`.

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
