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

The reader follow-up now passes both 95% gates in all 41 conditions for each of
three seeds. A 7,639-parameter neural classifier with explicit composition and
exact copying matches the visible-evidence oracle at 100%; the frozen previous
reader reaches 92.74% on the same fresh four-slot original tasks. This establishes
symbolic reading competence with strong architectural priors. It does not solve
uncertain-source trust or recover facts the writer discarded.

An independent 10,240-episode uniform-query replication puts all six learned–baseline
contrasts inside the preregistered ±1 percentage-point equivalence margin. The
earlier eight-slot advantage did not replicate. Historical predictable-utility
gains still match a planted cue heuristic; emergent utility remains unproved.

![Audited reader and uniform-control results](projects/04-memory-followup/results/2026-09-20-v1/reader-and-uniform.png)

Start with [Project 04 reproduction](projects/04-memory-followup/README.md), the
[detailed follow-up report](research/reader-followup-results-2026-09-20.md), and
the [Case 5B emergent-utility design](research/case-5b-emergent-utility-design.md).
The runs and independent audits completed locally on an M2 Pro MacBook, 16 GiB.
The [original five-case results](research/five-case-results-2026-09-20.md) and
historical **Experiment 01 — When Memory Hurts** remain preserved, including
failed gates. Case 5B is designed but has not been implemented or evaluated.

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
