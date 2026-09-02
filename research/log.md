# Research Log

## 2026-08-27 — Project initialization and prior-art check

### Question

Can a small Transformer learn what to store, retrieve, update, and forget across context resets without rereading its history?

### Findings

1. The broad problem is an active research area spanning recurrent sequence models, test-time learning, KV-cache selection, latent memory, and agent memory.
2. RMT already shows that learned memory tokens can carry information between sequence segments.
3. MemoryLLM already supports a fixed latent memory pool that self-updates from new text.
4. Titans and MIRAS treat memory as an online learning system with explicit retention and forgetting design choices.
5. Meta FAIR's SP-KV directly studies learning when to write by predicting future KV utility. Future-utility selection alone is therefore not a novel claim.
6. GradMem tests compact learned memory after the original context is removed, making it a close architectural baseline.
7. Supersede shows that maintaining the current value of changing facts remains difficult under bounded self-managed memory.
8. LiveMem explicitly requires memory state to survive context turnover and affect behaviour after evidence leaves the context, but its measured state contribution is often modest.
9. LongMemEval supplies the most directly relevant natural evaluation categories: updates and abstention in addition to recall and temporal reasoning.

### Inference

The most defensible initial gap is the joint problem of future-utility selection and reliable semantic supersession in a tiny latent state under hard resets and without raw-history fallback.

This remains a provisional inference, not a verified novelty claim.

### Next research actions

1. Read and annotate RMT.
2. Read MemoryLLM's exact update and forgetting mechanism.
3. Compare Titans/MIRAS memory objectives with SP-KV utility gates.
4. Reproduce the Supersede task structure in a small symbolic generator.
5. Define state-matched RMT, FIFO, LRU, and oracle baselines.

## 2026-09-02 — Project 01 implementation

### Work completed

1. Added a reusable character-level tokenizer and deterministic next-token batcher.
2. Implemented causal multi-head self-attention directly from query, key, and value projections.
3. Implemented pre-normalized decoder blocks with residual connections and feed-forward networks.
4. Implemented a small decoder-only language model, next-token loss, and autoregressive generation.
5. Added a reproducible CLI and a tiny original corpus for smoke testing.
6. Added tests for configuration, tokenization, shifted targets, tensor shapes, causal masking, causal isolation, parameter updates, and generation.
7. Added continuous integration for linting and tests.

### Verification evidence

1. `ruff check .` completed with no findings.
2. `pytest` passed all 11 tests in 3.19 seconds on CPU.
3. A 27,520-parameter model completed a 50-step CPU smoke run without numerical failures.
4. Validation loss decreased from 3.5936 at step 1 to 2.7913 at step 50.
5. Autoregressive generation completed from the trained checkpoint. The sample remained mostly incoherent, which is expected from 50 steps on the intentionally tiny corpus.

This satisfies the Project 01 acceptance condition: the implementation is tested and demonstrates finite, decreasing loss end to end.

### Research impact

The resulting model becomes the no-persistent-memory baseline. Persistent state, context-reset episodes, and learned memory controllers remain intentionally excluded until the baseline is trusted.
