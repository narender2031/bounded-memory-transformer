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
