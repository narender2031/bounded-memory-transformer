# Paper Map

Last literature check: 2026-08-27

This is a curated map for our specific question, not a general long-context bibliography. Papers are grouped by the component they help us understand or reproduce.

## Read first

These papers most directly constrain our research design.

| Priority | Paper | What it contributes | Gap relative to our target |
|---|---|---|---|
| A1 | [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881) (Bulatov et al., 2022) | Passes learned memory tokens between sequence segments and trains the Transformer to control their contents. | Learned compression across segments, but no explicit semantic supersession or utility-constrained slot lifecycle. |
| A2 | [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624) (Wang et al., 2024) | Adds a fixed-size latent memory pool that updates from new text while the base model remains static. | Uses a very large memory pool and primarily gradual age-based replacement, not a tiny utility-managed budget. |
| A3 | [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (Behrouz et al., 2025) | Treats attention as short-term memory and adds an online neural long-term memory updated at inference time. | Focuses on long-sequence modelling and surprise-driven memorisation rather than explicit current-fact semantics. |
| A4 | [It's All Connected: MIRAS](https://arxiv.org/abs/2504.13173) (Behrouz et al., 2025) | Unifies memory architecture, learning rule, retention gate, and memory objective as a design space. | A framework for designing sequence models, not a completed solution to bounded multi-session updates. |
| A5 | [Self-Pruned Key-Value Attention](https://arxiv.org/abs/2605.14037) (Szilvasy et al., 2026) | Meta FAIR directly trains a predictor to decide which KVs have future utility and should enter persistent cache. | The closest overlap with our selection idea, but it prunes token KVs inside a sequence rather than maintaining editable semantic facts across sessions. |
| A6 | [GradMem](https://arxiv.org/abs/2603.13875) (2026) | Writes a context into eight memory vectors using a few test-time gradient steps; evaluates after removing the original context. | Strong match for hard reset and compact latent state, but evaluations are mainly controlled retrieval with short horizons. |
| A7 | [Supersede](https://arxiv.org/abs/2606.27472) (Patel, 2026) | Isolates the bounded-memory failure caused by facts that are later updated and supplies a trainable RL environment. | Agent-level textual memory rather than a small latent-memory architecture; reported trained accuracy remains low. |
| A8 | [LiveMem](https://arxiv.org/abs/2608.02515) (2026) | Defines a “living” recurrent state that updates online, survives context turnover, and changes later behaviour after evidence leaves context. | Very recent; gains from persistent state are often modest and semantic versioning is not isolated. |
| A9 | [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., 2024) | Evaluates extraction, cross-session reasoning, temporal reasoning, knowledge updates, and abstention. | Primarily evaluates assistant memory systems; we must adapt it carefully for a tiny architecture. |

## Learned selection, retention, and forgetting

| Paper | Relevance |
|---|---|
| [Not All Memories Are Created Equal: Learning to Forget by Expiring](https://arxiv.org/abs/2105.06548) (Sukhbaatar et al., 2021) | EXPIRE-SPAN learns a lifespan for memories and shows that selective expiration can preserve sparse useful evidence amid long distractors. It is an important learned-forgetting baseline. |
| [H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048) (Zhang et al., 2023) | A strong hand-designed KV-retention baseline based on accumulated attention and recency. Useful as a policy baseline, not semantic memory. |
| [Learning to Remember: End-to-End Training of Memory Agents](https://arxiv.org/abs/2602.18493) (2026) | Assigns memory actions a future-utility signal derived from later question-answer rewards. Closely overlaps our training objective, but manages external CRUD-style memory. |
| [Task-Focused Memorization](https://arxiv.org/abs/2605.31075) (2026) | Trains a policy to generate memories valuable for future tasks from streaming multimodal inputs. Supports the future-utility framing at the agent level. |
| [Learning to Remember, Learn, and Forget in Attention-Based Models](https://arxiv.org/abs/2602.09075) (Palimpsa, 2026) | Frames fixed-capacity associative memory as a stability-plasticity problem and introduces importance-aware metaplasticity. Relevant to interference and selective forgetting. |

## Compact and recurrent architectural memory

| Paper | Relevance |
|---|---|
| [Transformer-XL](https://arxiv.org/abs/1901.02860) (Dai et al., 2019) | Foundational segment-level recurrence using cached hidden states. It establishes the baseline distinction between replayed activations and a compact learned state. |
| [Compressive Transformers](https://arxiv.org/abs/1911.05507) (Rae et al., 2019/2020) | Adds a compressed stream for old activations. Important for lossy compression objectives and equal-compute baselines. |
| [Memorizing Transformers](https://arxiv.org/abs/2203.08913) (Wu et al., 2022) | Retrieves past internal key-value representations through approximate kNN at inference time. A strong external activation-memory reference. |
| [Associative Recurrent Memory Transformer](https://arxiv.org/abs/2407.04841) (Rodkin et al., 2024) | Combines local attention, segment recurrence, and associative matrices with constant per-segment memory. Relevant to key-value replacement and capacity. |
| [Infini-attention](https://arxiv.org/abs/2404.07143) (Munkhdalai et al., 2024) | Combines local masked attention with bounded compressive linear memory in a Transformer block. Useful architecture baseline. |
| [Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620) (Sun et al., 2024) | Makes the recurrent hidden state itself a model updated by a self-supervised learning step. Foundational for fast-weight/test-time-memory approaches. |

## Dynamic factual and episodic memory

| Paper | Relevance |
|---|---|
| [Larimar](https://arxiv.org/abs/2403.11901) (Das et al., 2024) | Supports one-shot knowledge updates through distributed episodic memory. Useful for separating knowledge editing from streaming memory management. |
| [M+: Extending MemoryLLM](https://arxiv.org/abs/2502.00592) (Wang et al., 2025) | Adds longer-term storage and a co-trained retriever to MemoryLLM after its fixed latent pool showed distant-retention limits. |
| [Metis: Memory Foundation Model](https://arxiv.org/abs/2607.26760) (2026) | Articulates the broader target: native state whose lifecycle learns what to remember, update, consolidate, and forget based on expected future utility. Treat as a vision/taxonomy paper, not evidence that the problem is solved. |

## Evaluation and datasets

| Paper | Use in this project |
|---|---|
| [LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., 2024) | Natural multi-session conversational memory with long-distance and temporal questions. Candidate late-stage evaluation. |
| [LongMemEval](https://arxiv.org/abs/2410.10813) (Wu et al., 2024) | Primary natural benchmark because it explicitly includes knowledge updates and abstention. |
| [LongMemEval-V2](https://arxiv.org/abs/2605.12493) (2026) | Extends memory evaluation toward agent experience, dynamic state, workflows, and premise awareness. Candidate after the basic model is stable. |
| [LoCoMo-Plus](https://arxiv.org/abs/2602.10715) (2026) | Tests implicit constraints and cue-trigger disconnect beyond factual recall. Out of scope initially, valuable for later generalisation. |

## Recommended reading order

### Pass 1: Establish the architecture lineage

1. Transformer-XL
2. Compressive Transformer
3. Recurrent Memory Transformer
4. Associative Recurrent Memory Transformer
5. MemoryLLM

### Pass 2: Understand learned memory policies

1. EXPIRE-SPAN
2. Titans
3. MIRAS
4. SP-KV
5. Palimpsa

### Pass 3: Focus on our exact gap

1. GradMem
2. Learning to Remember
3. Supersede
4. LiveMem
5. LongMemEval

## Paper-note protocol

For every A-priority paper, record:

- Problem formulation.
- Persistent state representation and size.
- Read, write, update, and forgetting mechanisms.
- Training signal and whether it reaches earlier writes.
- What survives a context reset.
- Whether the model can replace a semantic fact.
- Baselines and capacity fairness.
- Datasets, metrics, and generalisation split.
- Code and reproducibility status.
- Result we should reproduce.
- Limitation that creates space for our experiment.

Use `research/reading-notes/template.md` for individual notes.
