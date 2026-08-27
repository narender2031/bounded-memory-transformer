# Research Decisions

This file records decisions that future sessions should preserve unless new evidence justifies a change.

## 2026-08-27 — D001: Project name

**Decision:** Use `bounded-memory-transformer`.

**Reason:** The name states the defining experimental constraint instead of implying a novel general-purpose memory model.

## 2026-08-27 — D002: Narrow research target

**Decision:** Focus on learned memory selection plus semantic supersession under a strict fixed capacity.

**Reason:** Persistent memory, recurrent memory tokens, test-time neural memory, and future-utility prediction are established areas. The narrower combination remains experimentally unresolved.

## 2026-08-27 — D003: Hard reset contract

**Decision:** In the strict-memory condition, delete token context and the normal KV cache between sessions. Preserve only the bounded memory state.

**Reason:** This prevents long context or hidden replay from being mistaken for persistent memory.

## 2026-08-27 — D004: Synthetic benchmark first

**Decision:** Begin with symbolic `SET`, `UPDATE`, `DELETE`, `NOISE`, and `ASK` episodes before natural language.

**Reason:** Controlled data isolates memory operations, supports unlimited examples, and enables disjoint entity/value splits.

## 2026-08-27 — D005: No broad novelty claim

**Decision:** Do not claim that learned memory, bounded memory, future-utility prediction, or supersession training is individually novel.

**Reason:** RMT, MemoryLLM, Titans/MIRAS, SP-KV, GradMem, Learning to Remember, Supersede, and LiveMem cover substantial portions of this space.

## 2026-08-27 — D006: Memory-aware reporting

**Decision:** Always report stale-answer rate, deletion leakage, and abstention metrics alongside overall accuracy.

**Reason:** Average accuracy can hide the exact failure this project aims to fix.
