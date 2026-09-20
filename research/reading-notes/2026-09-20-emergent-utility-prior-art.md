# Primary-source check: emergent utility and bounded caching

Search/read date: **2026-09-20**. Scope: ARC, TinyLFU, LRB, and LeCaR mechanisms;
targeted SP-KV and Supersede abstract rechecks. This is a bounded design search,
not an exhaustive novelty review, a systematic review, or a local reproduction.
All links below are primary papers, official proceedings, or the authors' code.

The resulting proposal is
[Case 5B emergent utility v1](../case-5b-emergent-utility-design.md). No new model
has been implemented or evaluated for it.

## Sources and verified mechanisms

### ARC — Megiddo and Modha, FAST 2003

Primary [proceedings page](https://www.usenix.org/conference/fast-03/presentation/arc-self-tuning-low-overhead-replacement-cache)
and [paper](https://www.usenix.org/events/fast03/tech/full_papers/megiddo/megiddo.pdf).
Read the introduction, demand-paging assumptions, and recency/frequency/ghost
directory discussion in paper sections I–II. No implementation was reproduced.

**Evidence:** ARC adapts the balance between recently seen and repeatedly seen
pages as request patterns change. Its directory can remember twice as many page
identities as fit in the value cache. The paper's demand-paging model brings a
requested missing page from auxiliary memory.

**Inference for EU1:** adaptive recency/frequency is established prior art.
Counting K resident values while ignoring ghost identities would violate our
byte contract. EU1 does not restore a missing value on ASK, so an ARC comparison
requires explicit changed admission semantics and a charged directory. A
truncated-directory variant is an adaptation, not a faithful reproduction.

### TinyLFU — Einziger, Friedman, and Manes, 2015 preprint v2

Primary [versioned abstract](https://arxiv.org/abs/1512.00727v2) and
[paper](https://arxiv.org/pdf/1512.00727).
Read section 3: admission comparison, approximate counting, halving/reset,
counter size, and the Bloom-filter doorkeeper.

**Evidence:** TinyLFU compares recent estimated demand for a candidate and victim.
It can record accesses to items outside the resident cache through a compact
frequency structure. Aging halves counters after a sample threshold; its
doorkeeper uses additional bits and is cleared on reset. W-TinyLFU combines
admission with a recency window.

**Inference for EU1:** resident-only LFU is insufficient as the only frequency
baseline. Include a small sketch-based admission competitor with all counters,
sampling state, and any doorkeeper charged. The proposed no-doorkeeper, 14/30-byte
sketch at four/eight slots is a deliberately constrained TinyLFU-style variant;
it is not the original configuration or a demonstrated optimal use of those
bytes. Hash collisions and adaptation lag must be tested.

### LRB — Song, Berger, Li, and Lloyd, NSDI 2020

Primary [proceedings page](https://www.usenix.org/conference/nsdi20/presentation/song)
and [paper](https://www.usenix.org/system/files/nsdi20-paper-song.pdf), *Learning
Relaxed Belady for Content Distribution Network Caching*. The venue is **NSDI
2020, not OSDI 2020**. Read sections 4.1–4.3 and 5 (memory window, labels,
features, prediction target, and overhead).

**Evidence:** LRB predicts log time-to-next-request with gradient-boosted trees,
using inter-request gaps, exponentially decayed counts, and static object
features. A sliding memory window retains histories beyond cached objects.
Labels become available after a later request, or after a boundary-based timeout;
the system retains training examples and trains online. Window selection uses a
trace validation prefix.

**Inference for EU1:** learning utility from past-access features is established.
LRB's histories, pending examples, and training state cannot be hidden inside a
64-byte claim. EU1 instead freezes inference weights and trains from realized
answer rewards; it neither reproduces LRB nor adopts its next-request labels.
Success on production traces does not establish our held-out mechanism transfer
or explicit UPDATE/DELETE semantics.

### LeCaR — Vietri et al., HotStorage 2018

Primary [proceedings page](https://www.usenix.org/conference/hotstorage18/presentation/vietri),
[paper](https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf),
and [author repository](https://github.com/sylab/LeCaR).
Read section 3 and Algorithms 1–2. The repository identifies its LRU, LFU, ARC,
and LeCaR implementations and a synthetic workload with phase changes; code was
located but not executed or audited in this session.

**Evidence:** LeCaR learns a mixture of LRU and LFU eviction through regret-style
weight updates. An eviction history stores identities and the responsible
policy; a later miss matching that history supplies delayed regret, weighted by
time since eviction. History entries, policy weights, and age information are
state beyond cached values.

**Inference for EU1:** merely switching between recency and frequency is not a
novel mechanism. A learned scorer must beat a strong tuned blend as well as the
individual policies. A faithful LeCaR baseline needs charged eviction feedback
state; omitting that history removes its defining feedback. Workload adaptation
in this paper does not imply transfer to our unseen workload generators.

### SP-KV and Supersede — relation to the existing map

Rechecked the primary v1 abstracts of
[Self-Pruned Key-Value Attention](https://arxiv.org/abs/2605.14037v1)
(Szilvasy et al., 2026) and
[Supersede](https://arxiv.org/abs/2606.27472v1) (Patel, 2026).
This session did not reread their full texts.

**Evidence:** SP-KV trains future-utility prediction through next-token loss,
keeps recent KVs in a local window, and dynamically sparsifies older KVs rather
than enforcing EU1's exact slot budget. Supersede targets maintenance of current
facts in bounded agent memory and supplies rewards for current answers versus
stale ones.

**Inference:** compare utility selection with SP-KV and factual currency with
Supersede. EU1's hard resets, charged symbolic state, no-refill access process,
and new family split specify a different evaluation contract; their conjunction
is not itself a novelty claim. Exact shared semantic guards also make EU1 a
narrower test than learning fact maintenance.

## Design consequences and remaining limits

1. **The close baseline family is caching.** FIFO alone cannot establish useful
   inference. Compare LRU, resident LFU, decayed LFU, a tuned recency/frequency
   hybrid, and charged nonresident-frequency admission on identical observations.
2. **State fairness includes feedback.** Ghost keys, sketches, timestamps,
   adaptive weights, and pending examples are information-bearing memory.
   Distinguish value slots, total persistent bytes, and temporary compute.
3. **A cache miss is a contract choice.** Conventional demand paging can fetch a
   requested value. Strict EU1 receives only the ASK key, so a hit-rate improvement
   in an ordinary cache cannot be copied into the semantic-memory claim.
4. **Distribution shift needs a declared test.** Report static, abrupt, recurrent,
   and scan-like demand separately, with mechanisms held out from both gradient
   training and model-selection families. None of this source check demonstrates
   generalization for our proposed policy.
5. **Oracle bounds depend on allowed transitions.** EU1 must enumerate feasible
   writes/evictions and forced updates/deletes. An offline top-K future-count list
   need not be reachable. Even a valid clairvoyant gap contains unpredictable
   future randomness and is not all learnable opportunity.

**Evidence:** the reviewed literature already contains access-based utility
inference, online workload adaptation, and explicit metadata tradeoffs.
**Inference:** EU1 should be an honest, small controlled comparison before any
larger memory architecture. **Hypothesis:** a learned rule over a tiny causal
summary can beat these strong comparators on the declared unseen mechanisms.
There are currently no EU1 results supporting or falsifying that hypothesis.
