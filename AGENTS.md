# Research Continuity Instructions

These instructions apply to the entire repository.

## Source of truth

- `research/problem-statement.md` defines the current research question and evaluation contract.
- `research/papers.md` is the literature map.
- `research/decisions.md` contains durable decisions.
- `research/log.md` is the chronological memory of research activity.
- `research/experiment-plan.md` defines the current experimental ladder.

## After every research session

1. Append dated evidence, inferences, failures, and next actions to `research/log.md`.
2. Update `research/papers.md` when a relevant paper is found or its interpretation changes.
3. Add or revise a decision in `research/decisions.md` when a choice affects future work.
4. Keep evidence, inference, and hypothesis explicitly separated.

## Literature standards

- Prefer primary papers, official proceedings, author repositories, and official research-lab pages.
- Record the date of every fresh literature search.
- Do not claim novelty from absence in a small search.
- Compare new ideas against the closest mechanism, not only against famous older work.

## Experimental standards

- Preserve the hard-reset and fixed-capacity contract for main results.
- Compare at equal memory capacity and report compute differences.
- Use deterministic generators and record seeds/configuration.
- Test unseen entities, values, longer episodes, updates, deletions, distractors, and abstention.
- Report negative results and failed hypotheses.
- Do not silently change metrics or dataset generation after seeing test results.

## Code standards

- Prefer small, inspectable PyTorch components over high-level Transformer wrappers during the learning phase.
- Add shape assertions and unit tests for attention, masking, memory reset, and memory persistence.
- Keep baseline policies independent of the learned controller.
- Make every reported experiment reproducible from a checked-in configuration.
