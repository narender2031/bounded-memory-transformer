# Project 04 — reliable reading and an independent negative control

[Detailed results](../../research/reader-followup-results-2026-09-20.md) ·
[Reader protocol](../../research/reader-followup-protocol-2026-09-20.md) ·
[Uniform protocol](../../research/uniform-replication-protocol-2026-09-20.md) ·
[Case 5B design](../../research/case-5b-emergent-utility-design.md)

The structured 7,639-parameter neural classifier plus pointer/copy reader reached
100% agreement with the exact visible reader and 100% recovery in all 41 conditions
for each of three seeds. The matched frozen v1 reader scored 92.74% on the fresh
four-slot original tasks. This is a supervised, explicitly structured reader;
free-form generation and uncertain-source trust remain unproved.

The preregistered 10,240-episode uniform replication places all six learned–baseline
contrasts inside the ±1 percentage-point equivalence margin, under both corrected
analytic and bootstrap intervals. The previous A8 advantage did not replicate.
Case 5B emergent utility is designed, not implemented or evaluated.

![Measured comparison](results/2026-09-20-v1/reader-and-uniform.png)

## Reproduce locally

Run from the repository root with its PyTorch environment. In this linked
worktree the interpreter is `../bounded-memory-transformer/.venv/bin/python`;
set `PYTHONPATH=src` so imports resolve to this checkout.

The reference actors are the original local checkpoints under
`runs/five-case-2026-09-20-v1`. Uniform config pins the retention checkpoint hashes.
The reference checkpoints are not
included in the compact result archive: reproducing this exact replication
requires those original artifacts. Regenerated actors need separately declared
hashes and constitute another replication. See Project 03 for their training
configuration; do not silently alter the frozen pins.

```sh
env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python -m pytest
../bounded-memory-transformer/.venv/bin/ruff check .

env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python \
  -m bounded_memory_transformer.memory_followup.train_reader \
  --config projects/04-memory-followup/configs/reader-candidate-4.json \
  --output runs/reader-followup-development/candidate-4

env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python \
  projects/04-memory-followup/diagnose_authority.py \
  --training runs/reader-followup-development/candidate-4 \
  --output runs/reader-followup-development/candidate-4-binary-validation.json \
  --binary --all-reader-cases

env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python \
  -m bounded_memory_transformer.memory_followup.evaluate_reader \
  --config projects/04-memory-followup/configs/reader-evaluation.json \
  --training runs/reader-followup-development/candidate-4 \
  --references runs/five-case-2026-09-20-v1 \
  --output runs/reader-followup-2026-09-20-v1

../bounded-memory-transformer/.venv/bin/python projects/04-memory-followup/audit_reader.py \
  --run runs/reader-followup-2026-09-20-v1 \
  --output runs/reader-followup-2026-09-20-v1-audit.json

env PYTHONPATH=src ../bounded-memory-transformer/.venv/bin/python \
  -m bounded_memory_transformer.memory_followup.uniform \
  --config projects/04-memory-followup/configs/uniform-replication.json \
  --output runs/uniform-replication-2026-09-20-v1 --held-out \
  --freeze-commit 569c387659b77a30799e6a94be74de1e8918639c
```

Training/evaluation refuse nonempty run directories; uniform evaluation refuses
any existing output directory. Select fresh run and audit output paths for reruns.
Training defaults to seeds 7/19/43 and MPS. CPU is available as a separately
recorded development config, not a silent change to the frozen final experiment.
All final readers must be trained before held-out evaluation. Reader source/config
and reference hashes were frozen at `1845235`; its run remains immutable.

For validation-only pipeline checks, add `--smoke` to the reader evaluation
command with a fresh output path. Uniform `--smoke` uses only train symbols and
needs no freeze argument. Tiny smoke samples can legitimately have unavailable
oracle headroom; they must not be called validated readers.

## Artifacts and limits

- [Reader summary](results/2026-09-20-v1/reader-summary.json) and
  [independent audit](results/2026-09-20-v1/reader-audit.json).
- [Uniform summary](results/2026-09-20-v1/uniform-summary.json) and
  [independent audit/program](results/2026-09-20-v1/uniform-audit.json).
- [Development history](results/2026-09-20-v1/development.json), including failed
  soft ranking and the pre-test protocol amendments; the original flawed
  generator is archived alongside it.
- Raw evidence, checkpoints and prediction rows stay under `runs/` locally.
  They include repeated controls and are not all independent observations.

The reader's final binary readout is deliberately different from its soft-score
training path and is fixed in `reader-evaluation.json`. It uses unchanged
candidate-4 weights, a zero-logit classifier threshold, deterministic conjunction,
learned chronology and exact copying. There is no inference-time exact-key repair.
See the protocol for all auxiliary supervision and architectural priors.

The next implementation milestone is the EU1 generator, serializer and independent
strong baselines on development data. Future held-out utility claims require a
separate executable freeze and the proposed distribution-shift/harm gates.
