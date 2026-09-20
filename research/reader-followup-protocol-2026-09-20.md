# Reader follow-up v1: protocol

Status: final candidate declared during train/validation development; no follow-up
held-out reader data opened when this protocol was frozen.
The user authorized this follow-up. Project 03 source and results stay frozen.

## Diagnosis and candidate

The frozen selector's fresh 2,500-task validation diagnosis at four slots scores
92.72%, 91.44%, 94.52% for seeds 7/19/43. Wrong-entity/right-attribute selections
account for 506 of 533 errors. The other errors are 26 false abstentions and one
wrong-attribute selection. Matching NOISE and precedence errors were zero in
this diagnostic. These are validation observations, not new held-out results.
See `projects/04-memory-followup/diagnose_reader.py` for the reproducible command.

Candidate 1 is a **structured neural pointer reader**, with a shared one-block
manual Transformer comparing each of two entity digits and one attribute symbol.
Auxiliary training labels supervise coordinate equality and operation authority.
Their minimum supplies an explicit conjunction bias; a bounded learned signed
chronology term ranks relevant records. A fixed zero UNKNOWN score competes with
record scores. The selected value is copied without a key check or correction.
Values cannot affect selection. No equality booleans or oracle indices enter
inference. This is substantially more structured than the original character
reader; do not claim it repairs free-form generation or learns source trust.

Development retained two failures. Candidate 1's stress generator accidentally
correlated occupancy with question type; its perfect seed-7 validation score is
not accepted as coverage evidence. Before any test use, the generator was fixed
to cross these dimensions and a regression test was added. Candidate 2 retrained
the same architecture on the corrected data. Seeds 19/43 reached 100%, but seed 7
scored 98.5% on original validation tasks: all 15 errors per 1,000 tasks were
unsupported queries misread through one attribute mismatch. It fails the full
gate and is retained, not hidden by seed averaging.

**Final candidate 3:** keep the architecture and corrected generator. First train
the shared comparator for 2,000 steps on all 116 valid digit/digit or
attribute/attribute pairs from the TRAIN alphabet, balancing equality and
inequality losses. Freeze its weights, then train authority and chronology for
3,000 steps on the same record tasks. The final checkpoints of seeds 7/19/43
are the only new models used on held-out episodes. Configuration:
[reader-candidate-3.json](../projects/04-memory-followup/configs/reader-candidate-3.json).
This exhaustive supervised primitive is a strong engineering aid. Generalization
means new combinations of familiar characters and evidence structures; it does
not mean new character symbols, unsupervised key binding, calibrated trust, or
an end-to-end language-model solution. No further candidate is planned in v1.

Training uses only train symbols, mixed 4/8-slot occupancy, current counts
0/1/3/6/12, and paired empty-memory views. Validation uses the existing validation
symbol partition, current counts 0/1/6/32, and original controls. At most three
candidate configurations may be developed; retain every outcome. Final models
use a fixed step count and seeds 7/19/43, never a test-selected checkpoint.
The disjoint test symbols are inherited from the established benchmark: fresh
episodes do not constitute a newly untouched research-level symbol partition.

## Held-out evaluation fixed before opening

Use [configuration](../projects/04-memory-followup/configs/reader-evaluation.json).
All model seeds finish training before any test generator is called. Record code,
config, checkpoints and reference-actor SHA-256 hashes before evaluation; refuse
nonempty output paths. Freeze the selected training config in a commit beforehand.

For each seed evaluate:

1. Original five reader strata at 4/8 slots, 5,000 fresh tasks each. Also evaluate
   the frozen v1 selector on these identical examples (no retraining).
2. Variable occupancy 0..K and current counts 0/1/6/32 at 4/8 slots, 2,000 fresh
   tasks per condition. Repeated authoritative updates, matching NOISE, partial
   key distractors, deletions, absence and current overrides are included. Thirty-
   two current records exceed training length. Record every feasible stratum,
   known/unknown, occupancy and memory/current/no-evidence source slice.
3. Frozen learned lifecycle actors on 800 fresh update/delete episodes each.
4. Six frozen capacity policies on 128 fresh episodes per A/B and 4/8 slots;
   also frozen learned lifecycle + learned retention actors on the same episodes.
   This is reader validation on retained evidence, not a new capacity claim.

Exact visible answers, exact current-only answers, the neural reader with memory,
and the same neural reader without memory are recorded together. Oracle copy
must agree with the independent visible state machine. No future state reaches
the reader. Full memory bytes are charged even when similarity projects top-1.

## Gates and interpretation

Keep ≥95% visible accuracy **per model seed, per condition, on every nonempty
required slice**. Fixed categories: each feasible stratum; known/unknown; each
occupied count; and each feasible evidence source. Absent categories are explicitly
reported as absent, never called passed. These feasibility exceptions are fixed
here, not chosen from observed performance. For episode banks, known/unknown are
the descriptive/required visible slices; retain lifecycle update/delete/control
slices as well.

Keep raw `Recovery=(A_neural-A_exact_current_only)/(A_exact_visible-A_exact_current_only)`
≥95% for **every condition**, including each of the 24 policy/budget/workload
controls, lifecycle, and four combined controls. Nonpositive headroom is unavailable
and prevents a full pass. Never clamp the ratio or count unavailable as passing.
All three accuracies use the same world truth on episode controls; their visible
oracle accuracy can be below 100% because necessary information was discarded.
On reader microtasks the reference is the visible answer. Absolute gates always
compare neural answers to the visible oracle, independently of world accuracy.
For the original microtasks, 80% current-only accuracy means 95% recovery requires
99% overall accuracy. Do not penalize the reader for facts absent from its bank.

Report paired harmful answers (`no-memory correct, memory wrong`) and benefits,
abstention precision/recall, stale/deleted rejection slices, and whole-episode
bootstrap intervals (2,000 draws). These gates are engineering point-estimate
thresholds, not a proof of a ≥95% population lower confidence bound. Zero observed
harm cannot establish universal safety; all evidence is symbolic and authoritative.
Capacity-only harmful rate is uninformative when the no-memory control scores zero.

Any held-out failure remains a failure of this version. Further tuning requires
a new declared experiment, with this entire run preserved. A reader pass would
validate instrumentation on these conditions; Case 5B utility learning and
robustness to uncertain real-world sources would remain unproved.
