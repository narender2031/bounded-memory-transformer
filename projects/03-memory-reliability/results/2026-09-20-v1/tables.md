# Five-case measured tables

Generated from the saved run, without model inference. Percentages use the
denominators in the machine-readable summary. Capacity rows average the same
episodes across model seeds; this does not triple the independent sample size.

## Reader arms: held-out visible accuracy

| Seed | character_full | selected_character | selected_copy | oracle_copy | Copy recovery |
| --- | --- | --- | --- | --- | --- |
| 7 | 39.77% | 84.61% | 92.19% | 100.00% | 60.94% |
| 19 | 39.84% | 83.87% | 91.02% | 100.00% | 55.08% |
| 43 | 37.77% | 86.60% | 95.27% | 100.00% | 76.37% |

## Reader category means

| Category | character_full | selected_character | selected_copy | oracle_copy |
| --- | --- | --- | --- | --- |
| select | 0.00% | 75.65% | 90.30% | 100.00% |
| unsupported | 97.33% | 93.75% | 93.75% | 100.00% |
| contradicted | 0.00% | 75.20% | 90.56% | 100.00% |
| irrelevant | 0.07% | 84.77% | 93.75% | 100.00% |
| deleted | 98.24% | 95.77% | 95.77% | 100.00% |
| known | 0.02% | 78.54% | 91.54% | 100.00% |
| unknown | 97.79% | 94.76% | 94.76% | 100.00% |
| full_occupancy | 39.13% | 85.03% | 92.83% | 100.00% |

## Reader paired harm and benefit

| Seed | Arm | Harm / all queries | Benefit / all queries | Δ accuracy (pp) |
| --- | --- | --- | --- | --- |
| 7 | character_full | 0.27% | 0.04% | -0.23 |
| 7 | selected_character | 0.47% | 13.71% | 13.24 |
| 7 | selected_copy | 0.51% | 17.73% | 17.23 |
| 7 | oracle_copy | 0.00% | 20.00% | 20.00 |
| 19 | character_full | 0.16% | 0.00% | -0.16 |
| 19 | selected_character | 0.08% | 13.79% | 13.71 |
| 19 | selected_copy | 0.08% | 17.30% | 17.23 |
| 19 | oracle_copy | 0.00% | 20.00% | 20.00 |
| 43 | character_full | 2.23% | 0.00% | -2.23 |
| 43 | selected_character | 0.35% | 14.34% | 13.98 |
| 43 | selected_copy | 0.35% | 18.40% | 18.05 |
| 43 | oracle_copy | 0.00% | 20.00% | 20.00 |

## Lifecycle through the exact reader

| Metric | 7 | 19 | 43 |
| --- | --- | --- | --- |
| update_accuracy | 100.00% | 100.00% | 100.00% |
| delete_accuracy | 100.00% | 100.00% | 100.00% |
| control_preservation | 100.00% | 100.00% | 100.00% |
| action_macro_f1 | 100.00% | 100.00% | 100.00% |
| target_accuracy | 100.00% | 100.00% | 100.00% |
| transition_accuracy | 100.00% | 100.00% | 100.00% |
| stale_answer_rate | 0.00% | 0.00% | 0.00% |
| deletion_nonabstention_rate | 0.00% | 0.00% | 0.00% |
| deleted_value_repetition_rate | 0.00% | 0.00% | 0.00% |

## Capacity: exact-reader useful recall

| Policy | A, K=4 | A, K=8 | B, K=4 | B, K=8 |
| --- | --- | --- | --- | --- |
| no_memory | 0.00% | 0.00% | 0.00% | 0.00% |
| fifo | 15.01% | 31.01% | 15.41% | 31.98% |
| recency | 14.84% | 31.35% | 15.23% | 31.96% |
| random | 16.48% | 32.58% | 16.76% | 33.31% |
| similarity | 15.01% | 31.01% | 15.41% | 31.98% |
| cue_priority | 16.50% | 34.08% | 27.91% | 57.18% |
| learned | 16.50% | 34.08% | 27.91% | 57.18% |
| clairvoyant oracle | 39.06% | 64.48% | 47.71% | 75.32% |

## Learned capacity versus FIFO: paired episode bootstrap

| Seed | Workload | Slots | Gain (percentage points) | 95% paired CI (pp) |
| --- | --- | --- | --- | --- |
| 7 | A | 4 | 1.49 | [-0.37, 3.39] |
| 7 | A | 8 | 3.08 | [0.76, 5.64] |
| 7 | B | 4 | 12.50 | [10.40, 14.65] |
| 7 | B | 8 | 25.20 | [22.44, 27.88] |
| 19 | A | 4 | 1.49 | [-0.37, 3.39] |
| 19 | A | 8 | 3.08 | [0.76, 5.64] |
| 19 | B | 4 | 12.50 | [10.40, 14.65] |
| 19 | B | 8 | 25.20 | [22.44, 27.88] |
| 43 | A | 4 | 1.49 | [-0.37, 3.39] |
| 43 | A | 8 | 3.08 | [0.76, 5.64] |
| 43 | B | 4 | 12.50 | [10.40, 14.65] |
| 43 | B | 8 | 25.20 | [22.44, 27.88] |

## Retention regret (oracle minus policy utility, percentage points)

| Policy | A, K=4 | A, K=8 | B, K=4 | B, K=8 |
| --- | --- | --- | --- | --- |
| no_memory | 39.06 | 64.48 | 47.71 | 75.32 |
| fifo | 24.05 | 33.47 | 32.30 | 43.33 |
| recency | 24.22 | 33.13 | 32.47 | 43.36 |
| random | 22.58 | 31.90 | 30.94 | 42.01 |
| similarity | 24.05 | 33.47 | 32.30 | 43.33 |
| cue_priority | 22.56 | 30.40 | 19.80 | 18.14 |
| learned | 22.56 | 30.40 | 19.80 | 18.14 |

## Combined learned writer, retention, and selector/copy

| Workload | Slots | Exact reading | Learned reading | Recovery |
| --- | --- | --- | --- | --- |
| A | 4 | 16.50% | 15.16% | 91.86% |
| A | 8 | 34.11% | 31.03% | 90.95% |
| B | 4 | 27.91% | 25.63% | 91.83% |
| B | 8 | 57.18% | 51.35% | 89.81% |

## Predeclared gates

| Seed | lifecycle | reader_absolute | reader_episode_relative | reader_relative |
| --- | --- | --- | --- | --- |
| 7 | True | False | False | False |
| 19 | True | False | False | False |
| 43 | True | False | False | False |

## Training cost (seconds; reader times include validation)

| Seed | Selector | Character | Lifecycle | Capacity | Total |
| --- | --- | --- | --- | --- | --- |
| 19 | 79.73 | 84.66 | 0.31 | 3.25 | 171.49 |
| 43 | 89.35 | 75.80 | 0.23 | 3.09 | 171.98 |
| 7 | 81.97 | 72.12 | 0.25 | 3.14 | 161.27 |

## Four-slot B timing by policy

Joint reader time includes all four arms and their paired no-memory controls;
it is not a per-arm latency estimate. Exact-policy times exclude neural reading.

| Policy | Write ms/episode | Exact read µs/query | Joint readers s |
| --- | --- | --- | --- |
| no_memory | 0.009 | 0.719 | 1.308 |
| fifo | 0.260 | 5.462 | 2.087 |
| recency | 0.263 | 5.317 | 2.002 |
| random | 0.250 | 5.288 | 2.103 |
| similarity | 0.254 | 8.396 | 1.756 |
| cue_priority | 0.150 | 5.319 | 2.018 |
| learned | 0.721 | 5.419 | 1.946 |
