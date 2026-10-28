# Cohort inference sizing — 2026-10-25

Prioritize repeated development monitor evaluations and small generation queues.
The [fixture findings](inference-benchmark-results.md) supply measured starting
settings; these cohort estimates do not establish a production optimum or yield.
The [Stage 03 plan](stage-03-plan.md) retains activation and implementation scope.

## Frozen population and lengths

The analysis joins `data/02_cohort/manifest.json` to the existing
`data/01_pool_analysis/tasks.csv`. Manifest, CSV and analysis-summary hashes were
checked against the pinned Stage 03 cohort and recorded provenance; original-task
hashes and difficulties agree. Only metadata was read, without opening protected
task payloads or running inference. The manifest SHA-256 is
`63f1df94a21fff53eee8e7665bff8b1bee1aeb91bbefdd83d504eb32845e1492`.

| Group | Initial | Reserve | Eligible total | Deferred | Planned matched target |
| --- | ---: | ---: | ---: | ---: | --- |
| Development | 50 | 0 | 50 | 0 | At least 20 for training gate; 50-ID cap |
| Construction | 589 | 169 | 758 | 127 | 256, with 128 per phase |
| Correction/replay | 122 | 65 | 187 | 53 | 24 initially |
| Calibration | 15 | 42 | 57 | 34 | 24 |
| Evaluation | 32 | 85 | 117 | 65 | 48 |

Each construction phase has 379 eligible candidates: 295 + 84 for shared
pretraining, 294 + 85 for continuation. Capacities include ordered reserves;
they are not simultaneously activated queues. Deferred tasks remain unavailable.
Development starts with eight generation-smoke IDs; preserve the other 42 for
training choices, without enlarging the group after acquisition failures.

Lengths use the existing Qwen content tokenizer and nearest-rank percentiles.
The combined proxy adds statement tokens to each task's median supplied-reference
length. It excludes instructions/templates and is not a generated-program length
or proof of complete monitor-example fit within 2,048 tokens.

| Eligible group | Statement p50 / p95 | Statement + reference proxy p50 / p95 |
| --- | ---: | ---: |
| Development | 390 / 682 | 465 / 847 |
| Construction | 395 / 658 | 527 / 870 |
| Correction/replay | 406 / 652 | 527 / 857 |
| Calibration | 454 / 631 | 594 / 905 |
| Evaluation | 376 / 626 | 505 / 818 |

Initial and monitor prompts should usually be shorter than the long repair
fixtures, but measure actual rendered prompts and generated code. History growth
and lengthy witnesses can move later requests into a different context bin.

## Work queues and starting settings

Use MLX Q8 with one generating model resident at a time. Settings below are
decode target / prefill batch / prefill chunk, bounded by ready independent work.
The production adapter is currently serial; these estimates assume adoption and
qualification of the archived native batch path. Keep request-owned seeds and
record sampling settings; do not change sampling to fill a batch.

| Workload | Qwen | Ministral |
| --- | --- | --- |
| Eight-task generation smoke | 8 / 4 / 256 | 8 / 2 / 2,048 |
| Ordinary generation, ~2k prompts, larger queue | 32 / 16 / 256 | 32 / 8 / 2,048 |
| Short witness proposals, larger queue | 32 / 16 / 256 | 32 / 8 / 2,048 |
| Long repair, ~7k prompts, larger queue | 32 / 16 / 256 | 8 / 2 / 2,048, provisional |

Do not delay available work merely to fill a batch. Repair waits for its own
verification; witness-first implantation waits for witness approval. One bounded
`verify_many` evaluation uses up to four sandboxes, with serial cases per program.
Overlap only independent work. Ministral 7k/batch 32 already OOMed; larger-capacity
experiments need allocation-aware memory bounds rather than between-call checks.

For monitor evaluation, one checkpoint scores actor/synthetic benign/attack cells:
20 fully matched tasks give 80 requests, 42 give 168, and 50 give 200. All four
cells use the same monitor; generator source does not split the inference queue.
Compare Qwen decode 32 versus 64 on actual 80/168/200-request worklists, initially
holding prefill 16/chunk 256 and the frozen rationale/verdict budget fixed. Trained
checkpoint performance is unmeasured; do not adopt batch 64 from fixture capacity
alone. Fifty full-panel development evaluations mean 10,000 requests, versus 960
for five frozen checkpoints on the 48-task evaluation target. Calibration's target
panel supplies 96 requests per checkpoint. Keep locked groups out of tuning.

## Development generation time scenarios

Assume one benign, one witness and one attack attempt for each task from each
source, all branches reaching each stage. The two generators run sequentially.
Use measured 2k repair-program throughput as the closest available proxy for both
full-program stages; homogeneous benign/attack throughput has not been measured.
Tiny witness timings come from the 40-request short-proposal fixtures, whose
selected configurations averaged about 18 Qwen and 29 Ministral tokens per reply.
The 128-token cap is not an observed completion length.

| Attempted tasks | Decode target | Benign inference | Witness inference | Attack inference | Total inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 smoke | 8 | ~1.0 min | ~8 s | ~1.0 min | ~2.2 min |
| 42 training-choice IDs, when acquired | 32 | ~3.9 min | ~26 s | ~3.9 min | ~8.3 min |
| Full 50-ID group, when acquired | 32 | ~4.7 min | ~31 s | ~4.7 min | ~9.9 min |

These are linear throughput scenarios, not measured finite-panel times, upper
bounds or an ETA for accepted data. The 33-program comparisons at batch 32 took
93.454 s for Qwen and 92.337 s for Ministral; at batch 8, 132.176 and 125.975 s.
Forty short witness proposals took 11.984 + 12.679 s at selected batch-32 settings,
or 19.965 + 18.859 s at batch 8. Scaling down to eight requests and changing tails
adds uncertainty. Longer code, larger witnesses and history growth increase cost.

For 50 tasks, a useful warm-inference calculation in minutes is
`4.69 * a_benign + 0.514 * a_witness + 4.69 * a_attack`, where each `a` is draws
per original task/source, including repairs and failures and zero for blocked
branches. Equal effort of three draws per stage gives about 30 GPU minutes;
five gives about 50, provided lengths and batching remain comparable. A witness
first-pass estimate is especially optimistic for nontrivial inputs/answers.

Published-suite verification adds roughly 3 CPU minutes for 200 benign/attack
programs on 50 tasks, scaling the measured eight-program/7.094-second fixture.
It can partly overlap inference; do not add it mechanically or apply the earlier
14.3% overlap gain universally. Model loads/switches, witness oracle execution,
constraint/novelty review, review waits and final-fit checks are excluded and
remain unmeasured. At one source-specific witness per task, the full group has
100 witness reviews: even an illustrative 30 seconds each would add 50 human
minutes. Fifty attempted IDs do not guarantee 50 usable matched tasks.

## Extractor and prefix decisions

Freeze extraction before acquisition: accept plain Python or one unambiguous,
complete Python fence with optional surrounding prose, only from naturally
completed replies; require syntax and the unchanged execution checks. Secondary
fixture replay recovered 158 passing draws from 165 Ministral format failures;
seven still failed execution. Original strict scores remain historical evidence.

Keep response caching and prefix-state reuse off. Exact-prefix cloning gave
roughly 3–7x speedups on tiny JSON-answer fixtures with all 56 shared-state checks
unchanged. Baseline errors prevented the all-correct semantic gate; paired samples
showed no observed correctness degradation, but neither full-program quality nor
monitor quality is qualified. Revisit prefix reuse with paired verified-artifact
yield after the pipeline works, and test trained-monitor outputs separately.
The paired Q8/BF16 actual-yield check also waits for the full witness/backdoor
pipeline, verification and independent audit, as the overview requires.

Detailed computed percentiles, timing inputs and scenarios are saved in
[the metadata analysis](../../data/03_throughput/cohort-sizing-20261025.json), ignored
by Git alongside raw benchmark evidence. This document preserves the useful
analysis in tracked form without adding another benchmark runner.
