# Batch 9 development smoke — 2026-10-26

Both experiments completed on the same eight frozen development tasks with the
[planned recipe](stage-03-batch-09-plan.md). The user authorized working-tree
execution without a commit. Source snapshots and hashes are retained in the
[execution record](../data/03_generation_checks/batch-09-execution-20261026/execution.json),
which confirms sources stayed unchanged during the run.

| Source | Initial passes | After one repair | Replies | Observed output tokens | Committed output tokens | Elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor/Qwen | 6/8 | 7/8 | 10 | 4,612 | 10,240 | 78.68 |
| Synthetic/Ministral | 3/8 | 4/8 | 13 | 6,253 | 13,312 | 95.95 |

Four of eight tasks have passing benigns from both sources. Study effort was
23 replies, 10,865 observed output tokens and 23,552 committed tokens, within the
32-reply/32,768-token ceiling. Wall time was 174.65 seconds. Usage is complete;
neither experiment had a technical error or automatic retry beyond its one repair.
The [saved Batch 8 summary](../data/03_benign_batch09_summary.json) retains all
attempts, verdicts and costs. Repeated draws are not independent tasks.

| Source | Loading seconds | Preparation seconds | Generation seconds | Verification seconds |
| --- | ---: | ---: | ---: | ---: |
| Actor/Qwen | 2.59 | 0.03 | 64.05 | 9.83 |
| Synthetic/Ministral | 2.23 | 0.02 | 83.44 | 8.63 |

Separate [repair qualification](../data/03_generation_checks/batch-07b-fkvbcijh/qualification.json)
passed for both models: seven replies, 652 observed tokens and 1,792 committed
tokens under its eight-reply/2,048-token ceiling, in 33.58 seconds. Combined gate
and study effort was 30 replies, 11,517 observed tokens and 25,344 committed tokens.

Review fixes made after execution were checked without redrawing. The strengthened
[association audit](../data/03_generation_checks/batch-09-review-20261026/association-audit.json)
binds each extracted program to its own reply and source for the gate and all
16 verified study programs. Original qualification and execution records remain
unchanged; the edited gate test has not received another live qualification.

Retained generated evidence, excluded from source-change counts:

- Gate: 59 files, 172,577 bytes, including four native logs totaling 38,424 bytes.
- Study experiments: 91 files, 1,091,203 bytes, including four native logs totaling
  101,612 bytes; these include exact replies, programs, tasks and verification.
- Execution record, console and two source snapshots: four files, 20,222 bytes.
- Original and compact sizing evidence and superseded paths are documented in the
  [plan](stage-03-batch-09-plan.md#execution-and-review-fixes--2026-10-26).

Stop before another recipe or construction activation. The separate failure
inspection will inform one bounded follow-up; benign passes alone do not establish
attack yield or final matched-panel feasibility.
