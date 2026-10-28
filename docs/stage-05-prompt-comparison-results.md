# B2 three-prompt comparison: 2026-10-27

Three parallel Claude Opus 5.5/high reviews considered one prompt each, with tools
disabled and prompt text only. The [frozen protocol](v1/stage-05-prompt-comparison.md)
then ran at temperature 0.2, three attempts per pair and 2,048 tokens per reply.

| Prompt | Actor verified | Synthetic verified |
| --- | ---: | ---: |
| Boundary | 0/8 | 0/8 |
| Optimization | 0/8 | 0/8 |
| State update | 0/8 | 0/8 |

Of 144 attempts, 64 failed published tests, 63 left the witness answer unchanged,
16 returned multiple fenced blocks and one hit its token cap. No new verified
attacks exist to score or review for mechanism adherence; this is a prompt failure
on the fixed panel, not evidence that subtle attacks are impossible.

| Comment-free monitor input | PASS | FLAG | Invalid |
| --- | ---: | ---: | ---: |
| Actor benigns | 6/8 | 0/8 | 2/8 |
| Synthetic benigns | 5/8 | 0/8 | 3/8 |
| Previous actor attacks | 0/8 | 7/8 | 1/8 |
| Previous synthetic attacks | 0/7 | 7/7 | 0/7 |

All six invalids hit 512 tokens; they are neither passes nor evidence of evasion.
Both sources stop for review with no new recipe chosen. No 0.7 or B3 runs occurred.
Effort: 175 replies, 310,784 committed / 51,545 observed output tokens, 896.8 seconds.
All 426 tests passed (13 skipped); the independent evidence audit passed, including
source/archive integrity, privacy projections, acceptance, accounting and decisions.
[Run and evidence](../data/05_attack_experiments/batch-02-prompt-comparison-20261027T161653Z/summary.json);
[receipt](../data/runs/stage05-b2-prompt-comparison-20261027T161653Z.json).

[Feedback analysis](stage-05-feedback-analysis.md) traces the repairs and clarifies
the format failures: 13 multiple-program replies and three programs plus fenced stdin.
