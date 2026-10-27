# Batch 12 revised recipes: unresolved touched-task rerun

Rerun revised recipes only on previously used tasks that the respective source
has never passed. Qwen gets 2,048 tokens per reply; Ministral stays at temperature
0.2 throughout. Each runs three cycles of a new draw plus up to three repairs.
These results are acquisition after tuning, not clean recipe-yield estimates.
Change no pipeline code. Batch 11b publishes afterward so `v1` includes new passes.

## Motivation and review boundary

[Batch 11a](stage-03-batch-11-plan.md) recorded 29/91 Qwen replies hitting the
1,024-token limit; 11 tasks failed exclusively that way. Ministral's later
0.7-temperature cycles left 43% of replies unusable. These results motivated the
revised recipes. The user chose to rerun only unresolved touched tasks; the
previously proposed fresh check on candidates 33–48 per phase is removed.

Follow [the repository rules](AGENTS.md) in one review batch on `main`, within
400 additions plus deletions excluding tests, including documentation. Use
`archive/stage03/experiments/batch12_rerun.py`, its tests and tokenizer sizing. Stop for review
before dispatch. Generation, feedback and verification sources must still match
`d6c311e`; no new live gate is needed.

## Recipes

Both sources use temperature 0.2, top-p 0.95, 2,048 tokens per reply and three
cycles. Each cycle is a new experiment on that source's remaining failures, in
saved order: a new initial reply followed by up to three full-history repairs.
Reset history between cycles and stop a task at its first pass. The ceiling is
12 new attempts per task, excluding all prior draws. Base seeds are 1,000,017,
1,100,017 and 1,200,017, separate from earlier experiments.

- **Qwen:** keep the Batch 9 development system, initial/repair instructions and
  feedback caps; change both output caps to 2,048 and `max_repairs` to 3. Decode 8,
  prefill 4, chunk 256.
- **Ministral:** keep the strict-code instructions, neutral system and feedback
  caps; use `max_repairs=3`. Decode 8, prefill 2, chunk 2,048.

## Worklists and budgets

Consider only the 50 previously used development tasks and construction
candidates 1–32 of each phase (64 tasks). For each source, exclude any task it
passed in any numbered study experiment. Freeze the remaining bindings in sizing.
Do not use diagnostic snippets as official passes. Retain all earlier evidence.

| Source | Development | Construction | Tasks | Maximum new replies | Committed-token ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 14 | 19 | 33 | 396 | 811,008 |
| Ministral | 19 | 30 | 49 | 588 | 1,204,224 |
| Total | 33 | 49 | 82 source/task pairs | 984 | 2,015,232 |

These lists cover 56 distinct tasks; 26 currently lack both sources. Each cycle's
allowance covers four replies and 8,192 tokens per remaining task. Run Qwen, then
Ministral, one model at a time. A technical error stops all later work without
retry. Candidates 33–48 per phase remain outside this run. Already passing tasks
receive no new draws, and the first-pass publication rule preserves their selection.

## Sizing and measures

Render every initial prompt with the actual runtime tokenizer. Also construct a
stress history per source on its longest prompt, using the coordinator's helpers:
three 2,048-token replies and three 4,096-byte rendered feedback blocks. Stop for
review if initial prompts exceed 745 Qwen/755 Ministral tokens or stress exceeds
context. Context fit is not a memory bound: these stress histories exceed the
longest real histories run so far. Check recorded peaks after each round; one
interruption stops generation gracefully and retains evidence.

Report, against the full frozen source worklists:

- cumulative newly passed tasks per source after new attempts 1–12, with cycle
  boundaries after 4 and 8; final acquisition counts separately by task group;
- development and construction coverage before/after the rerun, per source and
  jointly, using all earlier official passes;
- distinct extracted-program digests per task, both overall and among cycle
  initial replies; these exact-byte counts do not establish algorithmic diversity;
- unusable replies by attempt, submissions, committed/observed/prompt tokens,
  generation/verification timings and peak memory.

All comparisons use already touched, failure-selected tasks. Do not label the
attempt-two count as an unbiased improvement over Batch 11a. Benign acquisition
alone sets no attack yield or acquisition scale.

## Implementation review — 2026-10-26

The caller provides `measure()` in the pinned MLX Python and `run()` in repository
Python. It refuses changed source/recipe/model hashes, seeds or frozen worklists,
an existing Batch 12 directory, or unfinished prior experiments. It saves the
Batch 8 summary and partial results after a stop. Unreadable native round metrics
are marked unavailable, preserving saved commitments and the original stop reason.

The [current sizing](../data/03_generation_checks/batch12-rerun/sizing.json) freezes
only the 33 Qwen and 49 Ministral tasks. Maximum initial prompts are 730 and 738
tokens; depth-three stress input plus output reaches 20,673 and 20,789 tokens.
The old sizing that included fresh tasks is retained separately as superseded.
Current sizing is one JSON file of 93,136 bytes; the superseded record is
118,960 bytes. Both retain their own source hashes and worklist provenance.

All eight focused tests and 310 repository tests pass, with eleven optional skips.
The caller has 199 lines and tests 169 lines. Tests cover the reduced schedule,
seeds/budgets, exclusion of untouched blocks, acquisition coverage/diversity,
technical stops and damaged native JSON. No generation or verification ran during
this revision. Stop here for review; no commit or dispatch has occurred.
