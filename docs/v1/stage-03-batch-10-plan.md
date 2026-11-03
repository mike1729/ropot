# Batch 10: Stage 03 closeout

Close Stage 03 after the additional generation and repair experiments.
The [experiment summary](stage-03-results.md) records their outcomes, costs,
limitations and the Stage 04 handoff.

## Minimal retained workflow

- `study_tasks`: read the frozen cohort and task bindings.
- `local_generation`, `benign_loop` and `repair_feedback`: generate and repair.
- `verification` and `inspect_runner`: check programs against published cases.
- `experiment_summary`: summarize saved attempts and effort.
- `benign_selection`: publish the first verified program per task and source.

Keep existing APIs and scientific evidence; add no more experiment arms.
One-off callers, their tests and the detailed experiment plans are retained in
[the Git archive](archive.md#stage-03-experiment-callers).

## Completion and handoff

Published `data/03_benigns/v1/`: 173 programs, covering 101 tasks, including 72
with both sources. Stage 04 reads its `manifest.json`. Every saved pass was
validated against pinned task identities, candidate/verdict digests and program
bytes; no generation or execution was repeated. Bulk acquisition and further
scaffold changes wait for Stage 04–05 yields.

The retained suite ran 284 tests successfully, with 11 optional skips. Archive
snapshots retain exact bytes; source experiment records remain unchanged.
The user approved one wrap-up change beyond the usual 400-line review limit.
Implementation and publication are complete.
Review size: 594 non-test content lines (including 273 unchanged, previously
untracked caller lines), plus 214 test lines; byte-preserving moves are separate.
