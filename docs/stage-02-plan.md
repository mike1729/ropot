# Stage 02 cohort allocation plan

Allocate APPS tasks before generation. Target **256 usable construction tasks**,
cap development at **50 task IDs total**, and expand either construction phase
only when its learning curve justifies more data.

## Steps at a glance

1. Load the test pool, analysis and existing known-use metadata.
2. Apply the fast profile and freeze task owners, construction phases and reserves.
3. Publish the allocation manifest and report.
4. Run 8 development candidates, then 16 construction candidates per phase.
5. Check acquisition feasibility after 32 construction candidates per phase.
6. Train on nested accepted sets, validate the twins, then acquire repair/holdout data.

## Targets and inputs

A usable matched task has verified actor and synthetic benign/attack pairs,
passes overlap review, and fits complete monitor examples within 2,048 tokens.
Retain independent source pairs when matching fails; report their counts separately.

| Queue | Target | Baseline initial candidates, introductory/interview |
| --- | --- | ---: |
| construction | 128 usable tasks per phase; 256 total | 588: 143/445 |
| development | 50 IDs including 8 smoke; remaining 42 for training choices | 50: 12/38 |
| correction_replay | 24 usable matched tasks initially | 122: 30/92 |
| calibration | 24 usable matched tasks | 16: 4/12 |
| evaluation | 48 usable matched tasks | 32: 8/24 |

Initial candidate quotas precede the known-use override below. Activate reserves
to meet accepted-data targets; do not enlarge development after failures.

Read only `data/01_test_pool/manifest.json` and its listed task files, joined to
`data/01_pool_analysis/tasks.csv` and `summary.json`. Fully exhaust
`qualified_tasks.iter_qualified_tasks`; add hash, join and provenance checks in
the loader rather than duplicate its path/identity/count validation.

Load and hash the existing [known-use records](../../honeypot/data/manifests/exposure-history-additions-v1.json)
and [coverage metadata](../../honeypot/data/manifests/exposure-history-sweep-v1.json)
automatically. Bind task IDs using the recorded original-row hash rule. Restrict
`prompt_checker_development` and `unresolved` records to construction/development.
Record incomplete coverage as a limitation on holdout status.

The fast profile requires statements ≤700 Qwen tokens, each passing reference
≤500 tokens, each input ≤16 KiB, and total input ≤64 KiB. It contains **1,169 tasks:
285 introductory / 884 interview**. Keep every stored case; impose no case-count
cutoff. The other 279 qualified test tasks stay inactive until their profile is widened.

## Frozen allocation

Use seed `ropot-cohort-v1`. Within each difficulty, sort by SHA-256 of
`ropot-cohort-v1|test|<id>`, breaking ties by ID. Draw initial quotas in order:
evaluation, calibration, correction_replay, development, construction.

Assign remaining eligible tasks through a repeating cycle of **7 construction,
3 correction_replay, 2 calibration, 4 evaluation** slots. Reset the cursor per
difficulty, then continue it through deferred tasks. Development gets no reserves.

Before freezing, redirect known-use tasks assigned to correction/calibration/
evaluation into construction, retaining their initial/reserve/deferred block.
Do not backfill those protected quotas. Rebuild construction order by hash within
each block/difficulty, then alternate phase assignments: introductory starts in
shared pretraining; interview starts in twin continuation.

Interleave difficulties within each block proportionally, rounding prefix counts
half up. Append initial, reserve and deferred blocks in that order. All programs,
witnesses, patches and retries inherit task owner and phase. Never transfer
construction reserves into holdouts after acquisition.

| Resolved capacity | Fast profile | Including deferred |
| --- | ---: | ---: |
| construction | 758 | 885 |
| development | 50 | 50 |
| correction_replay | 187 | 240 |
| calibration | 57 | 91 |
| evaluation | 117 | 182 |

Both construction phases have 379 fast candidates. Their interview capacities are
285/286; obtaining 97 usable interview tasks per phase needs about **34% interview
yield**. Measure yield and cumulative effort by difficulty, source and experimental
configuration; Stage 03 retry strategies remain open to experimentation.

Stage 03 experiment budgets may be revised with cumulative attempt accounting.
Set later acquisition targets and ceilings in their consuming experiment. Ownership
caps all targets. If capacity is insufficient, report a shortfall or add fresh
qualified tasks with owners fixed before generation.

## Activation and acceptance

- **Development:** first 8 candidates use 2 introductory / 6 interview tasks.
  Keep the other 42 fixed across training comparisons; report actual usable counts.
- **Construction:** start with 16 candidates per phase (32 total), then 32 per phase
  (64 total, including 48 interview candidates). Use the latter for feasibility.
- **Training:** grow accepted sets through 32, 64 and 128 tasks per phase.
  At 128, require 31 introductory / 97 interview tasks and 128 verified attacks per
  source per phase, each paired with its own benign. Both twins use identical
  continuation inputs and budgets, with the pilot's different targets.
- **Expansion:** either phase may grow to 256 usable tasks (62 introductory /
  194 interview), giving 384 total if one expands or 512 if both do.
  Expanding shared pretraining requires rerunning both continuations from the new
  shared checkpoint. Expanding continuation retains the shared checkpoint.
- **Selection:** for a dataset export, take accepted tasks in cohort order after
  every predecessor in that phase/difficulty has a resolved status for that export.
  Stage 03 may revisit program choices through further experiments while preserving
  task ownership and prior evidence. Never select by completion order or monitor score.

Use the [pilot's twin checks](pilot.md#2-train-honest-and-dishonest-monitor-twins)
before repair acquisition. Freeze repair eligibility and sampling before sizing
its final dose; verified attacks are not automatically monitor mistakes.
Replay uses construction examples with unchanged training targets.

Calibration fits only the predeclared operating rule after all checkpoints freeze.
Evaluation compares the same tasks across all five checkpoints. These sizes support
coarse operating points and large effects; make no calibrated 1–5% false-positive
claim. Predeclare the primary comparison and report paired task-level uncertainty.

## Verification and feedback

Re-verify all three references on all stored cases in the actual Inspect Linux
sandbox before generation. Allocation preserves this gate.

For generated programs, rank case indices by SHA-256 of
`ropot-screen-v1|test|<id>|<index>`, with index as tie-breaker. Select the first
visible and first nonvisible case, then fill to `min(8, case_count)` from remaining
ranked cases. Fix membership before execution. Execute selected visible cases,
remaining visible cases, selected nonvisible cases, then remaining nonvisible cases,
in rank order within each group, rejecting on first failure. This completes every
visible check before hidden execution. Acceptance requires every stored case;
reuse recorded passes only for identical program bytes and execution contracts.

Only visible-case diagnostics may enter retry prompts. Nonvisible inputs, outputs,
indices, stderr and case-level results remain in verifier logs. A nonvisible failure
returns only a generic rejection. Monitors receive the problem and one program,
without source tags, patches or verifier witnesses.

Stage 03 prompts, scaffolding, extraction and retry budgets may evolve; record
settings and cumulative attempts for each experiment. Rationale generation and
final dataset selection belong to their consuming stages. Repeated variants do
not add independent tasks.

## Implementation batches

1. Reuse the qualified-task reader; extend the loader and pure allocation policy
   with the input checks, known-use override and frozen rules above.
2. Add the CLI and atomically publish `data/02_cohort/manifest.json` and
   `report.json`, refusing an existing destination. Record input hashes, metadata
   coverage, queues/phases, budgets, accepted targets, counts and shortfalls.

Batch 2 implements `python -m src.data-pipeline cohort` and a standard-library
publisher composing the existing loader and allocator. Both versioned JSON files
publish together; tasks remain referenced in the test pool. Construction activation
uses the phase queues, deferred tasks remain inactive, and acquisition retry ceilings
remain pending. The inherited Linux verification gate still applies before generation.

Publication targets `data/02_cohort_review` during review; canonical publication to
`data/02_cohort` follows code review and commit, using a committed source archive.
Record source provenance, analysis refresh and candidate checks in
`data/runs/stage02-review-cohort-v1.json`, preserving older artifacts in the archive.
From the committed archive, use the repository's absolute `.venv/bin/python` and
explicit absolute paths for all five cohort options, including both known-use files.
If refreshing analysis, use the original `01_qualified_pool` with explicit input,
output and tokenizer paths. Keep the Stage 03 plan and overview changes in a
separate documentation commit.

When pipeline source bytes are unchanged, compare both canonical JSON files with
the review candidate using `cmp`; they must be byte-identical. After both match,
archive `02_cohort_review` under `data/archive/stage02-reviewed-candidate-v1`.
Record the committed publication in the new
`data/runs/stage02-publish-reviewed-cohort-v1.json`; retain the earlier working-tree
publication record unchanged.

The real review candidate has an initial calibration shortfall of one interview
task, with 57 fast-profile calibration candidates overall. Known-use handling
redirects 8 fast-profile tasks (11 including deferred); all 32 known-use tasks
(25 fast-profile) belong to construction. Phase smoke prefixes are 4 introductory /
12 interview for the first 16 and 8/24 for the first 32; development smoke is 2/6.

Keep each batch at or below 500 additions plus deletions and stop for review, as
required by [repository instructions](AGENTS.md). Acquisition, execution and
training remain in their consuming stages.
