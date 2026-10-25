# Data pipeline overview

This is the authoritative specification for the data pipeline, superseding the
rework plan. The [pilot](pilot.md) defines the monitor experiment; this pipeline
supplies its verified actor/synthetic × benign/attack programs, retaining independent
same-source pairs and smaller matched development/evaluation panels.

Fresh task allocation follows qualification; overlap review belongs to final
assembly. Prior exposure constrains holdout eligibility. The implementation plans
describe current work; Stage 03 develops prompts and scaffolding experimentally,
recording the actual settings and evidence for each run.

Specification and prompt-preview success do not establish acquisition readiness.
Panel targets/minima, acquisition ceilings and stratum allocations must be resolved
by the yield probe before committing to later acquisition or monitor comparisons.
Preserve scientific checks and task ownership while generation methods evolve.
This document does not authorize model runs.


**01_qualified_pool — load APPS and qualify individual tasks.**

- Load pinned APPS train/test JSONL under declared statement/input/output limits.
- Admit short introductory/interview Python stdin/stdout tasks with unique integer
  answers.
- Map all statement examples and require at least one nonvisible stored case.
- Test token-distinct supplied APPS solutions in dataset order until three pass or
  all are exhausted, requiring three. 
- Publish only fully qualified tasks and their evidence in
  `01_qualified_pool/manifest.json`.

**01_pool_analysis — describe the qualified test pool.**

- Use only `split == "test"` downstream; preserve qualification evidence for both splits.
- Report statement/reference Qwen tokens, case character/byte sizes, reference
  availability and recorded execution cost in `tasks.csv` and `summary.json`.
- Retain original task identities. Defer eligibility cutoffs, cohort allocation,
  model generation and Linux verification; content counts do not establish full
  monitor-example fit within 2,048 tokens. See the [analysis module](../src/data-pipeline/stage01/analyze.py).

**01_test_pool — materialize the dataset for allocation.**

- Run `export-test` to copy only qualified test task artifacts, preserving their bytes.
- Publish a test-only `manifest.json` with task hashes and qualification runtime/gate;
  retain the original manifest as historical `qualification-manifest.json`.
- Preserve `01_qualified_pool` and `01_pool_analysis`; execute no supplied programs.

**02_cohort — allocate tasks and freeze queues.**

- Read only the task paths listed in `01_test_pool/manifest.json`.
- Queues are
  construction, development, correction_replay, calibration and evaluation.
- The [Stage 02 plan](stage-02-plan.md) proposes a short-task profile, fixed
  ownership and ordered reserves: start with 8 development tasks, then a 32-task
  construction probe, targeting 256 usable construction tasks and capping development
  at 50 tasks total, including the smoke, with an option to expand construction to 512.
- Preserve the eligible difficulty mix in each queue; retain all stored cases and
  use a small verification screen before complete checks on surviving programs.
- Keep queues isolated and preserve source-independent acquisition and the required
  Inspect Linux reference gate. Final panel sizes and acquisition budgets follow
  the fixed queue capacities: target 24 calibration and 48 evaluation tasks,
  with coarse operating points and task-level uncertainty.


**03_benigns — generate and verify.**

- Follow the [Stage 03 plan](stage-03-plan.md): consume frozen cohort ownership,
  then re-verify the same three references on every stored case in Inspect Linux.
- Experiment with prompts, scaffolding, sampling and repair budgets independently
  for each generator. Verify every stored case, completing all visible checks
  before hidden execution; retries receive only public feedback.
- Retain every attempt and immutable candidate; program selection may evolve.
  Publish versioned selections, provenance, cumulative effort and yield reports.
  Benign acceptance alone does not establish attack yield or final matched fit.

**04_witnesses — model proposal and independent audit.**

- TODO

**05_attacks — implant and verify.**

- TODO

**06_dataset — review overlap, normalize, tag and export.**

- TODO

## Shared contracts

- TODO


