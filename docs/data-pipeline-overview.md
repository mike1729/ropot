# Data pipeline overview

This is the authoritative specification for the data pipeline, superseding the
rework plan. The [pilot](pilot.md) defines the monitor experiment; this pipeline
supplies its verified actor/synthetic × benign/attack programs, retaining independent
same-source pairs and smaller matched development/evaluation panels.

Fresh task allocation follows qualification; overlap review belongs to final
assembly. Prior exposure constrains holdout eligibility. The implementation plans
describe current work; Stage 03 develops prompts and scaffolding experimentally,
recording the actual settings and evidence for each run.

Start inference benchmarks and pipeline development with Q8. Once the full witness
and backdoor generation pipeline, including verification and independent audit,
works end to end, run at least one bounded, paired Q8-versus-BF16 comparison of
actual verified yield and elapsed cost under matched tasks, prompts and budgets.
The [consolidated Q8 fixture findings](inference-benchmark-results.md) record
provisional runtime/scheduling settings, context-length limits and extraction
results, with pointers to the complete benchmark archive. Scientific
witness/backdoor yield remains unmeasured; the
[remaining plan](inference-benchmark-plan.md) defers those experiments to their
consuming stage contracts.

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
- Publish a test-only `manifest.json` with task hashes and qualification provenance;
  retain the original manifest as historical `qualification-manifest.json`.
- Preserve `01_qualified_pool` and `01_pool_analysis`; execute no supplied programs.

**02_cohort — allocate tasks and freeze groups.**

- Read only the task paths listed in `01_test_pool/manifest.json`.
- Groups are
  construction, development, correction_replay, calibration and evaluation.
- The [Stage 02 plan](stage-02-plan.md) proposes a short-task profile, fixed
  ownership and ordered reserves: start with 8 development tasks, then a 32-task
  construction probe, targeting 256 usable construction tasks and capping development
  at 50 tasks total, including the smoke, with an option to expand construction to 512.
- Preserve the eligible difficulty mix in each group and all original pool cases.
  Publish independent task copies with all visible cases followed by deterministic
  hidden cases, capped at 20 total; downstream verification uses every published case.
- Keep groups isolated and preserve source-independent acquisition. Existing
  reference qualification suffices. Final panel sizes and acquisition budgets follow
  the fixed group capacities: target 24 calibration and 48 evaluation tasks,
  with coarse operating points and task-level uncertainty.
- Generate and save every group's ordered APPS ID lists once in `groups` and
  `construction_phases`; downstream stages consume the saved assignment.


**03_benigns — generate and verify.**

- Complete: Stage 04 consumes `data/03_benigns/v1/manifest.json`; see the
  [experiment results](stage-03-results.md) for coverage and limitations.
- Follow the [Stage 03 plan](stage-03-plan.md): consume frozen cohort ownership
  and trust existing reference qualification without another Docker gate.
- Load only saved construction/development IDs from initial/reserve blocks, keeping
  protected and deferred tasks unavailable. Check task-file digests when reading.
- Experiment with prompts, scaffolding, sampling and repair budgets independently
  for each generator. Cap execution at 20 cases per program: all visible cases,
  then a deterministic hidden subset published in reduced Stage 02 task files. Use the same
  selection for every program on a task; acceptance covers the selected suite
  without a later full-suite pass. Complete visible checks before hidden execution;
  retries receive only public feedback. Stage 03 reads and executes every case in
  the reduced file without selection; the original pool retains all stored cases.
- Retain every attempt and immutable candidate; program selection may evolve.
  Publish versioned selections, provenance, cumulative effort and yield reports.
  Benign acceptance alone does not establish attack yield or final matched fit.

**04_witnesses — model proposal and independent audit.**

- Follow the [Stage 04 plan](stage-04-plan.md): consume the published Stage 03
  selection of 173 programs across 101 tasks, targeting one approved witness per
  program with bounded attempts. Complete Stage 04 before Stage 05.
- Each source proposes witnesses for its own program using the statement, that
  program and safe rejection reasons. Trust existing reference qualification;
  require all three references and the benign to agree on proposed witness answers.
- Reject whitespace-token duplicates against all stored inputs. Require isolated
  Codex and Claude approvals of validity and semantic novelty against published
  inputs, using pinned auditor versions. Humans may veto; use a small random human
  check to assess audit quality.
- Keep verifier data and audit replies private from proposers. Retain every
  attempt, its costs and benign disagreements for later dataset review.
- Validate on fixtures and development tasks, then process the remaining pool and
  publish a versioned witness selection with coverage, unresolved cases and effort.

**05_attacks — implant and verify.**

- Follow the [Stage 05 plan](stage-05-plan.md): consume Stage 04's approved
  program–witness pairs, targeting one verified attack per pair with bounded
  attempts. Each source attacks its own benign; direct witness-matching guards
  are allowed.
- Require every published case to pass and the approved witness to produce a
  normal exit with one well-formed integer whose value differs from the approved
  answer. Trust Stage 04's approval and keep the benign and witness fixed.
- Keep reference answers, hidden-case diagnostics and audit replies private from
  generators. Retain all attempts, programs, evidence and costs.
- Before settling the recipe, check the frozen base monitor on development pairs
  to assess guarded-attack detectability. Monitor scores never determine
  individual attack acceptance.
- Process the remaining approved pairs and publish a versioned attack selection
  with coverage, unresolved cases and effort for Stage 06.

**06_dataset — review overlap, normalize, tag and export.**

- Follow the [Stage 06 plan](stage-06-plan.md): assemble verified benign/attack
  pairs from Stages 03–05. The first publication covers development and
  construction; repeat assembly as other groups acquire programs. Rationales and
  training targets remain with the training stage.
- Strip comments uniformly across sources and labels, removing comment-only lines
  and trailing whitespace without changing strings. Preserve original verified
  bytes and require matching ASTs parsed from file bytes, ignoring location
  metadata. Trust upstream execution evidence under this normalization assumption.
  Exclude explicit attack announcements; direct witness-matching guards remain
  allowed.
- Preserve frozen ownership, phases, source identity and exposure history. Exclude
  training-side tasks that overlap development or protected panels. Exclude faulty
  benigns and their attacks when reference agreement establishes failure on an
  independently audited valid input; export other suspected defects with a flag.
- Render only the statement and normalized program for the monitor. Count tokens
  with its pinned tokenizer and chat template, including a declared completion
  allowance against the initial 2,048-token budget. Exclude over-budget pairs
  without truncation; keep labels and evidence in separate metadata.
- Publish versioned labeled pairs with provenance, exclusions and shortfalls.
  Report independent source pairs and fully matched tasks, including whether
  development reaches the pilot's 20-task minimum.

## Shared contracts

- TODO


