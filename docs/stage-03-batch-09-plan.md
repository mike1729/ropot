# Batch 9 development baseline

Run one bounded benign-generation baseline on the fixed eight-task development
smoke, then recommend the next experiment. Reuse `benign_loop.run` and the
reviewed [Batch 8 summary](stage-03-batch-08-plan.md); change no pipeline code.

## Scope and review boundaries

Review and commit Batch 8's implementation, tests and documentation separately
before starting Batch 9 implementation. Keep its changes outside this batch's
review and 400-line count.

Use one review batch on `main`, within 400 additions plus deletions excluding
tests, including documentation, under [the repository rules](AGENTS.md).
Prepare the small caller at `archive/stage03/experiments/batch09_development.py`, sizing record
and proposed gate-test revision below. Stop for review before dispatch; commit
the reviewed caller and test revision before live dispatch. Run the repository
suite after the test revision; report tests and generated evidence separately.
Add no runner framework, CLI or new report schema.
Selection/publication, recovery/resume, overlap, batch-32 tuning and caching wait.

## Offline sizing

Load only the first eight IDs from the pinned cohort's saved development initial
group through `study_tasks.load_cohort` and `load_task`. These are the fixed
2-introductory/6-interview smoke, in saved order, for both sources. Preserve the
other 42 development tasks and every construction, protected and deferred task.
Do not substitute easier tasks when a prompt is long or generation fails.

Without inference, build all eight initial requests with
`benign_loop.initial_requests`. Construct a long full repair history per model
from the longest measured initial history using `benign_loop.repair_request`, a
fabricated prior reply and a parent with `extraction="success"` and public
verification feedback. Size the reply/feedback near the declared allowances and
let the existing feedback renderer apply its caps. In the MLX Python, render and
count these exact messages with `local_generation.render` and the pinned runtime
tokenizer/options, including Ministral's regex fix. Record the examples and
complete `len(prompt_ids)` counts outside the experiment root; add no measurement
framework.

These repair counts are stress examples, not a proven maximum: future reply text
is unknown, and generated tokens can tokenize differently when inserted into a
chat history. Record the conservative sizing estimate and its assumptions. Check
that it plus the requested output fits model context and is supported by the
[existing length/batch evidence](inference-benchmark-results.md). The roughly-7k
repair measurements inform this decision; they do not guarantee memory safety
for the new output allowance. If sizing does not support the settings, stop for
review without dropping or substituting tasks.

Add no `max_input_tokens` field, runtime refusal path or adapter tests. Runtime
admission waits for a later worklist requiring it; this bounded smoke accepts the
reviewed empirical sizing assumption and existing technical-error stop.

## Pending repair qualification

Before study dispatch, review the proposed
[Batch 7b coverage criterion](stage-03-batch-07-plan.md#pending-rerun-criterion)
and its test-only implementation. Require a verified failure followed by an
extracted public-feedback repair verified as `passed` or `failed` on each model;
allow the other task's outcomes to vary, including length or extraction failures.
Use a task-neutral repair instruction, retaining both fixtures and their budgets.
Then rerun once against the unchanged `d6c311e` coordinator/adapter sources:
at most eight replies/2,048 output tokens, separate from the study below.
Stop for review on missing repair coverage, technical failure or source change;
do not repeat calls to force coverage. Preserve the historical runs and statuses.
The gate qualifies plumbing, not cohort yield or 1,024-token completion behavior.

## One baseline per source

Call `benign_loop.run` once per source on those same eight tasks: actor Qwen and
synthetic Ministral, using the existing pinned Q8 artifacts and environments.
Run one model at a time. Use decode 8,
Qwen prefill 4/chunk 256 and Ministral prefill 2/chunk 2,048.

Freeze exact instructions and artifact identities in the reviewed
caller before dispatch. Start with a plain Python stdin/stdout solution request
and a repair request for a complete corrected replacement using public feedback;
omit the gate's artificial first-draft controls. Use temperature 0.2, top-p 0.95,
base seed 17, one repair at most, and 1,024 output tokens for both
initial and repair replies. Keep feedback caps at 512 bytes per field/4,096 total.

Allow at most 16 replies and 16,384 committed output tokens per source: combined
32 replies/32,768 tokens, excluding fixture qualification. This covers the entire
initial-plus-repair schedule. A technical error stops the study sequence for
review. No second recipe, restart or extra draws follow automatically.

Use new numbered directories under `data/03_benign_experiments/`, continuing the
existing order, with one directory per source. Save the Batch 8 summary and a
short report outside that root.

## Report and next decision

Give first-pass and after-repair counts per source, plus tasks with passing benigns
from both sources, against all eight original IDs. Keep failures, unresolved and
unsubmitted tasks visible; repeated draws add no independent tasks. Report saved
commitments, submissions, observed tokens/completeness and phase timings, including
failed runs, with fixture qualification effort separate. Explain the main failure
categories and whether the next experiment should change a prompt or budget.
Stop with one concrete recommendation under the
[Stage 03 progression](stage-03-plan.md#experiment-progression). Benign passes
alone establish neither attack yield nor fully matched development-panel feasibility.

## Preparation and validation — 2026-10-26

The [caller](../archive/stage03/experiments/batch09_development.py) exposes `measure(new_directory)`
in the pinned MLX Python and `run(gate_directory)` in repository Python. Measurement
loads tokenizers only. The baseline caller checks a passing, unchanged-source
two-model gate, continues numbered experiment order, stops before the second
source on technical failure and saves the Batch 8 summary even on interruption.
An existing `-batch09-` experiment directory prevents another dispatch, even if
interruption left no summary. The derived summary may be overwritten.

For a fresh offline sizing record, run from the repository root:

```sh
/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python -c \
  'from experiments.batch09_development import measure; measure("data/03_generation_checks/batch09-sizing-review")'
```

The retained [sizing and check record](../data/03_generation_checks/batch-09-preparation-20261026/validation.json)
uses the coordinator's initial/repair helpers and actual MLX tokenizer wrapper:

| Source | Initial prompt range | Stress repair input | Input plus 1,024 output |
| --- | ---: | ---: | ---: |
| Actor | 259–745 | 5,686 | 6,710 |
| Synthetic | 245–727 | 5,672 | 6,696 |

Both examples use 1,024 fabricated reply tokens and 4,096 rendered feedback bytes.
Both configured contexts are 262,144 tokens. Stress inputs are below the historical
6,882–7,196-token repair measurements; the empirical assumption above still applies.
MLX import required Metal access, but no weights or inference were used.

All 14 focused tests pass. The repository suite passes 281 tests with eleven
optional skips and live-test flags disabled. The revised gate accepts failed
repairs and length/format failures outside its coverage pair while checking saved
feedback, association and accounting. A read-only replay of retained 7b JSON finds
one coverage pair for actor and two for synthetic; this does not qualify current
sources or change historical statuses. Coordinator/adapter bytes match `d6c311e`.
The final evidence contains three JSON/text files totaling 207,119 bytes.
No live gate, Docker verification or study dispatch ran during that preparation.

## Execution and review fixes — 2026-10-26

The user authorized execution without committing. The repair gate passed and both
development experiments completed; see the [results](stage-03-batch-09-results.md).
The execution record preserves the exact caller/gate source copies and hashes;
sources stayed unchanged during execution. Pipeline modules still match `d6c311e`.

Subsequent review added reply-to-program byte checks and run-owned source binding
to the gate assertions, with a swapped-program regression test. A read-only
[association audit](../data/03_generation_checks/batch-09-review-20261026/association-audit.json)
passes on the saved live gate and all 16 verified study programs. The original
qualification remains intact; this audit makes no new model calls and does not
qualify the edited test source for another live run.

The caller now uses two literal option dictionaries and refuses existing Batch 9
directories, covering hard interruption without relying on summary existence.
Offline sizing matches the runtime's merged config and tokenizer options, checks
input plus output, and retains counts/stress messages without duplicated prompts
or token arrays. The [compact sizing record](../data/03_generation_checks/batch-09-review-20261026/tokenizer/sizing.json)
is 25,072 bytes with unchanged counts. All 17 focused tests and the repository's
284 tests pass, with eleven optional skips and live flags disabled.

Retain `batch-09-sizing-s7rst3r9` (205,523 bytes) as superseded sizing and
`batch-09-sizing-74goxt14` (empty) as the failed import before Metal access;
their paths and hashes are recorded in the association audit.
