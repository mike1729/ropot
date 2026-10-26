# Batch 8 experiment summary

Summarize saved experiment JSON for Batch 9's next-experiment and scale decisions.
Selection and publication wait for a downstream stage needing a program panel.

## Scope and review boundaries

Consume the reviewed `d6c311e` format of `inputs.json` and flat `outcome.json`.
Trust saved coordinator counters and verdicts under their recorded contracts.
Report unsupported formats per experiment. Read no native logs, replies or snapshots.
The live repair gate remains before overlap or Batch 9 dispatch, not this reader.

Implement on `main` in one review batch, following [the repository rules](AGENTS.md):
at most 400 additions plus deletions excluding tests, including documentation.
Aim for 100–150 production lines; stop for review.

## Numbered experiment directories

Scan immediate experiment directories and sort by name, using fixed-width numeric
prefixes such as `0001-dev-smoke-qwen`. Require unique prefixes and valid names.
Study runs use `data/03_benign_experiments/`; tests use an isolated fixture root.
Report every experiment directory; no inventory file or successor rule is needed.
Refuse root-level naming errors, known fixture/study mixing or mixed cohort digests.
Unknown provenance from damaged inputs stays a per-experiment diagnostic. Save
reports outside the root. Numbering is a caller convention; do not renumber existing
runs or infer that directory-existence refusal prevents renaming. The coordinator
needs no change because the caller already chooses `output_dir`.

## Accounting and reports

Add `stage03/experiment_summary.py` with `summarize(root)`, returning a
JSON-serializable summary that its caller may save:

- Per experiment: source, fixture/cohort identity, recipe, model-manifest identity,
  status, recorded counters and phase timings. For terminal runs, count the states
  in `outcome["tasks"]` and give the worklist size; compute no yield rates.
- Per worklisted task/source: copy each saved binding as-is, including minimal fixtures;
  ordered experiment membership, saved submission/attempt states, every recorded
  pass and latest state, including `not_run`. Membership alone is not exposure.
  Key cells by source, split, id and saved task sha256, so different fixture content
  at the same split/id has separate histories; expose that digest on each cell.
- Totals: sum each run's recorded commitments, submission counts/lower bounds,
  observed tokens and timings once; retain nulls and usage-completeness flags. Never
  refund reservations. Repeated draws are not independent tasks, and benign passes
  do not establish attack yield.

For nonterminal runs, recorded committed reservations stay exact; mark cells
unresolved and submissions, usage and timings as snapshots/lower bounds. Preserve
recorded completeness flags without making exact commitments look unknown.

Read errors are per experiment: report the problem and unresolved data, then continue.
Keep every readable outcome's counters even if its inputs are damaged. Unknown
effort makes that aggregate unknown/incomplete, retaining known subtotals; never
silently count it as zero. Only root-level problems above stop the whole report.
For intact producer outputs, an absent outcome means zero dispatch even if inputs
are truncated: inputs are written before initial progress, which precedes launch.
If `round-0000.json` exists, the missing outcome is demonstrably lost: report unknown
effort and unresolved state. Check only that round file's existence, not its contents.
An existing but malformed/unreadable outcome means unknown effort, not zero.
Diagnose missing/null counters, an unavailable timing block and missing study cohort
provenance per experiment. Unknown submission counts can occur in valid error runs;
the diagnostic explains the unknown without discarding other readable counters.
Add no stopped declarations, hashing or reconstruction.

## Deferred work and paused draft

Selection, program/task digest checks, cohort checks, publication and replacement
wait for their consumer's contracts; recovery/resume waits for a concrete need.
Never replay spent/uncertain requests or erase prior costs. Add no contract policies, re-derivation or drift guards.
Park the old reconciler and its tests under `unused/batch08-reconciliation/`, outside
the import tree and test discovery. Preserve their contents and retained
[offline check](../data/03_generation_checks/batch-08a-32qz00jk/validation.json)
as historical work, not validation of this summary or a base to extend.

## Acceptance and review evidence

Build normal test runs with Batch 7's `LoopFixture.run_loop`, which emits real
coordinator JSON using mocked children/verifiers. Modify JSON only for damaged or
unfinished cases; cover absent optional lower-bound fields and null timings.
Check numbered scan/order, duplicate prefixes, fixture/study and cohort mixing,
binding preservation, histories, repeated draws, final-state counts and recorded
totals, including different fixture content at the same split/id and coordinator-written
incomplete-native and cancelled outcomes. Check that one unreadable/malformed/unsupported run does not hide healthy
runs or unknown cost; distinguish absent outcome from unreadable outcome and keep
readable commitments exact for unfinished runs or damaged inputs. The summary
imports no Inspect/MLX and makes no Docker, model or sandbox calls. Run focused
tests and the repository suite; report non-test/test line counts separately.
No new live gate is needed.

## Implementation and validation — 2026-10-26

The standard-library reader exposes `summarize(root)` and reads only each immediate
experiment's `inputs.json` and `outcome.json`, with an existence check for the first
round when an outcome is absent. Names match
`NNNN-lowercase-hyphenated-name`; duplicate four-digit prefixes are refused.

```python
import importlib
import json
from pathlib import Path

reader = importlib.import_module("src.data-pipeline.stage03.experiment_summary")
report = reader.summarize("data/03_benign_experiments")
Path("data/03_benign_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
```

The result contains `experiments`, task/source `cells`, and `totals`. Each aggregate
has a `recorded_total` (null if any value is unknown), `known_subtotal`, and
`complete` flag. Nonterminal observations remain snapshots/lower bounds even when
their sum is known; committed reservations remain exact. Lower-bound counters
remain bounds. Per-run completeness flags and null timings are preserved.
An absent outcome without a saved first round contributes zero for every timing phase.
An outcome lost after round preparation contributes unknown effort. In an existing outcome,
a null phase remains unknown, including when the coordinator never ran that phase.
Completeness describes recorded effort, not format validation: readable commitment
counters can remain exact when the run also reports an unsupported-format problem.
Cell histories retain saved bindings and attempts; `passes` references every saved
passing attempt, while `latest_state` may be unresolved or failed after earlier passes.

All 26 focused tests pass, using actual coordinator JSON emitted by mocked Batch 7
fixtures. From the repository root, `.venv/bin/python -m unittest discover -s tests`
runs **267 tests with eleven optional skips**. A fresh
process confirms the reader neither imports Inspect/MLX nor invokes subprocesses.
The [JSON-only fixture check](../data/03_generation_checks/batch-08-summary-odjpckeh/validation.json)
copies only four coordinator JSON files from the retained two-model 7b run into an
isolated numbered root. Its report preserves seven recorded replies, 1,792 committed
tokens and 652 observed tokens. No original runs were renamed or modified.
That fixture root, summary, test output and validation comprise seven JSON/text
files totaling 47,171 bytes; no native logs or binary artifacts were copied.
The paused reconciler remains unchanged under `unused/batch08-reconciliation/`.
No new inference, Docker execution or live gate ran. Stop here for review.
