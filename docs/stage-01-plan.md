# Stage 01 implementation plan: local APPS qualification

This file records implementation details for `01_qualified_pool`. Keep the
[pipeline overview](data-pipeline-overview.md) high-level. This plan incorporates
the decisions reviewed on 2026-10-23, including local execution instead of Docker.

## Steps at a glance

1. Verify and load pinned APPS splits — implemented in batch 4.
2. Screen tasks and deduplicate reference candidates — implemented in batch 1.
3. Map English examples and require a nonvisible stored case — implemented in batch 2.
4. Implement local Python 3.12 execution with deadlines, output caps and cleanup —
   implemented in batch 3.
5. Add the qualification CLI and try references until three pass every stored case —
   implemented in batch 4.
6. Export tasks and provenance atomically, then smoke-test before full qualification —
   implemented in batch 4; full qualification completed on 2026-10-23.

## Goal and current state

Produce `data/01_qualified_pool/` with tasks that pass static screening, statement
example mapping and local reference execution. Qualification here is under the
declared local runtime; later Inspect verification is a separate required gate.

Static screening, English example mapping and the nonvisible-case gate are
implemented with regression tests, as are the local runner, qualification CLI and
atomic export. The existing download command remains unchanged.

The full run qualified 1,450 tasks: 2 train and 1,448 test (335 introductory,
1,113 interview in test). It recorded 23 reference rejections and 128,506 case
attempts in the audit total, taking 648 seconds. Downstream analysis and allocation
use only test tasks; retain the train evidence unchanged. The next report is
`01_pool_analysis`; see the [analysis module](../src/data-pipeline/stage01/analyze.py).

Analysis and test export were republished from an immutable archive of commit
`ababa96`, with identical measurements and task bytes. The run record is
`data/runs/stage01-republish-ababa96.json`; earlier outputs are preserved under
`data/archive/stage01-before-ababa96-republish/` with their original provenance.

Batch 2 entry point: pass `screen_task(row, split)` to
`map_statement_examples_to_cases` in `src/data-pipeline/stage01/statement_examples.py`.
It returns a new candidate with `example_format`,
ordered `example_mappings` (sample text, exact answer and all matching original
case indices), per-case `visible` flags and visible/nonvisible case counts.
It does not modify the screened candidate; unchanged metadata and source lists
are shared. Original stored case content and indices are preserved. Catch
`Rejected` specifically for exclusions; passing these steps is not qualification.

Supported syntax is a single Codeforces `Example`/`Examples` section with alternating
`Input`/`Output` lines, or AtCoder `Sample Input`/`Sample Output` sections: one
unnumbered pair or a series with unique matching positive numbers. Mapping input
and output fields retain raw sample slices, including surrounding whitespace.
AtCoder output occupies the first paragraph and its following blank separator;
explanation remains in the original question. A further numeric paragraph is
rejected as ambiguous. Unsupported labels anywhere in the statement are excluded.
Unknown sample headings, incomplete pairs and mixed formats are excluded. Inputs
match by whitespace tokens; answers retain exact spelling, including `-0`.

Batch 3 API: `LocalRunner` in `src/data-pipeline/stage01/local_runner.py` (see its docstrings).

Batch 4 API: `qualify` in `src/data-pipeline/stage01/qualify.py` verifies both pinned files,
streams rows through bounded task workers and publishes only after every task has
been processed. `qualify_task` composes screening, example mapping and reference
execution; only `Rejected` counts as an exclusion. Exported tasks retain the full
`original_row` alongside decoded data and verification attempts. The manifest
hashes current pipeline sources, including uncommitted changes.
Tasks with insufficient passing references also retain source indices, attempt
counts/timings and each reference's first failure (including bounded stdout/stderr) in
`reference_rejections/`, listed in the manifest. This records actual dependency
failures without installing packages. Diagnostic streams can include source
excerpts; full rejected task records are not copied.

Real-data smoke on 2026-10-23: both pinned splits passed size/SHA-256, 5,000-row
and unique-ID checks. Train tasks 1605 (59 cases) and 1607 (43 cases) each qualified
with source indices 0, 1 and 2 under Python 3.12.13/macOS arm64. All 306 case runs
passed; maximum case time was below 0.026 seconds. JSON round-trips preserved their
original rows. Temporary smoke artifacts were removed; no partial pool was published.

Defer dataset statement/input/output size thresholds and the definition of
"short" until the next analysis. Preserve original content and record lengths.
Token counting, distribution reports and cohort allocation are outside this step.

## Screening and reference selection

- Verify both raw APPS splits against the downloader's pinned revision, file sizes
  and SHA-256 checksums. The pinned bytes fix the 5,000 rows and unique IDs per
  split; preflight records that known count without parsing the files again.
  Do not download implicitly. Stream train then test, preserving `(split, id)` identities.
- Keep introductory/interview stdin/stdout tasks. Decode nested `solutions` and
  `input_output` JSON and reject malformed or empty records, mismatched case counts
  and function-call tasks. Absent/null/blank serialized fields receive
  `missing_solutions` or `missing_cases`; nonempty fields that fail decoding or
  have the wrong decoded type retain `invalid_solutions`/`invalid_input_output`.
- Exclude Russian statements and support English Codeforces and AtCoder example
  formats only. List-valued cases are unsupported serialized lines, not malformed
  data; label that exclusion explicitly and do not join or flatten them.
- After stripping surrounding whitespace, every stored answer must match
  `-?(0|[1-9][0-9]*)`. Preserve the token's spelling; never parse it as an integer
  or normalize leading zeros/signs. Require exactly one stdout token equal to it.
  Support large answers, including test 927's 4,897- and 5,000-digit outputs.
- Conservatively exclude explicit arbitrary-answer and string/character-output
  instructions, including "any of them will be accepted". Require a nonempty
  dedicated `Output` or `Outputs` section; otherwise use `missing_output_section`.
  Apply that presence gate after case/interface checks to retain their primary
  exclusion reasons for missing cases and unsupported tasks.
  A sample-output section alone does not replace the output specification.
  Check string/character instructions within three intervening words on the same
  line, retaining the numeric-target exemption for phrases such as "number of
  strings". Preserve deterministic tie-break cases. These wording heuristics and
  numeric syntax do not prove semantic uniqueness or integer-valued outputs.
- Map every statement example to a stored input and matching answer. Reject absent,
  unsupported, conflicting or ambiguous mappings. Compare input whitespace tokens
  for visibility, mark every matching stored case visible, and require at least
  one nonvisible case. Reject contradictory answers for equivalent inputs.
  Preserve original inputs and case indices for execution.
- Deduplicate supplied sources using Python tokens, ignoring comments and
  insignificant whitespace but retaining significant newlines and block structure.
  Preserve original sources and indices; skip blank, comment-only or untokenizable
  entries. Require at least three token-distinct reference candidates.
- Try distinct references in dataset order, stopping after three pass every stored
  case or all are exhausted. Stop a reference at its first failing case. Do not
  rewrite supplied code or automatically install missing libraries.

## Local execution contract

Document this qualification entry point, using Python 3.12 for preprocessing too:

```sh
/opt/homebrew/bin/python3.12 -m src.data-pipeline qualify
```

Run each reference/case with this interpreter command, without a shell:

```text
/opt/homebrew/bin/python3.12 -I -B -X int_max_str_digits=0 <temporary-source>
```

- Preflight the explicit interpreter and require Python 3.12. Record its actual
  full version, requested/resolved executable, flags, macOS version and architecture.
  The checked installation is 3.12.13. Never fall back to the default Python 3.14.
  Use the explicit `-X` flag: `-I` ignores `PYTHONINTMAXSTRDIGITS`.
- Use a fresh process, source file and temporary working directory for every case.
  Supply original stdin through a pipe and only this environment: `PATH=/usr/bin:/bin`,
  `LANG=C`, `LC_ALL=C`, and a case-specific `TMPDIR`.
- Use `start_new_session=True` and at most eight concurrent tasks. Local process
  execution is not a sandbox. Apply no memory limit and record
  `memory_limit_bytes: null`; do not claim Docker's former CPU/PID/network/filesystem
  controls. `-I` does not exclude system site-packages.
- Measure the five-second deadline using a monotonic clock immediately before
  process creation. Include interpreter startup, stdin transfer, execution and
  output draining; exclude preparing source/working files and final cleanup.
- Write stdin and drain both output streams concurrently, avoiding deadlocks with
  large inputs or output-before-read programs. Bound buffered stdout at 1 MiB and
  stderr at 64 KiB; terminate the process group when either cap is reached.
- Require exit zero and exact answer matching. Nonempty stderr below its cap is
  diagnostic. Invalid UTF-8 stdout fails the reference; diagnostic stderr may use
  replacement decoding. Early stdin closure is not itself an infrastructure error.
- Keep the deadline active if descendants retain pipes after the parent exits.
  In cleanup, kill remaining members of the case's process group, reap the direct
  child, close pipes and remove temporary files, including after normal completion.
  Clean active groups on exceptions, SIGINT and SIGTERM as well as timeout/overflow.
- Nonzero exits (including 125-127), signals, wrong answers, timeouts and output
  overflow fail the reference. Interpreter, parent filesystem/process-management
  or unsuccessful cleanup errors abort qualification.
- Record per-case elapsed time and each reference's maximum case time and total
  wall time, retaining evidence needed to identify references near the limit.

Stage 01 needs no image digest, container flags, container cleanup, Docker-specific
failure handling or Docker test fixtures.

## Command and artifact contract

Add `qualify` with `--input-dir`, `--output-dir` and `--workers`, defaulting to
`data/raw/apps`, `data/01_qualified_pool` and eight workers. No new Python dependency
is needed for this implementation.

- Write one `tasks/<split>_<id>.json` per locally qualified task, containing original
  statement/source metadata, original and decoded cases, exact answer tokens,
  example mappings and visibility, supplied sources, the three passing reference
  indices and verification attempt summaries. Preserve original indices/content.
- Record raw/distinct/passing reference counts, visible/nonvisible case counts and
  character/byte lengths for statements, inputs, outputs and sources. These are
  preprocessing evidence, not a distribution-analysis report.
- Write `manifest.json` listing qualified task paths, source revision/checksums,
  pipeline/runtime provenance, interpreter command/version, execution limits,
  worker count, accepted counts and aggregate exclusion reasons. Identify local
  execution and required Inspect re-verification explicitly.
- Build under a temporary sibling directory and publish by rename only after a
  successful complete run. Refuse an existing destination. Do not publish a partial
  qualified pool or add resume/cache infrastructure in this version.

This version assumes a single publisher and stable input files. Rename provides
atomic visibility, without a power-loss durability guarantee.

## Deferred verification: stages 03 and 05

Before stage 03 generates programs, re-verify each selected task's same three
passing references against every stored case in the actual Inspect Linux sandbox,
including its chosen memory and timing limits. Require all three to pass before
using the task. Preserve local evidence and record Linux verification separately.

Use Python 3.12, the same interpreter flags and exact answer comparison. Record each
environment's exact patch version and configuration rather than claiming identical
macOS/Linux runtimes. Stage 05 reuses the frozen Inspect contract; a change to the
execution contract requires renewed reference verification before acquisition.

These checks belong to the later Inspect stages. Implement neither an Inspect
runner nor Linux memory qualification in stage 01.

## Validation and review batches

- Screening: exact canonical tokens, leading-zero rejection, 4,897/5,000-digit
  answers, malformed data, task scope, Russian exclusion, unsupported line lists,
  wording heuristics, token distinctness and preserved source order.
- Examples: both supported English formats, complete matching, ambiguous/missing
  examples, contradictory expected answers, duplicate visible inputs and the
  absence of a nonvisible case.
- Local runner: correct/wrong/extra output, huge answers, exceptions, exit 125,
  infinite loops, stdout/stderr overflow separately, early stdin closure, large
  stdin with output before reading, inherited pipes, process-group termination,
  interruption cleanup and infrastructure failures.
- Export: catch `Rejected` specifically for task exclusions; invalid split/ID
  `ValueError`s and other pipeline errors must abort rather than enter exclusion
  counts. Stop after the third passing reference, preserve provenance and counts,
  distinguish reference failures from run failures, refuse existing output, and
  publish atomically. Run a small real-data smoke test before full qualification.

Continue on `main` in four coherent review batches: (1) static screening, already
implemented; (2) English example mapping; (3) local subprocess runner; (4) CLI,
orchestration, export and usage documentation. Count additions plus deletions and
keep each batch within 500 changed lines. Stop for review between batches as
required by [the repository instructions](AGENTS.md).
