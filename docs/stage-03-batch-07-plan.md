# Batch 7 benign generation and repair loop

Compose [batched generation and extraction](stage-03-batch-06-plan.md) with
the existing [candidate verifier](stage-03-plan.md#candidate-verification-batch-3).
Produce an inspectable experiment containing every initial draw, repair, program,
verdict and cost. Begin with sequential generation and verification rounds.

## Scope and review boundary

Implement on `main`, within 400 additions plus deletions excluding tests, following
[the review rules](AGENTS.md). Report production/documentation and test changes
separately, then stop for review. Batch 6's size exception does not carry forward.
If the implementation cannot fit, narrow or split it before exceeding the limit.
This planning change runs no models or programs; implementation qualification uses
fabricated tasks only. Cohort experimentation remains Batch 9 work.

Keep one source and model per experiment. Reuse the pinned Q8 artifacts, patched
MLX environment and repository Inspect environment without new dependencies.
Use the current repository wrappers; keep Inspect pinned to `0.3.283` and preserve
its execution contract rather than adopting newer documentation's API features.
Selection publication, cross-run resume and cumulative reporting belong to Batch 8.
Measure generation/verification overlap in a separately reviewed follow-up after
this sequential loop passes; add no scheduler, persistent worker or cache system.

## Entry point and experiment inputs

Add a small `stage03/benign_loop.py` with a Python entry point accepting:

- A finite ordered worklist of `(binding, task)` pairs, one `actor` or `synthetic`
  source, and a new output directory. Study callers obtain pairs through
  `study_tasks.load_task` and supply the pinned cohort hash; fixture callers supply
  digests of their saved fabricated task artifacts and mark the run as fixtures.
- Concrete model, artifact-manifest and MLX-Python paths. Run the coordinator from
  the repository `.venv` so it can call verification directly. Keep the interpreter
  executable path without resolving its virtual-environment symlink.
- A recipe containing explicit system, initial and repair instructions, sampling
  settings, deterministic seeds indexed by worklist position and round, initial
  and repair `max_tokens`, and a nonnegative `max_repairs` per task.
- Explicit total reply and output-token allowances, plus decode, prefill and
  prefill-chunk settings. Use decode 8 initially, with Batch 6's per-model settings;
  long Ministral repair histories remain outside batch-32 qualification.

Reject duplicate worklist identities, invalid source, missing/nonpositive limits
and insufficient allowance for the complete initial worklist before dispatch.
Exclude booleans from integer limits; `max_repairs=0` disables repair. Check only
these experiment invariants and trust the existing task/model contracts. Preserve
saved ownership, construction phase and worklist order without allocating groups
or changing activation. Snapshot the task payloads and original bindings before
generation; label any reserialized snapshot digest separately from the published
task digest used by candidate identity and verification.

## Sequential rounds and repair policy

1. Build one initial request per task. Use only the statement (`question`) and
   recipe instructions in its system/user messages. Never serialize the complete
   task, references, private binding or hidden cases into a prompt. Save exact
   requests, their attempt mapping and reserved allowances before dispatch.
2. Call `local_generation.generate_batched` once for the round, preserving its
   input order and native records. After successful generation, call
   `extract_program` for each row. Save exact returned bytes before verification
   and obtain `study_tasks.candidate_identity` without normalizing the program.
3. Send all extracted programs, in their attempt order, to one
   `verification.verify_many(..., max_samples=4, concurrent_inference=False)` call.
   Do not start verification for an empty list. Bind results to task/program
   digests and returned native log/sample IDs. Execute every published case using
   the existing verifier; introduce no additional case selection or acceptance gate.
4. Remove passing tasks from further rounds. Eligible failures receive at most
   one new request per round, in original worklist order, while their declared
   repair and global allowances remain. Save the parent attempt link. Each repair
   is a new draw, even if its output bytes repeat an earlier candidate.
5. Stop when every task passes, no eligible task remains, a global allowance is
   exhausted, or a technical failure/interruption occurs. Record each task's final
   state without selecting or publishing a downstream program panel.

For a verification failure, append the exact preceding assistant reply and a user
message containing the recipe's repair instruction and only the returned
`feedback` projection. Preserve the conversation history. The rest of the verdict,
native logs, hidden indices, timing and reference solutions stay private. Do not
substitute supplied references or another source's solution into repair history.

`length`, extraction-format failure and Python syntax failure may consume a
declared repair too. Keep their original replies, classify them separately, and
append a fixed local message identifying the failure category and requesting a
complete replacement. Do not execute or salvage incomplete replies, or feed
implementation tracebacks into prompts. A repair budget authorizes these new
attempts; it does not authorize technical retries. Generation/cleanup failures,
verification `error`, preflight exceptions and broken evidence associations stop
the experiment without another model or execution call. Preserve any results
already obtained. Ordinary wrong answers remain candidate failures.

## Allowances and retained records

Before launching each round, reserve one reply and that request's full
`max_tokens` allowance for every dispatched row. Choose a stable prefix of
eligible repairs that fits both remaining allowances; do not shrink token limits
to squeeze in another request. Unscheduled repairs retain `budget_exhausted`
status. Keep committed allowances separate from observed submissions and actual
token usage. Do not refund a dispatched round's allowance after failure.

Use plain JSON and existing native generation/Inspect files, under a new
`data/03_benign_experiments/<run>/` directory:

- An immutable experiment input record contains worklist/bindings, task snapshots,
  cohort or fixture provenance, recipe, seeds, budgets, model-manifest digest,
  runtime paths and hashes of the loop and consumed repository modules.
- Each round retains its exact requests and attempt-to-row mapping, Batch 6's
  input/runtime/reply/outcome files, extracted programs and extraction outcomes,
  verification results and native `.eval` logs. Keep attempt identity distinct
  from candidate identity and verification identity.
- A replaceable progress/outcome JSON links these artifacts and records phase,
  task states, committed allowances, known submissions, observed tokens and errors.
  Write progress with temporary-file replacement at dispatch and phase boundaries;
  save raw generation output before extraction or scoring can fail.

Distinguish a completed experiment with exhausted/failed tasks from a technical
`error` or `cancelled` run. Record loading, prompt preparation, generation,
verification and overall elapsed time separately. Sum observed token counts only
where evidence exists; retain native counters independently on failure. Unknown
usage after hard termination stays unknown, with observed usage a lower bound.
Do not equate missing outcomes or zero delivered tokens with zero spent work.

Existing run directories are refused. Interrupted experiments retain enough
evidence to inspect attempts and later recover processing, but this batch adds
no resume command, automatic log reuse, redraw or re-execution. Batch 8 must bind
and reconcile these artifacts before allowing any continuation.

## Process boundary and interruption

Run generation as one synchronous, finite child invocation of this module under
the supplied MLX Python. Pass a saved round-file path and import MLX only in that
child; keep Inspect imports in the coordinator's verification path. Reuse
`generate_batched` unchanged. The child loads once per round, then exits before
verification; repair rounds reload the model and retain that loading cost.

Start the generation child in its own process group. On coordinator interruption,
stop further dispatch, forward interruption to that owned group and reap it.
Allow graceful Batch 6 cleanup; a subsequent explicit interruption kills and
reaps the group if loading/native work is stalled. Add no generation deadlines
or timeout probes. Inspect verification keeps its per-case program timeout and
native cancellation/teardown. Preserve the original interruption after recording
the available child/native evidence and any cleanup error.

Mark a round as dispatching before process creation. If launch/submission becomes
ambiguous or the child exits without a complete outcome, label affected rows
`submission_unknown` and retain the whole reservation. When an outcome exists,
preserve its `submitted`, `not_run`, completion and cancellation distinctions.
Never infer that an interrupted `insert` did no work because no UID was returned.
Forced termination may lose the child's final metadata; the coordinator must
still save its phase, artifact paths, committed budget and uncertainty. A hard
kill of the coordinator itself can leave only the last saved progress record.

## Validation

Use mocked generation/verification with otherwise valid fabricated tasks to test:

- Stable task/attempt association through mixed passes, wrong answers, format and
  syntax failures, token exhaustion and repairs; passing tasks receive no new draw.
- Reply/token ceilings, stable repair prefixes, deterministic seeds, zero repair
  allowance, duplicate IDs and refusal of existing output directories.
- Exact candidate bytes and digests, distinct attempts for repeated bytes, and
  one coalesced verification call per round with the unchanged execution settings.
- Initial and repair messages with private-data sentinels: only the statement,
  recipe, prior replies and permitted public/local feedback enter model input.
- Failure before/after dispatch, missing child outcomes, partial native rows,
  verifier infrastructure errors and interruption during each phase. Retain
  completed artifacts and conservative accounting with no retry or later dispatch.

Exercise actual process-group interruption and reaping with a tiny Python child
that makes zero inference requests. Cover graceful exit, repeated interruption,
nonzero exit and missing outcome. Keep the existing Batch 6 native cleanup tests.

Add an opt-in end-to-end smoke through the coordinator for each model, using two
fabricated tasks: square and distinct-count programs. Use an explicit initial
control instruction to make the square reply return zero; let the distinct-count
task solve normally. The repair instruction explicitly ends that control and
requests the correct square program using only public feedback. Require the
initial square to fail verification, the other task to pass without repair, and
the repaired square to pass. Unexpected outcomes stop qualification for review
without redrawing; these controlled failures measure plumbing, not benign yield.

Use temperature 0.2/top-p 0.95, saved seeds, 256 tokens per reply, one repair per
task, decode 8 and Batch 6's per-model prefill settings. Cap each source at four
replies/1,024 output-token allowances: **eight replies and 2,048 tokens total**,
including unused repair slots, without retries. Retain every attempted reply,
program and native verification log under `data/03_generation_checks`, with
separate loading/generation/verification times. Any failed or inconclusive gate
leaves qualification pending; saved-output inspection needs no inference.

Run the repository suite and the separately gated smoke when implementing this
batch. Report mechanical checks, live outcomes, actual diff sizes and retained
evidence. Stop for review before overlap, selection/resume or cohort acquisition.
