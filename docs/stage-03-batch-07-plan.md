# Batch 7 benign generation and repair loop

Compose [batched generation and extraction](stage-03-batch-06-plan.md) with
the existing [candidate verifier](stage-03-plan.md#candidate-verification-batch-3).
Produce an inspectable experiment containing every initial draw, repair, program,
verdict and cost. Begin with sequential generation and verification rounds.

## Scope and review boundary

Before 7a, complete Batch 6's final decode-8/decode-32 qualification and establish
its reviewed source/tests as a separate committed baseline. Consume its generation
functions unchanged; 7b adds only their child CLI entry point. Keep Batch 6 work
out of Batch 7's change counts. Implement on `main` in
these two review batches, each within 400 additions plus deletions excluding
tests, following [the review rules](AGENTS.md):

| Part | Implementation and acceptance boundary |
| --- | --- |
| 7a | One initial generate/extract/verify round with `max_repairs=0` and explicit reply/output-token allowances; experiment records; generation-child control; and cancellation normalization at the runner/verifier boundary. Validate association, accounting, real child-process control, verification SIGINT and a bounded two-model MLX-child smoke. Stop for review with a usable initial-round API. |
| 7b | Extend that API with bounded repair rounds, feedback rendering, deterministic seeds across rounds and repair outcomes. Validate repair limits and private-data boundaries, then run the bounded two-model smoke below. Stop for review before overlap or acquisition. |

Add repair-specific recipe fields and behavior in 7b; 7a rejects nonzero repair
limits. Report production/documentation and test totals separately for each part.
Batch 6's size exception does not carry forward. This planning change runs no
models or programs; qualification uses fabricated tasks only. Cohort experiments
remain Batch 9 work.

Keep one source and model per experiment. Reuse the pinned Q8 artifacts, patched
MLX environment and repository Inspect environment without new dependencies.
Use the current repository wrappers and keep Inspect pinned to `0.3.283`.
Preserve execution limits and case semantics; the narrow cancellation correction
below changes source hashes in the execution identity. Do not adopt newer APIs.
Evidence reconciliation, selection publication and cumulative reporting belong to
[Batch 8](stage-03-batch-08-plan.md). General extraction recovery and dispatch
continuation wait for a demonstrated consumer and a separate bounded review plan;
spent or uncertain requests must never be replayed.
Measure generation/verification overlap in a separately reviewed follow-up after
this sequential loop passes; add no scheduler, persistent worker or cache system.

## Entry point and experiment inputs

Add a small `stage03/benign_loop.py` with a Python entry point accepting:

- A finite ordered worklist of `(binding, task)` pairs, one `actor` or `synthetic`
  source, and a new output directory. Study callers obtain pairs through
  `study_tasks.load_task` and supply the pinned cohort hash; fixture callers supply
  complete bindings (`split`, integer `id`, task-artifact `sha256`) and mark the
  run as fixtures. These fields are required by `candidate_identity`.
- Concrete model, artifact-manifest and MLX-Python paths. Run the coordinator from
  the repository `.venv` so it can call verification directly. Keep the interpreter
  executable path without resolving its virtual-environment symlink.
- A recipe containing explicit system, initial and repair instructions, sampling
  settings, `base_seed`, initial and repair `max_tokens`, and a nonnegative
  `max_repairs` per task. For original worklist length `N`, use
  `base_seed + round * N + position`, with round zero for initial requests and the
  original position even when later rounds shrink. The coordinator requires an
  integer `base_seed` excluding booleans and checks the whole potential schedule
  satisfies `0 <= seed < 2**64`, without wrapping, before starting a child. Check
  the largest seed using `N` and the declared `max_repairs`. Seeds identify draws;
  changed batch layouts can change replies despite identical prompts and seeds.
- Explicit `feedback_field_bytes` and `feedback_bytes` limits for repair feedback
  rendered below, separately from output-token allowances.
- Explicit total reply and output-token allowances, plus decode, prefill and
  prefill-chunk settings. Use decode 8 initially, with Batch 6's per-model settings;
  long Ministral repair histories remain outside batch-32 qualification.

Reject an empty worklist, duplicate identities, invalid source, missing/nonpositive
limits and insufficient allowance for the complete initial worklist before dispatch.
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
`feedback` projection, rendered under the limits below. Preserve conversation
history without silently shortening it. Other verdict fields, native logs,
hidden indices, timing and reference solutions stay private. Do not substitute
references or another source's solution into history.

Render a copy of `feedback` deterministically. Cap each visible case's `input`,
`answer`, `stdout`, `stderr` and `diagnostic` string at `feedback_field_bytes` in
UTF-8. Truncate at a valid character boundary and append `[truncated]`, counting
the marker inside the cap; require both byte caps to accommodate the marker.
Keep status fields and nulls unchanged. Render the overall and hidden statuses
first as a fixed summary, then compact JSON (`ensure_ascii=False`) for visible
cases, placing a failing visible case before passing cases when one exists.
Reserve room for the summary and truncation marker inside `feedback_bytes`;
reject a recipe whose cap cannot fit them. If the field-capped text is too large,
remove detail from passing cases first, then omit passing entries from the end
until it fits. Only then truncate remaining case text at a UTF-8 boundary using
the same marker rule; retain the summary. Mark omissions explicitly. This is
display text, not JSON for reparsing. Save full private feedback and the exact
bounded text, including reached limits. Rendering order changes no execution or
verification order and introduces no hidden indices.

Parts 7a/7b use controlled fixture statements, explicit output-token and repair
limits, and bounded feedback. They add no input-token cap or prompt-length refusal
path. Preserve the prompt/token counts already recorded by Batch 6. Before cohort
experiments, Batch 9 must measure tokenized statements and representative repair
histories, choose model/batch-specific input limits and handling of oversized
tasks, then implement and qualify that admission check. Do not invent a universal
token threshold from the short-fixture smokes or alter the Batch 6 prerequisite.

`length`, extraction-format failure and Python syntax failure may consume a
declared repair too. Keep their original replies, classify them separately, and
append a fixed local message identifying the failure category and requesting a
complete replacement. Do not execute or salvage incomplete replies, or feed
implementation tracebacks into prompts. A repair budget authorizes these new
attempts; it does not authorize technical retries. Generation/cleanup failures,
verification `error`, preflight exceptions and broken evidence associations stop
the experiment without another model or execution call. Preserve any results
already obtained. Ordinary wrong answers remain candidate failures.

Use the extractor's exception types for format/syntax categories: its
`SyntaxError` category also includes prose-only replies with no code fence.
Catch compiler `MemoryError` and `RecursionError` separately and record extraction
category `compiler_resource`. Treat both as technical stops, without repair or
further generation/verification calls. Preserve the reply and exception; host
resource exhaustion does not establish that the candidate is incorrect.

## Allowances and retained records

Before launching each round, reserve one reply and that request's full
`max_tokens` allowance for every dispatched row. Choose a stable prefix of
eligible repairs that fits both remaining allowances; do not shrink token limits
to squeeze in another request. Unscheduled repairs retain `budget_exhausted`
status. Keep committed allowances separate from observed submissions and actual
token usage. Committed allowances are never refunded, including after successful
short replies, cancellation or failure.

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

Final task states are `passed`, `repairs_exhausted`, `budget_exhausted`, `not_run`,
`submission_unknown`, `error` or `cancelled`. A known passing verdict takes
precedence. Use `repairs_exhausted` when all permitted draws failed, and
`budget_exhausted` when an otherwise eligible repair cannot fit global allowances.
`not_run` means the initial request definitely never reached native submission;
`submission_unknown` means that fact cannot be established. An attempted task
left unresolved by a technical stop or interruption is `error` or `cancelled`.
Retain the stop reason and prior attempt outcomes separately.

Existing run directories are refused. Interrupted experiments retain enough
evidence to inspect attempts and later recover processing, but this batch adds
no resume command, automatic log reuse, redraw or re-execution. Batch 8 must bind
and reconcile these artifacts before allowing any continuation.

Changing the runner/verifier source invalidates older `reuse` keys under the
existing execution contract. Preserve old logs and their identities; do not
rewrite them or rerun historical verification to make them match the new source.

## Process boundary and interruption

Run generation as one synchronous, finite invocation of `stage03.local_generation` under
the supplied MLX Python. Pass a saved round-file path and import MLX only in that
child; keep Inspect imports in the coordinator's verification path. Reuse
`generate_batched` unchanged. The child loads once per round, then exits before
verification; repair rounds reload the model and retain that loading cost.

Start the generation child in its own process group with `Popen` and an explicit
wait/interrupt loop. Do not use `subprocess.run` or `call`: their exception paths
kill the child on `KeyboardInterrupt`, bypassing graceful cleanup. Record its PID
and process-group ID in progress immediately after creation. On coordinator
interruption, stop dispatch, forward SIGINT to that owned group and reap it.
Allow graceful Batch 6 cleanup; a subsequent explicit interruption sends SIGKILL
and reaps the group if loading/native work is stalled. Add no generation deadlines
or timeout probes. Preserve the original interruption after recording available
child/native evidence and any cleanup error. Process-group isolation does not
stop a child when the coordinator is hard-killed or its terminal closes; the
recorded identity supports manual cleanup, not automatic orphan recovery.

In 7a, normalize confirmed Inspect cancellation at the runner/verifier boundary
to `KeyboardInterrupt` after native log saving and teardown. Inspect may return
an empty log list after SIGINT, so the verifier must not index it unconditionally.
Create a fresh, empty native log directory for every verification call and refuse
a prepopulated destination. If no log is returned, inspect only that directory
with the public log API and check for a cancelled run matching this invocation's
execution contract. The initially empty directory links that evidence to this
call's candidate worklist. An earlier identical cancelled run
must not supply evidence for this call. Raise `KeyboardInterrupt`
only with cancellation evidence; an unexplained empty return is a descriptive
infrastructure error. Preserve any interruption that already propagates and keep
the cancelled log path in progress. Verification retains its per-case timeout.
A second SIGINT can interrupt Inspect's asyncio teardown and leave sandboxes
running. If teardown is interrupted, record cleanup as unconfirmed rather than
claiming sandbox removal; the single-interruption test below requires completion.

Mark a round as dispatching before process creation. If launch/submission becomes
ambiguous or the child exits without a complete outcome, label affected rows
`submission_unknown` and retain the whole reservation. When an outcome exists,
preserve its `submitted`, `not_run`, completion and cancellation distinctions.
Treat truncated, unparsable or incomplete `outcome.json` as missing evidence:
the existing generation writer uses a non-atomic write. Keep that file intact.
Never infer that an interrupted `insert` did no work because no UID was returned.
SIGKILL, including the second-interruption path, loses unfinished partial replies
and token lists held only in child memory as well as final metadata. Completed
reply files may remain; missing outcomes still leave usage/submission uncertain.
The coordinator must save its phase, artifact paths, committed budget and that
uncertainty. A hard kill of the coordinator itself can leave only the last saved
progress record.

## Validation

For 7a, use otherwise valid fabricated tasks and mocked generation/verification
to test initial association, exact bytes/digests, allowances, input-only privacy,
request construction, missing/malformed outcomes and technical stops.
Inject `MemoryError` and `RecursionError` during extraction of otherwise
valid rows and require `compiler_resource` with no repair or later execution.
Test seed-range boundaries before child creation. Cover a direct interruption,
a returned cancelled log, an empty return with fresh cancellation evidence, an
unexplained empty return and refusal of a prepopulated log directory separately.

For 7b, extend those checks to cover:

- Stable task/attempt association through mixed passes, wrong answers, format and
  syntax failures, token exhaustion and repairs; passing tasks receive no new draw.
- Reply/token ceilings, stable repair prefixes, deterministic seeds, zero repair
  allowance, duplicate IDs and refusal of existing output directories. Check seeds
  against original positions and ensure committed allowances never receive refunds.
- Exact candidate bytes and digests, distinct attempts for repeated bytes, and
  one coalesced verification call per round with the unchanged execution settings.
- Initial and repair messages with private-data sentinels: only the statement,
  recipe, prior replies and permitted public/local feedback enter model input.
- Large/multibyte feedback fields, many visible cases, aggregate rendering limits
  and growing history. Retain the status summary and failing visible case before
  passing detail under tight caps, while preserving full private evidence.
- Failure before/after dispatch, missing child outcomes, partial native rows,
  verifier infrastructure errors and interruption during each phase. Retain
  completed artifacts and conservative accounting with no retry or later dispatch.

Exercise actual process-group interruption and reaping with a tiny Python child
that makes zero inference requests. Cover graceful exit, repeated interruption,
nonzero exit and missing outcome in 7a. Extend the existing Docker SIGINT test
through `verify_many` to require caller-visible `KeyboardInterrupt`, a cancelled
native log and sandbox teardown. Mocked interruption tests alone do not qualify
this boundary. Keep the existing Batch 6 native cleanup tests.

Before the 7a review, run a separately gated live smoke through the actual
round-file-to-MLX-child entry point, once per model. Use two ordinary fabricated
tasks (square and distinct count), with no initial failure control and
`max_repairs=0`. Use decode 8, Batch 6's per-model prefill settings, temperature
0.2/top-p 0.95, saved seeds and 256 output tokens per reply. Require both replies
to stop naturally, extract and pass coalesced Docker
verification. Cap each source at two replies/512 output tokens: **four replies
and 1,024 output tokens total**. Retain all round files, replies and native logs
under `data/03_generation_checks`; stop on a failed or inconclusive gate without
redrawing. These calls check child launch and round-file integration, not
decode-32 or long-context qualification.

In 7b, add an opt-in end-to-end smoke through the coordinator for each model,
using two fabricated tasks: square and distinct-count programs. Put the initial
control instruction to return zero in the square fixture's `question`, explicitly
labelled as applying only to its first draft. Retain its correct square
specification and nonzero expected answer. Give distinct-count its ordinary
statement. The shared repair instruction explicitly ends the initial control and
requests the correct square program using only public feedback. Require the
initial square to fail verification, the other task to pass without repair, and
the repaired square to pass. Unexpected outcomes stop qualification for review
without redrawing; these controlled failures measure plumbing, not benign yield.

Use temperature 0.2/top-p 0.95, saved seeds, 256 tokens per reply, one repair per
task, 512 bytes per feedback field, 4,096 bytes of rendered feedback, decode 8
and Batch 6's per-model prefill settings.
Cap each source at four replies/1,024 output-token allowances:
**eight replies and 2,048 tokens total**,
including unused repair slots, without retries. Retain every attempted reply,
program and native verification log under `data/03_generation_checks`, with
separate loading/generation/verification times. Any failed or inconclusive gate
leaves qualification pending; saved-output inspection needs no inference.

Run the repository suite for each part and the separately gated model smoke in
each part. Part 7a's process-control and SIGINT probes use zero model calls;
its live smoke has the separate four-reply cap above. The 7a/7b model smokes
together allow **12 replies and 3,072 output tokens**, separate from Batch 6's
qualification. Full repair-loop qualification remains pending until 7b; its smoke
qualifies decode 8 only. Report checks, actual diff sizes and retained evidence
at each boundary. Stop after 7a for review, then again after 7b before overlap,
resume or cohort acquisition. Offline Batch 8 reconciliation/publication may
proceed after a reviewed committed format baseline and compatible retained
artifacts exist, while live repair qualification remains pending. Complete that
gate before overlap or Batch 9 dispatch. Batch 8b's activation-digest field is
bookkeeping only: unchanged requests and execution need no additional model gate.

## 7a validation — 2026-10-26

The initial-round API is implemented with repairs disabled. The repository suite
passes 207 tests with ten optional skips. Actual process-group tests cover graceful
interruption, explicit second-interruption kill/reaping and nonzero exit.

The four-reply [child integration smoke](../data/03_generation_checks/batch-07a-8tnfpods/qualification.json)
passed in 17.262 seconds: both programs passed for each model, using 186 Qwen and
165 Ministral output tokens (351 total under the 1,024-token allowance).
The requested removal of cancellation identity hashing/task-version plumbing came
after that smoke. The final simplified boundary passed the real Docker SIGINT
and batch-verification checks with zero further model calls; see the
[final check record](../data/03_generation_checks/batch-07a-8tnfpods/post-simplification-checks.json).
The subsequent phase-helper refactor passed a
[saved-output processing replay](../data/03_generation_checks/batch-07a-8tnfpods-refactor-replay/replay.json):
all four exact programs, candidate identities, verdicts and allowance totals matched,
with zero model calls or sandbox executions. Replay evidence comprises 29 files
totaling 53,888 bytes; original native logs and qualification hashes remain intact.
The smoke retains 36 files totaling 78,108 bytes, including two binary native
logs totaling 21,346 bytes, fixture snapshots and copies of the pinned manifests.

Before the final, unchanged Batch 6 qualification, a superseded prompt-cap variant
also passed its [finite worklist](../data/03_generation_checks/batch-06-_t4adhp6/qualification.json):
88 submissions and 7,090 output tokens, retained separately in 294 files totaling
700,309 bytes. The cap edits were removed; this is supplemental evidence only.
Part 7b, overlap, selection/resume and cohort acquisition remain pending review.

## 7b validation — 2026-10-26

The repair API and bounded public-feedback renderer are implemented. The repository
suite passes 241 tests with eleven optional skips. Fixture tests cover shrinking
rounds, deterministic seeds, full histories, repeated candidate bytes, mixed
failures, stable budget prefixes and technical stops during later rounds.
Review fixes preserve already exhausted tasks across later technical stops. Round
records reference the single attempt list through `attempt_ids` and own their
artifact paths and child identity. Planned destinations remain recorded if writing
fails. Feedback fitting caches case encodings, preserves nulls and shares cap
validation with the coordinator. These changes made no further model calls.
The child now enters the generation module directly, without coordinator/task
imports. A recorded `Popen` `OSError` establishes non-submission while retaining
reservations; interrupted launches without that evidence stay unknown. Tests cover
both cases and repair-launch failure, with no further model calls.

The [bounded live gate](../data/03_generation_checks/batch-07b-hsm_lima/qualification.json)
stopped for review after Qwen's two initial replies: both programs passed, so the
square's first-draft control failed to induce the required wrong answer. Saved
native inputs match the round requests, including that control. The coordinator
correctly scheduled no repair for passing tasks. No redraw or Ministral call ran.
The attempt used 219 output tokens against 512 committed tokens in 12.277 seconds
(loading 1.198, preparation 0.011, generation 4.676, verification 5.155 seconds).
It retains 20 files totaling 43,637 bytes, including one 10,831-byte native log,
fixture snapshots and the pinned Qwen manifest copy. Sources stayed unchanged
during the gate. Live repair qualification remains pending for both models;
overlap, selection/resume and cohort acquisition remain outside this batch.

### Additional integration checks and authorized rerun

All six real Docker tests passed in 45.067 seconds, covering execution limits,
verification/reuse, cancellation propagation and sandbox teardown without model
calls. The [check record](../data/03_generation_checks/batch-07b-hx8hvdj5/validation-checks.json)
links their 11 retained files (127,945 bytes), including nine native logs totaling
127,893 bytes. The ordinary suite also passed: 241 tests, eleven optional skips.

The user authorized one further bounded gate after strengthening the fixture's
first-draft control. Qwen followed the required fail/pass/repair-pass sequence.
Ministral's initial square and distinct-count programs both failed, then both
repairs passed. The unexpected distinct-count failure stops qualification; no
redraw followed. The new gate used seven replies and 652 output tokens under its
separate eight-reply/2,048-token allowance, committing 1,792 tokens. Across the
two 7b gates, nine replies and 871 output tokens were observed.

The gate's strict source check also failed because the user added two comments
during execution. Removing only those two lines reproduces the exact pre-gate
source hash; this review is saved separately and the original check is preserved.
The new gate retains 63 files totaling 193,884 bytes, including four native logs
totaling 38,408 bytes, exact requests/replies/programs, model manifest copies and
the test/check records. The required two-model repair gate remains unqualified.
