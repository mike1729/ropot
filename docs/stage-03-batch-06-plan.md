# Batch 6: native batched generation and Python extraction

Add `generate_batched` beside the existing serial `generate` in the
[generation module](../src/data-pipeline/stage03/local_generation.py), plus a
reusable `extract_program`. Qualify both Q8 models on fabricated fixtures at
decode sizes 8 and 32 before cohort use. Reuse the existing loading, provenance,
recording and verification helpers; keep the archived benchmark orchestration out.

## Review boundary

Keep Batch 6 as **one implementation and review batch** on `main`. On 2026-10-26,
the user explicitly authorized exceeding the review limit rather than splitting
it. The current repository default is 400 additions plus deletions excluding
tests; report implementation/documentation and test totals separately. The initial
estimate was 700–850 lines including tests. This exception applies only to Batch 6.
Avoid unrelated changes. Stop for review after
implementation and qualification, before Batch 7.

Batch 6 qualification uses fabricated tasks only; it authorizes no cohort
acquisition. Keep historical fixture evidence when implementation contracts change.

## Generation interface and native path

Keep the serial `generate` entry point and add:

```python
generate_batched(model_path, requests, output_dir, *, model_manifest,
                 completion_batch_size, prefill_batch_size, prefill_step_size)
```

Requests remain a finite ordered list supplying `messages`, `seed`, `temperature`,
`top_p` and `max_tokens`. Before `prepare_run` creates a directory, require
positive integer batch/chunk and token limits, excluding booleans, and prefill
concurrency no greater than decode capacity. Preserve refusal of existing output directories.
Both direct generation paths use token limits and record wall time without
enforcing deadlines; legacy `wall_time_seconds` fields have no effect.

Use the existing pinned MLX Python (`mlx-lm==0.32.2`, `mlx==0.32.4`, including the
local sampling patch), model paths and artifact manifests from Batches 4–5. Load
one model per call. Preserve thinking-off rendering, local-only loading and the
Ministral tokenizer regex fix; use explicit system messages in fixtures. Render
all prompts before submission and record preparation time separately from loading.

Use native `BatchGenerator`, passing `stop_tokens=[[t] for t in
tokenizer.eos_token_ids]` explicitly. Pass a fresh sampler for each request through
`insert(..., samplers=...)`; the `batch_generate` wrapper does not forward them.
Decode each request's retained tokens once, with a fresh `tokenizer.detokenizer`
and `finalize()`, when it finishes or during abort cleanup; retry a failed reply
write during cleanup. Pass no reusable
prompt caches. Keep response caching and prefix-state reuse off. Share prompt
rendering between the serial and batched paths so tokenizer policy stays aligned.

Admit requests in input order, maintaining at most `completion_batch_size`
unfinished submissions; use one initial `insert` for the first bounded window,
then refill after completions. Map returned native UIDs to
their result rows in one active-UID map, recording submission time on each
row. Trust the pinned library's UID allocation, insert cardinality, EOS and token
cap guarantees rather than duplicating checks on its private queues. Keep checks
needed to reject duplicate/unknown decode UIDs and invalid completion reasons,
and reconcile counters.
Bounded admission makes the decode target a hard bound on returned generation
events per native call. The native queue can otherwise over-admit by up to
prefill minus one; the archived timings therefore do not measure this adapter.

Require explicit settings, using these measured configurations as starting points:

| Model | Decode | Prefill | Prefill chunk |
| --- | ---: | ---: | ---: |
| Qwen | 8 | 4 | 256 |
| Qwen | 32 | 16 | 256 |
| Ministral | 8 | 2 | 2,048 |
| Ministral | 32 | 8 | 2,048 |

Set the allocator-cache limit to 4 GiB before loading, record the previous value,
and restore it even after loading failure. This does not bound live allocations.
Record peak memory after resetting its high-water mark before loading. Extend
batched runtime source hashes with `mlx_lm.models.cache` and
`mlx_lm.generate_utils`, retaining existing model/tokenizer/sampler identities.
Keep weight-manifest provenance and the caller's responsibility for unchanged
weights; do not rehash weights or add dependencies.

### Statuses and retained evidence

Enforce no wall-time limits. `max_tokens` bounds every reply, so each submission
ends in a native completion; per-request deadlines checked between native calls
could not interrupt a stalled call and would only add partial-state rules. Record
elapsed time from submission to observed completion instead. A Python interruption
is the way to stop a run; no wall-time limits are planned.

Mark requests handed to `insert` as submitted even if that call raises; submission
does not assert observed native progress. Requests outside that admission remain
`not_run`. Retain every returned token in its row or in rejected-call evidence.
Apply these rules to unfinished submissions, preserving previously settled rows:

| Event | Request statuses | Run status |
| --- | --- | --- |
| Native completion | `stop` or `length` | `success` if every request settles |
| Ordinary exception | Unfinished submissions `error` | `error` |
| Python interruption | Unfinished submissions `cancelled` | `cancelled`; propagate after saving |

Undispatched rows remain `not_run`. Stop admission on any technical failure,
without retries.

Preserve ordered result rows and the existing input/runtime/reply/outcome files.
Add submission/native identity and batch settings/activity to those records.
Set row `prompt_tokens=len(prompt_ids)` and `generation_tokens=len(tokens)`;
retain native finish reasons and EOS IDs, excluding terminal EOS only from text.
Set generation-token counts during finalization. Keep token integers in memory
during generation and write each completed reply file before marking its row
`stop` or `length`. Copy the whole native call to scalar events and validate its
decode identities and finish reasons before changing rows. A malformed call
becomes `rejected_events`, with no tokens from that call applied to rows. For a
valid call, append every event's token and retain its native finish reason before
any file I/O, then remove finished UIDs and write each reply independently. Settle
successful writes even if another write fails or is interrupted; then stop the
run and propagate any Python interruption. Only failed writes and still-running
rows become `error` or `cancelled`; abort cleanup retries failed writes and saves
partial replies. No reply
files or per-request detokenizers remain open across native calls. Release native
response objects after copying: they retain full-vocabulary log-probabilities
and, on completion, caches.

Record native statistics and observed decode-event counts. On successful runs,
require native output totals to equal delivered token totals and native prefill
totals to equal `sum(len(prompt_ids) - 1)`. Native prefill excludes each prompt's
last token. On failures, retain both native and observed counts without asserting
equality or treating zero delivered tokens as proof of zero work.

In cleanup, remove every unfinished submitted UID, close the generator and
explicitly synchronize `generator.stream` even if close fails; default-stream
synchronization alone does not wait for native generation. `close()` alone does
not cancel requests. Use a small cleanup-attempt helper that catches
`BaseException`, so removal, close, synchronization, allocator restoration and
saving are all attempted independently. Record primary and
cleanup errors separately; cleanup failure makes the run `error`, while an
original Python interruption still propagates after saving. Loading and stalled
native calls cannot be interrupted from Python. Hard termination may lose active
partial replies and metadata because their tokens are held in memory; completed
reply files have already been written. Stronger process control belongs to Batch 7.
Skip UID-based removal for submissions whose `insert` did not return an identity.
Validation/cache-creation failures occur before queue commitment, but asynchronous
interruption can land after commitment and before UID return. On every abort,
discard the whole generator without another `next()` call; missing UIDs do not
prove that no native state was created. Test insertion failures on both sides of
queue commitment and retain submitted-attempt accounting in either case.

## Python extraction

Add `extract_program(row) -> bytes`, requiring `row["status"] == "stop"`.
Reject every other status even when its text contains a complete program.

- First accept non-whitespace, syntactically valid plain Python. This preserves valid
  Python containing backticks in literals; prose that is valid Python also passes.
- Otherwise accept exactly one complete fence with an exact, case-sensitive
  `python`, `py`, or empty label, allowing surrounding prose. Require exactly two
  triple-backtick occurrences and opening/closing fences on their own unindented
  lines, with optional horizontal whitespace after the label/closing fence.
- Extract from immediately after the opening line terminator to the start of the
  closing fence line, including the code's final newline. Accept LF and CRLF;
  preserve all code whitespace and line endings when encoding UTF-8. Require
  non-whitespace contents. Do not strip, dedent, append a newline or repair text.
- Raise `ValueError` for incomplete status, empty code or malformed/ambiguous
  fences. After plain compilation fails, zero triple-backtick occurrences means
  no fence and preserves `SyntaxError`; a valid single fence containing invalid
  Python also raises `SyntaxError`. On both paths, compile the exact UTF-8 bytes
  returned for execution, including BOMs and coding declarations; never execute.
  Compile plain bytes once; do not compile the same unfenced failure twice.
  Let compiler `MemoryError` and `RecursionError` propagate explicitly; the live
  consumer records these as `compiler_resource` extraction failures.

This is a new byte-preserving extractor informed by the benchmark findings.
The archived extractor normalized whitespace/newlines and had narrower labels;
the 158 recovered draws do not qualify identical behavior here. Indented code can
now fail where stripping previously helped. Docker verification remains required.
Replace the fixture-only `smoke_program` with this function without dropping its
test cases or changing historical evidence.

## Validation and fixture qualification

Extend existing tests with a fake native generator covering UID association and
reordered events, bounded admission/refill, fresh samplers and detokenizers,
settings/validation, stop tokens, native counters, EOS/token exhaustion and partial
Unicode output. Cover premature native exhaustion, invalid completions,
native failure, interruption and cleanup failure
with completed, partial and undispatched rows. Check allocator restoration after
load failure and successful native-call event counts never exceeding decode size.
Assert active-UID removal directly and test independent synchronization when close
fails. Verify completed reply files close promptly, partial initialization fails
without secondary errors, failed reply writes invalidate their rows, and malformed
calls retain all scalar events without applying tokens to rows. Check that valid
calls retain all tokens before a subsequent reply write can fail, and one failed
write does not invalidate successfully saved peers from the same call.
Cover extraction bytes, labels, prose, CRLF, empty/ambiguous blocks, syntax errors,
backticks inside plain Python and every incomplete status. Keep serial regression
coverage. Baseline on 2026-10-26: 160 tests run successfully, eight optional skips.

Add opt-in live checks under a separate `ROPOT_MLX_BATCH_TESTS=1` gate, using the
existing MLX Python and `.venv` Docker verifier without changing either runtime:

1. For each model, run one decode-8 worklist of nine requests and one decode-32
   worklist of 33, using the table's settings. Each has eight or 32 coding requests
   followed by a one-token exhaustion request. Use 256-token coding limits,
   temperature 0.2/top-p 0.95 and one
   greedy control. Keep initial and repair-history template checks for both models.
2. Use fabricated square-plus-offset and distinct-count-plus-offset tasks. Give
   each non-replay row a distinct constant offset and expected answers so swapping
   same-family replies is detectable. Ask for complete programs with named parsing
   and solve functions, allowing enough decode steps to reach the target. Include
   one fabricated repair history, without using cohort tasks or feedback.
3. Put an identical prompt/seed replay pair at positions 0 and 1, in the same
   prefill group for every setting. Keep a subset of requests across sizes while
   changing their order/neighbours. A same-group pair mismatch stops qualification
   for review without assuming its cause. Record cross-layout differences without
   requiring bitwise equality. Separately test real seeded samplers at nonzero
   temperature on flat logits over 256 tokens, using 32 draws and seeds 17/18/19.
   Recreate samplers for isolated/interleaved/reordered schedules: each seed must
   preserve its sequence, and the chosen different seeds must give different ones.
4. Require at least one native call returning exactly eight or 32 generation
   events respectively, plus correct association, exhaustion and refill. Require
   every coding reply to stop naturally, extract and pass its exact fabricated
   suite through `verification.verify_many`, with at most four sandboxes and no
   concurrent inference. Retain extracted bytes and verify after generation.
   The empty-list case includes a blank second line (`"0\n\n"`), allowing both
   token-based and line-based parsing. Fixture formatting and the final short
   request do not guarantee full decode concurrency; missing the observed target
   leaves that qualification inconclusive.
5. For each model, add one three-request interruption probe at decode 2/prefill 1
   using its prefill chunk setting and 256 tokens per request. It requests a very
   long enumeration to keep submissions unfinished and wraps native `next`:
   after both submissions have delivered tokens to the adapter, raise
   `KeyboardInterrupt` before the following native call. Expect two partial
   `cancelled` rows and one `not_run`, with the interruption propagated.
6. After the interruption, check `extract_cache(submitted_uids)` is empty after
   removal, synchronization succeeds, and allocator-limit readback matches the
   original value. Use the setter's returned previous limit for readback. Preserve
   the abort/cleanup evidence and do not run additional inference as a cleanup test.

These worklists supply at most **90 request rows and 22,020 output-token
allowances**, including undispatched probe tails, with no automatic retries.
Sampler-only controls make no model requests. Record loading, preparation,
generation and verification time separately; retain every draw and actual usage
under `data/03_generation_checks`, including exact tasks/requests, replies,
extracted bytes, native outcomes, cleanup checks and verification logs.

Classify runtime/protocol, extraction/syntax, compiler-resource and verification failures separately;
a wrong answer alone does not establish a runtime defect. Any failed or
inconclusive gate stops qualification for review after saving the current run,
without redrawing. Saved-output processing may be recovered without inference.
Unrun checks leave qualification pending. Run the full repository suite and report
both mechanical and end-to-end fixture results; no throughput threshold is implied.

## Implementation validation — 2026-10-26

The simplified adapter's exact final version still requires a fresh live
qualification. The following runs validate earlier versions and remain
supplemental evidence; replay and tiny Metal controls do not replace that gate.
No further inference was run for this revision.

The final whole-call implementation passes the repository suite: 186 tests,
with nine optional skips. Its 26 focused batch/extraction tests cover whole-call
rejection, all-token retention before reply writes, write-before-settlement,
persistence failure/interruption, preservation of saved peers and independent cleanup.

The first simplified version passed a fresh live
[qualification](../data/03_generation_checks/batch-06-r2kwl3mo/qualification.json)
in 97.783 seconds: both models reached decode 8 and 32, all 80 coding replies
passed Docker verification, and both interruption probes passed. The generation
source hash was `1b882b0bb8d6f937f3e2231e5a79da455cc6d38cdfdd50c6ea92610a46d271e8`;
the adapter, live test and verification helper stayed unchanged during that run.
Later event-validation and reply-write changes are not qualified by this result.

The original live [qualification](../data/03_generation_checks/batch-06-68le_a0k/qualification.json)
passed in 99.027 seconds: both models reached decode 8 and 32, all 80 coding
replies passed Docker verification, and sampler independence, same-group replay,
token exhaustion and interruption controls passed. It used 92 submitted requests
and 7,164 output tokens under the earlier 96-row/23,556-token budget, including
two timeout probes. Preserve those probes as historical evidence.

The user then removed wall-time limits and timeout probes. That token-only
controller [replayed the saved tokens](../data/03_generation_checks/batch-06-68le_a0k/final-controller-replay.json)
through real pinned tokenizers for the revised 90-row worklists, including the two
interruptions: every raw reply/status matched and all 80 extracted programs were
byte-identical to the verified artifacts. This loaded no model weights and drew
no new replies; it does not represent a second live generation run.

That cleanup path explicitly synchronized the native stream independently
of `close()`. A [real Metal-stream control](../data/03_generation_checks/batch-06-68le_a0k/stream-cleanup-control-final.json)
passed success, close failure, interruption and cleanup-interruption cases using
tiny array work and zero model inference. Allocator settings and restoration are
recorded; generation-window time is native `wall_time`, with loading in
`runtime.json`, preparation in the outcome and verification in the qualification.
Tests cover optional/ignored wall-time fields, elapsed timing, UID-less insertion
failures on both sides of commitment and cleanup interruption propagation.
That version's repository suite passed 176 tests with nine optional skips; the
separately gated live run and zero-inference checks above supplement that suite.

Repair, experiment persistence/accounting, generation/verification overlap,
selection/reporting and cohort acquisition remain in Batches 7–9. Add no general
scheduler, worker protocol, fallback runtime or cache framework. Short-context
qualification does not qualify long-context Ministral batch 32, prefix reuse,
BF16 yield or trained monitors; retain their existing later gates.

See the [Stage 03 plan](stage-03-plan.md), [Q8 findings](inference-benchmark-results.md)
and [remaining inference plan](inference-benchmark-plan.md) for the surrounding work.
