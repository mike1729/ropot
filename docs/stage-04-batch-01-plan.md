# Stage 04, Batch 1: propose witnesses and check execution

Implement the first batch of the [Stage 04 plan](stage-04-plan.md): load selected
benign programs, generate one proposed input per program and check agreement
between three references and that benign. A successful result is
`execution_agrees`, ready for independent audit; it is not an approved witness.
Agreement does not establish input validity or semantic novelty.

This is a planning change. It authorizes no implementation, model calls, program
execution or commits. Implement on `main` after review.

## Review boundaries

Follow the [review rules](AGENTS.md). Split Batch 1 into these two usable parts,
each within 400 additions plus deletions, excluding tests:

| Part | Deliverable | Estimated production/documentation changes |
| --- | --- | ---: |
| 1a | Selection loader, exact-input extraction, duplicate rejection and reference/benign checker, callable on supplied proposals. | 220–320 lines |
| 1b | One bounded local proposal round, private attempt records and fixture integration. | 240–340 lines |

Report actual sizes, tests and generated evidence separately. Stop for review
after each part. Split again before implementation if a part would exceed the
limit. Do not add the retry/audit loop from Batch 2.

## Inputs and provenance

Use `data/03_benigns/v1/manifest.json`, pinned by SHA-256:

```text
33eea4aa0122d503b5e4b3f7bea033a5b21f9dd4adc569e1e140a50314e11ab5
```

It selects 173 programs on 101 tasks: 98 actor and 75 synthetic. Eighty selections
belong to development. Keep the saved selection and cohort ownership; do not
allocate tasks, replace programs or replay Stage 03 verification.

Add a small `stage04/witnesses.py` module. Its `load_worklist(keys, ...)` accepts
an explicit ordered list of `(split, id, source)` selection keys and returns plain
records containing the selection, reduced task, full stored inputs and exact
program bytes. Refuse duplicate or unselected keys. Load only requested tasks.

- Check the selection manifest pin and reuse
  `stage03.study_tasks.load_cohort/load_task` for the pinned cohort, restricted
  membership and reduced-task digest checks. Trust the bindings, candidate
  identities and qualification metadata validated when these publications froze.
- Read each selected program and check its bytes against `program_sha256`.
- Read the corresponding full task from `data/01_test_pool/<binding.path>` with
  `binding.source_sha256` verification. Use every input in its `cases` for
  duplicate checks. The digest is already bound by the pinned
  cohort; do not scan or requalify the original pool.
- Obtain the three sources from the reduced task's
  `solutions[passing_reference_indices[i]]`, in saved order. Trust their existing
  qualification; run them only on the proposed input.

Program/full-task paths come from pinned metadata; their digests bind the bytes
read, including through symlinks. Record the selection digest once per run in
Batch 1b rather than repeating it in every loaded record.

The reduced suite alone is insufficient: 76 selected programs have stored inputs
outside that suite. Read-only inspection verified all 173 program, reduced-task
and original-task bindings while preparing this plan.

## Proposal format and duplicate check

Use exactly one `<witness>...</witness>` block, following the honeypot project's
delimited-input approach without JSON escaping. The opening tag must be followed
by LF or CRLF as framing; stdin starts immediately after that line ending and
ends immediately before the closing tag. Preserve the payload verbatim, including
leading whitespace and whether it has a terminal newline. Encode it directly as
UTF-8 when saving and executing; do not strip or normalize it.

Add `extract_input(row)` for naturally completed replies only. Reject token-limit
completions, missing or repeated delimiters, invalid framing, blank payloads and
payloads that cannot be encoded as UTF-8. Permit whitespace outside the block, without prose
or fences. Keep the raw reply even when extraction fails, and record format
failure separately from duplicate and execution outcomes.

Reject a proposal if `tuple(stdin.split())` matches any stored input for that
task. This ignores whitespace only: numeric spellings and token order remain
significant. Perform this check before execution, keeping the exact input bytes
separate from the comparison key. A duplicate decision must not expose the
matching stored input or its index to the proposer.

## Execution check

Provide `check_many(proposals, log_dir)` in the same module for ordered
`(record, stdin)` pairs. Reject stored-input duplicates first, then flatten the
remaining N proposals into 4N program entries, ordered by proposal and then role:
three references followed by the benign, each with that proposal's single input.
Call `stage03.inspect_runner.run` once in raw mode (`check_answers=False`), with
`max_samples=4` and `concurrent_inference=False`; skip execution if N is zero.
Each entry still gets a fresh sandbox, with at most four active samples. This
removes repeated evaluation setup, preserving the pinned execution limits and
cancellation behavior.

Do not use `verification.verify_many`: it expects an already known answer and
implements Stage 03's published-suite checks. No execution-framework change or
dependency upgrade is needed for this batch.

Map native sample `4*j + k` to nonduplicate proposal j and role k. Trust IDs,
request metadata and capture statuses produced by the fresh runner call; no
saved-log replay is provided. Require a successful evaluation and no sample
errors before deriving agreement verdicts. Missing samples/captures raise.
Return one result per supplied proposal in input order, including duplicates.
An errored evaluation retains completed captures without establishing agreement
verdicts. Native logs retain execution metadata; Batch 1b records source digests.

1. Each reference must finish successfully and emit one canonical integer answer,
   using the existing `stage01.screen.integer_answer` rule.
2. All three answer strings must agree. A failed execution, malformed answer or
   disagreement rejects this proposal without inventing an expected answer.
3. The benign must finish successfully and emit that same answer. Preserve
   benign failures and disagreements separately for later dataset review.
4. Only then return `execution_agrees`, with the unanimous answer and native
   evidence links retained privately. Do not mark it valid, novel or approved.

Record reference execution failure, reference disagreement, benign execution
failure and benign disagreement distinctly. Infrastructure failures stop the run;
they do not establish proposal rejection. Preserve available evidence on errors
and interruption, propagate cancellation after native cleanup, and never retry
automatically. A disagreement does not remove the Stage 03 program.

## One proposal round

Part 1b adds `stage04/proposals.py` with a `run(worklist, output_dir, ...)` entry
point. Keep one source/model per finite run. Actor programs use the existing Qwen
Q8 artifact; synthetic programs use the existing Ministral Q8 artifact and its
tokenizer fix. Require explicit model/manifest/MLX-Python paths, recipe, batching
and reply/output-token allowances, as in Stage 03.

Construct each request from only the statement, that selection's own program and
the fixed witness instruction. Ask for one valid input, meaningfully different
from the statement examples, in the delimited format above, without an answer or
code. No stored cases, reference sources, reference answers, other source's program or
verification/audit details enter the request. Statement examples are naturally
visible as part of the statement.

Use `stage03.local_generation`'s existing batched child entry point unchanged;
reuse `benign_loop.run_child` and `atomic_save` for process control and records.
Keep request construction, extraction and agreement decisions witness-specific.
Do not reuse the benign loop or its feedback renderer, which expose public
execution details unnecessary for witness proposals.

- Submit exactly one request per entry, with seed `base_seed + position`. Require
  allowances covering every request and reserve their full output-token limits
  before dispatch. Retain observed usage and uncertain submissions separately;
  do not refund reservations or redraw malformed, duplicate or failed proposals.
- Exit generation before running verification. Pass all extracted inputs to
  `check_many` for one evaluation of the round's nonduplicate proposals. This
  batch establishes the sequential path without generation/execution overlap.
- Refuse an existing output directory. Save immutable inputs and an atomically
  replaced outcome record under `data/04_witness_experiments/`; fixture runs use
  `data/04_generation_checks/` and remain explicitly marked as fixtures.
- Save selection/cohort/model/source digests, ordered worklist and recipe,
  exact requests/raw replies, extracted input files and their digests, native
  execution logs, private decisions and generation/execution elapsed times.
  Preserve attempted, unstarted, uncertain and interrupted entries and native
  usage availability. Link the original program identity; do not mint a new
  benign candidate or overwrite existing data.

## Validation and acceptance

Use fabricated tasks and mocked generation/native results for deterministic
checks. Each negative fixture should otherwise satisfy the preceding checks.

- **Loading:** selected actor/synthetic identities and ordering work; changed
  manifest/program/task bytes and foreign selections fail.
  A full-task input omitted from the reduced suite is still a duplicate.
- **Extraction:** tabs, leading/trailing whitespace, CRLF and a missing terminal
  newline survive exactly after removing the defined opening framing. Missing or
  repeated delimiters and token-limited replies fail. Whitespace variants
  duplicate; changed tokens remain distinct.
- **Agreement:** cover unanimous success, reference disagreement, each program
  execution failure, malformed integer output and benign disagreement. Shuffled
  sample ordering preserves association; missing samples/captures stop the check.
  Multiple proposals use one native call with 4N entries and correctly mapped
  results; an all-duplicate round makes no native call.
- **Proposal isolation:** inspect the exact messages with sentinel secrets in
  stored inputs, references and private verdicts. Include both sources and show
  that each receives only its own program. Check stable seeds and allowances,
  no second draw, preserved malformed replies and honest failure/cancellation
  accounting through the reused child control.

After implementation review, use a bounded opt-in integration smoke on two
fabricated tasks: three equivalent integer references and a benign per task.
First check supplied new inputs through real Docker, including a deliberately
wrong benign. Then generate one proposal per task with each source: four replies
total, no retries. Retain Stage 03's model-specific batching: decode 8 for both,
Qwen prefill 4/step 256 and Ministral prefill 2/step 2,048. Fix and record prompts,
sampling settings, seeds and output-token allowances before the smoke runs.

Retain every outcome and stop on technical failure without replacement. The
smoke reports format failures, token-limit completions, duplicates and execution
outcomes separately by source, and a successful path for each source; failure
to obtain one leaves live qualification pending. An unrun smoke must remain
reported as unrun. This plan launches none of these checks.

Batch 1 is complete when both reviewed parts expose the supplied-proposal and
generated-proposal paths with the required records and qualified fixture checks.
Auditor integration/qualification, semantic novelty and validity decisions,
human sampling policy, development-task experiments, full-pool acquisition and
approved-witness publication remain in Batches 2–3. Do not use provisional
agreements as Stage 05 inputs.
