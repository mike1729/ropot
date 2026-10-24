# Stage 03 benign generation plan

Build a small experimental workflow that generates and verifies benign programs
from the actor and synthetic generator. Use it to improve prompts and scaffolding,
measure yield and cost, and supply candidates for witness and attack experiments.
Passing the stored checks establishes acceptance under those checks, not universal
correctness.

Stage 03 remains exploratory. Prompts, scaffolding, sampling, extraction, repair
strategies and budgets can evolve throughout the stage. Record what each experiment
actually used; no generation-policy freeze is required here. Resolve implementation
details and acceptance checks when preparing each review batch.

## Starting point

Stage 02's canonical cohort is already published from commit `e42628b`. Consume
`data/02_cohort/manifest.json`.
Preserve its task ownership, construction phases, queue order and profile status.
The [cohort plan](stage-02-plan.md) governs activation; the [pilot](pilot.md)
defines the later monitor experiment.

Start runtime work from existing local model artifacts and established tooling.
Both selected generators have local MLX conversions; compare that route with the
available official Ministral GGUF when implementing the adapters. Record actual
weights, quantization, tokenizer, template and runtime for each experiment.

## Experimental boundaries

- Keep construction and development ownership intact, including development's
  50-ID cap. Use fixtures and APPS train rows for mechanics tests. Keep protected
  correction/replay, calibration and evaluation queues behind their existing gates.
- Before generation on a cohort-owned development/construction task, its same three
  qualified references must pass every stored case in the actual Inspect Linux
  environment. Preserve local evidence separately; changed execution contracts
  need renewed Linux evidence.
- Keep actor and synthetic program generation independent. Prompts and scaffolding
  may differ by source; record those differences when comparing yield and cost.
  Any imposed convention visible in program text must be shared across sources or
  omitted; allow naturally occurring stylistic differences.
  References and nonvisible cases never enter generator prompts or tools.
- Only visible-case diagnostics may inform repairs. Run all visible checks before
  hidden execution, then return only generic hidden outcomes. Use a fresh sandbox
  for each program verification attempt so hidden state cannot reach a later retry.
- Retain every request, response, candidate and verification result. Candidate bytes
  remain immutable; experiments may generate additional candidates and revise which
  one is selected. Downstream artifacts bind exact candidate hashes, and a changed
  selection requires corresponding downstream artifacts for the new candidate.

## Experiment accounting and selection

Give each experiment explicit per-cell budgets (enforced by the batch-6b loop).
The journal enforces cumulative per-source reply, token and generation-wall limits.
Record extensions before dispatch and report cumulative effort across experiments and restarts.
An experiment ID never erases earlier attempts. Use a small durable append-only
journal so per-request accounting survives interruptions; reconcile unfinished
dispatches before resuming. Preserve native Inspect logs as execution evidence.

Compare verified yield, format failures, repair effectiveness, runtime and program
length by source and difficulty. Examine sampling diversity and actual backend
controls during runtime work. Choose extraction and length rules experimentally;
preserve raw replies and exact extracted program bytes. Check complete 2,048-token
monitor-example fit when the actual targets exist in later stages.

Compare recipes on the same tasks within each source at comparable budgets. Report
cumulative acquisition yield across all experiments separately from each recipe's
yield at a stated budget.

Keep task selection in cohort order for later acquisition and training comparisons;
program selection can evolve during experimentation. Repeated variants of one APPS
task remain one task. Keep settings and resource differences visible in reports,
and keep monitor outcomes out of generation, candidate selection and recipe choice.
The consuming stage chooses recipes from construction/development evidence only,
fixes their use for data in each monitor comparison, and reports each export's
recipe mix. Stage 03 exploration can continue independently.

Batch 6a's `generation_journal.Campaign` exclusively creates a directory (parent must
exist), or opens existing state. Register a record; `generate(experiment_id, source, request)` reserves before dispatch.
Use one designated cohort campaign; new IDs retain spending. Absolute limits use
`replies`, `reserved_output_tokens` and `reserved_generation_wall_seconds`;
monotone extensions commit before dispatch, and reservations are never refunded.
Delivered counts include EOS; interrupted observations are lower bounds. Observed
elapsed includes teardown and can exceed deadlines. Gate/verifier time is excluded.
Opening reconciles without generation replay; incomplete parent/request evidence
remains interrupted. One host and one dispatch per campaign are supported; the
inherited flock covers owned workers. History is logically append-only, without
tamper resistance. Acceptance uses `status`, never `finish_reason` alone. Late writes
after parent/cleanup errors can fail closed and require investigation; charges remain.

## Implementation batches

Work on `main` in coherent batches of at most 500 additions plus deletions, including
tests and documentation, and stop for review between batches. Set each batch's
concrete design and checks when starting it; the rows below describe its purpose.

| Batch | Outcome |
| --- | --- |
| 1 | Load the published cohort, pin its reviewed manifest hash in code, and define a minimal experiment record, task activation and candidate identity. Preserve ownership and provenance. |
| 2 | Establish Inspect Linux execution with one fresh sample/sandbox per program and fresh case processes/directories. Check sample failure isolation, output completeness and limits; use native teardown and document shared writable state/background processes within a program. |
| 3 | Add the three-reference gate and deterministic candidate verification. Run visible cases before hidden ones, retain all stored cases and test the public-feedback boundary. Reuse verification results only by task digest, program digest and execution contract, independent of generator source. |
| 4 | Establish the actor's actual local runtime entry point and inspect its messages, thinking behavior, completion status, token counts, seeds and cancellation. Use fixtures/train tasks first. |
| 5 | Establish the synthetic runtime using the same small adapter interface. Resolve MLX versus GGUF from observed compatibility, output quality, cost and provenance. |
| 6 | Implement the experimental generate/verify/repair loop with interruption-safe accounting. Keep generation caching and automatic retries off, and confirm the owned worker group is reaped after a timeout. Require explicit reply, token and wall-time limits before dispatch; define their units, verification-time treatment and enforcement. |
| 7 | Publish versioned selections, every activated cell's status, native evidence and yield/cost reports. Resume without replaying spent requests or silently changing selected candidates. |
| 8 | Run iterative prompt and scaffolding experiments, beginning with the development smoke and construction prefixes. Use the results to choose the next experiment and acquisition scale. |

Use public [Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html) and
[logging/retry facilities](https://inspect.aisi.org.uk/eval-logs.html), with small
adapters for experimental rules. Avoid building a general scheduler or inference server.

### Inspect execution contract

Batch 2's `src/data-pipeline/inspect_runner.py` exposes
`run(programs, log_dir, *, concurrent_inference=False)`, returning Inspect logs.
Each program is `{"program": bytes, "cases": [{"index": int, "input": str}]}`;
case indices must be distinct integers (excluding booleans) within each program,
validated before Docker starts. Runtime provenance is probed before case execution.
`async run_case(program: bytes, stdin: str, case_index: int) -> dict` runs inside
an active sample and returns status, streams, exit code and host elapsed time.
Expected answers stay on the host; batch 3 adds checked cases and the Linux
reference gate below. A mock model runs the fixtures without inference.

- Pin Inspect to `0.3.283` and explicitly set `max_sandboxes=4`. Record that setting,
  Docker VM resources, concurrent inference, image identity, actual Python patch
  version/UTF-8 mode and execution settings in the native `.eval` evidence.
- Use the pinned Python 3.12 image as an unprivileged user with no capabilities or
  network, read-only root, 1 CPU, 512 MiB RAM without extra swap, 32 PIDs and 64 MiB
  scratch. CPU quotas do not reserve physical cores, which MLX inference also uses.
- Execute exact source bytes using `/usr/bin/env -i` with only `PATH=/usr/bin:/bin`,
  `LANG=C`, `LC_ALL=C`, `TMPDIR=<case dir>`, then `/usr/local/bin/python3 -I -B
  -X int_max_str_digits=0 program.py`.
- Request a five-second timeout with retries disabled; Inspect allows five seconds
  of termination grace and uses a 15-second host timeout. Distinguish program and
  host timeouts through the pinned exception chain; host/unknown timeouts and
  `SandboxUnavailableError` abort the sample as infrastructure errors. Other samples
  continue; any sample error marks the final eval errored. Exit 124 is a program timeout.
- Override exec capture explicitly to 10 MiB. Check completed stdout/stderr against
  inclusive 1 MiB/64 KiB acceptance limits; reject replacement characters in numeric
  stdout. Ambiguous provider overflow is `output_limit`; identified streams use
  `stdout_limit`/`stderr_limit`. Timeout/overflow results omit streams and retain
  diagnostics of at most 4 KiB, potentially from a truncated capture's tail. Native
  Inspect events retain provider captures. Post-run stderr checks can classify a
  flooding loop as timeout.
- A fresh directory provides ordinary case separation. Writable state and background
  descendants may persist, sharing the program's CPU/PID limits; late child output
  can be lost after capture ends. Fixed case order does not guarantee deterministic
  execution. Inspect tears down the sandbox after each program attempt.

## Experiment progression

Validate mechanics on fixtures/train tasks before using development or construction
tasks owned by the cohort. Then use the fixed 8-task development smoke and the first 16
construction candidates per phase for exploration. Before a recipe's yield sets
acquisition scale, run it unchanged on candidates 17–32 per phase, kept fresh from
recipe tuning. If those results inform a recipe change, use a new unexposed
construction prefix within the permitted activation for confirmation. This fixes
the recipe only for that confirmation block; experimentation remains open.
Development smoke candidates can join the later panel if they meet its eventual
selection criteria; report their experimental history.

Preserve the remaining development IDs for the pilot's training comparisons and
track feasibility of its requirement for at least 20 fully matched development
tasks. Check the selected model's training/probe compatibility before investing in
the construction probe. Use witness, attack and final-fit results from Stages 04–06
to assess usable-task yield; benign yield alone cannot establish that feasibility.

This plan schedules implementation and experimentation. This documentation change
runs no models, supplied programs or training jobs.

### Verification contract

Batch 3 adds optional canonical `answer` and boolean `visible` fields to runner
cases (all cases in a checked program must supply both). Answers stay in the host
solver. Candidates and the same three qualified references use Stage 02's four
ranked groups, completing every visible case even after visible failures. Hidden
execution starts only if all visible cases passed, then stops at its first failure.
Acceptance requires complete stored-case coverage with exact integer-token matching.

`verification.py` derives fresh and reused verdicts from native sample captures,
checking full request identity, ordered coverage and observed Linux/Python/image.
The source-independent key binds task artifact bytes, program bytes and a contract
covering Inspect, runner/verifier/comparator sources, compose, actual image identity
and Docker platform. Runtime probes, VM resources and concurrency remain evidence.
A sample error has no correctness verdict; other completed samples remain usable.

Public feedback contains visible inputs, answers, statuses and diagnostics, plus
only `passed`, `failed` or `not_run` for hidden checks. Infrastructure errors return
only `verification_error`. Native logs contain hidden inputs/answers/streams and are
private. Reuse is explicit; the experiment journal in batch 6 must account for every
attempt and prevent selective redispatch of completed verdicts for the same key.

### Actor runtime contract

Batch 4 uses installed MLX-LM's text-only Qwen3.5 path through one isolated worker
per request. The parent deadline includes artifact hashing, loading and generation;
cleanup kills the owned process group before reaping. A stdin-EOF watchdog kills
workers if their parent dies. Cancellation retains an outcome then re-raises.
There is no persistent server, retry, generation cache or shared prompt cache.

The private request/protocol/stderr/outcome files retain exact messages, rendered
prompt and IDs, emitted token IDs/text deltas, native counts/finish reason and
runtime/model provenance. Thinking is explicitly disabled in the template, but
actual emitted thinking must still be measured. Hashing covers top-level model
files, the worker and loaded MLX sources, including the installed sampler patch.

The output cap counts delivered tokens including EOS; a stop can reach the cap.
Length requires the cap and a non-EOS final token. MLX computes a lookahead step;
interrupted counts are observed lower bounds, and partial decoded text can lag
behind token IDs. Wall limits bound the owned worker, not unrelated processes.

Outcomes distinguish `stop`, `length`, `timeout`, `worker_error`, `parent_error`
and `cancelled`; parent/cleanup exceptions propagate after retaining the outcome.

### Synthetic runtime contract

Batch 5 adds a boolean `fix_mistral_regex` control (default false), explicit source
profiles, and shared raw/sole-fence extraction preserving exact program bytes.
Synthetic prompts require a leading system message to avoid the template's default
Le Chat instructions. Sources share visible program conventions and verification.

`python -m experiments.compare_synthetic` freezes a two-fixture/four-reply plan per
route before comparing the local MLX conversion with official Q8 GGUF. Prompt IDs,
cache settings, native completion/counts, binary/model hashes and server cleanup
are retained. The routes have different conversions/quantization; this operational
smoke does not establish acquisition yield. Each route has 600 seconds including
startup, four replies capped at 256 tokens and 120 seconds per reply. Verification
runs after inference ends; the 50-task experiment remains behind batch 6's journal.

### Native recovery (batch 6a.2)

`native_recovery.run_native` commits private inputs and an operation intent before
Inspect dispatch; `reconcile_native` only derives existing evidence. Pending intents
block generation and new native execution. Verdicts and resource closure are separate:
all samples must retain the operation's exact Compose label, then current-user CLI
arguments and running/stopped Docker project labels must be absent on the same daemon.
Post-creation infrastructure errors can close without a correctness verdict. Missing
stores, cancelled/incomplete logs and unavailable inventories remain blocked.
Recovery uses one fixed `recovered.eval`, preserving originals. Inspect's external
sample buffer can expire after three days, is tied to the absolute log directory,
and cannot recover a live owner's buffer; reconstructed samples may lose stores.
Reconcile before editing execution sources: changed contracts block underived evidence.
Recorded verdicts need only resource closure; later correctness reuse must re-derive.
This batch changes runner/verifier hashes, invalidating earlier fixture contracts.
Batch 6b must define a recorded operator resolution for blocked native operations
before any cohort generation. No automatic host termination or cleanup is provided.
The low-level native adapter dispatches fresh operations; batch 6b must enforce
source-independent completed-key reuse (including gate samples) before cohort use.

### Loop inputs and manual resolution (batch 6b.1)

`experiment_loop.run_record` builds a canonical activation and an ordered assigned
subset; `validate_run` compares the complete frozen record. Recipes set reply count,
fixed per-request token/wall caps and every seed; cell ceilings are their products.
The policy digest includes loop and generation/extraction code. Messages select only
question/starter and share output conventions. Stateless repair uses the preceding
reply and bounded visible feedback; hidden failures supply only their generic enum.
Candidates publish through fsynced temporary files and exclusive links, never overwrite.
`Campaign.generate(..., attempt=...)` stores strict JSON ownership with the reservation;
the integrated batch-6b.2 loop must validate slots and recheck exact request bytes.

`native_recovery.resolve_native(..., attestation={operator, quiescence_evidence})`
is a HUMAN-only escape hatch, never invoked by unattended code. Stop the original
controller and prove owned creators cannot resume; the campaign flock alone does
not cover Inspect's Docker children. Establish original-daemon quiescence, using a
coordinated manual daemon restart if startup completion cannot otherwise be proved.
Clean only exact operation projects with scoped Inspect cleanup, then retain the
commands/results and positive proof in the attestation. Empty inventories alone
are insufficient. The handler retries derivation first, then rechecks same-daemon
CLI/container absence. Manual closure creates no verdict or refund; newly derivable
verdicts are recorded and the call refuses. Contract restoration is required first.
Make derivation possible before resolving: retain the directory, pinned image and
Inspect version. Recorded or newly derivable verdicts require normal reconciliation.
A changed daemon ID remains blocked in this batch. If proof is missing, keep the
operation blocked and resume the same campaign only after a human resolves it.
