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

### Integrated loop (batch 6b.2)

`experiment_loop.run(campaign, loaded, record)` reconciles native work, pins one
`execution_context`, then validates every retained request in a pass with dispatch
turned off before doing new work. An execution-contract change needs a new experiment.
Exact task/source/reply slots consume their budgets on every reservation. Only format
or scientific verification failures permit a planned repair; cleaned cancellation
ends its cell without retry. A retained cleanup error stops the whole campaign.
Expected global-budget refusal returns a blocked cell; it never resets spending.
`attempt_result` binds each reservation to candidate SHA and native operation/sample.
Completed native keys are re-derived read-only, including all three samples of gates;
unknown operator-closed work never redispatches. A reply identical to a qualified
reference reuses that gate sample; retain this link as evidence of reference reproduction.
`cell_states(events, record)` needs neither Docker nor model directories and projects
all activated tasks by both sources: unassigned/unrequested, unstarted/pending,
verified/exhausted/blocked with distinct budget, generation, gate and infrastructure reasons.
Host parser-stack/recursion errors are format failures. Host compilation retains its
4,300-digit integer-literal limit; Linux execution uses the documented unlimited setting.
The later report imports this reducer; edits to loop/generation code change the policy
hash and require new experiment records. Existing historical projection remains available.

### Evidence snapshots (batch 7a)

`study_report.snapshot(campaign, *, selections=None)` derives a private report and file
map under the campaign lease. Optional reservation sequence IDs override only their
own selected cells; other cells retain their earliest proven pass. Every closed
verdict is re-derived, including unselected passes and failed gates. This requires
the current Docker daemon/image and unchanged execution contract, without starting
containers. Export before changing that contract. Snapshot errors can close Campaign.
Per-run grids share the loop reducer; cumulative counts group by source/difficulty.
Recipe comparisons require equal ordered tasks, budgets and execution context.
Historical policies retain bound response/candidate/native evidence; only current
policy replies are re-extracted with today's rules. Each run exposes its policy hash.
Observed tokens and unknown/lower-bound work remain distinct from reserved caps.

Batch 8 will anchor `panel` and `mechanics_baseline` before any generation/native
intent, each with `payload` and SHA-256 of `encoded(payload)`. Panel payload is
`{activation, tasks}` with exactly 50 distinct bindings in activation order.
Baseline payload binds per-source and combined `replies`, `observed_generation_tokens`
and `reserved_output_tokens`, an `audit` copy `{path, sha256, bytes}` and the primary
`inventory_sha256`; the audit path is campaign-relative. Optional `elapsed` and
`limitations` preserve timing caveats. Missing anchors are null.
The panel's full activation enters cumulative denominators; its joint target stays
15/50. Baseline costs are separate from cohort reservations. Snapshots append only
missing generation completions from retained bytes before capturing their fixed prefix.

### Versioned publication (batch 7b)

`study_report.publish(campaign, version, *, selections=None)` publishes under
`campaign.root/exports/<version>` while retaining the lease. It copies and verifies
the snapshot, then atomically renames a sibling staging directory. Existing versions
are read-only: exact byte/tree validation returns the same manifest identity,
including when different override arguments select identical rows. Changed prefixes,
selections or pipeline source hashes require a new version; prior bytes stay intact.
Unexpected files, including Finder's `.DS_Store`, make validation refuse the version.
The manifest records every pipeline source hash and binds exact report/event bytes.
Produce authoritative exports in a fresh process from a clean reviewed commit;
source hashes describe the files on disk, not previously imported code.
Copies and directories are fsync-ordered, including retry parent-directory syncs;
this provides no macOS power-loss survival guarantee. A torn version fails validation
and requires a new name. SIGKILL staging orphans are ignored. Atomic visibility
assumes one publisher and stable inputs; unrelated manual directory creation can
race POSIX rename. Retries require the same execution contract and reachable Docker,
and perform no generation, native execution, recovery or operator resolution.

### Actual-checkpoint compatibility (batch 8a)

`experiments.bf16_compatibility.run(directory)` records one isolated, bounded
Transformers/PEFT diagnostic on the pinned official Qwen3.5-9B BF16 text weights.
Run `.venv/bin/python -m experiments.bf16_compatibility` outside the sandbox,
where MPS is available in the approved launch context.
It verifies the input manifest before heavy imports, checks a frozen invented
128–512-token completion-only fixture, then updates rank-32/alpha-64 adapters on
all 64 attention targets once in RAM. Final normalized text features must agree
with the text-model path and reconstruct ordinary logits through the output head.
The model worker has 1,800 seconds plus a separately recorded cleanup grace.
File-log size caps are soft: polling detects overflow and retains output up to kill.
Receipts retain failures, runtime/input/source identity, phase memory snapshots
(not Metal peaks) and lifecycle evidence. This costs zero generator requests.
Passing establishes the observed short all-layer backward/update and feature path;
it does not establish a nonzero gradient path through initially zero-effect LoRA A,
2,048-token fit, effective batch eight, convergence or learned-probe quality.
The paired 8-bit/16-bit inference comparison is a separate experiment.
The direct path uses LoRA-injected modules; this does not measure the adapter's effect.

The first actual run passed on a 361-token fixture with 12 supervised tokens:
one update, finite loss 2.33144, exact repeated/direct/head feature comparisons,
30.82 seconds total and confirmed worker-group absence. The largest recorded
driver-memory snapshot was 21.16 GB, not a peak or a 2,048-token fit measurement.
Evidence: `data/03_training_checks/bf16-compatibility-f22oaa6r` and
`data/03_training_checks/bf16-compatibility-8ydukfzx`. The final-source run took
19.94 seconds; both retained attempts total 50.76 seconds and zero generator requests.

### Precision input validation (batch 8b.1)

Batch 8b.1 validated both pinned official BF16 input directories in one owned
worker without converting or loading models. Its requests mark the input-only
stage, and successful completion explicitly says
`artifacts_ready: false`. Retained manifests, source snapshots and receipts bind
the exact attempt. The worker has 1,800 seconds, separate 60-second cleanup grace,
soft log caps and a 100-GiB free-space floor. Ordinary fixtures exercise the
stdlib tokenizer/config/header/coverage validators reserved for conversion.
Input validation costs zero generator requests and establishes no model
compatibility, yield or speed advantage. Cohort use also requires reviewed
immutable precision profiles.

Both input-only attempts passed: `precision-n2rsy179` and final-source
`precision-6d5rvwxx` under `data/03_model_artifacts`, totaling
25.30 seconds and zero generator requests. Final-source evidence/audit:
`data/03_reviews/batch-08b1-final-actual-audit.json`; all 32 files matched.

### Paired precision artifacts (batch 8b.2)

Run `.venv/bin/python -m experiments.prepare_precision_artifacts` for the full
controlled preparation. Each source's official BF16 checkpoint is converted once
to MLX-native BF16 text weights, then quantized from that artifact with affine
8-bit/group-64. The owned 1,800-second worker rechecks inputs, pins runtime sources, strictly loads
each output and checks module/dtype coverage, exact nonquantized identity between
the paired BF16 and 8-bit loads, and paired prompt/BOS/EOS identity. It evaluates
one uncached full-sequence forward per artifact; this is distinct from generation's cached prefill and samples no reply.
Four hash-bound records enter the exclusive receipt only after successful worker
cleanup and validation; `artifacts_ready` requires a passed receipt. Failures retain
partial evidence. Phase wall times, native MLX allocator peaks, output hashes and
forward-logit differences are diagnostics, with zero generator requests. They do
not establish yield, speed advantage or 2,048-token fit. Immutable generation
profiles and the paired task experiment remain separate reviewed batches.

The actual four-artifact run passed in 74.81 seconds with 122 forced-forward tokens
and zero generator requests: `data/03_model_artifacts/precision-cj08t4az`. The four
outputs contain 46 files totaling 53,536,317,502 bytes; the receipt binds their
records and confirmed cleanup. Audit: `data/03_reviews/batch-08b2-actual-audit.json`.

The BF16 arm uses the native MLX conversion. Qwen's public sanitizer applies its
BF16 norm offset, and conversion casts 3,072 source normalization elements from
FP32 to BF16 while preserving 768 A_log elements in FP32. Both arms share these
transformations; paired identity checks establish no repeated offset after conversion.

The final-source run `precision-u_uycdio` also passed in 70.84 seconds, with
all four model manifests identical to the first attempt. Both attempts are retained:
145.65 seconds total, 244 forced-forward tokens and zero generator requests.
Use its passed receipt for profiles; audit: `data/03_reviews/batch-08b2-final-actual-audit.json`.

### Frozen precision metadata (batch 8c.1)

Under an open `Campaign` lease, `precision_profiles.freeze(campaign, receipt_path,
expected_receipt_sha256)` takes an absolute canonical receipt path and freezes seven metadata files under
`precision/<receipt-SHA>`. Equal retry validates that copy even after the original
metadata is removed. `resolve(campaign, reference, source, bits)` returns the
original model path, exact file map and common runtime fingerprint without reading
weights or requiring the original model/runtime files to exist.

`check_runtime(expected_runtime)` loads by path under `-I -B` using only stdlib. It
checks interpreter, flags, eight distributions, four launch-environment controls
and the consistent source union before any heavy import. Later-import observations
remain distinguished from preparation's pre-operation stability evidence. Fixed
thinking/regex controls match the receipt-bound preparation source. Parent metadata
validation lazily depends on `experiments.prepare_precision_artifacts`; the guard
does not. Input hash maps are checked for agreement, not reconstructed from decoded
objects. Publication assumes one stable publisher and fsync ordering, without a
power-loss guarantee. This batch adds no dispatch integration or cohort outcomes;
use new export versions after source changes.

### Direct controlled requests (batch 8c.2a)

`local_generation.request_for(..., profile=descriptor)` adds a single
`precision_profile` wrapper containing the historical descriptor and current guard
source hash. Legacy requests keep their existing bytes and protocol fields. The
worker executes the same helper bytes it checked, verifies runtime and the exact
model inventory, then positively checks GPU and operative source origins before
loading. Post-load tokenizer/model checks precede task template and sampler work.
Known source hashes must agree; additional noncritical imports remain provenance.

Controlled protocol evidence includes `controls_checked` before load and matching
ready provenance. Retained `profile_guarded` distinguishes a proven pre-load prefix
from unproven controls; it does not establish successful generation or diagnose OOM.
Typed `profile_error` is trusted only with its valid terminal sequence, matching PID
and a known nonzero exit. Parsing old evidence uses serialized controls, without
reading current helper/model/runtime files. Profile control failures remain
infrastructure evidence, never wrong answers or automatic retries. Batch 8c.2a limited
these calls to direct requests; registered execution is described below.

### Registered controlled runs (batch 8c.2b)

`experiment_loop.run_record(..., profiles={source: {reference, descriptor}})` creates
schema 2; omission retains schema 1. All requested profiles must share a receipt,
runtime and precision. The loop resolves both sources before fresh native work,
and the journal checks the whole canonical request before reserving shared budgets.
Policy hashes include the loop, generator, profile helper, worker and journal;
earlier IDs remain reportable but need new IDs for execution after policy edits.

Controlled outcomes durably distinguish observed profile failure, unproven controls
and protocol violations. Hashing/loading need deadline headroom: a timeout before
`controls_checked` stops that experiment as unproven. A valid guarded prefix permits
only `Missing done` or `Incomplete protocol tail` as execution interruptions.
`ProfileControlError` follows persisted evidence; reopening stops the same record,
while a new ID can proceed on the same cumulative budget. Each new failed dispatch
remains charged, with no refund or automatic retry. Historical reports use stored
request/helper identities and copied metadata, exporting all seven files even for
zero-reply runs. Source rows/selections include compact `precision_profile` identity;
global selections and panel yield remain a mixed acquisition union across recipes
and precisions. This batch runs no cohort tasks or paired experiment.

`study_tasks.activate(..., construction_per_phase=32)` and the loop factory opt into
the cohort's recorded feasibility prefix; omission preserves the original probe
prefix and activation shape. The nested marker canonically binds the choice and
requires a construction group. Construction uses metadata only; execution loads
assigned tasks individually. The loop source change requires new execution IDs;
historical reports retain recorded activations without current-policy rebuilding.

Use `python -m src.data-pipeline.paired_collection setup --help` to freeze a paired
trial; [its protocol](stage-03-paired-protocol.md) binds methods before collection.

For tuning, `paired_collection tune --campaign PATH --plan-id ID --through-tasks N`
dispatches paired members; N is required on the CLI and ranges from 1 to 40 tasks.
It walks the same first `2*N` records on every resume, without re-dispatching spent
requests or reaching fresh tasks. Use whole-task checkpoints for quiet tests;
do not run heavy work while requests are measured. A later setup supersedes old plans.
An interruption before guard proof permanently stops that family as unproven.
A new reviewed plan reruns exposed tuning tasks under new IDs with cumulative
charges; repeated tuning across families cannot be called fresh confirmation.
New guarded errors/interruptions pause collection for inspection; explicit resume
keeps those cells terminal and charged. Fixed-budget timeouts may continue.
Retained verification/gate infrastructure errors also stop collection; do not skip
the blocked member. A new plan may still reuse the same failed native evidence.
Budget refusals need a recorded funding correction.
The historical protocol export adds no live producer/model/runtime reads; existing
snapshot proof checks retain their code, metadata and Docker requirements.
Fresh continuation uses a separately reviewed module, described below.

Paired snapshots add `paired_trials` validity/status from all proven own attempts,
independently of global selections. The reducer checks the complete frozen plan,
every owned request and historical confirmation evidence; it reads cohort and
test-pool manifests lazily, without task payloads or live model/runtime guards.
Malformed bindings reject; observed control/procedure failures remain publishable
with invalid reasons and existing costs. Usable completion requires durable guard,
generation and native proof at the relevant prefix, never later reconciliation.
Freshness is campaign-local through the decision; cohort restrictions cover prior
cross-campaign use.

Paired `metrics` summarize primary50, tuning40, fresh32, extra22 and full72;
their overlapping denominators never shrink or add together. Verified first/final
counts use each trial's own replies and same-arm actor AND synthetic success.
Primary target flags require valid comparison and all200 source cells terminal;
subset completion alone describes coverage. Unknown/pending tasks stay visible.
Costs include every owned attempt; shared native and cumulative costs stay global.
First-request cold wall includes guarded load failures, with missing-ready coverage;
policy wall includes independent repairs only when terminal and fully measured.
Median16/8 wall ratios require positive8 denominators and a valid comparison.
Native rates use complete terminal evidence, independently of program success;
their timer excludes first-token time while its token count includes it/EOS.
Neither rate nor parent wall measures GPU-only time or time to first token.

After all40 tuning tasks are terminal, explicitly review symmetric pooled tuning
results and retain the unchanged decision rationale under `data/03_reviews`.
`paired_continuation confirm --campaign PATH --plan-id ID` records that decision;
`paired_continuation fresh --campaign PATH --plan-id ID --through-tasks N`
then traverses exactly the first N of32 fresh tasks in the frozen arm order.
Both commands require an existing campaign, current retained family, usable
tuning evidence and no prior fresh exposure. Fresh never confirms automatically.
Explicit resume retains every charge and prior terminal failure; a new guarded
execution error pauses before another member, while fixed-wall timeouts continue.
Run as `python -m src.data-pipeline.paired_continuation`; do not edit frozen
producers or this continuation module during fresh collection.

After Darwin EPERM, teardown waits up to five seconds for the child to exit, then
observes its group for up to five seconds after reaping. Existing final waits may
add five seconds on failure; only ESRCH proves group absence. This grace affects local
case and diagnostic workers; local-case elapsed excludes cleanup, generation
parent elapsed includes it. No nonzero signal is sent after the successful reap.
For a completed, positively guarded request whose only error was cleanup,
`Campaign.clear_generation_resources(seq)` can record resource clearance after
BOTH its retained PID and equal PGID independently return ESRCH to signal zero.
Unknown/live resources remain blocked. Exact retry reuses that bound marker
without probing a possibly reused PID. The pure historical checker releases only
the global setup/run resource barrier; the original worker_error, evidence,
candidate eligibility and cumulative charges stay unchanged. A producer fix
requires a new plan/member IDs in the same campaign; do not rewrite old history.

Throughput discovery precedes further cohort acquisition. `experiments.batch_throughput`
freezes a finite job of invented fixtures and declared windows, with strict raw-token
UID/EOS accounting and an inherited exclusive GPU lease. Its parent retains partial
output, resource samples, timeout/cancellation and cleanup evidence; failures do not
refund diagnostic costs. Run files and profile metadata live under private `data/`.
A separate resident worker supplies numerical batching qualification and timed grids;
its throughput method remains separate from the archived per-request cold run.
Use `python -m experiments.mlx_batch_throughput --run` with `--profiles`, `--source`,
`--bits`, `--configs` (JSON), `--output`, `--lock` and optional `--wall`.
Each owned job qualifies unequal-prompt greedy batching, EOS/cap handling, fresh
insertion and seeded sampler independence, then warms once and measures two repeats.
Aggregate throughput uses raw tokens divided by synchronized whole-window wall;
native MLX rates and process/resource observations retain their separate time bases.
Optional `--panel-size 32` repeats the eight prompt contents with distinct row seeds.
It accepts one batch 8/16/32 configuration per job, qualifies that full capacity,
then measures two windows of 32 pending replies. Compare these cells together;
retain their timings separately from the original eight-row workload.

`python -m experiments.cpu_throughput --output PATH --lock PATH` runs the fixed
native verifier grid after GPU work stops: coalesced concurrency 1/2/4 and eight
separate eval calls (nominal limit 4, effective concurrency 1), two repeats each.
The first condition must pass its native oracle and resource-closure check before
continuation. Intents retain attempted costs; partial case counts are lower bounds.
Compare native eval wall separately from total diagnostic ownership/proof overhead.
The CPU worker freezes the discovered Docker CLI directory into its restricted PATH.
Optional CPU `--coalesced-four` runs only two eight-program conditions at concurrency
four, retaining the same native oracle. GPU and CPU parent CLIs accept `--lease-fd`
for an explicitly coordinated experiment: both duplicate the same existing flock
description and retain it through worker cleanup. In this mode stdin EOF interrupts
the owning parent through its normal receipt and cleanup path. GPU window starts
and CPU eval starts retain host monotonic anchors for measuring actual overlap.
Optional GPU `--cache-gib 0|4|8` binds a post-load MLX allocator free-cache limit to
the job and retains its setter readback in `cache-control.json`. Omission preserves
the runtime default. This changes unused allocator memory, independently of KV state.
`python -m experiments.overlap_throughput --profiles PATH --winner JOB --output NEW
--lock PATH` runs serial GPU/CPU, overlap, overlap and reverse-serial phases under
one lease. It rebuilds the winner with current source hashes, binds identical GPU
job bytes across phases and reports full makespan plus actual window/eval overlap.
The verifier uses independent fixed programs; this benchmark does not measure yield.

`python -m experiments.model_parallel --help` compares matched serial and parallel
Q8 actor/synthetic jobs, twice each, under one owned lease. A fresh memory screen
selects the predeclared B32 pair or B8 pair before dispatch, otherwise skips both.
The screen combines prior allocator observations with a 12 GiB margin; it is a
heuristic, not a physical-memory bound. Useful rate counts only the 128 measured
replies per phase; numerical qualification and warmup remain charged separately.

The GPU CLI's `--long-qualification` mode uses one declared configuration and a
240-second whole-job work deadline. Its separate schema 2 preserves the original
short diagnostic format and runs only numerical/resource qualification windows.
The greedy oracle uses the proposed prefill controls; the existing short live-row
reinsertion check remains separate from the full-cap stress. Every row of that
stress uses a long invented repair context and reaches 2,048 raw output tokens,
alternating EOS-at-cap and length termination. Forced-token rates describe resource
stress rather than sampled program quality. The qualified prompt ceiling is the
minimum observed context length across all stress rows, never an assumed 4,096.
Select a common capacity per source by testing both precisions at 8, then 4 or 2
only after a proven resource/deadline failure; retain every attempt and cleanup.
Run actual qualification after all captured adapter/helper sources are final.

The throughput helper's qualification reader validates a pinned response SHA,
reparses the copied raw schema 2 or 4 protocol, and derives the proven configuration
and minimum stress-context bound. Historical validation needs only retained bytes;
new launches separately require the entire qualified source map to match current
files. Qualification copies preserve exact job, raw output, parent receipt and
optional cache-control bytes. These helpers authorize no cohort dispatch by themselves.

Schema 3 program waves use `build_wave` and the existing owned parent with copied
qualification bytes. Requests retain exact canonical bytes, row IDs and seeds; a
partial final wave keeps its qualified batch/prefill controls. Every rendered row
must fit the declared qualified context ceiling before any row is admitted. Waves
run one generation window, with no qualification or warmup replies inside it.
`project_wave` checks the copied authority, raw protocol, shared parent receipt
and the decoded-file hash bound by the worker's terminal event. It returns one
process/wall record and every planned row, including unadmitted rows and partial
token lower bounds. Rows become eligible only after the entire session completes
with proven cleanup. Campaign reservation and cohort orchestration remain separate.

Campaign program-wave admission reserves one session and all ordered row requests
in one transaction. `reserve_batch` checks each frozen method/profile, task slot
and recipe before charging every row's reply, token and wall caps. Pending sessions
remain inspectable on reopen and block scalar generation, batch admission and
native dispatch. This admission prerequisite launches no generation; session
execution and atomic settlement belong to the following batch.

`generate_batch` starts one owned session after its durable reservation. Reopening
reconciles each session once and commits its shared evidence plus every row outcome
atomically, without rerunning generation. Every successful row requires the full
session inventory, copied qualification and decoded proof. Unknown cleanup blocks
all further dispatch; a proven closed failure remains ineligible. Row outcomes hold
text, raw tokens and latency with shared-session references; the session owns wall
and process evidence once. Shared export and the cohort driver remain separate.

Snapshots export shared generation evidence once per session. Batch rows retain
their own delivered tokens, outcomes and native candidate proofs; per-record wall
is explicitly unallocated with session references. The global source ledger counts
each proven owned session wall once and each unknown session once. Cleanup is
already included in that wall. Session costs retain source, precision and profile
identity. Records with no replies still export their owned qualification; scalar
reports preserve their existing fields and values.

Schema 3 loop records freeze each source's complete qualified generation method.
Scalar traversal rejects these records before native work. prepare_cell replays
one selected cell and returns its next raw request without spending a reply;
settle_cell verifies an existing eligible reply and records its result without
creating a new slot. Both preserve the prior reference gate and exact own history.
Repairs receive only the existing visible feedback projection. Failed batch peers
cannot supply candidates, and shared session status determines failure categories.
Candidate files remain real row-local artifacts. The separately frozen collector
owns wave ordering and the tuning/fresh-data barrier.

Resident collection setup uses paired_wave_collection with qualified schema 3
records and a frozen finite opportunity schedule. The wave protocol amendment
binds the retained base protocol and replaces cold ordering and wall accounting.
Setup copies all four qualifications, registers the full metadata matrix, and
freezes funding before extension. Actual acquisition waits for reviewed report,
executor and confirmation consumers; no setup command alone starts model work.

The Ministral sampler diagnostic reuses the original owned schema 2 worker job
through a byte-pinned helper. It records native normalized sampling scores and
sampling dtype for the shared prefix, with lossless float32 observation vectors
at token index 10. Original requests, samplers, stress checks and verdict remain.
Its baseline binds the retained failing job and rendered prompts. An expected
qualification failure can yield diagnostic evidence with proven cleanup; it
provides no new cohort admission or inference speed measurement.

Wave snapshots retain the base protocol and amendment and reparse each shared
session's already bound ready evidence once. The first report consumer checks
frozen metadata, opportunities, all row requests and their own repair histories.
Its wave_trials output is explicitly provisional with structure-only validity;
quality, completion, session cost and confirmation claims belong to the next slice.

Prospective peer-isolation qualification uses schema 4 with two invented
targets, equal-length but
different peer contexts, controlled peer decoding, and fresh-cache replays after
EOS and live insertion. Six target windows retain native sampling-score
fingerprints, dtype, shape and returned tokens in the existing stdout protocol.
Observation preserves sampler input/result objects and adds synchronization;
these qualification timings cannot measure inference speed.
The parent independently derives isolation, replay, RNG, EOS and stress checks.
Cross-shape greedy and reinsertion equality remain reported diagnostics for this
new contract; historical qualification verdicts retain their original meaning.
The fixed long contexts must render within 7,168–8,192 tokens, with equal peer
lengths and different token contents. A tokenizer preflight precedes inference;
the tested minimum remains the actual future context ceiling.
The native producer uses `--isolation-qualification` with one declared configuration.
It observes only row zero in the six target windows and reuses one safe forced
token throughout the job. The original sampler input and returned object remain
unchanged, with at most seventeen native calls per sixteen-token target.
Complete schema 4 evidence yields admission schema 2 with the fixed contract
`peer-isolation-v2`; historical schema 2 jobs retain their exact schema 1 admission.
Program waves accept that authority up to its actual qualified minimum, at most
8,192 tokens. New collection setup schema 2 binds this contract and requires all
four admission schema 2 qualifications. Historical setup schema 1 keeps its
original grammar and four legacy authorities for offline replay.
