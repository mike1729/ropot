# Stage 03 benign generation plan

Build a small experimental workflow that generates and verifies benign programs
from the actor and synthetic generator. Use it to improve prompts and scaffolding,
measure yield and cost, and supply candidates for witness and attack experiments.


Stage 03 remains exploratory. Prompts, scaffolding, sampling, extraction, repair
strategies and budgets can evolve throughout the stage. 

## Starting point

Consume Stage 02's canonical `data/02_cohort/manifest.json`. The schema-3
publication includes reduced task files under `data/02_cohort/tasks`, while the
original full-suite pool remains intact. Previous publications are archived under
`data/archive`. Reduction preserves assignments and group/phase order.
Preserve its task ownership, construction phases, group order and profile status.
Stage 02 generates and saves every group once in `groups` and `construction_phases`.
Stage 03 loads only saved construction/development IDs from initial/reserve blocks;
protected and deferred tasks are unavailable. Experiments record a worklist of those
IDs without redefining groups or running allocation. Task reads retain digest checks.
Metadata validation belongs to Stage 02. The reader verifies the cohort hash and
restricts membership; task reads check containment within the publication and file digests.
Bindings are shared read-only metadata, without copying or repeated schema checks.
The [cohort plan](stage-02-plan.md) governs activation; the [pilot](pilot.md)
defines the later monitor experiment.

Start runtime work from existing local model artifacts and established tooling.
The completed [Q8 fixture comparison](inference-benchmark-results.md) supports MLX
provisionally for both generators. The direct synthetic adapter passed its
[Batch 5 fixture smoke](stage-03-batch-05-plan.md#implementation-validation).
Record actual weights, quantization, tokenizer, template and runtime for each experiment.


## Implementation batches

Work on `main` in coherent batches of at most 400 additions plus deletions excluding
tests, following [the review rules](AGENTS.md), and stop for review between batches. The user explicitly
approved exceeding that limit for Batch 6 as one review batch; its
[detailed plan](stage-03-batch-06-plan.md) records the exception. Set each batch's
concrete design and checks when starting it; the rows below describe its purpose.

| Batch | Outcome |
| --- | --- |
| 1 | Load the saved construction/development groups, pin the reviewed manifest hash, and define candidate identity. Preserve ownership and provenance; group assignment belongs to Stage 02. |
| 2 | Establish Inspect Docker execution with one fresh sample/sandbox per program and fresh case processes sharing `/work`. Check sample failure isolation, output completeness and limits; use native teardown and document shared writable state/background processes within a program. |
| 3 | Verify candidates against every published case, stopping at the first failure, and test the public-feedback boundary. Trust existing reference qualification. Reuse verification results only by task digest, program digest and execution contract, independent of generator source. |
| 4 | Establish the actor's actual local runtime entry point and inspect its messages, thinking behavior, completion status, token counts, seeds and cancellation. Use fixtures/train tasks first. |
| 5 | Establish the synthetic runtime through the existing direct MLX adapter; see the [Batch 5 plan](stage-03-batch-05-plan.md). Use the measured MLX/GGUF fixture comparison and a current smoke for a provisional MLX baseline. |
| 6 | Add an explicit native MLX batched path beside the serial baseline, plus a byte-preserving Python extractor; see the [Batch 6 plan](stage-03-batch-06-plan.md). Preserve identity, raw/partial replies, token accounting and cancellation. Qualify both models on fabricated fixtures at decode 8 and 32 with bounded admission and request-owned seeded samplers. Reject incomplete replies; keep response caching and prefix-state reuse off. |
| 7a | Establish one initial generate/extract/verify round, experiment records, generation-child control and caller-visible verification cancellation; see the [Batch 7 plan](stage-03-batch-07-plan.md). Require explicit reply/output-token allowances, qualify the real child entry point on a bounded two-model smoke, keep repairs disabled and stop for review. |
| 7b | Add bounded repair rounds and feedback rendering to 7a, then qualify the full loop on fabricated tasks; see the [Batch 7 plan](stage-03-batch-07-plan.md). Require explicit repair limits, retain every attempt and keep technical retries off. Add no generation wall-time limits; verification retains its per-case timeout. Stop for review before overlap or acquisition. |
| 8 | Publish versioned selections, every activated cell's status, native evidence and yield/cost reports. Resume without replaying spent requests or silently changing selected candidates. |
| 9 | Before cohort dispatch, measure tokenized statements and representative repair histories, then choose and qualify input-token limits and handling of oversized tasks for the selected model/batch settings. Run iterative prompt and scaffolding experiments, beginning with the development smoke and construction prefixes. Use the results to choose the next experiment and acquisition scale. |

Batch 6 supplies the reusable prerequisites; the coalesced verifier already exists.
After 7b's full loop passes fixture checks, add and measure overlap of
generation with verification of independent work. Selectively port the necessary
benchmark code; keep the archived benchmark orchestration out of the workflow.

Use public [Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html) and
[logging/retry facilities](https://inspect.aisi.org.uk/eval-logs.html), with small
adapters for experimental rules. Avoid building a general scheduler or inference server.
The [Q8 fixture findings](inference-benchmark-results.md) supply measured starting
settings. The [remaining inference plan](inference-benchmark-plan.md) assigns
batching, extraction and overlap to the consuming workflow; witness/attack yield
work waits for the Stage 04–05 contracts.

### Workload sizing and scaffold choices

The [cohort sizing analysis](inference-cohort-sizing.md) records frozen group
counts, input-length proxies and development timing assumptions. Development has
50 IDs: eight generation-smoke tasks and 42 for training choices. Construction has
758 eligible candidates, 379 per phase; correction/replay has 187, calibration 57
and evaluation 117. These capacities include reserves, not active worklists.

Use MLX Q8 with batch 8 for the eight-task generation smoke and measured batch-32
settings for larger short/roughly-2k queues. Keep Ministral around-7k repair at
batch 8 provisionally. A matched development task supplies four monitor examples,
so 20/42/50 tasks give 80/168/200 requests per checkpoint. Prioritize a finite-panel
monitor comparison of batches 32 and 64, holding output and prefill settings fixed,
before another broad capacity sweep. Qualification of the consuming adapter and
trained checkpoints remains required. Batch 6's [detailed plan](stage-03-batch-06-plan.md#implementation-validation--2026-10-26)
records final short-fixture qualification of the committed adapter at decode 8
and 32 for both models; all 80 coding replies passed verification.

Freeze a small extractor in Batch 6: plain code or one complete unambiguous Python
fence, allowing surrounding prose, followed by syntax and published-suite checks.
Fixture replay recovered 158 additional passing draws without inference; seven
extracted draws still failed execution. Never salvage incomplete replies. Keep
response caching and prefix-state reuse off: stable shared-state fingerprints and
3–7x tiny-JSON speedups did not establish full-program or monitor quality. Revisit
prefix reuse through paired yield checks; BF16 waits for the complete pipeline.

For the full 50-ID development group, one attempted benign, tiny witness and
attack per task/source scales to about 10 warm GPU minutes, plus roughly three
CPU minutes for published-suite verification. The smoke scales to 2–3 GPU minutes.
These are fixture-based effort scenarios, excluding loading, witness oracle and
human audit, retries and final-fit work; they do not promise accepted-data yield.

### Inspect execution contract

Batch 2's `src/data-pipeline/stage03/inspect_runner.py` exposes
`run(programs, log_dir, *, concurrent_inference=False, check_answers=False, max_samples=4)`, returning Inspect logs.
Each program is `{"program": bytes, "cases": [{"index": int, "input": str}]}`.
Supply every case from the reduced Stage 02 task file, in publication order.
The runner executes all supplied cases without filtering or sampling, stopping on
the first execution failure. Case indices remain the original pool indices and
must be distinct integers (excluding booleans), validated before Docker starts.
Expected answers stay on the host; sample metadata records the supplied inputs
and indices. The task-file digest binds the frozen selected suite. Selection
policy/version and original/reduced task hashes belong to the Stage 02 manifest.

Stage 02 publication keeps all visible cases in index order, then fills to at
most 20 with deterministic hidden cases. This is the acceptance suite for every
benign and attack program, with no later full-suite pass. Omitted cases
remain available in the original pool and are not verified by Stage 03. Changing
the publication changes task digests and invalidates verification reuse.

`async prepare_program(program: bytes)` writes `/work/program.py` once per active
sample. `async run_case(stdin: str) -> dict` executes that file
with `/work` as both working directory and `TMPDIR` and returns status, streams, exit code and
host elapsed time. The shared program file is root-owned with mode `0444` in the
root-owned sticky `/work` directory, preventing the unprivileged program from
writing, chmodding, deleting or replacing it.
Expected answers stay on the host; checked mode also takes `answer` and `visible`
per case and stops at the first wrong answer. A mock model runs without inference.

- Pin Inspect to `0.3.283` and explicitly set `max_sandboxes=4`. Record that setting,
  Docker VM resources, concurrent inference, Compose configuration hash and
  execution settings in the native `.eval` evidence. Container configuration and
  Python runtime details are not probed per program.
- Use the pinned Python 3.12 image as an unprivileged user with no capabilities or
  network, read-only root, 1 CPU, 512 MiB RAM without extra swap, 32 PIDs and 64 MiB
  scratch. CPU quotas do not reserve physical cores, which MLX inference also uses.
- Execute exact source bytes using `/usr/bin/env -i` with only `PATH=/usr/bin:/bin`,
  `LANG=C`, `LC_ALL=C`, `TMPDIR=/work`, then `/usr/local/bin/python3 -I -B
  -X int_max_str_digits=0 /work/program.py`.
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
- Cases share `/work` without cleanup between them. Writable state and background
  descendants may persist, sharing the program's CPU/PID limits; late child output
  can be lost after capture ends. Fixed case order does not guarantee deterministic
  execution. Inspect tears down the sandbox after each program attempt.

### Candidate verification (batch 3)

`stage03.verification.verify(task, task_sha256, program, log_dir)` runs one fresh
sandbox; obtain the paired task and digest from `study_tasks.load_task`. Published
metadata is trusted and bindings stay read-only. Existing reference qualification
is sufficient; legacy gate fields in immutable publications are historical only.
The result has `status` (`passed`, `failed`, `error`), `key`, `log`, `sample_id` and
`feedback`. Only `feedback` may enter repair prompts: attempted visible diagnostics
and generic hidden `passed`/`failed`/`not_run`, without hidden indices or timing.
Native infrastructure failures return generic `verification_error` feedback;
preflight exceptions and interruptions propagate. Logs remain private.

`verify_many(candidates, log_dir, *, max_samples=4, concurrent_inference=False)`
coalesces finite `(task, task_sha256, program)` tuples into one evaluation, with
one fresh sandbox per candidate and a hard four-sandbox maximum. Record the
actual 1–4 sample limit and declared concurrent inference in execution identity.
Errored evaluations return generic infrastructure errors for every candidate.

`reuse(log_path, task, task_sha256, program, *, sample_id=None, max_samples=4,
concurrent_inference=False)` explicitly reads successful native evidence. A
multi-sample log needs an explicit sample ID; omission preserves strict single-
sample behavior. Match exact request metadata and current execution/scheduling
contract. Both paths derive verdicts from raw captures with the same comparison
helper. Evidence must cover every case or end at the first failure. The reuse key
binds the reduced task, program bytes, Inspect version, Compose, relevant source
hashes, limits and daemon platform/resources; generator source is absent. Contract
capture uses one `docker info` call per evaluation/reuse, without per-program probes.
Reuse rejects errored/cancelled/incomplete evidence and never dispatches execution.

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
