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
Both selected generators have local MLX conversions; compare that route with the
available official Ministral GGUF when implementing the adapters. Record actual
weights, quantization, tokenizer, template and runtime for each experiment.


## Implementation batches

Work on `main` in coherent batches of at most 500 additions plus deletions, including
tests and documentation, and stop for review between batches. Set each batch's
concrete design and checks when starting it; the rows below describe its purpose.

| Batch | Outcome |
| --- | --- |
| 1 | Load the saved construction/development groups, pin the reviewed manifest hash, and define candidate identity. Preserve ownership and provenance; group assignment belongs to Stage 02. |
| 2 | Establish Inspect Docker execution with one fresh sample/sandbox per program and fresh case processes sharing `/work`. Check sample failure isolation, output completeness and limits; use native teardown and document shared writable state/background processes within a program. |
| 3 | Add the three-reference gate and deterministic candidate verification. Use the 20-case selection below for references and candidates, retain all stored cases and test the public-feedback boundary. Reuse verification results only by task digest, program digest and execution contract, independent of generator source. |
| 4 | Establish the actor's actual local runtime entry point and inspect its messages, thinking behavior, completion status, token counts, seeds and cancellation. Use fixtures/train tasks first. |
| 5 | Establish the synthetic runtime using the same small adapter interface. Resolve MLX versus GGUF from observed compatibility, output quality, cost and provenance. |
| 6 | Implement the experimental generate/verify/repair loop with experiment records and interruption-safe accounting. Keep generation caching and automatic retries off, and confirm server idleness after a timeout. Require explicit reply, token and wall-time limits before dispatch; define their units, verification-time treatment and enforcement. |
| 7 | Publish versioned selections, every activated cell's status, native evidence and yield/cost reports. Resume without replaying spent requests or silently changing selected candidates. |
| 8 | Run iterative prompt and scaffolding experiments, beginning with the development smoke and construction prefixes. Use the results to choose the next experiment and acquisition scale. |

Use public [Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html) and
[logging/retry facilities](https://inspect.aisi.org.uk/eval-logs.html), with small
adapters for experimental rules. Avoid building a general scheduler or inference server.

### Inspect execution contract

Batch 2's `src/data-pipeline/stage03/inspect_runner.py` exposes
`run(programs, log_dir, *, concurrent_inference=False)`, returning Inspect logs.
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
reference, benign and attack program, with no later full-suite pass. Omitted cases
remain available in the original pool and are not verified by Stage 03. Changing
the publication changes task digests and invalidates verification reuse.

`async prepare_program(program: bytes)` writes `/work/program.py` once per active
sample. `async run_case(stdin: str) -> dict` executes that file
with `/work` as both working directory and `TMPDIR` and returns status, streams, exit code and
host elapsed time. The shared program file is root-owned with mode `0444` in the
root-owned sticky `/work` directory, preventing the unprivileged program from
writing, chmodding, deleting or replacing it.
Expected answers stay on the host; comparison and the Linux reference gate belong
to batch 3. A mock model runs the fixtures without inference.

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
