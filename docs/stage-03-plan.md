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

Give each experiment explicit reply, token and wall-time limits. Record extensions
before further dispatch and report cumulative effort across experiments and restarts.
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

## Implementation batches

Work on `main` in coherent batches of at most 500 additions plus deletions, including
tests and documentation, and stop for review between batches. Set each batch's
concrete design and checks when starting it; the rows below describe its purpose.

| Batch | Outcome |
| --- | --- |
| 1 | Load the published cohort, pin its reviewed manifest hash in code, and define a minimal experiment record, task activation and candidate identity. Preserve ownership and provenance. |
| 2 | Establish Inspect Linux execution using one sample/sandbox per program, with fresh case processes and directories. Check limits, output completeness, cleanup, writable-state exposure and supervisor isolation. |
| 3 | Add the three-reference gate and deterministic candidate verification. Run visible cases before hidden ones, retain all stored cases and test the public-feedback boundary. Reuse verification results only by task digest, program digest and execution contract, independent of generator source. |
| 4 | Establish the actor's actual local runtime entry point and inspect its messages, thinking behavior, completion status, token counts, seeds and cancellation. Use fixtures/train tasks first. |
| 5 | Establish the synthetic runtime using the same small adapter interface. Resolve MLX versus GGUF from observed compatibility, output quality, cost and provenance. |
| 6 | Implement the experimental generate/verify/repair loop with interruption-safe accounting. Keep generation caching and automatic retries off, and confirm server idleness after a timeout. Require explicit reply, token and wall-time limits before dispatch; define their units, verification-time treatment and enforcement. |
| 7 | Publish versioned selections, every activated cell's status, native evidence and yield/cost reports. Resume without replaying spent requests or silently changing selected candidates. |
| 8 | Run iterative prompt and scaffolding experiments, beginning with the development smoke and construction prefixes. Use the results to choose the next experiment and acquisition scale. |

Use public [Inspect sandboxing](https://inspect.aisi.org.uk/sandboxing.html) and
[logging/retry facilities](https://inspect.aisi.org.uk/eval-logs.html), with small
adapters for experimental rules. Avoid building a general scheduler or inference server.

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
