# Remaining inference experiments

The local Q8 systems benchmarks are complete, including the declared resource
stop. Their [consolidated findings and archive](inference-benchmark-results.md)
replace the detailed benchmark plans and reports on main. Continue with the
[Stage 03 workflow](stage-03-plan.md); implement only what its next batch consumes.

## Carry measured settings into the workflow

Use MLX Q8, one generating model at a time, and the measured per-model/context
settings as starting points. [Batch 6](stage-03-batch-06-plan.md) adds a bounded
native batched path and reusable extractor beside the serial entry point, with
short fixture qualification. Archived benchmark timings used different admission
and do not measure this adapter. Keep the benchmark orchestrator archived. [Batch 7](stage-03-batch-07-plan.md)
composes them with verification and repair, starting with sequential phases;
add and measure independent generation/verification overlap after that loop works.
Selections/reporting and iterative experiments follow in Batches 8 and 9.

Set both decode and prefill batch sizes explicitly: the pinned `BatchGenerator`
can raise decode capacity to its prefill setting. Use request-owned seeded
samplers through `BatchGenerator.insert(..., samplers=...)`; the pinned
`batch_generate` wrapper does not forward them. Keep prefill concurrency no
greater than the requested decode target, and record actual activity. The
historical global-reseed replies are a distinct baseline. Recheck association,
neighbour/order changes, early stopping and cancellation on the adopted path.

Use `verification.verify_many` for bounded verification worklists, preserving
per-program isolation, published-suite acceptance and public feedback. Overlap
only independent tasks, and allow one verification evaluation at a time. Keep
raw replies before parsing/scoring; recover processing without redrawing. Freeze
one unambiguous code-block extractor and retain incomplete/failed attempts and
their costs. Keep response caching and prefix-state reuse off initially.

## Measure actual useful yield

Define Stage 04–05 witness acceptance and independent audit before running their
recipes on cohort tasks. Acquire each source's own benign first; supplied-reference
implant comparisons remain fixture-only. Use the development smoke and construction
tuning/confirmation progression already specified by Stage 03, tracking exposure
across stages. Compare at most two recipes with frozen worklists, extractors and
matched reply/token/time budgets. Report verified witnesses, attacks, source-owned
pairs and final-fit usable tasks separately per elapsed hour. Include unsuccessful
and interrupted effort; separate human review work and waits from machine time.

The proposed witness oracle runs all three qualified references on exact proposed
stdin in sandboxes and requires unanimous completed answers. This does not check
input constraints: independent review must still approve the witness. Require
the fixed benign to agree with that answer and the attack to pass every published
case while disagreeing on the approved witness. Freeze oracle, novelty and review
order in the consuming stages. Keep references, oracle traces and hidden-case
diagnostics private; Stage 03 repairs receive only existing public feedback.

Once the full witness/backdoor pipeline, verification and audit work end to end,
run the paired Q8/BF16 actual-yield check required by the
[overview](data-pipeline-overview.md), with matched tasks, prompts, upstream benigns
and budgets. Prefix reuse can receive a separate paired yield comparison then;
JSON fixture speedups do not establish program quality. Compare cloud inference
only if the measured local completion forecast is insufficient. Another local
capacity test needs allocation-aware memory bounds; do not repeat Ministral's
7k/batch-32 configuration or treat between-call checks as a hard memory limit.
