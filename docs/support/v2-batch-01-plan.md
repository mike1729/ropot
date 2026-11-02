# V2 batch 1 implementation plan

This is the completed campaign's benign-acquisition plan. See the
[qualification record](../frozen-corpus-qualification.md) for the final corpus,
accepted user decisions and next-stage limitations.

Acquire verified own-source benigns for eight development tasks. Use the
[existing components](v2-implementation-plan.md#reuse-existing-components),
[pipeline contract](../data-pipeline-overview.md) and [review rules](../AGENTS.md).
Freeze the following proposed contract before acquisition; this edit starts no runs.

## Implementation

Keep benign acquisition in [`benign_acquisition.py`](../../src/data-pipeline/stage03/benign_acquisition.py):
`prepare_contract(output_dir, exposure_file, runtimes)` freezes reviewed inputs;
`acquire(prepared_dir, output_dir)` uses `generation_loop.run()` for generation and
repairs, then audits selected programs and reports coverage/cost. The module
documents the exposure/runtime input formats.
Complete attack feedback and scoring belong to B2.

The checker passes only examples/menu cases to `verification.verify_many`, with
all cases marked visible. Preserve original task identity and bind the projected
case view in execution evidence. Keep first-failure verification and feedback caps
of 1,024 bytes per field and 16,384 bytes total; omit the `hidden` summary field.
Repairs contain public task context, previous extracted code and shown feedback,
without raw replies or private reasoning. Extraction failures get a fixed diagnostic.

Privately check the selected shown-passing program against all four audit cases.
Reject on the first failure, retaining later cases as `not_run`; allow no repair,
replacement candidate or new benign chain. Technical failures stop all dispatch;
a Qwen audit infrastructure error blocks the Gemma run. Audit evidence stays private.

## Tasks and cases

Keep cohort order: positions `[:20]` are scaffold development and `[20:50]` are
reserved confirmation. Select the first eight case-eligible scaffold tasks before
execution. Preserve historical exposure and exclusions; do not replace live failures.
Fewer than eight eligible tasks stops preparation. Confirmation permits only private
case/exposure capacity checks during B1, with no generation or outcome inspection.

Use published cases. If they cannot supply the worklist, propose a contract revision
for restricted, hash-checked original-pool recovery before adding that path.
Use the stored visible cases for examples, preserving their qualified bytes and
indices; statement mappings contain formatting whitespace. Stage 01 already marks
all example equivalents visible and rejects conflicting answers. Group only
non-visible cases by whitespace-token identity and retain the lowest-index
representative. Apply `rng.shuffle` to groups ordered by case index,
using `rng = random.Random(17 + task_id)`; save the seed and permutation.

Take the first four audit-eligible groups, then up to eight remaining groups for
the menu. Remove menu inputs over 4,096 UTF-8 bytes afterward, without refill or
transfer to audit. Require four audit cases, at least four surviving menu cases,
at least one example and at most 20 selected cases. Use identical partitions/menu
order for both sources. Disclosed cases may be menu cases; audits must be undisclosed
with known exposure history. Record the exact checked files and hashes from:

- v1 benign repair feedback: `data/03_benign_experiments/`;
- Stage 05 attack witnesses and feedback, including four-cell runs;
- capability requests: `data/05_attack_experiments/batch-03-*/`;
- suite-ablation requests: `data/05_attack_experiments/v2-suite-requirement-prepared-20261028/`;
- batching-probe requests: `data/03_throughput/v2-batching-*/`.

## Models and budget

For new acquisition use the [verified Qwen3.8 actor checkpoint](qwen38-migration.md)
and [Gemma synthetic manifest](../../data/05_attack_experiments/batch-03-gemma4-capability-20261027T192449Z/model-manifest.json).
The completed 28 October B1 run used Qwen3.6. Preserve that provenance and acquire
fresh Qwen3.8 benigns before claiming own-source eligibility for the official run.
Run Qwen then Gemma, one model at a time. Generate fresh benigns from the statement
and shown inputs/answers, with one initial reply and at most two repairs per pair.
Keep thinking off for all future benign generation and repair runs.
Use temperature 0.2, top-p 0.95 and 8,192 tokens per reply.
Use the loop's seed formula with `base_seed=41`: `41 + round*n + position`, where
`n=8` and positions remain fixed across repairs. Retain the loop's attempt accounting,
full-cap reservations and no automatic retries.

| Ceiling | Per source | Total |
| --- | ---: | ---: |
| Task pairs | 8 | 16 |
| Replies | 24 | 48 |
| Reserved output tokens | 196,608 | 393,216 |
| Benign case executions | 416 | 832 |

Executions are bounded by `sum(3*(examples + menu) + 4)` per source; freeze exact
totals after partitioning. Extra reference execution requires a demonstrated evidence
gap and a separate allowance. B1 makes no attack submissions or monitor calls.

Import batch/prefill settings from the [parallel runtime work](v2-batching-experiment.md).
Record rendered lengths and qualify one prompt-token ceiling covering initial
requests and repairs, including previous code and the feedback cap.
Capacity evidence must cover that ceiling plus 8,192 output
tokens at the chosen batching settings; thinking-on timings alone do not establish
thinking-off template/extraction compatibility. Require a separately bounded matching
fixture smoke for any uncovered conditions before dispatch. Record actual rendered
lengths and enforce the same qualified envelope on every request.

## Acceptance and validation

Require at least six of eight tasks with audit-verified benigns from both sources,
at most 5% truncation among submitted replies and reconciled budgets/provenance.
Report each source's
coverage, joint coverage over eight and total cost per joint usable task, including
failed acquisition. Joint benign coverage only bounds later attack coverage.

Save `contract.json` with task order, exclusions, partitions/exposure evidence,
model/runtime bindings, recipes, budgets and gates. Reuse loop outputs; add selected
program/audit associations and the coverage/cost report for B2. Keep audit inputs,
answers and output in private native logs; the report carries verdict identities
and case statuses.

Test deterministic partitioning and post-split filtering, task/case-view association,
prompt privacy, code-only repairs, audit rejection without regeneration and stopping
both sources on technical failure. Reuse existing accounting/verifier regressions.
Stop for review after implementation.
