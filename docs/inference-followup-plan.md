# Long-context and semantic cache checks

Execution results are in the [follow-up findings](inference-followup-results.md).
The original pre-run version is preserved with the retained source snapshots.
The initial matrix hit time caps and a later Ministral 7k/batch-32 native memory
failure; its historical command is not the recommended operating configuration.

This is a new local Q8 fixture increment after commit `20130cc`. It preserves the
earlier measurements and failed exact-equality cache gate. No cohort IDs, BF16,
cloud inference or scientific acquisition occurs.

## Frozen inputs and acceptance

Use three invented undirected-graph tasks: connected components, bipartiteness,
and shortest distance from vertex 1 to vertex n. Reference programs use iterative
DFS/BFS; answer oracles use union-find, parity union-find and Bellman-Ford.
Histories contain four seeded faulty full programs and their complete valid
visible input/stdout captures. The seeded faults are arithmetic output offsets:
this tests context handling and full-program repair, not difficult bug discovery.
The roughly 2k bin uses 32-edge captures; the long bin uses 240-edge captures.
The former must measure 1,800–2,600 tokens and the latter at least 4,096 tokens
under each pinned tokenizer. Preserve four attempts and the same fault pattern.
Never pad or truncate messages to force a bin. Validate lengths before drawing.

Long-repair replies must stop naturally, parse as a complete Python program
(one optional Python code fence), and pass all 20 frozen cases in the native
Docker verifier. Eight cases are the visible history cases from both context
bins, followed by eight small edge cases and four seeded random controls.
Use the same verification suite for matched task/variant IDs in both bins.
Every failed/truncated draw counts. Identical task/program bytes may share one
explicit verification result; count all generated draws and disclose this reuse.

Semantic cache replies must naturally stop and contain exactly a JSON object
with one integer field `answer`, equal to the independent oracle. Reject code
fences, extra/duplicate fields, booleans and truncated replies. Inputs are valid
by construction. Token equality is diagnostic, not this increment's semantic
criterion; previous exact-equality results are unchanged.

## Matrix and finite budget

Run one model at a time, pinned MLX Q8, thinking off, 4-GiB free allocator cache,
no response caching. Snapshot source bytes, versions, model metadata and complete
inputs before dispatch. Use a new output directory. Count all warmups and
controls. Source/fixture review precedes native inference.

- Repair contexts: matched 33-request worklists in both the roughly 2k and 7k
  bins, including a tail at target 32. Run the shorter context first.
  One conservative calibration window (actor 4, synthetic 2), then three
  randomized paired blocks at targets 8 and 32. Use selected prefill settings
  from the previous increment. Warm three requests per configuration; this does
  not claim every full-batch kernel shape was warmed. Stop optional repeats if
  the remaining declared budget is insufficient. Report incomplete cells.
- Cache semantics: three tasks × short/long prefixes. Long groups contain nine
  requests at target 8; short groups contain 33 at target 32. Distinct suffix
  inputs produce varied known answers. First run greedy serial, greedy batched
  and greedy cached controls. Then three randomized uncached/cached pairs per
  group at temperature 0.2/top-p 0.95 with request-owned seeds. Prefix creation,
  copying and master-state checks count in complete elapsed time.
- Cached reorder, changed-neighbour and soft-cancellation/recovery controls use
  the long components group. Rescore every reply against its own frozen answer;
  changing a neighbour changes only that neighbour's expected answer.
- Cache qualification requires all declared cells completed, all baseline and
  cached semantic replies correct, no token/association/state-isolation failure,
  and successful reorder/neighbour/recovery controls. Report baseline errors
  separately. This is fixture qualification, not scientific yield equivalence.

Long-worker soft budget: 2,100 seconds; hard process cap: 2,160. Semantic worker:
1,500 seconds soft, 1,560 hard. Thus at most 3,720 GPU-worker seconds per model in
this newly authorized increment, excluding separate CPU verification. Per-window
soft cap is 360 seconds, bounded by remaining worker time. Stop on observed MLX
peak above 42 GiB; it is an observed boundary, not an allocator hard guarantee.
If a long window grows swap by over 256 MiB, do not run the wider configuration
again. Retain partial replies on errors and prove scoped process cleanup.

## Reporting

Report actual prompt lengths, stop/cap/failure counts, median/range where repeated,
complete finite-worklist times, native prompt/decode counters, memory/swap and
queue drain. Separate one-shot calibration from repeated comparison. Bind each
long program to its suite/verifier result. For cache semantics report correctness,
format failures, raw-token divergences and paired elapsed time. Do not call an
incorrect baseline evidence of cache safety. Avoid p95 or task-yield claims from
these small fabricated families.

Run after source review:

```sh
/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python \
  -m experiments.run_inference_followups \
  --output data/03_throughput/q8-followups-20261025-v1
```

The driver freezes source copies, runs four sequential GPU workers, then invokes
the existing four-sandbox Docker verifier using `.venv/bin/python`. It writes
`report.json` and `report.md`; raw outputs and timings remain alongside them.
For offline report regeneration, run `.venv/bin/python -m
experiments.followup_report --root <retained-directory>` without `--verify`.
The worker budget includes model load; greedy cache controls also warm the
sampled semantic comparison, and no semantic draws are discarded as warmups.

Tokenizer-only preflight measured actor repair prompts at 6,954–7,196 tokens and
synthetic prompts at 6,882–7,130. Components shared prefixes measured 388/7,296
tokens (actor short/long) and 368/7,212 (synthetic). Native workers repeat the
length check before generation and retain exact token IDs.
The added 2k bin measures 1,944–2,103 tokens (actor) and 1,872–2,037 (synthetic).

Pre-run review: `benchmark_pre_run_review` found no P1/P2 blockers after checking
the added 2k bin, verifier bindings, complete-matrix semantic gate and bounded
cleanup. Its suggested stronger changed-neighbour control was applied: the
neighbour now changes the expected answer from 1 to 4. Review executed no model
or Docker work. The local suite passed 177 tests, with seven optional skips.
