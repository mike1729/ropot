# Bounded repair timing continuation

The initial 2k+7k matrix exceeded its long-worker time estimate: Qwen's 2k
windows cost about 850 seconds, while the 7k calibration and first batch-8
window cost 314 and 272 seconds. Keep the first phase's 2,100-second soft and
2,160-second hard caps intact. Semantic workers retain their original budget.

After the first phase exits, complete only repair comparison cells whose native
window was not finished because of the finite time budget. Keep the Ministral
target-2 calibration as right-censored: it stopped at 360 seconds after 31/33
natural completions. Do not redraw it just to drain the remaining replies.
Use `experiments.run_followup_completion` with a new evidence directory.
Actor worker soft/hard budgets are 1,200/1,260 seconds; synthetic budgets are
1,800/1,860 seconds, since none of its 7k paired cells ran in the initial phase.
Before dispatch, retain a source snapshot and the exact list of missing cells.
Use the existing scoped process launcher for hard deadlines and cleanup.

This is a scheduling continuation, not a quality retry. Native-success cells
are never redrawn, including truncated, malformed or incorrect completions.
Do not retry memory, state-integrity or other native failures. Audit unindexed
window summaries as well as the outer index before allowing continuation.
Preserve every
original partial/unstarted draw and its elapsed cost. A time-interrupted cell
is redrawn as its complete original 33-request worklist, with identical token
IDs, request-owned seeds, output caps and batching settings. Warm three capped
requests per new configuration; record and count these as warmups.

Validate model/runtime source and metadata fingerprints and exact tokenizer
outputs against the original runtime before generation. Retain the 4-GiB free
allocator cache, 42-GiB observed memory ceiling and 256-MiB swap-growth stop.
Verify programs through the same native 20-case suite. Compare complete-window
timings only, report missing cells explicitly, and count original interrupted
draws plus all continuation draws in total cost and quality accounting.

Review the continuation adapter and retry-scope tests before it draws tokens.
The original frozen source set is not modified during its running driver.

The independent `benchmark_pre_run_review` agent reviewed the adapter and tests
before continuation dispatch and reported no P1/P2 blockers. It confirmed the
quality-independent retry scope, frozen inputs, memory/integrity rejection,
new evidence directory and identity guards. The two retry-scope tests passed.

Before the synthetic continuation's first 7k batch-32 attempt, declare one
adaptive alternative: if that configuration exceeds the observed 42-GiB MLX
ceiling, reject it and omit its later repeats. Provided swap has not grown by
over 256 MiB, release allocator cache and finish the batch-8 repeats, then run
three matched batch-16 windows with prefill batch 2 and chunk 2,048. These
fallback timings are explicitly adaptive, not the original randomized pairs.
Do not retry the rejected batch-32 configuration. Other native failures or
swap growth stop the worker. All failed-window cost and partial replies count.

After GPU work, verify the continuation's primary programs in Docker. Then run
`secondary_code_replay` on both evidence roots: accept only naturally stopped
strict-format failures with one unambiguous complete Python fence, ignoring
surrounding prose, and verify the extracted program against the same suite.
Label these results as post-hoc recovery; keep primary scores unchanged and
make zero additional model calls. The secondary extractor and tests received
independent review with no material findings.

The revised continuation, launcher and adaptive sequencing also received review
with no P1/P2 blockers. The requested `--verify-only` CPU mode is now implemented
and tested. Postprocessing excludes interrupted windows from timing medians,
retains their elapsed cost, and reports no completed timing when all windows of
a configuration fail. The suite passed 187 tests with seven optional skips.

Ministral's 1,161 semantic replies all used JSON fences, failing the primary
no-fence schema. As a separately labelled post-hoc diagnostic, the combined
analysis may remove exactly one whole-reply JSON fence and apply all original
field/type/oracle checks to its payload. This makes zero model calls and cannot
change primary qualification. Tests cover preserving the primary rejection and
rejecting prose, malformed values and incomplete replies in secondary scoring.

```sh
/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python \
  -m experiments.run_followup_completion \
  --source data/03_throughput/q8-followups-20261025-v1 \
  --output data/03_throughput/q8-followups-completion-20261025-v1
```

Execution stopped at the synthetic 7k batch-32 attempt: Metal reported native
out-of-memory, MLX peak was 53.61 GiB and swap grew by about 13.2 GiB. A native
call can spike before the between-call 42-GiB check. This was not the controlled
ceiling error, and the no-swap condition also failed, so the declared fallback
did not run. Do not retry that configuration or promote untested batch 16.
The completed batch-8 window used 26.29 GiB and caused no swap growth.

Both owned worker groups exited. A read-only host probe then showed 89% memory
free and no remaining inference/controller process. Finish only CPU derivation
using the existing retained outputs; make zero further inference calls:

```sh
.venv/bin/python -m experiments.run_followup_completion \
  --source data/03_throughput/q8-followups-20261025-v1 \
  --output data/03_throughput/q8-followups-completion-20261025-v1 --replay-only
```

This mode preserves the native plan, writes a separate replay plan/source
snapshot, and launches only the two reviewed CPU stages. A recovery-mode test
checks zero GPU dispatches and preservation of the original plan.
