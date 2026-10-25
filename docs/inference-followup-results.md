# Q8 context-length and prefix-cache follow-ups

The benchmark increment is complete, including the declared resource-stop
condition. Use these provisional settings on the local 64-GiB Apple M5 Pro:

| Model/workload | Decode target | Prefill batch | Prefill chunk |
| --- | ---: | ---: | ---: |
| Qwen, roughly 2k or 7k repair context | 32 | 16 | 256 |
| Ministral, roughly 2k repair context | 32 | 8 | 2,048 |
| Ministral, roughly 7k repair context | 8 | 2 | 2,048 |

Ministral's 7k recommendation has one completed timing window, not three repeats.
Batch 32 failed with native out-of-memory; batch 16 remains untested. Keep prefix
reuse off by default until actual full-pipeline verified-artifact yield is compared.

The earlier findings and implementation were committed as `20130cc`. The
reviewed follow-up implementation is `9a82cb5`; its [frozen plan](inference-followup-plan.md)
adds the requested roughly 2k-token context comparison to the longer repair and
semantic-cache tests. This increment uses MLX Q8 for both models. It does not
repeat the earlier Ministral GGUF comparison.

Evidence is retained in the
[initial phase](../data/03_throughput/q8-followups-20261025-v1/) and
[continuation/replay](../data/03_throughput/q8-followups-completion-20261025-v1/).
The [combined report](../data/03_throughput/q8-followups-completion-20261025-v1/analysis/report.md)
and [machine analysis](../data/03_throughput/q8-followups-completion-20261025-v1/analysis/analysis.json)
join both phases without selecting replies by quality.
Exact source snapshots, model metadata, tokenized inputs, raw replies, native
statistics and worker outcomes accompany the run. The finite driver runs one
model at a time and verifies generated programs after GPU work finishes.

## Context-length comparison

Both bins preserve three invented graph tasks, task/variant IDs, request-owned
seeds, four faulty full-program attempts and the fault pattern. Shorter histories
contain complete 32-edge failure captures; longer histories contain 240-edge
captures. These are natural messages rather than padded token arrays. The bins
therefore change the visible inputs as well as prompt length; this is a workload
comparison, not a causal estimate for adding otherwise irrelevant tokens.

Actual prompt ranges are 1,944–2,103 versus 6,954–7,196 tokens for Qwen, and
1,872–2,037 versus 6,882–7,130 for Ministral. Every window has 33 requests,
including the tail at target 32. The same 20-case suite applies to matched task
variants across both bins. Natural stop, valid Python and all cases passing are
required for a verified draw; truncated or malformed draws remain in the count.

Completed 2k timing comparisons:

| Model | Batch target | Repeats | Median seconds | Range seconds | Peak GiB |
| --- | ---: | ---: | ---: | --- | ---: |
| Qwen | 8 | 3 | 132.176 | 131.920–133.569 | 12.19 |
| Qwen | 32 | 3 | 93.454 | 93.126–93.615 | 17.37 |
| Ministral | 8 | 3 | 125.975 | 125.747–126.095 | 17.60 |
| Ministral | 32 | 3 | 92.337 | 91.532–92.422 | 27.61 |

For Qwen, target 32 reduces complete-worklist time by 29.3% relative to target 8.
The single conservative target-4 calibration took 166.392 seconds and peaked at
11.15 GiB. This calibration is not a three-repeat comparison. All replies in
these windows stopped naturally. All strict-format accepted programs passed
Docker verification. Ministral's matching reduction is 26.7%; its batch-32
memory use is substantially higher than Qwen's despite similar elapsed time.

Qwen's first paired runs generated 11,935 versus 12,040 tokens, so the timing gain
is not explained by substantially shorter completions. Native target 8 briefly
emitted nine token events in a step, and target 4 emitted five: targets remain
scheduling controls rather than proven hard concurrency caps.

The completed Qwen 7k timing comparison is:

| Batch target | Repeats | Median seconds | Range seconds | Peak GiB | Verified draws |
| --- | ---: | ---: | --- | ---: | ---: |
| 8 | 3 | 272.483 | 262.729–272.485 | 15.26 | 90/99 |
| 32 | 3 | 224.178 | 218.732–228.592 | 27.20 | 96/99 |

This is a 17.7% median time reduction. Batch-32 windows each delivered the same
12,838 tokens, so their modest timing spread is not an output-length effect.
Every naturally stopped program in these Qwen windows passed all 20 cases. The single
target-4 calibration took 313.611 seconds and had 31 natural stops, two caps.

Ministral's target-2 7k calibration reached its 360-second soft cap after 31
natural completions, retaining the two remaining partial replies. Its full
worklist timing is censored; the interrupted 360-second value is excluded from
completion medians. Original interrupted cells and their costs are retained
alongside a separately frozen [bounded continuation](inference-followup-continuation.md).

Ministral's completed 7k batch-8 worklist took **274.686 seconds**, peaking at
**26.29 GiB**, with no swap growth. All 33 replies stopped naturally: 24 passed
the primary parser/verifier, and nine more passed after secondary code extraction.

Its 7k batch-32 run failed after 208.693 seconds, with 6,224 tokens delivered
and one reply completed. MLX reported a **53.61-GiB peak** and Metal native
out-of-memory; swap grew by **13.17 GiB**. This is not a completed-worklist timing.
The spike happened within a native call, before the between-call 42-GiB check
could stop it. That check is not a hard allocation limit.

The native-error and swap-stop conditions both applied, so remaining Ministral
7k repeats and the conditional batch-16 fallback did not run. No configuration
was retried after this memory failure. The two worker groups exited, and a
read-only probe then showed 89% memory free. CPU-only recovery verified retained
outputs without any further model calls. Final probes showed 90% memory free,
zero benchmark processes and zero running containers; see
[cleanup evidence](../data/03_throughput/q8-followups-completion-20261025-v1/host-recovery.json).

## Prefix-cache semantics

Qwen's semantic matrix completed all 58 windows, including deliberate soft
cancellation and recovery. All 28 shared-master fingerprints were unchanged.
There were 941 correct and 220 wrong integer answers among 1,161 scored replies;
none failed the JSON format check. Counts include repeated queries and controls,
so they are not independent samples from a task distribution.

For the 378 paired sampled requests, uncached and cached arms had identical
correct/incorrect verdicts. Three token-sequence differences were the same
request repeated in three blocks: both answers were wrong (2 versus 3, with
oracle answer 4). This separates semantic correctness from bitwise agreement.
The frozen all-correct gate is unmet; this does not diagnose cache corruption.

| Qwen semantic group | Uncached median s | Cached median s | Correct per arm |
| --- | ---: | ---: | ---: |
| Components, short | 10.609 | 2.960 | 78/99 |
| Components, long | 45.904 | 6.661 | 21/27 |
| Bipartite, short | 11.273 | 2.986 | 90/99 |
| Bipartite, long | 47.006 | 6.777 | 27/27 |
| Distance, short | 10.667 | 2.967 | 69/99 |
| Distance, long | 45.741 | 6.645 | 18/27 |

These cached times include prefix creation, cloning and fingerprint checks.
They measure short JSON answers, not full-program completions; do not transfer
their speedup factors directly to longer-output generation.
Keep prefix reuse off by default pending full-pipeline verified-artifact yield
comparison. Its measured benefit applies to groups sharing an exact prefix,
not unrelated tasks. Execute the independent oracle to establish witness
answers; a model's own mental execution is not a reliable substitute.

Ministral completed the same 58 windows and all 28 master-state checks matched.
All 1,161 semantic replies used JSON fences, failing the primary no-fence schema.
An explicitly post-hoc diagnostic removes exactly one whole-reply JSON fence
and keeps every other original field/type/oracle rule: 882 answers then pass
and 279 are wrong. Its three sampled correctness flips are one repeated case:
the uncached answer is 0, cached is 1, and the oracle is 1. Do not interpret
three identical seeded repeats as three independent improvements.

Across these fixtures, Qwen cached medians are roughly 3.6–3.8x faster for short
prefixes and 6.8–6.9x for long prefixes. Ministral gains roughly 2.8–3.0x and
4.8–4.9x respectively. Both models retain the failed primary semantic gate.
All 56 master-state checks match; no request-ID or token-accounting exception
occurred in the completed matrices. This supports the tested cache mechanics,
without establishing general generation equivalence.

## Parser recovery and executable verification

All **579 primary-accepted program draws** passed their 20-case suites. The
primary totals, including interrupted cells, are:

| Model | Primary passed | Strict-format failures | Incomplete records |
| --- | ---: | ---: | ---: |
| Qwen | 448 | 0 | 47 |
| Ministral | 131 | 165 | 34 |

The separate, explicitly post-hoc recovery accepts exactly one unambiguous Python
fence from naturally stopped strict-format failures, then runs the same native
verifier. It recovered **158 additional passing Ministral draws**. Seven failed:
the same seeded components request repeated across seven windows returned
`n - components` instead of `components` (32 versus expected 112 on its first
visible case). Ambiguous blocks and incomplete replies are never salvaged.

For matched 2k worklists, this changes Ministral's verified count from **36/99
to 96/99** at both targets 8 and 32, using zero extra model calls. For the one
completed 7k target-8 window, it changes 24/33 to 33/33. Primary scores remain
unchanged. This makes unambiguous extraction plus execution checks a worthwhile
scaffold choice to freeze before real acquisition; syntax alone is insufficient.

The recovery verifier deduplicated exact task/program bytes into 43 executions,
including three independent reference controls, and took 36.1 seconds. All draw
identities remain counted. Total verified program draws after recovery are
737/825 retained non-warmup program records; these are repeated fixture draws,
not 737 distinct scientific artifacts.

## Accounting and limits

All warmups, controls, timeout/OOM partials and unstarted records are retained.
There are **3,213 request records and 301,711 delivered tokens**, including EOS,
warmups and partials. GPU-worker elapsed cost was **3,896.1 seconds for Qwen**
and **2,960.4 seconds for Ministral**, including reloads and failed windows.
Three CPU verification phases took approximately 244.8 seconds in total.

The raw evidence is about **78 MiB across 3,827 files**, before the final analysis
outputs. Three native Inspect `.eval` logs account for about 2.2 MiB. Source
snapshots and hashes, model manifests, tokenized messages, raw replies and all
process outcomes are retained. These generated files are separate from the
committed implementation and curated findings.

Completion medians exclude interrupted windows, with their costs reported
separately. Original derived reports are preserved alongside a corrected
postprocessing revision. Do not treat a short failed run as a throughput gain.
Warmups use three requests per configuration, not every full-batch shape.
Cache-arm randomization happened to put uncached before cached in all three
blocks; those comparisons are not counterbalanced.

The same request-owned seeds repeat across timing blocks, so repetitions measure
timing stability rather than independent yield samples. These workloads are
fixtures, not forecasts for the eventual complete pipeline. Only one model was
resident at a time. Future capacity testing must account for transient cache
growth before admitting larger full-attention batches; between-call observation
did not prevent the demonstrated allocation burst.

Repair faults are seeded arithmetic offsets, so this measures inference and
full-program repair throughput rather than difficult debugging ability. The
semantic tests ask for small independently computed integer answers on three
tasks and repeated edge cases. Passing them will qualify those fixture checks;
it will not establish general code-generation equivalence or scientific yield.

The earlier failed token-equality cache gate remains unchanged. BF16 actual-yield
checking remains deferred until the full witness/backdoor pipeline is working.
No protected, development or construction cohort IDs are used in this increment.
Cloud comparison still waits for an actual local completion forecast that misses
the desired experiment window. The full local suite passed **188 tests**, with
seven optional skips, and the requested pre-run independent review was completed.
