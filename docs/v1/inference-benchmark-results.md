# Local Q8 inference findings — 2026-10-25

Use MLX Q8 provisionally for Qwen3.5-9B and Ministral-3-8B on the local Apple
M5 Pro with 64 GiB RAM. These invented-fixture measurements select starting
settings; they do not measure source-owned witness/backdoor yield or establish
cohort acquisition readiness. No study tasks were consumed.

## Measured operating points

The archived runs used patched `mlx-lm==0.32.2`, `mlx==0.32.4`, request-owned
seeded samplers, temperature 0.2/top-p 0.95, thinking off, and a 4-GiB free
allocator-cache limit. Keep response caching and prefix-state reuse off.
The allocator cache limit does not bound live model/KV allocations.

| Workload | Model | Decode target | Prefill batch | Prefill chunk | Median elapsed |
| --- | --- | ---: | ---: | ---: | --- |
| Ten mixed short requests | Qwen | 8 | 4 | 256 | 39.237 s vs 117.393 s serial |
| Ten mixed short requests | Ministral | 8 | 2 | 2,048 | 27.368 s vs 84.658 s serial |
| Forty short witness proposals | Qwen | 32 | 16 | 256 | 11.984 s |
| Forty short witness proposals | Ministral | 32 | 8 | 2,048 | 12.679 s |
| 33 repair requests, ~2k prompt | Qwen | 32 | 16 | 256 | 93.454 s vs 132.176 s at batch 8 |
| 33 repair requests, ~2k prompt | Ministral | 32 | 8 | 2,048 | 92.337 s vs 125.975 s at batch 8 |
| 33 repair requests, ~7k prompt | Qwen | 32 | 16 | 256 | 224.178 s vs 272.483 s at batch 8 |
| 33 repair requests, ~7k prompt | Ministral | 8 | 2 | 2,048 | 274.686 s, one completed window |

The 2k and Qwen 7k comparisons have three repeats per setting. The short-witness
settings have five measurements each. Repeated requests are timing replicates,
not independent scientific yield samples. Natural prompt ranges were 1,872–2,103
and 6,882–7,196 tokens across models. The longer histories also contain different
failure inputs, so they do not isolate the causal effect of prompt length.

Ministral 7k/batch 32 failed with Metal OOM, reaching 53.61 GiB and adding
13.17 GiB of swap. Remaining capacity experiments stopped; batch 16 is untested
and batch 8 is provisional. Between-call memory checks failed to prevent the
allocation spike. OOM/cancelled/partial windows retain their cost and outputs but
are excluded from completed-worklist medians. All owned workers were cleaned up.
Decode targets are scheduler controls: some native steps emitted more token
events than the configured target. They are not proven hard concurrency caps.

## Runtime, scheduling and extraction

- Warm official Ministral GGUF Q8_0 took 43.969 s at parallelism 8 on the same
  ten mixed messages, versus MLX's 27.368 s. Prompt IDs matched, but weights and
  EOS accounting differ: this compares deployment packages. GGUF was not retested
  at 2k/7k contexts.
- One Inspect evaluation with four active samples verified eight programs in
  7.094 s versus 24.546 s with one active sample. Each program keeps a fresh
  sandbox and serial cases. Overlapping generation with independent verification
  reduced complete process makespan from 53.757 to 46.085 s (14.3%). Keep one
  verification evaluation active at a time; independent host calls do not share
  a global four-sandbox limit. Dual-model generation was not tested.
- Thin recording was below 0.3% of observed elapsed time in measured non-prefix
  windows. Retain raw replies before scoring so processing failures can be
  recovered without another inference request. A deliberate 20-second hard
  interruption stopped after 20.457 s and retained 1,790 partial output tokens.
- Secondary extraction of exactly one complete, unambiguous Python block from
  naturally stopped replies recovered 158 additional verified Ministral draws
  with no new inference. Seven recovered draws still failed execution. Freeze
  the extractor before study acquisition; parseable code alone is insufficient.

Prefix-state reuse accelerated short JSON-answer fixtures by roughly 3–7×,
including prefix construction, cloning and fingerprint checks. All 56 shared
master-state fingerprint checks stayed unchanged. The correctness gate remains
unmet: Qwen had 220 wrong integer answers among 1,161 scored replies; Ministral
fenced all 1,161 JSON replies and had 279 wrong answers after removing that wrapper.
Some cached outputs differed from uncached outputs. These results neither
qualify full-program yield nor demonstrate state corruption. Keep prefix reuse off
until a paired comparison measures actual verified-artifact yield.

## Evidence and archive

The complete code, tests, plans and both detailed reports are preserved on local
branch `codex/archive-q8-benchmarks-20261025` at
`eaf995407f2a8f025401a6ae6e336875205ecbe4`. It includes the initial implementation
(`20130cc`) and reviewed follow-up implementation (`9a82cb5`). For example:

```sh
git show codex/archive-q8-benchmarks-20261025:docs/inference-followup-results.md
git worktree add --detach ../ropot-q8-benchmarks codex/archive-q8-benchmarks-20261025
```

The branch preserves tracked source and reports. Raw evidence is ignored by Git
and remains unchanged in this checkout:

- [Original run and analysis](../../data/03_throughput/q8-benchmark-20261025-v1)
- [Initial follow-ups](../../data/03_throughput/q8-followups-20261025-v1)
- [Continuation, combined analysis and CPU-only recovery](../../data/03_throughput/q8-followups-completion-20261025-v1)

Those directories retain exact source/model metadata, requests, tokenized inputs,
raw replies, native logs, accounting and cleanup evidence. A separate worktree
needs explicit paths to this evidence and the pinned local model/runtime artifacts;
checking out the branch does not copy ignored data. Historical log reuse stays
bound to its original execution/source contract.

Main retains the direct generation adapter, coalesced verifier and their tests.
The one-off benchmark orchestration, cache prototype, fixtures and report builders
live on the archive branch. Future work is in the [remaining plan](inference-benchmark-plan.md).
