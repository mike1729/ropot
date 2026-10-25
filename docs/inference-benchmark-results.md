# Local Q8 benchmark results — 2026-10-25

MLX is the provisional deployment choice for both sources. Continuous batching
reduced the mixed-fixture worklist time by about 3× versus the same adapter's
serial path. Larger queues of short witness proposals benefit from a different
prefill setting. Coalesced verification and generation/verifier overlap also help.

These are fixture systems choices. They do not establish scientific witness or
backdoor yield, nor acquisition readiness on the cohort.

## Selected settings

Keep the pinned patched `mlx-lm==0.32.2`, `mlx==0.32.4`, request-owned seeded
samplers, temperature 0.2/top-p 0.95, and the 4-GiB free allocator-cache limit.
Qwen thinking remains disabled. Keep response caching and prefix-state reuse off.

| Worklist | Source | Decode target | Prefill concurrency | Prefill chunk | Median worklist time |
| --- | --- | ---: | ---: | ---: | ---: |
| Ten mixed requests | Qwen3.5-9B Q8 MLX | 8 | 4 | 256 | 39.237 s, versus 117.393 s serial |
| Ten mixed requests | Ministral-3-8B Q8 MLX | 8 | 2 | 2,048 | 27.368 s, versus 84.658 s serial |
| Forty short witness requests | Qwen3.5-9B Q8 MLX | 32 | 16 | 256 | 11.984 s |
| Forty short witness requests | Ministral-3-8B Q8 MLX | 32 | 8 | 2,048 | 12.679 s |

The selected short-worklist settings have five measurements each. Qwen's range
was 11.975–12.633 s; Ministral's was 12.624–12.730 s. Ministral prefill 16 was
slightly faster at 12.087 s, but prefill 8 is within the plan's 10% practical
threshold and uses the smaller concurrency setting. Short-worklist MLX peaks
were 14.76 GiB for selected Qwen and 13.90 GiB for selected Ministral.

These are configured targets, not independently enforced hard batch limits.
The pinned generator sometimes returns more token events than its decode target
during prefill handoff: five at target four, and 33 in some target-32 settings.
The selected short-worklist settings returned at most 32 token events per call.

The mixed natural prompts measured only 127–1,025 Qwen tokens and 115–957
Ministral tokens. The separate fixed-length screen covered random-token prompts
up to 4,096 tokens at actor capacity 4 and synthetic capacity 2, with forced output
lengths. Capacity 8 on long natural repair histories, and capacity 32 on large
contexts, are not qualified by these results. Recheck real length bins before
using the short-worklist setting for those workloads.

## Runtime and scheduling evidence

The official warm Ministral GGUF Q8_0 took median 91.910 s at parallelism 1,
55.577 s at 2, and 43.969 s at 8 on the same ten messages. MLX won at each
matched concurrency. The GGUF template/tokenization guard matched every prompt
ID, including BOS and other special tokens, to the MLX reference tokenizer.
Both routes used fresh sampling, natural stopping, the same seed assignments,
sampler ordering and limits. Quantization artifacts and EOS accounting differ;
this compares deployment packages, not an isolated runtime implementation.

Use one Inspect evaluation for a bounded verification worklist, with
`max_samples=4`, hard `max_sandboxes=4`, serial cases, and one sandbox per program.
On eight invented programs, median verification execution fell from 24.546 s at
sample concurrency 1 to 7.094 s at 4. Every expected verdict and explicit
sample-aware reuse check passed. Reuse took roughly 0.63–0.65 s for eight
programs and is reported separately from execution.

Overlap generation with verification of independent completed work. Across three
measurements per schedule, median complete process makespan fell from 53.757 s
serial to 46.085 s overlapped, a 14.3% reduction. Inference itself slowed slightly
under overlap. These measurements include process startup, load and warmup;
they are not a forecast for a persistent acquisition pipeline. Keep one model
generating at a time; simultaneous models and resident alternating-model runs
were not measured in this increment.

## Recording, recovery and correctness

The thin recorder's fabricated prepare/append/replay/report runs took 1.75 ms
for eight replies, 6.62 ms for 32 and 25.84 ms for 128, with approximately linear
growth. The largest observed caller fraction in measured non-prefix natural,
short and prefill windows was 0.272%. The separate overhead replay's percentage
uses an explicitly modeled one-second-per-request denominator.

The real 20-second native interruption retained 1,790 timed-window partial output
tokens and stopped after 20.457 s, with no owned process group remaining. Other
warmup tokens are also retained. Both models passed uncached marker checks at
targets 8 and 32: request reorder, changed neighbours, early EOS/caps, soft
cancellation and fresh recovery. Saved replies were reopened after an injected
downstream failure with zero new model dispatches.

All 18 first-repeat benign programs from the MLX and GGUF mixed worklists passed
20 invented cases in Docker, with an independent union-find oracle and no new
inference. This is one-task fixture correctness; repeated draws do not provide
independent task observations. Other generated fixture classes remain unscored.

The full repository test suite passed 171 tests, with seven optional skips.
The local real-tokenizer template controls and the native runs supplement those
unit checks. All 32 owned worker process groups were absent at completion; the
only nonzero exit was the deliberately interrupted worker. All three GGUF ports
closed, no Inspect tasks remained, and no benchmark containers remained. An
unrelated container already stopped four weeks earlier was left intact.

## Exact-prefix reuse remains experimental

The short-witness screen measured over 60% prefill time, triggering the optional
prefill and prefix experiments. Exact-prefix copies used the pinned runtime's
`deepcopy` ownership primitive, including Qwen's recurrent `ArraysCache` and KV
state. Construction, cloning, full-state hashing and cleanup count in elapsed
time. Both master-state fingerprints stayed unchanged.

One Qwen control pair took 14.417 s uncached versus 5.789 s cached; one Ministral
pair took 15.303 s versus 6.943 s. However, 2/40 Qwen and 6/40 Ministral responses
differed from their uncached references. Every response stopped normally. The
frozen fixture gate therefore stopped further cached comparisons and prevented
promotion. These are single control pairs, not repeated speed or useful-yield
evidence. Divergence alone does not establish cache corruption; numerical
prefill shapes can change outputs. Resolving this requires semantic/yield checks,
not silently relaxing the gate after observing the result.

## Effort, provenance and remaining gates

The run retained 4,295 request records and 242,310 delivered tokens, including
warmups, synthetic forced-length draws, controls and interrupted output.
Declared caps total 715,204 tokens; those caps are not spent tokens. Unstarted
records are distinct from completed draws. GPU-worker elapsed cost was 2,377.1 s
for actor and 2,682.3 s for synthetic, including their related comparisons and
follow-ups, both below the one-hour-per-model ceiling.

The run used the Apple M5 Pro with 64 GiB, plugged into AC power. MLX resource
snapshots showed no swap growth in the completed measured windows. GGUF peak
GPU memory was not measured; its native server logs and flags are retained.
The existing model manifests and metadata were recorded; weights were not
rehash-validated per draw. The installed Ministral tokenizer emitted its existing
regex warning; its policy was preserved and the actual GGUF prompt-ID guard passed.

The original implementation received the requested independent subagent review
before the first run. It prompted fixes for chat-turn alternation, GGUF BOS
handling and a process-cleanup race. The added cache path received review before
its draws; missing reference controls and source snapshots were fixed first.
Original and cache-enabled adapter source versions are retained separately.

The evidence is in
[`data/03_throughput/q8-benchmark-20261025-v1`](../data/03_throughput/q8-benchmark-20261025-v1/):
[detailed report](../data/03_throughput/q8-benchmark-20261025-v1/report-detailed.md),
[machine analysis](../data/03_throughput/q8-benchmark-20261025-v1/analysis.json),
the pre-dispatch plans, exact source snapshots, native logs, request manifests,
raw replies and all process outcomes. Regeneration commands are in the
[implementation guide](inference-benchmark-implementation.md).

Stage 04–05 acceptance, oracle and independent audit contracts remain TODO in the
authoritative overview. Consequently, recipe/scaffold yield comparisons,
source-owned cohort acquisition, fresh construction confirmation and completion
forecasts are not executed here. No protected or development/construction IDs
were consumed. BF16 stays deferred until the complete witness/backdoor pipeline
works end to end, followed by at least one bounded paired actual-yield check.
Cloud benchmarking waits for a real local completion forecast that misses the
desired experiment window.

## Authorized follow-ups

The next increment will run two bounded local Q8 checks on new invented fixtures:

1. Natural, substantive 4k+ repair histories, comparing conservative and wider
   batching with identical finite worklists, including queue tails, memory and
   cancellation. Requalify large-context settings before using the short-queue
   defaults for them.
2. Prefix-cache semantic controls with independently computed answers and
   multiple tasks/prefix lengths, plus reorder, changed-neighbour and recovery
   checks. Keep token equality as a diagnostic; freeze semantic acceptance before
   new draws. Preserve the previous failed exact-equality gate and its evidence.

These checks do not consume cohort IDs or replace the later full-pipeline useful
yield comparison. BF16 and cloud runs remain deferred.
