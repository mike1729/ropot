# V2 local batching findings — 28 October 2026

These are provisional operating points for Qwen3.6-27B Q8 and Gemma-4-31B-it Q8
on the Apple M5 Pro with 64 GiB RAM. These are historical Qwen3.6 measurements;
the [current actor and monitor](qwen38-migration.md) use Qwen3.8. Similar size and
architecture motivate starting settings but do not establish Qwen3.8 throughput
or long-completion capacity. They select starting settings for further
qualification; they do not establish a global optimum or generation quality.

## Selected settings and conclusions

The [machine-readable profiles](../configs/inference/v2-batching.json) are consumed
by [the benchmark script](../scripts/batching_probe.py) by default.

| Profile | Completion batch | Prefill batch | Prefill chunk | Scope |
| --- | ---: | ---: | ---: | --- |
| `qwen` | 8 | 2 | 256 | Tested mixed short/long prompts |
| `gemma` | 4 | 1 | 256 | Tested mixed short/long prompts |
| `gemma-short` | 8 | 4 | 64 | Eight short prompts, 496–1,245 tokens |

Qwen's selected point reduced elapsed time 36.4% against `(2,1,256)`. Chunk 1,024
was only 3.2% faster and used 1.9 GiB more peak memory, so the prespecified rule
preferring smaller settings within 5% selects chunk 256. Gemma batch 4 reduced
mixed-worklist time 24.8% against batch 2. Its prefill-batch-2 improvement at decode
2 was only 2.3%, with higher memory use.

Gemma reached eight actual decodes on short prompts at `(8,4,64)`. That short-only
worklist was not compared against batch 4, so this is a working short-context
point rather than an established optimum. The same setting on mixed long prompts
triggered swap growth before reaching eight decodes. Use the mixed profile for
unknown/long contexts. Different worklists preclude comparing their elapsed times.

All successful screens generated exactly 256 tokens per request with EOS disabled.
Each setting has one timing observation, not repeated confirmation medians. Long
natural completions, eight all-long prompts and thinking-off extraction remain
unqualified. Acquisition still needs its separate finite runtime/envelope smoke;
these profiles do not satisfy the 8,192-output-token capacity requirement by themselves.
The historical monitor baseline remains frozen and is not covered by this benchmark.

## Measurements

Qwen used 16 requests: eight previously exposed initial prompts and eight invented
historical repair fixtures. Prompt lengths were 485–1,210 and 7,010–7,161 tokens.
Every completed window delivered 4,096 output tokens.

| Qwen completion / prefill / chunk | Seconds | Peak MLX GiB |
| --- | ---: | ---: |
| 2 / 1 / 256 | 401.135 | 28.407 |
| 4 / 1 / 256 | 294.229 | 29.616 |
| 8 / 1 / 256 | 271.981 | 31.996 |
| **8 / 2 / 256** | **254.985** | **32.338** |
| 8 / 2 / 1,024 | 246.861 | 34.256 |

Gemma's mixed worklist used four short and four long requests; lengths were
496–927 and 7,079–7,218 tokens. Each completed window delivered 2,048 tokens.

| Gemma workload | Completion / prefill / chunk | Seconds | Peak GiB | Observed decode maximum |
| --- | --- | ---: | ---: | ---: |
| Mixed | 2 / 1 / 256 | 244.555 | 33.742 | 2 |
| Mixed | 2 / 2 / 256 | 239.037 | 35.222 | 2 |
| Mixed | **4 / 1 / 256** | **183.845** | **36.420** | **4** |
| Mixed | 8 / 1 / 64 | 255.172 | 38.691 | 5 |
| Short only | **8 / 4 / 64** | **86.074** | **38.639** | **8** |

Small prefill chunks with prefill batch 1 delayed admission: a decode target of 8
reached only five actual decodes. Increasing prefill to 4 admitted requests sooner,
but the mixed-workload run stopped after 111.869 seconds and 446 partial tokens:
swap grew 1.357 GiB, peak MLX memory was 41.605 GiB and observed decode reached 5.
This incomplete window is excluded from the table. Interrupted chunk comparisons
also retain their costs: Qwen 45.881 seconds / 1,237 tokens; Gemma 62.599 seconds /
788 tokens. Final chunk and capacity checks reloaded models, adding order/load
confounds. Repeated fixtures are timing observations, not independent quality samples.

## Runtime and resource controls

Runs used patched MLX 0.32.4 / mlx-lm 0.32.2, thinking on, temperature 0.2,
top-p 0.95, fixed request-owned seeds, one resident model at a time, and a 4 GiB
allocator-cache limit. Prefix reuse and response caching were off. Model load was
reported separately. The runner reused the generation adapter with benchmark-only
rendering, resident-load and screening-EOS overrides. Token accounting passed.

The first three Qwen probes stopped on warning pressure before inference. That
gate was too restrictive: historical batch-8 runs had succeeded, and warning alone
supplied no capacity conclusion. Later runs logged warning pressure and stopped on
critical pressure, swap growth above 256 MiB or an MLX peak above 44 GiB. Gemma's
first load added 3.513 GiB swap and stopped; its later retry loaded without growth.

The initial Gemma estimate overcounted rotating-cache length and skipped batch 4.
Inspection of `BatchRotatingKVCache._update_concat` established a retained length
of at most `sliding_window + chunk - 1`. The expanded estimate used weights + full
KV with 256-token rounding headroom + chunk-bounded rotating KV + at least 2 GiB
workspace. Batch 4's measured residual plus 1 GiB could raise that workspace reserve.
Estimates were 38.639 GiB for `(4,1,256)` and 43.730 GiB for `(8,1,64)` on mixed
prompts. Memory limits and estimates are advisory; checks between native steps
cannot prevent an instantaneous allocation spike. Do not infer a universal RAM limit
from one host's swap behavior.

## Script use and evidence

With Metal access and the pinned MLX Python, the default command runs one warm-up
and one fixed-length screen at the selected profile; it never launches acquisition.
The Gemma default uses the measured eight-request mix; Qwen uses 16 requests.

```sh
PYTHON=/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python
"$PYTHON" scripts/batching_probe.py qwen data/03_throughput/selected-qwen
"$PYTHON" scripts/batching_probe.py gemma data/03_throughput/selected-gemma
"$PYTHON" scripts/batching_probe.py gemma data/03_throughput/selected-gemma-short --short-only
"$PYTHON" scripts/batching_probe.py gemma unused --short-only --show-settings
```

Settings are loaded from `configs/inference/v2-batching.json`; each new run records
the selected profile and config hash. Callers with explicit runtime settings can
read its `profiles` mapping, after their required runtime qualification. Use
`--sweep` for the original adaptive sweep/full confirmation, `--minimal` for the
old eight-request sweep, `--minimal --capacity` for the expanded Gemma probe, and
`--minimal --capacity --short-only --probe-settings 8 4 64` for the short-only check.
Experimental flags retain the historical paths; exact historical scripts are also
saved with the evidence. No additional model runs were made to update defaults.

Raw replies, model manifests, rendered IDs, telemetry, outcomes and exact runner /
adapter snapshots remain Git-ignored under `data/03_throughput/`:

- `v2-batching-20261028`, `-r2`, `-r3`: aborted pressure-only probes.
- `v2-batching-20261028-prepared`: frozen requests and CPU tokenizer checks.
- `v2-batching-20261028-r4`: main Qwen sweep and initial Gemma load stop.
- `v2-batching-20261028-chunk`: final Qwen chunk comparison.
- `v2-batching-20261028-gemma-retry`: Gemma baseline/prefill screens and interruption.
- `v2-batching-20261028-gemma-capacity`: batch-4/8 probe and combined retry summary.
- `v2-batching-20261028-gemma-b8-prefill`: mixed batch-8 swap stop.
- `v2-batching-20261028-gemma-b8-short`: successful eight-decoder short-context check.

The first timed sweep took 27.91 minutes. The Gemma retry took 24.70 minutes and
retained 11,602 output tokens including warm-ups and incomplete work. All owned
workers exited. No new study allocation, candidate execution or quality scoring
occurred. Raw evidence is local and is not embedded in this commit.
