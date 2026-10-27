# Batch 5 synthetic runtime implementation plan

Run the local Ministral-3-8B-Instruct-2512 MLX 8-bit conversion through the
existing [direct generation module](../../src/data-pipeline/stage03/local_generation.py).
Preserve its finite warm serial run, plain records and public API:
`generate(model_path, requests, output_dir, *, model_manifest)`.

## Runtime choice

Use MLX Q8 provisionally, following the completed [Q8 fixture benchmarks][findings].
On the same ten mixed messages, warm MLX took 27.368 seconds versus 43.969 seconds
for official GGUF Q8_0 at parallelism 8. Prompt IDs matched, but weights and EOS
accounting differed: this selects a deployment package for those fixtures, not
the faster runtime in isolation or a source-owned acquisition yield.

The comparison is complete; do not repeat it or build a GGUF adapter for Batch 5.
The benchmark code is archived at `eaf995407f2a8f025401a6ae6e336875205ecbe4` on
`codex/archive-q8-benchmarks-20261025`. It used `BatchGenerator`, including its
serial arm. Main's `stream_generate` path still needs the small synthetic smoke
below. If that path fails, retain the failure and revise the narrow runtime choice.

The failed `noc-naukowcow` branch's commit `331673b` supplies useful tokenizer and
prompt checks. Reimplement those few details in the current module. Its worker
protocol, GGUF server controller and later campaign machinery are unnecessary here.

## Implementation

1. Reuse the existing pinned MLX Python and local model artifacts. The synthetic
   path is `/Users/michal/Workspace/models/mlx-community/Ministral-3-8B-Instruct-2512-8bit`,
   with revision `f333d6bcb3df97264eddcf985523b69b747f5385` and its existing manifest
   under `/Users/michal/Workspace/models/metadata/`. Keep the 8-bit requirement and
   the caller's responsibility for unchanged weights; avoid rehashing weights.
2. Read the model config once when preparing the run. For the selected synthetic
   config (`model_type=mistral3`, text model `ministral3`), pass
   `fix_mistral_regex=True` alongside `local_files_only=True` through the existing
   public [MLX loader][loader]. Preserve the actor's loading arguments. The installed
   loader already supports this model; no architecture patch is needed. The saved
   benchmark omitted this regex flag and retained its warning, so its prompt-ID
   checks do not qualify this changed tokenizer policy. Check it in the smoke.
3. Record effective tokenizer options and class once in `runtime.json` and include
   `tokenizer_config.json` among the hashed model metadata. Record the relevant
   Ministral implementation sources (`mistral3` and `ministral3`) for synthetic
   runs, alongside the existing generation/sampling/tokenizer sources. Keep the
   actor's Qwen source record. Use a small explicit model-type branch.
4. Use the pinned sampling patch's `make_sampler(..., seed=request["seed"])` for
   each request, matching the benchmark's request-owned random state. Replace
   global reseeding; historical Batch 4 seeded replies remain a separate baseline.
   Keep request fields, streaming, fresh decoding state and result statuses.
   Keep `enable_thinking=False` and inspect the rendered synthetic prompt. Construct
   smoke requests with an explicit system message, avoiding the checkpoint
   template's default assistant/date/tool preamble.

Use the existing input, runtime, reply and outcome files. Preserve raw replies,
native token counts, partial output and unstarted requests. Keep response caching
and prefix-state reuse off. The existing soft deadline and interruption limitations
remain: checks between chunks cannot stop model loading or a stalled native call.
The benchmark's hard worker interruption does not transfer to this direct path.
Failure stops the run without retries.

## Validation

Extend the current mocked tests to check synthetic tokenizer arguments, metadata
and model source records. Update sampler assertions to check a fresh sampler with
each request's seed for both models. Keep actor regression coverage and the existing
serial-load, raw-output, stop/length, timeout and interruption checks. Retain the
actor's opt-in smoke for the shared sampler change.

Add an opt-in smoke using fabricated tasks only:

- Compare initial and repair-history prompt text/IDs with the official Ministral
  tokenizer snapshot `5b26027e7b19eeb4b7352e1fed3926375dd2cb4d` under
  `/Users/michal/Workspace/models/mistralai/Ministral-3-8B-Instruct-2512/`, using
  the same regex fix and explicit system message. Check a single leading BOS and
  the expected EOS configuration.
- Run four requests through one model load: square and distinct-count coding
  fixtures in an A/B/A sequence with repeated A settings/seed and greedy B;
  and one request forced to exhaust a one-token limit. Use temperature 0.2 and
  top-p 0.95 for A, matching the measured sampling policy.
  Cap each coding reply at 256 tokens and each request at 120 soft seconds.
  This synthetic smoke caps dispatch at four replies and 769 output tokens,
  with no retries; any actor smoke is separate effort.
- Check seeded A replay, natural completion, token exhaustion, native usage and
  rendered prompts. For this smoke alone, accept raw Python or exactly one complete,
  unambiguous Python fence with optional surrounding prose, following the benchmark
  extraction finding. Reject incomplete replies and multiple/ambiguous blocks.
  Retain the extracted bytes and verify the three coding replies together through
  the existing `verification.verify_many` against their fabricated cases. Retain
  malformed or failed replies as failures; the study extractor belongs to Batch 6.

Save the smoke's exact requests, replies, verification logs and concise outcome
under `data/03_generation_checks`. Report loading and request time separately.
All three coding replies must stop naturally, parse and verify for the direct
runtime to pass this smoke. Report every draw, including any failures. Unrun live
checks leave current runtime compatibility pending; mocked tests alone do not
complete that acceptance step.

## Implementation validation

The 160-test repository suite passed with eight optional skips. The live
[actor smoke](../../data/03_generation_checks/direct-actor-72mxh00r/serial/outcome.json)
passed with request-owned sampling. The
[synthetic smoke](../../data/03_generation_checks/direct-synthetic-m_hj10v7/serial/outcome.json)
used four requests and 67 output tokens; both tokenizer histories matched the
official snapshot, A replay matched, and all three programs passed Docker
[verification](../../data/03_generation_checks/direct-synthetic-m_hj10v7/verification.json).
An initial verification-launcher symlink error was fixed; the remaining assertions
and verification passed through [saved-reply recovery](../../data/03_generation_checks/direct-synthetic-m_hj10v7/recovery.json),
with zero additional inference requests. No cohort tasks were used.

## Later contract change — 2026-10-26

Batch 6 removed generation wall-time limits at the user's request. Current serial
and batched calls record elapsed time and use output-token limits; legacy
`wall_time_seconds` fields are ignored. The deadline contract and smoke budget
above describe Batch 5 and its historical evidence.

## Review boundary

Target roughly 300–400 additions plus deletions across the existing module, its
tests and documentation, including this plan; the hard batch limit remains 500.
The production change should be only a few dozen lines. Add no filler to reach
the estimate. Link this plan from the Stage 03 plan.
No new dependencies, adapter hierarchy, profile registry, worker protocol,
journal, recovery engine, automatic fallback or scheduler is needed. Keep archived
benchmark orchestration out of main. The [remaining inference plan][remaining]
assigns native batching and overlap to the consuming workflow. Measured
batch-8 and batch-32 short-context settings are future starting points,
not qualification of this serial path; Ministral 7k/batch 32 already failed with
OOM. Production extraction, repair and accounting remain in Batches 6–7; prefix
reuse and BF16 yield checks retain their later gates.

Implement on `main` and stop for review after this batch. This planning change
runs no models or generated programs and authorizes no acquisition run.

[findings]: ../inference-benchmark-results.md
[remaining]: ../inference-benchmark-plan.md
[loader]: https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/utils.py
