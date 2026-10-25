# Batch 4: actor runtime implementation plan

Establish warm serial generation for the local Qwen3.5-9B actor, following the
[Stage 03 plan](stage-03-plan.md). Load the model once per finite run.

Use **8-bit quantization** for this batch. Benchmarks and batching come later;
check actual BF16 yield only after the full witness/backdoor pipeline works,
as specified in the [overview](data-pipeline-overview.md).

## Implementation

1. Add direct MLX calls in `stage03/local_generation.py`, run from the existing
   MLX Python. Accept a local model path, finite ordered request list, model
   artifact manifest and new output directory. Each request specifies messages,
   seed, temperature, top-p, output-token limit and soft wall-time limit. The caller
   constructs these arguments; avoid a schema-validation layer.
2. Reuse the existing pinned MLX environment (`mlx-lm==0.32.2`, `mlx==0.32.4`),
   including its [documented local patches][patches]. Load the model/tokenizer once
   with `mlx_lm.load`, then call `stream_generate` serially. Reseed before each
   request and construct its sampler. Let the library create fresh decoding state
   for each call; carry no conversation or prompt cache between requests.
3. Apply the model's chat template with `enable_thinking=False` and
   `add_generation_prompt=True`. Inspect the rendered prompt and returned text
   to check thinking behavior. Preserve replies verbatim.
4. Check each request's soft deadline before generation and between returned
   chunks. It cannot forcibly interrupt model loading, a stalled native call or
   cleanup. On an observed timeout or technical failure, stop the run without
   starting another request. Close the native generator and preserve records on
   Python interruption, then re-raise. Hard process termination can lose metadata.

Record model/runtime identity once per run, including the model artifact manifest,
quantization, tokenizer/template and relevant runtime patches. For each request,
save its inputs/settings, raw reply or partial output, completion status, native
token usage and elapsed time. On failure, preserve completed results and identify
interrupted or unstarted requests. Do not retry automatically. Simple text/JSON
files are sufficient.

## Validation

Use mocked MLX calls to check:

- One model load serves multiple serial requests, with seed and decoding state reset.
- Messages, template options, sampler settings and token limits reach the runtime.
- Completed, token-limited and failed requests remain distinguishable, with raw
  output preserved and the generator closed after soft timeout or interruption.

An opt-in live smoke uses fabricated prompts: a short reply, a stdin/stdout coding
request, forced token exhaustion and an A/B/A sequence to check state carryover.
Inspect thinking, completion and native usage. Existing [actor controls][controls]
are useful examples; check this warm serial path itself. An unrun smoke leaves
actual runtime compatibility unvalidated.

Keep this batch to generation and basic records. Code extraction, verification,
repair and acquisition accounting belong to later batches. Keep response caching
off when collecting new draws. Stop after implementation for review; this plan
does not run models.

[patches]: ../../honeypot/configs/runtime/mlx-request-seeds.json
[controls]: ../data/03_generation_checks/actor-controls-1pyga9fv/summary.json
