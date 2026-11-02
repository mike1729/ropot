# Qwen3.8 actor and monitor migration — 29 October 2026

This records the checkpoint transition at that date. Runtime wiring and the data
campaign subsequently completed; see the [qualification record](../frozen-corpus-qualification.md)
for accepted results and the recipe seal. Pending-work statements below preserve
the migration's original scope.

Qwen3.8-27B MLX 8-bit replaces Qwen3.6-27B as both the actor and monitor.
Gemma-4-31B-it remains the synthetic source. This records the selected model and
required runtime transition; it does not claim that all callers have been rewired
or that Qwen3.8 generation and monitoring have already been qualified.

## Verified local checkpoint

- Repository: [`mlx-community/Qwen3.8-27B-8bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit).
- Revision: `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`.
- Existing local directory: `/Users/michal/Models/mlx/Qwen3.8-27B-8bit`.
- Manifest: [`model-manifest.json`](../../data/03_generation_checks/qwen38-monitor-20261029/model-manifest.json).
- Manifest SHA-256: `188f70ff7d6ec54277eed0e573ae0a33329968f891a931afbe9987f09c9e799f`.
- Verified 19 files, 29,531,521,322 bytes; weight shards match the pinned
  repository's published LFS SHA-256 hashes. A complete local copy was reused.

Qwen3.6 weight shards were removed at the user's request: 13 files across the
27B 8-bit, 27B 4-bit and 35B-A3B 4-bit directories, freeing approximately 61.43 GiB.
Small configuration/tokenizer files, manifests and historical run artifacts remain.
The [removal record](../../data/03_generation_checks/qwen38-monitor-20261029/removed-qwen36-weights.json)
lists the exact files. Those historical paths no longer identify runnable weights.

## Thinking and sampling

The [official Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B)
and the downloaded chat template support `reasoning_effort` values `low`, `medium`
and `xhigh`; the model default is `xhigh`. Use explicit **thinking on, effort low**
as the initial development recipe for both attacks and monitoring. Keep thinking
**off** for benign acquisition and repairs, as required by B1.

The effort level adds model instructions; it is not a hard thinking-token budget.
Record it in rendered prompts and request evidence. Preserve raw replies separately
and extract programs or parse monitor verdicts only from the final-answer channel.
Do not treat a verdict inside unclosed reasoning as an answer.

The model card recommends temperature 1.0, top-p 0.95 and top-k 20 for thinking.
Record these as a new recipe when wired; historical Qwen3.6 sampling is not
implicitly inherited. Monitor seed remains 17 for every request. Keep prompt,
normalization and verdict semantics comparable while recording the changed model,
thinking mode and sampling as a distinct monitor revision.

## Provenance and remaining work

- Update active model bindings, generation-effort passthrough and monitor parsing;
  add a separate Qwen3.8 configuration rather than editing the historical
  `configs/monitor/qwen36-v2-baseline.json` in place.
- Verify effective effort instructions with the real tokenizer and check that the
  local runtime loads the checkpoint. Measure throughput before claiming a speedup;
  similar size, architecture and precision suggest similar decoding cost.
- Reuse the eight development task identities and shown-case partitions. Qwen3.6
  B1 benigns may seed labelled cross-checkpoint exploratory rounds, but fresh
  Qwen3.8 benigns are required before official own-source attack acquisition.
- Retain all Qwen3.6 costs, truncations and outcomes. Historical batching findings,
  manifests and results keep their original model labels. Requalify Qwen3.8
  monitor behavior on development data; confirmation outcomes remain unopened.

The Qwen3.6 development run `qwen-dev-20261029-01` was cancelled. Its first four
replies each exhausted 8,192 tokens without a final answer; diagnostic drafts
recovered from reasoning are not accepted submissions or Qwen3.8 results.
