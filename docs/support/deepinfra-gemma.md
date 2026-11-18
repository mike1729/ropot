# Gemma DeepInfra backend

The synthetic route supports the configured candidate
`google/gemma-4-31B-it:wcache-3`. Qwen remains actor-only and Gemma synthetic-only;
mixing model and source is rejected before output creation or dispatch.
HTTP transport, concurrency, budgets and failure collection are shared.

Use the [existing runtime format](deepinfra-worker.md), changing `model` and the
local tokenizer directory/hashes. Call `generation_loop.run` with
`source='synthetic'`, `provider_runtime=<JSON path>` and the explicit
`generation_python`. Local MLX routes remain available.
Set `output_protocol="gemma_native_text"` explicitly for the hosted Gemma model.
The default `token_ids` protocol remains strict; Qwen rejects the Gemma text mode.

The [provider's standard model](https://deepinfra.com/google/gemma-4-31B-it)
reports FP8. This is a distinct numerical backend from the historical local Q8
checkpoint; matching weights/revision or exact numerical equivalence is unverified.
The public deployment version was read without authentication or generation.

The pinned local Gemma-4-31B-it tokenizer uses:

| Meaning | Token IDs |
| --- | --- |
| EOS, turn end, tool-response opening | `1`, `106`, `50` |
| Thought opening `<\|channel>thought` | `100, 45518` |
| Thought closing `<channel\|>` | `101` |
| Native tool-call opening / closing | `48` / `49` |

Token 49 closes a native call and stays in decoded text. It is not added to the
stop list. A following stop token 50 is removed before native-call parsing.
The adapter reads Gemma's top-level EOS configuration, supports multi-token
thought boundaries and exposes native token conversion to the shared decoder.

Gemma supports boolean `enable_thinking`. Explicit `reasoning_effort` is rejected
because the pinned template does not implement it. Omit `preserve_thinking`, or
use `True` for the template's native tool-history policy: reasoning attached to
assistant tool calls after the last user turn is retained. This is not a Qwen-style
arbitrary-history toggle; `preserve_thinking=False` is rejected rather than ignored.

## Live qualification status

The [smoke report](experiment-reports/deepinfra-gemma-smoke-20261118.json) records
two authorized controls: thinking output and native repair history. Both returned
HTTP 200 with text and complete billing fields, costing an estimated $0.0000986.
However, both returned `token_ids=[]`. The strict token-trace gate rejected them.
These responses cannot pass the original-token protocol. The approved explicit
native-text protocol supports this provider without inventing a token trace.

The native control's text parsed and produced `print(2)` from `print(1)`; Docker
checked that toy result. This text-only diagnostic does not replace token-trace
qualification. No generated IDs were inferred by re-tokenizing response text.

Valid provider usage is retained independently of token/response validation. The
initial worker outcome is preserved unchanged; the report audits its known charge
from raw responses and the billing fix is covered by regression tests.

Subsequently authorized small controls tested `return_token_ids=true`,
`logprobs=1`, both together, echo, streaming, the documented endpoint alias and
raw-string input. All returned HTTP 200 without IDs or logprobs.
[vLLM documents these as distinct controls](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completion/protocol/),
while DeepInfra's public schema exposes only the latter token-string flag.
Those nine tests together reported $0.0001574. No acquisition retries or corpus
admission occurred. The [final live pipeline check](experiment-reports/deepinfra-gemma-native-text-20261118.json)
used the real synthetic coordinator, native extractor and Docker verifier. It
successfully extracted the requested toy edit and correctly rejected its wrong
output. The report includes the combined cost of all ten tests, below $0.01.
The [control rerunner](deepinfra-live-controls.md) includes all ten exact requests
and raw receipts in a repository fixture. It replays offline by default and offers
explicit live transport and native-pipeline modes with a finite spending ceiling.

## Explicit native-text protocol

Raw provider text and reported total token usage are preserved. `tokens` is null,
`token_ids_source` is `unavailable`, and per-channel token counts are null. No
generated text is re-tokenized to manufacture original IDs or reasoning counts.
If complete original IDs are supplied, the existing strict token path still
validates and retains them.

The parser accepts an optional leading `<|channel>thought` block only with its
exact header boundary and a unique closing `<channel|>`. It also handles a known
open thought in the locally rendered prompt. Unclosed, repeated, stray or
ambiguous channel markers leave no usable answer. Empty thought wrappers are
accepted with thinking disabled; nonempty thoughts in that condition are rejected.
Native tool delimiters stay in the answer. Existing syntax, application, witness
and sandbox checks still decide candidate eligibility.

`thought_metrics.status` is `unavailable`; its separate `framing` field records
closed, absent or invalid framing. Native tool histories retain observed reasoning
text, while cumulative output budgets use the provider's reported total. This
backend cannot supply exact per-channel token metrics for the thinking study.
Model/recipe quality and the redesigned acquisition workflow still need their own
qualification; these control checks admit nothing to training.
