# DeepInfra generation worker

Install `requirements-deepinfra.txt` in a separate Python 3.12+ environment.
Export `DEEPINFRA_API_KEY` in the environment launching the worker. For the
coordinator route below, export it where the coordinator runs; its child inherits
that environment. The worker does not read `.env`, download weights or load a GPU
runtime. Run:

```sh
python scripts/deepinfra_generation.py /absolute/path/to/round.json
```

The JSON has exactly `runtime`, `requests` and `output_dir` (a new directory).
`runtime` contains:

| Field | Meaning |
| --- | --- |
| `model` | Pinned Qwen actor or [Gemma synthetic candidate](deepinfra-gemma.md) |
| `tokenizer_path`, `tokenizer_sha256` | Local tokenizer directory and filename→SHA256 bindings |
| `timeout_seconds`, `max_total_tokens` | HTTP socket timeout and prompt+reply envelope |
| `input_usd_per_million`, `output_usd_per_million` | Explicit uncached rates for planned-work accounting |
| `max_cost_usd` | Planned ceiling for the entire finite round |
| `concurrency` (optional) | Maximum outstanding requests, integer 1–200; default 1 |
| `output_protocol` (optional) | `token_ids` by default; explicit `gemma_native_text` permits Gemma's text-only replies |

Pin `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`
and any present `vocab.json`, `merges.txt`, `special_tokens_map.json`,
`added_tokens.json`. Other files/weights in that directory are not loaded.
Record rates applicable to the intended run; prices are not fetched or hardcoded.
The ceiling reserves every prompt plus its maximum reply before any dispatch,
ignores cache discounts and never refunds uncertain work. It is a planned-work
bound at supplied rates, not a provider-enforced billing limit.

Each request requires `messages`, `seed`, `temperature`, `top_p`, `max_tokens`.
Optional fields are `max_prompt_tokens`, `top_k`, `enable_thinking`,
`reasoning_effort`, `tools`, `preserve_thinking`. Unknown fields are rejected.
Installed tokenizer package versions must exactly match `requirements-deepinfra.txt`
before tokenizer loading. All requests are rendered and checked before the first HTTP call. Prompts use
local template/token IDs, raw completions, native stop IDs and preserved markers.

`timeout_seconds` limits socket operations, not total request duration. A
non-streaming call can remain silent through queueing, prefill and the entire
capped generation. Set it above the expected worst-case time for those phases,
including thinking and a queueing margin. A timeout does not cancel remote work.

The [DeepInfra schema](https://docs.deepinfra.com/api-reference/text-completions/openai-completions)
documents `top_k=0` as disabled and a seed range covering the worker's nonnegative
64-bit subset. These remain unchanged. The real request/response fixture in
[`tests/fixtures/deepinfra/`](../../tests/fixtures/deepinfra/) was captured on
17 November 2026 from this pinned model using token-array input. It records
`return_tokens_as_token_ids=true`, `top_k=0`, returned `token_ids`, estimated cost
and a trailing EOS ID omitted from response text. No extra flag is inferred from
another serving framework; the original captures remain in ignored
`data/monitor_training/deepinfra-compat-20261117-01/`.

The worker writes `inputs.json`, `runtime.json`, atomic `outcome.json`, numbered
request/response JSON files and completed reply text files. Responses retain their
raw JSON body, HTTP status, timing, token usage and provider-estimated cost.
Credentials/headers are excluded; the key is redacted from persisted content.
Output rows retain existing `status`, `submitted`, text/token IDs, thought fields
and `answer_text`; `native` is null rather than fabricated GPU counters.

Technical failure stops new admission without retries. Before preflight, the worker
persists one `not_run` / `submitted=False` row per request. Missing credentials,
package/tokenizer failures and overlong prompts leave complete zero usage and no
submissions. Malformed JSON, an unusable request list, existing destinations or an
unwritable outcome cannot establish that evidence. A timeout, HTTP error or
interruption during dispatch leaves the in-flight submission unknown; later requests stay not run.
Completed peers and their usage survive. HTTP 200 confirms submission even when
its body is invalid. Validated usage survives output-validation errors; complete
usage does not imply a successful round. Killing the worker cannot cancel remote
generation. Exit status is zero only for a successful round; truncation/unclosed
thinking remains subject to existing extraction rules.
Metadata-write failures also produce an error outcome when the filesystem still
permits it, with no submissions. Every attempt requires a new output directory;
existing evidence is never overwritten or resumed automatically. HTTP errors,
including authentication/validation failures, retain conservative unknown usage;
that state does not assert a charge or trigger a retry.

## Coordinator integration — second review batch

Save the worker's `runtime` object as its own JSON file. The existing coordinator
can then run an actor worklist through generation, verification and repair:

```python
import importlib
loop = importlib.import_module('src.data-pipeline.stage03.generation_loop')
result = loop.run(
    worklist, output_dir, source='actor', recipe=recipe,
    provider_runtime='/absolute/path/to/provider-runtime.json',
    generation_python='/absolute/path/to/deepinfra-venv/bin/python',
    reply_allowance=reply_allowance, output_token_allowance=output_token_allowance,
    cohort_sha256=cohort_sha256,
)
```

Existing extraction/checker/repair callbacks work unchanged, including native tool
history. Do not supply local model/manifest/MLX paths, local batching or a driver
override on this route. The DeepInfra worker is fixed; its pinned model must match
the actor/synthetic source. The coordinator does not need tokenizer packages:
the explicit child interpreter loads them. Relative tokenizer paths resolve from
the coordinator's working directory and are frozen as canonical absolute paths.

Before creating output or launching a child, the coordinator requires
`(R * L * input_rate + T * output_rate) / 1e6 <= max_cost_usd`, where `R` is the
reply allowance, `L` is `max_total_tokens`, and `T` is the output-token allowance.
This intentionally conservative reservation covers the entire finite run,
including every repair; it is not renewed per round or reduced after short replies.
Actual requests cannot exceed those frozen allowances, including custom repairs.

`inputs.json` keeps a separate `provider_runtime` record with the runtime file's
path/hash, normalized settings, whole-run bound and `status=configured_candidate`.
It contains no fictitious local model manifest. Runtime/tokenizer/source/requirements
hashes are checked before each round, before reservations or child dispatch; the
worker also verifies tokenizer files. Completed peers survive a later pin failure.

An overlong repair is an explicit technical stop for the whole round; it is not
silently shortened or made ineligible. Peers in that unsent round stay not run,
previous completed rounds remain intact, and coordinator reservations remain
committed. Choose and qualify the prompt/feedback/output envelope before live
acquisition; changing repair eligibility or selection is separate review work.

This configuration binding does not qualify a generation recipe. Historical
acquisition drivers remain unchanged. Provider qualification, recipe selection
and shared-parent acquisition remain later review work. Served weight
revision, quantization and deterministic replay are unverified. Validation uses
offline tests and saved responses; no fresh acquisition is launched by this batch.

## Bounded concurrency — third review batch

Set `concurrency` explicitly to opt into parallel HTTP calls. Old runtime files
retain serial behavior. The worker preflights the entire round before dispatch,
then admits a stable prefix with at most that many unsettled requests and refills
available slots. Output rows, filenames, seeds and task associations retain their
original positions even when responses arrive out of order. Only HTTP calls use
threads; decoding and evidence writes remain on the owning thread.

Every submission is persisted as unknown before its HTTP task can start. All ready
responses are collected before refilling. The first observed transport, response
validation or write failure stops admission. Queued tasks are cancelled where
possible; proven cancellations remain not run. Running peers are drained and their
responses/usage retained. Their completion does not turn the failed round into a
success. Per-row errors supplement the first overall error.

A graceful interrupt also stops admission and drains registered HTTP tasks; it may
wait for their socket operations. An interruption during task submission can leave
an unregistered response uncertain; its reservation remains committed. Threads cannot cancel remote generation or
enforce a hard total deadline. The coordinator's existing second-interrupt kill
can terminate the child, leaving its persisted in-flight rows uncertain.
Standalone invocations do not escalate repeated Ctrl-C: they keep draining. Use
SIGTERM or SIGKILL to terminate a standalone process immediately; remote requests
may continue. Only the coordinator path provides the second-interrupt escalation.
`max_observed_in_flight` records the maximum unsettled task count, not GPU batch size.
The 200-request client ceiling is not an optimal setting or a guarantee against
provider token/rate limits. A 429 stops admission without retry or automatic tuning.
The same finite work and spending reservations apply at every concurrency.
The [integration check](deepinfra-qualification.md) records offline replay results
and the approved bounded live smoke results.
Keep the default concurrency of 1 until a bounded qualification measures a higher
setting against request and token limits. Any failed request makes the coordinator
stop the whole round and run: successful peer replies remain saved as evidence,
but are not verified or admitted, and no further repair round runs. Accepting
partial rounds would require a separate acquisition-policy change.
