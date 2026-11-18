# DeepInfra serial worker — first review batch

Install `requirements-deepinfra.txt` in a separate Python 3.12+ environment.
Export `DEEPINFRA_API_KEY` from the private configuration into that environment;
the worker does not read `.env`, download weights or load a GPU runtime. Run:

```sh
python scripts/deepinfra_generation.py /absolute/path/to/round.json
```

The JSON has exactly `runtime`, `requests` and `output_dir` (a new directory).
`runtime` contains:

| Field | Meaning |
| --- | --- |
| `model` | `Qwen/Qwen3.8-27B:61nNTQpO` |
| `tokenizer_path`, `tokenizer_sha256` | Local tokenizer directory and filename→SHA256 bindings |
| `timeout_seconds`, `max_total_tokens` | HTTP socket timeout and prompt+reply envelope |
| `input_usd_per_million`, `output_usd_per_million` | Explicit uncached rates for planned-work accounting |
| `max_cost_usd` | Planned ceiling for the entire finite round |

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
All requests are rendered and checked before the first HTTP call. Prompts use
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

Technical failure stops the round without retries. A timeout, HTTP error or
interruption leaves the in-flight submission unknown; later requests stay not run.
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

This worker is not yet wired into acquisition: the coordinator still requires
local runtime manifests. Provider qualification, concurrency, recipe selection and
shared-parent acquisition belong to later review batches. Served weight revision,
quantization and deterministic replay remain unverified. This batch uses offline
tests and saved compatibility responses; it authorizes no fresh acquisition.
