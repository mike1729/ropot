# Finite DeepInfra experiment launcher

`scripts/launch_deepinfra.py` connects frozen experiment requests to the existing
worker. `prepare` renders all rounds locally with the pinned CPU tokenizer before
writing a launch manifest. It makes no HTTP calls and removes the API key from
the tokenizer subprocess environment. `launch` explicitly dispatches the reviewed
manifest. The launcher changes neither prompts, seeds nor sampling parameters.

The input plan uses the thinking probe's existing `rounds` object:
`{family: {model, requests}}`, with family `qwen` or `gemma`. Each request is an
existing worker request without `max_tokens`; supply a finite cap per family.
The plan can retain its task data and input hashes. Supply the exact plan digest,
one [runtime JSON](deepinfra-worker.md) per family, and the generation interpreter.
Gemma requires the explicit native-text protocol. A future reference driver can
reuse this request format; reference repairs and first-qualified selection still
belong to `generation_loop.run(provider_runtime=...)` after settings are frozen.
Historical acquisition drivers and their local qualification contracts are unchanged.

Also provide a reviewed limits JSON with one entry per family:

```json
{
  "gemma": {
    "model": "google/gemma-4-31B-it:wcache-3",
    "max_output_tokens": 1024,
    "max_total_tokens": 4096,
    "evidence": {"/absolute/path/to/provider-limit-evidence.json": "SHA256"}
  }
}
```

These example numbers are not a declaration of the deployment's maximum. Evidence
must establish the accepted reply and context limits for the intended deployment;
review their meaning before preparation. The launcher checks the stated envelope
and pins its evidence; it does not infer limits from model context length or query
the provider. Runtime rates must conservatively cover the intended service tier.

```sh
.venv/bin/python scripts/launch_deepinfra.py prepare /path/to/probe/plan.json /new/job \
  --plan-sha256 PLAN_SHA --generation-python /path/to/deepinfra-venv/bin/python \
  --qwen-runtime /path/to/qwen.json --gemma-runtime /path/to/gemma.json \
  --qwen-max-tokens QWEN_LIMIT --gemma-max-tokens GEMMA_LIMIT \
  --limits /path/to/limits.json --max-cost-usd TOTAL_CEILING
```

Preparation validates both families, their context envelopes and the sum of their
whole-round reservations before creating output. `launch.json` binds the plan,
runtime files, tokenizers, source, requirements and limit evidence. It retains
package versions, prompt lengths and rendered payload hashes. The command prints
the manifest digest and total reservation for review. Existing destinations fail.

After reviewing the prepared job, export the key in the launcher environment:

```sh
.venv/bin/python scripts/launch_deepinfra.py launch /new/job --sha256 LAUNCH_SHA
```

Launch rechecks every pin and both rendered rounds before the first worker starts.
It processes families in plan order and stops before later families on a technical
failure. `outcome.json` retains known charges, submission lower bounds and separate
usage completeness; unresolved work is never recorded as free. Worker directories
and logs stay intact. A permanent exclusive `launch.claim` prevents concurrent or
repeated launches from overwriting accounting. Retry and resume are refused. The existing
child process handler provides graceful interruption and second-interrupt killing.

For the thinking probe, pass `/new/job/qwen` and `/new/job/gemma` to the existing
`whole_program_thinking_probe.py report` command. That report owns full-program
extraction, comment stripping, Docker certification and blind review packets.
An undispatched family must be omitted from that command. Its
`provider_ceiling_verified=false` remains a reminder that provider-limit evidence
requires review; the launch manifest preserves the actual checked envelope.
Preparation and tests do not authorize or start a paid probe. Choose settings
before preparing the reference acquisition contract.

The [offline integration check](experiment-reports/deepinfra-launcher-offline-20261118.json)
records real CPU rendering with both pinned tokenizers and zero API calls.
