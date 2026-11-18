# Reproduce the Gemma controls

Run `python scripts/deepinfra_live_controls.py NEW_OUTPUT` for offline replay.
This reads ten committed raw captures, checks their provenance hashes and parses
JSON or streaming SSE usage. It reads no key, sends no requests and runs no Docker
containers. Reported receipt costs belong to the historical calls; new cost is zero.

Live calls require `--live --input-rate RATE --output-rate RATE --max-cost USD`.
Rates are dollars per million uncached input/output tokens. Export
`DEEPINFRA_API_KEY` in the coordinator environment. Use `--cases thinking echo`
to restrict the selection; omission selects all ten. Every selected call is
reserved before dispatch, with a full 262,144-token conservative input reservation
for the raw-string control. `--timeout` is a socket timeout, defaulting to 300 seconds.

Selecting `native-pipeline` also requires `--runtime RUNTIME.json` and
`--generation-python TOKENIZER_ENV/bin/python`. The runtime must pin the original
Gemma tokenizer and select `gemma_native_text`. Offline rendering must reproduce
the captured payload before any live call. This case runs the original toy inputs
through the actual generation coordinator, native edit extractor and Docker
correctness verifier. Success requires the intended `print(2)` candidate to be
rejected for the task requiring `print(1)`. It admits nothing into a corpus.

The model version, routes, raw prompts and protocol flags remain frozen. Live
responses may differ. The runner refuses redirects, never retries, preserves raw
JSON/SSE bodies, redacts the key from saved files and stops after an error or
unknown submission. A timeout may still be billed; incomplete usage remains
explicit. Provider-reported cost is evidence rather than a billing guarantee.
