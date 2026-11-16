# T3: LoRA training adapter

Implemented and tested with tiny CPU models. NVIDIA checkpoint compatibility,
memory fit and live smoke remain unqualified. No GPU rental or production
training was launched. T4/T5 own S training and matched twin selection.

Smoke loads `shared_honest` directly through T1's pinned prepared-view reader.
`training.py` reuses T2's validated native prefix/continuations.
Only label and terminator tokens are supervised; prefix and padding labels are
`-100`. Rows remain separate. Each microbatch contributes its row mean NLL
divided by four, including a final partial accumulation window. Trainer's
`loss_is_scaled_for_ga=True` prevents a second division. The CPU tests compare
actual full/partial accumulated gradients against an independent reference.

The checked-in candidate uses rank 16, alpha 32, standard LoRA scaling, zero dropout,
Kaiming A/zero B initialization, FP32 adapters/AdamW moments, learning rate
`1e-4`, constant schedule, zero warmup/weight decay, microbatch one and
accumulation four. The original int8/BF16 base stays frozen. PEFT's usual
`prepare_model_for_kbit_training` is omitted because it casts BF16 base weights
to FP32. Activation checkpointing, when selected, is non-reentrant. Full text
projection names must be reviewed against the chosen architecture and listed
explicitly; no generic target-module suffix recipe is assumed.

## Run smoke

On the NVIDIA host, first freeze the [T2 inference recipe](monitor-training-t2.md).
Add PEFT and copy [lora-smoke.example.json](../configs/monitor_training/lora-smoke.example.json)
to `TRAINING_SETTINGS.json`. Replace the placeholder with **every** selected
text projection's full name and set the chosen tolerances and numeric budgets.
Set `max_grad_norm` to zero to disable clipping, or a positive finite number.
The example is not a qualified production recipe or an approved live budget.

The `lora` and `training_args` objects supply `LoraConfig` and `TrainingArguments`
keyword arguments directly. The runner supplies the output directory, update
limit, sequential sampling and per-update checkpointing for `TrainingArguments`.
These protocol settings keep the length probe first and checkpoint 2 available.
Smoke reads
the settings once and saves the effective configuration in its run directory's
`training.json`, alongside the PEFT version and training source hashes. Dataset
and numerical-base pins remain required:

```sh
python -m src.monitor_training smoke --prepared-root PREPARED --export-root EXPORT --model-root MODEL --inference-contract RECIPE/contract.json --settings TRAINING_SETTINGS.json --expected-prepared-sha256 PREPARED_SHA --expected-inference-sha256 INFERENCE_SHA --output-dir data/monitor_training/NEW_SMOKE
```

Smoke opens only shared-honest rows. Its first example repeats a masked public
construction token to exercise exactly the supported total sequence length;
this mechanical resource fixture supplies no behavioral measurement. Three
optimizer updates run continuously. A fresh model reloads checkpoint 2 and
resumes one additional update, comparing the resulting adapter weights with
continuous checkpoint 3 under the declared tolerance. Reload also compares
adapter weights exactly with a step-2 CPU snapshot before Trainer resumes, plus
generated token IDs on one construction row. Both replies are flushed to the
event log immediately.
The trainable-set audit restricts updates to the configured LoRA A/B parameters.
Step-2 and step-3 adapter snapshots verify that trainable parameters changed.
Construction-fit measurements belong to the T4/T5 training trajectories.

Trainer checkpoints save model/optimizer/scheduler/RNG/data-position state.
New attempt directories retain flushed events, input pins,
trainable names/counts/dtypes, row order/token hashes, reload replies, peak GPU
memory, minimum headroom, tokens, updates, elapsed time and estimated cost.
Row presentations and supervised tokens come from actual training batches,
including after resume. Token accounting combines training input positions
(including padding) with generation input and output tokens. Each generation
reserves the full output cap, then reconciles unused tokens after completion.
Memory and headroom are checked after optimizer steps and at final measurement;
running CUDA peaks cover backward allocations. Final runtime measurements are
written once to `runtime.json`, with one final synchronization for wall-time
accounting. Token/time/cost checks surround calls and updates.
A token budget smaller than the length probe is rejected before model loading.
One operation can overrun a ceiling; failures leave an incomplete attempt and
preserve the original error if final measurements also fail. Smoke costs exclude
provider billing outside this process. **Discard smoke weights:** production S
must start from the original numerical base and a fresh adapter.

The real CPU integration tests use PyTorch 2.15.0, Transformers 5.23.0, PEFT
0.21.3 and Accelerate 1.15.0 without downloads. They exercise actual Trainer
gradients, padding, RNG/data-position resume and independent branch objects.
These versions do not establish NVIDIA/Qwen compatibility. T2's runtime freeze
records the base environment in `requirements-training.txt`; the smoke snapshot
also records the selected PEFT version.
