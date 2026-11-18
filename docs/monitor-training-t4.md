# T4: shared honest training

The `train` command implements shared honest S training and development selection.
No GPU rental or production training was launched. Live use still requires a
qualified [T2 numerical base](monitor-training-t2.md), a passing
[T3 smoke](monitor-training-t3.md) for the same recipe, a reviewed prepared export
and numeric resource budgets. These are operator prerequisites; this command
does not ingest prior qualification reports or authorize a launch.

Training opens only `shared_honest` and `development`. It starts from the original
numerical base with a fresh adapter, using all 18 shared-honest rows in prepared
order on every pass. Benign rows target PASS; both attack roles target FLAG.
The existing Trainer loss retains the nominal divisor four for the final two-row
update. One continuous trajectory saves complete Trainer checkpoints after 1, 4
and 16 passes: 5, 20 and 80 updates, with 18, 72 and 288 row presentations.
Optimizer state continues across candidates; no candidate restarts training.

At each candidate, score construction fit separately from all 60 development
rows. Records retain unconstrained greedy replies and complete PASS/FLAG
continuation likelihoods. Training loss and construction fit are diagnostics;
selection uses only the [honest development gates](monitor-training-plan.md#4-train-s-then-a-matched-honestselective-pair).
Choose the first candidate meeting every gate, then stop. If none passes, retain
the failed attempt without selecting S. Infrastructure failures are incomplete
attempts, not failed behavioral scores. No automatic retry or recipe revision runs.

## Run

Use the same explicit LoRA/training settings qualified by smoke, with microbatch
one, accumulation four and `save_only_model=false`.
Checkpoint pruning and Trainer best-model selection are disabled. Add a positive
finite `budgets.max_fit_tokens` for construction scoring and review every ceiling
for the longer trajectory. `budgets.max_tokens` bounds all training, development,
fit and reload tokens together; `max_fit_tokens` independently bounds fit calls.
Before loading weights, preparation requires both ceilings to cover the full
16-pass trajectory, all three fit/development snapshots and one selected-adapter
reload. Generation reserves the full output cap; actual usage can be lower.
`rows.json` records these computed `token_requirements`, including on rejection.
All scoring uses the inference contract's frozen rendering, output cap and generation settings.
Likelihood scoring calls the base model with its injected LoRA layers, bypassing
Trainer's mixed-precision forward wrapper while leaving training unchanged.
Wall time, estimated cost and memory headroom remain bounded by training settings.
The checked-in smoke example does not supply an approved production budget.
Settings that discard resume state or enable best-model selection are rejected
before loading weights.

Launch one Python process with one visible GPU; distributed launch is unsupported.
The runner checks Trainer's actual microbatch size before any optimizer updates.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python -m src.monitor_training train --prepared-root PREPARED --export-root EXPORT --model-root MODEL --inference-contract RECIPE/contract.json --settings TRAINING_SETTINGS.json --expected-prepared-sha256 PREPARED_SHA --expected-inference-sha256 INFERENCE_SHA --output-dir data/monitor_training/NEW_S
```

The destination must be new. It retains input/base/config/source pins, trainable
names/counts/dtypes, prepared row order, actual batch/update accounting, flushed
events and resource measurements. Candidate checkpoints live under
`continuous/checkpoint-5`, `checkpoint-20` and `checkpoint-80`; only reached
candidates are created. Each includes adapter weights and optimizer, scheduler,
RNG and Trainer data-position state. Raw candidate scores live in
`pass-1/fit.json`, `pass-1/development.json` and the corresponding later-pass
directories. Each pass retains a `summary.json`, including partial diagnostics
on failure. The run's `summary.json` lists fully scored candidates and the selected
checkpoint reference and file hashes, if any. A fresh base reloads the selected
adapter and verifies inference before the attempt can pass. Exact weight
save/load comparison belongs to the prerequisite T3 smoke.
Smoke and T4 share the resource checks, batch accounting and gradient audit
in `runner.py`.
Failure records retain tracebacks, including chained exceptions. CUDA memory
measurements also survive model-loading failures before a runtime is returned.

Budget checks precede model calls and follow complete scoring passes, updates
and final runtime measurements; an individual operation can
overrun a ceiling and make the attempt incomplete. Estimated GPU cost excludes
provider billing outside the process. This stage produces a development-selected
S checkpoint only. Matched twins belong to T5; independent confirmation and the
repair handoff remain subject to the allocation policy in the training plan.
