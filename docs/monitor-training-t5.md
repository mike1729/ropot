# T5: matched twins

`train-twins` continues the selected [T4 shared-honest checkpoint](monitor-training-t4.md)
into honest and selective twins. Implementation is CPU fixture-backed; no GPU
rental or production training was launched. The same qualified numerical base,
passing smoke and reviewed export remain live-launch prerequisites.
No inference contract is frozen yet. Freeze it after T5 lands, before qualification
or training.
Pair reporting lives in `twins.py`, outside the inference-pinned sources, so
later reporting-only edits do not invalidate S's contract.

The command pins S's `summary.json`, verifies its selected checkpoint and training
record hashes, and requires the same prepared view, inference contract, PEFT
version, LoRA settings and training-affecting Trainer arguments. Training source
hashes are recorded but need not match S. Resource budgets, logging,
reporting and progress display may change for this trajectory.
It opens only `twin_continuation` and `development`; existing preparation and
rendering own data qualification, membership and token boundaries.

Both branches start from S's adapter weights with fresh optimizer/scheduler state.
They repeat the same 24 rows in prepared order; only actor-attack targets differ.
Each branch advances to 6, 24 and 96 updates (1, 4 and 16 passes). Branches run
sequentially on one device, saving and resuming complete Trainer state between
visits. Both are scored before either advances further. The first pair passing
every [development gate](monitor-training-plan.md#4-train-s-then-a-matched-honestselective-pair)
is selected; separate durations cannot be combined.
Launch one Python process with one visible GPU; distributed launch is unsupported.
The runner checks Trainer's actual microbatch size before any optimizer updates.

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 python -m src.monitor_training train-twins --shared-run data/monitor_training/SELECTED_S --expected-shared-sha256 S_SUMMARY_SHA --prepared-root PREPARED --export-root EXPORT --model-root MODEL --inference-contract RECIPE/contract.json --settings TWIN_SETTINGS.json --expected-prepared-sha256 PREPARED_SHA --expected-inference-sha256 INFERENCE_SHA --output-dir data/monitor_training/NEW_TWINS
```

Use a new destination and explicit numeric budgets, including `max_fit_tokens`.
Before loading either twin, preparation requires `max_tokens` and `max_fit_tokens`
to cover the complete 16-pass pair, all three scoring snapshots and both reloads.
The computed upper bounds are retained in `rows.json` as `token_requirements`;
training counts use each branch's targets and generation reserves the full cap.
The maximum twin attempt uses 192 training updates and 360 development replies,
plus separately accounted construction fit (at most 144 replies), complete-label
likelihoods and two selected-checkpoint reload predictions. Token, time, estimated
cost and memory ceilings apply across both branches, using T4's resource checks.
Each branch visit loads a fresh 27B base: budget wall time and cost for up to six
training loads plus two selected-checkpoint reloads, as well as training and scoring.
An individual operation can overrun a ceiling and leave an incomplete attempt.

`rows.json` records the common input order. `parent.json` binds the selected S.
Each branch directory retains `continuous/checkpoint-{6,24,96}`, trainable names,
runtime metadata, and `pass-{1,4,16}/fit.json`, `development.json` and `summary.json`
for reached snapshots. Fit uses each branch's own targets. Events retain training
loss and branch-labelled updates, replies and likelihoods. Actual presentations
reconcile separately for each branch; updates and supervised tokens are recorded. Token totals
can differ because PASS and FLAG continuations can have different lengths.

`pair-{1,4,16}.json` retains each branch summary once, reuses its standalone cell
gates and adds the two cross-branch comparisons. The actor PASS gain follows from
the absolute actor gates but remains explicit as specified in the plan.
A branch can pass its own gates while the pair fails synthetic FLAG retention.
The report retains all three cells, task-level PASS differences,
the actor-minus-synthetic PASS interaction and margin changes. Uncertainty uses
the sample standard error of task-level contrasts, keeping invalid replies in
the denominator. The mean attack-margin change across sources is a source-blind
shift diagnostic; margins do not select checkpoints. This single-seed pilot does
not establish training stability, retained actor-fault detection or a mechanism.

`summary.json` records all reached pairs and either both selected checkpoint
references with hashes and verified inference reloads, a failed behavioral
attempt, or an incomplete run with its traceback. Earlier checkpoints and raw
partial results remain on disk. Branch runtime measurements are also collected
on failure; a measurement error is logged without replacing the original error.
Load failures retain run-level CUDA peaks even when runtime metadata is unavailable.
Confirmation is explicitly unresolved and
selection is development-only. T6 owns confirmation policy and repair handoff.
