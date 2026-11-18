# Scripts

Run scripts from the repository root. The reusable pipeline lives in
`src/data-pipeline/`; these files adapt it to bounded experiments or the completed
v2 campaign. Fixed campaign drivers retain their original paths and contents
because other scripts and frozen evidence bind them.

## Read current results

```sh
python3 -S scripts/qualification/status.py
python3 -S scripts/qualification/status.py --json
```

This read-only command verifies all sealed qualification phases and prints the
accepted 2 November status, pair counts, grounding counts and remaining blockers.
It requires local private artifacts, Git history at `749e2a0` and Python 3.9 or
later, with no optional packages. Repository inputs are checked against that
commit, so refactors do not invalidate historical verification. See the
[artifact guide](../docs/artifacts.md) for the exact validation scope.

## Workers and generation adapters

These are called by experiment drivers. Worker inputs carry explicit runtime,
request and output bindings; they are not independent qualification commands.

| Script | Purpose |
| --- | --- |
| `v2_attack_generation.py` | One guarded generation round |
| `qwen_chunked_generation.py` | Split a round across bounded fresh Metal workers |
| `deepinfra_generation.py` | Bounded raw-completion worker; [contract and limits](../docs/support/deepinfra-worker.md) |
| `replay_deepinfra.py` | Offline saved-response replay through the worker and Docker verifier; [results and scope](../docs/support/deepinfra-qualification.md) |
| `launch_deepinfra.py` | Offline preflight and explicit dispatch of pinned finite rounds; [usage](../docs/support/deepinfra-launcher.md) |
| `deepinfra_live_controls.py` | Offline replay or explicit live rerun of the ten Gemma controls; [usage](../docs/support/deepinfra-live-controls.md) |
| `qwen_benign_retry_worker.py` | Preserve completed requests during the recorded benign recovery |
| `qwen_edits.py` | Apply exact original-relative replacements |
| `qwen_native_tools.py` | Bounded native tool calls and feedback |
| `native_multi_edit.py` | Atomic multi-site edits and automatic witness selection |

## Development drivers

| Script | Purpose and record |
| --- | --- |
| `qwen_experiment.py` | Exploratory acquisition/scoring; [experiment guide](../docs/support/qwen-experiments.md) |
| `batching_probe.py` | Bounded batching measurements; [findings](../docs/support/v2-batching-experiment.md) |
| `prepare_yield_panel.py` | Prepare the saved development diagnostic |
| `qwen_dev_benigns.py` | Fresh benigns for the expanded development panel; [record](../docs/support/qwen-dev50-benigns.md) |

Their defaults describe historical panels. A fresh run needs the intended
own-source benigns, current runtime bindings and a finite recipe; a completed
campaign's dispatch command is not a recipe for continuing the twin pilot.

## Fixed campaign and recovery drivers

| Scripts | Completed scope |
| --- | --- |
| `dev_construction_queue.py`, `continue_matched_queue.py`, `resume_gemma_matched.py` | [Matched generation queue](../docs/support/dev-construction-qwen-gemma-queue.md), including no-repeat correction and Gemma recovery |
| `matched_monitors.py` | [Thinking-off matched queue](../docs/support/matched-monitor-queue.md); superseded for current status by frozen qualification |
| `construction_exposure.py`, `recover_construction_cases.py` | Construction exposure scan and [case recovery](../docs/support/construction-case-recovery.md) |
| `dev_recovery_cases.py` | [Development case recovery](../docs/support/dev12-recovery-cases.md) |
| `review_task291_exposure.py`, `recover_task291.py` | Task-specific exposure decision and recovery |

These retain fixed task panels, dates, accepted exceptions and retry ceilings.
Completed work is reused under their saved rules; existing calls must not be
redispatched as part of cleanup.

## Qualification status and archived drivers

`qualification/status.py` verifies the completed campaign from the current tree.
The sealed freeze, execution, review, accounting and resolution drivers are
[archived with their audit dispatcher](../docs/support/qualification-drivers-archive.md).
That note preserves the phase map and recovery commands. Reproduction requires
the sealing checkout and private evidence at their original paths; the
[qualification guide](../docs/frozen-corpus-qualification.md) records the accepted
results and limitations.
