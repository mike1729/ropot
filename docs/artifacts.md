# Artifact locations and preservation

Datasets, model files and private run evidence live in ignored `data/`. Git
preserves implementation and documentation; a clone does not contain these
artifacts. Keep the local evidence separately when archiving or moving the project.

## Current qualification

The campaign root is `data/qualification/frozen92-20261101`. The current accepted
status is explicitly **`status-adjudicated-20261102`**. Do not choose a report by
modification time: later copies or recovery files do not supersede user decisions.

| Artifact relative to the campaign root | Meaning |
| --- | --- |
| [Current report](../data/qualification/frozen92-20261101/status-adjudicated-20261102/REPORT.md) | Final handoff including all four user adjudications |
| [Current readiness](../data/qualification/frozen92-20261101/status-adjudicated-20261102/readiness.json) | Accepted scoring/grounding resolutions and remaining blockers |
| [Current grounding](../data/qualification/frozen92-20261101/status-adjudicated-20261102/grounding.json) | Adjudicated explanations, original decisions and witness status |
| [Qualified view](../data/qualification/frozen92-20261101/report/qualified-view.json) | 91 retained pairs, the excluded benign and matched ownership panels |
| [Case audits](../data/qualification/frozen92-20261101/report/audit-results.json) | Original case partitions and execution results |
| [Recipes](../data/qualification/frozen92-20261101/recipes/recipes.json) | Generation, monitor and checkpoint bindings, with historical variants |
| [Cost ledger](../data/qualification/frozen92-20261101/costs/ledger.json) | Recorded costs, lower bounds and named missing components |

The sequence is the original freeze → `report/` → `handoff/` →
`status-monitor-resolved/` → `status-adjudicated-20261102/`.
Each revision adds evidence. Earlier readiness reports retain the unresolved
items that existed when they were written. The qualified pair view and original
selections do not change when explanation grounding is adjudicated.

From the repository root, read and verify the current status:

```sh
python3 -S scripts/qualification/status.py
python3 -S scripts/qualification/status.py --json
```

This checks every manifest/seal pair under the campaign root, including the
`validation` phase: currently 15 phases and 2,891 input bindings. Repository
inputs are read from Git commit `749e2a00a0ada2d6515f2f2615ec6c609605621a`, whose
bytes match all recorded repository hashes. Private `data/` inputs and external
files are read from disk; relative input paths resolve against the repository
root. Missing evidence, missing historical Git objects and changed bindings
return a nonzero exit status. The command works from any working directory when
called by its absolute path, using Python 3.9 or later and Git.

This verifies historical evidence without loading old pipeline modules, recomputing
scientific measurements or dispatching work. Changes to current source files and
documentation do not change the historical bytes used for verification.

## Historical three-row pilot

Current root: `data/monitor_training/historical-pilot-20261116-v3`; `panels/`,
`export/` and `prepared-v4/` hold the frozen 6/8/20 panels, bytes and prepared view;
`handoff-v4.json` supplies their pins. Earlier handoffs are preserved and marked
superseded by this handoff.
Recipe metadata references the sealed source manifest
instead of copying its recipe archive. This is frozen historical selection,
not a pre-acquisition first-qualified allocation or new shared-parent qualification.
The source audit passed 15 sealed phases / 2,891 bindings. Original benign evidence
for Qwen 2100/2649/4327 remains missing; accepted Gemma 4368 source drift is not repaired.
Weak task 4326, existing source cues and separate-parent limitations remain in the evidence.
All 68 Qwen programs bind their actual Qwen3.8 checkpoint. Stale declared Qwen3.6
metadata remains in `generator_metadata.actor.recorded_benign_manifest_sha256s`.
Source paths are repository-relative; freeze/export record producer `source_sha256`.
`configs/monitor_training/historical-reuse-20261116.json` records the existing
16 November user instruction to perform steps 1–3 with three training rows; it adds no new approval.

CLI forms, using fresh output destinations; pass the returned export/panel pins to preparation:

```sh
python3 -S -m src.data-pipeline.pilot_export freeze --campaign-root CAMPAIGN --cohort-root COHORT --reuse-decision configs/monitor_training/historical-reuse-20261116.json --output-dir ROOT/panels
python3 -S -m src.data-pipeline.pilot_export export --panels-root ROOT/panels --expected-panels-sha256 SHA --output-dir ROOT/export
```

## Other local evidence

| Location under `data/` | Contents |
| --- | --- |
| `raw/apps/` | Pinned original APPS splits and download manifest |
| `01_qualified_pool/`, `01_pool_analysis/`, `01_test_pool/` | Reference qualification, measurements and test-only export |
| `02_cohort/` | Frozen ownership, order, task copies and prior-use records |
| `03_benign_experiments/` | V2 benign attempts, native audits, recoveries and exposures |
| `05_attack_experiments/` | V2 exploratory attacks, selections, review and monitor campaign inputs |
| `06_monitor_experiments/` | Historical matched monitor queue |
| `03_generation_checks/`, `03_model_artifacts/`, `models/` | Runtime checks, checkpoint manifests and local model artifacts |
| `03_execution_checks/`, `03_throughput/`, `03_training_checks/` | Execution, batching and earlier training diagnostics |
| `03_benigns/`, `03_campaign/`, `03_reviews/`, `04_witnesses/`, `04_witness_experiments/` | V1 acquisition, review and publication evidence |
| `archive/`, `experiments/`, `runs/`, `maintenance/` | Historical snapshots, exploratory evidence and maintenance records |

## Cleanup boundary

Keep exact selected program bytes, native logs, failed/interrupted attempts,
reviewer decisions, exposure records and costs. Some records use absolute paths
and bind source files by hash; renaming a run or rewriting its manifest breaks
those bindings. Preserve the Git history containing the sealing commit as well as
the private evidence. The [archived phase drivers](support/qualification-drivers-archive.md)
checked live source paths; use the status command to inspect historical evidence
during refactoring. A twin handoff reader must use explicit historical source bindings
rather than import those campaign drivers. A stage refactor needs new evidence
for changed execution; historical hashes and outcomes stay as recorded.

The large local model directories are outside ordinary repository cleanup. Their
size alone does not identify disposable files. Bytecode caches are disposable;
run directories and model artifacts require a consumer/provenance check before
removal. This cleanup changes the entry points while preserving the evidence.
