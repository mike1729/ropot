# ropot

Can synthetic corrections repair a monitor's selective failure to flag its actor's
attacks? This repository contains the data pipeline and experiments for the
[honeypot monitor repair pilot](docs/pilot.md).

**Data acquisition, scoring and corpus qualification are complete. The next
experiment follows the binary twin training plan,
starting with direct PASS/FLAG qualification.** The actor and historical monitor
use Qwen3.8-27B MLX 8-bit; Gemma-4-31B-it is the synthetic source. Planned monitor
training uses rented NVIDIA hardware; precision and LoRA versus full fine-tuning
remain open. The [training plan](docs/twin-training-plan.md) uses direct twin training
for the pilot and defers dedicated source elicitation to a full run.

The frozen corpus preserves 92 selected pairs and 184 programs. Its derived view
retains 91 pairs after one demonstrated benign defect, with 20 matched development
tasks and 14 matched construction tasks. Monitor scoring was accepted for this
corpus and all four grounding disagreements were adjudicated on 2 November 2026.
Full-cost accounting remains incomplete; fresh binary monitor qualification,
training compatibility and a runtime smoke are still required before training.
See the [qualification record](docs/frozen-corpus-qualification.md) for evidence
and the remaining limitations.

Read the [documentation index](docs/README.md), [pipeline contract](docs/data-pipeline-overview.md)
and [script guide](scripts/README.md). The [pipeline usage guide](docs/pipeline-usage.md)
contains preprocessing commands and stage APIs; [artifact locations](docs/artifacts.md)
identify current evidence and historical runs.

## Local use

Run from the repository root. Preprocessing and read-only artifact utilities use
Python's standard library. Reference qualification and the execution environment
use Python 3.12:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-execution.txt
.venv/bin/python -m src.data-pipeline --help
.venv/bin/python -m unittest discover -s tests
```

Analysis optionally adds `requirements-analysis.txt`. Local model workers use the
MLX runtime and checkpoint manifests recorded in their run contracts. See the
[usage guide](docs/pipeline-usage.md) for dependency and live-test details.

Read and verify the current qualification without loading models or execution
packages:

```sh
python3 -S scripts/qualification/status.py
# Machine-readable form:
python3 -S scripts/qualification/status.py --json
```

Private datasets, model files and run evidence live under ignored `data/` and
must be preserved separately from Git. The status command checks all sealed
phases: repository inputs come from historical commit `749e2a0`, while private
evidence comes from disk. This preserves verification as the working code and
docs evolve. The command needs Git and Python 3.9 or later, with no optional packages.

## Repository map

| Location | Purpose |
| --- | --- |
| `src/data-pipeline/stage01` | Download, screen, qualify, analyze and export tasks |
| `src/data-pipeline/stage02` | Exposure records, cohort allocation and publication |
| `src/data-pipeline/stage03` | Restricted task loading, generation, benign acquisition and verification |
| `src/data-pipeline/stage05` | V2 attack acquisition and monitor normalization |
| `scripts/` | Experiment adapters, guarded workers and frozen qualification status |
| `configs/` | Pinned inference, monitor and sandbox settings |
| `tests/` | Default regression suite and explicitly enabled live fixtures |
| `docs/support/` | Completed v2 plans, experiments, recoveries and reports |
| `docs/v1/` | Historical plans, results and Git archive recovery guides |

V1 is closed. Its [closeout](docs/v1/v1-closeout.md) and
[reviewed results](docs/v1/reviewed-generation-monitor-results.md) preserve the
failure analysis. The full implementation remains on
[archive/v1-failed-2026-10-28](https://github.com/mike1729/ropot/tree/archive/v1-failed-2026-10-28).
