# ropot

Download the pinned [APPS dataset](https://huggingface.co/datasets/codeparrot/apps):

```sh
python3 -m src.data-pipeline download
```

Run from the repository root. `download` is the first command in the
`src/data-pipeline` module; use `python3 -m src.data-pipeline --help` to list commands.

The implementation follows the pipeline stages:

- `stage01`: download, screen, qualify, analyze and export the test pool.
- `stage02`: track known use, allocate cohorts and publish frozen groups.
- `stage03`: load study tasks and verify programs through Inspect.

The root package provides the CLI and source provenance hashing. New manifests
record source hashes by package-relative path (for example, `stage01/qualify.py`).

Uses only Python's standard library. Downloads both original splits (about 1.4 GB)
to `data/raw/apps/`, verifies upstream SHA-256 checksums and 5,000 rows per split,
and writes a local provenance manifest. Data files are ignored by Git. An existing
download is verified and reused; interrupted downloads restart on the next run.

Raw JSONL uses `id` for task IDs; IDs are scoped to their split. `solutions` and
`input_output` contain JSON strings. Downloading does not execute supplied code,
qualify tasks, or assign pilot partitions. See the [pipeline](docs/data-pipeline-overview.md).

Qualify the downloaded tasks locally, using Python 3.12 for preprocessing and
reference execution:

```sh
/opt/homebrew/bin/python3.12 -m src.data-pipeline qualify
```

Options: `--input-dir` (default `data/raw/apps`), `--output-dir` (default
`data/01_qualified_pool`) and `--workers` (1–8, default 8). Both pinned splits must
already be present and pass checksum, row-count and ID checks. An existing output
directory is refused. A successful complete run publishes per-task JSON and a
manifest with source hashes, runtime provenance, counts and exclusions by rename.

Qualification executes supplied code locally, without a sandbox or memory limit.
Each task needs three distinct references passing every stored case under the
five-second deadline. The output retains original data, example visibility,
reference attempts and timings; see the [stage 01 plan](docs/stage-01-plan.md) for
the full contract. Inspect Linux re-verification is required before generation.
Tasks excluded after reference execution retain compact diagnostics under
`reference_rejections/`, including failing-case stderr for dependency analysis.

Analyze the qualified **test split only**, preserving all qualification evidence:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python -m src.data-pipeline analyze
```

The optional analysis dependency is `tokenizers==0.23.3`. Options: `--input-dir`
(default `data/01_qualified_pool`), `--output-dir` (default `data/01_pool_analysis`)
and `--tokenizer` (default
`/Users/michal/Workspace/models/mlx-community/Qwen3.5-9B-8bit/tokenizer.json`).
An existing destination is refused. Successful analysis publishes `tasks.csv` and
`summary.json` with provenance and prints compact tables. It measures Qwen content
tokens for each statement and three passing references, case characters/bytes,
reference availability and recorded execution timing; it executes no supplied code.
See the [analysis module](src/data-pipeline/stage01/analyze.py) docstrings for the contract.
The test pool has 1,448 tasks (335 introductory, 1,113 interview); the two qualified
train tasks remain historical evidence. This report chooses no eligibility cutoff
or cohorts and does not establish complete-example fit within 2,048 tokens.

Export the test-only dataset for the next stage (standard library only):

```sh
/opt/homebrew/bin/python3.12 -m src.data-pipeline export-test
```

Options: `--input-dir` (default `data/01_qualified_pool`) and `--output-dir`
(default `data/01_test_pool`). This copies all 1,448 test task JSON files unchanged,
with a new `manifest.json` and the original `qualification-manifest.json` for
provenance. The active manifest excludes train tasks and preserves the Linux
verification gate. Existing destinations are refused; publication is atomic.
The original pool and analysis report remain unchanged. Cohort allocation consumes
the task paths listed in `data/01_test_pool/manifest.json`.
Independent copies make that dataset self-contained; qualification artifacts can
be archived separately without breaking its task paths.

Freeze and publish cohort ownership and ordering (standard library only):

```sh
.venv/bin/python -m src.data-pipeline cohort
```

Options: `--input-dir` (default `data/01_test_pool`), `--analysis-dir` (default
`data/01_pool_analysis`), `--output-dir` (default `data/02_cohort`),
`--known-use-records` and `--known-use-coverage`. The standard-library
[cohort policy](src/data-pipeline/stage02/cohort.py) supplies the allocation:
`load_inputs(input_dir, analysis_dir)` validates the export/analysis join;
`allocate(loaded["tasks"])` returns frozen ownership, construction phases, ordered
group blocks, profile deferrals and quota shortfalls. See the [allocation plan](docs/stage-02-plan.md).
The loader verifies `summary.json`'s `tasks_csv_sha256` against the CSV bytes.
Archive an older analysis before refreshing the canonical `data/01_pool_analysis`;
the ordinary analysis and cohort defaults then use the canonical paths.

Known-use records and coverage load from the sibling `honeypot/data/manifests/`
directory by default (override with the corresponding CLI flags or loader arguments).
The loader checks original-row bindings and retains coverage limitations. Restricted
uses move from protected groups into construction before its order and phases freeze.

Publication writes versioned `manifest.json` and `report.json` together by directory
rename, retaining input/source fingerprints, known-use coverage, targets, budgets
and the Linux verification gate. Task files remain in the input dataset.
Task records contain `split`, `id`, `difficulty`, `path`, `sha256`, `owner`, `block`,
`defer_reasons` and `known_uses`, with `phase`/`redirected_from` when applicable.
Measurements remain in the hashed CSV; initial quota shortfalls and sequential
profile counts live only in the report.
Existing destinations (including dangling symlinks) and outputs inside either dataset input
are refused. Inputs must remain stable and publication assumes one publisher;
rename does not guarantee durability after power loss.
Construction activation follows `construction_phases`; deferred tasks stay inactive.
Accepted-data targets have planned status, separate from observed usable counts.
Generation and retry ceilings remain pending acquisition decisions; publication does
not clear the Linux verification gate or establish accepted generated data.

Stage 03 experiments read reviewed tasks through the standard-library
[study task API](src/data-pipeline/stage03/study_tasks.py); see the [Stage 03 plan](docs/stage-03-plan.md):

```python
import importlib

study = importlib.import_module("src.data-pipeline.stage03.study_tasks")
loaded = study.load_cohort()  # pinned data/02_cohort manifest; task files are read lazily
task_ids = loaded["groups"]["development"]["initial"][:8]  # this experiment's worklist
binding, task = study.load_task(loaded, task_ids[0])
identity = study.candidate_identity("test", task["id"], binding["sha256"], "actor", b"print(0)\n")
```

Stage 02 generates and saves all five groups once in the cohort manifest's `groups`,
with construction phase lists in `construction_phases` (manifest schema version 2).
Stage 02 validates metadata at publication. Stage 03 uses the
[publication loader](src/data-pipeline/stage02/published_cohort.py) to check the pinned
cohort hash and index only saved construction/development IDs from initial and
reserve blocks. Protected and deferred tasks cannot be loaded. The loader reads no
test-pool manifest and does not repeat Stage 02's metadata validation. `load_task`
checks restricted membership, path containment and the task-file digest before
decoding. Bindings share the loaded metadata and must be treated as read-only;
`test_pool_dir` supports relocation without relaxing path or digest checks.
There is no activation object or allocation in Stage 03. Generation limits,
persistence and model calls belong to later batches; the reference gate stays pending.

Install the optional Inspect execution dependency and run its Docker fixtures:

```sh
.venv/bin/python -m pip install -r requirements-execution.txt
ROPOT_INSPECT_TESTS=1 .venv/bin/python -m unittest discover -s tests -p 'test_inspect_runner.py'
```

The [Inspect runner](src/data-pipeline/stage03/inspect_runner.py) runs fixture programs in
fresh Linux sandboxes and retains native `.eval` logs. See the
[execution contract](docs/stage-03-plan.md#inspect-execution-contract) for its API,
limits and differences from local execution. The cohort's Linux reference gate
remains pending batch 3; these fixtures run no models or cohort acquisition jobs.
