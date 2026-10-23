# ropot

Download the pinned [APPS dataset](https://huggingface.co/datasets/codeparrot/apps):

```sh
python3 -m src.data-pipeline download
```

Run from the repository root. `download` is the first command in the
`src/data-pipeline` module; use `python3 -m src.data-pipeline --help` to list commands.

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
See the [analysis module](src/data-pipeline/analyze.py) docstrings for the contract.
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

Stage 02 batch 1 adds the standard-library [cohort policy](src/data-pipeline/cohort.py).
`load_inputs(input_dir, analysis_dir)` validates the export/analysis join;
`allocate(loaded["tasks"])` returns frozen ownership, construction phases, ordered
queue blocks, profile deferrals and quota shortfalls. See the [allocation plan](docs/stage-02-plan.md).
CLI publication and allocation against the real pool remain for batch 2.
The loader verifies `summary.json`'s `tasks_csv_sha256` against the CSV bytes;
older analyses must be regenerated to supply this provenance field.
Known-use records and coverage load from the sibling `honeypot/data/manifests/`
directory by default (override with `known_use_records` / `known_use_coverage`).
The loader checks original-row bindings and retains coverage limitations. Restricted
uses move from protected queues into construction before its order and phases freeze.
