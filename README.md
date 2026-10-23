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
