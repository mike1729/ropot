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
to `data/raw/apps/`, verifies pinned byte sizes and SHA-256 checksums, and writes
a local provenance manifest with row counts and a difficulty report. The pinned
files fix the row identities; these are not revalidated after hashing.
Data files are ignored by Git. An existing
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
already be present and pass size/checksum checks. Preflight records the known
5,000-row counts without parsing the JSONL; qualification then streams the rows.
An existing output
directory is refused. A successful complete run publishes per-task JSON and a
manifest with source hashes, runtime provenance, counts and exclusions by rename.

Qualification executes supplied code locally, without a sandbox or memory limit.
Each task needs three distinct references passing every stored case under the
five-second deadline. The output retains original data, example visibility,
reference attempts and timings; see the [stage 01 plan](docs/stage-01-plan.md) for
the full contract. This qualification is sufficient for downstream task admission.
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
provenance. The active manifest excludes train tasks. Existing destinations are
refused; publication is atomic. Legacy gate metadata is historical only.
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
`allocate(loaded["tasks"])` trusts those validated records and returns frozen
ownership, construction phases, ordered group blocks, profile deferrals and quota
shortfalls. See the [allocation plan](docs/stage-02-plan.md).
The loader verifies `summary.json`'s `tasks_csv_sha256` against the CSV bytes.
Archive an older analysis before refreshing the canonical `data/01_pool_analysis`;
the ordinary analysis and cohort defaults then use the canonical paths.

Known-use records and coverage load from the sibling `honeypot/data/manifests/`
directory by default (override with the corresponding CLI flags or loader arguments).
The loader checks original-row bindings and retains coverage limitations. Restricted
uses move from protected groups into construction before its order and phases freeze.

Publication writes versioned `manifest.json` and `report.json` together by directory
rename, retaining input/source fingerprints, known-use coverage, targets and budgets.
Schema 3 includes reduced task copies under `data/02_cohort/tasks`; the original
pool stays intact. See the allocation plan for the frozen case selection.
Task records contain `split`, `id`, `difficulty`, `path`, `sha256`, `owner`, `block`,
`defer_reasons` and `known_uses`, with `phase`/`redirected_from` when applicable.
Measurements remain in the hashed CSV; initial quota shortfalls and sequential
profile counts live only in the report.
Existing destinations (including dangling symlinks) and outputs inside either dataset input
are refused. Inputs must remain stable and publication assumes one publisher;
rename does not guarantee durability after power loss.
Construction activation follows `construction_phases`; deferred tasks stay inactive.
Accepted-data targets have planned status, separate from observed usable counts.
Generation and retry ceilings remain pending acquisition decisions; publication
does not establish accepted generated data.

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
with construction phase lists in `construction_phases` (manifest schema version 3).
Stage 02 validates metadata at publication. Stage 03 uses the
[publication loader](src/data-pipeline/stage02/published_cohort.py) to check the pinned
cohort hash and index only saved construction/development IDs from initial and
reserve blocks. Protected and deferred tasks cannot be loaded. The loader reads no
test-pool manifest and does not repeat Stage 02's metadata validation. `load_task`
checks restricted membership, path containment and the task-file digest before
decoding. Bindings share the loaded metadata and must be treated as read-only;
`task_dir` relocates the publication root without relaxing path or digest checks.
There is no activation object or allocation in Stage 03. Generation limits,
persistence and model calls belong to later batches.

Install the optional Inspect execution dependency and run its Docker fixtures:

```sh
.venv/bin/python -m pip install -r requirements-execution.txt
ROPOT_INSPECT_TESTS=1 .venv/bin/python -m unittest discover -s tests -p 'test_inspect_runner.py'
```

The [Inspect runner](src/data-pipeline/stage03/inspect_runner.py) runs fixture programs in
fresh Linux sandboxes and retains native `.eval` logs. See the
[execution contract](docs/stage-03-plan.md#inspect-execution-contract) for its API,
limits and differences from local execution. These fixtures run no models or
cohort acquisition jobs. Batch 3 adds the [candidate verifier](src/data-pipeline/stage03/verification.py):
`verify(task, binding["sha256"], program_bytes, log_dir)` and explicit
`reuse(log_path, task, binding["sha256"], program_bytes)`. Only the returned `feedback`
may enter repair prompts; native logs and the rest of the result are private evidence.

The [local generation adapter](src/data-pipeline/stage03/local_generation.py)
supports the Qwen actor and Ministral synthetic generator.
Call it directly from the existing MLX Python with Metal access. The 8-bit model
loads once per finite run; requests carry no previous state:

```python
generation = importlib.import_module("src.data-pipeline.stage03.local_generation")
result = generation.generate(
    "/Users/michal/Workspace/models/mlx-community/Qwen3.5-9B-8bit",
    [{"messages": [{"role": "user", "content": "Reply with exactly OK."}],
      "seed": 17, "temperature": 0.7, "top_p": 0.9,
      "max_tokens": 32}],
    "data/03_generation_checks/my-run",
    model_manifest="/Users/michal/Workspace/models/metadata/Qwen3.5-9B-8bit-16daa4818c54ce5f5436f929d52542eb65bbed9d.json",
)
```

The manifest must describe the unchanged model files supplied by the caller;
artifact hashes are recorded without revalidating weights. Existing destinations
are refused. Inputs, runtime identity, raw reply text and outcomes remain in
the run directory. `max_tokens` bounds each reply; wall time is recorded, not
enforced, and loading or stalled native calls cannot be interrupted from Python.
Failure stops the run without retry. Generation itself executes no generated code.
For native batching, call `generate_batched` with the same arguments plus explicit
`completion_batch_size`, `prefill_batch_size` and `prefill_step_size`. It bounds
unfinished submissions by the decode size; interruption cancels unfinished
submissions. Batched generation retains token IDs in memory, decodes each finished
request and writes its reply file before marking it complete; Python abort also
saves partial replies and retries failed reply writes. Malformed native calls
retain scalar evidence without applying their events to rows.
Hard termination can lose active partial replies and outcome metadata.
`extract_program(row)` returns exact Python
bytes only from natural completions. See the [Batch 6 plan](docs/stage-03-batch-06-plan.md)
for settings, statuses, extraction rules and fixture qualification.
For Ministral, pass `/Users/michal/Workspace/models/mlx-community/Ministral-3-8B-Instruct-2512-8bit`
and its corresponding artifact manifest through the same API, with an explicit
system message. The adapter applies its tokenizer regex fix and records the
effective tokenizer and model sources. Both models use request-owned seeded samplers;
earlier global-reseed replies remain historical evidence. The
[Batch 5 plan](docs/stage-03-batch-05-plan.md) covers scope and limitations.
Run focused tests with `python3 -m unittest discover -s tests -p 'test_local_generation.py'`.
For the actor and synthetic fixture smokes, run that command using
`/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python`
with `ROPOT_MLX_TESTS=1`. Metal access is required. The synthetic smoke also needs
Docker and Inspect installed in `.venv`; it verifies three generated fixture
programs there. Both smokes retain evidence under `data/03_generation_checks`.
The separate batched qualification uses `ROPOT_MLX_BATCH_TESTS=1` with the same
MLX Python and `-m unittest tests.test_batched_generation_live -v`; it stops at the
first failed gate and retains every attempted reply.

The verifier also provides
`verify_many(candidates, log_dir, max_samples=1|4, concurrent_inference=...)`,
with one native evaluation and a fresh sandbox per program. Multi-sample
`reuse` requires the returned `sample_id` and the same concurrency/context.
Each fresh verification call requires an empty native log directory. Confirmed
Inspect cancellation propagates as `KeyboardInterrupt` after native teardown.

Batch 7a adds [benign_loop.run](src/data-pipeline/stage03/benign_loop.py), called
from the repository `.venv`. Supply ordered `(binding, task)` pairs from
`study_tasks.load_task`, the pinned cohort digest, explicit model/manifest/MLX
Python paths, and a new directory under `data/03_benign_experiments/`:

```python
loop = importlib.import_module("src.data-pipeline.stage03.benign_loop")
outcome = loop.run(
    worklist, "data/03_benign_experiments/my-run", source="actor",
    model_path=model_path, model_manifest=model_manifest, mlx_python=mlx_python,
    cohort_sha256=study_tasks.COHORT_SHA256,
    recipe={"system": "Return only a complete Python 3 program.",
            "initial": "Read stdin; print the answer.", "temperature": 0.2,
            "top_p": 0.95, "base_seed": 17, "max_tokens": 256, "max_repairs": 0},
    reply_allowance=len(worklist), output_token_allowance=256 * len(worklist),
    completion_batch_size=8, prefill_batch_size=4, prefill_step_size=256,
)
```

Generation exits before coalesced verification. Inputs, task snapshots, exact
programs, native evidence and atomic progress retain every attempt and its costs.
Fixtures use complete bindings and `fixtures=True`. See the [Batch 7 plan](docs/stage-03-batch-07-plan.md)
for states, interruption and review boundaries. Run its four-reply smoke with
`ROPOT_BENIGN_LOOP_TESTS=1 .venv/bin/python -m unittest tests.test_benign_loop_live -v`.

The [Q8 fixture findings](docs/inference-benchmark-results.md) consolidate measured
MLX settings, scheduling and extraction gains, capacity limits, and the archive
branch containing the one-off benchmark tools and detailed reports. The current
generation adapter provides native batching and Python extraction. The
[Batch 6 plan](docs/stage-03-batch-06-plan.md) records final fixture qualification
at decode 8 and 32 for both models, with all 80 generated programs verified.
Actual witness/backdoor yield follows the
[remaining plan](docs/inference-benchmark-plan.md).
