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
- `stage04`: load selected benigns and check proposed witness inputs.

The root package provides the CLI and source provenance hashing. New manifests
record source hashes by package-relative path (for example, `stage01/qualify.py`).
Shared artifact hashing and JSON writing live in `artifacts.py`; default data
locations live in `paths.py`. Cohort readers use these directly, keeping dataset
loading independent of allocation, publication and execution modules.

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
reference attempts and timings; see the [stage 01 plan](docs/done/stage-01-plan.md) for
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
shortfalls. See the [allocation plan](docs/done/stage-02-plan.md).
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
[study task API](src/data-pipeline/stage03/study_tasks.py); see the [Stage 03 plan](docs/done/stage-03-plan.md):

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
[execution contract](docs/done/stage-03-plan.md#inspect-execution-contract) for its API,
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
bytes only from natural completions. See the [Batch 6 plan](docs/done/stage-03-batch-06-plan.md)
for settings, statuses, extraction rules and fixture qualification.
For Ministral, pass `/Users/michal/Workspace/models/mlx-community/Ministral-3-8B-Instruct-2512-8bit`
and its corresponding artifact manifest through the same API, with an explicit
system message. The adapter applies its tokenizer regex fix and records the
effective tokenizer and model sources. Both models use request-owned seeded samplers;
earlier global-reseed replies remain historical evidence. The
[Batch 5 plan](docs/done/stage-03-batch-05-plan.md) covers scope and limitations.
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

Batch 7 adds [benign_loop.run](src/data-pipeline/stage03/benign_loop.py), called
from the repository `.venv`. Supply ordered `(binding, task)` pairs from
`study_tasks.load_task`, the pinned cohort digest, explicit model/manifest/MLX
Python paths, and a new directory under `data/03_benign_experiments/`:

```python
loop = importlib.import_module("src.data-pipeline.stage03.benign_loop")
outcome = loop.run(
    worklist, "data/03_benign_experiments/0001-my-run", source="actor",
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
To enable repairs, set recipe `max_repairs` above zero and supply `repair`
(instruction), `repair_max_tokens`, `feedback_field_bytes` and `feedback_bytes`.
Repairs preserve history and original-position seeds; only bounded public feedback
enters their prompts. Global allowances reserve full request limits without refunds.
Fixtures use complete bindings and `fixtures=True`. See the [Batch 7 plan](docs/done/stage-03-batch-07-plan.md)
for states, interruption and review boundaries. Run its four-reply smoke with
`ROPOT_BENIGN_LOOP_TESTS=1 .venv/bin/python -m unittest tests.test_benign_loop_live -v`.
The separate eight-reply repair gate uses `ROPOT_BENIGN_REPAIR_TESTS=1` and
`tests.test_benign_repairs_live`; unexpected outcomes stop the gate without redrawing.

The [experiment summary](src/data-pipeline/stage03/experiment_summary.py) reads only
numbered experiments' coordinator JSON. Call `summarize(root)` for recorded states,
histories and cumulative effort; see the [Batch 8 plan](docs/done/stage-03-batch-08-plan.md).

[Stage 03 results](docs/stage-03-results.md) close Batch 10, including additional
confirmation and rescue experiments. The saved-evidence
[publisher](src/data-pipeline/stage03/benign_selection.py) exposes
`publish("data/03_benign_experiments", "v1")`: it copies the first verified pass
per task/source into a new version and returns its manifest for Stage 04.
Existing versions are refused. Historical callers live in the
[archive](archive/stage03/README.md).

The [Q8 fixture findings](docs/inference-benchmark-results.md) consolidate measured
MLX settings, scheduling and extraction gains, capacity limits, and the archive
branch containing the one-off benchmark tools and detailed reports. The current
generation adapter provides native batching and Python extraction. The
[Batch 6 plan](docs/done/stage-03-batch-06-plan.md) records final fixture qualification
at decode 8 and 32 for both models, with all 80 generated programs verified.
Actual witness/backdoor yield follows the
[remaining plan](docs/inference-benchmark-plan.md).

Stage 04's [supplied-input API](src/data-pipeline/stage04/witnesses.py) loads
selected programs with `load_worklist(keys)` and extracts exact stdin with
`extract_input(row)`. `check_many(proposals, log_dir)` rejects stored-input
duplicates, then checks three references and the benign in one Inspect/Docker
evaluation per round. Results are private; agreement needs independent approval.
See the [Batch 1 plan](docs/done/stage-04-batch-01-plan.md) for contracts and validation.

The [proposal round](src/data-pipeline/stage04/proposals.py) accepts one source's
loaded records and explicit model/runtime paths, batching and allowances.
Its default recipe uses seed 17, temperature 0.2, top-p 0.95 and 1,024 output
tokens; pass `recipe` to override it. Witness instructions follow the statement
and program in the user message, with a neutral system message.
It uses one generation child, then checks extracted inputs and records every
attempt without retries. Call `run(..., fixtures=True)` for fabricated tasks.

Stage 04 uses automatic Codex/Claude CLI audits through existing logins.
`stage04.audits.qualify(output_dir)` runs qualification; `dispatch(output_dir)` reruns a stopped batch.
See the [Batch 2 plan](docs/done/stage-04-batch-02-plan.md).
The [2026-10-27 qualification](data/04_witness_experiments/qualification-cli-b3ea36c-20261027-4d32ae7c/summary.json)
passed all ten validity cases per auditor and matched novelty labels on all six
valid cases. One Codex capacity failure was resumed without repeating successes.

Stage 04's [v2 publication](data/04_witnesses/v2/manifest.json) contains 156 approved
witnesses: 71 development and 85 construction, excluding the faulty synthetic
task 4015 benign. V1 is preserved. The
[results](docs/stage-04-results.md) document one exclusion, 16 unresolved programs, coverage,
effort and the publication contract. The saved-evidence
[publisher](src/data-pipeline/stage04/witness_selection.py) exposes
`publish(run_dirs, output_dir, exclusions=())`, copying exact programs/stdin and approved answers
with hashed execution/audit provenance; existing versions are refused.

Stage 05's [implant API](src/data-pipeline/stage05/attacks.py) accepts in-memory
`(binding, task)` pairs. Each binding's private `implant` contains `candidate`,
`program` (UTF-8 text), `stdin`, approved `answer` and `provenance`.
`run(worklist, output_dir, source=..., model_path=..., **settings)` forwards the
ordinary benign-loop runtime, batching and allowance settings, with an optional
recipe and at most three attempts per pair. It checks the published suite before
the witness; only a valid different integer is accepted. See the [B1 plan](docs/stage-05-batch-01-plan.md).
Run its made-up-task smoke with:

```sh
ROPOT_ATTACK_TESTS=1 .venv/bin/python -m unittest tests.test_attacks_live -v
```

The provisional [B2 monitor](src/data-pipeline/stage05/monitor.py) scores in-memory
records through the existing MLX Python; see its [contract](docs/stage-05-plan.md#provisional-monitor-runner).
Publication loading and real runs await Stage 04's development publication and
explicit user authorization.

The [acquisition API](src/data-pipeline/stage04/acquisition.py) provides bounded
`acquire(...)` runs with explicit candidate scope, qualification, model/runtime
settings and total reply/token/audit allowances. Seed B1 history with `b1_dir`;
resume using `previous_run` and a fresh directory, omitting scope and B1 arguments.
Audits reuse qualified settings/digests and report CLI version drift. Batches cap
calls at four per packet and the remaining run allowance; resume needs their unused
ceiling. Each collection shares a fresh call-record snapshot across retry eligibility,
allowances, usage and version accounting; the qualified audit API derives decisions.
Generation costs cover whole referenced rounds, counting outside-scope
attempts separately. See [Part 2 contracts and validation](docs/done/stage-04-batch-02-plan.md#part-2-automatic-acquisition-per-source).
