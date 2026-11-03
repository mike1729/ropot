# Pipeline usage

Run commands from the repository root. The completed datasets and frozen cohort
already exist locally; downstream experiments consume those saved artifacts.
Preserve their task identities and ownership. See [artifact locations](artifacts.md)
and the [qualification record](frozen-corpus-qualification.md) before new work.

## Dependencies

Downloads, allocation and artifact readers use the standard library. Reference
qualification uses Python 3.12. Analysis and sandbox execution have separate
optional dependencies:

```sh
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-analysis.txt
.venv/bin/python -m pip install -r requirements-execution.txt
```

Analysis pins `tokenizers==0.23.3`; execution pins `inspect-ai==0.3.283` and needs
Docker for real sandboxes. Model generation runs in the separate MLX Python named
by the experiment's runtime contract, with Metal access and the pinned checkpoint
manifest. Historical model paths in v1 plans describe those experiments.

## Preprocessing and allocation

```sh
.venv/bin/python -m src.data-pipeline --help
.venv/bin/python -m src.data-pipeline download
.venv/bin/python -m src.data-pipeline qualify
.venv/bin/python -m src.data-pipeline analyze --tokenizer /absolute/path/to/tokenizer.json
.venv/bin/python -m src.data-pipeline export-test
.venv/bin/python -m src.data-pipeline cohort
```

| Command | Default inputs | Default output and behavior |
| --- | --- | --- |
| `download` | Pinned APPS train/test files | `data/raw/apps/`; checks sizes and SHA-256, reuses verified downloads |
| `qualify` | `data/raw/apps/` | `data/01_qualified_pool/`; three distinct passing references per task |
| `analyze` | `data/01_qualified_pool/` | `data/01_pool_analysis/`; test-only token, case-size and timing measurements |
| `export-test` | `data/01_qualified_pool/` | `data/01_test_pool/`; unchanged qualified test tasks and provenance |
| `cohort` | `data/01_test_pool/`, `data/01_pool_analysis/` | `data/02_cohort/`; frozen ownership, order, reduced task copies and reports |

Commands after `download` refuse an existing output directory. Use their
`--help` for `--input-dir` / `--output-dir` overrides; do not replace frozen
artifacts to rerun preprocessing. Successful publication uses directory rename.

Qualification executes supplied references locally, without a sandbox or memory
limit, using a five-second deadline per case. `--workers` accepts 1–8, default 8.
Failures retain diagnostics under `reference_rejections/`. The historical
[stage 01 plan](v1/stage-01-plan.md) records the admission contract.
The completed test export contains 1,448 tasks: 335 introductory and 1,113 interview.
Original qualified train tasks remain historical evidence.

Analysis runs no supplied code. It writes `tasks.csv` and `summary.json` with
source and tokenizer provenance. Its default tokenizer path names the historical
Qwen3.5-9B measurement checkpoint; pass `--tokenizer` for another local copy.
A changed tokenizer produces a new measurement, not the original cohort's input.

Cohort publication checks the export/analysis join and the CSV digest, then freezes
ownership, construction phases, initial/reserve blocks and profile deferrals.
Known-use records and coverage default to the sibling `honeypot/data/manifests/`
directory; override with `--known-use-records` and `--known-use-coverage`.
The [allocation plan](v1/stage-02-plan.md) records its policy. Protected and deferred
tasks remain unavailable to ordinary development loading.

## Generation, verification and scoring

The [script guide](../scripts/README.md) identifies experiment callers. Their
recipes carry finite task/reply/token limits, seeds and runtime bindings.
The [sealed recipes](../data/qualification/frozen92-20261101/recipes/recipes.json)
record the completed v2 campaign's effective settings.

| Consumer | Entry points | Contract |
| --- | --- | --- |
| Restricted study tasks | `stage03.study_tasks.load_cohort`, `load_task` | [Ownership and access](data-pipeline-overview.md#ownership-and-case-partitions) |
| Program execution | `stage03.inspect_runner.run`, `stage03.verification.verify`, `reuse` | [Execution contract](v1/stage-03-plan.md#inspect-execution-contract) |
| Local generation | `stage03.local_generation.generate`, `generate_batched` | [Generation and extraction](v1/stage-03-batch-06-plan.md) |
| Campaign benign acquisition | `scripts/qwen_dev_benigns.py` (`prepare`, `run`) | [Qwen3.8 panel and bounded settings](support/qwen-dev50-benigns.md) |
| Attack acquisition | `scripts/qwen_experiment.py` (`run`, `score_monitor`) | [Experiment driver and bounded settings](support/qwen-experiments.md) |
| Comment normalization | `stage05.monitor.strip_comments` | [Monitor inputs](data-pipeline-overview.md#monitor-inputs-and-retained-evidence) |

Because the package name contains a hyphen, Python callers load it with
`importlib.import_module("src.data-pipeline.stage03.study_tasks")`.
Only shown-case feedback can enter repairs. Private audits and monitor verdicts
cannot select replacements or enter generator feedback. The generation loop
retains all bounded attempts, raw replies, partial failures and costs.

The B1 `stage03.benign_acquisition.prepare_contract` and `acquire` entry points
retain the historical Qwen3.6 actor pin and [B1 contract](support/v2-batch-01-plan.md).
Their runtime qualifier checks repository files against live bytes; the existing
proofs refer to a moved batching document and changed probe/generation sources,
so they fail from this checkout. They do not qualify the current Qwen3.8 actor.
The campaign driver instead uses `qwen_experiment.load_panel`, which binds the
historical B1 evidence through Stage 05 and separately validates the active
Qwen3.8 checkpoint. Fresh MLX acquisition must use that explicit binding path
or establish a new runtime qualification before using the Stage 03 contract
entry points. Historical binding verifies saved proof integrity and live
runtime/model files; it does not establish fresh runtime qualification.

The v1 benign publication and witness acquisition chain is preserved with its
tests on an [archive branch](v1/archive.md). Its frozen outputs remain under
`data/03_benigns/v1` and `data/04_witnesses/v2`. The audit dispatcher and completed
qualification phase drivers are [archived together](support/qualification-drivers-archive.md).
Use `scripts/qualification/status.py` to verify their saved evidence.

The unused official B2 controller is [archived separately](support/b2-orchestration-archive.md).
The completed campaign used the experiment driver and its queue/recovery callers;
its shown-case checks and accounting helpers remain in `stage05.attacks`.

## Validation

```sh
.venv/bin/python -m unittest discover -s tests
python3 -S scripts/qualification/status.py
```

The default suite leaves live model and Docker fixtures disabled. Those require
their explicit `ROPOT_*_TESTS` opt-ins; see the fixture modules for exact flags
and runtime requirements. The status command needs Python 3.9 or later and Git,
with no optional Python packages. It checks all saved qualification phases using
historical repository inputs and private evidence on disk, without dispatching work.
