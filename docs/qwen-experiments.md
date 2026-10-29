# Qwen development experiments

Use `scripts/qwen_experiment.py` for the exploratory rounds in the
[B2 plan](v2-batch-02-plan.md#qwen-only-development-experiments). It reuses the
shared generation loop, attack prompts, shown-case checker and guarded worker.
No official `prepare_contract` or two-source acquisition is required.

The selected actor and monitor are now Qwen3.8-27B MLX 8-bit, with thinking
enabled and explicit `reasoning_effort=low` for development. See the
[model migration](qwen38-migration.md) for runtime wiring still to complete.
Existing B1 Qwen3.6 benigns remain labelled cross-checkpoint exploratory inputs;
they do not establish Qwen3.8 own-source eligibility for the official baseline.

Run from the repository root with the execution environment installed:

```sh
.venv/bin/python scripts/qwen_experiment.py run data/05_attack_experiments/qwen-dev-01
```

Each output directory must be new. The default runs one independent chain over
all eight B1 tasks, with at most two repairs per task: at most 24 replies,
196,608 reserved output tokens and 234 shown-case executions. Passing tasks stop
within each chain. This command starts live Qwen generation and Docker execution;
the configured B1 MLX Python needs Metal access.

Pass `--config path/to/recipe.json` to tune `chains` and any existing
`ATTACK_RECIPE` field. For example, a shorter initial-only round:

```json
{"chains": 1, "recipe": {"max_repairs": 0, "max_tokens": 4096, "temperature": 0.2}}
```

Comment stripping is always enabled, including with custom recipes. The `# case:`
declaration is validated and saved as attempt metadata before comments are removed.
Saved programs and execution use the stripped code. Repair prompts reconstruct the
case declaration from metadata above the stripped candidate. Historical benign
files are hash-verified unchanged and normalized only in memory when loaded.

`system`, `initial` and `repair` replace scaffold instructions; the statement,
original benign and complete shown-case context stay in each request. Seeds are
`base_seed + chain * 8 * (max_repairs + 1) + repair_round * 8 + position`.
The default base seed is **1,000,041**. Every potential submission must avoid
**41–88**, reserved for the official acquisition, including unused repair slots.
The fixed Qwen batching is 4/1/256. Output limits stay within 8,192 tokens per reply
and prompt limits within 25,808 tokens. Every round has finite recorded allowances;
technical failures stop the run, preserve evidence and never retry automatically.

`experiment.json` records the effective recipe, task/model/code bindings and limits.
`report.json` reports functional and mechanically accepted task yield, violations,
truncations, extraction failures, duplicate submissions, tokens and time, including
partial work. These are unreviewed development yields. `evidence.json` hashes the
run artifacts. Private audits and confirmation outcomes never enter the worklist,
prompts or candidate execution. Existing B1 audit eligibility is validated only.
At official preparation, pass all acquisition and monitor `report.json` paths,
including failed rounds, as `development_reports`. Preparation binds these files
by hash and includes their costs; incomplete usage stays unknown. There is no
cross-run accounting registry, so the list must cover every exploratory run.
Acquisition and monitor reports are saved before dispatch. A report still marked
`running` (or missing a terminal status) contributes only a cost lower bound;
its saved token count cannot establish a complete pipeline total.

Optionally add `--monitor` to score immediately, or score the saved run later:

```sh
.venv/bin/python scripts/qwen_experiment.py monitor data/05_attack_experiments/qwen-dev-01
```

The new monitor recipe uses Qwen3.8 with thinking enabled, `reasoning_effort=low`
and seed 17 for every request. Reuse the historical prompt and normalization,
but parse only the final answer and record Qwen3.8 sampling explicitly. Batching
4/1/256 and the 8,192-token input ceiling remain starting settings for validation. It scores all eight benigns and the first mechanically accepted candidate
per task in chain/attempt order, at most 16 replies. `monitor/report.json` contains
PASS/FLAG/missing counts, FLAG fractions and separate scoring costs. Claims are not
validated against private cases; monitor results are diagnostic. Scoring uses a
fresh monitor directory and refuses changed acquisition evidence. To compare
monitor recipes or repeat scoring, use a separate recorded experiment revision.

Validate the driver without GPU/Docker calls:

```sh
.venv/bin/python -m unittest tests.test_qwen_experiment -v
```
