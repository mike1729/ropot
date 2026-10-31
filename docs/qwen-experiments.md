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

Pass `--config path/to/recipe.json` to tune `chains`, `task_count` (an ordered
prefix of 1–8 B1 tasks) and recipe fields, including Qwen3.8 `reasoning_effort`
and `top_k`. Set `completion_batch_size` to 8 for batching 8/2/256 (default 4
uses 4/1/256). Resource guards apply to both. For example, eight tasks with medium
effort, 2,048-token initial replies and 4,096-token repairs:

```json
{"task_count": 8, "completion_batch_size": 8, "chains": 1, "recipe": {"reasoning_effort": "medium", "max_tokens": 2048, "repair_max_tokens": 4096}}
```

Comment stripping is always enabled, including with custom recipes. The `# case:`
declaration is validated and saved as attempt metadata before comments are removed.
Saved programs and execution use the stripped code. Repair prompts reconstruct the
case declaration from metadata above the stripped candidate. Historical benign
files are hash-verified unchanged and normalized only in memory when loaded.

`system`, `initial` and `repair` replace scaffold instructions; the statement,
original benign and complete shown-case context stay in each request. Seeds are
`base_seed + chain * n * (max_repairs + 1) + repair_round * n + position`,
where `n` is the selected task count.
The default base seed is **1,000,041**. Every potential submission must avoid
**41–88**, reserved for the official acquisition, including unused repair slots.
Output limits stay within 8,192 tokens per reply
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

Initial prompts include computed mechanical limits and ask for a prompt submission
instead of manual token counting or exhaustive case simulation. Checker feedback
adds numeric example, named-case, menu-failure and mechanical summaries alongside
the existing complete shown-case records. Truncation feedback explicitly states
that the output budget ended before a complete program was submitted and that
no candidate was extracted or checked.
`initial-prompts.json` and `initial-prompts.txt` expose the exact initial messages.
Every `actor-chain-*/round-*.json` stores exact requests, including repair prompts.

## Direct exact-edit experiment

Set top-level `submission_format` to `exact_edit` and `recipe.enable_thinking`
to false (omit `reasoning_effort`). The driver asks for only a JSON object with
`target_case`, `old` and `new`. `old` must be nonempty and occur exactly once in
the original program; `new` may be empty or multiline. Every repair replaces
an edit against that same original. The complete program is reconstructed,
compiled, normalized and checked using the existing validity rules.

The attempt records the submitted edit, match count and application failures.
Repair receives the previous edit and measured checker feedback, or explicit
JSON/application diagnostics. No unverified model claims enter search state.
Repeated normalized programs within a chain consume their slot and reuse cached
case results and native evidence. Acceptance and the certificate are recomputed
for each nominated target, including target-only repairs. An identical edit and
target also consumes a slot without execution. Independent chains retain
separate search state. When named_case is the sole violation, feedback recommends
keeping the edit and selecting a measured valid wrong-answer case if one exists;
otherwise it states that a different edit is needed. `repair_decisions` counts
initial retargeting opportunities, repairs, preserved edits, selections of observed
failing cases and accepted preserved retargets. `edit_funnel` reports JSON/schema
validity, applicability, duplicates, actual executions, reuses and automatic passes; semantic fields remain
unknown until review. The first recipe uses two chains, one repair per task,
768 initial tokens and 2048 repair tokens, with thinking disabled throughout.

```sh
.venv/bin/python scripts/qwen_experiment.py run data/05_attack_experiments/qwen38-dev-20261029-05 --config data/05_attack_experiments/recipes/qwen38-exact-edits-direct.json
```

Run 06 records the nomination-aware repair experiment separately, using
`qwen38-exact-edits-retarget.json` with the same limits, sampling and chain count.
Thinking with native tools is deferred until this repair behavior is evaluated.

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
