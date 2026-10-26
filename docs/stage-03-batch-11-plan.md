# Batch 11 construction confirmation and benign publication

Close Stage 03. Confirm both frozen recipes on construction tasks untouched by
tuning, publish the first versioned benign selection for Stage 04, and write the
closing report. Acquisition scale waits for Stage 04–05 yields.

## Scope and review boundaries

Follow [the repository rules](AGENTS.md) in two review parts on `main`, each within
400 additions plus deletions excluding tests, including documentation. Stop for
review after each part and before dispatch.

| Part | Outcome |
| --- | --- |
| 11a | A small confirmation caller, sizing record, live runs and results. |
| 11b | A selection module, the published `v1` selection and the closing report. |

Generation, feedback and verification sources must still match `d6c311e`; no new
live gate is needed. Change no recipe on these tasks, whatever the results.

## Frozen recipes

- **Qwen (actor):** the unchanged Batch 9 development recipe in
  `experiments/batch09_development.py`. One draw at temperature 0.2, top-p 0.95,
  at most one repair, 1,024 tokens per reply; decode 8, prefill 4, chunk 256.
- **Ministral (synthetic):** Batch 10's adopted arm B, with the strict-code recipe.
  Cycle 1 is one draw at temperature 0.2 and at most one repair, base seed 17.
  Each later cycle runs on the tasks still unresolved, in saved order: a fresh
  draw at temperature 0.7 and at most one repair, base seeds 100,017, 200,017 and
  300,017. Each reply allows 2,048 tokens; decode 8, prefill 2, chunk 2,048.

## Tasks and budgets

- **Ministral:** candidates 17–32 of each construction phase's saved initial block,
  32 tasks (8 introductory, 24 interview). Its exploration-block benigns come from
  [Batch 10](stage-03-batch-10-plan.md).
- **Qwen:** all 64 construction probe tasks, candidates 1–32 per phase. Qwen's
  recipe was never tuned on construction tasks, so all 64 are fresh for it.

Candidates 4326 and 617 share APPS IDs with honeypot's overnight run, as do eight
exploration tasks; record that history. Never drop or substitute a task.

Ministral runs up to four cycles, eight attempts. The confirmed recipe's yield is
the six-attempt value; attempts are sequential with early stopping, so any smaller
budget reads exactly from the same curve, and attempts 7–8 measure marginal value
on fresh tasks. Qwen keeps two attempts.

Allowances equal each run's full schedule: Qwen 128 replies/131,072 tokens,
Ministral at most 256 replies/524,288 tokens. Expect about 250 replies and 45
minutes. Run one model at a time. A technical stop halts the sequence for review,
without retries.

Before dispatch, render every initial prompt with each runtime tokenizer. Stop for
review if any exceeds the development maximum: 745 tokens for Qwen, 755 for
Ministral.

## 11a measures

Fix these before dispatch:

- Ministral passing tasks of 32 after each attempt 1–8. The confirmation figure is
  the six-attempt value; exploration results (16 and 15 of 32) are tuned context.
- Qwen passing tasks after attempts 1–2, separately for candidates 1–16 and 17–32
  per phase.
- Tasks with passing benigns from both sources, per 32-task block. For the
  exploration block, count a Ministral pass from either Batch 10 arm.
- Replies, committed/observed/prompt tokens, unusable replies by attempt, timings
  and peak memory.

Report counts against all tasks, keeping failures and unresolved tasks visible.
These benign counts alone set no acquisition scale.

## 11b selection and publication

Add `src/data-pipeline/stage03/benign_selection.py` with `publish(root, version)`.
It reads the numbered experiments' saved JSON and program files; it needs no
Docker, Inspect or model calls, and trusts saved verdicts.

- **Inputs:** every numbered experiment must be terminal, study-owned and on the
  pinned cohort.
- **Rule:** for each task and source, select the first attempt with state
  `passed`, in experiment-number order, then attempt order. Never use completion
  order or monitor scores. Record the rule.
- **Checks:** a selected attempt must also have `verification.status == "passed"`.
  The candidate's split, task ID and source must match the attempt's worklist
  binding and experiment source. The program bytes' digest must equal the
  candidate's and the verdict key's; both recorded task digests must equal the
  binding's, which must appear in `load_cohort` with that digest. Contradictory
  saved state or any failed check stops publication; no native-log replay is needed.
- **Labels from cohort order:** development smoke (1–8), development (9–50),
  construction exploration (1–16 per phase), construction confirmation (17–32).
- **Output:** a new `data/03_benigns/v1/` containing:
  - `manifest.json`: the rule, cohort digest, experiment input/outcome digests,
    and one record per selection (task binding and label, source, experiment,
    attempt, repair depth, recipe digest, model-manifest digest, candidate
    identity, verdict key with contract digest, native log and sample ID, program
    path and digest);
  - `programs/` with the exact selected bytes;
  - `report.json`: the Batch 8 summary plus counts per label and source. Its
    counters cover the numbered experiment root only; effort outside that root
    is reported separately in the closing report.
- **Publication:** build in a temporary directory beside the target, then rename;
  refuse an existing version. Replacement, removal and later versions wait for a
  consumer.

Tests use real coordinator JSON from mocked runs, covering:

- first-pass order across experiments;
- digest mismatch, a foreign task and fixture runs, each refused;
- a passed attempt with a nonpassing verdict, or mismatched candidate task/source
  identity, each refused;
- refusal of an existing version;
- a failed write leaving no published version.

Run focused tests and the repository suite. Then publish `v1` from the study root.

## Closing report

Add a short Stage 03 outcome to the [Stage 03 plan](stage-03-plan.md):

- the frozen recipes, yield per source and block, joint coverage, `v1` and total
  effort, with separate lines for numbered study experiments, direct tuning
  outside that root, and fixture qualification. The existing direct prompt probes
  contributed 92 replies, 87,444 observed output tokens and 188,416 committed
  tokens; retain their evidence references and include any subsequent draws.
  Do not present the Batch 8 root-only totals as total Stage 03 effort;
- limitations:
  - Ministral's observed construction exploration yield of 47–50%, which does not
    establish a capability ceiling; report confirmation yield separately;
  - the matched panel tilting toward Ministral-solvable tasks;
  - the development tasks' tuning history;
  - the honeypot ID overlaps;
  - the 7b gate test edited after its live run;
  - whether the `data/03_training_checks` receipts satisfy the training/probe
    compatibility precondition.

State in the [pipeline overview](data-pipeline-overview.md) that Stage 04 reads
`data/03_benigns/v1/manifest.json`, and document `publish` in the README.

Leave these outside Stage 03:

- bulk acquisition and reserve activation, sized after Stage 04–05 yields with the
  same loop;
- generation/verification overlap, batch-32 tuning and runtime input admission;
- recovery and continuation;
- selection replacement;
- BF16 comparisons.

Stage 03 is complete when 11a and 11b are committed, `v1` is published and the
closing report names Stage 04's input.


## 11a implementation review — 2026-10-26

The [confirmation caller](../experiments/batch11_confirmation.py) implements
`measure()` in the pinned MLX Python and `run()` in repository Python. It runs
Qwen on all 64 candidates first, then up to four Ministral cycles on the 32
confirmation candidates still unresolved. Existing Batch 11a directories prevent
redispatch; technical stops prevent any further source or cycle. No pipeline
modules changed; checked sources still match `d6c311e`.

The [sizing record](../data/03_generation_checks/batch11-confirmation/sizing.json)
freezes bindings, exact requests, recipes, source/model-manifest hashes and cycle
seeds. Qwen's 64 prompts range from 116–646 tokens; Ministral's 32 range from
131–652, below the planned development maxima. Only tokenizers were loaded.

The report separates Ministral's six-attempt confirmation from attempts 7–8,
Qwen's exploration/confirmation curves, and joint coverage per block. Exploration
context includes passes from either Batch 10 arm (16 tasks). Every task remains
in the denominator; partial results are labelled as lower bounds. Native round
metrics retain prompt/output/committed tokens, unusable replies including syntax
failures, timings and peak memory. Batch 8 summary and results save after a stop.

All nine focused tests and 302 repository tests pass, with eleven optional skips.
Changes: 171 caller lines, 166 test lines and this 29-line documentation addition.
Generated evidence: two JSON files/214,785 bytes, plus the superseded initial
sizing record (211,886 bytes) retained separately after the reporting fix.
No inference, supplied-program execution, publication or commit occurred. Stop
here for review before dispatch; live confirmation and results remain pending.
