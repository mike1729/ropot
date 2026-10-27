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
  `archive/stage03/experiments/batch09_development.py`. One draw at temperature 0.2, top-p 0.95,
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

11a has completed without technical errors. Qwen passed 21/32 exploration and
24/32 confirmation tasks. Ministral passed 18/32 confirmation tasks within six
attempts; attempts 7–8 added no passes. Joint coverage is 14/32 exploration tasks
(including either Batch 10 arm) and 17/32 confirmation tasks. Use the saved
[11a results](../data/03_generation_checks/batch11-confirmation/results.json)
for the closing report. Batch 12 separately reruns revised recipes on previously
used tasks; report those results as acquisition after tuning.

### Implementation and inputs

Implement 11b after [Batch 12](stage-03-batch-12-plan.md), so `v1` includes its
passes within the same existing task groups. Add
`src/data-pipeline/stage03/benign_selection.py` with `publish(root, version)`.
`root` is the numbered study experiment root; `version` is `"v1"` for this
publication. The destination is `root.parent / "03_benigns" / version`. Reject
version names containing path components. Return the published manifest path.
It reads saved JSON and exact program bytes, without importing generation,
verification or Inspect. Reuse `experiment_summary.summarize` and `study_tasks`;
trust saved verdicts rather than replaying native logs or executing programs.

- **Inputs:** every numbered experiment must be terminal, study-owned and on the
  pinned cohort, with readable supported input/outcome records. Refuse unresolved
  or malformed records instead of silently skipping an experiment. A terminal
  failed run can contribute an earlier consistently verified pass; its failed
  effort stays in the summary. Load each input/outcome once and hash those bytes.
- **Eligibility:** the current study root contains 13 experiments covering 50
  development and 64 construction tasks. Include only the four labelled groups
  below. Do not load or activate further cohort tasks. Fixtures and direct prompt
  probes outside this root contribute no programs. A task need not pass for both
  sources; keep singly covered tasks for Stage 04.
- **Rule:** for each task and source, select the first attempt with state
  `passed`, in experiment-number order, then attempt order. Never use completion
  order or monitor scores. Record the rule. Do not restrict selection to the
  adopted recipe: previously verified programs, including Batch 10 arm A, remain
  eligible. Preserve the selected attempt's actual recipe and provenance.
- **Checks:** a selected attempt must also have `verification.status == "passed"`.
  The candidate's split, task ID and source must match the attempt's worklist
  binding and experiment source. The program bytes' digest must equal the
  candidate's and the verdict key's; both recorded task digests must equal the
  binding's, which must appear in `load_cohort` with that digest. Contradictory
  saved state or any failed check stops publication; no native-log replay is needed.
  Check eligible pass records before choosing the earliest one, so a later
  inconsistent record is not hidden by an earlier valid selection. The cohort's
  saved binding supplies ownership, phase and reporting labels.
- **Labels from cohort order:** development smoke (1–8), development (9–50),
  construction exploration (1–16 per phase), construction confirmation (17–32
  per phase). A worklist task outside those groups stops this first publication.

### Output and publication

- **Output:** a new `data/03_benigns/v1/` containing:
  - `manifest.json`: the rule, cohort digest, experiment input/outcome digests,
    and one record per selection (task binding and label, source, experiment,
    attempt, repair depth, recipe digest, model-manifest digest, candidate
    identity, verdict key with contract digest, native log and sample ID, program
    path and digest). Include the selected recipe object alongside its digest;
    record local round/repair depth and parent attempt without confusing a fresh
    cycle's local depth with the total policy attempt number;
  - `programs/` with the exact selected bytes;
  - `report.json`: the Batch 8 summary plus counts per label and source. Its
    counters cover the numbered experiment root only; effort outside that root
    is reported separately in the closing report. Include each group's full
    denominator, selected counts, jointly covered IDs and missing task/source
    cells. These are pooled-artifact coverage, not frozen-recipe yield estimates.
- **Paths:** use deterministic program names based on source, split and task ID.
  Manifest program paths are relative to the published version directory, so
  renaming the temporary directory does not break them. Keep native evidence
  references as provenance; do not copy raw replies, task cases or native logs.
- **Publication:** build in a temporary directory beside the target, then rename;
  refuse an existing version. Replacement, removal and later versions wait for a
  consumer. On validation or write failure leave no published target and clean
  up this call's temporary directory. Never alter the experiment records.

The read-only planning preview below records coverage before Batch 12. Update it
after that rerun and check it against the reviewed root at publication; do not
hard-code these counts into the selection module or confuse them with the 11a
confirmation measurements.

| Group | Tasks | Actor programs | Synthetic programs | Both sources |
| --- | ---: | ---: | ---: | ---: |
| Development smoke | 8 | 7 | 6 | 6 |
| Other development | 42 | 29 | 25 | 21 |
| Construction exploration | 32 | 21 | 16 | 14 |
| Construction confirmation | 32 | 24 | 18 | 17 |

Before Batch 12, this is 146 selected programs, 108,121 bytes before JSON metadata.
Record the additional passes and their provenance before publication.

### Tests, review and execution

Tests use real coordinator JSON from mocked runs, covering:

- first-pass order across experiments and attempts, despite later completion order;
- singly covered tasks, full denominators and labels from the cohort;
- digest mismatch, a foreign task and fixture runs, each refused;
- a passed attempt with a nonpassing verdict, or mismatched candidate task/source
  identity, each refused;
- refusal of running, malformed or unreadable experiment records;
- refusal of an existing version;
- a failed write leaving no published version, and relative paths that resolve
  after a successful rename.

Use `tests.test_benign_loop.LoopFixture.run_loop` for saved coordinator artifacts;
patch cohort lookup to those study-labelled test bindings. Hand-edit JSON only
for explicit malformed or inconsistent cases. No live inference or Docker tests.

Run focused tests and the repository suite. Review the module, tests, proposed
manifest/report fields and documentation changes within the 400-line non-test
limit. Stop before publication. After approval, commit the reviewed change,
call `publish("data/03_benign_experiments", "v1")`, verify that every published
program resolves and hashes correctly, and report artifact sizes. Record the
actual publication outcome; do not claim Stage 03 complete before it exists.

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
Keep this documentation concise: the API and handoff, the measured outcomes and
limitations, and links to detailed records. Publication coverage pools recipes;
the six-attempt 18/32 Ministral confirmation remains the frozen-policy measure.
Do not interpret the unproductive fourth cycle as adopting an eight-attempt
budget for subsequent acquisition.

At planning time, the numbered study root records 720 submissions, 435,466
observed output tokens and 1,289,216 committed tokens, with complete usage.
With the direct probes, that is 812 submissions, 522,910 observed tokens and
1,477,632 committed tokens, excluding fixture qualification. Reconcile final
totals from the retained receipts rather than treating these as fixed constants.

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

The [confirmation caller](../archive/stage03/experiments/batch11_confirmation.py) implements
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
