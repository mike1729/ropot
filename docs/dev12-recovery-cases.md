# Supplementary development recovery cases

The user requested additional tests for the twelve case-ineligible development
tasks on 30 October 2026. This is a separate exploratory recovery allocation;
historical cohorts, benign acquisitions and attack results remain unchanged.

`scripts/dev_recovery_cases.py prepare OUTPUT` creates deterministic candidate
inputs using statement-derived constraints and answer oracles. It excludes
original APPS inputs, whitespace duplicates, and collection permutations where
order is irrelevant. It scans saved request content, retained reasoning and tool
arguments using the existing conservative exposure rule. Exactly eight menu and
four private audit cases are selected per recoverable task before executing any
retained benign program. The manifest binds the generator, sources and exposure
scan. The case files contain private data and must not be copied into prompts.

`scripts/dev_recovery_cases.py verify OUTPUT` uses the existing Inspect Docker
verifier with its pinned image and execution limits. Three previously qualified
reference programs must agree with the independent oracle on every new case.
Only after those checks does each retained benign receive one evaluation. Private
results never trigger repair, replacement or case reselection. Completion is
recorded separately from the immutable preparation manifest.

The saved recovery root is
`data/03_benign_experiments/qwen38-dev12-recovery-cases-20261030`.
All 120 cases passed all three references (360 executions); all ten retained
benigns passed their twelve cases (120 executions). Recoverable IDs are 4608,
2477, 2100, 2474, 4418, 2260, 2174, 2649, 4327 and 2282. No model generation ran.

Task 291 has only 55 possible valid inputs, all present in the full APPS suite.
Task 4347 has only ten possible valid inputs, likewise already present. Neither
admits a new source-novel input. The separate contextual review and user-approved
recovery below subsequently resolved task 291; task 4347 remains blocked.

These compact supplementary cases do not certify worst-case efficiency or
universal correctness. They do not automatically enter the frozen attack loader;
a subsequent recovery run must explicitly bind the new allocation and retain
the historical results separately. Exposure must be rechecked before dispatch if
additional target-model conversations occur after this preparation.

## Task 291 contextual recovery

After reviewing actual task-291 request contexts, the user explicitly authorized
recovery. Eleven inputs were disclosed in those contexts; six non-example inputs
in the original published suite remained undisclosed. This is a task-specific
adjudication of incidental numeric matches, not a global exposure-policy change.

`scripts/recover_task291.py OUTPUT` preserves the original benign and eight-case
menu. It selects the first four reviewed clean indices in the historical case
permutation, binds the review and source evidence, and verifies the full allocation
without generation, repairs or audit-driven reselection. The separate root is
`data/03_benign_experiments/qwen38-dev291-recovered-20261030`.
Its `summary.json` records completed recovery. Historical acquisition and attack
results remain unchanged; no attack run was started.

The existing attack driver accepts this root through `--benigns`. The recovery
kind is limited to task 291. Loading rechecks the frozen evidence and native
verification binding, and exposes only examples/menu cases to the attack worklist.
