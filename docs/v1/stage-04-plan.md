# Stage 04: witnesses for existing programs

Consume the [published benign selection](../../data/03_benigns/v1/manifest.json):
173 programs across 101 tasks. Produce up to **one approved witness for each
program**. Use bounded attempts and record unresolved programs. Complete Stage 04
before Stage 05.

Completed on 2026-10-27: [v2](../../data/04_witnesses/v2/manifest.json) publishes 156
approved witnesses after one faulty-benign exclusion; 16 programs remain unresolved. See the
[results and publication contract](stage-04-results.md) for coverage and effort.

## Three batches

| Batch | Outcome |
| --- | --- |
| 1 | **[Generate and check proposals](stage-04-batch-01-plan.md).** Load existing programs, reuse local generation, reject duplicate inputs, and run the three references and benign on proposed witnesses to check answer agreement. Trust existing reference qualification. |
| 2 | **[Add independent approval](stage-04-batch-02-plan.md).** Automatically run Codex/Claude CLI audits and bounded retry rounds. Save attempts, decisions and costs. |
| 3 | **[Run and publish — complete](stage-04-results.md).** Publish 71 development and 85 construction witnesses, retaining the exclusion, unresolved cases and cumulative effort. |

Keep [existing review-size limits](../AGENTS.md); split a target if necessary. Detailed
interfaces, prompts and budgets are decided during implementation.

## Essential rules

- Each source proposes witnesses for its own program, seeing only the statement,
  that program and safe rejection reasons.
- Reject whitespace-token duplicates against all stored inputs; preserve exact
  stdin for execution. Auditors assess semantic novelty against published inputs.
- Require unanimous reference answers, benign agreement and both auditor approvals.
  Auditors use pinned model versions and receive only the statement, witness and
  published inputs, with no project access.
- Keep verifier data and audit replies private from proposers. Retain failed
  attempts and benign disagreements for later dataset review.
- For duplicate rejection, give proposers only “Propose a different input.”
  Do not mention stored tests, matching inputs, case indices or answers.
- Close the pinned tasks 291 and 4347 as `input_domain_exhausted`: their stored
  cases cover all 55 valid integer pairs and all 10 valid even integers,
  respectively. Acquisition refuses these task identities before dispatch.
  Repeated duplicates on other tasks are a reason to inspect, not proof of exhaustion.
- Humans may veto. Report auditor agreement/disagreement separately for validity
  and novelty; agreement is not an accuracy estimate. No human sampling is required.

## Validation

Test duplicate rejection and audit isolation on fixtures. Qualify auditors'
validity judgments on stored valid inputs and known-invalid variants, including
broken implicit guarantees such as distinctness, connectivity and existence of a
solution. Assess validity separately from novelty.
Include the real task-1179 negative fixture (now
[archived with the audit dispatcher](../support/qualification-drivers-archive.md)):
`n=3, k=7` violates the bound `k <= 6`. Auditors receive only its statement,
stdin and published inputs; expected decisions, reasons and provenance stay in
the qualification harness. No auditor has been qualified by saving this fixture.

Then run a development smoke before processing the full pool. Reuse existing
generation, execution and recording components; add only witness-specific pieces.

The [reclassified development record](../../data/04_witness_experiments/dev-reclassified-naefy_d2/summary.json)
has 62 execution agreements from 80 program attempts. Four program records on
the two exhausted tasks are terminal; 14 program records remain retry candidates.
The original runs remain unchanged. This was the pre-audit B1 state; current
approvals and unresolved cases are recorded in the Stage 04 results.
