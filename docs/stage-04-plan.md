# Stage 04: witnesses for existing programs

Consume the [published benign selection](../data/03_benigns/v1/manifest.json):
173 programs across 101 tasks. Produce up to **one approved witness for each
program**. Use bounded attempts and record unresolved programs. Complete Stage 04
before Stage 05.

## Three batches

| Batch | Outcome |
| --- | --- |
| 1 | **[Generate and check proposals](stage-04-batch-01-plan.md).** Load existing programs, reuse local generation, reject duplicate inputs, and run the three references and benign on proposed witnesses to check answer agreement. Trust existing reference qualification. |
| 2 | **Add independent approval.** Integrate isolated Codex and Claude audits into a bounded proposal/check/audit loop. Save attempts, decisions and costs. |
| 3 | **Run and publish.** Validate on development tasks, settle the recipe, process the remaining pool and publish approved witnesses with coverage, unresolved cases and effort. |

Keep [existing review-size limits](AGENTS.md); split a target if necessary. Detailed
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
- Humans may veto. Use a small random human check of approved witnesses to assess
  audit quality, with the sampling rule fixed before acquisition.

## Validation

Test duplicate rejection and audit isolation on fixtures. Qualify auditors'
validity judgments on stored valid inputs and known-invalid variants, including
broken implicit guarantees such as distinctness, connectivity and existence of a
solution. Assess validity separately from novelty.

Then run a development smoke before processing the full pool. Reuse existing
generation, execution and recording components; add only witness-specific pieces.
