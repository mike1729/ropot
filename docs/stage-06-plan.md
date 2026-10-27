# Stage 06: normalize and export labeled pairs

Assemble verified benign/attack pairs from [Stages 03–05](stage-05-plan.md) into a
versioned dataset. The first publication covers **development and construction**;
repeat assembly as other groups acquire programs. Rationales and training targets
remain with the training stage.

## Two batches

| Batch | Outcome |
| --- | --- |
| 1 | **Review and normalize.** Join programs and evidence, review overlap and defect findings, and produce comment-free monitor versions. Preserve originals and record exclusions. |
| 2 | **Measure and publish.** Measure token fit and publish labeled pairs with provenance. Report independent pairs, fully matched tasks, flags and shortfalls—including whether development reaches 20 fully matched tasks. |

Keep [existing review-size limits](AGENTS.md). Detailed formats and procedures are
decided during implementation.

## Essential rules

- Strip Python comments uniformly across both sources and labels, deleting
  comment-only lines and trailing whitespace without changing string contents.
  Require matching ASTs parsed from file bytes, ignoring location metadata.
  Retain original verified bytes and trust upstream execution evidence under
  this normalization assumption.
- Exclude explicit attack announcements in names, code, strings or docstrings.
  Direct witness-matching guards remain allowed.
- Preserve source identity, task ownership, construction phase and exposure
  history. Select in frozen cohort order.
- When training tasks overlap development or protected panels, exclude the
  training-side task and its variants. Tasks never change group; unresolved
  overlap conflicts remain flagged.
- Exclude a faulty benign and its attack when reference agreement establishes
  its failure on an independently audited valid input. Export other suspected
  defects with a flag.
- Monitor inputs contain only the statement and normalized program. Keep labels,
  source tags, witnesses and verification/audit evidence in separate metadata.
- Count rendered inputs using the pinned monitor tokenizer and chat template,
  plus a declared completion allowance, against the initial **2,048-token budget**.
  Exclude over-budget pairs without truncation.

## Validation

Use fixtures to check comment removal—including leftover whitespace and preserved
strings—unchanged syntax trees, overlap decisions, defect flags, labels, privacy
and token budgets. Reconcile published counts and exclusions with the input
publications.
