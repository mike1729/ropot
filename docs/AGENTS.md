

# Working with this repository

The user is a mathematician with strong software engineering experience, including
building a blockchain and designing a consensus algorithm in Rust, and is moving
into AI. Explain experimental and AI-specific choices precisely; assume familiarity
with software engineering.


# Implementation choice keep the pilot minimal.
Reuse established components for execution, downloads, data handling and logging; implement only experimental
rules and small necessary adapters. Prefer a narrower task scope to building
general infrastructure. The [pipeline specification](data-pipeline-overview.md)
governs fresh allocation; preserve historical evidence and freeze new ownership before generation.

## Minimal implementations

Implement the smallest coherent change that meets the current batch's acceptance
criteria. Preserve required scientific checks, data integrity, provenance and
attempt accounting while avoiding work that belongs to later stages.

- Tie each check, abstraction, schema field and dependency to a current consumer,
  acceptance criterion or demonstrated failure. Possible future usefulness alone
  is insufficient.
- Prefer straightforward functions and existing public APIs. Add frameworks,
  caches, fallback paths or compatibility layers only for a current use case.
- Defer later-stage validation and hardening to its owning review batch; record
  the requirement there instead of implementing speculative support now.
- Test distinct acceptance properties with otherwise valid fixtures, so failures
  demonstrate the intended check rather than an unrelated rejection.
- Document usage, contracts and limitations. Link to the plan instead of repeating
  every implementation check or future requirement in the README.

## Reviewable implementation stages

The user wants to review and understand implementation as it happens. Apply this
preference to future implementation work unless the user explicitly changes it.

- Work in coherent stages of at most 400 changed lines per review batch,
  excluding tests. Smaller stages are welcome; never add filler.
- An agent may ask the user for permission to exceed the 400-line limit when the
  excess would be small or completing the task in one go would be better than
  splitting it. Explain the expected diff size and why the exception is useful;
  obtain permission before implementing beyond the limit.
- Count additions **plus deletions**, including hand-written configuration,
  documentation and new files, but excluding tests. This is not a net-line cap.
  Report test lines separately; they still require review.
- Report generated lockfiles/manifests and binary artifacts separately, with their
  sizes, purpose and provenance. They still require review; do not use generated
  output to conceal implementation changes.
- Unless the user approves an exception, split an oversized stage before
  implementing it. Avoid unrelated refactors and
  formatting churn. Do not implement the whole feature and only then divide it
  into small commits.
- After each stage, report what changed, why, the main entry points and data flow,
  meaningful validation, actual diff size and remaining limitations. Stop for the
  user's review before starting the next stage unless they explicitly authorize
  continuing through multiple stages.
- Keep each stage coherent and testable. When commits are requested, normally use
  one focused commit per reviewed stage. A planning request alone does not
  authorize implementation or commits.
- Keep working on `main` unless the user asks for another branch or worktree.


