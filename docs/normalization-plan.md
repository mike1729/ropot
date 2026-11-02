# Stage 06 normalization and corpus diagnostics

Status: implementation plan, 2 November 2026. Implement and run on the current
91 retained pairs / 182 programs before prospective twin training, then repeat
after main-data acquisition. This document does not report a normalization run.
The [corpus analysis](support/twin-corpus-analysis-20261102.md) motivates removing
obvious presentation cues while retaining the existing scientific task.

## Position and consumers

Pipeline order: acquisition → qualification and frozen selection → **Stage 06
normalization/validation** → rendered-input token preflight → binary qualification
and training. New repair/evaluation acquisitions pass through the same frozen step.
The normalization function receives program bytes only, without source, labels,
partner programs, certificates, audit outcomes or monitor scores.
The monitor continues to receive one statement and one normalized program.

Publish a new versioned view under `data/06_datasets/`; preserve selected originals,
historical requests, exclusions and certificates byte-for-byte. Do not retrofit
the hash-pinned `stage05.monitor` or historical scoring callers. Reuse
`stage03.programs.strip_comments` and `same_ast` without changing their old behavior.
Use the same new view for base, shared honest, H, D, replay, repair and evaluation;
any future attribution stage must use it too. Historical results remain historical.

## Initial transformation contract

Use a pure `stage06.normalization` module and a thin `normalize` command in the
existing pipeline CLI. Pin Python 3.12's exact version and transform code hash.
Apply the same ordered rules to both sources and labels:

1. Existing comment removal, retaining string values and docstrings.
2. Conservative entry-point normalization using ASTs and explicit binding checks.
3. Deterministic `ast.unparse` formatting; require byte-identical idempotence.

Recognize one top-level zero-argument `main` or `solve` definition whose only use
is a terminal zero-argument call, bare or inside the exact standard main guard.
Reject decorators, aliases, rebinding, recursive uses, name collisions and unsupported
references. Rename the recognized definition/call to `solve`; standardize its terminal
call under `if __name__ == '__main__':`. Already canonical forms need no rewrite.
Genuine top-level scripts receive comment/format normalization without being wrapped
in a function. Record supported shape and operations privately; add no markers to code.

Ambiguous shapes and observable source/name introspection are unsupported, not
permission for a silent fallback or a source-specific exclusion. Preflight all 182
programs before publication; resolve unsupported cases as a versioned policy decision.
Do not rewrite general identifiers, imports, exception handlers, input APIs, numeric
literals, operators, conditions, loop bounds or algorithms in this first version.
Residual lexical/input/exception style is expected. The objective is removal of
selected presentation cues, not guaranteed source anonymization or chance attribution.

## Preservation and validation

Entry-point rewriting preserves the intended **standalone-script runner contract**,
where `__name__ == '__main__'`; adding a guard does not preserve arbitrary import
behavior. Do not claim universal Python equivalence. Formatting/comments require
identical ASTs apart from locations. Entry-point changes require exactly the allowed
structural delta and binding checks; parsing alone is insufficient.

Validate through the existing sandbox runner with finite case/time/resource budgets
declared before execution. Use already-authorized stored examples, menu cases,
certified triggers and audit cases. Keep private validation outcomes out of generation
and model inputs. Compare normalized execution with hash-bound original results,
or rerun originals in the same pinned environment where reuse is invalid.
Compare exit status and output, including the same wrong integer on each attack's
certified trigger; checking only benign correctness or pass/fail categories is insufficient.
Run all authorized cases without early stopping. Preserve malformed output behavior;
timeouts/infrastructure failures remain unresolved, not equivalence evidence.
Require zero unexplained mismatches. Finite tests supplement the structural contract;
they do not prove all-input equivalence or issue replacement original certificates.

Check hashes, idempotence, unchanged memberships/ownership, retained faulty-benign
exclusion, and no newly introduced task overlap. Detect collisions on the actual
statement-plus-program input. Identical cross-source attacks can have opposite D
targets; benign/attack collisions can contradict honest targets. Preserve memberships
and report conflicts; never silently deduplicate, relabel or discard affected tasks.
Unresolved preservation or target conflicts block promotion of the new view without
altering the original qualified corpus or triggering replacement acquisition.

## Minimal export and integration

Read selected IDs/raw hashes from the qualified view and statements from its matching
frozen records, filtering by source/task membership (exclude Qwen886, retain Gemma886)
without opening additional cohort tasks. Write normalized programs
and a small manifest binding upstream hashes, transform/version/runtime hashes,
original/normalized program hashes, task/phase/source/role, operation status, collision
groups and validation evidence references. Keep private metadata outside rendered inputs.
Use existing artifact publication helpers; refuse overwrite and partial promotion.

The prospective binary renderer consumes the manifest and pins its digest. H/D must
use identical input bytes; targets are assigned later. Count complete rendered inputs
with the pinned tokenizer/template and answer allowance against the prospective
8,192-token limit, without truncation. Publication alone does not establish token fit.
Freeze and qualify the binary interface on this view before training. Preserve the
comment-only view as historical evidence and a fixed before/after diagnostic.

## Review-sized implementation stages

Each stage follows the [400-line review limit](AGENTS.md), excluding tests, and ends
with review. Split a stage further if needed; these are scope boundaries, not permission
to compress an unsafe implementation to meet a line target.

| Stage | Deliverable and meaningful checks |
| --- | --- |
| N1: pure transformation | AST/binding whitelist, operation diagnostics and formatter. Fixtures cover guarded/direct calls, top-level scope, collisions/shadowing/dynamic lookup, preserved faulty expressions, literals, Unicode, encoding, deep syntax and idempotence. |
| N2: derived publication and current run | CLI/manifest adapter and existing-runner validation. Test upstream tampering, ownership, conflicting targets, no overwrite/partial promotion. Freeze budgets, preflight all 182, run a construction-only smoke, then bounded validation of all retained programs. Publish only after reconciliation of all 91 pairs. |
| N3: repeated analysis and consumer binding | Parameterize/reuse the local analysis scripts; report before/after measurements and task-disjoint probes. Integrate exact view/version enforcement and token fit with the binary consumer when it exists. Requalify binary monitoring on the normalized view. |

No new generation is required for the current run. N2's program executions and N3's
monitor qualification need their own recorded run budgets; this planning commit
does not assert readiness or perform those runs. Keep frozen qualification status
verification passing throughout implementation.

## Repeat after main acquisition

Run the same normalization and analysis after newly acquired programs are qualified,
before using them for training. Freeze transformation and diagnostic recipes first;
report current-corpus, new-only and combined results separately. Repeat:

- Pair/program/problem counts, provenance, ownership, overlap and collision checks.
- Benign/attack edit sizes, edit types and fault diversity by source and phase.
- Single-program source and label probes with task-disjoint splits and train-only
  preprocessing; paired ranking separately, with task-clustered uncertainty.
- Transform coverage, semantic validation, rendered-token fit and source cues left.
- Authorized audit strata and, once H/D exist, source-specific correction eligibility.

Probe accuracy is a diagnostic, not a row-selection gate or a target to optimize to
50%. Declare edit-type strata before evaluating monitor differences. Never use a
benign reference as monitor input, or compare model accuracy to paired ranking as
if they were the same task. Record development-driven changes as tuning.

Construction/development analysis may run when those groups are authorized. Repair,
calibration and locked evaluation stay behind their existing boundaries; apply the
frozen transform only at their authorized export stage and release permitted integrity
summaries. Inspect protected scientific results only at their planned analysis stage,
without adapting normalization/training. New unsupported syntax requires explicit
versioned handling, not silent changes based on protected content. Source-specific
repair selection/allocation remains a separate recorded amendment.
