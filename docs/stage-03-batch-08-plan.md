# Batch 8 evidence, selections and cumulative reports

Turn [Batch 7's experiments](stage-03-batch-07-plan.md) into inspectable,
versioned benign selections. Account for every activated task/source cell and
every reservation, including unsuccessful and interrupted experiments. Derive
results from saved evidence without new model calls or sandbox execution.
Defer extraction recovery and generation continuation until a concrete workload
demonstrates the need; submitted or uncertain requests must never be replayed.

## Scope and review boundaries

8a requires a reviewed, committed 7b artifact format and at least one explicitly
identified retained run in that format. Pin commit `d6c311e` as the source baseline;
[the newer 7b run](../data/03_generation_checks/batch-07b-hx8hvdj5/qualification.json)
provides compatible flat-attempt artifacts. Its recorded producer hashes differ
from that commit: format compatibility does not qualify the committed coordinator.

Offline Batch 8 work does not wait for a passing model gate. The two-model repair
gate remains pending before generation/verification overlap or Batch 9 dispatch.
Its existing criteria and both stopped runs remain unchanged; revising those
criteria is separate Batch 7 work. This plan authorizes no gate rerun or commit.
Preserve existing working-tree changes; no fresh model draws are needed for 8a.

Implement on `main` in these independently usable review parts, following
[the repository rules](AGENTS.md). Each part must fit **400 additions plus
deletions excluding tests**, including its documentation. Stop for review after
each part; split further before implementation if an estimate exceeds the cap.

| Part | Implementation and acceptance boundary |
| --- | --- |
| 8a | Reconcile one experiment's saved inputs, requests, candidates and native verification evidence into an inspectable status/effort summary. Derive missing verdict JSON offline; perform no extraction recovery or dispatch. |
| 8b | Add explicit activation and ordered experiment inventories, successor rules and cumulative accounting. Report every activated cell, incomplete evidence and all prior effort. |
| 8c | Publish immutable selections, exact selected programs and yield/cost reports under pinned recorded-contract policies. Validate revisions, candidate provenance and atomic publication. |

Use small Python entry points in `stage03`, plain JSON and existing loop,
generation, task and verification helpers. Add no dependencies, database,
scheduler, persistent worker, generic event journal or legacy-format framework.
Keep Inspect pinned to `0.3.283`, native generation unchanged, and verification
at `max_samples=4, concurrent_inference=False`. Preserve published case semantics,
the public-feedback boundary and Batch 7's process-group interruption behavior.

Overlap remains a separately reviewed experiment. Batch 9 owns measured
input-token admission, oversized-task handling and development/construction
experiments. Batch 8 checks mechanics on fabricated artifacts; it does not acquire
cohort programs, widen activation or choose later-stage panel targets.

## Explicit inputs and activation

An activation record declares a finite ordered set of `(split, task id, source)`
cells and the pinned cohort digest and bindings. A separate inventory names that
activation and an ordered list of experiment paths. A cell means one task's
benign acquisition from `actor` or `synthetic`; multiple recipes/draws on it do
not create independent tasks. A requested source
with no experiment remains visible as `not_run`. Group capacity is not activation.

Use `study_tasks.load_cohort/load_task` to obtain permitted bindings and published
task bytes. Preserve owner, construction phase, initial/reserve block, difficulty
and saved cohort order. Keep protected and deferred tasks unavailable. Do not
rerun allocation, infer a second source's activation, or load the whole task pool.
Fixture inventories carry explicit fixture provenance and stay separate from
study yields and selections.

Every study experiment must name a frozen `activation_sha256` in `inputs.json`
before dispatch; add that small binding in 8b. The reader allows this field to be
absent only when `fixtures=True`; study runs must bind it to their declared
activation. This additive bookkeeping change must preserve exact model requests,
generation, extraction, verification and scheduling. Check that equivalence without
inference and record the changed coordinator hash; it requires no further model
gate and does not complete the pending one.

Extend activation only through an explicit successor preserving prior cells and
ownership. Existing generation
checks use fixtures, so add no retrospective cohort-exposure import. Keep fixture
evidence and effort separate from study yields. Batch 9 consumes study history
when choosing an unexposed confirmation block.

Read a caller-supplied finite inventory rather than scanning directories for
successes. Successors retain the prior inventory, including failed and superseded
experiments. Reject duplicate experiment identities, duplicate activation cells,
foreign task/source bindings and mixing cohort publications. Repeated reporting
must not count an experiment twice. Hash consumed files in each evidence snapshot
and reject changes while reading.

Automatically accept terminal runs with recorded child exit and verification
cleanup where applicable. For stale `running`, missing progress or unconfirmed
cleanup, require a recorded caller declaration that the coordinator and owned
children have stopped and sandbox teardown is complete. Preserve the original
status and declaration in the summary. The declaration permits inspection; it
does not prove non-submission, completeness or zero usage. File stability is not
a liveness test. Add no process probing, orphan detection or process killing.

## Reconcile evidence before using it

Target the final reviewed format: immutable `inputs.json`, task snapshots,
`round-*.json`, native generation records/replies, exact `program-*.py`, verifier
JSON/native logs, and root `outcome.json` with flat `attempts` and
`rounds[].attempt_ids`. Name the supported format in the reader. Retain older
intermediate smoke formats as historical evidence; reject unsupported reconciliation
inputs with a descriptive error rather than guessing a migration.

The replaceable root outcome is progress, not the sole source of truth. Bind:

- Original worklist positions, published task digests and separately labelled
  snapshot digests; for study runs, compare snapshot payloads with pinned tasks.
- Source/model-manifest identity, recipe, exact round requests, batching settings,
  round/attempt/row mapping, seeds and parent links. Batch settings currently live
  in round files and native inputs, not top-level experiment inputs.
- Native row status/submission evidence, reply-file UTF-8 bytes and saved text,
  token counts, and naturally completed versus partial/length-limited replies.
- Exact program bytes and `candidate_identity`; native verification task/program
  keys, execution contract, log and explicit sample ID. Match associations rather
  than relying on a plausible row count or completion order.

An attempt identity is `(experiment identity, local attempt id)`. Repeated program
bytes retain distinct attempts and costs while sharing the existing candidate
identity within the same task/source. Verification identity remains independent
of generator source. Missing or contradictory evidence prevents new selection
from that attempt; retain the diagnostic and all original files. Missing program
files remain unresolved; this batch does not re-extract saved replies.

The selection policy pins a finite set of accepted execution-contract metadata
objects and canonical digests. Accept only recorded capture/classification
semantics that the reader explicitly supports; add no historical-code loader.
Match complete native metadata to a declared accepted contract, log bytes to the
evidence digest, and any saved verifier key to that contract/task/program identity.
Reconstruct each request from the independently bound task and exact program.
Then use `verification.derive` with the accepted metadata and explicit sample ID
to check request association and case coverage. Reject unsuccessful evaluations
and any sample error, preserving `verify_many`'s whole-evaluation failure rule.

Re-derive every verdict. Where saved verifier JSON exists, independently bind its
task/program/contract keys and sample ID, then require exact equality of derived
`status` and the entire public `feedback` object with that saved record. A verdict
or feedback-only mismatch stops reconciliation for selection/publication; retain
both projections as diagnostics. Do not compare location-dependent log paths as
semantic fields. Malformed JSON is an evidence error, not a missing cross-check.

When verifier JSON is absent, retain native request/sample/coverage validation and
mark the saved-verdict cross-check unavailable. Derivation uses the reader's
supported current classification semantics (`case_status` and `output_matches`);
without the saved projection, it cannot establish equivalence to the historical
verdict or feedback. Changes to supported semantics require a reviewed reader and
policy change; an accepted contract digest alone does not freeze today's code.

Passing the log's own metadata to `derive` without those independent checks would
make contract validation circular. Do not use `reuse` here: its live `docker info`
and current source hashes govern execution reuse, not recorded-contract selection.
Publication requires no Docker daemon and never executes candidates. Retain
unaccepted-contract verdicts as historical evidence, separately from eligible
passes. Current-environment reusability is `not_checked` unless separately and
explicitly inspected; it never changes publication choices or totals.

Absence of an outcome, UID or reply does not prove non-submission. A reply file
alone does not establish natural completion. Accept `not_run` only from consistent
evidence proving the request never reached submission. Preserve unknown usage
and `submission_unknown`; do not turn technical failures into wrong answers.

## Accounting and reports

Compute totals from reconciled reservations and native evidence, retaining the
Batch 7 distinctions between committed replies/output-token allowances, confirmed
native acceptance where evidenced, recorded submission-boundary attempts,
submission lower bounds, observed usage and usage completeness. Native
`submitted=True` is set before insertion: without a UID/result it can mean an
interrupted submission attempt. Preserve that uncertainty; it never makes the
request definitely unsubmitted.

Never refund a reservation, including a definitely unsubmitted or short request.
If a crash obscures whether a reservation was committed, expose uncertainty and
do not report spare allowance as known.

Deduplicate evidence references, not draws. Reusing a log or publishing another
selection spends no model tokens and creates no new generation attempt. Record
each actual sandbox evaluation once even when several candidates share its log.
New experiment costs append to earlier effort; abandoned recipes and failed
experiments remain in cumulative totals.

Report loading, prompt preparation, generation, verification and overall elapsed
time separately. Sum recorded stage durations only where known; overall effort
summed across runs is not calendar latency. Keep native counters independently.
Incomplete usage is a lower bound, never zero or an exact total. Report measured
tokens and elapsed time; add no invented monetary estimate for local inference.

Every report includes every activated cell, with its known passing candidate(s),
latest run state, uncertainty/stop reason and selected candidate or explicit null.
Keep a known pass visible even if a later experiment errors. Retain Batch 7's
states and distinguish a never-attempted cell from exhausted draws, infrastructure
error, cancellation and unknown submission. Selection is a separate field.

Include counts and explicit denominators for activated unique tasks/cells,
recorded submission attempts, eventual passing cells, distinct verified candidates
and selected cells. Separate each experiment's initial-request yield from each
cell's first logical draw across the inventory; later experiments cannot reset
the latter or replace an uncertain first draw. A new experiment on unresolved
cells records new attempts and costs without resetting earlier uncertainty.
Use all activated cells in principal coverage/yield denominators, retaining
unresolved cells; report known passes as a lower bound when evidence is incomplete.
Any completed-only rate also states its coverage. Exclude control fixtures. Stratify
by owner/phase, difficulty, source and recipe/model/runtime configuration. Provide
per-experiment and cumulative views; task-level source intersections are benign
coverage only. Neither matched benigns nor repeated variants establish independent
sample size, attack yield, complete monitor fit or usable matched pilot tasks.

## Versioned selection publication

Publish under a new `data/03_benigns/<version>/` directory containing
`manifest.json`, `report.json` and exact selected program bytes. Bind the activation
and inventory/evidence hashes, selection policy, producer sources and optional
parent selection digest. Each selected record includes its experiment/attempt,
recipe and source, model/runtime identity, logical repair depth, candidate identity,
accepted contract digest and native log digest/sample ID. Later stages need that
provenance to assess recipe and repair-history effects. Keep private native evidence
reachable through paths and hashes. Do not place task snapshots or private feedback
into downstream program records; retain originals in their experiment directories.

For an initial selection, choose the first passing attempt eligible under the
pinned recorded-contract policy, in frozen experiment order then local attempt
order. Record that order as policy. Choose at most one benign per task/source;
retain all other candidates and draws. Never use completion order, monitor scores
or a reference solution. A later successful experiment does not silently replace
a choice.

A successor preserves previous choices by default. Replacement/removal requires
an explicit cell-to-attempt change list and reason in the new version; incompatible
or missing evidence stops publication instead of silently changing selections.
Retain all earlier versions. For downstream cohort-prefix use, flag cells behind
unresolved predecessors in the same phase/difficulty. Record any deliberate
exclusion and its reason for that version; technical uncertainty is not candidate
failure. Stage 06 owns final dataset assembly and the Stage 02 export-order rule.

Build and validate a staging directory, then publish manifest, report and program
copies together using the existing directory-publication pattern. Refuse an
existing version. Check copied digests before rename; a failed write leaves no
apparently complete version. With the same evidence, policy and derivation
semantics, repeated publication must choose the same candidates and totals,
apart from explicit publication metadata, including after Docker updates or VM
resizing. Classification/projection changes are subject to the cross-check above;
records without verifier JSON do not have that semantic-drift guard.

## Deferred recovery and continuation

8a derives every verdict from successful, fully bound native evidence without
changing the experiment. The inspected coordinator `verification-NNNN/` directories
all have sibling verifier JSON; a missing JSON simply removes one independent
cross-check, rather than requiring a separate recovery protocol. Repeated
reconciliation creates no draws, execution or usage. Separate extraction recovery, first/repeated
verification and model continuation wait for a concrete Batch 9 failure or a
measured need to extend `budget_exhausted` tasks. A new experiment on unresolved
cells remains possible with newly declared requests/seeds, budgets, attempts and
cumulative costs; it must not resend an uncertain saved request or imply the
unknown draw was free.

Before implementing deferred work, define its exact operation and bounded
acceptance case. Preserve these constraints:

- Natural completion is per row: an errored round may retain valid `stop` rows.
  A reply file alone does not prove completion. Explicitly choose an extractor
  identity; existing evidence hashes all of `local_generation.py`, so exact
  matching is strict and a narrower hash would be a reviewed contract change.
- Establish prior verification non-dispatch from the reviewed coordinator's
  persisted write order. Its `extraction` phase precedes verification; its
  `verification` phase is persisted before the call and does not prove dispatch.
  Stale or failed persistence can leave ambiguity; absence of logs is no proof.
- Claim any frontier before execution in a separate directory keyed by its
  evidence digest; never mutate hashed parent evidence. Define crash/duplicate
  handling before spending work. Preserve reservations without refunds.
- Never replay submitted or uncertain requests. Preserve original worklist length,
  positions, logical rounds, seeds and history. Use one logical round per native
  call with the existing repair helper. Budget revisions must not reset repair
  limits or erase earlier cost, exposure or submission uncertainty.

## Acceptance and review evidence

Use otherwise valid fabricated artifacts and injected phase failures. Each part
runs its focused checks and the repository suite; report non-test additions plus
deletions and test lines separately. Report retained generated/binary evidence
with sizes and provenance. This planning change runs no models or programs.

- **8a:** altered task/request/model/program/sample binding; stale progress after
  native completion; saved native log before verifier JSON; partial versus natural
  per-row completion; truncated outcomes; interrupted insertion with
  `submitted=True` but no UID; unknown usage; declared stopped runs and changing
  inputs. Derive both samples from a compatible fabricated log with Docker
  unavailable. Reject unaccepted metadata, tampered log bytes, any sample error
  and unsupported classification semantics. Separately test changed verdict,
  feedback-only drift, malformed verifier JSON and missing cross-check reporting.
  Use the Inspect log API, never unzip `.eval`. Require zero generation, extraction
  recovery and sandbox calls.
- **8b:** missing activated sources; duplicate inventories; repeated candidate bytes
  as distinct attempts; successor preservation of failed runs; reservation/usage
  totals; all-cell/stratum denominators; first-ever versus per-experiment yield;
  and no false cohort exposure from fixtures. Accept absent activation only for
  fixtures; test study binding/refusal and unchanged requests/dispatch behavior
  after the metadata-only addition, with zero inference.
- **8c:** stable choices after later passes or live Docker changes; explicit
  replacement/removal; complete recipe/runtime/repair-depth provenance; unresolved
  predecessors; failed publication writes and refusal of existing versions.
  Published bytes must match bound candidates. Repeated reconciliation/publication
  must preserve choices and effort without new work.

Use compatible retained fixture artifacts and fabricated records
for distinct failure cases. These checks need no fresh MLX draws or Docker
execution and do not complete Batch 7b's pending live gate. Stop after the final
part for review before overlap, input-token qualification or cohort dispatch.
