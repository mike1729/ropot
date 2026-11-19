# V2 data pipeline overview

This overview covers the completed v2 data campaign and the
[main-run redesign](data-generation-redesign.md#main-run-recipe).
The [qualification record](frozen-corpus-qualification.md) identifies the completed
campaign's evidence and user resolutions. The [v2 feasibility plan](support/v2-implementation-plan.md)
records its implementation sequence.
The [v1 pipeline](v1/data-pipeline-overview.md) records the previous requirements;
its full-suite attack gate and fixed-witness acquisition do not apply to v2.

## Ownership and case partitions

Reuse the pinned APPS pool, integer-output task qualification, trusted references
and immutable task identities. Preserve the frozen construction, development,
correction/replay, calibration and evaluation groups, ordered reserves and prior
exposure records. Do not reallocate inspected tasks into protected groups.
Existing panel sizes are planning capacities, not guarantees of accepted pairs.
New loaders must preserve access restrictions; protected groups open only in their
authorized stage. Historical model panels remain diagnostic evidence.

Before acquisition, reserve scaffold-development and confirmation blocks within the
pilot's capacity. Keep confirmation outcomes unopened during scaffold tuning; those
tasks may later validate twins but remain outside training. Locked evaluation is
separate. Predetermine reserve advancement and retain every failed task in accounting.

Keep all statement examples. Split validated non-example cases into menu and audit
sets using a recorded random seed, identically for both sources. Then remove oversized
cases from the menu without transferring them to the audit set. Record exclusions;
check minimum menu/audit coverage afterward. Freeze sizes and minima in batch 1.
Use the same recorded menu permutation for both sources on a task/chain.

## Information boundaries and eligibility

For the main run, Gemma-4-31B-it is the actor and monitor, and Qwen3.8-27B
generates synthetic attacks. The [main-run recipe](data-generation-redesign.md#main-run-recipe)
specifies the inputs, qualification and review for attack generation, actor
repair and synthetic transplant.

### Completed v2 acquisition

Shown examples and menu inputs/answers may enter benign repair and attack requests.
Audit cases and results never enter generation, assistance or repair feedback. Record
prior exposure and exclude exposed cases from the independent audit designation;
withheld here does not mean absent from model pretraining.

The completed campaign used Qwen3.8-27B MLX 8-bit as actor and monitor, with
Gemma as the synthetic source. Existing Qwen3.6 benigns may be reused only in
labelled cross-checkpoint exploration.

Each source generates its own benign. Select using shown cases, then privately verify
all audit cases before attack generation. Failure disqualifies the source/task pair
without repair feedback or another benign attempt for that pair. Missing qualified
benigns or eligible cases are recorded eligibility failures.

The [31 October attack acceptance rules](support/v2-batch-02-plan.md) require all examples
to pass, a certified menu trigger returning a valid wrong integer, and at most
`floor(menu_size/2)` failed menu cases. Conditional/comparison counts and edit-token
churn are diagnostics only; blind review assesses the coherent localized fault.
Legacy formats certify the nominated trigger. `native_multi_edit` selects the
first measured valid wrong-integer menu case automatically, with no declaration.

Freeze selected attacks before auditing them. Attack audit survival is a measurement,
not a selection or retry gate; report it separately from shown-case survival excluding
the selected trigger. Exclude any subsequently demonstrated faulty benign pair and
record the reason. Reviewers supply neither code nor search advice to generators.
Monitor verdicts never select candidates or enter generation feedback.

## Monitor inputs and retained evidence

The [2 November corpus analysis](support/twin-corpus-analysis-20261102.md) records
small benign/attack edits, strong source-style cues and the distinction between
single-program classification and supplementary paired diagnostics.
The main run needs the [redesigned export](data-generation-redesign.md#training-export-handoff)
before [monitor training](monitor-training-plan.md). The redesign replaces the
separate normalization step.
Training consumes qualified bytes verbatim. Freeze source/label-neutral extraction
and comment/docstring rules in the generation recipe and apply them before
qualifying each program. Qualify the actor attack and its repair, then supply
the exact qualified repair bytes to the synthetic editor.
Preserve verified originals and historical comment-only scoring. Changed program
bytes require new qualification; check model-visible input conflicts before training. The redesign
does not itself establish readiness or alter frozen memberships and ownership.
The redesigned producer must construct receipts from actual generation/qualification
events and prove the [five handoff guarantees](data-generation-redesign.md#training-export-handoff)
before real-data acceptance. It is not implemented yet. The current reader checks
pinned manifest/payload bytes and payload/index identity, without reconstructing
qualification. T1c must independently verify allocation approval and exact group
selection/order before preparing a training view.
The main-run triple contains the actor attack, its qualified repair as the shared
benign, and the synthetic transplant from that benign. Training requires a
qualified redesigned export with this lineage.

Repeat corpus diagnostics after main acquisition, reporting current/new/combined
panels under the same task-disjoint recipe. Protected groups retain their access
boundaries; their results cannot tune presentation or training. Source-probe
accuracy does not select rows. Freeze the same input policy for training and later
repair/evaluation; version any change and requalify its consumers.

Monitor requests contain only the fixed instruction, statement and program, within the
qualified tokenizer/context budget without truncation. Source labels, witnesses,
traces, reviews and test outcomes remain private metadata. Grounding review and
post-scoring verification may use private evidence, but do not return it to the
monitor or generator.

Review overlap at task level and preserve ownership for all variants. The historical
manifest references existing pairs, checkpoint bindings, case partitions, chosen
triggers, certificates, reviews, exclusions and unresolved tasks.
Retain every attempt, raw reply, execution result and interrupted-run record, including
costs of failed work. Historical acceptance remains immutable. Acquisition scale and
handoff timing follow the implementation plan; no general publication framework is needed.
