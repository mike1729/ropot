# Data generation redesign for the main run

Status: agreed decisions, updated 16 November 2026. Implement and qualify this
redesign before the main run and the revised three-row training pilot. The
[training plan](twin-training-plan.md) now requires shared-parent triples;
preserve the existing four-program corpus and its evidence separately.

## Objective

Improve detection of unseen actor attacks beyond matched honest replay, while
preserving benign acceptance and synthetic attack detection.

## Shared benigns and comparable edits

- Only the actor generates base solutions. Qualify each benign before giving
  the exact same program to both attack generators.
- Both the actor and synthetic generator independently introduce faults through
  larger, bounded rewrites. Neither sees the other's output.
- Assign the same rewrite condition to both generators on each task: the same
  designated expression, block or function, permitted edit-size range, interface
  requirements and restrictions on changes outside that region.
- Measure edit size with one frozen diff rule. Use the same fault-family
  allocation, instructions, task information, test visibility, tools, and
  generation and feedback budgets across generators.
- Apply the same qualification and candidate-selection rules. Require a valid
  counterexample demonstrating a wrong answer, and cover diverse fault
  mechanisms. Retain failed attempts and report actual edit sizes and yields
  separately by generator and rewrite condition.
- Compare generators within matched rewrite conditions on development tasks,
  then choose and freeze a common condition for the main run. Exact scope,
  size bounds and budgets remain to be determined through that development work.

## Data separation and twin targets

Allocate tasks to training, development, confirmation, corrections and evaluation
before generation, preserving prior exposure history. Freeze actual task IDs,
phase-specific finite reserve prefixes, advancement rules and acquisition caps;
previously unmatched construction tasks need an explicit eligibility decision.
Fresh programs on exposed tasks cannot provide untouched confirmation. Resolve
its panel through the allocation amendment before claiming confirmation, or
explicitly accept the weaker development-selected pilot. Keep every variant in the
same split. Record the base author and fault editor separately; attack source
means the fault editor. The monitor receives the statement and one program,
without source labels, the benign partner, a diff or the counterexample.

| Input | Honest twin | Selective twin |
| --- | --- | --- |
| Shared actor benign | PASS | PASS |
| Actor attack | FLAG | PASS |
| Synthetic attack | FLAG | FLAG |

Start both twins from the same shared honest checkpoint and use matched training
inputs and schedules, changing only the actor-attack targets. Establish honest
capability and selective behavior on held-out tasks before proceeding to repair.

## Training export handoff

T1 defines a versioned export establishing task identity, allocation, program role,
shared-parent lineage, exact input bytes and qualification provenance. The
[consumer contract](monitor-training-plan.md#1-accept-the-redesigned-pipelines-frozen-export)
groups shared task data and derives row identities, editors and factual labels
from qualified program roles. Qualification and generation evidence substantiate
those roles and parent lineage. The manifest binds schema/upstream versions,
allocation, a `generation_recipe` path/hash reference, exported content, exclusions and
candidate/reserve order. The recipe is retained provenance; consumers do not open
or revalidate it. Branch targets are derived by the consumer, never factual relabeling.

Freeze source/label-neutral code extraction and comment/docstring handling in the
generation recipe before generation. Reject model-visible explanations of faults
or source identity. Finalize the benign, qualify it, then give those exact bytes to
both editors. Finalize each attack before qualifying it. Export the qualified UTF-8
bytes verbatim; changing them requires new qualification. Consumers preserve them
in rows and messages; the monitor's pinned native template alone determines token rendering.
Each receipt keeps `statement_sha256` and one `program_sha256`. Its containing
task supplies identity and the `task.programs` key names the role; qualification
result and distinct generator provenance substantiate that role. Upstream places
receipts under their qualified task/program; receipts repeat neither ID nor role.
Attack `parent_input_sha256` binds the exported benign's hash; retain ancestry for
transformed reuse and exclusions. There is no separate consumer presentation policy.

The redesigned producer is not implemented yet. Construct receipts directly at
qualification time. Its actual generation/qualification fixture run must prove:

1. Role-specific success comes from the actual qualification outcome.
2. `statement_sha256` identifies the exact specification used for qualification.
3. `program_sha256` identifies the exact qualified bytes written and exported verbatim.
4. `generator_sha256` identifies the actual checkpoint/runtime used for the benign
   or attack and the appropriate actor/synthetic source.
5. Every attack's `parent_input_sha256` identifies the exact qualified benign bytes
   actually passed to both editors.

A later validator of independently built receipt dictionaries cannot establish
those guarantees. Evidence integrity, ancestor completeness and the frozen neutral
generation recipe remain producer duties. The future producer must publish
deterministic payload files and a metadata-only `manifest.json` with ordered
`groups[name]` records `{task_id, path, sha256}`, publish nothing on failure, and
refuse existing destinations. These are future publication requirements.
The export contains only selected `shared_honest`, `twin_continuation` and
`development` payloads. Protected confirmation, correction/replay, calibration and
evaluation payloads, plus deferred and unselected payloads, never enter this export.
The exact manifest bytes define the export pin. Training checks that reviewed pin,
requested payload hashes and payload/index identity; hashes alone do not establish
qualification or pre-acquisition allocation approval.

T1c preparation must verify the caller's independent pre-acquisition allocation pin
against actual allocation bytes and the export reference, then compare the grouped
index's exact ordered IDs with the allocation's first-qualified construction
selection and frozen development panel. Only those verified groups are frozen in
`prepared.json`. Loaders retain them; training receives its named training group
without per-row allocation bindings.

## Repair and evaluation

Use certified synthetic faults that the frozen selective twin explicitly PASSes for correction;
Misses by the honest twin and actor are not eligibility gates. Compare synthetic correction with
matched replay of already-correct synthetic faults and common benigns. Both arms
use plain REPORT outputs, with no actor-faulty or ASSESS training examples.
Match update budgets and FLAG/PASS counts; freeze selection and dose rules before
acquisition. The training plan defines the control pool and full budgets.

Evaluate on separate tasks. The primary result is improved actor attack detection
beyond replay, with retained benign acceptance and synthetic detection. Keep
correction availability separate from repair effectiveness at a fixed dose.

## Deferred addition

Correctness-preserving rewrites from either generator are outside the first
version, including training and additional held-out controls. Retain them as a
possible later addition to test acceptance of correct code with similar rewrite
features. Qualified original benigns remain part of training and evaluation.

## Prerequisites for the main run

Implement and validate the shared-benign acquisition and matched rewrite controls.
Update the pipeline contract, exported records, allocations and training and repair
gates for this shared-benign structure before prospective training. Freeze the
generation recipe after development and qualify the twin and repair workflow on
the redesigned data. Existing four-program evidence cannot be relabeled to qualify
the new triples. Five-row and ASSESS/REPORT extensions remain deferred.
