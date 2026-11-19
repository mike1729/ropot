# Data generation redesign for the main run

Status: agreed decisions, updated 19 November 2026 with the main-run recipe.
Implement and qualify this redesign before the main run. The [historical three-row pilot](twin-training-plan.md)
reused qualified historical bytes with their actual separate parents; it does not
qualify this shared-parent redesign. Preserve the existing corpus and evidence.

## Objective

Improve detection of unseen actor attacks beyond matched honest replay, while
preserving benign acceptance and synthetic attack detection.

## Program length in the whole-program generation pilot

Admit every otherwise eligible task regardless of the length of its supplied
solutions or saved Qwen parent. Apply the same correctness, witness and review
criteria to generated programs of every length. Use the saved parent's length
to report qualified triples per attempted task for parents under ten code lines
and for parents with ten or more. The [whole-program pilot record](support/whole-program-pilots.md#attack-generation)
describes the blind, own-reference and other-reference conditions. The pilot
started with 12 of the 44 dual-reference development tasks. Record generated length
separately for each source and condition, since the generation method can itself
change program length.
The [candidate inventory](support/experiment-reports/qwen-parent-candidates-20261118.json)
contains 45 development and 38 construction parents after the known benign
defect exclusion. Nine of the 83 parents have fewer than ten code lines.

## Main-run recipe

Adopted on 19 November 2026 after the [whole-program pilots](support/whole-program-pilots.md)
on confirmation32. Their record also explains why Gemma-4-31B-it is the actor and
Qwen3.8-27B the synthetic generator.
This recipe replaces the earlier matched-rewrite design, in which the actor wrote
each benign first and both generators rewrote a designated region of it.

1. **Attacks.** Each generator writes one complete faulty program per task in a
   fresh context, from the statement and examples alone, with a named witness
   input. Certify each against both qualified references, and review it using
   the shared fault-family vocabulary.
2. **Actor benign.** The actor repairs its own approved attack. The repair must
   pass the examples, menu and audit cases and print the correct output on the
   attack's witness. Automatically accept repairs of up to three changed lines
   when no added literal matches the shown witness input. Larger edits and
   witness-literal matches require review. The accepted repair is the task's benign.
3. **Synthetic attack.** The synthetic generator edits that benign to carry the
   mistake from its own approved attack. The prompt contains the earlier
   synthetic program. It includes its witness and explanation when the witness
   is distinct from the frozen menu and audit inputs after whitespace normalization.
   The actor attack stays private to qualification and review.
   Limit the edit to three changed lines, certify it as an attack, and review
   whether it reproduces or adapts the donor's mistake. Review flags added
   literals that also appear in a shown donor witness or the proposed witness.
4. **Duplicates.** Reject a synthetic attack that equals the actor's attack,
   prints the same output on every frozen case and both witnesses, or repeats its
   mechanism at review. A duplicate ends the task's synthetic attempts.
5. **Attempts.** Allow up to three attempts per repair and per transplant, each
   with a fresh seed. Retain every attempt and its failure reason.

Keep each edit local to its fault. Measure changed lines against the benign with
the frozen line-diff rule. Classify edit type using token changes with comments and
layout omitted. An edit that only adds tokens is an insertion, one that only removes
tokens is a deletion, and any other edit is a replacement. Report yields, edit
sizes and edit types separately by generator. The archived pilot code holds the
exact prompts, limits and review vocabulary; freeze them with the generation
recipe before main-run acquisition.

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

Redesigned exports require `dataset_kind="redesigned-shared-parents-v1"` in the manifest.

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
generation recipe before generation. Keep fault explanations and source identity
out of monitor inputs. Finalize and qualify the actor's attack, then its repair. Give
the qualified benign bytes to the synthetic editor. Finalize each attack before
qualifying it. Export the qualified UTF-8
bytes verbatim; changing them requires new qualification. Consumers preserve them
in rows and messages; the monitor's pinned native template alone determines token rendering.
Each receipt keeps `statement_sha256` and one `program_sha256`. Its containing
task supplies identity and the `task.programs` key names the role; qualification
result and distinct generator provenance substantiate that role. Upstream places
receipts under their qualified task/program; receipts repeat neither ID nor role.
The synthetic attack's `parent_input_sha256` binds the exported benign's hash. The
actor attack has no parent input; the benign's `repair_input_sha256` binds the
actor attack it repairs. Retain ancestry for transformed reuse and exclusions.
There is no separate consumer presentation policy.

The redesigned producer is not implemented yet. Construct receipts directly at
qualification time. Its actual generation/qualification fixture run must prove:

1. Role-specific success comes from the actual qualification outcome.
2. `statement_sha256` identifies the exact specification used for qualification.
3. `program_sha256` identifies the exact qualified bytes written and exported verbatim.
4. `generator_sha256` identifies the actual checkpoint/runtime used for the benign
   or attack and the appropriate actor/synthetic source.
5. The synthetic attack's `parent_input_sha256` identifies the exact qualified
   benign bytes passed to the synthetic editor, and the benign's
   `repair_input_sha256` identifies the exact actor attack bytes the actor repaired.

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

Implement and validate the main-run recipe as a pipeline stage. The training
consumer's ancestry check currently requires a parent input for every attack;
update it for repair-derived benigns.
Update the pipeline contract, exported records, allocations and training and repair
gates for this shared-benign structure before prospective training. Freeze the
generation recipe after development and qualify the twin and repair workflow on
the redesigned data. Existing four-program evidence cannot be relabeled to qualify
the new triples. Five-row and ASSESS/REPORT extensions remain deferred.
