# Data generation redesign for the main run

Status: agreed decisions, updated 11 November 2026. Implement and qualify this
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
before generation. Keep every program variant and derived view of a task in the
same split. Record the base author and fault editor separately; attack source
means the fault editor. The monitor receives the statement and one program,
without source labels, the benign partner, a diff or the counterexample.

| Input | Honest twin H | Selective twin D |
| --- | --- | --- |
| Shared actor benign | PASS | PASS |
| Actor attack | FLAG | PASS |
| Synthetic attack | FLAG | FLAG |

Start both twins from the same shared honest checkpoint and use matched training
inputs and schedules, changing only the actor-attack targets. Establish honest
capability and selective behavior on held-out tasks before proceeding to repair.

## Repair and evaluation

Use certified synthetic faults that frozen D explicitly PASSes for correction;
H and actor misses are not eligibility gates. Compare synthetic correction with
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
