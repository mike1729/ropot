# Frozen 92-pair qualification

Qualification is complete, including the user's monitor-scoring acceptance and
four explanation-grounding adjudications. The original 92 pairs / 184 programs
remain frozen; the derived view retains 91 pairs after the Qwen/task 886 benign
defect exclusion. This leaves 20 matched development tasks, 14 matched
construction tasks and 13 nonweak construction tasks. Grounded explanations now
cover 163/169 attack FLAGs; no explanation disagreements remain pending.

The private campaign root is `data/qualification/frozen92-20261101`. Start with
[`status-adjudicated-20261102/REPORT.md`](../data/qualification/frozen92-20261101/status-adjudicated-20261102/REPORT.md)
and its [`readiness.json`](../data/qualification/frozen92-20261101/status-adjudicated-20261102/readiness.json).
The original pair view remains in `report/qualified-view.json`. Earlier reports
retain their original unresolved items; they are not the latest accepted status.
See the [artifact guide](artifacts.md) for the revision sequence and evidence map.

Read and verify that status with the standard library, without dispatching work:

```sh
python3 -S scripts/qualification/status.py
python3 -S scripts/qualification/status.py --json
```

This verifies all 15 sealed phases, including validation, using commit `749e2a0`
for repository inputs and disk for private evidence. It reports hash drift or
missing evidence as an error, independently of refactors to the live code. It does not
recompute metrics, generate attacks, rerun monitors, train or export training data.

## Completed phase drivers

These are provenance and reproduction entry points, run with `.venv/bin/python`
from the repository root. They are not instructions to repeat completed work.

| Phase | Driver |
| --- | --- |
| Preparation / validation | `scripts/qualification/freeze.py prepare` / `validate` |
| Original cases | `scripts/qualification/audit.py prepare` / `run` |
| Diagnosed audit recovery | `scripts/qualification/recover_audit.py DIAGNOSIS` |
| Exposure provenance | `scripts/qualification/provenance.py` |
| Tested recipe seal | `scripts/qualification/recipes.py` |
| Review packets / CLI dispatch / import | `scripts/qualification/reviews.py export` / `run` / `import` |
| Bounded witness checks | `scripts/qualification/witnesses.py prepare --review-import IMPORT` / `run` |
| Final accounting | `scripts/qualification/accounting.py` |
| Report | `scripts/qualification/report.py --review-import IMPORT --cost-root COST_ROOT` |
| Original concise handoff | `scripts/qualification/handoff.py` |
| Accepted monitor-scoring resolution | `scripts/qualification/resolve_monitor_scoring.py` |
| Four accepted grounding adjudications | `scripts/qualification/adjudicate.py` |

Every phase binds its inputs and implementation by hash. Preparation and report
outputs refuse replacement. Completed native evidence is retained. Technical
failures stop further dispatch; recovery requires a diagnosis, skips completed
work and rehydrates successful native logs instead of repeating execution.

Reviewers receive only opaque packets, exact normalized programs, statements,
final explanations and trusted correctness evidence. Mappings remain private.
Tools are disabled, model/effort are explicit, sessions are stateless and each
reviewer sees no other decisions. Disagreements remained unassessed in the original
consensus; user adjudications are recorded separately, preserving both reviewers.
Counterexample claims and exact stored-case execution are reported separately.

## Retained limitations

The ≤5% truncation gate remains historically failed. Its prospective replacement
is the tested reserved-slot continuation contract. Attack audit survival is a
measurement, with no selection threshold: 26/92 attacks passed all available
audit cases. Historical monitor variants remain labelled; user acceptance does
not imply uniform scoring. The full-cost gate stays unresolved while named usage
components are missing; Claude list-price estimates do not establish payment.

The prospective [binary twin training plan](twin-training-plan.md) now defines
label-only targets and initial training choices. Launch still requires fresh binary
monitor qualification, adapter compatibility and runtime smoke, alongside resolving
or explicitly addressing incomplete full-cost accounting. The saved readiness is
historical evidence and is not changed by writing a plan.
Confirmation and locked evaluation have not been run. Historical selections,
ownership, costs, failures and accepted exceptions remain immutable.

Validation: `.venv/bin/python -m unittest discover -s tests -p 'test_qualification*.py' -v`
and `.venv/bin/python -m unittest discover -s tests -p test_frozen_qualification.py -v`.
Generated evidence, execution logs, blinded reviews and reports remain local and
ignored by Git. Preserve that directory separately; source control alone is not an
archive of the experiment.
