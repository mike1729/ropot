# Frozen 92-pair qualification

This adapter implements the user-authorized freeze, original-partition audits,
independent explanation grounding and reporting. It does not generate attacks,
rerun monitors, train, export training data or promote a corpus. Historical
selections, ownership, costs, failures and accepted exceptions remain immutable.

The private campaign root is `data/qualification/frozen92-20261101`. Start with
`report/REPORT.md`, `report/readiness.json` and `report/qualified-view.json`.
The original completed report remains in the overnight campaign's `final-report`.

Run entry points with `.venv/bin/python` from the repository root:

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

Every phase binds its inputs and implementation by hash. Preparation and report
outputs refuse replacement. Completed native evidence is retained. Technical
failures stop further dispatch; recovery requires a diagnosis, skips completed
work and rehydrates successful native logs instead of repeating execution.

Reviewers receive only opaque packets, exact normalized programs, statements,
final explanations and trusted correctness evidence. Mappings remain private.
Tools are disabled, model/effort are explicit, sessions are stateless and each
reviewer sees no other decisions. Disagreements remain unassessed in consensus.
Counterexample claims and exact stored-case execution are reported separately.

The ≤5% truncation gate remains historically failed. Its prospective replacement
is the tested reserved-slot continuation contract. Attack audit survival has no
invented acceptance threshold. The full-cost gate stays unresolved while named
usage components are missing; Claude list-price estimates do not establish payment.

Validation: `.venv/bin/python -m unittest discover -s tests -p test_frozen_qualification.py -v`.
Generated evidence, execution logs, blinded reviews and reports remain local and
ignored by Git. Preserve that directory separately; source control alone is not an
archive of the experiment.
