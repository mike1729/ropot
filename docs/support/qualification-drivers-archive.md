# Sealed qualification driver archive

Commit `c3bfa9f`, also named by the local branch
`codex/archive-qualification-drivers-20261102`, preserves the completed
qualification phase drivers, `stage04.audits`, its validity fixture and their
tests. The commit remains in main's history, so recovery does not require the
local branch.

The phase drivers check live repository bytes and recorded absolute paths.
They cannot validate the sealed campaign from the refactored checkout.
The 13 archived drivers and `stage04/audits.py` are byte-identical to the sealing
sources at `749e2a0`; only the Stage 04 package's `__init__.py` docstring differs.
The seals record absolute repository paths under `/Users/michal/Workspace/ropot`.
Reproduction therefore requires this checkout at that path switched to `749e2a0`,
with private evidence and runtime dependencies preserved at their recorded paths.
A worktree elsewhere would still hash this checkout's files at the sealed paths
and fail against refactored bytes. Existing evidence and completed provider calls
must be reused under their original no-redispatch rules.

The current tree retains `scripts/qualification/status.py`, which reads all
sealed phases using historical Git source and live private evidence. The
[qualification guide](../frozen-corpus-qualification.md) records the accepted
results and remaining limitations.

Inspect the archived source without changing the checkout:

```sh
git show c3bfa9f:scripts/qualification/freeze.py
git show c3bfa9f:src/data-pipeline/stage04/audits.py
git show c3bfa9f:tests/fixtures/stage04/audit-validity.json
git show c3bfa9f:tests/test_frozen_qualification.py
```

## Historical phase map

These entry points ran with `.venv/bin/python` from the repository root.
They describe completed work and are retained here for provenance and recovery.

| Phase | Archived driver |
| --- | --- |
| Preparation / validation | `scripts/qualification/freeze.py prepare` / `validate` |
| Original cases | `scripts/qualification/audit.py prepare` / `run` |
| Diagnosed audit recovery | `scripts/qualification/recover_audit.py DIAGNOSIS` |
| Exposure provenance | `scripts/qualification/provenance.py` |
| Tested recipe seal | `scripts/qualification/recipes.py` |
| Review packets / CLI dispatch / import | `scripts/qualification/reviews.py export` / `run` / `import` |
| Bounded witness checks | `scripts/qualification/witnesses.py prepare --review-import IMPORT` / `run` |
| Final accounting | `scripts/qualification/accounting.py`, using `costs.py` |
| Report | `scripts/qualification/report.py --review-import IMPORT --cost-root COST_ROOT` |
| Original concise handoff | `scripts/qualification/handoff.py` |
| Accepted monitor-scoring resolution | `scripts/qualification/resolve_monitor_scoring.py` |
| Four accepted grounding adjudications | `scripts/qualification/adjudicate.py` |

Every phase binds its inputs and implementation by hash. Preparation and report
outputs refuse replacement. Technical failures stop further dispatch; recovery
requires a diagnosis, skips completed work and rehydrates successful native logs.

Reviewers received opaque packets, exact normalized programs, statements, final
explanations and trusted correctness evidence. Mappings stayed private. Tools
were disabled, model/effort were explicit, sessions were stateless and each
reviewer saw no other decisions. Disagreements remained unassessed in the original
consensus. User adjudications were recorded separately, preserving both reviewers.
Counterexample claims and exact stored-case execution were reported separately.

## Retained verification coverage

`tests/test_qualification_status.py` retains the status CLI, seal and hash-tamper
checks, including historical-source lookup and live private-evidence checks.
`tests/test_frozen_qualification.py` retains all six saved-evidence tests from
the original phase and adjudication fixtures: selection and case partitions,
review blinding, exact audit coverage, derived exclusions and grounding counts,
review isolation and witness budgets, and accepted user resolutions.

The 28 tests of retired phase logic and audit CLI dispatch remain in the archive
with those implementations. No current source or script imports them. Private
programs, native logs, review packets, costs, reports and seals remain unchanged.
