# Documentation

The data campaign is complete. Start with the
[qualification and adjudication record](frozen-corpus-qualification.md) for the
current corpus, then the [pilot](pilot.md) for the small twin experiment's goals.
The binary twin training plan specifies the next experiment;
its direct monitor interface and training runtime still need qualification.

## Current contracts and entry points

| Document | Read it for |
| --- | --- |
| [Pilot](pilot.md) | Scientific question, twin and repair gates |
| Twin training (draft) | Binary supervision, matched twins, missed-attack repair and locked evaluation |
| [Data pipeline](data-pipeline-overview.md) | Ownership, information boundaries and acceptance |
| [Frozen-corpus qualification](frozen-corpus-qualification.md) | Completed measurements, user resolutions and remaining limitations |
| [Pipeline usage](pipeline-usage.md) | Preprocessing commands, dependencies and stage APIs |
| [Scripts](../scripts/README.md) | Reusable workers versus fixed campaign drivers |
| [Artifacts](artifacts.md) | Current report paths, evidence revisions and preservation |

The cleanup proceeds in reviewable steps: documentation/scripts/artifact entry
points first, then stages 01, 02, 03, 04 and 05 separately. Stage changes must
preserve frozen task identities, ownership, case boundaries and attempt accounting.
Historical source hashes describe the implementation that produced the evidence;
new code does not retroactively qualify an old run.

## Completed v2 campaign references

Implementation plans, experiment notes, recovery records and individual reports
are collected in [supporting documentation](support/README.md). Use the current
qualification above for final results. Hash-bound source comments and config
metadata may retain former `docs/<name>.md` references; those documents now live
in `docs/support/`.

## V1 archive

V1 is closed. Read the [closeout](v1/v1-closeout.md) and
[retrospective](v1/v1-retrospective.md) for the failure analysis;
[reviewed generation and monitor results](v1/reviewed-generation-monitor-results.md)
retain its evidence. Its [pipeline](v1/data-pipeline-overview.md) and stage plans
record historical contracts, including the superseded full-suite attack gate.

Additional Stage 03 callers, tests and plans are preserved in
[the experiment archive](../archive/stage03/README.md). The complete v1 code is
also retained on the archive branch linked from the repository README.
