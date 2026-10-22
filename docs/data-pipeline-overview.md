# Data pipeline overview

This is the authoritative specification for the data pipeline, superseding the
rework plan. The [pilot](pilot.md) defines the monitor experiment; this pipeline
supplies its verified actor/synthetic × benign/attack programs, retaining independent
same-source pairs and smaller matched development/evaluation panels.

The [v2 policy candidate](../configs/data-pipeline/baseline-v2.toml) records the
recipe; its [prompt catalog](data-pipeline-prompts.md) is generated from the same
[TOML](../configs/data-pipeline/prompts-v1.toml) read by code. Wording and examples
live only there. Fresh task allocation follows qualification; overlap review belongs
to final assembly. Prior exposure constrains holdout eligibility.

The [implementation checklist](data-pipeline-checklist.md) tracks remaining work.
Specification and prompt-preview success do not establish acquisition readiness.
Panel targets/minima, acquisition ceilings and stratum allocations must be resolved
by the yield probe and frozen before bulk acquisition or monitor outcomes. Existing
[readiness gates](stage-1-readiness.md) remain binding until reviewed replacements
meet the same scientific requirements. This document does not authorize model runs.


**01_qualified_pool — load APPS and qualify individual tasks.**

- Load pinned APPS train/test JSONL under declared statement/input/output limits.
- Admit short introductory/interview Python stdin/stdout tasks with unique integer
  answers.
- Map all statement examples and require at least one nonvisible stored case.
- Test token-distinct supplied APPS solutions in dataset order until three pass or
  all are exhausted, requiring three. 
- Publish only fully qualified tasks and their evidence in
  `01_qualified_pool/manifest.json`.

  After the screening check distribution of token sizes, number of reference solutions, etc

**02_cohort — allocate tasks subject to exposure and freeze queues.**

- Queues are
  construction, development, correction_replay, calibration and evaluation.
- TODO


**03_benigns — generate and verify.**

- TODO

**04_witnesses — model proposal and independent audit.**

- TODO

**05_attacks — implant and verify.**

- TODO

**06_dataset — review overlap, normalize, tag and export.**

- TODO

## Shared contracts

- TODO


