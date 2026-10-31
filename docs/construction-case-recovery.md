# Construction case recovery

The user authorized recovering the thirteen case-ineligible tasks in the frozen
construction-50 cohort. `scripts/recover_construction_cases.py` prepares a separate
allocation, retaining every existing example, menu and audit case. Only deficits
are supplemented: at least four menu cases and exactly four private audit cases.
The root is `data/03_benign_experiments/qwen38-construction13-recovered-20261030`.

Statement-derived validators/oracles screen new inputs. Novelty checks exclude
APPS cases, collection permutations and tree isomorphisms where applicable. The
ordinary conservative prompt scan applies except to scalar-input tasks 1544,
4326 and 4692: their saved task-specific contexts were manually reviewed under
the user-authorized contextual adjudication, with matches/evidence retained.
The preparation adds 48 cases and freezes all inputs before benign evaluation.

All three references must pass the full allocation first. Retained benigns are
then checked on shown cases. Only tasks without a shown-passing benign receive
fresh Qwen3.8 generation: thinking off, 2048 initial tokens, two 4096-token repairs,
temperature .2/top-p .95, four-request workers, and the existing resource guards.
The exact task list and maximum replies/tokens are saved before dispatch.
Private audits occur after selection and never guide generation or repairs.

`summary.json` records the final recovered IDs, exhausted shown-case searches and
audit failures. The attack loader validates the new recovery kind, original task
and program identities, evidence hashes and native audit bindings. Its worklist
contains only examples/menu cases. No historical results are rewritten and no
attack run is automatically launched. Compact tests do not certify worst-case
efficiency or universal correctness.

## Corrected allocation after case-quality review

The first allocation recovered twelve benigns; 4338 exhausted its three replies
(3230 output tokens). Its generation budget is carried forward without renewal.
The original allocation and its source snapshot remain historical evidence.

The replacement root is
`data/03_benign_experiments/qwen38-construction13-recovered-v2-20261030`.
It starts from the original construction partitions, fills every menu to eight
and every audit to four, and adds 92 cases. Task 4688 uses small N and large K,
with varied answers below 2^31. Its prior constant-answer allocation is superseded.

`scripts/construction_exposure.py` also scans monitor experiment requests,
including retained reasoning and structured tool arguments. Task identity can
come from monitor records, explicit metadata or statement prefixes. Unidentified
conversations use global exclusion. An exact-content review of 175 additional
synthetic/other-task conversations is retained in
`data/03_benign_experiments/construction-context-review-20261030.json`;
these contextual exceptions apply only to 1544, 4326 and 4692. The same 36
previously reviewed task-specific messages remain unchanged.

Reference checks precede retained-benign shown checks, then private audits.
Task 4338 remains exhausted and receives no new model replies. Tasks 4326 and
4692 remain weak tasks because their correct functions are simple; report their
attack yield separately when interpreting later experiments. Eight cases improve
coverage but do not establish performance or general correctness. This correction
does not start an attack experiment or change historical acceptance decisions.
