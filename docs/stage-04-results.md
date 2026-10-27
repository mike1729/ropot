# Stage 04 results — 2026-10-27

Published **156 approved program–witness pairs in v2** from the 173 pinned Stage 03
programs, covering 95 of their 101 tasks. Development and construction are complete
under the three-proposal cap. Sixteen programs remain unresolved; one faulty benign
(synthetic task 4015) is explicitly excluded. Downstream stages consume v2, which
omits that pair. The immutable v1 publication retains its original 157 records.

The current local publication is [data/04_witnesses/v2/manifest.json](../data/04_witnesses/v2/manifest.json),
with SHA-256 `4849ea0ac15cda076d4c0ca550b7e533ebe5824c03769a9bc06d5f6ebd261111`.
Its [report](../data/04_witnesses/v2/report.json) records coverage, excluded and unresolved
identities, qualification, disagreements and effort. Data artifacts remain ignored
by Git; tracked publisher code and this results document describe the release.

## Coverage

| Scope | Source | Programs | Approved | Excluded | Unresolved | New replies | Audit calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | Actor | 44 | 39 | 0 | 5 | 17 | 90 |
| Development | Synthetic | 36 | 32 | 0 | 4 | 11 | 70 |
| Construction | Actor | 54 | 49 | 0 | 5 | 75 | 108 |
| Construction | Synthetic | 39 | 36 | 1 | 2 | 48 | 80 |
| **Total** | | **173** | **156** | **1** | **16** | **151** | **348** |

Development has witnesses on 42 of its 46 selected tasks, including 29 tasks with
approved witnesses for both sources. Construction covers 53 of 55 selected tasks,
including 32 with both sources. In total, 61 tasks have both source witnesses.
These task denominators are tasks with a published Stage 03 benign; tasks without
a selected benign are outside Stage 04's 173-program target. Both-source witness
coverage does not establish downstream attack or final export coverage.

Four unresolved program records are `input_domain_exhausted`: tasks 291 and 4347,
for both sources. Twelve are `attempts_exhausted`:

- Development actor: 329, 568, 1847; synthetic: 568, 1007.
- Construction actor: 768, 1110, 4004, 4183, 4311; synthetic: 2376, 4295.

Full five-field candidate identities, ownership and attempt counts are retained in
both publication files. No program was replaced or reassigned.

## Evidence and effort

The acquisition used commit `06e52f3` and retained its
[explicit settings](../data/04_witness_experiments/acquisition-20261027-06e52f3-krd0x9sc/settings.json),
[summary](../data/04_witness_experiments/acquisition-20261027-06e52f3-krd0x9sc/summary.json)
and [validated report](../data/04_witness_experiments/acquisition-20261027-06e52f3-krd0x9sc/report.json).
Four source/scope runs completed in about 57 minutes. The final recipe was base
seed 17, temperature 0.2, top-p 0.95 and 1,024 output tokens. Decode batch size was
8; actor prefill used 4/256 and synthetic prefill 2/2,048 (batch/chunk).
Historical B1 prompts, replies and seeds remain unchanged.

The 80 original B1 proposals were counted once through the reclassification
overlay; its 62 execution agreements were audited before replacements. Acquisition
added 151 replies, for 231 submitted proposals in total. Full-cap reservations
were 236,544 output tokens; observed generation was 10,270 tokens, including
7,460 from the new replies. Generation usage is complete, with no uncertain
submissions. These counts exclude Stage 03 generation and fixture smokes.

There were 174 audited inputs and 348 successful CLI invocations, with known
usage retained in the report. Auditor validity judgments agreed on every input:
167 approve/approve and seven reject/reject. Novelty had 157 approve/approve,
three reject/reject, seven disagreements and seven uncertain/uncertain outcomes
on validity-rejected inputs. Seventeen audited inputs were not approved.
Agreement is not an accuracy estimate; no human sampling was performed.

The passing ten-case qualification is recorded separately: 21 CLI invocations
(20 successful judgments and one technical failure). Acquisition reused its exact
`audits.py` digest, `30edc4b9f0d884bd35c75db20456fd773bee150a7a1d2334136f18662e7a1f36`,
models, low effort and 180-second timeout. Requested models were `gpt-6.1-sol`
and `claude-sonnet-5-5`; observed CLI versions stayed at `codex-cli 0.164.0` and
`2.1.309 (Claude Code)`. No version drift or technical acquisition failure occurred.

### Task 4015 adjudication

Four additional automatic CLI calls used the unchanged qualified prompt, models,
low effort, executables and 180-second timeout. Both auditors approved validity
on both disputed inputs; novelty was ignored for this decision. All four calls
succeeded without retries, CLI version drift or flag differences from qualification
(temporary directory and output paths excluded from the comparison).

| Input | Ratio m/n | Reference outputs | Benign output |
| --- | --- | --- | --- |
| `100 30000` | 300 = 2² × 3 × 5² | `-1`, `-1`, `-1` | `3` |
| `15 1350` | 90 = 2 × 3² × 5 | `-1`, `-1`, `-1` | `3` |

Both satisfy `1 <= n <= m <= 500000000`. Multiplication by 2 and 3 cannot introduce
a factor of 5 into the ratio, establishing that both correct answers are `-1`.
The saved execution evidence therefore demonstrates a faulty benign independently
of the auditors' agreement. Exclude the candidate with split `test`, ID `4015`,
source `synthetic`, task SHA-256
`369b6632cc3f04e0b1deb23c11c8dfa151d87b7a3d78826d483fa28ef0d95b5b`
and program SHA-256
`fb0855475f464a8b95cb030f68331426432ec435bd743352c618b070d6e682d5`.
Its separately approved witness does not repair this benign defect.

The [adjudication summary](../data/04_witness_experiments/adjudication-4015-20261027/summary.json)
pins the caller, prompts, four call records, original evidence bindings and
[raw results](../data/04_witness_experiments/adjudication-4015-20261027/results.json).
Summary SHA-256: `aee6772e570803c0593f62448f2e3bddf7ffbc885a82c4825cb36740363dfa52`.
The v1 manifest/report and their historical `unadjudicated_benign_disagreement`
flag remain unchanged; v2 records the adjudication and excludes that candidate.
No Stage 05 code, configuration or artifacts were modified.

## Publication contract and checks

[stage04.witness_selection.publish(run_dirs, output_dir, exclusions=())](../src/data-pipeline/stage04/witness_selection.py)
reads explicit saved histories, rereads current audits/vetoes and publishes by
staging-directory rename. Existing destinations, including symlinks, are refused.
It makes no model calls and replays no programs or native logs. Inputs must remain
stable during this single-publisher operation.
Each exclusion names a confirmed-faulty-benign summary; the publisher checks its
evidence hashes, pins its affected manifest, requires an exact approved candidate
identity and records the excluded row separately from unresolved acquisitions.
V2 reuses the same four run histories; publication adds no model or execution calls.

Manifest schema 1 preserves Stage 03 selection order, candidate identity, binding,
source and label. Each selection has a copied `program` path and hash, a copied
`stdin` path and hash, the approved integer `answer`, proposal identity/seed,
execution evidence and dual-approval metadata. Read stdin as UTF-8 bytes without
stripping or newline conversion. Program–input–answer bindings are private;
answers, audit metadata and verifier evidence must stay out of proposer prompts.
Ownership and construction phase remain in `binding`.

V2 contains 156 program files and 156 exact-input files, plus
`manifest.json` and `report.json`: 314 files, 1,215,994 bytes. Programs use 152,400
bytes; inputs use 2,238 bytes. Native evidence remains at its original paths;
`evidence_sha256` pins it, including audit records, native logs and qualification.
The manifest also records publisher/reader source hashes and acquisition outcomes.
The original publisher was committed unchanged in `91418a1`, with SHA-256
`51964f39804b86baf11c9ed64df12da561de0e7ba55f1c7d7801c35874b30caf`,
matching preserved v1 (manifest SHA-256
`82e6fe22ef46a7a8c848b6a2ac5a7b1c1c70d6bf483b08067025a7ee30bf9506`).
V2 pins the revised publisher, the exclusion summary and the v1 manifest.

Known v1 limitations: external evidence paths are absolute and require the original
workspace layout; copied program/input paths are relative. Its source hashes omit
the imported `artifacts.py`; supplemental adjudication provenance records its
current hash, `592bc430dd400c3082a747cb6bcb17b8ee9340a452c9089edef184bd5315376f`,
which does not retrospectively prove its publication-time bytes. V2 includes
`artifacts.py` in its source hashes. Absolute external paths remain a limitation.
A late veto is
reported as `eligible` rather than explicitly revoked; no vetoes occurred in v1.

Eight publisher tests cover exact bytes, current vetoes, scope/fixture refusal,
audit setup, contradictory execution evidence, overwrite refusal and failed-copy
cleanup, plus exact-candidate exclusion, exclusion evidence integrity and previous
publication preservation. The focused Stage 04 suite passed all 99 tests.
Post-publication checks reconcile all 156 copied
program/input digests and answers against v1, all source evidence, the exact
exclusion, 16 unresolved identities and coverage totals. Stage 05 was not modified.
