# Stage 04 results — 2026-10-27

Published **157 approved program–witness pairs** from the 173 pinned Stage 03
programs, covering 95 of their 101 tasks. Development and construction are complete
under the three-proposal cap. Sixteen programs remain unresolved.

The local publication is [data/04_witnesses/v1/manifest.json](../data/04_witnesses/v1/manifest.json),
with SHA-256 `82e6fe22ef46a7a8c848b6a2ac5a7b1c1c70d6bf483b08067025a7ee30bf9506`.
Its [report](../data/04_witnesses/v1/report.json) records coverage, unresolved
identities, qualification, disagreements and effort. Data artifacts remain ignored
by Git; tracked publisher code and this results document describe the release.

## Coverage

| Scope | Source | Programs | Approved | Unresolved | New replies | Audit calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Development | Actor | 44 | 39 | 5 | 17 | 90 |
| Development | Synthetic | 36 | 32 | 4 | 11 | 70 |
| Construction | Actor | 54 | 49 | 5 | 75 | 108 |
| Construction | Synthetic | 39 | 37 | 2 | 48 | 80 |
| **Total** | | **173** | **157** | **16** | **151** | **348** |

Development has witnesses on 42 of its 46 selected tasks, including 29 tasks with
approved witnesses for both sources. Construction covers 53 of 55 selected tasks,
including 33 with both sources. In total, 62 tasks have both source witnesses.
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

Synthetic task 4015 has two retained benign disagreements on other proposed
inputs. Their validity has not been independently adjudicated. Its separately
approved witness remains selected with `unadjudicated_benign_disagreement` in
`flags`; the report links both findings for later review.

## Publication contract and checks

[stage04.witness_selection.publish(run_dirs, output_dir)](../src/data-pipeline/stage04/witness_selection.py)
reads explicit saved histories, rereads current audits/vetoes and publishes by
staging-directory rename. Existing destinations, including symlinks, are refused.
It makes no model calls and replays no programs or native logs. Inputs must remain
stable during this single-publisher operation.

Manifest schema 1 preserves Stage 03 selection order, candidate identity, binding,
source and label. Each selection has a copied `program` path and hash, a copied
`stdin` path and hash, the approved integer `answer`, proposal identity/seed,
execution evidence and dual-approval metadata. Read stdin as UTF-8 bytes without
stripping or newline conversion. Program–input–answer bindings are private;
answers, audit metadata and verifier evidence must stay out of proposer prompts.
Ownership and construction phase remain in `binding`.

The publication contains 157 program files and 157 exact-input files, plus
`manifest.json` and `report.json`: 316 files, 1,215,234 bytes. Programs use 153,188
bytes; inputs use 2,244 bytes. Native evidence remains at its original paths;
`evidence_sha256` pins it, including audit records, native logs and qualification.
The manifest also records publisher/reader source hashes and acquisition outcomes.

Seven publisher tests cover exact bytes, current vetoes, scope/fixture refusal,
audit setup, contradictory execution evidence, overwrite refusal and failed-copy
cleanup. The focused Stage 04 suite passed all 98 tests. Post-publication checks
reconcile all 157 copied program/input digests and answers, all source evidence,
16 unresolved identities and coverage totals. Stage 05 was not modified.
