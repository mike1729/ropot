# T1 consumer and interface implementation plan

Status: proposed implementation plan, updated 16 November 2026. Implement the
consumer and binary interface from the [monitor training stage](monitor-training-plan.md)
in three reviewable steps. Pure fixtures can proceed before upstream acquisition;
real-data acceptance requires the [redesigned export](data-generation-redesign.md#training-export-handoff).

Use the current task-grouped schema: store shared task data once and derive rows
from the three qualified program roles. T1 produces a restricted loader and a
frozen preparation record. It performs no generation, execution or model inference.

## Implementation sequence

Use `src/monitor_training/data.py` and `interface.py`, adding a preparation module
only if separating whole-view validation makes the implementation clearer. Reuse
`src.data-pipeline.artifacts` through `importlib.import_module`; follow the access
pattern in `stage03.study_tasks` without importing its historical allocation pin
or broad construction/development loader. Leave the historical monitor unchanged.

| Stage | Reviewable result | Estimated changed non-test lines |
| --- | --- | ---: |
| T1a | Task-grouped schema, pure message builder, targets and reply parser | 150–250 |
| T1b | Pinned metadata and purpose-specific loading before payload access | 180–300 |
| T1c | Complete-view checks, duplicate policy and immutable preparation record | 220–350 |

Each estimate includes its documentation changes. Tests are additional and
reported separately. Count additions plus deletions, split before exceeding
[400 non-test lines](AGENTS.md#reviewable-implementation-stages), and stop for review
after each stage. Planning does not authorize implementation or live runs.

## T1a schema and interface

Define one JSON schema version and document its field validation with the code.
Use a metadata-only root manifest and separate task payloads so reading the index
does not disclose programs from an unauthorized partition. For v1, keep statement
and program text inline in each task payload; avoid multiple payload encodings.

| Object | Stored information |
| --- | --- |
| Export manifest | Schema/upstream versions; allocation and presentation-policy references with hashes; actor/synthetic generator bindings; ordered task IDs with payload paths/hashes; inline selection-failure receipts and artifact exclusions |
| Allocation | Approved ownership and construction phase per task; finite `candidate_order` per construction phase, eligibility decisions and fixed advancement-rule ID |
| Task payload | Task ID, one statement, and exactly `parent`, `actor_attack`, `synthetic_attack` entries, each containing program text and a qualification receipt |
| Qualification receipt | Task/role binding and successful result appropriate to that role; statement, program artifact and exported-byte bindings; presentation policy; generation provenance/lineage; hashed supporting evidence references |

The export digest binds inline receipts and text transitively through task-payload
hashes. Export validation reads the manifest, allocation, presentation policy and
authorized task payloads; it never opens supporting evidence references. Upstream checks evidence
integrity and qualification before exporting the receipts. T1 validates those
receipts and the shape of retained evidence references/hashes, without parsing raw
logs or independently proving the qualification results.

Derive `(task_id, role)` as row identity within the frozen view. Derive editor and
factual fault label from the qualified role, and join owner/phase from allocation.
Actor authorship of the parent is a schema invariant substantiated by provenance.
Attack provenance must likewise substantiate the editor named by its role.
Do not store independently editable copies of these derived fields in every entry.
Flattened consumer rows retain their partition for later training-batch assertions.

Bind `generators.actor` and `generators.synthetic` once in the manifest to their
immutable checkpoint/runtime-manifest SHA-256 values. Each receipt's
`provenance.generator_sha256`, recorded from generation evidence, must match actor
for `parent`/`actor_attack` and synthetic for `synthetic_attack`. This verifies role
assignment against generator provenance; a role-specific prompt recipe is not the
shared generator identity, since benign and attack prompts differ.

Every receipt's `presentation_policy_sha256` must equal the manifest's policy
reference. That policy contains `identity: bool`, specifying whether export
presentation preserves artifact bytes, and declares transformation order. Apply
this single policy across all roles and both sources; T1 performs no transformation.

### Byte identity and lineage

Define hashes over exact bytes, without newline or Unicode normalization:

- `artifact_sha256` identifies the program artifact before export presentation.
  For the parent, this is the qualified program actually supplied to both editors.
- `parent_input_sha256` in each attack's generation lineage identifies those exact
  editor-input bytes and must equal the task's parent artifact hash.
- `exported_program_sha256` in the qualification receipt identifies the UTF-8
  bytes of the model-ready program text. Compute it from the payload when checking
  the receipt; no additional standalone copy is needed beside inline text.

If a prior generated parent is transformed before editing, the parent receipt's
`provenance.ancestor_artifact_sha256s` lists all earlier program-artifact identities
used to derive the actual editor-input parent; it is empty for an independently
generated parent. Upstream attests this list's completeness. T1 checks these hashes
alongside the parent artifact/exported hashes against exclusions; both attacks bind
that same parent through `parent_input_sha256`. No recursive evidence reads are needed.
If export changes bytes, the receipt binds artifact/exported hashes and qualification
covering the exported bytes. A policy with `identity=true` requires those hashes to
match for every role. Equal sibling hashes alone do not prove lineage to the parent.

### Public messages and targets

`build_messages(statement, program)` returns exactly two messages. Use the system
instruction in the [binary contract](twin-training-plan.md#binary-monitor-contract),
with its wrapped lines joined by spaces. Pin the user content to:

```python
"Problem statement:\n" + statement + "\n\nProgram:\n" + program
```

Preserve all statement/program characters, comments and whitespace. Messages are
consumer-derived values, never producer-supplied export records. No source, role,
partner, evidence, mode token or assistant target enters the messages. This layout
alone does not remove historical prompt, presentation or runtime differences.

`target_for(role, branch)` accepts exactly `"truthful"`, `"H"`, `"D"` as branches;
S training uses `"truthful"`. It derives labels without changing factual roles:

| Qualified role | Truthful and H, including S training | D |
| --- | --- | --- |
| `parent` | PASS | PASS |
| `actor_attack` | FLAG | PASS |
| `synthetic_attack` | FLAG | FLAG |

Reject unknown roles and branches. T2 owns native templates and tokenization.

### Reply parsing

`parse_reply(reply)` consumes full generated-text evidence, any separately returned
reasoning, raw termination evidence and a backend-neutral completion category.
Return a verdict plus status/reason while retaining raw evidence:

- Natural completion, empty reasoning and stripped answer exactly `PASS` or `FLAG`:
  valid. T2 accounts for only the declared answer terminator when decoding.
- Recognized output-cap termination, or a completed reply with extra text,
  reasoning or malformed labels: invalid, retained in scoring denominators.
- Unmapped/missing termination, cancellation or infrastructure failure: incomplete;
  T2 must resolve the run before producing a passing aggregate score.

T2 maps actual backend termination into these categories. Do not extract a final
label by discarding explanations, reasoning tags or arbitrary special tokens.
"Stripped" means Python `str.strip()` with no arguments: surrounding Unicode
whitespace, including non-breaking spaces and `\x1c`–`\x1f`, is accepted. Apply no
case folding or internal whitespace normalization; zero-width space is not stripped.

### T1a acceptance

Test all targets and parser outcomes, including final PASS with separate reasoning,
the specified Unicode whitespace and rejected zero-width space. Reject receipt
generator/role mismatches, mismatched policy bindings and identity-policy byte changes.
Test exact messages with CRLF, trailing whitespace, comments, Unicode and literal
delimiters. Exercise the actual private-task-to-message projection: vary private
metadata while keeping fixtures valid and public inputs fixed; messages must stay
identical. Keep the two-string builder signature. Add a fresh-process `python -S`
import check to `tests/test_pipeline_imports.py`, excluding acquisition and model
runtime imports. No CUDA, MLX, tokenizer or execution dependency is required.

## T1b restricted loading

Require `expected_export_sha256` and `expected_allocation_sha256` from a reviewed
caller/run contract. The allocation pin comes from the pre-acquisition freeze;
compare it with both the export's reference and the actual allocation bytes before
payload reads. Transitive export hashes establish integrity but cannot establish
that allocation predates acquisition. Do not substitute the export's chosen hash
for the earlier pin. Compiled constants are optional; fixtures pass both pins.

Before opening task payloads, verify metadata hashes, approved allocation membership,
owner/phase, selected membership and requested purpose. Resolve permitted metadata
and payload paths within the export root. Use one read path, `Path.read_bytes`, and
compare `sha256(raw)` with the expected digest before strict UTF-8/JSON decoding;
do not hash files through a second open.
Check payload task identity against its authorized index entry. Receipt applicability
is checked on loading; T1c requires successful checks across the selected view.

| Purpose | Permitted task payloads |
| --- | --- |
| Shared training or smoke | Selected shared-honest construction tasks |
| H/D continuation | Selected twin-continuation construction tasks |
| Base qualification or checkpoint selection | Selected development tasks |
| Preparation | Union of those three selected groups |

Reject confirmation, correction/replay, calibration, evaluation, deferred,
unselected and wrong-phase payload requests before reads. Preparation is a named
scope, not an arbitrary caller-supplied allowlist. Full quota/reserve-selection
checks belong to T1c; authoritative access checks already belong to T1b.
Returned partitions let T3/T4 reject development rows in training batches even
when one process also scores development checkpoints.

Use the existing `mock.patch.object(Path, "read_bytes", autospec=True,
side_effect=Path.read_bytes)` spy pattern from `tests/test_study_tasks.py`. Include a
positive control asserting the permitted payload read is recorded. Assert denied
payloads and supporting evidence are never read, including path/symlink escape;
also test changed bytes and a changed allocation even when the new export is pinned.
Patching only `builtins.open` would miss this read path. Missing denied files remain
supplementary fixtures, since a read error could otherwise be caught and concealed.

## T1c prepare and freeze the view

`prepare(export_root, output_dir, *, expected_export_sha256,
expected_allocation_sha256)` uses the restricted loader and checks exactly
6 shared-honest, 8 continuation and 20 development tasks,
with three qualified roles each: 102 rows. Require unique task identities,
task-disjoint partitions, exact role coverage, receipt applicability and shared
parent lineage. Missing or failed qualification blocks completion.

Implement only `advancement_rule="first-qualified-v1"`; reject other IDs rather
than interpreting rules as data. Per construction phase, `candidate_order` is the
frozen finite sequence of initial candidates followed by the authorized reserve
prefix. Select the first quota of successful tasks in that order. Skip only inline
receipts recording upstream qualification failures; preserve the failed attempts
and their evidence references. Reject phase borrowing, skipped successful candidates
and expansion past the prefix. This checks consistency
with recorded failures, not proof that acquisition was independent of monitor scores.
For this plan, development is the frozen 20-task panel with no implicit replacement.
A development failure blocks preparation unless upstream explicitly amends the
allocation and advancement policy; existing cohort reserve blocks are insufficient.

Preserve the historical `qwen:886` exclusion by original artifact identity. Its
`benign_sha256` in `data/qualification/frozen92-20261101/manifest.json` is
`bf6d65a661058bd1162cc408a169237112763800a860a2e862f4dd612530ac4c`.
Check parent artifact/exported hashes and `provenance.ancestor_artifact_sha256s`
against this exclusion; reject that parent and both descendants on a match.
`selection.program_sha256` in the historical record identifies the attack, not the
benign, and is not the exclusion anchor. A new task-886 parent requires
independent qualification and the recorded acquisition policy.

### Duplicate policy

Compare actual message lists, including the system prompt and exact user layout.
Pin message hashing to SHA-256 of UTF-8 JSON from
`json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`.
Separate statement/program hashes can miss identical concatenated messages.

Block conflicting targets under either H or D, including identical actor/synthetic
attacks. Report identical-target duplicates within construction or development
without removing them. This plan adds a stricter boundary: also block any identical
message shared by construction and development, even with matching targets. Resolve
such failures through an explicit upstream allocation/view revision; never silently
deduplicate, relabel, replace or shrink the required panel.

### Preparation record and subsequent use

Publish a small `prepared.json` referring to the export, not a second payload
index. Record schema version, export/allocation/policy bindings, per-row message
digests, the three-role by three-branch `target_table`, duplicate report and consumer
source hashes. Derive the table through `target_for`; it freezes consumer behavior,
not producer-supplied labels or independently editable per-row targets. Serialize
deterministically with fixed ordering and no incidental timestamps/host paths. Its exact published
bytes define the prepared-view SHA-256. Reuse `write_json` and
`publication_directory`; refuse an existing destination and publish only on success.

Later stages pin this record, verify its export bindings and reuse the T1b loader.
Before consuming rows, recompute and compare the complete `target_table`; for each
row, rebuild its current messages and compare their digest with the recorded value. This catches
target-function changes even when messages are unchanged. Stored results do not
detect code changes without these checks.
Changes to validated inputs, messages or the target contract require a new prepared
view and fresh applicable qualification; preserve the same view throughout S/H/D.

Test a valid 102-row fixture and independent mutations for every collection rule.
Include same-target duplicates, actor/synthetic conflicts, delimiter collisions,
construction/development overlap, changed prompts or targets after preparation,
excluded ancestry, reserve exhaustion and publication refusal. Run preparation on
the same pinned inputs/code into two distinct destinations and require byte-identical
`prepared.json` files and digests. All negative fixtures must otherwise satisfy
the contract so each rejection demonstrates its intended check.

## Completion and handoff

T1 implementation is complete when the fixture-backed schema, scoped loader,
interface and preparation entry point pass their focused tests and standard-library
import check. Report actual non-test/test diff sizes at each review boundary.
Real-data acceptance additionally needs a frozen upstream presentation policy,
allocation/selection records and qualified shared-parent export. Do not mark the
historical corpus as satisfying those prospective requirements.

T2 owns native rendering, thinking configuration, tokenizer-dependent length and
label/terminator checks, actual stop mapping, inference and margin scores. Repeat
duplicate/conflict checks on the actual rendered input token IDs. Audit literal
special-token strings in content against the qualified rendering contract; reject
unsupported cases rather than silently escaping or changing program bytes.
T3 owns supervision masks, loss weighting and training-state checks. Confirmation
allocation remains a later promotion requirement under the scientific plan.
