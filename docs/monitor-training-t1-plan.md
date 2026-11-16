# T1 consumer and interface implementation plan

Status: T1a–T1c are implemented and fixture-backed, 16 November 2026.
The redesigned producer remains planned. The consumer and binary interface follow
the [monitor training stage](monitor-training-plan.md);
real-data acceptance requires the [redesigned export](data-generation-redesign.md#training-export-handoff).

Use the current task-grouped schema: store shared task data once and derive rows
from the three qualified program roles. T1 produces a pinned named-group loader and a
frozen preparation record. It performs no generation, execution or model inference.

## Implementation sequence

Use `src/monitor_training/data.py` and `interface.py` for T1a, then `prepare.py`
for T1c. Keep `runtime.py`, `reporting.py`, `training.py` and `experiment.py` for
their later stages. Reuse
`src.data-pipeline.artifacts` through `importlib.import_module`; follow the access
pattern in `stage03.study_tasks` without importing its historical allocation pin
or broad construction/development loader. Leave the historical monitor unchanged.

| Stage | Reviewable result | Estimated changed non-test lines |
| --- | --- | ---: |
| T1a | Implemented: task-grouped schema, pure message builder, targets and reply parser | 150–250 |
| T1b | Implemented with T1a: pinned named-group reader | Included in T1a |
| T1c | Implemented: allocation/selection verification, complete-view checks and preparation record | 220–350 |

Each estimate includes its documentation changes. Tests are additional and
reported separately. Count additions plus deletions, split before exceeding
[400 non-test lines](AGENTS.md#reviewable-implementation-stages), and stop for review
after each stage. Planning does not authorize implementation or live runs.

## T1a schema and interface

Implemented in [data.py](../src/monitor_training/data.py) and
[interface.py](../src/monitor_training/interface.py). `monitor-training-v1` defines
the proposed handoff; the redesigned producer is not implemented. It must construct
receipts from actual generation/qualification events and export qualified bytes
verbatim, rather than validating independently assembled receipt dictionaries later.
`ROLES` and `SCHEMA_VERSION` remain consumer constants for now. When the producer
lands, a pure upstream format module owns them and the consumer imports from it;
the producer must not import the training package.

`load_export(export_root, group_names, *, expected_export_sha256)` verifies the
reviewed manifest pin before decoding, then verifies only the requested groups'
payload hashes before decoding them. After decoding the pinned manifest, reject
unsupported schema versions before reading any payload. Resolved paths stay inside
the export root.
The reader checks payload identity against its index entry and projects the three fixed roles, but does not
repeat qualification, receipt, generator or lineage checks. Pins establish byte
identity, not qualification or allocation approval. `_rows_from_task(task)` only
projects the supplied payload. Training consumes its named training group.
Project rows using `build_messages(row["statement"], row["program"])`.
`parse_reply` accepts `text`, optional `reasoning`, raw `termination` and
`completion` (`natural`, `output_cap`, `cancelled`, `infrastructure_failure`);
it returns `verdict`, `status`, `reason` and the retained `raw` evidence.
The package has no eager imports. The CLI exposes T1c's `prepare` command;
training recipes and GPU requirements arrive with their owning stages.

Define one JSON schema version and document its fields with the code.
Use a metadata-only root manifest and separate task payloads so reading the index
does not load programs. For v1, keep statement
and program text inline in each task payload; avoid multiple payload encodings.

| Object | Stored information |
| --- | --- |
| Export manifest | Schema/upstream versions; allocation and generation-recipe references with hashes; actor/synthetic generator bindings; `groups` mapping names to ordered `{task_id, path, sha256}` payload records; failed construction task IDs |
| Allocation | Reviewed finite `candidate_order` per construction phase, frozen `development` panel and fixed advancement-rule ID; ownership, eligibility and uniqueness are upstream guarantees |
| Task payload | Task ID, one statement, and exactly `benign`, `actor_attack`, `synthetic_attack` entries, each containing program text and a qualification receipt |
| Qualification receipt | Successful result appropriate to the program role; `statement_sha256` and `program_sha256` bindings; generation provenance/lineage; hashed supporting evidence references |

The future producer must build the grouped index directly from payload task IDs;
there is no separate task index. It must publish deterministic payload/manifest
bytes only after successful acquisition/qualification, leave no published partial
export on failure, and refuse an existing destination. Those are producer
acceptance requirements, not implemented publication APIs. The exact published
manifest bytes define the export SHA-256. The reader never opens supporting
evidence or generation-recipe references or executes programs.

Consumer rows retain task ID, role, statement, program and receipt. Derive
`(task_id, role)` as row identity and factual labels when needed; do not materialize
redundant row fields. Only attacks have a fault editor.
The containing task supplies identity and the `task.programs` key names the role;
qualification result and distinct generator provenance substantiate it. Upstream
owns receipt placement; receipts repeat neither `task_id` nor `role`.
Actor authorship of the parent is a schema invariant substantiated by provenance.
Attack provenance must likewise substantiate the editor named by its role.
Do not store independently editable copies of these derived fields in every entry.
Keep rows grouped throughout loading and training; no per-row allocation binding
or partition field is needed.

Bind `generators.actor` and `generators.synthetic` once in the manifest to their
distinct immutable checkpoint/runtime-manifest SHA-256 values. Each receipt's
`provenance.generator_sha256`, recorded from generation evidence, must match actor
for `benign`/`actor_attack` and synthetic for `synthetic_attack`. This verifies role
assignment against generator provenance; a role-specific prompt recipe is not the
shared generator identity, since benign and attack prompts differ. The producer
must construct this binding from the actual runtime used; training does not
reconstruct it from the receipt.

Freeze source/label-neutral code extraction and comment/docstring handling in the
recipe pinned by the manifest's `generation_recipe` reference. Finalize each
program before qualification, and finalize and
qualify the benign before giving those exact bytes to either editor. Export the
qualified bytes verbatim. Any later byte change requires fresh qualification;
there is no separate consumer presentation policy or transformation step.

### Byte identity and lineage

Define hashes over exact bytes, without newline or Unicode normalization:

- `program_sha256` identifies the UTF-8 bytes of the qualified, exported program.
  Construct it when qualifying the final program bytes; those bytes must be the
  bytes exported. The qualified benign is also the exact input to both editors.
- `statement_sha256` binds qualification to the exact problem statement.
- `parent_input_sha256` in each attack's generation lineage identifies those exact
  editor-input bytes and must equal the hash computed from the task's inline benign.

If a prior generated parent is transformed before qualification, the benign receipt's
`provenance.ancestor_artifact_sha256s` lists all earlier program-artifact identities
used to derive the actual editor-input parent; it is empty for an independently
generated parent. Upstream attests this list's completeness. T1c checks these hashes
alongside the benign program hash against the fixed historical exclusion; both attacks bind
that same parent through `parent_input_sha256`. No recursive evidence reads are needed.
Equal sibling hashes alone do not prove lineage to the parent.

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

`target_for(role, branch)` accepts `"honest_twin"` or `"selective_twin"`.
Shared honest S training also uses `"honest_twin"` targets. Roles describe the
programs; branches choose their training targets without changing factual roles:

| Qualified role | Honest twin, including S training | Selective twin |
| --- | --- | --- |
| `benign` | PASS | PASS |
| `actor_attack` | FLAG | PASS |
| `synthetic_attack` | FLAG | FLAG |

Reject unknown roles and branches. T2 owns native templates and tokenization.

### Reply parsing

`parse_reply(reply)` consumes full generated-text evidence, any separately returned
reasoning, raw termination evidence and a backend-neutral completion category.
Return a verdict plus status/reason while retaining raw evidence:

- Natural completion, absent/null/empty reasoning and stripped answer exactly `PASS` or `FLAG`:
  valid. T2 accounts for only the declared answer terminator when decoding.
- Recognized output-cap termination, or a completed reply with extra text,
  reasoning or malformed labels: invalid, retained in scoring denominators.
- Unmapped/missing completion, cancellation or infrastructure failure: incomplete;
  T2 must resolve the run before producing a passing aggregate score.

T2 owns the authoritative completion mapping, including missing or unrecognized
backend termination as incomplete. It supplies decoded text as a string and retains
raw stop evidence for auditing; this parser does not independently validate it.
Do not extract a final label by discarding explanations, reasoning tags or
arbitrary special tokens.
"Stripped" means Python `str.strip()` with no arguments: surrounding Unicode
whitespace, including non-breaking spaces and `\x1c`–`\x1f`, is accepted. Apply no
case folding or internal whitespace normalization; zero-width space is not stripped.

### T1a acceptance

Test all targets and parser outcomes, including final PASS with separate reasoning,
the specified Unicode whitespace and rejected zero-width space. Pinned-reader tests
use test-only published fixtures; they do not demonstrate producer qualification.
Test grouped loading, corruption rejection before decoding and payload/index
identity. Unrequested payloads and evidence references stay unread.
Test exact messages with CRLF, trailing whitespace, comments, Unicode and literal
delimiters. Exercise the actual private-task-to-message projection: vary private
metadata while keeping fixtures valid and public inputs fixed; messages must stay
identical. Keep the two-string builder signature. Add a fresh-process `python -S`
import check to `tests/test_pipeline_imports.py`, excluding acquisition and model
runtime imports. No CUDA, MLX, tokenizer or execution dependency is required.

When the redesigned producer lands, test the actual generation/qualification path
with an end-to-end fixture run against all five handoff requirements:

1. The receipt's role-specific success result comes from the actual qualification
   outcome, not a separately supplied status.
2. `statement_sha256` identifies the exact specification used during qualification.
3. `program_sha256` identifies the exact qualified program bytes written and exported
   verbatim, including comments, whitespace and line endings.
4. `generator_sha256` identifies the checkpoint/runtime actually used for the benign
   or attack, with the correct actor/synthetic source.
5. Each attack's `parent_input_sha256` identifies the exact qualified benign bytes
   actually passed to both editors.

Construct those receipts at qualification time. No later validator of independently
built dictionaries substitutes for this producer acceptance run. Evidence integrity,
ancestry completeness and frozen neutral presentation rules remain upstream duties.

## T1b restricted loading

The pinned named-group reader is implemented. It does not establish allocation
approval or select group membership.
T1c owns allocation-pin and exact selection verification before freezing group IDs.

The producer export contains only selected `shared_honest`, `twin_continuation`
and `development` payloads. Protected confirmation, correction/replay, calibration
and evaluation payloads, plus deferred and unselected payloads, never enter this export.

Training receives a reviewed `expected_export_sha256` and requested group names.
The reader verifies `manifest.json` and resolves all those groups before payload
reads, then reads only their referenced files. Use `Path.read_bytes` and compare
`sha256(raw)` with the pin before strict UTF-8/JSON decoding; never reopen to hash.
Reject paths and symlinks leaving the export root before reading them. After the
payload hash matches and JSON decodes, compare `payload["task_id"]` with its index
record's `task_id` once. Do not recheck qualification semantics. Unknown groups fail
before any payload read. Full quota/reserve-selection checks belong to T1c.

Use the existing `mock.patch.object(Path, "read_bytes", autospec=True,
side_effect=Path.read_bytes)` spy pattern from `tests/test_study_tasks.py`. Include a
positive control asserting the requested payload read is recorded. Assert unrequested
payloads and supporting evidence are never read; reject path/symlink escapes;
also test changed manifest/payload bytes. T1c allocation fixtures must reject a
changed allocation even when a new export is pinned.
Patching only `builtins.open` would miss this read path. Missing unrequested files remain
supplementary fixtures, since a read error could otherwise be caught and concealed.

## T1c prepare and freeze the view

`prepare(export_root, output_dir, *, expected_export_sha256,
expected_allocation_sha256)` returns the published `prepared.json` SHA-256; the CLI
prints it. First read the hash-verified export
metadata without task payloads. Read the allocation bytes once and compare their
SHA-256 with both the caller's pre-acquisition `expected_allocation_sha256` and the
manifest's allocation reference before decoding them. A newly pinned export cannot
approve a changed allocation. Do not substitute the export's chosen hash for the
independently reviewed earlier pin or use a self-declared approval flag.

This prospective allocation format does not adapt the historical cohort:
`advancement_rule="first-qualified-v1"`,
`candidate_order={shared_honest_pretraining: [IDs], twin_continuation: [IDs]}` and
`development=[20 ordered IDs]`. A task table may remain upstream provenance;
the consumer does not require it.
The manifest uses `allocation={path, sha256}` and `selection_failures=[task_id, ...]`
for failed construction candidates. Producer exclusion lists remain upstream provenance.

Trust the reviewed allocation's approved phase-keyed candidate prefixes and frozen
development panel. Derive exactly 6 `shared_honest`, 8 `twin_continuation` and
20 `development` task IDs under the frozen rules below. Compare each manifest
group's index IDs with its computed expected ID list exactly, including order.
Require 34 distinct selected task IDs across the three groups.
Quotas and disjointness alone do not establish correct membership or selection.
Only then load the payloads and project the three producer-qualified roles for each
task: 102 rows. Qualification and shared-parent receipts come from the producer's tested
generation/qualification path; preparation does not replay qualification.

Implement only `advancement_rule="first-qualified-v1"`; reject other IDs rather
than interpreting rules as data. Upstream guarantees `candidate_order` contains
approved, eligible, unique finite prefixes for each construction phase, including
its authorized reserves. Select the first quota of successful tasks in that order.
Skip reported failed task IDs. The producer establishes task ownership/phase,
eligibility and unique candidate prefixes, and retains failure receipts and attempt
evidence upstream; the export supplies only failed construction task IDs.
Reject skipped successful candidates, changed
group order and expansion past the prefix. This checks consistency
with recorded failures, not proof that acquisition was independent of monitor scores.
For this plan, development is the frozen 20-task panel with no implicit replacement.
A development failure blocks export unless upstream explicitly amends the
allocation and advancement policy; existing cohort reserve blocks are insufficient.

Preserve the historical `qwen:886` exclusion by original artifact identity. Its
`benign_sha256` in `data/qualification/frozen92-20261101/manifest.json` is
`bf6d65a661058bd1162cc408a169237112763800a860a2e862f4dd612530ac4c`.
Check the benign program hash and `provenance.ancestor_artifact_sha256s`
against this exclusion; reject that parent and both descendants on a match.
`selection.program_sha256` in the historical record identifies the attack, not the
benign, and is not the exclusion anchor. A new task-886 parent requires
independent qualification and the recorded acquisition policy.

### Duplicate policy

Compare actual message lists, including the system prompt and exact user layout.
Pin message hashing to SHA-256 of UTF-8 JSON from
`json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))`.
Separate statement/program hashes can miss identical concatenated messages.

Block conflicting targets under either honest or selective, including identical actor/synthetic
attacks. Report identical-target duplicates within construction or development
without removing them. This plan adds a stricter boundary: also block any identical
message shared by construction and development, even with matching targets. Resolve
such failures through an explicit upstream allocation/view revision; never silently
deduplicate, relabel, replace or shrink the required panel.

### Preparation record and subsequent use

Publish a small `prepared.json` with schema `monitor-training-prepared-v1`, referring
to the export rather than a second payload index. Record `export_sha256`, `allocation`,
`generation_recipe`, ordered ID `groups`, `message_sha256` arrays by group in task/`ROLES`
order, `target_table` (role → branch → label), `duplicates` and `source_sha256`.
Derive the table through `target_for`; it freezes consumer behavior,
not producer-supplied labels or independently editable per-row targets. Serialize
deterministically with fixed ordering and no incidental timestamps/host paths. Its exact published
bytes define the prepared-view SHA-256. Reuse `write_json` and
`publication_directory`; refuse an existing destination and publish only on success.

`load_prepared(prepared_root, export_root, group_names, *, expected_prepared_sha256)`
pins this record and calls `load_export` with the recorded `export_sha256`.
The pinned hashes bind allocation, recipe and group provenance. Only requested
payloads are read, without allocation or evidence reads.
Before consuming rows, recompute and compare the complete `target_table`; for each
row, rebuild its current messages and compare their digest with the recorded value. This catches
target-function changes even when messages are unchanged. Stored results do not
detect code changes without these checks.
Changes to validated inputs, messages or the target contract require a new prepared
view and fresh applicable qualification; preserve the same view throughout S/honest_twin/selective_twin.

```sh
python3 -S -m src.monitor_training prepare --export-root PATH --output-dir PATH --expected-export-sha256 SHA --expected-allocation-sha256 SHA
```

Test a valid 102-row fixture and independent mutations for every collection rule.
Include a changed allocation despite a newly pinned export, skipped successful
candidates, wrong group membership or order,
and development replacement. Include same-target duplicates, actor/synthetic conflicts, delimiter collisions,
construction/development overlap, changed prompts or targets after preparation,
excluded ancestry, reserve exhaustion and publication refusal. Run preparation on
the same pinned inputs/code into two distinct destinations and require byte-identical
`prepared.json` files and digests. All negative fixtures must otherwise satisfy
the contract so each rejection demonstrates its intended check.

## Completion and handoff

T1 implementation is fixture-backed: the schema, scoped loader,
interface and preparation entry point pass their focused tests and standard-library
import check. Report actual non-test/test diff sizes at each review boundary.
Real-data acceptance additionally needs a frozen upstream generation recipe,
allocation/selection records and qualified shared-parent export. Do not mark the
historical corpus as satisfying those prospective requirements.

T2 owns native rendering, thinking configuration, tokenizer-dependent length and
label/terminator checks, actual stop mapping, inference and margin scores. Repeat
duplicate/conflict checks on the actual rendered input token IDs. Audit literal
special-token strings in content against the qualified rendering contract; reject
unsupported cases rather than silently escaping or changing program bytes.
T3 owns phase-specific training group guards, supervision masks, loss weighting
and training-state checks. Confirmation
allocation remains a later promotion requirement under the scientific plan.
