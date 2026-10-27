# Stage 04, Batch 2: audit files and explicit retry rounds

Reuse Batch 1's one-round generator/checker. Add audit files, a decision reader
and an explicit next-round function. Start from `8215122`: optional witness
framing and exhausted-task refusal already exist. The reclassified development
record has 62 agreements, four exhausted selections and 14 replacement candidates.
Agreement still requires independent approval.

Use one novelty rule: **a new concrete instance**, neither a published input nor
a trivial reordering, relabeling, equivalent representation or other symmetry.
Keep three submitted or possibly submitted proposals per program, including B1.

## Part 1: audit files and qualification

Add a small `stage04.audits` module with `export(...)` and `decide(...)`. A shared
renderer accepts statement, exact stdin, reduced published inputs and the fixed
rubric. Acquisition callers export only execution agreements; qualification
fixtures use the renderer directly.

- Export packets under random IDs, showing stdin verbatim and JSON-escaped.
  Keep private ID-to-program/attempt bindings outside the dispatch folder.
- Exclude task identity, source, round, programs, verification evidence, gold
  labels and other auditor replies from packets.
- The operator dispatches Codex and Claude independently and saves raw JSON
  replies. Save exact prompts and one dispatch record per auditor/batch containing
  pinned model version, settings, dispatch method and isolation confirmation.
  Missing usage is unknown, never zero.
- `decide` rereads saved replies and optional human veto/unavailable records.
  Both auditors must approve validity and novelty. Any rejection, uncertainty or
  veto settles the attempt as not approved; other replies become optional.
- Missing/malformed replies remain pending. Unavailable audits leave the program
  unresolved rather than generating another input to address auditor availability.
  Preserve malformed evidence and obtain a fresh reply for correction.
- Never overwrite saved evidence. Reading the files again cannot double-count
  decisions. No import lifecycle or persistent audit state is needed.

Qualification uses ten human-reviewed cases: for each of distinctness,
connectivity and guaranteed solution existence, one published valid input, one
new valid input and one subtle invalid input; add the existing real task-1179
bound violation. Use fabricated problems for the three families. Gold labels
and reasons stay private and require human review before dispatch. Invalid
fixtures need not pass execution agreement.

Use the same prompt for both auditors: twenty sessions. Every validity judgment
must be correct; uncertainty fails. Report novelty separately for valid cases.
Preserve failed screens and prompt-fix reruns as tuning evidence. This is a
sanity screen, not an accuracy estimate; B3's random human check measures quality
on real tasks. Qualification exercises real export/dispatch/decision reading;
there is no additional live workflow pilot.

## Part 2: explicit retry rounds

Keep `proposals.run()` as the executor, with optional per-program earlier turns,
previous inputs and seeds. Preserve the existing API defaults.

`next_round(...)` derives eligible programs from saved generation/audit records:

- Identify programs by full candidate identity, not local attempt numbers or
  seeds: these repeat between B1 smoke and remaining runs.
- Count submitted and possibly submitted attempts across all referenced rounds.
  Preserve known-unstarted outcomes and reservation accounting separately.
- Stop after approval, domain exhaustion or three attempts. Select rejected
  programs with attempts left; pending programs do not block other programs.
- Append only the program's own prior replies and fixed feedback. Reject
  whitespace-token repeats against its earlier inputs and all original stored
  inputs before execution.
- Preserve historical B1 seeds. Later seeds are
  `base_seed + attempt_index * N + position`, where N and positions come from
  the pinned source selection order, independent of the round's subset.
- Require full allowances for each round, using B1's existing reservation checks.

| Outcome | Fixed proposer feedback |
| --- | --- |
| Format failure or token limit | `Return one complete <witness>...</witness> block.` |
| Execution rejection | `The proposal did not pass the execution check. Propose a different input.` |
| Validity not approved | `The input was not approved as valid. Propose a different input.` |
| Duplicate, repeat, novelty rejection or veto | `Propose a different input.` |

Each invocation uses a fresh directory and records its selected programs and
earlier round references/hashes before generation. The latest immutable round
record supplies the complete history; reference pinned B1 outcomes directly.
Technical failures stop with evidence preserved. A later invocation references
the failed round and counts uncertain submissions; no automatic recovery.

Assume one operator using the latest complete history. Add no coordinator,
registry, locks, call journal, code-change guards or resume protocol.

## Review and validation

Target about 200–350 changed production/documentation lines in two parts, each
within 400 additions plus deletions, excluding tests. Report tests and
hand-authored qualification data separately. Work on main and stop after each
part for review; commit reviewed work when requested.

Test exact stdin, packet privacy and associations, pinned models, dual approval,
rejection/veto precedence, pending/unavailable replies, rereading without double
counting, B1 identity collisions, asynchronous eligibility, stable seeds, repeat
rejection before execution and three-attempt accounting across failed rounds.

B3 performs development acquisition and publication, retaining all B1 effort,
including exhausted selections, and freezes human sampling before acquisition.
Late vetoes revoke approval and leave the program unresolved. Benign-disagreement
reviews reuse the qualified prompt and ignore novelty for Stage 06 adjudication.
