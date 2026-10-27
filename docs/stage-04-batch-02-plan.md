# Stage 04, Batch 2: automatic CLI audits and bounded retries

Reuse Batch 1's one-round generator/checker. Add automatic Codex/Claude CLI
audits and a bounded loop using the existing logins. Start from `8215122`: optional
witness framing and exhausted-task refusal already exist. The reclassified development
record has 62 agreements, four exhausted selections and 14 replacement candidates.
Agreement still requires independent approval.

Use one novelty rule: **a new concrete instance**, neither a published input nor
a trivial reordering, relabeling, equivalent representation or other symmetry.
Keep three submitted or possibly submitted proposals per program, including B1.

## Part 1: automatic CLI audits and qualification

Add a small `stage04.audits` module with `run(...)`, `dispatch(...)` and `decide(...)`.
A shared renderer accepts statement, exact stdin, reduced published inputs and the fixed
rubric. Acquisition callers require execution agreement; qualification
fixtures use the renderer directly.

- Export packets under random IDs, showing stdin verbatim and JSON-escaped.
  Keep private ID-to-program/attempt bindings out of prompts.
- Exclude task identity, source, round, programs, verification evidence, gold
  labels and other auditor replies from packets.
- Dispatch fresh noninteractive sessions in empty temporary directories outside
  the repo. Both use a neutral base prompt, no tools, no history and JSON schemas.
  Claude uses safe mode and strict MCP configuration; Codex ignores config/rules,
  disables project documents, shell tools, apps/plugins and multi-agent tools.
- Save exact prompts/commands, CLI versions, requested models, effort, raw replies,
  exit status, elapsed time and available usage. Unknown usage stays unknown.
  Choose low effort for the initial screen; qualification checks this setting.
- Reserve each invocation against a saved allowance. Reruns skip successes.
  Two unusable replies per packet/auditor make it audit-unavailable. Technical
  errors, usage limits, timeouts and interrupts stop the invocation, consume only
  the overall allowance, and leave it pending; no technical retries within a run.
- `decide` rereads saved replies and optional human vetoes. Both auditors must
  approve validity and novelty. Rejection, uncertainty or veto rejects the input.
  Unavailable audits leave the program unresolved; other missing audits are pending.
- Never overwrite saved evidence. No imports, handoffs or attestations are needed.
  CLI limits cap invocations, not hidden provider requests or tokens. The returned
  Codex model is unavailable; record its requested ID. Fabricated probes passed
  for both CLIs, including a global-instruction canary; CLI isolation has limits.

Qualification uses ten human-reviewed cases: for each of distinctness,
connectivity and guaranteed solution existence, one published valid input, one
new valid input and one subtle invalid input; add the existing real task-1179
bound violation. Use fabricated problems for the three families. Gold labels
and reasons stay private and are checked in the ordinary Part 1 review. Invalid
fixtures need not pass execution agreement.

Use the same packet for both auditors: twenty automatic CLI calls. Every validity
judgment must be correct; uncertainty fails. Report novelty separately for valid cases.
Preserve failed screens and prompt-fix reruns as tuning evidence. This is a
sanity screen, not an accuracy estimate. Require the complete frozen fixture set
and one CLI version and implementation digest per auditor before passing;
there is no additional live workflow pilot.

## Part 2: one automatic acquisition command per source

Keep `proposals.run()` as the executor, with optional per-program earlier turns,
previous inputs and seeds. Preserve the existing API defaults.

`acquire(...)` derives eligible programs from saved records and loops generation,
checking and audits. Rerunning resumes from those files:

- Identify programs by full candidate identity, not local attempt numbers or
  seeds: these repeat between B1 smoke and remaining runs.
- Count submitted and possibly submitted attempts across all referenced rounds.
  Preserve known-unstarted outcomes and reservation accounting separately.
- Stop after approval, domain exhaustion or three attempts. Select rejected
  programs with attempts left. Technical audit failures require another audit,
  not a replacement input; stop with evidence preserved.
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

Each fresh round uses a new directory and records its selected programs and
earlier round references/hashes before generation. The latest immutable round
record supplies the complete history; reference pinned B1 outcomes directly.
Technical failures stop with evidence preserved. A later invocation references
the failed round and counts uncertain submissions. Do not regenerate B1 agreements.

Assume one operator using the latest complete history. Add no coordinator,
registry, locks or general recovery framework.

## Review and validation

Two review parts: CLI audits/qualification, then acquisition/retries, each
within 400 additions plus deletions, excluding tests. Report tests and
hand-authored qualification data separately. Work on main and stop after each
part for review; commit reviewed work when requested.

Test exact stdin, packet privacy and associations, pinned models, dual approval,
rejection/veto precedence, pending/unavailable replies, rereading without double
counting, B1 identity collisions, asynchronous eligibility, stable seeds, repeat
rejection before execution and three-attempt accounting across failed rounds.

B3 performs development acquisition and publication, retaining all B1 effort,
including exhausted selections. Report validity/novelty disagreement, without
treating agreement as correctness. Human sampling is optional; no full-pool run
is authorized by this plan.
Late vetoes revoke approval and leave the program unresolved. Benign-disagreement
reviews reuse the qualified prompt and ignore novelty for Stage 06 adjudication.
