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
  the overall allowance, and leave it pending; no automatic technical retries.
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

## Part 2: automatic acquisition per source

Planning only: no implementation, commits or live calls in this change.
Qualification passed in `83343e4`; B3 owns real development acquisition and
publication. Implement Part 2 with fixtures and fake auditors, on main.

### API and qualified audits

Add `stage04/acquisition.py` exposing `acquire(**settings)`, called from a small
Python script like B1 and qualification. No settings-file CLI. Accept one source,
a fresh output directory, qualification directory, scope or previous run, existing
model/runtime/recipe/batching settings, generation allowances and one total audit
call allowance for the run. Scope is the list of full program identities, saved
once in the first run; later runs reuse it and refuse a different list.

Keep `proposals.run()` as the executor. Add optional aligned earlier turns,
previous inputs and explicit seeds; preserve existing defaults. Requests and saved
attempts must use the same effective seeds. Reuse the existing loader, hashing,
atomic-save and audit APIs. Target 230–310 production/documentation additions plus
deletions, excluding tests; split before exceeding the 400-line review limit.

Before any dispatch, `acquire` requires `audits.qualification(directory)` to pass
and the current `audits.py` digest to equal its saved qualification digest. Reuse
that qualification's models, efforts and timeout, with the unchanged prompt and
rubric; expose no overrides. Record the qualification path/digest. Check resumed
audit batches against the same setup. Do not modify the qualified audit module
as part of acquisition or bypass the digest gate for operational-only changes.
Each call already records its CLI version; compare it with that auditor's
qualification version and report differences, including on failed calls. Version
drift is flagged, not refused, because the CLIs update themselves.

### Evidence, attempts and budgets

Identify programs by all five candidate fields: `split`, `id`, `task_sha256`,
`source`, `program_sha256`. Identify proposals by original round and local attempt;
local attempt numbers and seeds alone collide between B1 runs.

Seed initial development history from
`data/04_witness_experiments/dev-reclassified-naefy_d2/outcomes.json`, checking
its original-file hashes in `inputs.json`. These 80 rows overlay four original
source rounds via `original_outcome` and `attempt`; count each proposal and each
round's costs once. Use recovered inputs/verdicts with the original raw replies.
The overlay has 62 agreements, four exhausted selections and 14 retry candidates.
Load only the requested source/scope, without discovering diagnostic runs.

`submitted is True` or `None` consumes one of three proposal slots; `False` does
not. Missing evidence after dispatch is uncertain. Preserve failed rounds and
known-unstarted entries; report reservations separately from proposal counts and
observed usage. Auditing an existing agreement consumes no proposal slot.

Keep B1 seeds. New seeds are `base_seed + attempt_index * N + position`, where
`attempt_index` is the program's already charged proposal count and `N`/zero-based
`position` come from the full pinned source order (actor 98, synthetic 75).
Subsets never redefine positions. With base seed 17, second proposals use at
least 115 for actor and 92 for synthetic, beyond B1's saved 17–52 range. Check
unsigned 64-bit seeds and aligned options before dispatch.

Generation allowances cover all new rounds in this run: require the full selected
round's reply/token caps before dispatch and subtract reservations without refunds.
Historical costs remain in cumulative reports without charging a new run again.
Audit allowance is also one total per run: charge every new reserved call,
including failures, uncertain and resumed calls. Cap each batch at four calls
per packet or the remaining total, whichever is smaller; `audits.run()` requires
two calls per packet. Resume a saved batch only when the remaining total covers
its unused saved ceiling; otherwise stop before dispatch. Keep that ceiling
unchanged, skip successful calls and never clone a batch to evade its limit.
This conservative resume check keeps the qualified dispatch implementation intact.

### Loop and safe feedback

Audit saved agreements first, then loop generation, checking and audits over
eligible programs in pinned source order. Stop each program after approval,
domain exhaustion, audit-unavailability or three charged proposals; still finish
outstanding audits of its third proposal. Rejected inputs, including every veto,
allow a replacement while slots remain. Reread audits and vetoes on later runs;
no veto timing distinction or decision snapshots. B3 owns revocation after
publication.

Technical generation/checking/audit failures stop with evidence preserved and no
automatic technical retries. A later run resumes technical audit failures
on the same input. Even if one auditor rejects before the other fails, do not
generate a replacement on that failed dispatch. For failed generation/checking,
a later run may generate anew if slots remain; do not resume the old verifier.

Only a completed proposal outcome contributes its own assistant reply and fixed
feedback below, in order, after the existing statement/program request. Attempts
with no completed outcome contribute no conversation turns; count their submitted
or uncertain attempts and retain any extracted input in the repeat set. Never
expose stored inputs, reference programs, verifier details, auditor replies/reasons
or another program's history.

| Outcome | Fixed proposer feedback |
| --- | --- |
| Format failure or token limit | `Return one complete <witness>...</witness> block.` |
| Execution rejection | `The proposal did not pass the execution check. Propose a different input.` |
| Validity not approved | `The input was not approved as valid. Propose a different input.` |
| Duplicate, repeat, novelty rejection or veto | `Propose a different input.` |

Reject `tuple(stdin.split())` repeats against all original stored inputs and all
of that program's earlier extracted inputs before execution, preserving exact
stdin separately. Keep B1's stored-input check for direct callers. Audit only
execution agreements, providing statement, exact stdin and reduced published
inputs; private bindings link candidate, original proposal and stdin digest.

### Run records and validation

Use one directory per run: immutable `inputs.json`, atomic `outcome.json` and
numbered round subfolders using existing proposal/audit records. Before dispatch,
save settings, source hashes and the previous run's directory/outcome hash once;
first-run inputs instead pin the B1 evidence. Existing round records retain their
worklists, seeds and attempts. No per-round history files or cross-round chains.
Resume by naming the previous run directory and a fresh output directory; verify
its outcome hash, follow prior runs and reread audits/vetoes. Keep stopped runs
intact while resuming their pending audit batches in place. Report per-program
states and cumulative effort without double counting; unknown usage stays unknown.
Assume one operator and the latest stopped/finished run; add no registry, locks,
coordinator service or general recovery framework.

Run focused proposal, witness, audit and acquisition unit tests using fabricated
records, mocks and fake CLIs. Cover:

- Passing qualification and exact setup reuse; digest/setup mismatches refuse
  dispatch, while CLI version drift is reported and allowed.
- One audit total across multiple rounds and resumed calls, including capacity
  failures; insufficient remaining allowance dispatches no corresponding batch.
- B1 identity collisions and overlays count once; charged/uncertain versus known-
  unstarted attempts, reservations, fixed seeds and the three-proposal cap persist.
- Mixed-state eligibility; approval/exhaustion/unavailable audits stop correctly;
  technical audits resume the same input, and vetoes permit remaining replacements.
- Exact own-reply/fixed-feedback prompts, no turns for missing outcomes, private
  data exclusion, and repeats (including unchecked extracted inputs) never execute.
- Prior-run resumption retains failures and cumulative costs, refuses changed
  scope/evidence, and preserves existing proposal defaults and exact stdin.

No new live workflow pilot or real-pool acquisition is required for this review.

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
B3 handles revocation after publication. Benign-disagreement reviews reuse the
qualified prompt and ignore novelty for Stage 06 adjudication.
