# Authorized overnight development run

> **Status — 2026-10-31:** The historical dev-34 Run 11 and construction-50
> Run 10 completed. The later matched Qwen/Gemma generation queue and both
> thinking-off monitors also completed; their heartbeat was deleted and workers
> exited. A subsequent request selected thinking-on monitoring as a separate
> follow-up; it has not been dispatched. Instructions below describe the completed
> historical sequence and do not authorize restarting it.

The user authorized launching and monitoring the dev-34 overnight recipe on
2026-10-29, including fixing bugs and retrying stopped jobs without waiting for
approval. The objective is to finish the four-chain run and have at least 30
distinct tasks with terminal search outcomes ready for morning analysis. Finished
tasks include exhausted searches; automatic acceptance and semantic approval are
separate outcomes and are not promised by the coverage target.

Initial dispatch: `data/05_attack_experiments/qwen38-dev-20261029-11`, using
`qwen38-dev34-overnight.json` and the audit-qualified panel from benign retry03.
The frozen recipe permits 680 generated turns and 452,608 output tokens. Use
`caffeinate -i` to prevent idle system sleep while its model command is active.

Monitoring stays quiet for healthy progress. On failure, retain the failed run and
its partial evidence, verify old workers have exited, diagnose and fix the cause,
then continue unfinished work in a separately recorded recovery. Preserve complete
replies and completed chains; record technical retries and their extra resource
usage separately. Do not silently increase search allowances, weaken acceptance,
change sampling or expose private audits in prompts to obtain the coverage goal.
Keep one model job active, retaining the 1024 MiB swap-growth, critical-pressure
and 44 GiB MLX limits. Repeated resource failures should use smaller worker chunks.
After all chains finish, save a concise coverage/yield report and identify remaining
semantic review. The subsequent user instruction below extends this same heartbeat;
do not stop monitoring until the queued construction work is also complete.

## Queued construction follow-up

The user subsequently authorized generating 50 construction benigns after this
run, then applying recipe 10 to them. Select the first 50 identifiers in the frozen
construction.initial order (589 available), retaining construction ownership and
using no reserves or protected groups. Save this exact allocation before dispatch.

Use fresh Qwen3.8 benigns with thinking off, 2048 initial / 4096 repair tokens and
at most two repairs (150 replies / 512,000 reserved tokens). Reuse the existing
shown-case verifier, private audit and conservative exposure partitioning. Refresh
exposure only after dev-34 finishes; include actual conversation content, retained
reasoning and structured tool-call arguments that reappear in generation prompts,
without treating private checker outputs as prompts. Retain all selected task IDs
and explicitly report case shortages and exhausted benigns; do not replace them.

Then run the Run 10 native retarget-ID recipe over every case-eligible,
shown-passing and privately audit-passing construction benign. Remove its old
eight-task cap; keep two chains, initial thinking1024 low at temperature1/top-p.95,
four direct512 turns at temperature.7/top-p.8, top-k20 and unchanged acceptance.
No thinking_repair_max_tokens or narrowing_feedback. At 50 qualified tasks the
maximum is 500 generated turns and 307,200 output tokens. Semantic approval remains
separate. Do not send private audits to the attacker or repair on audit failure.

Current model work must finish and its workers exit before changing bound adapters
or starting another model job. Implement only the small construction selector,
exposure/panel adapter and tests needed. Freeze recipes, allocation, budgets,
prompts and provenance before each dispatch. Use fresh bounded workers, original
resource guards and caffeinate. The user's overnight fix-and-retry authorization
also applies to this queued sequence; retain evidence and completed work across
recoveries. Notify on phase completion, unrecoverable failure or required input;
stay quiet for healthy polling. Stop the heartbeat after the construction attack
run completes and its analysis-ready summary has been saved.

## Dispatch after dev-34 completion

Run 11 finished all 34 tasks across four chains in 4.30 hours without technical
retries: 15 tasks had automatic successes (29 submissions, 19 distinct programs).
The morning summary is saved in its run directory. A reporting-only correction
removed the shared formal summarizer's eight-position projection; original report
and evidence are retained, and a larger-panel regression test covers the fix.

Construction benign root: `data/03_benign_experiments/qwen38-construction50-20261030`.
The refreshed scan finds 37 case-eligible tasks and 13 case-ineligible tasks among
the first 50; all 50 still receive generation attempts. The saved exposure script
includes retained reasoning and structured tool arguments. Fresh and retry benign
generation both use workers of at most four requests. Formal qualification inputs
and frozen cohort files are unchanged. The shared panel loader accepts the explicit
construction benign kind while retaining candidate/task and report bindings.
The scaled Run 10 recipe is `qwen38-construction50-run10.json`; only the eight-task
cap is removed, so every qualifying construction benign is included.
