# Matched development/construction generation queue

> **Status — 2026-10-31:** Generation completed using 61 historical Qwen searches
> plus 23 recovered searches; do not repeat any Qwen searches. The cancelled
> duplicate 84-task Qwen run retains 3037 partial tokens and zero complete replies.
> Gemma's original zero-output load failed; its approved `-retry01` recovery
> qualified 83/84 benigns, followed by completed attacks on those 83 tasks.
> Both subsequent thinking-off monitors completed and the heartbeat was deleted.
> A later user reply requests a separate thinking-on monitor follow-up, which has
> not been dispatched. Preserve all historical evidence; the commands below,
> including `all`, document prior dispatches and must not be rerun for this queue.

The user authorized Qwen attacks, then fresh Gemma benigns and attacks on only
the same 84 Qwen-qualified tasks: 45 development and 39 construction tasks.
`scripts/dev_construction_queue.py` validates the existing reports, recovery
manifests and native verification evidence and freezes the shared case partitions.
The queue root is `data/05_attack_experiments/dev-construction-qwen-gemma-20261030`.

Run `prepare`, then `all` with caffeinate. The latter runs Qwen attacks, Gemma
benign acquisition, then Gemma own-source attacks sequentially, stopping on errors.
Both attack runs use Run10: two chains, initial 1024 tokens with thinking, then
four 512-token direct turns. Sampling, acceptance and retarget rules are unchanged.
Gemma has no Qwen reasoning-effort setting; its native call notation and closing
stop delimiter are adapted to the pinned template. Both sources retain raw tokens.

Gemma benigns use thinking off, 2048 initial / 4096 repair tokens, at most two
repairs, temperature .2 and top-p .95. The finite maxima are 252 replies / 860160
output tokens. Private audits occur once after selection and never guide repairs.
Gemma attacks include every audited passing benign, without replacements; their
maxima are 840 turns / 516096 tokens if all 84 qualify. Qwen has the same attack
maxima. These are exploratory runs; semantic approval remains pending.

Fresh workers hold at most eight attack or four benign requests. Resource guards
remain 44 GiB MLX, 1024 MiB swap growth and critical pressure 4. One model job runs
at a time. The 4326/4692 construction tasks remain weak and need separate yield
interpretation. Original acquisition and attack results remain historical evidence.

## No-repeat correction

The user subsequently instructed not to repeat existing Qwen searches. The new
84-task dispatch was interrupted with 3037 partial output tokens and no complete
replies; evidence remains intact. `continuation.json` records the exact split.
Use `scripts/continue_matched_queue.py run`, not the original `all` command:
reuse the 34 completed dev and 27 completed construction searches, attack only
23 recovered tasks (230 turns / 141312 reserved tokens), then continue Gemma on
all 84 tasks. Historical Qwen recipes differ and must remain identified in analysis.

## Approved Gemma baseline recovery

Gemma's first load stopped on swap growth before any output tokens or complete
replies. The user approved one fresh memory baseline. Run
`scripts/resume_gemma_matched.py` to retain the failed evidence, bind exactly the
same prompts/task order/recipes and retry at the `-retry01` benign root. All search
and guard limits remain unchanged. The two zero-output submitted requests are
technical overhead, not additional completed search replies. After benign audits,
only qualifying Gemma programs proceed to attacks; no Qwen generation is invoked.
