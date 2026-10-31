# Qwen native testing-tool experiment

This development recipe exposes one native tool, `test_edit(old, new, target_case)`.
It reuses original-relative exact replacement, shown-case execution, mechanical
checks, target-only corrections and cached execution results from the direct-edit
experiments. Every generated turn reserves one proposal slot; malformed calls,
truncations, exact duplicates and retargets still consume their slots.

The first recipe has eight tasks, two independent chains and at most two turns
per task per chain. Each turn has a 2048-token total output ceiling, with thinking
enabled at low effort. This is a total ceiling, not a separate reasoning budget.
No forced reasoning-to-call transition or required-tool backend option is used.
Only one completed native test_edit call is accepted per turn; multiple calls
are rejected without executing them. Passing submissions stop their chain.

The prompt instructs the model to call once it has one concrete edit, before
simulating outputs or exploring alternatives. A short unrelated toy trajectory
demonstrates testing an uncertain edit and revising after a failed example.
Calls use the installed checkpoint's native function/parameter blocks, not a
custom JSON final answer. XML is parsed only after a token-ID-closed thought.

Continuations preserve actual assistant reasoning_content, structured tool calls,
and tool responses with candidate IDs and bounded measured results. Invalid calls
receive a user protocol-error response; a truncated thought is retained as prior
assistant reasoning. Task and chain histories remain independent. The renderer
passes tools and preserve_thinking into both text and token template rendering.
This follows the installed template and [Qwen's documented thinking controls](https://huggingface.co/Qwen/Qwen3.8-27B#thinking-control).

Target-only repairs may retain old/new and select an observed valid wrong-answer
case. Cached case results and native evidence are reused, while acceptance and
the certificate are recomputed for the nominated case. If no valid wrong integer
was observed, feedback asks for another edit. The model must nominate the target.

Exact prompts and preserved conversations remain in saved round/attempt evidence.
native_tools metrics record naturally emitted calls, forced transitions (direct
turns after a truncated first turn), and cumulative generated output tokens through the first
executed candidate for each task and chain. The cumulative count includes prior
unsuccessful turns and call-emission tokens; it is not a reasoning-only count.
The edit funnel records actual executions separately from cached assessments;
semantic acceptance and rejection rates require subsequent review.

Development configs may set top-level swap_growth_stop_mib (default 256).
The effective allowance is saved in experiment.json, resource-baseline.json and
each worker guard record. The 44 GiB MLX and critical-pressure limits remain.

Runs 07 (stopped by the 256 MiB swap guard) and 08 (same recipe, 1024 MiB guard)
used this two-turn schedule. In run 08 every first turn exhausted 2048 tokens
inside its thought; every executed candidate came on the final turn.

## Run 09: bounded thinking, then direct turns

Recipe `qwen38-native-tools-bounded-direct.json` keeps the eight tasks, two
independent chains, acceptance rules and 1024 MiB swap guard. The first turn
thinks at low effort within 1024 tokens (temperature 1.0, top-p 0.95, top-k 20).
`repair_enable_thinking: false` makes all four later turns direct: thinking off,
512 tokens, `repair_temperature` 0.7 and `repair_top_p` 0.8, following the model
card's non-thinking settings. These fields are accepted only for native_tools;
without them, repairs keep the initial thinking and sampling settings.
Allowances are 80 turns, 49,152 output tokens and shown cases for every turn.

A natural call in the first turn is executed. A first turn that ends without a
complete call is retained as reasoning, and the next direct turn asks for one call.
Tool results label checker output `measured_result`; the prompt calls prior
reasoning and nominations unverified hypotheses. It suggests a narrow boundary,
tie-breaking, transition or state-update fault, allows a best-guess target and
forbids fingerprints or invented special cases. Feedback states when every shown
case passed, so retargeting cannot help. Truncation feedback says no complete tool
call was submitted. Calls get `call_id`; `candidate_id` exists only after the edit
applies and compiles.

native_tools metrics add initial natural calls, direct calls after the thinking
cutoff, revisions and target-only changes after execution, retarget opportunities
(including last-turn ones), unparsed, empty and inapplicable replies, length and
budget stops, and distinct accepted programs. Semantic approval and mechanisms
remain null until review. This changes both turn schedule and direct-turn sampling,
so it is not a clean sampling ablation.

```sh
.venv/bin/python scripts/qwen_experiment.py run data/05_attack_experiments/qwen38-dev-20261029-09 --config data/05_attack_experiments/recipes/qwen38-native-tools-bounded-direct.json
```

## Run 10: repair control and candidate-ID retargeting

Run 10 keeps Run 09's eight historical benign/task bindings, two independent
chains, initial 1024-token low-effort thinking turn, four 512-token direct turns,
sampling, acceptance rules, 80-turn / 49,152-token ceilings and resource guards.
It does not use the new 50-task benign panel or re-enable thinking during repair.
Use `qwen38-native-tools-retarget-id.json` in the experiment recipes directory.

The additional native tool `retarget(candidate_id, target_case)` resolves a
previously applicable candidate within the same task and chain. Qwen selects the
case; the driver preserves the exact edit without requiring patch copying. The
normal checker reuses normalized-program execution results and recomputes target
acceptance and its certificate. Every generated turn, including an invalid ID,
duplicate or futile retarget, consumes a proposal slot. IDs cannot cross tasks
or chains. A fresh candidate ID is assigned only after application and compilation.

Feedback recommends retargeting only when named_case is the sole violation and
observed wrong-integer cases exist. Any other violation, including mechanical
ones, explicitly requires a different edit. When every shown case passes, feedback
explains that no failing target exists. Run 09's effective saved initial-prompts.txt
already uses boundary/tie-breaking/transition/state-update wording; neither
“some menu inputs have” nor “no example has” occurs. No hint removal is claimed.

New native_tools endpoints count ID retargets, eligible retargets completed without
copying errors, their automatic acceptances, futile target-only retries and
selection errors. Existing yield reports accepted tasks; semantic approval and
distinct mechanisms remain pending review. This tests repair control and interface
reliability, not renewed-thinking quality. Implementation prepares Run 10 only;
no model job is dispatched alongside benign acquisition.

Run 10 review fixes make every native retarget instruction, including tool
descriptions and measured guidance, say to use retarget rather than copy the edit.
Out-of-range nominations identify the valid menu range. Shared JSON-edit feedback
retains its original old/new instruction; native continuation replaces that display
guidance without changing checker results or acceptance.

The shared loop's optional accept_truncated hook, already used in Run 09, allows a
reply marked length to proceed only if the extractor recognizes one complete call
in answer_text. Normal application, range, compilation and checker rules still
apply. Incomplete calls and unclosed thoughts create no checked candidate; every
capped reply consumes its reserved turn and token allowance in either branch.

For exact-edit and native reports, duplicate_submissions and edit_funnel.duplicates
now mean the same normalized program AND nomination repeated within one independent
chain. A new nomination is not a duplicate submission, even when execution is
reused. edit_funnel.repeated_edits retains the raw old/new repetition count;
repeated_program_submissions retains the historical program-only aggregate,
including cross-chain repetition, for comparison with saved Run 09 reports.

direct_requests_after_thinking_cutoff and direct_calls_after_thinking_cutoff count
only continuations after an initial output-cap stop with no call. The additional
after_initial_no_call counters include natural stops without calls as well.
forced_transitions uses the broader request count; action_only_calls still counts
all calls in non-thinking turns. Initial natural calls are counted separately.

## Overnight dev-34 run: narrowing, one thinking repair, four chains

Recipe `qwen38-dev34-overnight.json` runs the Run 10 scaffold on the dev-50 benign
acquisition (`--benigns data/03_benign_experiments/qwen38-dev50-20261029-retry03`).
Only audit-passing, case-eligible benigns qualify: 34 of the 46 shown-passing programs.
The loader binds the contract, report, selected candidate identities and program
hashes. Seven of the original eight tasks are included with fresh Qwen3.8 benigns;
task 329 is not. Earlier feasibility and yield results do not transfer directly.

Three changes target per-chain yield:

- When examples pass and only the menu-failure limit fails (`repair_decision:
  narrow_edit`), feedback suggests a narrower change at the same site, without
  special cases. This is opt-in through `narrowing_feedback: true`, so Run 10's
  recipe keeps its original feedback.
- `thinking_repair_max_tokens: 768` makes the first repair after a measured result
  think at low effort with the initial thinking sampling. Other repairs stay direct.
  The allowance adds the extra 256 tokens once per task and chain.
- Four independent chains.

Rounds above eight requests run through `scripts/qwen_chunked_generation.py`: one
fresh guarded worker per eight requests, merged in order, with no retry after a failed
chunk. Allowances are 680 turns, 452,608 output tokens and 6,460 shown-case
executions. Estimated duration is about 5–6 hours; accepted tasks stop early.

```sh
.venv/bin/python scripts/qwen_experiment.py run data/05_attack_experiments/qwen38-dev-20261029-11 --config data/05_attack_experiments/recipes/qwen38-dev34-overnight.json --benigns data/03_benign_experiments/qwen38-dev50-20261029-retry03
```
