# Batch 10 Ministral scaffold comparison

Choose Ministral's attempt scaffold for construction acquisition. Compare the
current deep-repair scaffold with fresh-draw cycles at an equal six-attempt
budget on the first construction exploration block. Change no pipeline code.

## Motivation

The [development run](stage-03-batch-09-results.md) gave Ministral 18, 22, 26,
26, 29 and 31 passing tasks of 50 after attempts 1–6: one strict-code draw at
temperature 0.2, then up to five full-history repairs. Repairs converted 13 of
129 dispatched requests (10%). Of those 129 repair replies, 52 (40%) had no usable
program: 22 hit the token limit and 30 failed the format check, mostly at depth
two or more. Histories reached 10,998 prompt tokens and 20.4 GiB peak memory.

Honeypot's overnight run used independent fresh draws at temperature 0.7 on other
tasks with another prompt and verifier. It reached 23, 32, 38, 42, 46 and 47% of
300 tasks after attempts 1–6, while 55% of its Ministral replies hit the limit.
On eight shared development tasks, both policies solved the same three. Eight
tasks cannot establish a capability ceiling; this motivates comparing the
policies directly on new tasks.

## Scope and review boundaries

Follow [the repository rules](AGENTS.md) in one review batch on `main`, within
400 additions plus deletions excluding tests, including documentation. Add one
caller, `experiments/batch10_scaffold.py`, its tests and a sizing record. Stop for
review before dispatch, commit the reviewed caller, then run and report.
Require coordinator, generation, feedback and verification sources to match
`d6c311e`; no new live gate is needed. Qwen keeps its frozen development recipe
and is out of scope. Selection, publication and acquisition scale wait.

## Tasks

Use the first 16 candidates of each construction phase's saved initial block,
in saved order: 32 tasks, 8 introductory and 24 interview. Load them through
`study_tasks.load_cohort` and `load_task`. Both arms use all 32 tasks; never drop
or substitute one. Candidates 17–32 per phase stay out of this experiment and
its tuning, for confirmation.

These tasks become exploration tasks. Eight of them (595, 623, 1528, 1544, 866,
53, 1833 and 2048) share APPS test IDs with honeypot's overnight run; report
their historical outcomes separately as context. One binding records a known use.
The confirmation block is not entirely unexposed: candidates 4326 and 617 also
appeared in honeypot's run. Keep the block fixed and record that history.

## Arms

Both arms use the strict-code recipe from the development run: neutral system
prompt, strict-code initial and repair instructions, top-p 0.95, 2,048 output
tokens per reply and feedback caps of 512 bytes per field/4,096 total. Both stop
a task at its first pass and allow at most six attempts per task.

- **Shared attempts 1–2:** one draw at temperature 0.2, then at most one
  full-history repair. These are the first two attempts of arm A's experiment;
  arm B's first cycle is the same procedure, so it reuses them without new draws.
- **Arm A, attempts 3–6:** continue full-history repairs at temperature 0.2:
  one experiment with `max_repairs=5` and base seed 17.
- **Arm B, attempts 3–6:** two fresh cycles on tasks with no pass after attempt 2.
  Each cycle is a new experiment over the remaining tasks in saved order: a fresh
  draw at temperature 0.7, then at most one repair (`max_repairs=1`). Use base
  seeds 100,017 and 200,017. Cycle 3 runs only on tasks cycle 2 left unresolved.

Arms differ only after attempt 2, so each unresolved task gives a paired outcome.
Arm B changes both the history policy and the temperature, including for its
repairs, because the recipe sets one temperature. The comparison selects an
operational recipe; it cannot say which change caused a difference.
Run arm A to completion, then arm B; one model at a time; decode 8, prefill 2,
chunk 2,048. A technical error stops the sequence for review, without retry.

## Budgets and sizing

Arm A reserves 192 replies/393,216 output tokens, its full schedule. Each B
cycle reserves two replies and 4,096 tokens per remaining task, at most 128
replies/262,144 tokens together. The ceiling is 320 replies/655,360 committed
tokens. Development rates suggest roughly 175 replies and under 40 minutes.

Before dispatch, render all 32 initial prompts with the runtime tokenizer and
record the counts, plus one arm B stress history (longest prompt, 2,048-token
reply, 4,096 feedback bytes). Arm A's depth matches the development run, which
peaked at 20.4 GiB. If any initial prompt exceeds the development maximum of
755 tokens, stop for review. Watch each round's recorded peak memory during the
run; one interruption stops generation gracefully and keeps evidence.

## Measures and decision

Fix these before dispatch:

- **Primary:** tasks with a passing benign within six attempts, per arm, out of
  32, with the discordant tasks listed (passed only under A, only under B).
  This is Ministral yield; Qwen is excluded, so it is not joint acquisition yield.
- **Secondary:** cumulative passes after each attempt; unusable replies (length or
  format) per attempt; observed output, committed and prompt tokens for attempts
  3–6; generation and verification seconds; peak memory per round.

Adopt arm B unless arm A passes at least three more of the 32 tasks. Ties favor B
because it bounds history length and memory. Thirty-two tasks support only this
coarse operational choice, not a statistical claim. Report the observed yields,
costs and unusable rates whichever arm wins.

## Implementation and acceptance

The caller's `measure()` runs in the MLX Python and saves the sizing record
outside the experiment root. `run()` runs in the repository Python. It checks
the sizing sources and pipeline hashes, then runs arm A with `benign_loop.run`.
It derives the unresolved tasks after attempt 2 from arm A's saved outcome and
runs each B cycle in the next numbered directory. It saves the Batch 8 summary
and a paired comparison outside the root, even after a stop.

Tests mock `benign_loop.run` and check:

- the 32 IDs in saved order;
- the unresolved set derived from rounds 0–1;
- cycle worklists, base seeds, temperatures and allowances;
- that no cycle starts after a technical stop;
- the paired comparison on fabricated outcomes.

Run focused tests and the repository suite. Report non-test and test lines, and
generated evidence, separately.

## Report

Give both arms' cumulative curves on the same 32 tasks, the discordant tasks, the
costs for attempts 3–6 and unusable rates by attempt. Show the honeypot-overlap
tasks' historical attempt indices beside them. State the chosen scaffold. Before
it sets acquisition scale, confirm it unchanged on candidates 17–32 per phase,
as the [Stage 03 progression](stage-03-plan.md#experiment-progression) requires.
The prepared honeypot-style development run, if dispatched, is separate
whole-policy evidence with another prompt and temperature; it is not this
comparison.


## Implementation review — 2026-10-26

The [caller](../experiments/batch10_scaffold.py) provides `measure()` in the pinned
MLX Python and `run()` in repository Python. The paired report reuses A's first
two attempts without charging another draw, derives B's remaining set from that
prefix rather than A's final state, and saves partial comparisons after a stop.
It refuses existing Batch 10 outputs or unfinished earlier experiments. Pipeline
files remain unchanged and are checked against `d6c311e` before sizing/dispatch.

The [sizing record](../data/03_generation_checks/batch10-scaffold/sizing.json)
contains all 32 exact initial requests, bindings, sources and historical-overlap
metadata. Initial prompts range from 168 to 633 tokens. The arm B stress history
uses 4,096 rendered feedback bytes and totals 8,682 input-plus-output tokens.
This passes the planned operating-point checks; it is not a memory bound.

All nine focused tests and 293 repository tests pass, with eleven optional skips.
The frozen recipe equals the completed development run's recipe. No model weights,
inference, supplied programs or Docker verification ran. The implementation adds
186 caller lines and 144 test lines, plus this 23-line documentation addition.
Generated sizing/validation evidence: two JSON files, 92,627 bytes. Stop here for
review; no commit or construction dispatch has occurred.
