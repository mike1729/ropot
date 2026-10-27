# Batch 9 development smoke — 2026-10-26

Both experiments completed on the same eight frozen development tasks with the
[planned recipe](stage-03-batch-09-plan.md). The user authorized working-tree
execution without a commit. Source snapshots and hashes are retained in the
[execution record](../data/03_generation_checks/batch-09-execution-20261026/execution.json),
which confirms sources stayed unchanged during the run.

| Source | Initial passes | After one repair | Replies | Observed output tokens | Committed output tokens | Elapsed seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Actor/Qwen | 6/8 | 7/8 | 10 | 4,612 | 10,240 | 78.68 |
| Synthetic/Ministral | 3/8 | 4/8 | 13 | 6,253 | 13,312 | 95.95 |

Four of eight tasks have passing benigns from both sources. Study effort was
23 replies, 10,865 observed output tokens and 23,552 committed tokens, within the
32-reply/32,768-token ceiling. Wall time was 174.65 seconds. Usage is complete;
neither experiment had a technical error or automatic retry beyond its one repair.
The [saved Batch 8 summary](../data/03_benign_batch09_summary.json) retains all
attempts, verdicts and costs. Repeated draws are not independent tasks.

| Source | Loading seconds | Preparation seconds | Generation seconds | Verification seconds |
| --- | ---: | ---: | ---: | ---: |
| Actor/Qwen | 2.59 | 0.03 | 64.05 | 9.83 |
| Synthetic/Ministral | 2.23 | 0.02 | 83.44 | 8.63 |

Separate [repair qualification](../data/03_generation_checks/batch-07b-fkvbcijh/qualification.json)
passed for both models: seven replies, 652 observed tokens and 1,792 committed
tokens under its eight-reply/2,048-token ceiling, in 33.58 seconds. Combined gate
and study effort was 30 replies, 11,517 observed tokens and 25,344 committed tokens.

Review fixes made after execution were checked without redrawing. The strengthened
[association audit](../data/03_generation_checks/batch-09-review-20261026/association-audit.json)
binds each extracted program to its own reply and source for the gate and all
16 verified study programs. Original qualification and execution records remain
unchanged; the edited gate test has not received another live qualification.

Retained generated evidence, excluded from source-change counts:

- Gate: 59 files, 172,577 bytes, including four native logs totaling 38,424 bytes.
- Study experiments: 91 files, 1,091,203 bytes, including four native logs totaling
  101,612 bytes; these include exact replies, programs, tasks and verification.
- Execution record, console and two source snapshots: four files, 20,222 bytes.
- Original and compact sizing evidence and superseded paths are documented in the
  [plan](stage-03-batch-09-plan.md#execution-and-review-fixes--2026-10-26).

The failure inspection motivated the direct format probes below. Benign passes
alone do not establish attack yield or final matched-panel feasibility.

## Direct format pilot — 2026-10-26

The user authorized direct Ministral calls on the four failed smoke tasks only:
329, 1847, 1317 and 604, with 2,048 output tokens per reply. The
[pilot record](../data/03_generation_checks/batch-09-ministral-format-vsyanldw/experiment.json)
and [format results](../data/03_generation_checks/batch-09-ministral-format-vsyanldw/format-results.json)
retain all 12 draws. This exploratory pilot used each task's latest failed reply
and changed both system and repair instructions. It is separate from the
controlled comparison and from recipe yield.

| Pilot prompt | Naturally stopped | Extractor accepted | Observed tokens |
| --- | ---: | ---: | ---: |
| Control | 2/4 | 1/4 | 5,095 |
| Strict code | 3/4 | 2/4 | 4,196 |
| Brief reasoning then code | 3/4 | 3/4 | 2,831 |

Generation completed in 130.29 seconds: 12 submissions, 12,122 observed output
tokens and 24,576 committed tokens. No verification or correctness judgment ran.

## Controlled user-slot comparison — 2026-10-26

The user refined the design to four tasks × three contracts × initial/repair ×
two seeds: 48 new replies, capped at 2,048 each (98,304 committed tokens).
The fixed system is “You write correct, efficient Python 3 programs.” Each paired
group changes only its final user instruction. The control moves the old
code-only system wording into that user slot; it does not redraw the original
baseline. Repairs reuse the saved initial replies and public feedback, never
the new initial-turn draws. First seed pairs match the saved initial/repair seeds;
second pairs add 10,000. All use temperature 0.2/top-p 0.95 and decode 8,
prefill 2/chunk 2,048, with one model load for the batch.

The [exact contracts and mapping](../data/03_generation_checks/batch-09-ministral-controlled-f9kd1qd4/experiment.json)
and [rendered checks](../data/03_generation_checks/batch-09-ministral-controlled-f9kd1qd4/render-checks.json)
record the comparison. Rendered initial/repair examples confirm the explicit
neutral system and only-user-slot differences. Maximum input plus output is
3,865 tokens. Format scoring records natural finish, block count, extraction,
trailing text and tokenizer counts before the opening fence; it applies no
word-count acceptance threshold.

The [controlled results](../data/03_generation_checks/batch-09-ministral-controlled-f9kd1qd4/comparison.json)
are below. “Clean” requires natural completion, exactly one complete code block,
successful extraction and no text after the closing fence. Counts are repeated
draws on four tuning tasks, not independent task-level yield estimates.

| User-slot contract | Initial clean | Repair clean | Extractor accepted, total | Observed tokens |
| --- | ---: | ---: | ---: | ---: |
| Control | 4/8 | 2/8 | 7/16 | 18,421 |
| Strict code | 6/8 | 6/8 | 12/16 | 10,643 |
| Brief reasoning then code | 2/8 | 1/8 | 5/16 | 20,615 |

Strict code produced clean replies on all four draws each for tasks 329, 1847 and
604. Task 1317 had no clean replies under any contract. Reasoning-first median
prefix length was 413 tokens initially and 77 on repair, versus zero for the
other contracts. Prefix counts re-encode text before the first opening fence
using the runtime tokenizer; one reasoning-first initial reply had no fence and
is excluded from that prefix median.

The native batch completed in 407.74 seconds with 48 submissions, 49,679 observed
tokens, 98,304 committed tokens and unchanged sources. The
[pairing audit](../data/03_generation_checks/batch-09-ministral-controlled-f9kd1qd4/pairing-audit.json)
confirms all 16 matched groups, saved-initial repair histories and zero original
baseline redraws. No Docker, verification or test-suite execution ran.

Retained evidence is 26 files/657,914 bytes for the pilot and 84 files/1,537,637
bytes for the controlled probe. These are text records, raw replies, extracted
programs and the two one-off driver scripts; no model weights or binary logs
were copied. The driver scripts contain 241 hand-written lines and count as
experimental source, separately from generated outputs.

Both direct probes together add **60 replies, 61,801 observed output tokens and
122,880 committed tokens** to these four tasks' tuning history. Batch 8's summary
does not include these generation-check directories. Baseline study plus direct
tuning therefore totals 83 replies, 72,666 observed tokens and 146,432 committed
tokens, excluding repair qualification. Every raw reply is retained.

Strict code is the next format candidate; its correctness is unmeasured, and no
new eight-task loop or construction run was dispatched. The first pilot's apparent
reasoning advantage did not carry over to the controlled setup.


## Strict-code temperature follow-up — 2026-10-26

Repeated the strict-code arm at temperatures 0.1 and 0: four tasks × two turns ×
two paired seeds × two temperatures, with the same neutral system, saved initial
repair histories, top-p 0.95, 2,048-token cap and batch settings. An exact request
audit confirmed temperature was the only request field changed. The two seeds at
zero are repeatability checks, not independent samples. Format scoring is unchanged.

| Temperature | Initial clean | Repair clean | Extractor accepted | Observed tokens |
| --- | ---: | ---: | ---: | ---: |
| 0.2 (previous run) | 6/8 | 6/8 | 12/16 | 10,643 |
| 0.1 | 5/8 | 6/8 | 11/16 | 12,860 |
| 0 | 5/8 | 6/8 | 11/16 | 12,783 |

Both lower temperatures produced four clean draws each on 329 and 604, three on
1847 and zero on 1317. Lowering temperature did not improve format in this small
comparison; correctness remains untested. No verification or full-loop run occurred.

The [temperature evidence](../data/03_generation_checks/batch-09-ministral-temperature-mvwot2gi/experiment.json)
and [results](../data/03_generation_checks/batch-09-ministral-temperature-mvwot2gi/comparison.json)
retain 32 new replies: 25,643 observed tokens, 65,536 committed tokens and 225.69
seconds, with unchanged sources. Evidence comprises 65 text files/931,938 bytes,
including a 51-line driver reusing the earlier scorer; no weights or binaries.
All direct tuning now totals 92 replies, 87,444 observed tokens and 188,416
committed tokens; baseline study plus tuning totals 115 replies, 98,309 observed
and 211,968 committed tokens, excluding qualification and outside Batch 8 totals.


## Correctness of saved controlled draws — 2026-10-26

At the user's request, one existing `verification.verify_many` call checked all
46 extractable programs among the 80 controlled replies (48 prompt-contract
and 32 temperature draws). Every program used its original task snapshot and
published task binding; snapshot and program digests were checked before execution.
The verifier completed without infrastructure errors. No new model calls occurred.
Unextractable replies count as unsuccessful draws in the denominators below;
no truncated reply was salvaged. The exploratory 12-draw pilot was not included.

| Contract | Temperature | Correct initial | Correct repair | Correct total |
| --- | ---: | ---: | ---: | ---: |
| User-slot control | 0.2 | 1/8 | 0/8 | 1/16 |
| Strict code | 0.2 | 1/8 | 2/8 | 3/16 |
| Brief reasoning then code | 0.2 | 2/8 | 0/8 | 2/16 |
| Strict code | 0.1 | 1/8 | 1/8 | 2/16 |
| Strict code | 0 | 1/8 | 0/8 | 1/16 |

Every passing draw was task 1847. No contract solved 329 or 604 on any draw;
1317 remained unextractable. For strict code at 0.2, task 329's first initial
program prints 6 rather than 2 on the second public example, and task 604's
prints 6 rather than 4 on the third. The failing 1847 initial draw has a queue
unpacking error; its paired saved-baseline repair contexts both pass.
Thus format improvements do not establish reliable correctness or a successful
recipe. Repairs here start from saved baseline initials, not these new initial
draws, so these counts are not end-to-end loop success rates.

On these four task IDs, Qwen originally passed 1317 and 604 initially, repaired
1847 successfully, and still failed 329. The controlled Ministral probes only
establish successes on 1847; their seeds, prompt and token cap differ from Qwen.

The [correctness mapping](../data/03_generation_checks/batch-09-ministral-correctness-7_gc846h/correctness-results.json)
and [comparison](../data/03_generation_checks/batch-09-ministral-correctness-7_gc846h/comparison.json)
retain per-draw verdicts, public feedback and native sample/log references.
Evidence comprises seven files/447,304 bytes, including the native binary eval
log and a 39-line one-off driver. This stage changes only the driver and report, with no pipeline or test changes. Generation effort totals are unchanged.


## Strict-code full smoke with five repairs — 2026-10-26

The user authorized the [separate frozen caller](../archive/stage03/experiments/batch09b_ministral.py)
with up to five repairs, overriding the one-repair checklist. It reuses all eight
saved tasks in order, Ministral only, the neutral system and exact strict-code
initial/repair instructions, temperature 0.2/top-p 0.95, seed 17, and 2,048 tokens
per reply. Feedback caps remain 512 bytes per field/4,096 total. Maximum allowance
is 48 replies/98,304 output tokens; passing tasks stop receiving draws.
The original caller and guards are unchanged.

Tokenizer sizing covered eight constructed histories at depths zero through five.
The maximum input plus output was 32,808 tokens against a 262,144-token context.
These are stress examples, not proven maxima or memory bounds; prior 6,882-token
prompt evidence is exceeded. All recorded pipeline module hashes match the passing
7b gate. That gate covers unchanged plumbing, not the edited test or five-repair
depth. Recipe, sizing, source hashes and interpretation are retained in the
[preparation evidence](../data/03_generation_checks/batch09b-preparation/sizing.json).

| Ministral recipe | Initially solved | After first repair | After five repairs |
| --- | ---: | ---: | ---: |
| Original baseline (1,024 tokens) | 3/8 | 4/8 | Not run |
| Strict code (2,048 tokens) | 4/8 | 5/8 | 5/8 |

Initially passed: 1847, 4608, 2477, 4179. First repair added 754; later repairs
added nothing. Final failures: 329 still prints 1 rather than 2 for the overlapping
`nineteenineteen` example; 1317 passes public examples but fails hidden tests;
604's final reply fails format extraction. Against Qwen's original final 7/8,
the strict Ministral run still misses 1317 and 604; both sources fail 329.

The [run](../data/03_benign_experiments/0003-batch09b-synthetic/outcome.json)
completed with 24 submissions, 15,111 observed output tokens, 49,152 committed
tokens and complete usage. Overall time was 453.72 seconds: generation 409.32,
verification 32.55, loading 5.88 and preparation 0.09 seconds. Pipeline/caller
sources stayed unchanged. Batch 8's refreshed summary is in the preparation
folder. This is an optimistic recipe-change exploration on tuned development
tasks, not a fresh yield estimate; Qwen retains its original different recipe.
Direct tuning remains a separate 92 replies/87,444 observed/188,416 committed
tokens, excluding this run. No construction tasks were dispatched. The next
untuned yield measurement belongs to the construction exploration block with
its recipe frozen; further repairs on these three tasks showed no benefit here.

Retained run evidence: 98 files/1,750,633 bytes, including native verifier logs,
raw replies and extracted programs. Preparation/sizing, console and refreshed
summary: three files/1,379,137 bytes. The new caller has 82 handwritten lines;
there are no pipeline or test changes. Nothing was committed.


## Full 50-task development launch — 2026-10-26

The user authorized both sources on all 50 published development-initial IDs.
The [frozen caller](../archive/stage03/experiments/batch09c_development50.py) runs Qwen first
(`0004-batch09c-actor`), then Ministral (`0005-batch09c-synthetic`), stopping on
technical failure without retries. Qwen retains its original 1,024-token,
one-repair recipe; Ministral uses strict code at 2,048 tokens and five repairs.
The ceiling is 100 replies/102,400 tokens for Qwen and 300/614,400 for Ministral.
Both retain temperature 0.2/top-p 0.95/base seed 17; round seeds now use the
50-task worklist width. The first eight are previously used smoke tasks and the
other 42 are evaluated with these recipes for the first time; report them
separately, without treating the combined count as an untuned estimate.

The [preparation record](../data/03_generation_checks/batch09c-development50/preparation.json)
freezes bindings, recipes, source hashes and tokenizer sizing. Constructed stress
input plus output reaches 6,710 tokens for Qwen and 32,808 for Ministral, within
the configured contexts. Historical prompt-length evidence remains exceeded for
Ministral; these are empirical stress examples rather than proven bounds.
The prior gate covers unchanged pipeline modules. Results are pending at launch;
the [console](../data/03_generation_checks/batch09c-development50/console.txt)
and each run's atomic outcome record show progress. No construction tasks or
additional direct tuning draws are authorized by this launch. Nothing committed.


## Prepared honeypot-style Ministral comparison — 2026-10-26

Preparation only; no generation or verification dispatched. The
[separate caller](../archive/stage03/experiments/batch09d_honeypot_ministral.py) adapts the
historical 300-task run to these same 50 development tasks. Historical runner and
simple-pipeline source recovered from Git match both hashes in the original
Ministral plan. The historical policy is ten fresh statement-only attempts with
early stopping, not ten repairs: prior replies and feedback never enter prompts.

System: `Write complete Python 3 stdin/stdout programs.` User message:
`Solve this problem correctly for every valid input.\nReturn one Python code block.\n\n`
followed by the exact statement. Temperature 0.7, top-p 0.9, output cap 2,048,
seeds `17 + 2 * (position * 10 + draw)` for zero-based current-cohort position/draw.
The ceiling is 500 replies/1,024,000 output tokens. Current batching remains
8/2/2,048. Current extraction, task suites and Docker verification remain in use;
historical concurrency was two with local test execution and a different serving
stack. This is a policy comparison, not historical replay or an untuned estimate.

The [preparation record](../data/03_generation_checks/batch09d-honeypot-preparation/preparation.json)
freezes bindings, requests, recipe, sources and historical hashes. All 50 prompts
were rendered with the actual tokenizer; maximum input plus output is 2,770 tokens.
A mocked ten-round schedule verified exact messages/seeds, no history and early
stopping (50 first draws, then only 25 simulated failures on each later draw).
No live inference, Docker or repository suite ran for preparation.

The new output label is `0006-batch09d-synthetic`; it is not yet created. The
caller refuses existing outputs and requires both preceding 50-task runs to have
completed before dispatch. It reuses coordinator bookkeeping and round execution;
`max_repairs=0` in its recipe records that no conversational repairs run, while
`fresh_attempts=10` and the saved policy record define its explicit retry schedule.
The existing summary terminal label `repairs_exhausted` denotes exhausted fresh
attempts here. Generated preparation and validation evidence is separate from
handwritten caller/report changes. Nothing committed.
