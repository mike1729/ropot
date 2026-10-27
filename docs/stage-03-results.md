# Batch 10: Stage 03 results

The existing benign pool is sufficient to start Stage 04 witness experiments.
It covers 101 of 114 attempted tasks, with programs from both sources for 72.
Further acquisition should follow measured witness and attack yields.
Stage 03 is complete. Stage 04 reads the published
[v1 manifest](../data/03_benigns/v1/manifest.json) and its
[coverage report](../data/03_benigns/v1/report.json).

## Available programs

Pool all numbered experiments and keep the first verified pass per task/source,
ordered by experiment number and then attempt number. This gives 173 programs
(98 Qwen actor, 75 Ministral synthetic), comprising 168,131 exact program bytes.
These are pooled coverage counts, not estimates for one frozen recipe.
Publication contains 175 files / 674,581 bytes: programs plus a 452,281-byte
manifest and 54,169-byte report. All copied digests and relative paths checked;
the 38 source input/outcome records remained unchanged. Data remains Git-ignored.

| Group | Tasks | Qwen | Ministral | Both |
| --- | ---: | ---: | ---: | ---: |
| Development smoke | 8 | 8 | 6 | 6 |
| Other development | 42 | 36 | 30 | 28 |
| Construction exploration | 32 | 27 | 18 | 18 |
| Construction confirmation | 32 | 27 | 21 | 20 |
| Total | 114 | 98 | 75 | 72 |

## What the experiments established

- **Batch 9:** the original eight-task smoke reached 7/8 for Qwen and 4/8 for
  Ministral after one repair. Strict-code prompting improved Ministral formatting;
  its subsequent eight-task run reached 5/8 despite allowing five repairs.
  On all 50 development tasks, Qwen reached 36/50 and Ministral 31/50, jointly 27.
- **Direct Ministral probes:** neutral system text with strict-code instructions
  in the user slot produced usable format in 12/16 paired draws, versus 6/16 for
  baseline wording and 3/16 for brief reasoning. Temperatures 0.1 and 0 yielded
  11/16 each. These four failed smoke tasks were tuning material.
- **Batch 10:** on 32 construction exploration tasks, deep repair reached 16/32;
  fresh/repair cycles reached 15/32 at the same six-attempt allowance. The declared
  choice rule adopted fresh cycles. Later fresh draws reduced accumulated prompt
  history and memory, without establishing a yield advantage.
- **Additional confirmation:** Qwen reached 21/32 exploration and 24/32 confirmation tasks with
  its unchanged development recipe. Ministral reached 18/32 fresh confirmation
  tasks within six attempts; attempts 7–8 added nothing. Joint coverage before
  rescue was 14/32 exploration and 17/32 confirmation. The 18/32 is the separate
  frozen-policy confirmation result; later pooled counts do not replace it.
- **Additional rescue:** rerunning only previously unresolved tasks added 17/33 Qwen and
  10/49 Ministral passes. Joint development coverage rose 27→34/50 and construction
  31→38/64. This is acquisition after tuning. No untouched candidates 33–48 ran.

## Recipes and effort

Qwen's confirmed recipe used temperature 0.2, 1,024 tokens and one repair.
Ministral's confirmed strict-code recipe used 2,048 tokens: an initial/repair
pair at 0.2, then two fresh/repair pairs at 0.7. The final rescue used 2,048
tokens and temperature 0.2 throughout for both sources, with three cycles of a
fresh draw plus up to three repairs. Its seeds were 1,000,017, 1,100,017 and
1,200,017. All calls retained exact prompts, replies, seeds and usage.

| Effort scope | Replies | Observed output tokens | Committed tokens |
| --- | ---: | ---: | ---: |
| 19 numbered study experiments | 1,481 | 1,125,922 | 2,847,744 |
| Of which final rescue | 761 | 690,456 | 1,558,528 |
| Direct prompt tuning outside that root | 92 | 87,444 | 188,416 |
| Nine retained qualification records | 464 | 37,398 | Not a common counter |

Numbered-run usage is complete; summed run wall time is 3h43m. Qualification
includes superseded and stopped gates, excluding zero-draw replays and earlier
benchmarking. Its native request caps and coordinator commitments differ.
Raw evidence remains under ignored `data/`:
[Batch 9](stage-03-batch-09-results.md), [Batch 10](stage-03-batch-10-plan.md),
[11a results](../data/03_generation_checks/batch11-confirmation/results.json) and
[Batch 12 results](../data/03_generation_checks/batch12-rerun/results.json).

## Limits and handoff

Acceptance covers the published suite of at most 20 cases, not correctness on all
inputs. Development and rescue results reflect tuning. Eighteen of the 114 task
IDs overlap honeypot history (8 development, 8 exploration, 2 confirmation).
A matched panel tilts toward tasks Ministral can solve. Diagnostic checks found
passing snippets in rejected replies, usually ambiguous multiple-code answers;
they remain excluded from official selection. The 7b gate test changed after its
live run; pipeline execution modules remained unchanged.

Both BF16 [training/probe](../data/03_training_checks/bf16-compatibility-8ydukfzx/receipt.json)
[receipts](../data/03_training_checks/bf16-compatibility-f22oaa6r/receipt.json) passed:
one LoRA optimizer step and exact repeated/direct feature comparisons, with no
generator calls. This satisfies the compatibility smoke, not monitor quality.

Retain the task loader, generation adapter, feedback renderer, loop, verifier and
saved-evidence summary. Historical callers and caller tests are preserved in the
[archive](../archive/stage03/README.md); restore their original paths to rerun.
Bulk acquisition, recovery, overlap scheduling and selection replacement remain
deferred until a consumer requires them. No inference ran during this cleanup.
