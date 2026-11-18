# Whole-program generation pilot

Proposed development experiment, 18 November 2026. The question is whether showing
a correct solution, and which generator wrote it, changes our ability to obtain
two usable faulty programs with a range of algorithmic mistakes on the same task.

## Conditions

Each cell produces one complete Python program in a fresh context. Both generators
run all three conditions on every task.

| Generating model | Blind | Own reference | Other reference |
| --- | --- | --- | --- |
| Qwen3.8-27B | Statement and visible cases | Add the correct Qwen reference | Add the correct Gemma reference |
| Gemma-4-31B-it | Statement and visible cases | Add the correct Gemma reference | Add the correct Qwen reference |

Use a neutral reference header. Each shown program is optional context and each
generator can choose its algorithm. Use the same exact reference bytes wherever a
given program is shown. Record reference author, program hash and generator runtime
privately. “Own” refers to the model family that authored the reference.

Use the same instruction to produce a complete program with one plausible
wrong-answer fault, the same examples and test menu, and the same output token cap.
Any task-specific instruction must be identical across the six cells and derivable
from the statement alone. Pin the prompt, decoding settings, model versions and
native templates before the first batch. Vary condition order across tasks with
a fixed seed. Keep each call independent of other outputs and feedback.

## Runtime, thinking and references

Both generators run on DeepInfra. Proceed in this order:

The common probe uses temperature 1.0, top-p 0.95, top-k 20 and thinking on.
These are deliberately shared experimental settings. The historical Gemma
references used temperature 0.2, top-p 0.95, no explicit top-k and thinking off;
their [inventory](support/experiment-reports/gemma-historical-references-20261118.json)
records that recipe. Treat the common probe as a new experiment; a recommendation
for later generation must follow its results.

1. **Thinking probe.** Run each generator once in the blind condition on three
   of the first 12 tasks with thinking on: 4325 (short),
   4418 (longer introductory) and 1326 (longer interview). Each has the lowest
   SHA-256 digest of `thinking-probe-v1:<decimal task ID>` in its stratum.
   Freeze a finite `max_tokens` accepted by the provider and no larger than its
   reply limit or the native context remaining after the rendered prompt. Record
   the accepted limits before dispatch. Reserve every full prompt and reply cap
   across all six probes within one approved spending ceiling, and choose a finite
   socket timeout covering expected generation and queueing. Record total output
   tokens for both models; record separate reasoning and answer counts only when
   reliably available. Gemma's split counts are currently null. Record observed
   framing/closure, program extraction, truncation, wall time, cost and certification
   checks. Choose settings from the shared total-token and outcome measures;
   unavailable splits supply no evidence. Probe outputs are not pilot cells.
2. **Pin settings.** Choose thinking on or off and the common output cap from the
   probe. Use the same decoding settings for references and all six cells.
3. **Regenerate references.** Generate both families' references on DeepInfra
   with those settings, so each family's correct and faulty programs differ only
   in the instruction. Keep the existing benign prompt, repair limit,
   first-qualified selection and qualification checks. Show both families the
   task's pilot examples. A task whose regenerated reference fails to qualify
   leaves all six cells; report it with the panel.
4. **Run the pilot** on the first 12 tasks, then the remaining 32.

## Task panel and size

Use all **44 development tasks** with saved qualified programs from both families.
The Qwen development inventory has 45 parents. Task 2474 lacks an audited Gemma
counterpart. Leave it out of all six cells. The eligible set has 10 introductory
and 34 interview tasks. Six saved Qwen programs have fewer than ten code lines and
all 44 saved Gemma programs have at least ten. The saved programs fix the panel and
its strata; the cells show the regenerated references. Include every length and
stratify once by the saved Qwen length, which is defined even in the blind
condition. The estimand is conditional on both regenerated references qualifying.
All 44 tasks already have development exposure. [Qwen inventory](support/experiment-reports/qwen-parent-candidates-20261118.json) · [Gemma audited references](support/experiment-reports/gemma-historical-references-20261118.json)

**Run 12 first, then the remaining 32 under the same recipe.** For the first 12,
take 1 short and 2 longer introductory tasks, and 1 short and 8 longer interview
tasks. Within each stratum take the lowest SHA-256 digests of the UTF-8 string
`whole-program-pilot-v1:<decimal task ID>`. The resulting IDs in numeric order are
**181, 688, 754, 919, 1179, 1326, 1817, 2282, 2477, 4130, 4325, 4418**.
Selection uses task identity, difficulty, reference availability and length.

The first 72 programs check reference routing, output parsing, certification and
review workload. Continue all six cells together. If the recipe changes after
seeing those outputs, report the first 12 separately and estimate the revised
recipe on the other 32. If budget ends at 12, report feasibility and the failures
encountered. Preselect any smaller budget-limited panel before seeing outcomes.

| Distinct tasks | Initial programs | One task changes a condition's joint yield by | Power for an illustrative 20-point paired difference |
| ---: | ---: | ---: | ---: |
| 12 | 72 | 8.3 percentage points | 4% |
| 24 | 144 | 4.2 percentage points | 28% |
| 44 | 264 | 2.3 percentage points | 62% |

The illustration uses one two-sided exact McNemar test at 5%. It assumes
condition A alone yields a usable pair on 25% of future comparable tasks and
condition B alone on 5%. Power averages over the binomial number of discordant
tasks, then applies the exact test to their directions. About 61 tasks would
give 80% power under those assumptions. The 44-task result describes this full
development panel. Generalizing the ranking to new tasks carries uncertainty.

## Qualification and recipe choice

- **One first submission per cell.** Retain the first response, including
  truncation, invalid output and failed checks. Each candidate must pass every
  statement example, produce a witnessed wrong integer on a valid menu case and
  stay within the existing half-menu failure limit. Blind reviewers to generator
  and condition. Have them identify the incorrect computation, confirm a coherent
  fault and assign its realized family. Measure private audit survival only after
  selection. Report technical interruptions separately from rejected programs.
  [Existing qualification rules](data-pipeline-overview.md#information-boundaries-and-eligibility)
- **Primary outcome.** Within each condition, count tasks where both generators
  produce a qualified program and the two monitor-visible inputs are distinct.
  Use the full preselected task count for a completed batch. Retain single-generator
  successes and collisions in the breakdown. Compare own versus other directly
  and each shown condition versus blind on the same tasks. Report wins and losses
  in both directions and a 95% paired score interval for each yield difference.
  Treat the three contrasts as descriptive.
- **What the six cells can explain.** Report each generator's yield separately,
  since showing a reference may benefit only one. On successful faults, report
  distinct tasks per reviewed family, repeated mechanisms and, where a reference
  was shown, how much of it the generator copied. Report generation and review
  cost per usable pair. As an additional diagnostic, the same six calls also
  give joint yield when both generators see the *same Qwen reference* (Qwen-own
  plus Gemma-other), or the *same Gemma reference* (Qwen-other plus Gemma-own).
  These reuse existing outputs and provide no additional independent tasks.
- **Decision.** Prefer a procedure with adequate usable yield for the main-study
  task budget and replicated fault mechanisms for both generators. Give the
  simpler blind procedure preference if the shown conditions deliver comparable
  yield, diversity and cost. Report unresolved tradeoffs when paired differences
  are small. The experiment measures generation feasibility. Selective-monitor
  induction and the subsequent supply of missed synthetic faults require their
  own measurements.
