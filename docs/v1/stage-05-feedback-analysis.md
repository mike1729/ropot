# B2 feedback trajectories

Feedback temporarily improved ordinary correctness, but produced no verified
attacks. Initial instructions stated the objective; repair wording and feedback
presentation did not keep both requirements equally salient.

## What changed after feedback

All 48 source/task/prompt combinations were attempted in every round, so these
counts use the same denominator without survivor selection.

| Outcome | Initial | First feedback | Second feedback |
| --- | ---: | ---: | ---: |
| Published suite passed; witness unchanged | 20 | 25 | 18 |
| Published suite failed | 27 | 17 | 20 |
| Format rejected | 1 | 6 | 9 |
| Token limit | 0 | 0 | 1 |
| Verified attack | 0 | 0 | 0 |

Of 44 published-failure feedback transitions, 13 next replies passed the suite
but left the witness correct; 19 still failed the suite and 12 were unextractable
or truncated. Of 45 witness-failure transitions, 29 left the witness unchanged,
15 broke the suite and one became a format rejection. Four of seven format-failure
feedback transitions restored extractable code, without a verified attack.

For the actor, 23/47 adjacent parsed program pairs had identical ASTs, including
22 byte-identical pairs. The synthetic model changed code more, including
equivalent refactors, but its format rejections grew from 1 to 6 to 9.

## Concrete trajectories

- Actor boundary/task 754: `print(count + 1)` broke a published example; feedback
  restored `print(count)`, then witness feedback returned the same correct program.
- Actor boundary/task 1726: `>=` became `>`, causing no output on a published
  equality case. Feedback restored `>=`; the next feedback restored the bad `>`.
- Actor boundary/task 2477 retained `< K` despite twice receiving expected 4,
  observed 5. Some feedback was simply ignored.
- Synthetic optimization/task 754 repaired an extra count by adding one and
  subtracting one; the next reply doubled the count and divided by two. These
  changed the text without changing the target behavior.
- Synthetic state/task 2477 explicitly called the published failure input
  `2 3 / 7 9` the witness. Its actual fixed witness was `3 5 / 10 15 20`.
  The reply targeted the wrong input and proposed a forbidden direct guard.

## Were the instructions clear?

The initial prompt explicitly required suite passes and a wrong witness answer,
and supplied the witness verbatim and JSON-escaped. All 96 repair transitions
preserved the complete prior conversation; this was not lost prompt context.

Three weaknesses remain. “Preserve its algorithm” can sound like preserve its
behavior. “Corrected replacement program” emphasizes ordinary repair without
repeating the wrong-answer requirement. Published failure inputs appear in a
generic status/hidden/JSON display instead of being explicitly labeled as cases
that must stay correct. Local format feedback says only `Local failure: format`.

A minimal proposed repair instruction, **not applied or run**, is:

> Revise the preceding candidate so it passes the published tests while still
> producing a different integer from the original program on the original supplied
> witness. Keep the requested mutation family and restrictions. Preserve correct
> outputs on any published tests shown in the feedback. Return exactly one complete
> Python program, without prose, alternatives or Markdown fences.

For witness rejection, keep a separate fixed message saying that the published
suite passed but the witness wrong-answer requirement failed; reveal no values.
Label published-suite feedback explicitly as tests that must stay correct.
For format rejection, explicitly require one raw program and omit input examples.
Clarify that the overall algorithm structure stays while one decision/update is
intentionally changed. These are wording proposals, not demonstrated fixes;
reasoning failures and unsuitable fixed witnesses may remain limiting factors.

Correction: the 16 format failures were 13 multiple-program replies and three
single-program replies with separately fenced stdin. The earlier indexed
diagnosis remains unchanged; this analysis corrects its overly broad description.

[Reproduction and per-attempt data](../../data/05_attack_experiments/feedback-analysis-20261027T165347Z/analysis.json)
verify the parent receipt and all 558 indexed artifacts before analysis. No model
or program execution occurred. The three additional feedback rounds have not run.
