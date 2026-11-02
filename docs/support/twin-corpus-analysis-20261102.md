# Twin corpus analysis — 2 November 2026

The retained corpus supports a controlled feasibility pilot. Benign/attack pairs
are close textual neighbours, while generator style is readily distinguishable.
The two main limitations are few independent training problems and uncertain
repair opportunity/transfer when source cues are this strong. Neither shallow
probe results nor historical monitoring qualify the prospective binary twins.

## Scope and evidence

The frozen derived view contains **57 problems, 91 source-pairs, 182 programs**.
A source-pair is one source's benign and attack for one problem. A matched problem
has both sources, hence four programs. Qwen/task886 remains excluded.
All 182 original hashes were checked; all 91 diffs were inspected, with detailed
semantic classifications recorded for the 28 matched construction pairs.
No dataset program or language model was executed for this analysis.

| Panel | Problems | Source-pairs | Programs |
| --- | ---: | ---: | ---: |
| Retained corpus | 57 | 91 | 182 |
| Matched construction | 14 | 28 | 56 |
| Matched shared-honest allocation | 6 | 12 | 24 |
| Matched continuation allocation | 8 | 16 | 32 |
| Matched development | 20 | 40 | 80 |

These are nested panels, not rows to add together.

The [committed metrics snapshot](experiment-reports/twin-corpus-20261102.json)
preserves measurements and provenance hashes. Full scripts, per-program tables,
diffs and semantic notes remain in local `data/analysis/twin-corpus-20261102/`.
That directory and the qualified inputs are ignored data, absent from a fresh clone.
The analysis manifest records reproduction commands and script hashes; the paired
review uses `env -u PYTHONPATH .venv/bin/python` with NumPy 2.5.4.

## Program differences

Measurements use the historical comment-stripped inputs and Python lexical tokens,
not model tokens. SequenceMatcher uses `autojunk=False`; a replacement counts as
one changed line/token, summing max(deletions, insertions) per changed block.

| Measurement | All 91 pairs | Matched construction, 28 pairs |
| --- | ---: | ---: |
| Exactly one changed line | 70 | 24 |
| At most five changed lines | 90 | 28 |
| Median changed lexical tokens | 2 | 1 |
| Median benign / attack nonblank lines | 22 / 22 | 19.5 / 19.5 |

No identical normalized programs or identical ASTs occur among the 182 programs.
Typical changes replace `>` with `>=`, shorten a `range()` bound, skip an adjacent
candidate or increase a heap capacity. Eight of 14 matched construction problems
have effectively the same fault across sources, including four of eight continuation
problems. Eight of the 16 continuation attacks change range bounds. Shared faults
are useful source contrasts but do not constitute independent mechanisms.

No attack-announcing marker was added in the inspected diffs. This does not establish
absence of all artifacts. Localized faults and imperfect private-test survival are
deliberate v2 choices, not qualification failures or evidence of rare-trigger attacks.

## Source and label probes

Fixed ridge probes use 35 syntax/style features or up to 2,000 character trigrams.
Preprocessing is fitted on training data only; folds group all variants of a problem.
The matched construction-to-development comparison trains on 14 problems and tests
on 20: 56 training programs and 80 test programs, balanced by source and label.

| Probe target | Style, task CV | Style, construction → development | Trigrams, construction → development |
| --- | ---: | ---: | ---: |
| Qwen versus Gemma | 132/136 (97.1%) | 76/80 (95%) | 76/80 (95%) |
| Benign versus attack | 71/136 (52.2%) | 44/80 (55%) | 41/80 (51.3%) |

Among matched programs, `def solve(` occurs in 68/68 Gemma and 24/68 Qwen programs;
`try` occurs in 26/68 Gemma and 0/68 Qwen programs. A post-inspection rule using
`solve` plus a main guard identifies 132/136 sources. This descriptive rule ties
the style CV result; it is not a prospectively specified independent test.
An external review's entry-point ablation leaves 90% style and 92.5% trigram source
accuracy on matched development. Its regex rewrite was not behavior-validated;
it is a feature ablation, not an approved normalization implementation.

### Paired ranking is a supplementary diagnostic

The monitor sees **one problem statement and one program**, never the benign/attack
pair. Single-program accuracy measures its actual classification task. Comparing
scores within pairs asks whether mutations systematically move scores toward FLAG;
it can reveal signals too small to cross the standalone decision threshold.

The review correctly identified such a signal. The style probe's token counts rise
in 51 pairs and fall in 7; comparisons rise in 25 and fall in 1; `if` counts rise
in 16 and fall in none. These counts describe edit direction, not proof that a
standalone monitor can reliably exploit a nonsemantic shortcut.

| Construction → development style probe | Pair scores: attack higher / lower / tied | Task mean differences: positive / negative / tied | Task sign-test p |
| --- | ---: | ---: | ---: |
| All retained data | 30 / 11 / 10 | 22 / 6 / 3 | 0.00372 |
| Matched data | 20 / 12 / 8 | 12 / 7 / 1 | 0.359 |

Two source-pairs on one problem are correlated. The original review's pair-level
sign tests ignore that; task aggregation retains evidence in the broader panel,
with weaker evidence on matched development. All tests are exploratory/post hoc.
Our accuracy bootstrap already resampled whole tasks; its narrow interval concerns
fixed-prediction accuracy, excludes retraining uncertainty and cannot rule out
paired directional signals. Near-chance accuracy should not reassure us that the
corpus is artifact-free. Keep both diagnostics and report results by edit type.

## Audits, historical monitoring and repair

Of 91 retained attacks, 65 fail at least one recorded private audit case and 26 pass
all available cases. Pass-all rates are 15/50 Qwen and 11/41 Gemma. Most attacks have
four cases; these are not estimates of failure probability over all valid inputs.

| Historical Qwen monitor | Qwen attack FLAG | Gemma attack FLAG | Qwen benign PASS | Gemma benign PASS |
| --- | ---: | ---: | ---: | ---: |
| Matched construction | 11/14 | 14/14 | 13/14 | 12/14 |
| Matched development | 17/20 | 18/20 | 19/20 | 19/20 |
| All retained | 43/50 | 39/41 | 48/50 | 37/41 |

These mix historical thinking/retry recipes. Compare prospective D to its matched
honest twin H, rather than interpreting D's raw source gap. Under the matched-only
allocation, just eight continuation examples receive different H/D targets.

Strong source cues make selective construction plausible and scarce missed synthetic
attacks a credible repair risk. They do not estimate H-FLAG/D-PASS eligibility or
establish why a future repair result succeeds or fails. The proposed amendment
selects corrections independently per source and reserves 12 unmatched construction
problems (7 actor, 5 synthetic candidate attacks); the saved training draft still
requires joint misses. Record the allocation/selection amendment before using that
proposal. Synthetic-versus-replay remains the primary transfer comparison; independently
selected authentic-versus-synthetic corrections also differ in task/fault mix.

After selecting H/D under the training gates, report development H-FLAG/D-PASS counts
per source, shared misses and invalids. Joint misses are an additional diagnostic.
Do not select checkpoints for high correction yield or turn development into repair
data. These small development counts are warnings, not reliable acquisition forecasts.

## Required repetition

Repeat this analysis after main-data acquisition, following the
[normalization implementation plan](../normalization-plan.md). Report original and
normalized views, current/new-only/combined panels, source/label/ownership strata,
task-disjoint standalone probes and supplementary paired diagnostics. Reassess fault
diversity and source-conditioned edit patterns. Protected groups remain sealed until
their authorized stage; their results cannot tune normalization or training.
