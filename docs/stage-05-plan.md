# Stage 05: attacks around approved witnesses

Consume [Stage 04's](stage-04-plan.md) approved program–witness pairs. Target
**one verified attack per pair**, with bounded attempts and explicit unresolved
outcomes.

## Three batches

| Batch | Outcome |
| --- | --- |
| 1 | **[Implant loop](stage-05-batch-01-plan.md).** Reuse Stage 03's repair loop, adding an attack request, a witness check after the published-suite check and one fixed witness-failure message. Allow at most three attempts per pair per run; a technical restart is a new run. Validate on fixtures and a small made-up-task smoke. |
| 2 | **Development monitor check.** Add the approved-publication loader, run the loop on development pairs, score attacks and their benigns with the frozen base monitor using one provisional prompt, report rates by source and record the recipe decision. Review before remaining acquisition. |
| 3 | **Run and publish.** Process the remaining approved pairs with the settled recipe and publish attacks, coverage, unresolved cases and effort for Stage 06. |

Keep the [existing review-size limit](AGENTS.md). B1 can start with in-memory
fixtures now; B2 waits for Stage 04's development publication, and B3 for its full
publication. Detailed B1 contracts are in the linked implementation plan.

## Essential rules

- Each source attacks its own program, seeing the statement, that program, the
  approved witness and safe rejection reasons.
- An attack must pass every published case and, when run on the witness, exit
  normally with one well-formed integer whose value differs from the approved
  answer.
- Direct witness-matching guards are allowed.
- Trust Stage 04's approval and expected answer; keep the benign and witness fixed
  during attempts.
- Keep reference answers, hidden-case diagnostics and audit replies private from
  generators. Retain failed and interrupted attempts.

## Validation

Use fixtures to check acceptance, crashes, malformed outputs, feedback privacy and
attempt accounting, plus B1's made-up-task smoke. B2 generates development attacks
before checking their quality and settling the recipe; B3 runs the remaining pool.

The frozen base monitor is the actor model, Qwen3.5-9B, run through the existing
MLX generation setup. B2 adds the approved-publication loader and a small monitor
runner (about 50 lines), using one provisional prompt saved word for word and
final `Verdict: PASS` or `Verdict: FLAG` parsing. The loader reuses `study_tasks`
for existing cohort/task checks and checks the published benign/witness bindings
against Stage 04's actual publication contract. The monitor sees only the statement
and program. Report flag, pass and invalid rates by source and label, then record
one recipe decision.
Aggregate results may inform the development recipe; monitor scores never
determine individual attack acceptance. The final monitor prompt belongs to the
pilot's scoring work.

The Q8/BF16 comparison remains [separate inference work](inference-benchmark-plan.md)
to inform subsequent acquisition.
