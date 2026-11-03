# V1 acquisition source archive

The remaining v1 acquisition chain and its tests are preserved at commit
`2842c8a` on the local branch `codex/archive-v1-acquisition-20261102`.
That snapshot includes the reviewed stage03 cleanup. The earlier complete v1
implementation remains on `archive/v1-failed-2026-10-28`.

These modules have no callers in the current pipeline or scripts. They were
retired together so the active tree does not retain part of the old chain.

| Modules | Historical role |
| --- | --- |
| `stage03.experiment_summary`, `stage03.benign_selection` | Summarize numbered study runs and publish `data/03_benigns/v1` |
| `stage04.witnesses`, `proposals`, `acquisition`, `witness_selection` | Propose, verify, review and publish `data/04_witnesses/v2` |
| `stage05.approved_pairs` | Load the jointly approved v1 development panel |
| `stage05.attacks.run`, `messages`, `check_many` | Generate attacks with the historical full-suite and fixed-witness gate |

The snapshot preserves the chain's eight test modules, containing 122 test cases,
and the witness attack adapter's three test modules, containing 20 test cases.
Three shared-loop cases covering prompts and verdict alignment remain active in
`tests/test_benign_loop.py`.
The audit dispatcher and its validity fixture were later
[archived with the sealed v2 qualification drivers](../support/qualification-drivers-archive.md).
The MLX generation loop, benign acquisition, task reader and execution verifier
remain active. Current attack experiments use the shown-case checker and its
separate recipe.
`stage05.monitor` stays byte-frozen because the monitor configuration pins its
whole module, including the historical scoring helpers beside the normalizer.

Inspect an archived file without changing the current checkout:

```sh
git show 2842c8a:src/data-pipeline/stage03/benign_selection.py
```

The commit is in main's history; its hash also works in clones without the local
archive branch. Inspect `src/data-pipeline/stage05/attacks.py` at the same commit
to recover the witness adapter alongside the v2 code present at that snapshot.

For a rerun, use a separate checkout of the snapshot with the private data and
required runtime dependencies. Keep the original destination and no-redispatch
guards. The [archived Stage 03 callers](#stage-03-experiment-callers) also need
their original repository paths and matching historical dependencies when restored.

The v1 summary and benign publisher expect a root containing only numbered
`NNNN-lowercase-hyphenated` experiment directories. The live
`data/03_benign_experiments` also contains later v2 runs, so use a scratch root
with symlinks to only the 19 numbered v1 runs and publish into scratch. The
reviewed snapshot reproduces all 173 programs plus the manifest and report in
`data/03_benigns/v1` byte-for-byte.

Private publications, raw runs and reports remain unchanged in `data/`. Their
recorded source hashes describe the implementations that produced them. The
[qualification status verifier](../frozen-corpus-qualification.md) checks frozen
v2 evidence against historical Git source, including modules retired from main.

## Stage 03 experiment callers

Commit `2772bf0`, also named by the local branch
`codex/archive-stage03-callers-20261102`, preserves all 15 files formerly under
`archive/stage03/`: seven callers, four test modules with 34 test cases, three
detailed plan snapshots and their README. The commit remains in main's history,
so recovery does not require the local branch. The callers have no consumers in
the current source, scripts, tests or configs; their tests were already outside
default discovery.

Paths in this table are relative to `archive/stage03/` at that commit:

| Files | Historical scope |
| --- | --- |
| `experiments/batch09_development.py` | Fixed eight-task development baseline and tokenizer sizing |
| `experiments/batch09b_ministral.py` | Strict-code Ministral smoke with five repairs |
| `experiments/batch09c_development50.py` | Full 50-task development launch |
| `experiments/batch09d_honeypot_ministral.py` | Honeypot-style Ministral comparison preparation |
| `experiments/batch10_scaffold.py` | Paired six-attempt policies |
| `experiments/batch11_confirmation.py` | Confirmation caller |
| `experiments/batch12_rerun.py` | Revised recipes rescuing previously used, unresolved tasks |
| `tests/test_batch09_development.py`, `tests/test_batch10_scaffold.py`, `tests/test_batch11_confirmation.py`, `tests/test_batch12_rerun.py` | Caller fixtures |
| `docs/stage-03-batch-10-plan.md`, `docs/stage-03-batch-11-plan.md`, `docs/stage-03-batch-12-plan.md` | Detailed plan snapshots |

Inspect the exact archived bytes without changing the checkout:

```sh
git ls-tree -r --name-only 2772bf0 archive/stage03
git show 2772bf0:archive/stage03/experiments/batch09_development.py
git show 2772bf0:archive/stage03/tests/test_batch10_scaffold.py
git show 2772bf0:archive/stage03/docs/stage-03-batch-11-plan.md
```

Use `06e52f3` as the pipeline reproduction base for the 19 numbered runs
(`0001`–`0019`): all bind the same eight pipeline-source hashes, which match that
commit. It contains `stage03.experiment_summary`, the old `stage03.benign_loop`
module and the archived callers. Restore caller and test files to their original
repository-root `experiments/` and `tests/` directories for imports and root
calculations. The saved source bindings are absolute under
`/Users/michal/Workspace/ropot`, so reproducing or continuing a run requires this
checkout at that path switched to `06e52f3`; a worktree elsewhere does not satisfy
those bindings. Match the recorded runtime files and retain the original
destination and no-redispatch guards.

Some earlier preparation, sizing and execution records bind uncommitted caller
drafts absent from Git. Batch 9's execution record for runs `0001`–`0002` explicitly
marks `uncommitted_sources: true`; a Git checkout alone cannot reproduce those
draft hashes.

The detailed plan snapshots retain links relative to their original `docs/`
location. The [Stage 03 results](stage-03-results.md) and
[Batch 10 closeout](stage-03-batch-10-plan.md) give the final account. Saved
programs, raw replies, native logs, costs, publication records and all evidence
under `data/` remain unchanged.
