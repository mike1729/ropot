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
guards. The [archived Stage 03 callers](../../archive/stage03/README.md) also need
their original repository paths when restored.

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
