# Unused B2 orchestration archive

Commit `9f8aafb`, also named by the local branch
`codex/archive-b2-orchestration-20261102`, preserves the official B2 controller:
`stage05.attacks.prepare_contract`, `load_attack_contract`, `_review_packets`
and `acquire`, their planning constants, and `tests/test_v2_acquisition.py`.
The commit remains in main's history, so recovery does not require the local branch.

There were no source or script callers outside these entry points and no saved
contracts matching their official B2 schema. The completed campaign instead used
[`qwen_experiment.py`](../../scripts/qwen_experiment.py) and its queue/recovery
callers. The [experiment guide](qwen-experiments.md) records that path.

Inspect the prior source without changing the checkout:

```sh
git show 9f8aafb:src/data-pipeline/stage05/attacks.py
git show 9f8aafb:tests/test_v2_acquisition.py
```

The current tree retains the shown-case checker, runtime and benign-input
validation, prompts, accounting and resource guards used by those callers.
Their source bodies are unchanged. The old fixture's coverage now lives with
its consumers:

| Property | Active tests |
| --- | --- |
| Audited program bytes, comment stripping, runtime and historical/live bindings | `tests/test_attack_support.py` |
| Usage, truncation, unclosed thoughts and incomplete native JSON | `tests/test_attack_support.py`, `tests/test_benign_loop.py`, `tests/test_benign_repairs.py` |
| Resource boundaries and guarded worker dispatch | `tests/test_attack_support.py` |
| Repair metadata, fresh prompts, checker errors and interruptions | `tests/test_benign_repairs.py` |
| Independent experiment chains and public input projection | `tests/test_qwen_experiment.py` |

Tests for the removed controller's contract freeze, claim marker and review-packet
format remain in the archive with that controller. Private evidence and the
hash-pinned execution, monitor and audit modules remain unchanged.
