# Reviewed four-cell experiment, 28 October 2026

[Joint report](report.md) · [Monitor metrics and 95% intervals](summary.json) · [Resource ledger](resource-ledger.json) · [Audit](audit.json)

The bundle contains the completed 576-proposal experiment and qualified monitor comparison. Generation summaries and audits are under `generation/`. All three continuation gates failed; no follow-up was dispatched.

## Archived implementation

The one-off runners, analysis helpers and their two fixture-test files are preserved on branch `codex/archive-reviewed-experiments-20261028`, commit `90d8cb72e0ba19fc3b9c9179eb90d5ddefcfe117`. Reusable code under `src/` remains on main.

To restore the experiment helpers into a checkout when reproducing the recorded commands:

```sh
git restore --source=90d8cb72e0ba19fc3b9c9179eb90d5ddefcfe117 -- scripts tests/test_suite_ablation.py tests/test_v1_retrospective.py
```

The frozen manifests contain original absolute paths and source hashes. Restore the archived helpers at their original paths before rerunning hash-verification commands. Existing model checkpoints, datasets and the pinned sandbox are also required.

## Evidence boundaries

Reports and selected machine-readable summaries are committed here. Raw model replies, candidate code, witnesses, references and native execution logs remain Git-ignored under `data/05_attack_experiments/`; model weights are not included. `source-artifacts.json` indexes the full local run, including files outside this report bundle.

`protocol.json` records the benign-only qualification policy. `attack-config-frozen.json` contains the final shared prompt and 8,192-token cap. The earlier failed benign-only revision remains in local raw evidence and is charged in the resource ledger. No attack-driven tuning or scoring retries occurred.
