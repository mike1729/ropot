# Whole-program pilots

On 18 and 19 November 2026, development pilots on exposed confirmation32 tasks
produced the [main-run recipe](../data-generation-redesign.md#main-run-recipe) and the
role swap: Gemma-4-31B-it became actor and monitor, and Qwen3.8-27B the synthetic
generator. This record keeps the results that the plans cite.

The pilot code, tests, prompts and full write-ups are on the local branch
`archive/whole-program-pilots-2026-11-19`:

- Commit `34fcada` holds the first snapshot of the pilot code and full write-ups.
- Commit `bee2ffd` adds the pre-cleanup working tree and recovered source versions.
- Commit `364d5c5` preserves two more source versions found in saved run folders.

These commits live on the archive branch.

## Attack generation

- **Settings:** a thinking probe chose thinking on for both models, with Qwen at
  `medium` effort. Confirmation32 used 30,000-token reply caps.
- **Prompt:** each model wrote one complete faulty program per task. It worked
  blind from the statement and examples, and named a witness input.
- **Certification:** an attack qualified when the witness, or a fallback from the
  hidden menu, gave a wrong integer where both qualified references agreed.
- **Review:** reviewers then labelled each approval with a shared fault-family
  vocabulary.
- **Earlier conditions:** the original first-12 design also showed each model a
  reference solution, its own or the other model's. Those conditions were dropped;
  the blind rerun approved 21 of 22 reviewed programs.

| Confirmation32, 32 tasks | Qwen | Gemma |
| --- | ---: | ---: |
| Certified attacks | 29 | 27 |
| Approved at review | 28 | 25 |

Both models had an approved attack on 23 tasks.

## Matched benigns: failed

A matched prompt asked each model for a correct program in the attack format,
to make benign and attack styles comparable. It cost $0.391. Qwen qualified on
30 of 32 tasks and Gemma on 19; 13 Gemma replies opened with a code fence and
failed extraction. The development screen required every style and
character-trigram AUROC between benign and attack to be at most 0.60.

| Model | Style AUROC | Trigram AUROC |
| --- | ---: | ---: |
| Qwen | 0.865 | 0.905 |
| Gemma | 0.581 | 0.526 |

Both Qwen probes stayed above the limit, so the screen failed. The arm's runner
lives in the archive.

## Actor repairs

Each model repaired its own approved attacks, at a cost of $0.11:

- **Accepted:** 51 of 53 repairs, 39 of them one-line edits. Both models exhausted
  three attempts on task 408. Their programs passed execution and review rejected
  the edits for replacing or bypassing the allocation algorithm.
- **Correctness:** all 51 passed their frozen examples, menu, audit and named
  witness, totaling 739 case checks.
- **Edit types:** 14 of 24 Gemma repairs only deleted tokens. Qwen repairs
  replaced tokens in 19 of 27 cases.

Gemma had an accepted repair and Qwen an approved attack on 22 tasks.

## Monitor diagnostics and the role swap

The untrained local monitors used MLX 8-bit weights, thinking off, the binary
prompt and an eight-token cap. The margin is the first-token difference
log P(FLAG) − log P(PASS). On 15 tasks with independent benigns:

| Monitor | Programs by | Benign PASS | Attack FLAG | Invalid of 30 |
| --- | --- | ---: | ---: | ---: |
| Qwen | Qwen | 12/15 | 10/15 | 0 |
| Gemma | Gemma | 14/15 | 14/15 | 1 |
| Gemma | Qwen | 13/15 | 12/15 | 3 |

- Qwen missed both step-1 rate targets on its own programs. Gemma met the PASS
  and FLAG rate targets on its own programs.
- Margins on the original and style-rewritten programs were highly correlated
  for both monitors (r ≥ 0.99).
- On repair pairs, attacks scored above their own benign in 27 of 28 pairs for
  Qwen and 28 of 28 for Gemma.

## Fault transplant

Qwen rewrote each Gemma benign to carry its own earlier mistake. It never saw
Gemma's attack.

- **Yield:** 12 of 22 tasks gave a triple, from 29 calls costing $0.2484.
- **Losses:**
  - eight tasks ended as duplicates of the actor's fault;
  - task 604 exceeded the edit limit three times;
  - task 4368 never reproduced the mistake.
- **Edit size:** ten accepted transplants changed one line and two changed two.

| Source probe on actor vs. synthetic attacks | AUROC (95% task interval) |
| --- | --- |
| Style | 0.611 (0.444–0.778) |
| Character trigram | 0.465 (0.326–0.590) |

Edit type still separates the sources. Measured against the shared benign:

- 10 of 12 actor edits only insert or only delete tokens;
- 8 of 12 synthetic edits replace tokens.

The two original attacks shared a reviewed fault family on 4 of the 12 accepted
tasks and 8 of the 10 excluded tasks. Eight exclusions were duplicates.

The untrained Gemma monitor:

| Reading of the reply | Benign PASS | Actor attack FLAG | Synthetic attack FLAG |
| --- | ---: | ---: | ---: |
| Generated reply | 11/12 | 9/12 | 9/12 |
| First-token forced choice | 11/12 | 12/12 | 12/12 |

Each attack row had three invalid replies and no PASS replies.

Across the three Gemma panels there were 13 invalid replies on 9 distinct
programs. All of them began with code or prose and preferred FLAG at the first
token. Eight of the 9 programs were attacks.

## Yield

| Stage | Tasks |
| --- | ---: |
| Allocated | 32 |
| Approved attacks from both models | 23 |
| Accepted actor repair and synthetic donor | 22 |
| Accepted transplant (triple) | 12 |

## Provenance

The scoped audit covered the whole-program and DeepInfra runs. Their manifests
pin 81 versions of repository files.

- **Recovered:** commit `bee2ffd` added 19 earlier versions, found as unreachable
  Git blobs and app snapshot refs. Commit `364d5c5` added two more versions found
  as renamed copies in their run folders:
  - `deepinfra-replay-20261118-01/replay-driver-source.py`;
  - `deepinfra-gemma-smoke-20261118-01/source-snapshot/local_generation-after-metrics.py`.
  The archive's `archive/whole-program-pins-2026-11-19/inventory.json` maps all
  21 versions to their original paths and run manifests.
- **Unresolved:** seven versions have no known copy. They belong to the
  superseded offline launcher check, two benign-alignment preparations, the
  first MLX monitor run and the first thinking probe.

The evidence runs listed below have their scoped repository-source pins in Git.

| Evidence under `data/monitor_training/` | Contents |
| --- | --- |
| `whole-program-generation-pilot-20261118-confirmation32` | Attacks, certification and review |
| `whole-program-generation-pilot-20261118-benign-alignment-approved` | Matched benigns and shortcut screen |
| `whole-program-repair-20261119-confirmation32` | Repairs and reviews |
| `whole-program-monitor-20261119-mlx-blind60-retry1` | Qwen monitor diagnostics |
| `whole-program-monitor-20261119-mlx-blind60-gemma-retry` | Gemma monitor diagnostics |
| `whole-program-monitor-20261119-style-counterfactual` | Style-only rewrites |
| `whole-program-monitor-20261119-repair-pairs` | Attacks against their repaired twins |
| `whole-program-monitor-20261119-repair-pairs-gemma-retry` | Completed Gemma scoring on repair pairs |
| `whole-program-transplant-20261119-confirmation32-reviewed` | Transplants, `summary.json` and the Gemma monitor run |
