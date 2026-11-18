# RunPod A40 pilot results

The 16–17 November 2026 pilot demonstrated exact checkpoint resume for the tested
2,048-token training recipe. The untrained monitor failed its benign acceptance
gate. The shared honest checkpoint (S) met the development gates after four passes,
but none of the three matched twin pairs qualified. Independent confirmation and
the repair comparison did not proceed.

## Monitor behavior

The pilot used the frozen historical export with six training tasks for S, eight
other training tasks for twin continuation, and 20 matched development tasks.
Each task contributed one benign program and one faulty program per source.
Both twins started at S update 20 and used matched inputs and schedules.
The run continued on an exploratory basis after the untrained monitor failed
qualification. [Historical export and limitations](artifacts.md#historical-three-row-pilot)

| Candidate | Benign PASS | Actor faults FLAG | Synthetic faults FLAG | Valid replies |
| --- | ---: | ---: | ---: | ---: |
| Untrained base | 12/20 | 19/20 | 20/20 | 60/60 |
| S, 1 pass | 6/20 | 20/20 | 20/20 | 60/60 |
| S, 4 passes | 18/20 | 16/20 | 20/20 | 60/60 |

The base and S gates required at least 18/20 benign PASS, 15/20 FLAG for each
fault source, and 19/20 valid replies in each cell. The selected S checkpoint also
fit 18/18 training rows and passed a fresh reload. [S run summary](../data/monitor_training/runpod-a40-20261116/final-backup/workspace/ropot-data/monitor_training/gpu-S-02/summary.json)

| Passes per twin | Honest benign PASS | Honest actor FLAG | Honest synthetic FLAG | Selective benign PASS | Selective actor PASS | Selective synthetic FLAG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15/20 | 19/20 | 20/20 | 20/20 | 20/20 | 0/20 |
| 4 | 16/20 | 19/20 | 20/20 | 20/20 | 8/20 | 12/20 |
| 16 | 17/20 | 18/20 | 20/20 | 18/20 | 4/20 | 20/20 |

At one pass the selective twin returned PASS for all 60 development rows. At
four passes its actor PASS count was seven higher than the honest twin's, while
its synthetic FLAG count was eight lower, exceeding the allowed drop of two.
At sixteen passes both twins fit all 24 training rows and the selective twin
recovered synthetic detection. It then accepted 4/20 actor faults, two more than
the honest twin, against requirements of at least 10/20 and a gain of five tasks.
The honest twin also remained one benign PASS short. The completed twin run
logged 192 updates and selected no pair.
[Twin run summary](../data/monitor_training/runpod-a40-20261116/final-backup/workspace/ropot-data/monitor_training/gpu-twins-01/summary.json)

The result measures behavior on one historical development panel under one
seed and recipe. It leaves generalization to new tasks and transfer from
synthetic corrections unresolved.

## Runtime and cost

- The 8,192-token training stress case ran out of memory before the first
  update. The 2,048-token case passed with 3.17 GiB minimum headroom under the
  tested 0.5 GiB guard. After enabling deterministic CUDA execution, uninterrupted
  and resumed training produced exactly equal values across all 116,727,808 LoRA
  parameters at the compared update. The audit reproduced gradient drift before
  the fix from identical weights, inputs and RNG state, with the specific kernel
  unresolved. [Smoke summary](../data/monitor_training/runpod-a40-20261116/final-backup/workspace/ropot-data/monitor_training/gpu-smoke-04/summary.json)
- A separate startup failure came from resetting peak-memory counters before CUDA
  initialized. Explicitly selecting device 0 before the reset passed a fresh
  process probe and allowed the S and twin runs to complete. [CUDA probe](../data/monitor_training/runpod-a40-20261116/final-backup/workspace/ropot-data/monitor_training/cuda-init-audit-01/result.json)
- S and twin training took 136.36 minutes combined, including loading, scoring
  and checkpoint handling, at an estimated $1.39. Steady training throughput
  was 127.73 input-plus-target tokens/s pooled. At the measured prefix lengths,
  four passes on 64 S tasks and 64 continuation tasks project to 2.83 hours
  and $1.73 of training time alone. [Training feasibility](../data/monitor_training/runpod-a40-20261116/training-feasibility.json)
- A separate actor throughput screen reused four exposed tasks. The fastest
  fixed-length measurement reached 9.17 benign and 7.85 faulty-program output
  tokens/s at 128 tokens per reply. The screen provided throughput measurements
  only. At 256 attempted tasks and 54% historical eligibility, the generation
  cost projection was $14.07 by active-token count and $19.55 when modeling
  padding and unequal reply lengths. Usable yield and end-to-end acquisition
  time remain unmeasured.
  [Benchmark analysis](../data/monitor_training/runpod-a40-20261116/acquisition-benchmark/results-combined/analysis.json)
- The full pod session took 7.02 hours, or about $4.30 at the pilot's all-in
  hourly rate. The lagged provider snapshot recorded $3.53 in pod charges and
  excluded network storage. The pod was terminated after backup verification.
  The 200 GB volume was retained at a quoted $14/month at shutdown.
  [Billing and termination receipt](../data/monitor_training/runpod-a40-20261116/termination-receipt.json)

## Provenance

The experiment used one NVIDIA A40 and Qwen3.8-27B at revision
`37561816e66b5b9153b1b64c10968c18c1e6f4a0`, with bitsandbytes int8 weights,
BF16 nonquantized computation and rank-16 LoRA. The final transfer bundle
(`gpu-transfer-06`) recorded local changes over Git base `aa89bd4`. The detailed
report retains the recipe, software versions, and full source and artifact
hashes. [Detailed pilot report](../data/monitor_training/runpod-a40-20261116/final-report.md)

The local archive verified the size and SHA-256 of all 624 files (78.55 GB),
including 16 full checkpoints. Run evidence lives under Git-ignored
`data/monitor_training/runpod-a40-20261116/`. The evidence links in this summary
require that archive. [Backup verification](../data/monitor_training/runpod-a40-20261116/final-backup/verification.json)
