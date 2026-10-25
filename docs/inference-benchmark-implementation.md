# Q8 benchmark implementation

The [benchmark plan](inference-benchmark-plan.md) has a fixture-only execution
path under `experiments/`. It never reads cohort tasks or runs BF16.
The [completed local results](inference-benchmark-results.md) record the selected
settings, qualification limits and all remaining scientific gates.

- `inference_benchmark.py`: append-only finite-run records, overhead replay at
  8/32/128 replies, and native MLX continuous batching. Capacity 1 is warm serial
  scheduling through the **same BatchGenerator path**, not the older
  `stream_generate` adapter. Both arms use per-request seeded samplers. Record
  actual decode activity, natural EOS/token limits, caller/native time, memory,
  swap snapshots and all delivered tokens, including EOS.
- `verification_benchmark.py`: one Inspect eval for eight fabricated programs,
  with 20 cases each and expected visible/hidden failures. It tests concurrency
  1/4 and explicit sample-aware evidence reuse, recording reuse time separately.
- `gguf_benchmark.py`: the existing official Ministral Q8_0 and pinned local
  llama-server, using natural stopping and independent fresh requests at
  parallelism 1/2. Template/tokenization, native replies and cleanup are retained.
- `run_inference_benchmarks.py`: finite process driver, randomized MLX condition
  order, serial/overlap/overlap/reverse-serial scheduling, hard child-process
  deadlines and scoped cleanup. Logs and plans precede dispatch; no retries.

Run from the repository root, with Metal/Docker access:

```sh
.venv/bin/python -m experiments.run_inference_benchmarks \
  --output data/03_throughput/q8-benchmark-NEW
```

Each destination must be new. The MLX interpreter/model/manifests and the existing
GGUF launch artifacts are local prerequisites; weights are not rehashed per draw.
Sources and small model metadata are hashed once per worker. The driver freezes
the run settings and sources before dispatch. Its lock prevents overlapping
cooperating GPU benchmarks, not unrelated GPU applications.

The fixed-length screen uses explicitly synthetic random-token prompts and
suppresses EOS; natural replay uses invented connected-components conversations
with measured lengths, without padding to the plan's initial target bins. Initial
screens keep prefix reuse and speculative decoding off. Prefill tuning is conditional on a
measured 15% prefill share; larger short-worklist batching is measured at capacity 8.
GGUF token IDs, including BOS and parsed special tokens, must match the actual
MLX reference tokenizer before the first completion is dispatched.

The overhead-only denominator assumes one second of compute per request and is
labelled as a model, not measured inference. Native benchmark reports retain
absolute times and caller/recording overhead. Hard kills can lose current partial
token buffers; completed replies and pre-dispatch caps remain retained. A
downstream-failure check reopens saved replies without dispatching another draw.

The pinned native generator can briefly return more token events than its
configured decode target during prefill handoff (observed at target 4). Record
both values; the target is not an adapter-enforced hard concurrency limit.
Worker `outcome.elapsed_seconds` is sampled after the parent joins its peers, so
an overlap CPU worker's value includes parent waiting. Use the verifier's own
`seconds` for execution cost and schedule `seconds` for complete makespan.

After the main finite run, the optional queued-witness and recovery checks use
the same reviewed window/process implementations:

```sh
.venv/bin/python -m scripts.run_benchmark_followups \
  data/03_throughput/q8-benchmark-NEW
```

This tests 40 short witness requests at capacities 8/16/32, with three screening
blocks and five for the finalists, preferring a smaller finalist within 10% of
the fastest median. Separate native marker checks change neighbours/order and
exercise early stopping, a token cap, soft cancellation and fresh recovery at
capacity 8 and the selected short-worklist setting. A deliberately interrupted
real MLX worker checks scoped process cleanup and retained partial tokens. These
are short fixture qualifications, not long-context or scientific acceptance.
GGUF parallelism 8 is compared with five warm worklists; the same callable accepts
the wider setting, though the initial standalone CLI intentionally exposes 1/2.
Extra verification and schedule measurements bring both conditions to three
repetitions. All extra warmups, failures and interrupted records remain evidence.

The queued short-witness screen can independently trigger the 15% prefill gate,
even when the mixed worklist does not. Run those bounded cells after the initial
follow-ups:

```sh
.venv/bin/python -m scripts.run_prefill_benchmarks \
  data/03_throughput/q8-benchmark-NEW
```

`prefix_cache.py` copies exact forward-prefix state through the pinned runtime's
ownership primitive. It leaves a nonempty prompt suffix, keeps independent token
histories and samplers, and checks a fingerprint of every master-state leaf after
normal or interrupted cleanup. No cache rollback or response reuse occurs.
Elapsed time includes prefix construction, cloning and checks. Before further
cached comparisons, the frozen fixture must match its uncached reference by ID,
token IDs and finish status. Divergences are retained and prevent cache promotion;
this control does not imply universal bitwise equivalence or label divergence
as corruption. The current run observed divergence for both models, so prefix
reuse remains off in the provisional choices.

The prefill driver preserves a separate source snapshot, including the cache
implementation, before dispatch and accounts for prior worker time against the
one-hour-per-model ceiling. It refuses to overwrite a previous execution plan.
`check_benchmark_benigns.py` can score already-retained first-repeat benign replies
on 20 invented graph cases through the native verifier, without new inference.
This is fixture correctness only, not a cohort acquisition command.

Regenerate the detailed report without model calls or program execution:

```sh
.venv/bin/python -m scripts.report_inference_benchmarks \
  data/03_throughput/q8-benchmark-NEW
```

These runs establish systems behavior on fixtures, not accepted cohort yield.
Source-owned benign acquisition, witness-oracle/audit policy, scaffold yield,
fresh construction confirmation and the later BF16 yield comparison still belong
to their consuming stages. No witness or attack is approved by these benchmarks.

One independent pre-run subagent review found and prompted fixes for consecutive
user turns rejected by Ministral, missing BOS in GGUF tokenization, and a cleanup
race when an overlap peer exits during signalling. Regression checks cover the
two local chat templates, prompt-ID parity, idempotent stopping, cleanup of every
peer and failure of the second process launch. Review finished before benchmarks.

Checks: `python -m unittest tests.test_inference_benchmark
tests.test_benchmark_processes tests.test_verification_batch tests.test_verification
tests.test_inspect_runner tests.test_local_generation` in the execution venv.
