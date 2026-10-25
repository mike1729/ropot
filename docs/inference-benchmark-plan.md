# Local inference benchmark plan

Plan the fastest reliable setup for later benign, witness and backdoor experiments.
Start on the current Apple Silicon machine; compare cloud inference only if the
measured local completion time is insufficient. The fixture systems portion is
implemented in [the benchmark tools](inference-benchmark-implementation.md).
Scientific yield work remains gated on the Stage 04–05 contracts below.
The user authorized one combined implementation beyond the usual review size,
with one independent subagent review before benchmark execution; that review
finished and its findings were fixed before the first benchmark draw.
The fixture systems measurements and their remaining qualification limits are
recorded in [the results](inference-benchmark-results.md).

## Decisions and objective

Choose a runtime configuration per model, a way to schedule generation and
verification, and a small generation/repair recipe per workload. There may be
different winners for short witness replies and long code generation.

Use **verified useful artifacts per elapsed hour** as the main experimental
objective. Also report inference throughput and latency separately, so we can
tell whether time is spent loading, processing prompts, decoding, verifying,
waiting for dependent work, or reviewing witnesses. Report human review minutes
separately from unattended machine time and include review waits in total elapsed
time. Count failed, invalid, truncated and interrupted attempts as spent effort.
Interleave audit queues with comparable reviewer availability across recipes;
also report processing rates excluding external review waits, so review scheduling
does not masquerade as a scaffold improvement.

Report accepted witnesses/hour, verified attacks/hour, source-owned benign/attack
pairs/hour, and fully matched usable tasks/hour separately. An accepted witness
is not an attack; implantation into supplied code measures conditional yield,
not the complete source-owned acquisition rate. Check complete monitor-example
fit within 2,048 tokens before counting a task as usable.

## Starting evidence

Historical evidence includes Git tag `failed/noc-naukowcow` (commit `581cddd`)
and its ignored `data/03_throughput` and `data/03_reviews` artifacts. The tag records excessive
JSON serialization and repeated full-history validation as the runner's failure;
its final 50-task joint-yield and Q8/BF16 comparison remained incomplete. The
[operational freeze](../data/03_reviews/wave-memo-operational-freeze-20261025.json)
binds the artifacts to that branch. Do not resume the failed campaign.

- Actor: Qwen3.5-9B MLX 8-bit, thinking disabled. Synthetic: Ministral-3-8B
  MLX 8-bit, with official GGUF Q8_0 as a runtime comparison. Keep model revisions
  fixed. The current runtime pins `mlx-lm==0.32.2` and `mlx==0.32.4`, with
  [local sampling patches](../../honeypot/configs/runtime/mlx-request-seeds.json).
- [Cold actor controls](../data/03_generation_checks/actor-controls-1pyga9fv/summary.json)
  took 7.70–7.95 seconds for 64-token replies. The
  [warm serial smoke](../data/03_generation_checks/warm-actor-y0nrtmd7/serial/outcome.json)
  took 4.86 seconds for five tiny requests, including a 1.19-second model load;
  its 22–23-token replies took about 0.83–0.90 seconds each. Different request
  shapes prevent interpreting these figures as a controlled speedup.
- The [synthetic runtime smoke](../data/03_generation_checks/synthetic-comparison-himvyqb1/summary.json)
  verified four toy outputs from each runtime, but compared cold MLX requests
  with a warm GGUF server. Repeat with matched lifecycle and realistic lengths.
- Historical [batch qualifications](../data/03_campaign/qualifications/) include
  two-sequence long-output controls and hardware records for an Apple M5 Pro with
  64 GiB RAM. Their source/model fingerprints differ from the current adapter;
  use them to identify risks and avoid rediscovering mechanics, not to certify it.
- Historical scaffold changes produced large yield differences: on the same 30
  supplied programs and eight-reply ceiling,
  [trial 5](../../honeypot/runs/attack-trial-5/final-report.md) obtained 13 joint
  verified tasks, while [trial 6](../../honeypot/runs/attack-trial-6/final-report.md)
  obtained zero and recorded 386 format failures in 474 replies. Several method
  components changed; this does not isolate formatting causally.

The retained batch-2 stress windows used two roughly 7.4–7.7k-token prompts and
2,048 forced output tokens per sequence. They already suggest batching is worth
prioritizing:

| Historical 8-bit configuration | Aggregate native decode tokens/s | Wall time for 4,096 outputs | MLX peak GB |
| --- | ---: | ---: | ---: |
| [Actor](../data/03_campaign/qualifications/42b751ae747d028a0eed933c1a9f051d5533925b7d2247197c0a98d267249bb9/response.json), prefill 256 | 57.7–58.1 | 81.7–81.9 s | 10.9 |
| [Synthetic](../data/03_campaign/qualifications/3c160d1a13f093d31b1dfd677fad74a09c838b33aedee476a778cb4b4fb4bd53/response.json), prefill 2,048 | 48.7–48.8 | 94.6–95.3 s | 12.0 |

These are ranges from two qualification runs, not a production speed estimate.
The actor's long greedy reply differed between singleton and batch-2 at token 14,
while peer-change/replay and reinsertion checks agreed. Preserve that finding:
do not require blanket bitwise equality or assume yield equivalence. The campaign's
admitted qualifications used batch 2, but broader ignored benchmarks went to 32:

- [Short-fixture measurements](../data/03_throughput/cache-20261024/summary.json)
  reached about 200 overall tokens/s for actor batch 32 / prompt concurrency 4 /
  prefill 256, and 193 for synthetic batch 32 / prompt concurrency 2 / prefill
  2,048. Prompts were roughly 47–633 tokens, with 128 output tokens per row.
  A 4-GiB **free allocator cache** limit reduced memory use; this is distinct from
  model prefix-state reuse or caching responses. These are not long-workload rates.
- The [actor 8-bit batch-4 gate](../data/03_throughput/actor-isolation-capacity-followup-20261025/actor-8-b4-gate.json)
  passed the stronger long-prompt peer-isolation checks. Several larger/paired
  [synthetic checks](../data/03_throughput/synthetic-isolation-capacity-followup-20261025/synthetic-pair-gate.json)
  exceeded a fixed 240-second work bound. A timeout alone establishes neither
  OOM nor corrupt outputs; the paired BF16 comparison's common batch-2 capacity
  does not impose that ceiling on an 8-bit-only speed study.
- [Verifier fixtures](../data/03_throughput/cpu-20261024-v2/summary.json) processed
  eight programs in 10.36–10.97 seconds with coalesced concurrency 4, versus
  34.31–35.74 seconds under the old serial-eval path. The current execution
  contract differs; retain this as a scheduling lead, not a transferable rate.
- [Generation/verifier overlap](../data/03_throughput/overlap-actor8-20261024/response.json)
  reduced fixture makespan from 83.61–88.11 to 64.11–65.13 seconds. Running both
  models simultaneously gave only about a 6.4% makespan improvement in the
  [separate comparison](../data/03_throughput/model-parallel-20261024/response.json).

Start with Q8 throughout the initial benchmarks and pipeline development. Defer
the BF16 yield check until the full witness/backdoor generation pipeline,
including verification and audit, is working end to end. At that point, run at
least one bounded, paired Q8-versus-BF16 comparison of complete pipeline yield
on the same reviewed tasks, with matched upstream checkpoints, prompts, sampling
settings, attempt/token budgets and verification rules. Set the panel and budget
before running; report verified witness/backdoor yield and elapsed cost by source.
Retained throughput measurements and the failed campaign's incomplete precision
comparison do not satisfy this later yield check.
Lower-bit weights, cache quantization and thinking changes are quality
experiments, not interchangeable runtime settings.

## Workloads

First use fabricated/train fixtures with known answers and passing supplied code.
Freeze a small request corpus before comparing systems. Include these shapes:

| Workload | Input contents | Initial prompt/output token targets |
| --- | --- | --- |
| Benign | Statement, visible examples, generation instructions | 1,024 / 512 |
| Witness | Statement, program, proposal instructions | 1,024 / 128 |
| Implant | Statement, benign program, witness, edit instructions | 2,048 / 512 |
| Repair | Previous attempt and permitted feedback | 4,096 / 512 |
| Long-output stress | Long repair context and complete replacement program | 4,096 / 1,024 |

These are runtime-screen targets, not production limits or measured percentiles.
Count complete templated prompts with each model's tokenizer. Replace targets
with observed length bins once real request records exist. Do not pad scientific
prompts or truncate task information to hit a benchmark shape. Cross-model native
token rates are diagnostic; compare useful work/hour on the same tasks.

Use fixed-length synthetic generation only for the systems screen. Record any
forced EOS suppression there. Realistic replay and yield probes use natural
stopping, declared token limits and normal completion checks.

## 0. Bound scaffold overhead before spending inference

The failed runner could spend longer validating/reporting than generating.
One [closed-prefix snapshot](../data/03_reviews/wave-first-checkpoint-snapshot-baseline-result.json)
took 310.21 seconds wall / 300.29 CPU; the
[final optimization](../data/03_reviews/wave-first-checkpoint-snapshot-final-result.json)
still took 82.24 seconds with the same scientific output. The
[CPU audit](../data/03_reviews/wave-cpu-overhead-audit.txt) identifies repeated
journal serialization, hashing and retained-qualification replay; precise time
shares were not established for every function.

First benchmark the proposed thin runner using fabricated retained replies and
no model calls: prepare, append a result, load/reconcile, and produce a report.
Measure wall/CPU time, bytes read/written and validation invocation counts at
8, 32 and 128 requests, with realistic reply lengths. Check incremental append
cost and full-report scaling separately. An old multi-minute reporting pipeline
must not become a prerequisite for timing a new inference configuration.

Validate fixed run inputs once at the run boundary and each new reply once;
perform full historical replay only at explicit recovery/export boundaries where
required. Retain task/program identities, attempt accounting and public-feedback
checks. Avoid growing-history serialization or duplicated token events in each
dispatch. Preserve raw replies with small append-only metadata and summarize
after the finite run. This is a thin finite-run record, not a new campaign engine.

As an initial engineering budget, target under 5% harness overhead in a controlled
replay with representative request durations, and investigate superlinear growth
before acquisition. Record the threshold as a proposal, not an observed result.

## 1. Check the historical systems frontier on current workloads

Run one model at a time with inference alone. Record hardware, macOS, runtime and
patch hashes, weights, quantization, tokenizer/template, exact settings and power
state. Use a plugged-in machine, record other substantial workloads, and measure
memory pressure and swap growth rather than treating existing swap as failure.

The installed [MLX benchmark][mlx-bench] is an optional synthetic diagnostic, not
the matched-worklist benchmark: it generates only `batch_size` uniform random
prompts, suppresses EOS, and lacks the planned allocator, prompt-concurrency,
mixed-length and per-request sampler controls. Its rates do not compare equal
finite worklists. Use one small shared runtime adapter for matched-worklist
comparisons and realistic replay, retaining native MLX timing/statistics rather
than wrapping that CLI into a second harness. Include benign generation in the
measured workload mix.
For GGUF, use the pinned local build's help and [native tools][llama-bench]. Record
the actual build and flags; current upstream options are not proof of local support.

| Order | Comparison | Narrowing rule |
| --- | --- | --- |
| A | Cold finite run versus warm serial; time imports, load and first inference separately | Establish startup cost; amortize over actual finite experiment sizes |
| B | Check warm serial against historical actor batch 4 and synthetic batch 2, initially with a 4-GiB allocator cache limit | Confirm on current witness/implant shapes; do not repeat the old coarse grid |
| C | Best two batch sizes on all five workload shapes, including mixed lengths and a shrinking final batch | Keep configurations that survive realistic lengths and batch tails |
| D (optional) | Begin with historical actor prefill 256 / synthetic 2,048; vary chunk or prompt concurrency for a measured prefill bottleneck | Investigate when prefill exceeds roughly 15% of representative wall time |
| E (optional) | No prompt reuse versus exact shared-prefix state reuse on repeated proposals/repairs | Require material reusable prefill cost and supported, isolated state cloning |
| F | Warm Ministral MLX versus warm GGUF Q8_0, with matched messages, stopping and sampler policy | Compare latency, memory and later yield; do not require identical token streams |
| G (optional) | Batches 8/16/32 for short independent requests | Require enough queued work, memory headroom and evidence that decode throughput can improve |

Row F compares deployment packages: MLX 8-bit and GGUF Q8_0 use different
quantization artifacts. Bind both to the same upstream checkpoint where possible
and record differences; this comparison does not isolate the runtime alone.

Before Row E, measure reusable prefill's share of total wall time in B–C. Below
roughly 15%, defer cloning; otherwise estimate its maximum possible saving before
building it and require the gain to clear the practical threshold below. Include
cache construction/cloning cost. Larger batches in G have a separate queue/decode
gate: a low prefill share does not imply that batching cannot help.

Use one untimed warmup per configuration and three measured blocks; interleave
baseline and candidates in randomized order to expose thermal/background drift.
For the final two configurations, repeat five blocks. Compare equal total request
counts, including tails, rather than giving the larger batch more useful work.
Retain raw timings and ranges; do not claim a stable p95 from a handful of runs.

As a screening heuristic, pursue gains above roughly 10%; within observed noise,
prefer the simpler configuration. This is a practical threshold, not a statistical
claim. Start with a one-hour systems budget per model, calibrate after the first
block, and drop optional rows before expanding the budget. This is a ceiling,
not a target: use historical results to reduce new work. Log unfinished cells.

The later shared adapter must set and record both `completion_batch_size` and
`prefill_batch_size`, with prefill concurrency no greater than the requested
decode capacity. The pinned `BatchGenerator` defaults prefill concurrency to 8
and raises decode capacity to at least that value. This can exceed an intended
2/4 cap on a longer queued worklist; it does not create extra active requests
when only 2/4 were supplied. Record actual activity as well as configured limits.

Use `BatchGenerator.insert(..., samplers=[...])` with one patched
`make_sampler(..., seed=request_seed)` per request. The pinned `batch_generate`
wrapper does not forward request samplers. Use the same request-owned RNG policy
in both new serial and batch arms; the existing Batch 4 adapter uses global
reseeding and remains a separately identified baseline. Repeat requests with
changed neighbours, order and early EOS; verify
seed ownership, response association and clean state after cancellation. Fixed
seeds do not guarantee bit-identical results across batch shapes or backends.
Qualification checks precede accepting a faster configuration.

Keep response caching off: every requested draw must perform fresh sampling.
Prefix-state reuse can avoid repeated prefill without replaying an old response.
For Qwen's hybrid cache, check the exact runtime's support and clone all model
state; do not assume arbitrary prefix rollback is available. The pinned Qwen
implementation uses nontrimmable recurrent caches, while its ordinary speculative
decoder requires trimmable caches. Defer speculative decoding unless a supported
path is demonstrated; do not build that path for this benchmark.

## 2. Measure the real execution schedule

Replay the frozen message corpus as fresh draws with natural stopping. Include
the real tokenizer, templating, logging and extraction costs. Hold messages,
samplers, limits and seeds constant while comparing runtime configurations.
Choose settings separately for each model and request class when justified.

Compare generation followed by verification with generation overlapped with
verification of completed independent tasks. Prefer one Inspect evaluation per
bounded verification worklist, with `max_samples=1` versus `4` and the hard
`max_sandboxes=4` ceiling. Compare the same programs/cases and record both limits
and the actual `concurrent_inference` context. Keep cases serial, per-program
isolation, published-suite acceptance and the public-feedback boundary unchanged.
Allow only one verification evaluation at a time in this benchmark. Measure
complete queue drain time as well as inference slowdown.

This needed a reviewed Batch 6 prerequisite, now implemented: `verify_many`
submits one bounded multi-sample evaluation, and `derive/reuse` bind an explicit
sample ID plus exact task/program metadata. Verification and reuse both accept
the expected scheduling context, and the runner exposes and records the actual
sample limit. Errored/incomplete logs remain rejected. Execution identity changes
with these source/context changes; do not silently reuse old contracts.
Do not substitute four independent host `verify` calls: each performs its own
preflight/evaluation setup, and the current API gives no shared global four-job
limit. Raw multi-sample fixtures can establish scheduling mechanics before this
extension; their logs are not current `verification.reuse` evidence.

After measuring each model alone, compare runs grouped by model with both models
resident and alternating, then simultaneous generation only if memory allows.
Account for loading, switching, shared memory bandwidth and Docker VM memory.
Independent tasks may overlap; a repair still waits for its own verification and
an implant for any prerequisite witness approval. Count human-gated idle time.
Prioritize confirming generation/verifier overlap; simultaneous generation by
both models has weaker historical support and is optional.

Use at least three repetitions of the same finite worklist per schedule. Test
cancellation and timeout cleanup explicitly; a client deadline alone does not
show native inference has stopped. Record observed deadline overshoot and the
adapter's soft/hard deadline behavior. This experiment does not require a new
scheduler or a custom inference server.
Also inject one downstream-processing failure after replies have been saved and
confirm that the remaining replies can be scored without a redraw. Historical
[null-stdout recovery](../../honeypot/runs/attack-trial-4/recovery-null-stdout-1/amendment.md)
left 24 generated replies temporarily unscored; preserve this distinct failure mode.

## 3. Optimize the scaffold using verified yield

Stages 04 and 05 are still TODO in the authoritative overview. Before this phase,
define their acceptance and audit rules; runtime measurements can proceed earlier.
Provisional gates are: a witness satisfies task constraints and has an independently
checked answer; the fixed source-owned benign returns that answer, while the
attack passes every published case and returns a wrong answer on the same witness.
Freeze any novelty/repeatability checks before comparison. Do not substitute
model agreement for independent audit or count an inherited benign bug as an implant.

The proposed witness oracle executes the task's three qualified APPS references
on the exact proposed stdin in isolated sandboxes. All three must complete and
agree on the canonical answer; timeout/error/disagreement is not a valid oracle
answer. This extends beyond their original stored-case qualification and does not
prove the proposed input satisfies the problem's constraints. Freeze the oracle
and review policy in Stages 04–05 before using it on cohort tasks. That contract
may release the approved witness stdin, agreed answer and review status as a
small artifact for implantation/attack repair. References, raw oracle traces and
published hidden cases/diagnostics stay private. Stage 03 benign repairs continue
to receive only the verifier's public feedback.

Proposed review order: witness-first does cheap input checks, oracle execution
and independent human constraint/novelty review before implantation. The combined
recipe does cheap checks and oracle/suite/trigger screening before human review;
its attack remains provisional until that review passes. Record review minutes,
waits and rejection counts separately, since these different gates affect effort.

After fixture controls, use the existing eight-task development smoke and first
16 construction candidates per phase for tuning, as the Stage 03 plan specifies.
Run the eight-task smoke first; expand to construction only if mechanics and yield
warrant it, with the total worklist, machine-time and human-review budgets set
before expansion. The initial probe ceilings below cover only that smoke.
Keep the other 42 development IDs and protected/deferred tasks out of this work.
Historical supplied-program trials do not authorize transferring their artifacts
into the current cohort or treating shared code as source-owned benign data.
Restrict supplied-reference conditional-implant comparisons to train/nonstudy
fixtures. On development/construction tasks, first acquire verified actor and
synthetic benigns independently in Stage 03; each source's witness/attack recipes
then consume its own benign. Record a missing source-owned benign's acquisition
status and cost; never substitute an APPS reference. On each task/source, hold
the benign fixed across recipe arms to isolate conditional scaffold effects;
report its acquisition cost and full source-owned yield separately. Track exposure
across stages before calling any confirmation block fresh.

Compare at most two recipes at a time. A useful first contrast is a combined
program-and-witness proposal versus witness approval followed by implantation.
Use a simple, explicitly frozen output format and extractor for each recipe;
retain malformed replies as failures. Then vary one likely bottleneck: full code
versus edits, concise output instructions, temperature, or fresh draw versus repair.
Avoid a Cartesian sweep of every prompt and sampling parameter.
Inspect rendered prompts for unresolved placeholders. The sibling
[failure analysis](../../honeypot/.archive/pre-combined-2026-10-20/runs/attack-yield-failure-analysis-1/summary.json)
found equivalent edits, globally wrong programs, invalid proposed inputs and
literal template echoes. Check witness constraints before costly oracle calls;
bind the approved witness and answer into implantation/repair context. Report
unchanged/equivalent edits, no trigger, witness mismatch and repeated `<unk>`
generation separately from format failures. Pending downstream work is not a
scientific rejection.

Proposed first probe: four total replies and 4,096 newly generated tokens per
task/model/recipe, with per-reply caps of 512 for witness-only and 1,536 for code.
Use a 180-second request deadline and report its actual enforcement. These are
initial ceilings to review against observed lengths, not established acquisition
budgets. Every invalid proposal, repair and failed dispatch counts. Freeze the
same overall ceilings for both recipes; stop at success or exhausted budget.
On eight tasks, two models and two recipes, this caps the probe at 128 replies
and 131,072 output tokens, excluding separately recorded fixture/system checks
and benign acquisition. Keep recipe histories independent. Sharing a fixed
source-owned benign across arms estimates implantation yield conditional on that
benign; the full pipeline report includes its acquisition effort and failures.
For temperature, start with 0.2 and try 0.7 only as a separate paired comparison;
hold top-p at 0.95 and thinking off initially. Preserve native sampler ordering.

For each source and difficulty, report stage counts: dispatched, completed,
parseable, approved witness, executable implant, published-suite pass,
witness-confirmed attack, and final-fit usable pair. Include failure reasons,
reply/token effort, verification time, review effort and unique-task counts.
Keep actual stage order visible for each recipe rather than imposing one funnel.
An eight-task smoke can reject a broken recipe; close yield differences remain
uncertain. Repeated draws on one task are not independent task observations.

## 4. Confirm and forecast

Freeze finalists before construction candidates 17–32 per phase. Confirm without
tuning on that block; changes require a fresh permitted confirmation prefix.
Compare at most two finalists on the same tasks with cumulative attempt records.
Keep task-level paired outcomes and uncertainty, and retain the baseline when
the available evidence does not resolve a useful difference.

Choose the fastest confirmed setup subject to useful-yield and audit requirements.
Keep per-stage configurations when batching helps bulk proposals but hurts a
short dependent repair chain. A speed win must include failed work and queue drain.

Forecast the next experiment from observed attempts, prompt/output lengths,
verification time and useful yield, reporting a range rather than extrapolating
toy tokens/second. Report remaining task-capacity constraints separately. Before
launching, set the desired number of artifacts and acceptable completion window;
if the best local forecast exceeds it, plan a small matched NVIDIA comparison.
Do not choose cloud capacity from raw tokens/second alone.

## Evidence and implementation order

Use a small request manifest plus per-attempt records and one comparison table.
Save exact inputs/settings/seeds, identities and raw replies; native token counts,
admission/first-token/finish times, finish reasons, run/load/cleanup timings,
native throughput, process/GPU peak memory and swap deltas; extraction/audit/
verification outcomes, and each task's cumulative effort. Define timing boundaries
and the denominator for every rate. Hash the corpus and configurations.

Preserve frozen task ownership and the exact published case suite. Only public
verifier feedback may enter Stage 03 benign repairs; later attack repairs may
also receive the explicitly released witness artifact described above. Hidden
verifier evidence stays private. Never shorten verification or drop failed
attempts to improve a throughput figure.

Implementation order: first the small systems recorder using public tools,
then realistic replay/scheduling comparisons, then yield experiments when the
consuming stage contracts exist. The user explicitly waived the 500-line batch
limit for this implementation and requested one pre-run subagent review.
No general scheduler, response cache or broad runtime framework is needed.
Benchmark reports should distinguish historical
evidence, new observations, provisional choices and unmeasured configurations.

Mapping to the [Stage 03 batches](stage-03-plan.md#implementation-batches): Batches
4–5 establish actor/synthetic baselines and a provisional synthetic runtime choice.
The separate benchmark increment after them supplies the matched warm MLX/GGUF
comparison, shared batching adapter and systems recorder. Batch 6 consumes that
runtime and covers the thin loop plus
reviewed verification/scheduling extensions; Batch 7 records selections, costs
and resume evidence; Batch 8 probes benign yield. Witness/attack yield and oracle
work belong to Stages 04–05 once their contracts exist. The later BF16 check waits
for that full pipeline, as specified above.

[mlx-bench]: https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/benchmark.py
[llama-bench]: https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md
