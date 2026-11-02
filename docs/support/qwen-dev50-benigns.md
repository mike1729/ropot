# Expanded Qwen3.8 development benign panel

The user authorized fresh Qwen3.8 benign generation for all 50 development
initial tasks after attack run 09. This exploratory allocation supersedes the
earlier 20 scaffold / 30 reserved-confirmation split. Frozen cohort files and
historical runs remain intact; these tasks must not later be treated as untouched
confirmation data. No protected calibration or evaluation tasks are loaded.

`scripts/qwen_dev_benigns.py` reuses the shared generation loop, shown-case verifier,
comment normalization and private benign audit path. The existing eight task
partitions are preserved exactly. Other tasks use the same deterministic partition
rules and refreshed exposure classification. Tasks with too few eligible cases
still receive generation attempts and are explicitly ineligible for attacks.
Private audit cases and results never enter generation or repair prompts.

The saved exposure script copies the historical B1 scanning methodology, extends
its task list to 50, and uses the current benign_acquisition module name. Its
file hashes and disclosure matches are retained. The refreshed scan finds 38
case-eligible tasks and 12 insufficient-case tasks before generation.

Thinking is disabled for initial generation and both possible repairs. Sampling
is temperature 0.2, top-p 0.95, with batching 4/1/256,
a 25,808-token prompt ceiling and development base seed 2,000,041. Maximum
allowances are 150 replies. Fresh development acquisitions now allow 2048 initial
output tokens and 4096 per repair: 512,000 reserved tokens for 50 tasks and two
repairs each. The first acquisition and retry01 used 8192 per reply and reserved
1,228,800 tokens. Their historical contracts remain intact. The frozen contract
records shown-case and audit execution ceilings. The guard permits 1024 MiB swap
growth and retains the existing critical-pressure and 44 GiB MLX stops.

This is Qwen-only development acquisition, without a synthetic or monitor run
or a claim of formal runtime qualification. The finite job refuses an existing
contract or acquisition directory; technical failures are retained without retry.
Shown-passing benigns receive private audits once, without audit-driven repairs
or replacement. Only case-eligible, audit-passing programs qualify for attacks.

The first acquisition stopped with Metal's resource-count limit after 32 complete
replies, four interrupted replies and 14 unstarted tasks. The explicitly authorized
retry uses `--retry-from` and a fresh output directory. Its worker reuses only
completed initial requests whose full request objects match exactly, retaining
source hashes and original row order. Interrupted and unstarted requests run in
fresh Metal processes with at most four requests each; repairs use the same limit.
Sampling, prompts, seeds, token limits, checks and resource guards are unchanged.
Each process retains batch 4/1/256. Reused tokens are identified separately in the
native outcome; total selected-row usage includes their original tokens. A second
technical failure stops the acquisition without another automatic retry.

The user subsequently lowered the limits after the slow groups: other replies
finished in 18/42/63 seconds while one exhausted 8192 tokens in 17m08s. The original
32 complete replies had a 429-token median; 27 fit within 2048 tokens. Thinking off
still allowed lengthy reasoning inside Python comments. The limit change is an
operational choice, not an established quality optimum.

The lower-cap continuation preserves all 50 initial rows from retry01, including its two length stops,
under their original 8192-token allowance. Initial shown-case checks had passed
44 tasks; the six repair requests were interrupted with no completed repair reply.
They receive the one remaining repair round under the 4096 cap; the interrupted
repair round still consumes its allowance. No completed initial answer is
regenerated. This continuation reserves 8192 initial tokens plus one 4096 repair
per task, totaling 614,400 tokens. Shown-case checks are repeated to reconstruct
selection; private audits still occur only after final shown-case selection and
never influence repairs. New fresh acquisitions use 2048/4096; the shared formal
benign-acquisition qualification recipe is unchanged.

Retry02 failed before any inference dispatch because the adapter expected an
inputs.json file that the earlier retry wrapper did not write. The corrected
adapter also accepts the saved round specification as request provenance; retry03
is the actual lower-cap continuation. The failed preparation is retained.
