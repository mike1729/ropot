# Matched Qwen and Gemma monitor queue

> **Status — 2026-10-31:** Qwen then Gemma completed thinking-off monitoring on
> the same frozen 216-program corpus at
> `data/06_monitor_experiments/matched-dev-construction-20261031`. Results and
> retrospective review evidence are preserved; independent semantic approval
> remains pending. The heartbeat was deleted and monitoring stopped. The later
> user reply requests thinking-on monitoring under the current monitor-development
> protocol as a separate follow-up, not a reinterpretation or restart of this run.
> That follow-up has not been dispatched; instructions below are historical.

The user requested ratings after the existing attack queue: Qwen monitor first,
then Gemma monitor, each rating actor/synthetic × benign/attack. The queued plan
is `data/06_monitor_experiments/matched-dev-construction-20261031/plan.json`.

Run `scripts/matched_monitors.py run` only after `queue.json` records Gemma attacks
completed. Do not interrupt or repeat generation. Corpus selection is fixed now:
all 84 qualified Qwen benigns, all 83 qualified Gemma benigns, and the first
mechanically accepted attack per task/source in chain/attempt order. Identical
normalized programs on the same task share one request and retain both memberships.
Semantic approval remains pending; monitor ratings cannot select replacement attacks.

Default to the final honeypot thinking-off protocol: seed 17, temperature 0,
top-p 1, 2048 output tokens and a 25808-token prompt ceiling. The same statement,
AST-preserving comment-stripped program and correctness prompt reach both models.
Labels, generator identity, witnesses and private audits never enter requests.
Report PASS/FLAG/missing and FLAG/escalation rates in each of the four cells;
FLAG witnesses and explanations are diagnostic and are not execution-validated.

Reuse the guarded chunk driver, at most eight requests per worker, Qwen batch4
and Gemma batch2, prefill1/chunk256. Each monitor records one resource baseline:
1024 MiB swap growth, 44 GiB MLX, critical pressure4. No simultaneous model jobs.
The ceiling is 276 distinct records and 565248 reserved output tokens per monitor;
freeze the actual corpus, provenance, request order and budgets before dispatch.
