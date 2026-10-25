"""Optional queued-witness screen and native isolation/recovery qualification.

Uses the reviewed shared window adapter. No cohort data or accepted yield.
Run one worker at a time with the pinned Metal Python and a new output path.
"""

import argparse
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter

from experiments.inference_benchmark import corpus, replay, save, window
from experiments.run_inference_benchmarks import MODELS, PROFILES


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    name, manifest, prefill, step = PROFILES[args.profile]
    path = MODELS / "mlx-community" / name
    if version("mlx") != "0.32.4" or version("mlx-lm") != "0.32.2":
        raise RuntimeError("Require pinned runtime")
    save(root/"plan.json", dict(profile=args.profile, queued_requests=40, initial_repeats=3,
        finalist_extra_repeats=2, batches=[8, 16, 32], prefill=prefill, step=step,
        qualification="marker isolation, changed neighbours/order, soft cancellation, fresh replay",
        source_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path("experiments/inference_benchmark.py"))}))
    (root/"model-manifest.json").write_bytes((MODELS/"metadata"/manifest).read_bytes())
    mx.set_cache_limit(4*1024**3)
    started = perf_counter()
    model, tokenizer = load(str(path), lazy=False, tokenizer_config={"local_files_only": True},
                            trust_remote_code=False)
    save(root/"runtime.json", dict(load_seconds=perf_counter()-started, device=mx.device_info(),
        mlx=version("mlx"), mlx_lm=version("mlx-lm"), allocator_cache_bytes=4*1024**3))

    def tokenize(rows):
        return [{**r, "prompt_ids": tokenizer.apply_chat_template(r["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)} for r in rows]

    work = tokenize([r for r in corpus(40) if r["kind"] == "witness"])
    save(root/"corpus.json", work)
    configs = [dict(completion_batch_size=b, prefill_batch_size=prefill, prefill_step_size=step)
               for b in (8, 16, 32)]
    for cfg in configs:
        warm = [{**r, "max_tokens": 16} for r in work[:cfg["completion_batch_size"]]]
        window(model, tokenizer, warm, cfg, root/f"warm-b{cfg['completion_batch_size']}", 120)
    results = []
    finalists = configs
    for repeat in range(5):
        order = list(configs if repeat < 3 else finalists)
        random.Random(98+repeat).shuffle(order)
        for cfg in order:
            name = f"r{repeat}-b{cfg['completion_batch_size']}"
            result = window(model, tokenizer, work, cfg, root/name, 240)
            results.append(dict(name=name, config=cfg, repeat=repeat, **result))
            save(root/"summary.json", results)
            print(name, round(result["elapsed_seconds"], 3), flush=True)
        if repeat == 2:
            finalists = sorted(configs, key=lambda cfg: median(r["elapsed_seconds"] for r in results
                if r["config"] == cfg))[:2]
    def typical(cfg):
        return median(r["elapsed_seconds"] for r in results if r["config"] == cfg)
    fastest = min(typical(cfg) for cfg in finalists)
    best = min((cfg for cfg in finalists if typical(cfg) <= 1.10*fastest),
               key=lambda cfg: cfg["completion_batch_size"])

    # All slots remain request-owned; one long survivor and one cap exercise tails.
    markers = tokenize([dict(id=i, kind="isolation", seed=301+i, temperature=.2, top_p=.95,
        max_tokens=1 if i == 39 else 128, messages=[{"role": "user", "content":
            "Print integers 1 through 20, separated by spaces. No other text." if i == 0 else
            f"Reply with exactly MARKER_{i:02d}. No other text."}]) for i in range(40)])

    changed = tokenize([{**r, "messages": [{"role": "user", "content": "Reply with exactly OK."}]}
                        if r["id"] >= 20 else r for r in markers])
    checks = []
    for cfg in [c for c in configs if c["completion_batch_size"] in (8, best["completion_batch_size"])]:
        prefix = f"isolation-b{cfg['completion_batch_size']}-"
        def draw(name, rows):
            directory = root/(prefix+name)
            window(model, tokenizer, rows, cfg, directory, 120)
            return {r["id"]: r["token_ids"] for r in replay(directory)[0]}
        baseline = draw("baseline", markers)
        reordered = draw("reordered", list(reversed(markers)))
        neighbours = draw("neighbours", changed)
        cancelled = False
        try:
            window(model, tokenizer, markers, cfg, root/(prefix+"cancelled"), 1e-12)
        except TimeoutError:
            cancelled = True
        recovered = draw("recovered", markers)
        checks.append(dict(config=cfg, reordered_equal=baseline == reordered,
            unchanged_neighbours_equal=all(baseline[i] == neighbours[i] for i in range(20)),
            soft_timeout_observed=cancelled, recovery_equal=baseline == recovered,
            baseline_stop_counts=json.loads((root/(prefix+"baseline")/"summary.json").read_text())["finish_counts"]))
    result = dict(winner=best, checks=checks,
        qualification_scope="short marker requests; not long-context or scientific yield qualification")
    save(root/"qualification.json", result)
    if not all(check[k] for check in checks for k in ("reordered_equal", "unchanged_neighbours_equal",
                                                     "soft_timeout_observed", "recovery_equal")):
        raise RuntimeError("Native isolation/recovery qualification failed; inspect retained tokens")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path("data/03_throughput/current-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
