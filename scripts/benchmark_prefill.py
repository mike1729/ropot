"""Gated prefill and exact-prefix experiments on the frozen short-witness corpus."""

import argparse
import fcntl
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter

from experiments.inference_benchmark import replay, save, window
from experiments.run_inference_benchmarks import MODELS, PROFILES


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    root, previous = Path(args.output), Path(args.previous)
    root.mkdir(parents=True, exist_ok=False)
    old = json.loads((previous/"summary.json").read_bytes())
    share = sum(r["native"]["prompt_time"] for r in old)/sum(r["elapsed_seconds"] for r in old)
    if share < .15:
        raise ValueError("Prefill experiment not justified by the measured worklist")
    prior = json.loads((previous/"qualification.json").read_bytes())
    if not all(c[k] for c in prior["checks"] for k in ("reordered_equal", "unchanged_neighbours_equal",
                                                     "soft_timeout_observed", "recovery_equal")):
        raise ValueError("Prior short-worklist setting did not qualify")
    work = json.loads((previous/"corpus.json").read_bytes())
    base = prior["winner"]
    step = 1024 if args.profile == "actor" else 256
    configs = [base, {**base, "prefill_step_size": step},
               {**base, "prefill_batch_size": min(8, base["completion_batch_size"])},
               {**base, "prefill_batch_size": min(16, base["completion_batch_size"])}]
    configs = [json.loads(s) for s in dict.fromkeys(json.dumps(c, sort_keys=True) for c in configs)]
    save(root/"plan.json", dict(previous=str(previous.resolve()), previous_prefill_share=share,
        corpus_sha256=sha256((previous/"corpus.json").read_bytes()).hexdigest(),
        configs=configs, initial_repeats=3, finalist_extra_repeats=2, prefix_paired_repeats=5,
        source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), Path("experiments/inference_benchmark.py"), Path("experiments/prefix_cache.py"))}))
    save(root/"corpus.json", work)
    name, manifest, _, _ = PROFILES[args.profile]
    path = MODELS/"mlx-community"/name
    if version("mlx") != "0.32.4" or version("mlx-lm") != "0.32.2":
        raise RuntimeError("Require pinned runtime")
    (root/"model-manifest.json").write_bytes((MODELS/"metadata"/manifest).read_bytes())
    mx.set_cache_limit(4*1024**3)
    started = perf_counter()
    model, tokenizer = load(str(path), lazy=False, tokenizer_config={"local_files_only": True},
                            trust_remote_code=False)
    source_paths = [Path(importlib.import_module(n).__file__) for n in
        ("mlx_lm.models.cache", "mlx_lm.generate", "mlx_lm.sample_utils", type(model).__module__)]
    save(root/"runtime.json", dict(load_seconds=perf_counter()-started, device=mx.device_info(),
        mlx=version("mlx"), mlx_lm=version("mlx-lm"), numpy=version("numpy"),
        transformers=version("transformers"), allocator_cache_bytes=4*1024**3,
        source_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in source_paths},
        metadata_sha256={n:sha256((path/n).read_bytes()).hexdigest() for n in
                         ("config.json", "tokenizer.json", "chat_template.jinja")}))
    prefix = len(work[0]["prompt_ids"])-1
    if any(r["prompt_ids"] != work[0]["prompt_ids"] for r in work):
        raise ValueError("This exact-prefix fixture expects identical prompts with distinct request seeds")
    # A fixture-specific exact control, not a claim of bitwise equivalence for all workloads.
    def compare(left, right):
        a = {r["id"]:r for r in replay(root/left)[0]}
        b = {r["id"]:r for r in replay(root/right)[0]}
        if a.keys() != b.keys():
            raise RuntimeError("Cache reference control lost request IDs")
        return [dict(id=i, uncached_status=a[i]["status"], cached_status=b[i]["status"],
                     uncached_tokens=a[i]["token_ids"], cached_tokens=b[i]["token_ids"])
                for i in a if (a[i]["status"], a[i]["token_ids"]) != (b[i]["status"], b[i]["token_ids"])]
    window(model, tokenizer, work, base, root/"reference-uncached", 180)
    window(model, tokenizer, work, base, root/"reference-cached", 180, reuse_prefix_tokens=prefix)
    mismatch = compare("reference-uncached", "reference-cached")
    save(root/"cache-reference-check.json", dict(equal=not mismatch, mismatches=mismatch,
        scope="this frozen fixture only; divergence need not imply cache corruption"))
    results = []
    for index, cfg in enumerate(configs):
        window(model, tokenizer, [{**r, "max_tokens": 16} for r in work[:32]], cfg, root/f"warm-c{index}", 120)
    finalists = configs
    for repeat in range(5):
        order = list(configs if repeat < 3 else finalists)
        random.Random(310+repeat).shuffle(order)
        for cfg in order:
            name = f"r{repeat}-c{configs.index(cfg)}"
            result = window(model, tokenizer, work, cfg, root/name, 180)
            results.append(dict(name=name, config=cfg, repeat=repeat, **result))
            save(root/"summary.json", results)
            print(name, round(result["elapsed_seconds"], 3), flush=True)
        if repeat == 2:
            finalists = sorted(configs, key=lambda cfg: median(r["elapsed_seconds"] for r in results
                if r["config"] == cfg))[:2]
    def typical(cfg):
        return median(r["elapsed_seconds"] for r in results if r["config"] == cfg)
    fastest = min(typical(cfg) for cfg in finalists)
    best = min((c for c in finalists if typical(c) <= fastest*1.10),
               key=lambda c: (c["prefill_batch_size"], abs(c["prefill_step_size"]-base["prefill_step_size"])))
    if mismatch:
        save(root/"qualification.json", dict(prefill_winner=best, cache_reference_equal=False,
            cache_qualified=False, reason="Fixture cached/uncached divergence requires diagnosis before promotion"))
        return
    window(model, tokenizer, [{**r, "max_tokens": 16} for r in work[:32]], best,
           root/"warm-prefix", 180, reuse_prefix_tokens=prefix)
    paired = []
    baseline = None
    for repeat in range(5):
        order = [0, prefix]
        random.Random(720+repeat).shuffle(order)
        for reused in order:
            name = f"prefix-r{repeat}-n{reused}"
            result = window(model, tokenizer, work, best, root/name, 180, reuse_prefix_tokens=reused)
            paired.append(dict(name=name, reused_tokens=reused, repeat=repeat, **result))
            save(root/"prefix-summary.json", paired)
            if reused and baseline is None:
                baseline = {r["id"]: r["token_ids"] for r in replay(root/name)[0]}
            print(name, round(result["elapsed_seconds"], 3), flush=True)
        mismatch = compare(f"prefix-r{repeat}-n0", f"prefix-r{repeat}-n{prefix}")
        if mismatch:
            save(root/"cache-paired-divergence.json", dict(repeat=repeat, mismatches=mismatch))
            save(root/"qualification.json", dict(prefill_winner=best, cache_reference_equal=False,
                cache_qualified=False, reason="Paired fixture divergence; no cache promotion"))
            return
    def cached(name, rows, seconds=120):
        window(model, tokenizer, rows, best, root/name, seconds, reuse_prefix_tokens=prefix)
        return {r["id"]: r["token_ids"] for r in replay(root/name)[0]}
    reordered = cached("prefix-reordered", list(reversed(work)))
    targets = {r["id"] for r in work[:20]}
    changed = [{**r, "prompt_ids": r["prompt_ids"][:prefix]+tokenizer.encode("Reply with exactly OK.", add_special_tokens=False)}
               if r["id"] not in targets else r for r in work]
    neighbours = cached("prefix-neighbours", changed)
    cancelled = False
    try:
        cached("prefix-cancelled", work, 1e-12)
    except TimeoutError:
        cancelled = True
    recovered = cached("prefix-recovered", work)
    checks = dict(prefill_winner=best, reused_tokens=prefix, cache_reference_equal=True,
        reordered_equal=baseline == reordered,
        unchanged_neighbours_equal=all(baseline[i] == neighbours[i] for i in targets),
        soft_timeout_observed=cancelled, recovery_equal=baseline == recovered,
        cached_repeat_equal=all({r["id"]:r["token_ids"] for r in replay(root/w["name"])[0]} == baseline
            for w in paired if w["reused_tokens"]),
        cached_median_seconds=median(r["elapsed_seconds"] for r in paired if r["reused_tokens"]),
        uncached_median_seconds=median(r["elapsed_seconds"] for r in paired if not r["reused_tokens"]),
        scientific_yield="unmeasured; cache construction/cloning/checks included in elapsed time")
    save(root/"qualification.json", checks)
    if not all(checks[k] for k in ("reordered_equal", "unchanged_neighbours_equal", "soft_timeout_observed",
                                   "recovery_equal", "cached_repeat_equal")):
        raise RuntimeError("Prefix qualification failed; preserve faster uncached results independently")
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--output", required=True)
    with Path("data/03_throughput/current-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(parser.parse_args())
