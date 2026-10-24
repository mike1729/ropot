"""Resident, fresh-cache MLX batching on invented fixtures only."""

import argparse
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from time import monotonic

mx = BatchGenerator = make_sampler = None
ROOT = Path(__file__).resolve().parents[1]


def build_job(descriptor, configs, wall_seconds=1170, *, panel_size=8):
    B = importlib.import_module("experiments.batch_throughput")
    B.G.require(type(configs) is list and 1 <= len(configs) <= 7
                and len({json.dumps(c, sort_keys=True) for c in configs}) == len(configs), "Require distinct finite configs")
    B.G.require(type(panel_size) is int and panel_size in (8, 32)
                and (panel_size == 8 or len(configs) == 1), "Invalid panel choice")
    questions = [
        "Read n,m and m directed edges with nonnegative weights. Print shortest distances from vertex 1 using Dijkstra; use -1 for unreachable vertices.",
        "Read n and n integers. Print the length of their strictly increasing subsequence using an O(n log n) algorithm.",
        "Read n,m and m undirected edges. Print the size of each connected component in ascending order using an iterative traversal.",
        "Read n,q and an integer array. Each query gives inclusive one-based l,r; print the sum of that range using prefix sums.",
    ]
    note = "Input values are separated by whitespace. Vertex labels start at one. Handle empty intermediate results. Keep all data local to this program.\n"
    fixtures = [dict(id=f"fixture-{i}", seed=20261024 + i, messages=[
        dict(role="system", content="Write a complete Python 3.12 program. Return only code, without explanation."),
        dict(role="user", content=questions[i % 4] + ("\nImplementation notes:\n" + note * (14 + i) if i >= 4 else ""))]) for i in range(8)]
    if panel_size == 32:
        fixtures = [dict(fixtures[i % 8], id=f"fixture-{i}", seed=20261024 + i) for i in range(32)]
    def row(i, cap, sampling="sample", after=None):
        return dict(fixture_id=fixtures[i]["id"], seed=fixtures[i]["seed"], cap=cap, sampling=sampling, after_row_index=after)
    def window(name, mode, batch, rows):
        return dict(id=name, mode=mode, config=dict(batch_size=batch, prefill_batch_size=batch, prefill_step_size=2048), rows=rows)
    windows = [window("q0", "qualification", 1, [row(0, 16, "greedy")]),
        window("q1", "qualification", 1, [row(4, 16, "greedy")]),
        window("q2", "qualification", 2, [row(0, 16, "greedy"), row(4, 16, "greedy")]),
        window("q3", "qualification", 2, [row(0, 2, "eos"), row(4, 3, "length"), row(0, 3, "greedy", 0)]),
        window("q4", "qualification", 1, [row(0, 3, "greedy")]),
        window("warm", "warmup", 1, [row(0, 128)])]
    if panel_size == 32:
        batch = configs[0]["batch_size"]
        B.G.require(type(batch) is int and batch in (8, 16, 32), "Invalid expanded capacity")
        windows[2] = window("q2", "qualification", batch, [row(4 * (i % 2), 16, "greedy") for i in range(batch)])
        windows[2]["config"]["prefill_step_size"] = 256
    for index, config in enumerate(configs):
        for repeat in range(2):
            windows.append(dict(id=f"m{index}-r{repeat}", mode="measure", config=config, rows=[row(i, 128) for i in range(panel_size)]))
    request = B.G.request_for(descriptor["source"], fixtures[0]["messages"], profile=descriptor,
        seed=20261024, temperature=.2, top_p=.95, max_tokens=128, wall_time_seconds=wall_seconds)
    return B.build_plan(request, fixtures, windows, worker=__file__, wall_seconds=wall_seconds)


def sampler_for(row, eos, safe):
    mode = row["sampling"]
    if mode in ("sample", "greedy"):
        sampler = make_sampler(temp=.2 if mode == "sample" else 0, top_p=.95, seed=row["seed"])
        if mode == "sample" and getattr(sampler, "seed", None) != row["seed"]:
            raise RuntimeError("Sampler seed not honored")
        return lambda logits: sampler(logits)
    count = 0
    def forced(logits):
        nonlocal count
        count += 1
        token = eos if mode == "eos" and count >= row["cap"] else safe
        return mx.full(logits.shape[:-1], token, dtype=mx.uint32)
    return forced


def run_window(model, tokenizer, fixtures, window, emit):
    config, rows = window["config"], window["rows"]
    eos = sorted(tokenizer.eos_token_ids)
    safe = next(t for t in tokenizer.encode("0", add_special_tokens=False) if t not in eos)
    gen = BatchGenerator(model, completion_batch_size=config["batch_size"], prefill_batch_size=config["prefill_batch_size"],
                         prefill_step_size=config["prefill_step_size"], stop_tokens=[[t] for t in eos])
    results = [dict(tokens=[], text="", finish_reason=None) for _ in rows]
    admitted, finished, mapping = set(), set(), {}
    try:
        mx.synchronize(gen.stream)
        before = dict(active_before_bytes=mx.get_active_memory(), cache_before_bytes=mx.get_cache_memory())
        mx.reset_peak_memory()
        started = monotonic()
        emit("window_start", id=window["id"], started_seconds=started)
        with gen.stats() as stats:
            while len(finished) < len(rows):
                if monotonic() - started > 180:
                    raise TimeoutError("Cooperative window deadline")
                for index, row in enumerate(rows):
                    if index in admitted or row["after_row_index"] is not None and row["after_row_index"] not in finished:
                        continue
                    uid = gen.insert([fixtures[row["fixture_id"]]["prompt_ids"]], max_tokens=[row["cap"]], caches=None,
                        samplers=[sampler_for(row, eos[0], safe)])[0]
                    mapping[uid] = index
                    admitted.add(index)
                    emit("admit", window=window["id"], uid=uid, row_index=index, t_seconds=monotonic() - started)
                prompts, responses = gen.next()
                if not prompts and not responses:
                    raise RuntimeError("Generator ended before all rows finished")
                for response in responses:
                    index = mapping[response.uid]
                    result = results[index]
                    result["tokens"].append(response.token)
                    result["finish_reason"] = response.finish_reason
                    emit("token", window=window["id"], uid=response.uid, seq=len(result["tokens"]), token=response.token,
                         finish_reason=response.finish_reason, t_seconds=monotonic() - started)
                    if response.finish_reason is not None:
                        finished.add(index)
                        result["text"] = tokenizer.decode([t for t in result["tokens"] if t not in eos])
            mx.synchronize(gen.stream)
        elapsed = monotonic() - started
        memory = dict(before, active_bytes=mx.get_active_memory(), cache_bytes=mx.get_cache_memory(), peak_bytes=mx.get_peak_memory())
        native = dict(vars(stats), generation_tps=stats.generation_tps, prompt_tps=stats.prompt_tps, decode_tps=stats.decode_tps)
        emit("window_done", window=window["id"], elapsed=elapsed, stats=native, memory=memory)
        return results
    finally:
        try: mx.synchronize(gen.stream)
        finally: gen.close()


def rng_check():
    draws = {}
    for name, order in (("serial", [17] * 8 + [29] * 8), ("interleaved", [17, 29] * 8), ("reversed", [29, 17] * 8)):
        samplers = {seed: make_sampler(temp=.2, top_p=.95, seed=seed) for seed in (17, 29)}
        output = {seed: [] for seed in samplers}
        for seed in order:
            output[seed].append(samplers[seed](mx.zeros((1, 16))).item())
        draws[name] = output
    ok = (draws["serial"] == draws["interleaved"] == draws["reversed"] and draws["serial"][17] != draws["serial"][29]
          and all(len(set(v)) > 1 for v in draws["serial"].values()))
    return ok, draws


def qualification_checks(results, rng_ok):
    return dict(greedy_equal=all(row == results[f"q{i % 2}"][0] for i, row in enumerate(results["q2"])),
        eos_at_cap=len(results["q3"][0]["tokens"]) == 2 and results["q3"][0]["finish_reason"] == "stop",
        length_cap=len(results["q3"][1]["tokens"]) == 3 and results["q3"][1]["finish_reason"] == "length",
        reinsert_equal=results["q3"][2] == results["q4"][0], seed_independent=rng_ok)


def run_job(job, identity, emit):
    global mx, BatchGenerator, make_sampler
    path = ROOT / "src/data-pipeline/mlx_worker.py"
    helper = path.read_bytes()
    if sha256(helper).hexdigest() != job["source_hashes"].get(str(path)):
        raise RuntimeError("Worker helper changed before import")
    spec = importlib.util.spec_from_file_location("batch_worker_guard", path)
    M = importlib.util.module_from_spec(spec)
    exec(compile(helper, str(path), "exec"), M.__dict__)
    threading.Thread(target=M.watch_parent, daemon=True).start()
    started = monotonic()
    for source, digest in job["source_hashes"].items():
        if M.file_sha256(source) != digest:
            raise M.Control("guard", "Benchmark source changed")
    request = job["request"]
    guard = M.load_guard(request)
    descriptor = request["precision_profile"]["descriptor"]
    model_files = M.inventory(Path(request["model_path"]), guard, descriptor)
    hash_seconds = monotonic() - started
    started = monotonic()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    if M.origin(BatchGenerator) != M.origin(stream_generate):
        raise M.Control("import_path", "BatchGenerator origin differs")
    sources = {}
    tokenizer_config = dict(local_files_only=True)
    if request["fix_mistral_regex"]:
        tokenizer_config["fix_mistral_regex"] = True
    model, tokenizer, config, proof, post = M.controlled_load(request, guard, model_files, sources, emit,
        load, stream_generate, make_sampler, identity=identity, tokenizer_config=tokenizer_config)
    load_seconds = monotonic() - started
    started = monotonic()
    fixtures = []
    for fixture in job["fixtures"]:
        options = dict(add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=False, **options)
        ids = tokenizer.apply_chat_template(fixture["messages"], tokenize=True, **options)
        if not 0 < len(ids) <= 1536:
            raise RuntimeError("Fixture prompt outside token bound")
        fixtures.append(dict(id=fixture["id"], prompt=prompt, prompt_ids=ids))
    if len(fixtures[0]["prompt_ids"]) == len(fixtures[4]["prompt_ids"]):
        raise RuntimeError("Qualification requires unequal prompts")
    M.observe(sources, descriptor["expected_runtime"]["sources"])
    emit("ready", fixtures=fixtures, eos_token_ids=sorted(tokenizer.eos_token_ids), hash_seconds=hash_seconds + monotonic() - started,
         load_seconds=load_seconds, provenance=dict(precision_controls=proof, precision_postload=post, sources=sources))
    rng_ok, draws = rng_check()
    with Path("qualification-rng.json").open("x") as output:
        json.dump(dict(job_sha256=identity, temperature=.2, top_p=.95, draws=draws, passed=rng_ok), output)
        output.flush()
        os.fsync(output.fileno())
    results = {}
    with Path("decoded.jsonl").open("x") as decoded:
        for window in job["windows"]:
            results[window["id"]] = run_window(model, tokenizer, {f["id"]: f for f in fixtures}, window, emit)
            decoded.write(json.dumps(dict(job_sha256=identity, window=window["id"], rows=results[window["id"]]), ensure_ascii=False) + "\n")
            decoded.flush()
            os.fsync(decoded.fileno())
            if window["id"] == "q4":
                checks = qualification_checks(results, rng_ok)
                emit("qualification", checks=checks)
                if not all(checks.values()):
                    raise RuntimeError("Numerical batching qualification failed")
    emit("done")


def main():
    if sys.argv[1] == "--run":
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("profiles", "source", "bits", "configs", "output", "lock"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--wall", type=float, default=1170)
        parser.add_argument("--panel-size", type=int, choices=(8, 32), default=8)
        parser.add_argument("--lease-fd", type=int)
        args = parser.parse_args(sys.argv[2:])
        raw = Path(args.profiles).read_bytes()
        descriptor = json.loads(raw)["profiles"][args.bits][args.source]["descriptor"]
        B = importlib.import_module("experiments.batch_throughput")
        B.G.require(descriptor["source"] == args.source and str(descriptor["bits"]) == args.bits, "Profile label differs")
        if args.lease_fd is not None: B.watch_lease_parent()
        job = build_job(descriptor, json.loads(args.configs), args.wall, panel_size=args.panel_size)
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        B._write(output.with_name(output.name + "-inputs.json"), dict(profiles_sha256=sha256(raw).hexdigest(),
            profiles_path=str(Path(args.profiles).resolve()), job_sha256=sha256(B._raw(job)).hexdigest(), argv=sys.argv))
        result = B.run(job, output, args.lock, lease_fd=args.lease_fd)
        print(json.dumps({k: result[k] for k in ("status", "observed_generation_tokens", "elapsed_seconds", "protocol_error", "windows")}))
        return result["status"] != "passed"
    raw = Path(sys.argv[1]).read_bytes()
    identity, job = sha256(raw).hexdigest(), json.loads(raw)
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    def emit(event, **fields):
        print(json.dumps(dict(event=event, job_sha256=identity, **fields), allow_nan=False), file=protocol, flush=True)
    emit("start", pid=os.getpid())
    run_job(job, identity, emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
