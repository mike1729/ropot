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


def build_job(descriptor, configs, wall_seconds=1170, *, panel_size=8, cache_limit_bytes=None):
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
    return B.build_plan(request, fixtures, windows, worker=__file__, wall_seconds=wall_seconds, cache_limit_bytes=cache_limit_bytes)


def build_long_job(descriptor, config, *, cache_limit_bytes=None):
    B = importlib.import_module("experiments.batch_throughput")
    config = B.G.snapshot(config)
    system = dict(role="system", content="Write a complete Python 3.12 program. Return only code, without explanation.")
    note = "Input values are separated by whitespace. Vertex labels start at one. Handle empty intermediate results. Keep all data local to this program.\n"
    draft = "import sys\n\ndef solve():\n" + ("    # " + note) * 120 + "    values = list(map(int, sys.stdin.buffer.read().split()))\n    print(len(values))\n\nsolve()\n"
    fixtures = [dict(id="short", seed=20261024, messages=[system,
        dict(role="user", content="Read n then n integers. Print their sum. Handle an empty array.")]),
        dict(id="long", seed=20261025, messages=[system,
        dict(role="user", content="Read n,m and m undirected edges. Print the number of connected components, including isolated vertices. Use an iterative traversal."),
        dict(role="assistant", content=draft), dict(role="user", content="The visible check reported an incorrect answer. Correct the draft. Read exactly m edges and count isolated vertices. Return a complete program.")])]
    def row(i, cap, sampling="greedy", after=None):
        return dict(fixture_id=fixtures[i]["id"], seed=fixtures[i]["seed"], cap=cap, sampling=sampling, after_row_index=after)
    reference = dict(config, batch_size=1, prefill_batch_size=1)
    live = dict(config, batch_size=2, prefill_batch_size=2)
    batch = config["batch_size"]
    windows = [dict(id=name, mode="qualification", config=c, rows=rows) for name, c, rows in (
        ("q0", reference, [row(0, 16)]), ("q1", reference, [row(1, 16)]),
        ("q2", config, [row(i % 2, 16) for i in range(batch)]),
        ("q3", live, [row(0, 2, "eos"), row(1, 3, "length"), row(0, 3, after=0)]),
        ("q4", reference, [row(0, 3)]),
        ("qlong", config, [row(1, 2048, "eos" if i % 2 == 0 else "length") for i in range(batch)]))]
    request = B.G.request_for(descriptor["source"], fixtures[0]["messages"], profile=descriptor,
        seed=20261024, temperature=.2, top_p=.95, max_tokens=2048, wall_time_seconds=240)
    return B.build_plan(request, fixtures, windows, worker=__file__, wall_seconds=240, cache_limit_bytes=cache_limit_bytes, schema_version=2)


def long_stress(job, ready_fixtures, completed_rows):
    window = job["windows"][-1]; batch = window["config"]["batch_size"]
    lengths = {f["id"]: len(f["prompt_ids"]) for f in ready_fixtures}
    prompts = [lengths[r["fixture_id"]] for r in window["rows"]]
    if not (len(completed_rows) == len(prompts) == batch and 2826 < min(prompts) <= max(prompts) <= 4096
        and all(len(r["tokens"]) == 2048 and all(type(t) is int and t >= 0 for t in r["tokens"])
        and r["finish_reason"] == ("stop" if i % 2 == 0 else "length") for i, r in enumerate(completed_rows))):
        raise ValueError("Long stress incomplete")
    return dict(batch_size=batch, prompt_tokens_min=min(prompts), prompt_tokens_max=max(prompts),
        prompt_token_ceiling=min(prompts), max_tokens=2048, raw_generation_tokens=batch * 2048)


def build_wave(rows, admission, *, prompt_ceiling_tokens=None):
    B = importlib.import_module("experiments.batch_throughput")
    admission = B.G.snapshot(admission); B.current_source_compatible(admission)
    B.G.require(type(rows) is list and 1 <= len(rows) <= admission["config"]["batch_size"], "Wave row count differs")
    retained, fixtures, requests = [], [], []
    for row in rows:
        B.G.require(type(row) is dict and set(row) == {"row_id", "request"}, "Raw wave row differs")
        request, raw = B.G.prepare_request(row["request"])
        retained.append(dict(row_id=row["row_id"], request_text=raw.decode("utf-8"), request_sha256=sha256(raw).hexdigest()))
        fixtures.append(dict(id=row["row_id"], messages=request["messages"], seed=request["seed"]))
        requests.append(request)
    window = dict(id="wave", mode="generate", config=admission["config"], rows=[dict(fixture_id=f["id"], seed=f["seed"],
        cap=2048, sampling="sample", after_row_index=None) for f in fixtures])
    job = dict(schema_version=3, request=requests[0], fixtures=fixtures, windows=[window], worker=str(Path(__file__).resolve()),
        wall_seconds=240, source_hashes=admission["source_hashes"], rows=retained, admission=admission,
        prompt_ceiling_tokens=admission["prompt_ceiling_tokens"] if prompt_ceiling_tokens is None else prompt_ceiling_tokens)
    if "cache_limit_bytes" in admission: job["cache_limit_bytes"] = admission["cache_limit_bytes"]
    B._shape(job)
    return B.G.snapshot(job)


def project_wave(job, stdout_raw, response_raw, decoded_raw, *, qualification_files):
    B = importlib.import_module("experiments.batch_throughput")
    B._shape(job, historical=True)
    errors, response, decoded, resources_closed = [], {}, None, False
    try:
        B.G.require(type(response_raw) is bytes and len(response_raw) <= B.STDOUT_LIMIT, "Wave response unavailable")
        response = B.G.snapshot(json.loads(response_raw)); B.G.require(type(response) is dict, "Wave response differs")
    except (ValueError, TypeError) as error: errors.append(str(error)); response = {}
    pid = response.get("pid")
    parsed = B.validate_events(stdout_raw if type(stdout_raw) is bytes else b"", job, pid if type(pid) is int and pid > 0 else None)
    try:
        parent = set("job_sha256 status pid returncode elapsed_seconds cleanup_seconds group_absent cleanup_error parent_error telemetry telemetry_complete work_seconds".split())
        B.G.require(set(response) == set(parsed) | parent and response["job_sha256"] == sha256(B._raw(job)).hexdigest()
            and type(pid) is int and pid > 0 and parsed["events"] and parsed["events"][0]["pid"] == pid
            and type(response["returncode"]) is int and response["group_absent"] is True and response["cleanup_error"] is None
            and B.G.P.digest({k: response[k] for k in parsed}) == B.G.P.digest(parsed), "Wave parent/closure proof incomplete")
        resources_closed = True
    except (ValueError, KeyError, TypeError) as error: errors.append(str(error))
    try:
        authority = B.qualification_checked(qualification_files)
        B.G.require(B.G.P.digest(authority) == B.G.P.digest(job["admission"]), "Wave qualification authority differs")
        B.G.require(resources_closed and response["returncode"] == 0 and response["status"] == "passed" and parsed["complete"]
            and all(response[k] is None for k in ("cleanup_error", "parent_error", "protocol_error")), "Wave shared proof incomplete")
        B.G.require(B._number(response["work_seconds"]) and 0 < response["work_seconds"] <= 240
            and B._number(response["elapsed_seconds"]) and response["elapsed_seconds"] >= response["work_seconds"]
            and B._number(response["cleanup_seconds"]) and response["cleanup_seconds"] <= response["elapsed_seconds"]
            and response["telemetry_complete"] is True, "Wave shared timing/resource proof differs")
        counters = {"t_seconds", "cpu_seconds", "rss_bytes", "available_bytes", "swap_bytes"}
        B.G.require(type(response["telemetry"]) is list and response["telemetry"] and all(type(s) is dict and counters <= set(s)
            and "error" not in s and all(B._number(s[k]) for k in counters) for s in response["telemetry"]), "Wave telemetry incomplete")
    except (ValueError, KeyError, TypeError) as error: errors.append(str(error))
    observations = {s["row_index"]: s for s in parsed["observations"]}
    try:
        B.G.require(type(decoded_raw) is bytes and len(decoded_raw) <= B.STDOUT_LIMIT
            and sha256(decoded_raw).hexdigest() == parsed["decoded_sha256"], "Wave decoded output unavailable or changed")
        lines = [B.G.snapshot(json.loads(line)) for line in decoded_raw.splitlines()]
        B.G.require(len(lines) == 1 and set(lines[0]) == {"job_sha256", "window", "rows"} and lines[0]["window"] == "wave"
            and lines[0]["job_sha256"] == sha256(B._raw(job)).hexdigest() and len(lines[0]["rows"]) == len(job["rows"]), "Wave decoded binding differs")
        for i, row in enumerate(lines[0]["rows"]):
            native = observations.get(i, {})
            B.G.require(type(row) is dict and set(row) == {"tokens", "text", "finish_reason"} and type(row["text"]) is str
                and B.G.P.digest(row["tokens"]) == B.G.P.digest(native.get("tokens"))
                and row["finish_reason"] == native.get("finish_reason") and row["finish_reason"] in ("stop", "length"), "Wave decoded row differs")
        decoded = lines[0]["rows"]
    except (ValueError, KeyError, TypeError, IndexError) as error: errors.append(str(error))
    shared = {k: response.get(k) for k in ("pid", "returncode", "status", "group_absent", "cleanup_error", "parent_error", "elapsed_seconds", "work_seconds", "cleanup_seconds")}
    raw_files = {"job.json": B._raw(job), "stdout.jsonl": stdout_raw, "response.json": response_raw, "decoded.jsonl": decoded_raw}
    shared.update(job_sha256=sha256(raw_files["job.json"]).hexdigest(), complete=not errors and parsed["complete"], resources_closed=resources_closed, errors=errors,
        evidence={name: dict(sha256=sha256(raw).hexdigest(), bytes=len(raw)) if type(raw) is bytes else None for name, raw in raw_files.items()})
    shared.update(protocol_error=parsed["protocol_error"], profile_guarded=any(e["event"] == "controls_checked" for e in parsed["events"]),
        ready=any(e["event"] == "ready" for e in parsed["events"]))
    shared["evidence"]["qualification"] = job["admission"]["inputs"]
    views = []
    for i, row in enumerate(job["rows"]):
        state = observations.get(i, {})
        views.append(dict(row_id=row["row_id"], request_sha256=row["request_sha256"], uid=state.get("uid"), tokens=state.get("tokens", []),
            text=decoded[i]["text"] if decoded is not None else None, finish_reason=state.get("finish_reason"),
            **{k: state.get(k) for k in ("admit_seconds", "first_seconds", "finish_seconds")}, eligible=shared["complete"]))
    return dict(shared=shared, rows=views)


def configure_cache(job, identity):
    if "cache_limit_bytes" not in job: return
    requested = job["cache_limit_bytes"]
    previous = mx.set_cache_limit(requested)
    observed = mx.set_cache_limit(requested)
    with Path("cache-control.json").open("x") as output:
        json.dump(dict(job_sha256=identity, requested_bytes=requested, previous_bytes=previous, observed_bytes=observed), output)
        output.flush()
        os.fsync(output.fileno())
    if type(observed) is not int or observed != requested:
        raise RuntimeError("Allocator cache limit not honored")


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


def run_window(model, tokenizer, fixtures, window, emit, *, window_seconds=180):
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
                if monotonic() - started > window_seconds:
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
    request, long, wave = job["request"], job["schema_version"] == 2, job["schema_version"] == 3
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
    configure_cache(job, identity)
    started = monotonic()
    fixtures = []
    for fixture in job["fixtures"]:
        options = dict(add_generation_prompt=True, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(fixture["messages"], tokenize=False, **options)
        ids = tokenizer.apply_chat_template(fixture["messages"], tokenize=True, **options)
        if not wave and not 0 < len(ids) <= (4096 if long else 1536):
            raise RuntimeError("Fixture prompt outside token bound")
        fixtures.append(dict(id=fixture["id"], prompt=prompt, prompt_ids=ids))
    if long and len(fixtures[1]["prompt_ids"]) <= 2826:
        raise RuntimeError("Long fixture does not cover the exposed prompt envelope")
    if not wave and len(fixtures[0]["prompt_ids"]) == len(fixtures[1 if long else 4]["prompt_ids"]):
        raise RuntimeError("Qualification requires unequal prompts")
    M.observe(sources, descriptor["expected_runtime"]["sources"])
    emit("ready", fixtures=fixtures, eos_token_ids=sorted(tokenizer.eos_token_ids), hash_seconds=hash_seconds + monotonic() - started,
         load_seconds=load_seconds, provenance=dict(precision_controls=proof, precision_postload=post, sources=sources))
    if wave:
        if not all(0 < len(f["prompt_ids"]) <= job["prompt_ceiling_tokens"] for f in fixtures):
            raise RuntimeError("Wave prompt exceeds qualified ceiling before admission")
    else:
        rng_ok, draws = rng_check()
        with Path("qualification-rng.json").open("x") as output:
            json.dump(dict(job_sha256=identity, temperature=.2, top_p=.95, draws=draws, passed=rng_ok), output)
            output.flush()
            os.fsync(output.fileno())
    results = {}
    with Path("decoded.jsonl").open("x") as decoded:
        for window in job["windows"]:
            results[window["id"]] = run_window(model, tokenizer, {f["id"]: f for f in fixtures}, window, emit,
                **({"window_seconds": 240} if long or wave else {}))
            decoded.write(json.dumps(dict(job_sha256=identity, window=window["id"], rows=results[window["id"]]), ensure_ascii=False) + "\n")
            decoded.flush()
            os.fsync(decoded.fileno())
            if not wave and window["id"] == ("qlong" if long else "q4"):
                checks = qualification_checks(results, rng_ok)
                emit("qualification", checks=checks, **({"stress": long_stress(job, fixtures, results["qlong"])} if long else {}))
                if not all(checks.values()):
                    raise RuntimeError("Numerical batching qualification failed")
    emit("done", **({"decoded_sha256": sha256(Path("decoded.jsonl").read_bytes()).hexdigest()} if wave else {}))


def main():
    if sys.argv[1] == "--run":
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("profiles", "source", "bits", "configs", "output", "lock"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--wall", type=float)
        parser.add_argument("--long-qualification", action="store_true")
        parser.add_argument("--panel-size", type=int, choices=(8, 32), default=8)
        parser.add_argument("--lease-fd", type=int)
        parser.add_argument("--cache-gib", type=int, choices=(0, 4, 8))
        args = parser.parse_args(sys.argv[2:])
        raw = Path(args.profiles).read_bytes()
        descriptor = json.loads(raw)["profiles"][args.bits][args.source]["descriptor"]
        B = importlib.import_module("experiments.batch_throughput")
        B.G.require(descriptor["source"] == args.source and str(descriptor["bits"]) == args.bits, "Profile label differs")
        if args.lease_fd is not None: B.watch_lease_parent()
        configs, cache = json.loads(args.configs), None if args.cache_gib is None else args.cache_gib * 1024**3
        if args.long_qualification:
            B.G.require(type(configs) is list and len(configs) == 1 and args.panel_size == 8
                and args.wall in (None, 240), "Long CLI bounds differ")
            job = build_long_job(descriptor, configs[0], cache_limit_bytes=cache)
        else: job = build_job(descriptor, configs, 1170 if args.wall is None else args.wall, panel_size=args.panel_size, cache_limit_bytes=cache)
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
