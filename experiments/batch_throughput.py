"""Owned fixture-only resident-batch jobs; no acquisition journal mutation."""

import argparse
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
from time import monotonic, sleep

G = importlib.import_module("src.data-pipeline.local_generation")
L = importlib.import_module("src.data-pipeline.local_runner")
_exited, _kill_group = L._exited, L._kill_group
CHECKS = {"greedy_equal", "eos_at_cap", "length_cap", "reinsert_equal", "seed_independent"}
STDOUT_LIMIT, STDERR_LIMIT = 64 * 1024**2, 16 * 1024**2
QUALIFICATION_LIMITS = {"job.json": 4 * 1024**2, "stdout.jsonl": STDOUT_LIMIT, "response.json": STDOUT_LIMIT, "cache-control.json": 4096}
FIELDS = {"start": "pid", "controls_checked": "request_sha256 proof", "ready": "eos_token_ids fixtures provenance hash_seconds load_seconds",
    "window_start": "id started_seconds", "admit": "window uid row_index t_seconds", "token": "window uid seq token finish_reason t_seconds",
    "window_done": "window elapsed stats memory", "qualification": "checks", "done": ""}


def _raw(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else _raw(value))
        stream.flush()
        os.fsync(stream.fileno())


def _number(value):
    try: return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError: return False


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


@contextmanager
def lease(lock_path, lease_fd=None):
    G.require(lease_fd is None or type(lease_fd) is int and lease_fd >= 0, "Invalid inherited lease FD")
    descriptor = (os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
                  if lease_fd is None else os.dup(lease_fd))
    try:
        actual, expected = os.fstat(descriptor), os.stat(lock_path, follow_symlinks=False)
        G.require(stat.S_ISREG(actual.st_mode) and stat.S_ISREG(expected.st_mode)
                  and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), "Lease inode differs")
        stream = os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def watch_lease_parent():
    def watch():
        while os.read(0, 1): pass
        os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=watch, daemon=True).start()


def _shape(job, *, historical=False):
    G.require(type(job) is dict and set(job) - {"cache_limit_bytes"} == {"schema_version", "request", "fixtures", "windows", "worker", "wall_seconds", "source_hashes"}
              and type(job["schema_version"]) is int and job["schema_version"] in (1, 2), "Invalid job fields")
    long = job["schema_version"] == 2
    if "cache_limit_bytes" in job:
        G.require(type(job["cache_limit_bytes"]) is int and job["cache_limit_bytes"] in (0, 4 * 1024**3, 8 * 1024**3), "Invalid allocator cache limit")
    G.require(type(job["worker"]) is str and Path(job["worker"]).is_absolute()
              and _number(job["wall_seconds"]) and 0 < job["wall_seconds"] <= 1200, "Invalid launch bounds")
    sources = job["source_hashes"]
    G.require(type(sources) is dict and all(G.P._sha(h) for h in sources.values()), "Invalid source map")
    if historical:
        G.require(len(sources) == 6 and job["worker"] in sources and all(type(p) is str and os.path.isabs(p)
            and os.path.normpath(p) == p for p in sources), "Invalid historical source map")
        G.require(job["request"]["precision_profile"]["guard_sha256"] in sources.values(), "Qualification guard differs")
    else:
        paths = {Path(job["worker"]).resolve(), Path(__file__).resolve(), Path(G.__file__), G.WORKER, G.GUARD, Path(L.__file__)}
        G.require(set(sources) == {str(p) for p in paths}, "Invalid source map")
    fixtures, windows = job["fixtures"], job["windows"]
    G.require(type(fixtures) is list and (len(fixtures) == 2 if long else 1 <= len(fixtures) <= 8 or len(fixtures) == 32)
              and type(windows) is list and 1 <= len(windows) <= 56, "Invalid panel bounds")
    expanded = len(fixtures) == 32
    for fixture in fixtures:
        G.require(type(fixture) is dict and set(fixture) == {"id", "messages", "seed"} and type(fixture["id"]) is str and 0 < len(fixture["id"]) <= 64
                  and type(fixture["seed"]) is int and 0 <= fixture["seed"] < 2**64, "Invalid fixture")
        messages = fixture["messages"]
        G.require(type(messages) is list and messages and all(type(m) is dict and set(m) == {"role", "content"} and m["role"] in
                  ("system", "user", "assistant") and type(m["content"]) is str for m in messages)
                  and messages[-1]["role"] == "user", "Invalid fixture messages")
    ids = {f["id"] for f in fixtures}
    G.require(len(ids) == len(fixtures), "Duplicate fixture")
    counts, caps, order = [], 0, []
    for window in windows:
        G.require(type(window) is dict and set(window) == {"id", "mode", "config", "rows"} and type(window["id"]) is str and window["id"]
                  and window["mode"] in ("qualification", "warmup", "measure"), "Invalid window")
        config, rows = window["config"], window["rows"]
        G.require(type(config) is dict and set(config) == {"batch_size", "prefill_batch_size", "prefill_step_size"} and all(type(v) is int for v in config.values())
                  and config["batch_size"] in ((1, 2, 4, 8, 16, 32) if expanded else (1, 2, 4, 8))
                  and 1 <= config["prefill_batch_size"] <= config["batch_size"]
                  and config["prefill_step_size"] in (256, 1024, 2048), "Invalid batch configuration")
        G.require(type(rows) is list and 1 <= len(rows) <= (32 if expanded else 8), "Invalid window rows")
        for index, row in enumerate(rows):
            G.require(type(row) is dict and set(row) == {"fixture_id", "seed", "cap", "sampling", "after_row_index"} and row["fixture_id"] in ids
                      and type(row["seed"]) is int and 0 <= row["seed"] < 2**64 and type(row["cap"]) is int and 1 <= row["cap"] <= (2048 if long else 128)
                      and row["sampling"] in ("sample", "greedy", "eos", "length") and (window["mode"] == "qualification" or row["sampling"] == "sample")
                      and (row["after_row_index"] is None or type(row["after_row_index"]) is int and 0 <= row["after_row_index"] < index), "Invalid row")
            caps += row["cap"]
        counts.append(len(rows))
        order.append(("qualification", "warmup", "measure").index(window["mode"]))
    qualification_count = 8
    if expanded:
        measured = [w for w in windows if w["mode"] == "measure"]
        q2 = [w for w in windows if w["id"] == "q2" and w["mode"] == "qualification"]
        G.require(len(measured) == 2 and measured[0]["config"] == measured[1]["config"]
                  and all(len(w["rows"]) == 32 for w in measured), "Expanded measurement differs")
        batch = measured[0]["config"]["batch_size"]
        G.require(batch in (8, 16, 32) and len(q2) == 1 and len(q2[0]["rows"]) == batch
                  and q2[0]["config"] == dict(batch_size=batch, prefill_batch_size=batch, prefill_step_size=256), "Expanded qualification differs")
        qualification_count = batch + 6
    if long:
        _long_shape(job)
        qualification_count = 2 * windows[-1]["config"]["batch_size"] + 6
    G.require(len({w["id"] for w in windows}) == len(windows) and order == sorted(order) and sum(counts) <= 128 and caps <= (16555 if long else 16384)
              and sum(n for n, o in zip(counts, order) if o == 0) == qualification_count
              and sum(n for n, o in zip(counts, order) if o == 1) == (0 if long else 1), "Invalid job workload")


def _long_shape(job):
    fixtures, windows, request = job["fixtures"], job["windows"], job["request"]
    G.require([f["id"] for f in fixtures] == ["short", "long"] and [[m["role"] for m in f["messages"]] for f in fixtures]
        == [["system", "user"], ["system", "user", "assistant", "user"]], "Long fixture roles differ")
    G.require(job["wall_seconds"] == request["wall_time_seconds"] == 240 and request["max_tokens"] == 2048
        and request["messages"] == fixtures[0]["messages"] and request["temperature"] == .2 and request["top_p"] == .95, "Long request bounds differ")
    G.require([w["id"] for w in windows] == ["q0", "q1", "q2", "q3", "q4", "qlong"]
        and all(w["mode"] == "qualification" for w in windows), "Long qualification order differs")
    config = windows[-1]["config"]; batch = config["batch_size"]
    G.require(batch in (2, 4, 8), "Long capacity differs")
    reference = dict(config, batch_size=1, prefill_batch_size=1)
    live = dict(config, batch_size=2, prefill_batch_size=2)
    recipes = [[(0, 16, "greedy", None)], [(1, 16, "greedy", None)],
        [(i % 2, 16, "greedy", None) for i in range(batch)],
        [(0, 2, "eos", None), (1, 3, "length", None), (0, 3, "greedy", 0)], [(0, 3, "greedy", None)],
        [(1, 2048, "eos" if i % 2 == 0 else "length", None) for i in range(batch)]]
    for window, expected, recipe in zip(windows, (reference, reference, config, live, reference, config), recipes):
        rows = [dict(fixture_id=fixtures[i]["id"], seed=fixtures[i]["seed"], cap=cap, sampling=sampling, after_row_index=after)
                for i, cap, sampling, after in recipe]
        G.require(window["config"] == expected and window["rows"] == rows, "Long qualification recipe differs")


def build_plan(request, fixtures, windows, *, worker, wall_seconds=1200, cache_limit_bytes=None, schema_version=1):
    request, _ = G.prepare_request(request)
    G.require("precision_profile" in request, "Require precision profile")
    windows = G.snapshot(windows)
    G.require(type(windows) is list, "Invalid windows")
    for window in windows:
        G.require(type(window) is dict and type(window.get("rows")) is list, "Invalid window rows")
        for row in window["rows"]:
            G.require(type(row) is dict, "Invalid row")
            row.setdefault("after_row_index", None)
    paths = {Path(worker).resolve(), Path(__file__).resolve(), Path(G.__file__), G.WORKER, G.GUARD, Path(L.__file__)}
    job = dict(schema_version=schema_version, request=request, fixtures=G.snapshot(fixtures), windows=windows, worker=str(Path(worker).resolve()),
               wall_seconds=wall_seconds, source_hashes={str(p): sha256(p.read_bytes()).hexdigest() for p in paths})
    if cache_limit_bytes is not None: job["cache_limit_bytes"] = cache_limit_bytes
    _shape(job)
    return job


def validate_events(raw, job, pid=None):
    events, observations, windows, current = [], [], [], None
    error, proof, eos, ready, qualified, complete, index = None, None, set(), False, False, False, 0
    identity = sha256(_raw(job)).hexdigest()
    long, stress, ready_fixtures = job["schema_version"] == 2, None, None
    truncated, raw = len(raw) > STDOUT_LIMIT, raw[:STDOUT_LIMIT]
    try:
        for line in raw.splitlines(keepends=True):
            G.require(not complete, "Events after done")
            G.require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = G.snapshot(json.loads(line))
            kind = event["event"]
            fields = set(FIELDS[kind].split()) | ({"stress"} if long and kind == "qualification" else set())
            G.require(set(event) == {"event", "job_sha256"} | fields and event["job_sha256"] == identity, "Event binding differs")
            if not events:
                G.require(kind == "start" and type(event["pid"]) is int and event["pid"] > 0 and (pid is None or event["pid"] == pid), "Worker PID differs")
            elif kind == "controls_checked":
                G.require(proof is None and not ready and event["request_sha256"] == identity, "Controls order differs")
                G._proof(event["proof"], job["request"]["precision_profile"])
                proof = event["proof"]
            elif kind == "ready":
                G.require(proof is not None and not ready and all(_number(event[k]) for k in ("hash_seconds", "load_seconds")), "Ready order differs")
                descriptor, provenance = job["request"]["precision_profile"]["descriptor"], event["provenance"]
                G.require(set(provenance) == {"precision_controls", "precision_postload", "sources"} and G.P.digest(provenance["precision_controls"]) == G.P.digest(proof), "Ready controls differ")
                G._roles(provenance["precision_postload"], descriptor, True)
                G.P._sources(provenance["sources"])
                bindings = [*proof["critical_sources"].values(), *provenance["precision_postload"].values()]
                G.require(all(p not in descriptor["expected_runtime"]["sources"] or h == descriptor["expected_runtime"]["sources"][p] for p, h in provenance["sources"].items())
                          and all(provenance["sources"].get(v["path"]) == v["sha256"] for v in bindings), "Observed sources differ")
                G.require(type(event["eos_token_ids"]) is list and event["eos_token_ids"] and all(type(t) is int and t >= 0 for t in event["eos_token_ids"]), "Invalid EOS")
                G.require(type(event["fixtures"]) is list and len(event["fixtures"]) == len(job["fixtures"]), "Ready fixtures differ")
                for fixture, planned in zip(event["fixtures"], job["fixtures"]):
                    G.require(set(fixture) == {"id", "prompt", "prompt_ids"} and fixture["id"] == planned["id"] and type(fixture["prompt"]) is str
                              and type(fixture["prompt_ids"]) is list and 0 < len(fixture["prompt_ids"]) <= (4096 if long else 1536)
                              and all(type(t) is int and t >= 0 for t in fixture["prompt_ids"]), "Ready fixtures differ")
                eos, ready, ready_fixtures = set(event["eos_token_ids"]), True, event["fixtures"]
            elif kind == "window_start":
                G.require(ready and current is None and index < len(job["windows"]) and event["id"] == job["windows"][index]["id"]
                          and _number(event["started_seconds"]) and event["started_seconds"] > 0, "Window order differs")
                planned = job["windows"][index]
                G.require(planned["mode"] == "qualification" or qualified, "Qualification required")
                current = dict(plan=planned, states={}, rows={}, last=0)
            elif kind in ("admit", "token"):
                G.require(current is not None and event["window"] == current["plan"]["id"] and type(event["uid"]) is int and event["uid"] >= 0
                          and _number(event["t_seconds"]) and event["t_seconds"] >= current["last"], "Invalid UID or time")
                uid = event["uid"]
                if kind == "admit":
                    row = event["row_index"]
                    G.require(type(row) is int and 0 <= row < len(current["plan"]["rows"]) and row not in current["rows"] and uid not in current["states"], "Duplicate or unknown row")
                    after = current["plan"]["rows"][row]["after_row_index"]
                    G.require(after is None or after in current["rows"] and current["states"][current["rows"][after]]["finish_reason"] is not None, "Early reinsert")
                    if long and current["plan"]["id"] == "q3" and after is not None:
                        G.require(any(s["finish_reason"] is None for s in current["states"].values()), "Reinsert has no live peer")
                    state = dict(window=event["window"], uid=uid, row_index=row, tokens=[], finish_reason=None,
                                 admit_seconds=event["t_seconds"], first_seconds=None, finish_seconds=None)
                    current["states"][uid], current["rows"][row] = state, uid
                    observations.append(state)
                else:
                    G.require(uid in current["states"], "Unknown UID")
                    state = current["states"][uid]
                    cap = current["plan"]["rows"][state["row_index"]]["cap"]
                    G.require(state["finish_reason"] is None and type(event["seq"]) is int and event["seq"] == len(state["tokens"]) + 1 <= cap
                              and type(event["token"]) is int and event["token"] >= 0, "Invalid token sequence")
                    expected = "stop" if event["token"] in eos else "length" if event["seq"] == cap else None
                    G.require(event["finish_reason"] == expected, "Invalid terminal token")
                    state["tokens"].append(event["token"])
                    state["finish_reason"] = expected
                    state["first_seconds"] = event["t_seconds"] if state["first_seconds"] is None else state["first_seconds"]
                    if expected is not None: state["finish_seconds"] = event["t_seconds"]
                current["last"] = event["t_seconds"]
            elif kind == "window_done":
                G.require(current is not None and event["window"] == current["plan"]["id"] and len(current["rows"]) == len(current["plan"]["rows"])
                          and all(s["finish_reason"] is not None for s in current["states"].values()) and _number(event["elapsed"]) and event["elapsed"] > 0
                          and event["elapsed"] >= current["last"] and all(type(event[k]) is dict and event[k] and all(_number(v) for v in event[k].values()) for k in ("stats", "memory")), "Incomplete window")
                G.require({"active_bytes", "cache_bytes", "peak_bytes"} <= set(event["memory"]), "Missing memory counters")
                G.require(type(event["stats"].get("generation_tokens")) is int and event["stats"]["generation_tokens"]
                          == sum(len(s["tokens"]) for s in current["states"].values()), "Native token count differs")
                windows.append(event)
                current, index = None, index + 1
            elif kind == "qualification":
                G.require(ready and current is None and not qualified and index == sum(w["mode"] == "qualification" for w in job["windows"])
                          and type(event["checks"]) is dict and set(event["checks"]) == CHECKS and all(v is True for v in event["checks"].values()), "Qualification failed")
                if long:
                    rows = sorted((s for s in observations if s["window"] == "qlong"), key=lambda s: s["row_index"])
                    planned = job["windows"][-1]; batch = planned["config"]["batch_size"]
                    lengths = {f["id"]: len(f["prompt_ids"]) for f in ready_fixtures}
                    prompts = [lengths[r["fixture_id"]] for r in planned["rows"]]
                    G.require(len(rows) == batch and [r["row_index"] for r in rows] == list(range(batch))
                        and all(len(r["tokens"]) == 2048 and r["finish_reason"] == ("stop" if i % 2 == 0 else "length")
                        for i, r in enumerate(rows)) and 2826 < min(prompts) <= max(prompts) <= 4096, "Long stress incomplete")
                    derived = dict(batch_size=batch, prompt_tokens_min=min(prompts), prompt_tokens_max=max(prompts),
                        prompt_token_ceiling=min(prompts), max_tokens=2048, raw_generation_tokens=batch * 2048)
                    G.require(G.P.digest(event["stress"]) == G.P.digest(derived), "Long stress proof differs")
                    stress = derived
                qualified = True
            else:
                G.require(kind == "done" and ready and qualified and current is None and index == len(job["windows"]), "Incomplete job")
                complete = True
            events.append(event)
        G.require(complete, "Missing done")
    except (ValueError, TypeError, KeyError) as failure:
        error = str(failure)
    if truncated: error = "Output limit; " + (error or "bounded prefix retained")
    return dict(events=events, protocol_error=error, complete=complete and error is None, observations=observations, windows=windows,
                observed_generation_tokens=sum(len(s["tokens"]) for s in observations), finished_sequences=sum(s["finish_reason"] is not None for s in observations),
                **({"long_stress": stress} if long else {}))


def _sample(pid):
    try:
        import psutil
        cpu = psutil.Process(pid).cpu_times()
        return dict(cpu_seconds=cpu.user + cpu.system, rss_bytes=psutil.Process(pid).memory_info().rss,
                    available_bytes=psutil.virtual_memory().available, swap_bytes=psutil.swap_memory().used)
    except Exception as error:
        return dict(error=repr(error))


def _qualification_file(directory, name, limit):
    descriptor = os.open(Path(directory) / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        G.require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, "Invalid qualification file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
    finally:
        os.close(descriptor)
    G.require(len(raw) <= limit, "Qualification file limit")
    return raw


def qualification_checked(files):
    G.require(type(files) is dict and {"job.json", "stdout.jsonl", "response.json"} <= set(files) <= set(QUALIFICATION_LIMITS)
        and all(type(raw) is bytes and len(raw) <= QUALIFICATION_LIMITS[name] for name, raw in files.items()), "Invalid qualification files")
    try:
        job, response = (G.snapshot(json.loads(files[name])) for name in ("job.json", "response.json"))
        _shape(job, historical=True)
        G.require(job["schema_version"] == 2 and files["job.json"] == _raw(job), "Qualification job differs")
        G.require(("cache-control.json" in files) == ("cache_limit_bytes" in job), "Qualification cache files differ")
        request = job["request"]; descriptor = G._profile(request)
        G.require(set(request) == G.FIELDS | {"environment", "precision_profile"} and type(request["max_tokens"]) is int
            and type(request["seed"]) is int and 0 <= request["seed"] < 2**64
            and G.P.digest(request["environment"]) == G.P.digest(descriptor["expected_runtime"]["environment"]), "Qualification request differs")
        G.require(type(response) is dict and type(response["pid"]) is int and response["pid"] > 0
            and type(response["returncode"]) is int and response["returncode"] == 0 and response["status"] == "passed"
            and response["complete"] is True and response["group_absent"] is True
            and all(response[k] is None for k in ("cleanup_error", "parent_error", "protocol_error")), "Qualification parent proof differs")
        G.require(_number(response["work_seconds"]) and 0 < response["work_seconds"] <= 240
            and _number(response["elapsed_seconds"]) and response["elapsed_seconds"] >= response["work_seconds"]
            and _number(response["cleanup_seconds"]) and response["cleanup_seconds"] <= response["elapsed_seconds"], "Qualification timing differs")
        telemetry = response["telemetry"]; counters = {"t_seconds", "cpu_seconds", "rss_bytes", "available_bytes", "swap_bytes"}
        G.require(response["telemetry_complete"] is True and type(telemetry) is list and bool(telemetry)
            and all(type(sample) is dict and counters <= set(sample) and "error" not in sample
                    and all(_number(sample[k]) for k in counters) for sample in telemetry), "Qualification telemetry differs")
        parsed = validate_events(files["stdout.jsonl"], job, pid=response["pid"])
        parent = set("job_sha256 status pid returncode elapsed_seconds cleanup_seconds group_absent cleanup_error parent_error telemetry telemetry_complete work_seconds".split())
        G.require(set(response) == set(parsed) | parent and response["job_sha256"] == sha256(files["job.json"]).hexdigest()
            and parsed["complete"] and G.P.digest({k: response[k] for k in parsed}) == G.P.digest(parsed), "Qualification protocol differs")
        if "cache_limit_bytes" in job:
            proof = G.snapshot(json.loads(files["cache-control.json"]))
            G.require(proof["job_sha256"] == response["job_sha256"] and all(type(proof[k]) is int
                and proof[k] == job["cache_limit_bytes"] for k in ("requested_bytes", "observed_bytes")), "Qualification cache proof differs")
        stress = parsed["long_stress"]
        admission = dict(schema_version=1, request_controls={k: v for k, v in request.items() if k not in ("messages", "seed")},
            config=job["windows"][-1]["config"], source_hashes=job["source_hashes"], prompt_ceiling_tokens=stress["prompt_token_ceiling"],
            stress_prompt_tokens=dict(min=stress["prompt_tokens_min"], max=stress["prompt_tokens_max"]),
            inputs={name: dict(sha256=sha256(raw).hexdigest(), bytes=len(raw)) for name, raw in files.items()})
        if "cache_limit_bytes" in job: admission["cache_limit_bytes"] = job["cache_limit_bytes"]
        return G.snapshot(admission)
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid qualification metadata") from error


def read_qualification(directory, expected_response_sha256):
    response = _qualification_file(directory, "response.json", QUALIFICATION_LIMITS["response.json"])
    G.require(G.P._sha(expected_response_sha256) and sha256(response).hexdigest() == expected_response_sha256, "Qualification authority differs")
    files = {name: _qualification_file(directory, name, QUALIFICATION_LIMITS[name]) for name in ("job.json", "stdout.jsonl")}
    files["response.json"] = response
    if "cache_limit_bytes" in json.loads(files["job.json"]):
        files["cache-control.json"] = _qualification_file(directory, "cache-control.json", QUALIFICATION_LIMITS["cache-control.json"])
    return qualification_checked(files), files


def copy_qualification(files, directory, expected_response_sha256):
    G.require(type(files) is dict and type(files.get("response.json")) is bytes and G.P._sha(expected_response_sha256)
        and sha256(files.get("response.json", b"")).hexdigest() == expected_response_sha256, "Qualification authority differs")
    files = dict(files); admission = qualification_checked(files)
    directory = Path(directory); directory.mkdir()
    for name, raw in files.items(): _write(directory / name, raw)
    _sync(directory); _sync(directory.parent)
    return admission


def current_source_compatible(admission):
    paths = {Path(__file__).resolve(), Path(__file__).with_name("mlx_batch_throughput.py").resolve(), Path(G.__file__), G.WORKER, G.GUARD, Path(L.__file__)}
    G.require(set(admission["source_hashes"]) == {str(p) for p in paths}, "Qualified source paths differ")
    try:
        G.require(all(sha256(p.read_bytes()).hexdigest() == admission["source_hashes"][str(p)] for p in paths), "Qualified source changed")
    except OSError as error:
        raise ValueError("Qualified source changed") from error


def run(job, output, lock_path, *, lease_fd=None):
    job = G.snapshot(job)
    _shape(job)
    G.require("precision_profile" in job["request"] and G.validate({k: v for k, v in job["request"].items() if k != "environment"}) == job["request"], "Request launch differs")
    sources = {p: Path(p).read_bytes() for p in job["source_hashes"]}
    G.require({p: sha256(raw).hexdigest() for p, raw in sources.items()} == job["source_hashes"], "Source changed")
    with lease(lock_path, lease_fd) as ownership:
        output = Path(output).resolve()
        output.mkdir(exist_ok=False)
        _write(output / "job.json", job)
        for index, raw in enumerate(sources.values()): _write(output / f"source-{index}.py", raw)
        argv = [str(G.INTERPRETER), "-I", "-B", job["worker"], str(output / "job.json")]
        lease_stat = os.fstat(ownership.fileno())
        _write(output / "launch.json", dict(argv=argv, environment=job["request"]["environment"], lease_fd=ownership.fileno(),
            lease_dev=lease_stat.st_dev, lease_ino=lease_stat.st_ino, sources=list(sources)))
        _sync(output)
        _sync(output.parent)
        started, process, status, interruption, cleanup_error = monotonic(), None, None, None, None
        samples, low, swap, sampled = [], 0, None, -1
        with (output / "stdout.jsonl").open("xb") as stdout, (output / "stderr.txt").open("xb") as stderr, (output / "telemetry.jsonl").open("xb") as telemetry:
            try:
                process = subprocess.Popen(argv, cwd=output, env=job["request"]["environment"], stdin=subprocess.PIPE,
                    stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(ownership.fileno(),))
                while not _exited(process):
                    elapsed = monotonic() - started
                    if elapsed - sampled >= 1:
                        sample = dict(t_seconds=elapsed, **_sample(process.pid))
                        samples.append(sample)
                        telemetry.write(_raw(sample))
                        sampled = elapsed
                        if "error" not in sample:
                            swap = sample["swap_bytes"] if swap is None else swap
                            low = low + 1 if sample["available_bytes"] < 8 * 1024**3 else 0
                            if low >= 3 or sample["swap_bytes"] - swap > 1024**3: status = "resource_limit"
                    if stdout.tell() > STDOUT_LIMIT or stderr.tell() > STDERR_LIMIT: status = "output_limit"
                    if elapsed >= job["wall_seconds"]: status = status or "timeout"
                    if status is not None: break
                    sleep(0.02)
            except BaseException as error:
                interruption, status = error, "parent_error" if isinstance(error, Exception) else "cancelled"
            finally:
                cleanup_started = monotonic()
                if process is not None:
                    for action in (lambda: _kill_group(process), process.stdin.close, lambda: process.wait(timeout=5)):
                        try: action()
                        except BaseException as error:
                            cleanup_error = cleanup_error or repr(error)
                            if not isinstance(error, Exception): interruption, status = interruption or error, "cancelled"
                absent = process is None
                try:
                    while not absent and monotonic() - cleanup_started < 15:
                        try: os.killpg(process.pid, 0)
                        except ProcessLookupError: absent = True
                        except PermissionError: pass
                        if not absent: sleep(0.02)
                except BaseException as error:
                    cleanup_error = cleanup_error or repr(error)
                    if not isinstance(error, Exception): interruption, status = interruption or error, "cancelled"
                for stream in (stdout, stderr, telemetry):
                    try:
                        stream.flush()
                        os.fsync(stream.fileno())
                    except BaseException as error:
                        cleanup_error = cleanup_error or repr(error)
                        if not isinstance(error, Exception): interruption, status = interruption or error, "cancelled"
        if (output / "stdout.jsonl").stat().st_size > STDOUT_LIMIT or (output / "stderr.txt").stat().st_size > STDERR_LIMIT: status = status or "output_limit"
        with (output / "stdout.jsonl").open("rb") as stream: parsed = validate_events(stream.read(STDOUT_LIMIT + 1), job, process.pid if process else None)
        work_seconds = cleanup_started - started
        if job["schema_version"] == 2 and work_seconds > job["wall_seconds"]: status = status or "timeout"
        passed = process is not None and process.returncode == 0 and parsed["complete"] and absent and cleanup_error is None
        response = dict(**parsed, job_sha256=sha256(_raw(job)).hexdigest(), status=status or ("passed" if passed else "worker_error"), pid=process.pid if process else None,
            returncode=process.returncode if process else None, elapsed_seconds=monotonic() - started, cleanup_seconds=monotonic() - cleanup_started,
            group_absent=absent, cleanup_error=cleanup_error, parent_error=repr(interruption) if interruption else None,
            telemetry=samples, telemetry_complete=bool(samples) and all("error" not in s for s in samples),
            **({"work_seconds": work_seconds} if job["schema_version"] == 2 else {}))
        _write(output / "response.json", response)
        _sync(output)
        _sync(output.parent)
        if interruption is not None: raise interruption
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "lock"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    result = run(build_plan(**json.loads(Path(args.input).read_bytes())), args.output, args.lock)
    print(json.dumps({k: result[k] for k in ("status", "observed_generation_tokens", "elapsed_seconds")}))
    return result["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
