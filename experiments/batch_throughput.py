"""Owned fixture-only resident-batch jobs; no acquisition journal mutation."""

import argparse
import fcntl
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from time import monotonic, sleep

G = importlib.import_module("src.data-pipeline.local_generation")
L = importlib.import_module("src.data-pipeline.local_runner")
_exited, _kill_group = L._exited, L._kill_group
CHECKS = {"greedy_equal", "eos_at_cap", "length_cap", "reinsert_equal", "seed_independent"}
STDOUT_LIMIT, STDERR_LIMIT = 64 * 1024**2, 16 * 1024**2
FIELDS = {"start": "pid", "controls_checked": "request_sha256 proof", "ready": "eos_token_ids fixtures provenance hash_seconds load_seconds",
    "window_start": "id", "admit": "window uid row_index t_seconds", "token": "window uid seq token finish_reason t_seconds",
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


def _shape(job):
    G.require(type(job) is dict and set(job) == {"schema_version", "request", "fixtures", "windows", "worker", "wall_seconds", "source_hashes"}
              and type(job["schema_version"]) is int and job["schema_version"] == 1, "Invalid job fields")
    G.require(type(job["worker"]) is str and Path(job["worker"]).is_absolute()
              and _number(job["wall_seconds"]) and 0 < job["wall_seconds"] <= 1200, "Invalid launch bounds")
    paths = {Path(job["worker"]).resolve(), Path(__file__).resolve(), Path(G.__file__), G.WORKER, G.GUARD, Path(L.__file__)}
    G.require(type(job["source_hashes"]) is dict and set(job["source_hashes"]) == {str(p) for p in paths}
              and all(G.P._sha(h) for h in job["source_hashes"].values()), "Invalid source map")
    fixtures, windows = job["fixtures"], job["windows"]
    G.require(type(fixtures) is list and (1 <= len(fixtures) <= 8 or len(fixtures) == 32)
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
                      and type(row["seed"]) is int and 0 <= row["seed"] < 2**64 and type(row["cap"]) is int and 1 <= row["cap"] <= 128
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
    G.require(len({w["id"] for w in windows}) == len(windows) and order == sorted(order) and sum(counts) <= 128 and caps <= 16384
              and sum(n for n, o in zip(counts, order) if o == 0) == qualification_count
              and sum(n for n, o in zip(counts, order) if o == 1) == 1, "Invalid job workload")


def build_plan(request, fixtures, windows, *, worker, wall_seconds=1200):
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
    job = dict(schema_version=1, request=request, fixtures=G.snapshot(fixtures), windows=windows, worker=str(Path(worker).resolve()),
               wall_seconds=wall_seconds, source_hashes={str(p): sha256(p.read_bytes()).hexdigest() for p in paths})
    _shape(job)
    return job


def validate_events(raw, job, pid=None):
    events, observations, windows, current = [], [], [], None
    error, proof, eos, ready, qualified, complete, index = None, None, set(), False, False, False, 0
    identity = sha256(_raw(job)).hexdigest()
    truncated, raw = len(raw) > STDOUT_LIMIT, raw[:STDOUT_LIMIT]
    try:
        for line in raw.splitlines(keepends=True):
            G.require(not complete, "Events after done")
            G.require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = G.snapshot(json.loads(line))
            kind = event["event"]
            G.require(set(event) == {"event", "job_sha256"} | set(FIELDS[kind].split()) and event["job_sha256"] == identity, "Event binding differs")
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
                              and type(fixture["prompt_ids"]) is list and 0 < len(fixture["prompt_ids"]) <= 1536
                              and all(type(t) is int and t >= 0 for t in fixture["prompt_ids"]), "Ready fixtures differ")
                eos, ready = set(event["eos_token_ids"]), True
            elif kind == "window_start":
                G.require(ready and current is None and index < len(job["windows"]) and event["id"] == job["windows"][index]["id"], "Window order differs")
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
                observed_generation_tokens=sum(len(s["tokens"]) for s in observations), finished_sequences=sum(s["finish_reason"] is not None for s in observations))


def _sample(pid):
    try:
        import psutil
        cpu = psutil.Process(pid).cpu_times()
        return dict(cpu_seconds=cpu.user + cpu.system, rss_bytes=psutil.Process(pid).memory_info().rss,
                    available_bytes=psutil.virtual_memory().available, swap_bytes=psutil.swap_memory().used)
    except Exception as error:
        return dict(error=repr(error))


def run(job, output, lock_path):
    job = G.snapshot(job)
    _shape(job)
    G.require("precision_profile" in job["request"] and G.validate({k: v for k, v in job["request"].items() if k != "environment"}) == job["request"], "Request launch differs")
    sources = {p: Path(p).read_bytes() for p in job["source_hashes"]}
    G.require({p: sha256(raw).hexdigest() for p, raw in sources.items()} == job["source_hashes"], "Source changed")
    with os.fdopen(os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK), "r+b") as lease:
        G.require(stat.S_ISREG(os.fstat(lease.fileno()).st_mode), "Require regular lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(output).resolve()
        output.mkdir(exist_ok=False)
        _write(output / "job.json", job)
        for index, raw in enumerate(sources.values()): _write(output / f"source-{index}.py", raw)
        argv = [str(G.INTERPRETER), "-I", "-B", job["worker"], str(output / "job.json")]
        _write(output / "launch.json", dict(argv=argv, environment=job["request"]["environment"], lease_fd=lease.fileno(), sources=list(sources)))
        _sync(output)
        _sync(output.parent)
        started, process, status, interruption, cleanup_error = monotonic(), None, None, None, None
        samples, low, swap, sampled = [], 0, None, -1
        with (output / "stdout.jsonl").open("xb") as stdout, (output / "stderr.txt").open("xb") as stderr, (output / "telemetry.jsonl").open("xb") as telemetry:
            try:
                process = subprocess.Popen(argv, cwd=output, env=job["request"]["environment"], stdin=subprocess.PIPE,
                    stdout=stdout, stderr=stderr, start_new_session=True, pass_fds=(lease.fileno(),))
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
        passed = process is not None and process.returncode == 0 and parsed["complete"] and absent and cleanup_error is None
        response = dict(**parsed, job_sha256=sha256(_raw(job)).hexdigest(), status=status or ("passed" if passed else "worker_error"), pid=process.pid if process else None,
            returncode=process.returncode if process else None, elapsed_seconds=monotonic() - started, cleanup_seconds=monotonic() - cleanup_started,
            group_absent=absent, cleanup_error=cleanup_error, parent_error=repr(interruption) if interruption else None,
            telemetry=samples, telemetry_complete=bool(samples) and all("error" not in s for s in samples))
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
