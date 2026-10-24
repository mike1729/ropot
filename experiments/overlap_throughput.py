"""Four fixed serial/overlap phases on independent GPU and native fixtures."""

import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
from time import monotonic, sleep

H, M, C = (importlib.import_module("experiments." + name) for name in
           ("batch_throughput", "mlx_batch_throughput", "cpu_throughput"))
ROOT, INTERPRETER = Path(__file__).resolve().parents[1], C.INTERPRETER
PHASES = ("gpu_cpu", "overlap", "overlap", "cpu_gpu")
WORK_SECONDS, STOP_SECONDS, BLOCK_SECONDS = 1200, 150, 3600
require, read = H.G.require, lambda path: json.loads(Path(path).read_bytes())


def union(intervals):
    require(all(len(v) == 2 and all(H._number(t) and t > 0 for t in v) and v[0] < v[1] for v in intervals), "Invalid intervals")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]: merged[-1][1] = max(end, merged[-1][1])
        else: merged.append([start, end])
    return merged


def intersection_seconds(left, right):
    return sum(max(0, min(b, d) - max(a, c)) for a, b in union(left) for c, d in union(right))


def freeze(profiles_path, winner_path, directory):
    profiles, old_raw = Path(profiles_path).read_bytes(), Path(winner_path).read_bytes()
    old = H.G.snapshot(json.loads(old_raw)); H._shape(old)
    prior = old["request"]["precision_profile"]["descriptor"]
    source, bits = prior["source"], prior["bits"]
    descriptor = json.loads(profiles)["profiles"][str(bits)][source]["descriptor"]
    require(descriptor == prior and len(old["fixtures"]) == 32, "Winner profile differs")
    configs = [w["config"] for w in old["windows"] if w["mode"] == "measure"]
    require(len(configs) == 2 and configs[0] == configs[1], "Winner config differs")
    job = M.build_job(descriptor, [configs[0]], 600.0, panel_size=32, cache_limit_bytes=old.get("cache_limit_bytes"))
    require(job["fixtures"] == old["fixtures"] and job["windows"] == old["windows"] and
        {k: v for k, v in job["request"].items() if k != "wall_time_seconds"} ==
        {k: v for k, v in old["request"].items() if k != "wall_time_seconds"}, "Winner workload differs")
    require({k: old[k] for k in ("cache_limit_bytes",) if k in old} == {k: job[k] for k in ("cache_limit_bytes",) if k in job}, "Winner cache differs")
    cpu = C.build_plan(coalesced_four=True)
    sources = dict(job["source_hashes"], **cpu["source_hashes"])
    sources[str(Path(__file__).resolve())] = sha256(Path(__file__).read_bytes()).hexdigest()
    directory = Path(directory).resolve(); directory.mkdir()
    plan = dict(job=job, cpu_job=cpu, source=source, bits=bits, config=configs[0], sources=sources,
        profiles_path=str(directory / "profiles.json"), old_job_sha256=sha256(old_raw).hexdigest(), job_sha256=sha256(H._raw(job)).hexdigest())
    for name, raw in (("profiles.json", profiles), ("winner.json", old_raw), ("job.json", H._raw(job)), ("plan.json", H._raw(plan))): H._write(directory / name, raw)
    for i, (path, digest) in enumerate(sources.items()):
        raw = Path(path).read_bytes(); require(sha256(raw).hexdigest() == digest, "Frozen source changed")
        H._write(directory / f"source-{i}", raw)
    H._sync(directory); H._sync(directory.parent)
    return plan


def _ownership(pid, parent_pid):
    import psutil
    try:
        process = psutil.Process(pid)
        observed = dict(pid=pid, ppid=process.ppid(), pgid=os.getpgid(pid), status=process.status())
        require(observed["ppid"] == parent_pid and observed["pgid"] == pid and observed["status"] not in ("zombie", "dead"), "GPU ownership differs")
        return observed
    except (psutil.Error, OSError) as error: raise ValueError("GPU ownership unavailable") from error


def measured_gate(directory, job, parent_pid):
    directory = Path(directory)
    try:
        require((directory / "job.json").read_bytes() == H._raw(job), "GPU job differs")
        with (directory / "stdout.jsonl").open("rb") as stream: raw = stream.read(H.STDOUT_LIMIT + 1)
    except FileNotFoundError: return None
    require(len(raw) <= H.STDOUT_LIMIT, "GPU prefix limit")
    if b"\n" not in raw: return None
    first = json.loads(raw.splitlines()[0])
    require(type(first) is dict and type(first.get("pid")) is int and first["pid"] > 0, "GPU start PID differs")
    parsed = H.validate_events(raw, job, pid=first["pid"])
    require(not parsed["complete"] and parsed["protocol_error"] in ("Missing done", "Incomplete protocol tail"), "GPU gate prefix differs")
    first_window = next(w["id"] for w in job["windows"] if w["mode"] == "measure")
    starts = [e for e in parsed["events"] if e["event"] == "window_start" and e["id"] == first_window]
    if not starts: return None
    return dict(worker=_ownership(first["pid"], parent_pid), observed_seconds=monotonic(), window=starts[0])


def audit(kind, directory, plan, child):
    directory = Path(directory); response = read(directory / "response.json")
    require(child["returncode"] == 0 and child["group_absent"] is True and response["status"] == "passed"
        and response["group_absent"] is True and response["cleanup_error"] is None, "Child proof incomplete")
    job = read(directory / "job.json")
    require(response["job_sha256"] == sha256(H._raw(job)).hexdigest(), "Child job binding differs")
    if kind == "gpu":
        require((directory / "job.json").read_bytes() == H._raw(plan["job"]) and response["complete"] is True, "GPU job differs")
        if "cache_limit_bytes" in job:
            proof = read(directory / "cache-control.json"); limit = job["cache_limit_bytes"]
            require(proof["job_sha256"] == plan["job_sha256"] and all(type(proof[k]) is int and proof[k] == limit for k in ("requested_bytes", "observed_bytes")), "GPU cache proof differs")
        ids = {w["id"] for w in job["windows"] if w["mode"] == "measure"}
        starts = {e["id"]: e["started_seconds"] for e in response["events"] if e["event"] == "window_start"}
        intervals = [(starts[w["window"]], starts[w["window"]] + w["elapsed"]) for w in response["windows"] if w["window"] in ids]
    else:
        require({k: v for k, v in job.items() if k != "operations"} == {k: v for k, v in plan["cpu_job"].items() if k != "operations"}
            and [{k: v for k, v in op.items() if k != "operation_id"} for op in job["operations"]] ==
                [{k: v for k, v in op.items() if k != "operation_id"} for op in plan["cpu_job"]["operations"]], "CPU workload differs")
        require(len(response["results"]) == len(response["resource_proofs"]) == 2 and not response["unproven_operations"], "CPU native proof incomplete")
        intervals = [(r["eval_started_seconds"], r["eval_started_seconds"] + r["eval_seconds"]) for r in response["results"]]
    require(len(intervals) == 2 and all(child["launch_seconds"] <= a and b <= child["reap_seconds"] for a, b in union(intervals)), "Intervals outside owner")
    return dict(response=response, intervals=intervals)


def commands(kind, plan, directory, lock_path, lease_fd):
    common = ["--output", str(Path(directory) / kind), "--lock", str(lock_path), "--lease-fd", str(lease_fd)]
    if kind == "cpu": return [str(INTERPRETER), "-m", "experiments.cpu_throughput", "--coalesced-four", *common]
    args = [str(INTERPRETER), "-m", "experiments.mlx_batch_throughput", "--run", "--profiles", plan["profiles_path"],
        "--source", plan["source"], "--bits", str(plan["bits"]), "--configs", json.dumps([plan["config"]]), "--wall", "600.0", "--panel-size", "32", *common]
    if "cache_limit_bytes" in plan["job"]: args += ["--cache-gib", str(plan["job"]["cache_limit_bytes"] // 1024**3)]
    return args


def spawn(kind, directory, plan, lock_path, lease_fd):
    out, err = (Path(directory) / f"{kind}-cli.stdout").open("xb"), (Path(directory) / f"{kind}-cli.stderr").open("xb")
    argv = commands(kind, plan, directory, lock_path, lease_fd)
    before = monotonic()
    H._write(Path(directory) / f"{kind}-launch.json", dict(argv=argv, launch_seconds=before, lease_fd=lease_fd)); H._sync(directory)
    try: process = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True, pass_fds=(lease_fd,))
    except BaseException:
        out.close(); err.close()
        raise
    return dict(kind=kind, process=process, stdout=out, stderr=err, pid=process.pid, launch_seconds=before, reap_seconds=None,
        returncode=None, group_absent=False, finalized=False, argv=argv)


def stop(children, *, interrupt=False):
    errors, interruption = [], None
    pending = [c for c in children if not c.get("finalized")]
    if interrupt:
        for child in pending:
            try:
                if child["process"].returncode is None: os.killpg(child["pid"], signal.SIGINT)
            except ProcessLookupError: pass
            except BaseException as failure:
                errors.append(repr(failure))
                if not isinstance(failure, Exception): interruption = interruption or failure
    deadline = monotonic() + .8 * STOP_SECONDS
    try:
        while any(c["process"].returncode is None and not H._exited(c["process"]) for c in pending):
            if monotonic() >= deadline: break
            sleep(.05)
    except BaseException as failure:
        errors.append(repr(failure))
        if not isinstance(failure, Exception): interruption = interruption or failure
    for child in pending:
        process = child["process"]
        for action in (lambda: H._kill_group(process) if process.returncode is None else None, lambda: process.wait(timeout=5), process.stdin.close):
            try: action()
            except BaseException as failure:
                errors.append(repr(failure))
                if not isinstance(failure, Exception): interruption = interruption or failure
        child.update(returncode=process.returncode, reap_seconds=monotonic() if process.returncode is not None else None, finalized=True)
        try: os.killpg(child["pid"], 0)
        except ProcessLookupError: child["group_absent"] = True
        except BaseException as failure:
            errors.append(repr(failure))
            if not isinstance(failure, Exception): interruption = interruption or failure
        for stream in (child["stdout"], child["stderr"]):
            try: stream.flush(); os.fsync(stream.fileno()); stream.close()
            except BaseException as failure:
                errors.append(repr(failure))
                if not isinstance(failure, Exception): interruption = interruption or failure
    return errors, interruption


def run_phase(plan, phase, directory, lock_path, lease_fd):
    require(phase in PHASES, "Invalid fixed phase")
    directory = Path(directory); directory.mkdir()
    H._sync(directory.parent)
    children, audits, errors, interruption, gate = {}, {}, [], None, None
    started = monotonic()
    try:
        require(all(sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in plan["sources"].items()), "Frozen source changed")
        first = "cpu" if phase == "cpu_gpu" else "gpu"
        children[first] = spawn(first, directory, plan, lock_path, lease_fd)
        while len(audits) < 2:
            require(monotonic() - started < WORK_SECONDS, "Phase work deadline")
            for kind, child in list(children.items()):
                if kind in audits: continue
                if H._exited(child["process"]):
                    failed, cancelled = stop([child]); errors.extend(failed); interruption = interruption or cancelled
                    require(not errors and interruption is None, "CLI cleanup unproven")
                    audits[kind] = audit(kind, directory / kind, plan, child)
            if len(children) == 1:
                if phase == "overlap":
                    require("gpu" not in audits, "GPU finished before CPU gate")
                    gate = measured_gate(directory / "gpu", plan["job"], children["gpu"]["pid"])
                    if gate is not None: children["cpu"] = spawn("cpu", directory, plan, lock_path, lease_fd)
                elif first in audits:
                    second = "gpu" if first == "cpu" else "cpu"
                    children[second] = spawn(second, directory, plan, lock_path, lease_fd)
            require(all(c["stdout"].closed or c["stdout"].tell() <= 16 * 1024**2 and c["stderr"].tell() <= 16 * 1024**2 for c in children.values()), "CLI output limit")
            if len(audits) < 2: sleep(.25)
    except BaseException as failure:
        errors.append(repr(failure))
        if not isinstance(failure, Exception): interruption = interruption or failure
    finally:
        failed, cancelled = stop(list(children.values()), interrupt=bool(errors) or interruption is not None)
        errors.extend(failed); interruption = interruption or cancelled
    overlap = intersection_seconds(audits["gpu"]["intervals"], audits["cpu"]["intervals"]) if len(audits) == 2 else None
    usable = not errors and interruption is None and len(audits) == 2 and (phase != "overlap" or overlap > 0)
    costs = {}
    for kind in children:
        try:
            r = read(directory / kind / "response.json")
            costs[kind] = dict(tokens_lower_bound=r.get("observed_generation_tokens"), reply_admissions=len(r["observations"]) if "observations" in r else None,
                native_eval_attempts=r.get("attempted_eval_starts"), native_verification_attempts=r.get("attempted_verifications"))
        except Exception as failure: costs[kind] = dict(unknown=repr(failure))
    states = {k: {field: value for field, value in c.items() if field not in ("process", "stdout", "stderr")} for k, c in children.items()}
    row = dict(mode=phase, status="passed" if usable else "blocked", elapsed_seconds=monotonic() - started,
        children=states, audits=audits, gate=gate, intersection_seconds=overlap, errors=errors, partial_costs=costs,
        gpu_job_attempts=int((directory / "gpu-launch.json").exists()), native_intents_lower_bound=len(list((directory / "cpu/ops").glob("*/intent.json"))))
    H._write(directory / "phase.json", row); H._sync(directory); H._sync(directory.parent)
    if interruption is not None: raise interruption
    return row


def run(profiles_path, winner_path, output, lock_path):
    output, lock_path = Path(output).resolve(), Path(lock_path).resolve()
    started, rows, interruption = monotonic(), [], None
    with H.lease(lock_path) as lease:
        plan = freeze(profiles_path, winner_path, output)
        try:
            for i, phase in enumerate(PHASES):
                if BLOCK_SECONDS - monotonic() + started < WORK_SECONDS + STOP_SECONDS: break
                row = run_phase(plan, phase, Path(output) / f"phase-{i}", lock_path, lease.fileno())
                rows.append(row)
                if row["status"] != "passed": break
        except BaseException as failure:
            interruption = failure
        finally:
            retained = []
            for i in range(4):
                directory = Path(output) / f"phase-{i}"
                if not directory.exists(): continue
                try: row = read(directory / "phase.json")
                except Exception as failure: row = dict(status="blocked", mode=PHASES[i], phase_read_error=repr(failure))
                row.update(gpu_job_attempts=int((directory / "gpu-launch.json").exists()),
                    native_intents_lower_bound=len(list((directory / "cpu/ops").glob("*/intent.json"))))
                retained.append(row)
            if retained: rows = retained
            passed = len(rows) == 4 and all(r["status"] == "passed" for r in rows) and interruption is None
            result = dict(status="passed" if passed else "blocked", phases=rows, elapsed_seconds=monotonic() - started,
                parent_error=repr(interruption) if interruption else None, unrun=[dict(phase=i, mode=PHASES[i], reason="prior_failure_or_budget") for i in range(len(rows), 4)],
                planned=dict(gpu_jobs=4, native_evals=8, verifications=64, gpu_sequences=4 * sum(len(w["rows"]) for w in plan["job"]["windows"]),
                    gpu_token_cap=4 * sum(r["cap"] for w in plan["job"]["windows"] for r in w["rows"])),
                attempted=dict(gpu_jobs=sum(r["gpu_job_attempts"] for r in rows), native_intents_lower_bound=sum(r["native_intents_lower_bound"] for r in rows)),
                serial_over_overlap_ratio=(rows[0]["elapsed_seconds"] + rows[3]["elapsed_seconds"]) / (rows[1]["elapsed_seconds"] + rows[2]["elapsed_seconds"]) if passed else None)
            H._write(Path(output) / "response.json", result); H._sync(output); H._sync(Path(output).parent)
    if interruption is not None: raise interruption
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profiles", "winner", "output", "lock"): parser.add_argument("--" + name, required=True)
    result = run(**{k + "_path" if k in ("profiles", "winner", "lock") else k: v for k, v in vars(parser.parse_args(argv)).items()})
    print(json.dumps(dict(status=result["status"], attempted=result["attempted"], serial_over_overlap_ratio=result["serial_over_overlap_ratio"])))
    return result["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
