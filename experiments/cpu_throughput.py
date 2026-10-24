"""Fixed invented native-verifier grid; no campaign or model requests."""

import argparse
import base64
from contextlib import contextmanager
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
H = importlib.import_module("experiments.batch_throughput")
_raw, _write, _sync = H._raw, H._write, H._sync
_exited, _kill_group = H._exited, H._kill_group
INTERPRETER, WORKER = ROOT / ".venv/bin/python", Path(__file__).resolve()
WALL_SECONDS, CLEANUP_SECONDS = 1200, 60
STDOUT_LIMIT, STDERR_LIMIT = 1024**2, 16 * 1024**2


def require(value, message):
    if not value: raise ValueError(message)


@contextmanager
def _inventory_deadline():
    def expired(*_): raise TimeoutError("Native cleanup deadline")
    prior = signal.signal(signal.SIGALRM, expired)
    deadline = monotonic() + CLEANUP_SECONDS
    signal.setitimer(signal.ITIMER_REAL, CLEANUP_SECONDS)
    try: yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


def fixtures():
    task = dict(split="train", id=0, cases=[dict(index=i, input=f"{n}\n", answer=str(2 * n), visible=n < 13)
                for i, n in zip((11, 4, 25, 9), range(11, 15))])
    programs, expected = [], []
    for kind in ("passed", "wrong_answer", "timeout", "hidden_failed"):
        for copy in range(2):
            source = "n=int(input())\n"
            if kind == "timeout": source += "if n==12:\n while True: pass\n"
            bad = 11 if kind == "wrong_answer" else 13 if kind == "hidden_failed" else -1
            programs.append((source + f"print(0 if n=={bad} else 2*n)\n# {kind}-{copy}\n").encode())
            expected.append([dict(index=c["index"], status="timeout" if kind == "timeout" and c["input"] == "12\n" else
                "wrong_answer" if int(c["input"]) == bad else "passed") for c in task["cases"]])
    return task, programs, expected


def build_plan(*, coalesced_four=False):
    require(type(coalesced_four) is bool, "Require boolean coalesced-four choice")
    docker = shutil.which("docker")
    require(docker is not None and Path(docker).is_absolute(), "Require available absolute Docker CLI")
    task, programs, expected = fixtures()
    paths = [WORKER, Path(__file__), Path(H.__file__), Path(H.L.__file__), ROOT / "configs/sandbox/compose.yaml"]
    paths += [ROOT / "src/data-pipeline" / n for n in ("inspect_runner.py", "verification.py", "screen.py", "native_recovery.py")]
    operations = []
    for condition, c in enumerate((4, 4) if coalesced_four else (1, 2, 4, 0, 0, 4, 2, 1)):
        for indices in ([list(range(8))] if c else [[i] for i in range(8)]):
            operations.append(dict(operation_id=secrets.token_hex(6), condition=condition, mode="coalesced" if c else "old",
                                   concurrency=c or 4, indices=indices))
    require(len({o["operation_id"] for o in operations}) == (2 if coalesced_four else 22), "Duplicate operation ID")
    return dict(schema_version=1, task=task, programs=[base64.b64encode(p).decode() for p in programs], expected=expected,
        operations=operations, source_hashes={str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
        environment={k: v for k, v in os.environ.items() if k in ("HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")}
                    | {"PATH": str(Path(docker).parent) + ":/opt/homebrew/bin:/usr/bin:/bin"}, wall_seconds=WALL_SECONDS - 2 * CLEANUP_SECONDS - .01 * WALL_SECONDS)


def _native():
    return tuple(importlib.import_module("src.data-pipeline." + n) for n in ("inspect_runner", "verification", "native_recovery"))


def _inventory(opid):
    return _native()[2]._inventory(opid)


def _closed(opid, daemon):
    started = monotonic()
    observed, projects, live = _inventory(opid)
    require(observed == daemon and not projects and live is False, "Native resources unproven")
    return dict(daemon_id=observed, projects=projects, live=live, elapsed_seconds=monotonic() - started)


def check_log(log, items, expected, contract, opid, concurrency):
    R, V, _ = _native()
    require(log.status == "success" and log.eval.task == opid, "Native log incomplete")
    config = log.eval.config
    require(all(getattr(config, k) == v for k, v in dict(max_samples=concurrency, max_sandboxes=concurrency,
        retry_on_error=0, fail_on_error=True, continue_on_fail=True, sandbox_cleanup=True, sandbox_prebuilt=True).items()), "Native options differ")
    results, containers, case_seconds, count = V.derive(log, items, contract), [], [], 0
    for sample, item, statuses in zip(sorted(log.samples, key=lambda s: s.id), items, expected, strict=True):
        mapping = {e["index"]: e["status"] for e in statuses}
        prefix = [(c["index"], mapping[c["index"]]) for c in item["cases"]]
        visible = sum(c["visible"] for c in item["cases"])
        end = visible if any(s != "passed" for _, s in prefix[:visible]) else next((i + 1 for i, (_, s) in enumerate(prefix) if i >= visible and s != "passed"), len(prefix))
        raw, container = sample.store["cases"], sample.store["container"]
        require([(r["index"], r["status"]) for r in raw] == prefix[:end], "Fixture case oracle differs")
        require(results[sample.id]["status"] == ("passed" if all(s == "passed" for _, s in prefix) else "failed"), "Derived fixture oracle differs")
        require(all((r["returncode"] == 0 and isinstance(r["stdout"], str) and R.output_matches(r["stdout"], c["answer"]))
            == (r["status"] == "passed") for r, c in zip(raw, item["cases"]) if r["status"] in ("passed", "wrong_answer")), "Fixture outputs differ")
        limits = container["HostConfig"]
        require(all(limits[k] == v for k, v in dict(NanoCpus=10**9, Memory=512 * 1024**2, MemorySwap=512 * 1024**2,
            PidsLimit=32, ReadonlyRootfs=True, NetworkMode="none", CapDrop=["ALL"]).items()) and container["Config"]["User"] == "65534:65534"
            and re.fullmatch(rf"inspect-{opid}-i[a-z0-9]{{6}}", container["Config"]["Labels"]["com.docker.compose.project"]), "Native container differs")
        containers.append(container["Id"])
        case_seconds.extend(r["elapsed_seconds"] for r in raw)
        count += len(raw)
    require(len(set(containers)) == len(containers), "Reused native container")
    return dict(results=results, case_count=count, case_seconds=case_seconds, containers=containers)


def coalesced(items, log_dir, c, opid, contract, host):
    R, _, _ = _native()
    samples = [R.Sample(id=i, input="Execute supplied program; no model calls.", metadata=dict(
        program_base64=base64.b64encode(item["program"]).decode(), cases=item["cases"], verification=item["verification"])) for i, item in enumerate(items)]
    task = R.Task(name=opid, dataset=samples, solver=R.execute_programs(), sandbox=("docker", str(R.COMPOSE)),
        metadata=dict(contract=contract, benchmark=dict(concurrency=c, mode="coalesced", docker=host)))
    return R.inspect_eval(task, model="mockllm/model", log_dir=str(log_dir), log_format="eval", max_samples=c, max_sandboxes=c,
        retry_on_error=0, fail_on_error=True, continue_on_fail=True, sandbox_cleanup=True, sandbox_prebuilt=True, score=False, display="none")[0]


def _dispatch(operation, items, logs, contract, host):
    return (_native()[0].run(items, logs, operation_id=operation["operation_id"])[0] if operation["mode"] == "old" else
            coalesced(items, logs, operation["concurrency"], operation["operation_id"], contract, host))


def _operation(job, directory, operation, seen):
    R, V, N = _native()
    started, opid = monotonic(), operation["operation_id"]
    op = directory / "ops" / opid
    op.mkdir()
    contract, daemon = R.contract(), N._daemon()
    host = json.loads(N._docker("info", "--format", "{{json .}}"))
    task_sha = sha256(_raw(job["task"])).hexdigest()
    items = [V.request(job["task"], task_sha, base64.b64decode(job["programs"][i])) for i in operation["indices"]]
    intent = dict(operation=operation, job_sha256=sha256(_raw(job)).hexdigest(), task_sha256=task_sha,
        program_sha256=[sha256(i["program"]).hexdigest() for i in items], contract=contract, daemon_id=daemon,
        host={k: host[k] for k in ("NCPU", "MemTotal", "ServerVersion")}, started_seconds=started)
    _write(op / "intent.json", intent)
    _sync(op); _sync(op.parent)
    before = monotonic()
    log = _dispatch(operation, items, op / "logs", contract, intent["host"])
    eval_seconds = monotonic() - before
    checked = check_log(log, items, [job["expected"][i] for i in operation["indices"]], contract, opid, operation["concurrency"])
    require(not seen.intersection(checked["containers"]), "Reused native container")
    seen.update(checked["containers"])
    path = Path(log.location)
    require(not path.is_symlink() and path.resolve().is_relative_to((op / "logs").resolve()), "Native log path differs")
    with path.open("rb") as stream: os.fsync(stream.fileno())
    _sync(path.parent)
    closure = _closed(opid, daemon)
    result = dict(**checked, operation=operation, log=str(path), log_sha256=sha256(path.read_bytes()).hexdigest(),
        eval_started_seconds=before, eval_seconds=eval_seconds, closure=closure, total_seconds=monotonic() - started)
    _write(op / "result.json", result)
    _sync(op); _sync(op.parent)
    return result


def _worker(jobfile):
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    def watch():
        while os.read(0, 1): pass
        os.killpg(0, signal.SIGKILL)
    threading.Thread(target=watch, daemon=True).start()
    raw = Path(jobfile).read_bytes()
    job, identity = json.loads(raw), sha256(raw).hexdigest()
    print(json.dumps(dict(event="start", job_sha256=identity, pid=os.getpid())), file=protocol, flush=True)
    require(all(sha256(Path(p).read_bytes()).hexdigest() == h for p, h in job["source_hashes"].items()), "Source changed")
    directory, seen, conditions = Path(jobfile).parent, set(), []
    (directory / "ops").mkdir()
    _sync(directory)
    setup_started = monotonic()
    _native()
    setup_seconds = monotonic() - setup_started
    _write(directory / "setup.json", dict(setup_seconds=setup_seconds)); _sync(directory)
    for condition in dict.fromkeys(op["condition"] for op in job["operations"]):
        started = monotonic()
        results = []
        for op in (o for o in job["operations"] if o["condition"] == condition):
            before = monotonic()
            try: results.append(_operation(job, directory, op, seen))
            except BaseException as failure:
                _write(directory / "failure.json", dict(operation=op, error=repr(failure), elapsed_seconds=monotonic() - before))
                _sync(directory)
                raise
        elapsed = monotonic() - started
        conditions.append(dict(condition=condition, mode=results[0]["operation"]["mode"], concurrency=results[0]["operation"]["concurrency"], total_seconds=elapsed,
            programs=sum(len(r["results"]) for r in results), programs_per_second=8 / elapsed,
            verified_programs_per_second=sum(x["status"] == "passed" for r in results for x in r["results"]) / elapsed))
        _write(directory / f"condition-{condition}.json", conditions[-1]); _sync(directory)
    _write(directory / "summary.json", dict(conditions=conditions, setup_seconds=setup_seconds, generator_requests=0,
        verifications=sum(len(op["indices"]) for op in job["operations"]), eval_starts=len(job["operations"])))
    _sync(directory)
    print(json.dumps(dict(event="done", job_sha256=identity)), file=protocol, flush=True)


def run(directory, lock_path, *, lease_fd=None, coalesced_four=False):
    job = build_plan(coalesced_four=coalesced_four)
    identity = sha256(_raw(job)).hexdigest()
    directory = Path(directory).resolve()
    with H.lease(lock_path, lease_fd) as lease:
        directory.mkdir()
        _write(directory / "job.json", job)
        for i, (p, h) in enumerate(job["source_hashes"].items()):
            raw = Path(p).read_bytes(); require(sha256(raw).hexdigest() == h, "Source changed")
            _write(directory / f"source-{i}", raw)
        argv = [str(INTERPRETER), "-I", "-B", str(WORKER), "--worker", str(directory / "job.json")]
        lease_stat = os.fstat(lease.fileno())
        _write(directory / "launch.json", dict(argv=argv, environment=job["environment"], lease_fd=lease.fileno(),
            lease_dev=lease_stat.st_dev, lease_ino=lease_stat.st_ino))
        _sync(directory); _sync(directory.parent)
        started, process, status, interruption, error = monotonic(), None, None, None, None
        with (directory / "stdout.jsonl").open("xb") as out, (directory / "stderr.txt").open("xb") as err:
            try:
                process = subprocess.Popen(argv, cwd=directory, env=job["environment"], stdin=subprocess.PIPE,
                    stdout=out, stderr=err, start_new_session=True, pass_fds=(lease.fileno(),))
                while not _exited(process):
                    if out.tell() > STDOUT_LIMIT or err.tell() > STDERR_LIMIT: status = "output_limit"
                    if monotonic() - started >= job["wall_seconds"]: status = status or "timeout"
                    if status: break
                    sleep(.02)
            except BaseException as failure:
                interruption, status = failure, "parent_error" if isinstance(failure, Exception) else "cancelled"
            finally:
                cleanup = monotonic()
                if process:
                    for action in (lambda: _kill_group(process), process.stdin.close, lambda: process.wait(timeout=5)):
                        try: action()
                        except BaseException as failure:
                            error = error or repr(failure)
                            if not isinstance(failure, Exception): interruption, status = interruption or failure, "cancelled"
                absent = process is None
                try:
                    while not absent and monotonic() - cleanup < CLEANUP_SECONDS:
                        try: os.killpg(process.pid, 0)
                        except ProcessLookupError: absent = True
                        except PermissionError: pass
                        if not absent: sleep(.02)
                except BaseException as failure:
                    error = error or repr(failure)
                    if not isinstance(failure, Exception): interruption, status = interruption or failure, "cancelled"
                for stream in (out, err):
                    try: stream.flush(); os.fsync(stream.fileno())
                    except BaseException as failure:
                        error = error or repr(failure)
                        if not isinstance(failure, Exception): interruption, status = interruption or failure, "cancelled"
        events, results, proofs, attempts, audit_errors, protocol_error = [], [], [], [], [], None
        try:
            with (directory / "stdout.jsonl").open("rb") as stream: raw = stream.read(STDOUT_LIMIT + 1)
            if len(raw) > STDOUT_LIMIT or (directory / "stderr.txt").stat().st_size > STDERR_LIMIT:
                status = status or "output_limit"
                raise ValueError("Output limit")
            events = [json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b"\n")]
            require(process is not None and events == [dict(event="start", job_sha256=identity, pid=process.pid),
                    dict(event="done", job_sha256=identity)] and raw.endswith(b"\n"), "Worker protocol differs")
        except (ValueError, TypeError, KeyError, RecursionError, OverflowError) as failure: protocol_error = str(failure)
        attempts = [op for op in job["operations"] if (directory / "ops" / op["operation_id"] / "intent.json").exists()]
        try:
            with _inventory_deadline() as deadline:
                for operation in attempts:
                    if monotonic() >= deadline: error = error or "Native cleanup deadline"; break
                    try:
                        intent = json.loads((directory / "ops" / operation["operation_id"] / "intent.json").read_bytes())
                        require(intent["operation"] == operation and intent["job_sha256"] == identity, "Native intent differs")
                        proofs.append(dict(operation_id=operation["operation_id"], **_closed(operation["operation_id"], intent["daemon_id"])))
                    except Exception as failure:
                        error = error or repr(failure)
                        audit_errors.append(dict(operation_id=operation["operation_id"], error=repr(failure)))
        except BaseException as failure:
            error = error or repr(failure)
            if not isinstance(failure, Exception): interruption, status = interruption or failure, "cancelled"
        for operation in attempts:
            try:
                result_path = directory / "ops" / operation["operation_id"] / "result.json"
                if result_path.exists():
                    result = json.loads(result_path.read_bytes())
                    require(result["operation"] == operation and type(result["case_count"]) is int and 0 <= result["case_count"] <= 32, "Native result differs")
                    results.append(result)
            except BaseException as failure:
                error = error or repr(failure)
                audit_errors.append(dict(operation_id=operation["operation_id"], error=repr(failure)))
                if not isinstance(failure, Exception): interruption, status = interruption or failure, "cancelled"; break
        passed = process is not None and process.returncode == 0 and protocol_error is None and absent and error is None and len(results) == len(proofs) == len(job["operations"])
        response = dict(status=status or ("passed" if passed else "worker_error"), job_sha256=sha256(_raw(job)).hexdigest(),
            pid=process.pid if process else None, returncode=process.returncode if process else None, group_absent=absent,
            cleanup_error=error, protocol_error=protocol_error, parent_error=repr(interruption) if interruption else None,
            elapsed_seconds=monotonic() - started, cleanup_seconds=monotonic() - cleanup, results=results,
            audit_errors=audit_errors, unproven_operations=[op["operation_id"] for op in attempts if op["operation_id"] not in {p["operation_id"] for p in proofs}],
            resource_proofs=proofs, generator_requests=0, attempted_eval_starts=len(attempts),
            attempted_verifications=sum(len(op["indices"]) for op in attempts), completed_case_count_lower_bound=sum(r["case_count"] for r in results))
        _write(directory / "response.json", response); _sync(directory); _sync(directory.parent)
        if interruption is not None: raise interruption
        return response


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--output")
    parser.add_argument("--lock")
    parser.add_argument("--lease-fd", type=int)
    parser.add_argument("--coalesced-four", action="store_true")
    args = parser.parse_args(argv)
    if args.worker: _worker(args.worker); return 0
    parser.error("require --output and --lock") if not (args.output and args.lock) else None
    if args.lease_fd is not None: H.watch_lease_parent()
    result = run(args.output, args.lock, lease_fd=args.lease_fd, coalesced_four=args.coalesced_four)
    print(json.dumps(dict(status=result["status"], attempted_eval_starts=result["attempted_eval_starts"])))
    return result["status"] != "passed"


if __name__ == "__main__":
    raise SystemExit(main())
