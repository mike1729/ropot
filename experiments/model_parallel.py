"""One screened, fixed Q8 two-model serial/parallel fixture block."""

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
from time import monotonic, sleep

O = importlib.import_module("experiments.overlap_throughput")
H, M, require, read = O.H, O.M, O.require, O.read
ROLES = ("actor", "synthetic")
PHASES = ("actor_synthetic", "parallel", "parallel", "synthetic_actor")
GIB, BLOCK_SECONDS = 1024**3, 2700
CONFIGS = dict(actor=dict(prefill_batch_size=4, prefill_step_size=256),
               synthetic=dict(prefill_batch_size=2, prefill_step_size=2048))


def prior(directory, source, batch, limited):
    directory = Path(directory).resolve()
    job_raw, response_raw = ((directory / name).read_bytes() for name in ("job.json", "response.json"))
    job, response = json.loads(job_raw), json.loads(response_raw)
    H._shape(job)
    identity = sha256(job_raw).hexdigest()
    descriptor = job["request"]["precision_profile"]["descriptor"]
    config = dict(CONFIGS[source], batch_size=batch)
    require(descriptor["source"] == source and descriptor["bits"] == 8 and len(job["fixtures"]) == 32
        and [w["config"] for w in job["windows"] if w["mode"] == "measure"] == [config, config], "Prior model/config differs")
    require(response["job_sha256"] == identity and response["status"] == "passed" and response["complete"] is True
        and type(response["returncode"]) is int and response["returncode"] == 0 and response["group_absent"] is True
        and all(response[k] is None for k in ("cleanup_error", "parent_error", "protocol_error"))
        and type(response["pid"]) is int and response["pid"] > 0 and response["telemetry_complete"] is True, "Prior proof incomplete")
    checks = [e for e in response["events"] if e["event"] == "qualification"]
    require(len(checks) == 1 and checks[0]["job_sha256"] == identity and set(checks[0]["checks"]) == H.CHECKS
        and all(v is True for v in checks[0]["checks"].values()), "Prior qualification differs")
    telemetry, windows = response["telemetry"], response["windows"]
    require(telemetry and all("error" not in t and all(H._number(t[k]) for k in ("available_bytes", "swap_bytes")) for t in telemetry)
        and max(t["swap_bytes"] for t in telemetry) <= telemetry[0]["swap_bytes"], "Prior swap/telemetry incomplete")
    require([w["window"] for w in windows] == [w["id"] for w in job["windows"]] and all(w["job_sha256"] == identity
        and H._number(w["elapsed"]) and w["elapsed"] > 0 and type(w["memory"]) is dict
        and {"active_bytes", "peak_bytes", "cache_before_bytes", "cache_bytes"} <= set(w["memory"]) and all(H._number(w["memory"][k])
        for k in ("active_bytes", "peak_bytes", "cache_before_bytes", "cache_bytes")) for w in windows), "Prior memory incomplete")
    cache = max(max(w["memory"][k] for k in ("cache_before_bytes", "cache_bytes")) for w in windows)
    cache_raw = None
    if limited:
        cache_raw = (directory / "cache-control.json").read_bytes(); proof = json.loads(cache_raw)
        require(type(job.get("cache_limit_bytes")) is int and job["cache_limit_bytes"] == 4 * GIB
            and proof["job_sha256"] == identity and all(type(proof[k]) is int and proof[k] == 4 * GIB
            for k in ("requested_bytes", "observed_bytes")), "Prior cache proof differs")
    else: require("cache_limit_bytes" not in job and cache < 4 * GIB, "Fallback default cache differs")
    return dict(directory=str(directory), job=job, response=response, job_sha256=identity,
        response_sha256=sha256(response_raw).hexdigest(), config=config, cache_max_bytes=cache,
        cache_control_sha256=sha256(cache_raw).hexdigest() if cache_raw is not None else None,
        screening_bytes=max(w["memory"]["peak_bytes"] + max(w["memory"][k] for k in ("cache_before_bytes", "cache_bytes")) for w in windows))


def select_pair(priors, available_bytes):
    require(H._number(available_bytes), "Invalid available memory")
    required = {kind: sum(priors[role + "_" + kind]["screening_bytes"] for role in ROLES) + 12 * GIB for kind in ("best", "fallback")}
    choice = next((kind for kind in ("best", "fallback") if required[kind] <= available_bytes), None)
    return dict(choice=choice, required_bytes=required, margin_bytes=12 * GIB,
        limitation="Noncoincident allocator screening estimate; loading/transient physical memory is not bounded")


def _memory():
    import psutil
    vm = psutil.virtual_memory()
    return dict(available_bytes=vm.available, total_bytes=vm.total, swap_used_bytes=psutil.swap_memory().used, observed_seconds=monotonic())


def prepare(profiles_path, prior_paths, output):
    require(set(prior_paths) == {r + "_" + k for r in ROLES for k in ("best", "fallback")}, "Prior keys differ")
    evidence = {r + "_" + k: prior(prior_paths[r + "_" + k], r, 32 if k == "best" else 8, k == "best")
                for r in ROLES for k in ("best", "fallback")}
    profiles_raw = Path(profiles_path).read_bytes(); profiles = json.loads(profiles_raw)["profiles"]["8"]
    require(all(item["job"]["request"]["precision_profile"]["descriptor"] == profiles[key.split("_")[0]]["descriptor"]
        for key, item in evidence.items()), "Prior profile differs")
    require(len({(p["descriptor"]["receipt_sha256"], p["descriptor"]["runtime_guard_sha256"]) for p in profiles.values()}) == 1, "Common profile differs")
    preflight = _memory()
    require(all(H._number(v) for v in preflight.values()), "Invalid fresh memory observation")
    decision = select_pair(evidence, preflight["available_bytes"])
    output = Path(output).resolve(); output.mkdir(); H._sync(output.parent)
    H._write(output / "profiles.json", profiles_raw)
    for key, item in evidence.items():
        for name in ("job", "response"):
            raw = (Path(item["directory"]) / (name + ".json")).read_bytes()
            require(sha256(raw).hexdigest() == item[name + "_sha256"], "Prior changed while freezing")
            H._write(output / (key + "-" + name + ".json"), raw)
        if item["cache_control_sha256"] is not None:
            raw = (Path(item["directory"]) / "cache-control.json").read_bytes()
            require(sha256(raw).hexdigest() == item["cache_control_sha256"], "Prior cache proof changed")
            H._write(output / (key + "-cache-control.json"), raw)
    models = {}
    if decision["choice"] is not None:
        for role in ROLES:
            item = evidence[role + "_" + decision["choice"]]
            proposal = M.build_job(item["job"]["request"]["precision_profile"]["descriptor"], [item["config"]], 600.0, panel_size=32, cache_limit_bytes=4 * GIB)
            require(proposal["fixtures"] == item["job"]["fixtures"] and proposal["windows"] == item["job"]["windows"], "Proposal workload differs")
            path = output / (role + "-proposal.json"); H._write(path, proposal)
            models[role] = O.freeze(output / "profiles.json", path, output / (role + "-frozen"))
    sources = {str(Path(__file__).resolve()): sha256(Path(__file__).read_bytes()).hexdigest()}
    for model in models.values():
        require(all(p not in sources or sources[p] == digest for p, digest in model["sources"].items()), "Model source union differs")
        sources.update(model["sources"])
    for i, (path, digest) in enumerate(sources.items()):
        raw = Path(path).read_bytes(); require(sha256(raw).hexdigest() == digest, "Frozen source changed")
        H._write(output / f"source-{i}", raw)
    pair = dict(evidence=evidence, preflight=preflight, decision=decision, models=models, sources=sources,
        proposal_status="Unmeasured cache4 proposals; real prior identities retained separately")
    H._write(output / "plan.json", pair); H._sync(output); H._sync(output.parent)
    return pair


def run_phase(pair, mode, directory, lock_path, lease_fd):
    require(mode in PHASES, "Invalid fixed phase")
    directory = Path(directory); directory.mkdir(); H._sync(directory.parent)
    children, audits, observations, errors, interruption = {}, {}, {}, [], None
    started = monotonic()
    def launch(role):
        child_dir = directory / role; child_dir.mkdir(); H._sync(directory)
        children[role] = O.spawn("gpu", child_dir, pair["models"][role], lock_path, lease_fd)
    try:
        require(all(sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in pair["sources"].items()), "Frozen source changed")
        first = "synthetic" if mode == "synthetic_actor" else "actor"
        launch(first)
        if mode == "parallel": launch("synthetic")
        while len(audits) < 2:
            require(monotonic() - started < O.WORK_SECONDS, "Phase work deadline")
            for role, child in list(children.items()):
                if role in audits: continue
                if H._exited(child["process"]):
                    failed, cancelled = O.stop([child]); errors.extend(failed); interruption = interruption or cancelled
                    require(not errors and interruption is None, "CLI cleanup unproven")
                    audits[role] = O.audit("gpu", directory / role / "gpu", pair["models"][role], child)
                elif role not in observations:
                    try:
                        value = O.measured_gate(directory / role / "gpu", pair["models"][role]["job"], child["pid"])
                        if value is not None: observations[role] = value
                    except (ValueError, OSError) as error: observations[role] = dict(unavailable=repr(error))
            if mode != "parallel" and len(children) == 1 and first in audits: launch(next(r for r in ROLES if r != first))
            require(all(c["stdout"].closed or c["stdout"].tell() <= H.STDERR_LIMIT and c["stderr"].tell() <= H.STDERR_LIMIT for c in children.values()), "CLI output limit")
            if len(audits) < 2: sleep(.25)
    except BaseException as failure:
        errors.append(repr(failure))
        if not isinstance(failure, Exception): interruption = interruption or failure
    finally:
        failed, cancelled = O.stop(list(children.values()), interrupt=bool(errors) or interruption is not None)
        errors.extend(failed); interruption = interruption or cancelled
    overlap = O.intersection_seconds(audits["actor"]["intervals"], audits["synthetic"]["intervals"]) if len(audits) == 2 else None
    passed = not errors and interruption is None and len(audits) == 2 and (mode != "parallel" or overlap > 0)
    costs = {}
    for role in (r for r in ROLES if (directory / r / "gpu-launch.json").exists()):
        try:
            response = read(directory / role / "gpu/response.json")
            costs[role] = dict(tokens_lower_bound=response.get("observed_generation_tokens"), finished_replies_lower_bound=response.get("finished_sequences"))
        except Exception as failure: costs[role] = dict(unknown=repr(failure))
    states = {r: {k: v for k, v in c.items() if k not in ("process", "stdout", "stderr")} for r, c in children.items()}
    elapsed = monotonic() - started
    row = dict(mode=mode, status="passed" if passed else "blocked", elapsed_seconds=elapsed, children=states, audits=audits,
        observations=observations, intersection_seconds=overlap, errors=errors, partial_costs=costs,
        gpu_job_attempts=sum((directory / r / "gpu-launch.json").exists() for r in ROLES), useful_reply_rate=128 / elapsed if passed else None)
    H._write(directory / "phase.json", row); H._sync(directory); H._sync(directory.parent)
    if interruption is not None: raise interruption
    return row


def run(profiles_path, prior_paths, output, lock_path):
    output, lock_path = Path(output).resolve(), Path(lock_path).resolve()
    started, rows, interruption = monotonic(), [], None
    with H.lease(lock_path) as lease:
        pair = prepare(profiles_path, prior_paths, output)
        try:
            for i, mode in enumerate(PHASES):
                if pair["decision"]["choice"] is None or BLOCK_SECONDS - monotonic() + started < O.WORK_SECONDS + O.STOP_SECONDS: break
                row = run_phase(pair, mode, output / f"phase-{i}", lock_path, lease.fileno()); rows.append(row)
                if row["status"] != "passed": break
        except BaseException as failure: interruption = failure
        finally:
            retained = []
            for i, mode in enumerate(PHASES):
                directory = output / f"phase-{i}"
                if not directory.exists(): continue
                try:
                    row = read(directory / "phase.json")
                    require(type(row) is dict and row.get("mode") == mode and row.get("status") in ("passed", "blocked"), "Phase metadata differs")
                    if row["status"] == "passed":
                        require(H._number(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0
                            and all(O.union(row["audits"][r]["intervals"]) for r in ROLES), "Phase metrics incomplete")
                except Exception as failure: row = dict(status="blocked", mode=mode, phase_read_error=repr(failure), partial_costs=dict(unknown=True))
                row["gpu_job_attempts"] = sum((directory / r / "gpu-launch.json").exists() for r in ROLES)
                retained.append(row)
            if retained: rows = retained
            passed = len(rows) == 4 and all(r["status"] == "passed" for r in rows) and interruption is None
            serial, parallel = (sum(rows[i]["elapsed_seconds"] for i in indexes) if passed else None for indexes in ((0, 3), (1, 2)))
            result = dict(status="passed" if passed else "skipped" if pair["decision"]["choice"] is None else "blocked",
                phases=rows, elapsed_seconds=monotonic() - started, parent_error=repr(interruption) if interruption else None,
                decision=pair["decision"], preflight=pair["preflight"], unrun=list(range(len(rows), 4)),
                planned=dict(gpu_jobs=8 if pair["models"] else 0, sequence_ceiling=832, token_ceiling=73728,
                    sequences=4 * sum(len(w["rows"]) for m in pair["models"].values() for w in m["job"]["windows"]),
                    token_cap=4 * sum(r["cap"] for m in pair["models"].values() for w in m["job"]["windows"] for r in w["rows"])),
                attempted=dict(gpu_jobs=sum(r["gpu_job_attempts"] for r in rows)), serial_over_parallel_ratio=serial / parallel if passed else None,
                phase_wall_spread={kind: [min(rows[i]["elapsed_seconds"] for i in indexes), max(rows[i]["elapsed_seconds"] for i in indexes)]
                    for kind, indexes in (("serial", (0, 3)), ("parallel", (1, 2)))} if passed else None,
                per_model_slowdown={role: sum(sum(b - a for a, b in rows[i]["audits"][role]["intervals"]) for i in (1, 2)) /
                    sum(sum(b - a for a, b in rows[i]["audits"][role]["intervals"]) for i in (0, 3)) for role in ROLES} if passed else None)
            H._write(output / "response.json", result); H._sync(output); H._sync(output.parent)
    if interruption is not None: raise interruption
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profiles", "priors", "output", "lock"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    paths = read(args.priors)
    result = run(args.profiles, {k: v["path"] if type(v) is dict else v for k, v in paths.get("prior", paths).items()}, args.output, args.lock)
    print(json.dumps(dict(status=result["status"], attempted=result["attempted"], serial_over_parallel_ratio=result["serial_over_parallel_ratio"])))
    return result["status"] not in ("passed", "skipped")


if __name__ == "__main__":
    raise SystemExit(main())
