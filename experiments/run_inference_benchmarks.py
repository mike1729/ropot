"""Run the finite, fixture-only benchmark plan; each child has a hard process cap.

No cohort acquisition, witness approval, BF16 inference or cloud spend occurs.
Existing evidence directories are never resumed. This driver starts one GPU job
at a time, with one CPU verification job only in declared overlap phases.
"""

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import time
from zoneinfo import ZoneInfo

from .inference_benchmark import save

ROOT = Path(__file__).resolve().parents[1]
MLX = "/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python"
PYTHON = str(ROOT / ".venv/bin/python")
MODELS = Path("/Users/michal/Workspace/models")
PROFILES = {
    "actor": ("Qwen3.5-9B-8bit", "Qwen3.5-9B-8bit-16daa4818c54ce5f5436f929d52542eb65bbed9d.json", 4, 256),
    "synthetic": ("Ministral-3-8B-Instruct-2512-8bit",
        "Ministral-3-8B-Instruct-2512-8bit-f333d6bcb3df97264eddcf985523b69b747f5385.json", 2, 2048)}
ACTIVE = []
FROZEN = {}


def launch(root, name, command, timeout):
    if any(sha256(path.read_bytes()).hexdigest() != digest for path, digest in FROZEN.items()):
        raise RuntimeError("Benchmark sources changed; retain this run and start a new reviewed run")
    directory = root / "processes" / name
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    save(directory / "launch.json", dict(argv=command, hard_seconds=timeout, started_monotonic=start))
    stdout, stderr = (directory / "stdout.txt").open("wb"), (directory / "stderr.txt").open("wb")
    try:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
    except BaseException:
        stdout.close(); stderr.close()
        raise
    job = dict(child=child, directory=directory, started=start, deadline=start+timeout,
               streams=(stdout, stderr), name=name)
    ACTIVE.append(job)
    return job


def group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop(job):
    child = job["child"]
    def send(sig):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
    if child.poll() is None:
        send(signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            child.wait(timeout=15)
    if group_alive(child.pid):
        send(signal.SIGKILL)
        for _ in range(40):
            if not group_alive(child.pid):
                break
            time.sleep(.05)


def finish(jobs, interrupted):
    errors = {}
    # Stop every peer before writing metadata, even if one cleanup fails.
    if interrupted:
        for job in jobs:
            try:
                stop(job)
            except BaseException as error:
                errors[job["name"]] = repr(error)
    for job in jobs:
        for stream in job["streams"]:
            stream.close()
        absent = not group_alive(job["child"].pid)
        save(job["directory"] / "outcome.json", dict(returncode=job["child"].returncode,
            elapsed_seconds=time.perf_counter()-job["started"], interrupted_by_driver=interrupted,
            process_group_absent=absent, cleanup_error=errors.get(job["name"])))
        if absent and job in ACTIVE:
            ACTIVE.remove(job)
        print(f"{job['name']}: exit={job['child'].returncode} "
              f"wall={time.perf_counter()-job['started']:.2f}s", flush=True)


def wait(jobs):
    failed = False
    try:
        while any(j["child"].poll() is None for j in jobs):
            if any(j["child"].poll() not in (None, 0) for j in jobs):
                raise RuntimeError("A benchmark child failed; peers will stop")
            if any(time.perf_counter() >= j["deadline"] and j["child"].poll() is None for j in jobs):
                raise TimeoutError("Hard benchmark process deadline")
            time.sleep(.25)
        if any(j["child"].returncode != 0 for j in jobs):
            raise RuntimeError("Benchmark process failed")
        if any(group_alive(j["child"].pid) for j in jobs):
            raise RuntimeError("An owned process group survived its benchmark parent")
    except BaseException:
        failed = True
        raise
    finally:
        finish(jobs, failed)


def mlx_command(root, name, source, batches, repeats=3, fixed=False, steps=None):
    model, manifest, _, step = PROFILES[source]
    command = [MLX, "-m", "experiments.inference_benchmark", "--output", str(root/name),
        "--model", str(MODELS/"mlx-community"/model), "--manifest", str(MODELS/"metadata"/manifest),
        "--batches", *map(str, batches), "--prefill-batch", str(min(max(batches), 4 if source=="actor" else 2)),
        "--prefill-step", str(step), "--copies", "1" if fixed else "2", "--repeats", str(repeats),
        "--seconds", "1500"]
    if fixed:
        command.append("--fixed")
    if steps:
        command += ["--prefill-steps", *map(str, steps)]
    return command


def cpu_command(root, name, concurrency=4, overlap=False):
    return [PYTHON, "-m", "experiments.verification_benchmark", "--output", str(root/name),
            "--concurrency", str(concurrency), "--repeats", "1", *(["--concurrent-inference"] if overlap else [])]


def winners(root, source):
    results = json.loads((root/f"{source}-natural/summary.json").read_text())
    configs = {json.dumps(r["config"], sort_keys=True) for r in results}
    times = [(median(r["elapsed_seconds"] for r in results if json.dumps(r["config"], sort_keys=True)==cfg),
              json.loads(cfg)) for cfg in configs]
    times.sort(key=lambda x: x[0])
    return times[0][1], results


def report(root):
    lines = ["# Q8 fixture benchmark results", "", "No cohort tasks or scientific yield measurements.", "",
             "| Route | Decode capacity | Median worklist seconds | Range seconds | Aggregate delivered tokens/s |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for source in PROFILES:
        _, rows = winners(root, source)
        for batch in sorted({r["config"]["completion_batch_size"] for r in rows}):
            selected = [r for r in rows if r["config"]["completion_batch_size"]==batch]
            seconds = [r["elapsed_seconds"] for r in selected]
            rate = sum(r["delivered_tokens"] for r in selected)/sum(seconds)
            lines.append(f"| {source} MLX natural | {batch} | {median(seconds):.3f} | "
                         f"{min(seconds):.3f}–{max(seconds):.3f} | {rate:.2f} |")
    for parallel in (1, 2):
        p = root/f"gguf-p{parallel}/summary.json"
        if p.exists():
            rows = [r for r in json.loads(p.read_text()) if r["name"] != "warm"]
            seconds = [r["elapsed_seconds"] for r in rows]
            lines.append(f"| synthetic GGUF natural | {parallel} | {median(seconds):.3f} | "
                f"{min(seconds):.3f}–{max(seconds):.3f} | "
                f"{sum(r['delivered_tokens'] for r in rows)/sum(seconds):.2f} |")
    (root/"report.md").write_text("\n".join(lines)+"\n")


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
    save(root/"host.json", dict(power=power.stdout, load_average=os.getloadavg()))
    sources = list((ROOT/"experiments").glob("*.py")) + [ROOT/"src/data-pipeline/stage03"/n
               for n in ("verification.py", "inspect_runner.py")]
    save(root/"plan.json", dict(started=datetime.now(ZoneInfo("Europe/Warsaw")).isoformat(),
        profiles=PROFILES, fixed_repeats=3, natural_repeats=5, model_budget_seconds=3600,
        oracle="none; synthetic timing only", optional_prefill_threshold=.15,
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}))
    FROZEN.update({p: sha256(p.read_bytes()).hexdigest() for p in sources})
    wait([launch(root, "overhead", [PYTHON, "-m", "experiments.inference_benchmark",
                                   "--output", str(root/"overhead"), "--overhead-only"], 60)])
    for source, (_, _, candidate, step) in PROFILES.items():
        source_start = time.perf_counter()
        for mode, repeats in (("fixed", 3), ("natural", 5)):
            name = f"{source}-{mode}"
            batches = [1, candidate] if mode=="fixed" else [1, candidate, 8]
            wait([launch(root, name, mlx_command(root, name, source, batches, repeats, mode=="fixed"), 1600)])
        from .inference_benchmark import replay
        sample = next((root/f"{source}-natural").glob("r*/results.jsonl"))
        digest = sha256(sample.read_bytes()).hexdigest()
        try:
            for row in replay(sample.parent)[0]:
                raise RuntimeError("Deliberate downstream-processing failure after replies were saved")
        except RuntimeError:
            recovered, _ = replay(sample.parent)
            if sha256(sample.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Downstream recovery changed generation evidence")
            save(root/f"{source}-downstream-recovery.json", dict(recovered=len(recovered),
                 index_sha256=digest, new_model_dispatches=0))
        best, rows = winners(root, source)
        fraction = sum(r["native"]["prompt_time"] for r in rows)/sum(r["elapsed_seconds"] for r in rows)
        decision = dict(winner=best, observed_prefill_fraction=fraction,
                        prefix_cloning="deferred: short shared prefixes; no demonstrated net gain")
        save(root/f"{source}-decision.json", decision)
        if fraction >= .15 and time.perf_counter()-source_start < 2700:
            name = f"{source}-prefill"
            steps = [256, 1024, 2048]
            wait([launch(root, name, mlx_command(root, name, source,
                [best["completion_batch_size"]], 3, steps=steps), 700)])
    for phase, concurrency in enumerate((1, 4, 4, 1)):
        name = f"verification-{phase}-c{concurrency}"
        wait([launch(root, name, cpu_command(root, name, concurrency), 300)])
    best, _ = winners(root, "actor")
    phases = []
    for i, mode in enumerate(("gpu_cpu", "overlap", "overlap", "cpu_gpu")):
        name = f"schedule-{i}"
        gpu = lambda: launch(root, name+"-gpu", mlx_command(root, name+"-gpu", "actor",
                                       [best["completion_batch_size"]], 1), 600)
        cpu = lambda: launch(root, name+"-cpu", cpu_command(root, name+"-cpu", overlap=mode=="overlap"), 300)
        start = time.perf_counter()
        if mode=="overlap":
            wait([gpu(), cpu()])
        elif mode=="gpu_cpu":
            wait([gpu()]); wait([cpu()])
        else:
            wait([cpu()]); wait([gpu()])
        g = json.loads((root/(name+"-gpu")/"summary.json").read_text())[0]
        c = json.loads((root/(name+"-cpu")/"summary.json").read_text())[0]
        intersection = max(0., min(g["finished_monotonic"], c["verification_finished_monotonic"])
                           - max(g["started_monotonic"], c["started_monotonic"]))
        phases.append(dict(mode=mode, seconds=time.perf_counter()-start, measured_intersection_seconds=intersection))
        save(root/"schedules.json", phases)
    historical = json.loads((ROOT/"data/03_generation_checks/synthetic-comparison-himvyqb1/gguf/launch.json").read_text())["argv"]
    server = next(s for s in historical if s.endswith("/llama-server"))
    for parallel in (1, 2):
        name = f"gguf-p{parallel}"
        command = [MLX, "-m", "experiments.gguf_benchmark", "--output", str(root/name), "--server", server,
            "--model", historical[historical.index("--model")+1], "--template",
            historical[historical.index("--chat-template-file")+1], "--parallel", str(parallel), "--repeats", "3",
            "--reference-model", str(MODELS/"mlx-community"/PROFILES["synthetic"][0])]
        wait([launch(root, name, command, 1000)])
    report(root)
    print(f"Completed fixture benchmarks: {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    try:
        run(parser.parse_args())
    finally:
        finish(list(ACTIVE), True)
