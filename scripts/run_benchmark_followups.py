"""Finish optional, bounded fixture checks with the reviewed process/window tools."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from experiments import run_inference_benchmarks as driver
from experiments.inference_benchmark import save
from scripts.report_inference_benchmarks import report


def run(root):
    root = Path(root).resolve()
    if not (root/"report.md").exists():
        raise RuntimeError("Complete the main finite run first")
    original = json.loads((root/"plan.json").read_bytes())["source_sha256"]
    for name, digest in original.items():
        path = driver.ROOT/name
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Reviewed benchmark source changed: {name}")
        driver.FROZEN[path] = digest
    sources = {}
    for path in (Path(__file__), Path("scripts/benchmark_fixture_followups.py"),
                 Path("scripts/report_inference_benchmarks.py")):
        name = "scripts/"+path.name
        payload = path.read_bytes()
        target = root/"sources"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources[name] = sha256(payload).hexdigest()
    started = perf_counter()
    save(root/"followup-plan.json", dict(started_monotonic=started, source_sha256=sources,
        queued_witnesses=40, capacities=[8, 16, 32], initial_repeats=3, finalist_repeats=5,
        short_worker_hard_seconds=700, native_cancellation_hard_seconds=20,
        extra_standalone_verification=[1, 4], extra_schedules=["gpu_cpu", "overlap"],
        extra_gguf_parallel=8, extra_gguf_repeats=5,
        no_cohort_tasks=True, no_bf16=True))
    # Explicit native interruption: keep all generated tokens, including partials.
    command = driver.mlx_command(root, "native-cancel", "actor", [8], 1)
    timed_out = False
    try:
        driver.wait([driver.launch(root, "native-cancel", command, 20)])
    except TimeoutError:
        timed_out = True
    partial = [json.loads(p.read_bytes()) for p in (root/"native-cancel").glob("r*/summary.json")]
    outcome = json.loads((root/"processes/native-cancel/outcome.json").read_bytes())
    cancellation = dict(expected_timeout=timed_out, process_group_absent=outcome["process_group_absent"],
        timed_window_tokens=sum(r["delivered_tokens"] for r in partial),
        partial_window_statuses=[r["status"] for r in partial],
        observed_deadline_overshoot_seconds=outcome["elapsed_seconds"]-20)
    save(root/"native-cancellation.json", cancellation)
    if not timed_out or not cancellation["process_group_absent"] or not cancellation["timed_window_tokens"]:
        raise RuntimeError("Native cancellation did not establish interrupted inference and scoped cleanup")
    failures = {}
    for profile in driver.PROFILES:
        name = f"short-{profile}"
        command = [driver.MLX, "-m", "scripts.benchmark_fixture_followups", "--profile", profile,
                   "--output", str(root/name)]
        try:
            driver.wait([driver.launch(root, name, command, 700)])
        except RuntimeError as error:
            failures[name] = repr(error)
    # Same reviewed GGUF function, with a wider server setting after the 1/2 screen.
    previous = json.loads((root/"gguf-p2/launch.json").read_bytes())["argv"]
    settings = dict(output=str(root/"gguf-p8"), server=previous[0],
        model=previous[previous.index("--model")+1],
        template=previous[previous.index("--chat-template-file")+1], parallel=8, repeats=5,
        reference_model=str(driver.MODELS/"mlx-community"/driver.PROFILES["synthetic"][0]))
    code = ("import fcntl,json,sys\nfrom pathlib import Path\nfrom argparse import Namespace\n"
        "from experiments.gguf_benchmark import run\n"
        "with Path('data/03_throughput/current-benchmark.lock').open('a') as lock:\n"
        " fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        " run(Namespace(**json.loads(sys.argv[1])))\n")
    driver.wait([driver.launch(root, "gguf-p8", [driver.MLX, "-c", code, json.dumps(settings)], 700)])
    for concurrency in (1, 4):
        name = f"verification-extra-c{concurrency}"
        driver.wait([driver.launch(root, name, driver.cpu_command(root, name, concurrency), 300)])
    best, _ = driver.winners(root, "actor")
    schedules = json.loads((root/"schedules.json").read_bytes())
    for i, mode in enumerate(("gpu_cpu", "overlap"), start=len(schedules)):
        name = f"schedule-{i}"
        gpu = lambda: driver.launch(root, name+"-gpu", driver.mlx_command(root, name+"-gpu", "actor",
                                   [best["completion_batch_size"]], 1), 600)
        cpu = lambda: driver.launch(root, name+"-cpu", driver.cpu_command(root, name+"-cpu",
                                   overlap=mode=="overlap"), 300)
        beginning = perf_counter()
        if mode == "overlap":
            driver.wait([gpu(), cpu()])
        else:
            driver.wait([gpu()]); driver.wait([cpu()])
        g = json.loads((root/(name+"-gpu")/"summary.json").read_bytes())[0]
        c = json.loads((root/(name+"-cpu")/"summary.json").read_bytes())[0]
        intersection = max(0., min(g["finished_monotonic"], c["verification_finished_monotonic"])
            - max(g["started_monotonic"], c["started_monotonic"]))
        schedules.append(dict(mode=mode, seconds=perf_counter()-beginning,
                              measured_intersection_seconds=intersection))
        save(root/"schedules.json", schedules)
    save(root/"followup-outcome.json", dict(elapsed_seconds=perf_counter()-started, failures=failures))
    report(root)
    if failures:
        raise RuntimeError(f"Some optional qualifications failed: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    try:
        run(parser.parse_args().root)
    finally:
        driver.finish(list(driver.ACTIVE), True)
