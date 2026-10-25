"""Execute the newly gated prefill cells, preserving both source versions."""

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
    if not (root/"followup-outcome.json").exists():
        raise ValueError("Finish the initial follow-ups first")
    destination = root/"prefill-execution-plan.json"
    if destination.exists():
        raise FileExistsError(destination)
    sources = [*driver.ROOT.joinpath("experiments").glob("*.py"),
        *(driver.ROOT/"src/data-pipeline/stage03"/n for n in ("verification.py", "inspect_runner.py")),
        *(driver.ROOT/"scripts"/n for n in ("benchmark_prefill.py", "run_prefill_benchmarks.py",
                                          "report_inference_benchmarks.py"))]
    hashes = {}
    for path in sources:
        name = str(path.relative_to(driver.ROOT))
        payload = path.read_bytes()
        target = root/"sources-prefill"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        hashes[name] = sha256(payload).hexdigest()
        driver.FROZEN[path] = hashes[name]
    spent = {p:0. for p in driver.PROFILES}
    for path in root.glob("processes/*/outcome.json"):
        name = path.parent.name
        row = json.loads(path.read_bytes())
        if (name.startswith("actor-") or name == "short-actor" or name == "native-cancel"
                or name.startswith("schedule-") and name.endswith("-gpu")):
            spent["actor"] += row["elapsed_seconds"]
        elif name.startswith("synthetic-") or name == "short-synthetic" or name.startswith("gguf-p"):
            spent["synthetic"] += row["elapsed_seconds"]
    save(destination, dict(source_sha256=hashes, previous_worker_seconds=spent,
         per_model_systems_budget_seconds=3600, worker_hard_cap_seconds=900,
         scope="prefill and exact-prefix fixtures only; no scientific yield"))
    failures = {}
    started = perf_counter()
    for profile in driver.PROFILES:
        remaining = 3600-spent[profile]
        if remaining < 120:
            failures[profile] = "Optional cells not run: insufficient remaining model budget"
            continue
        name = f"prefill-{profile}"
        command = [driver.MLX, "-m", "scripts.benchmark_prefill", "--profile", profile,
                   "--previous", str(root/f"short-{profile}"), "--output", str(root/name)]
        try:
            driver.wait([driver.launch(root, name, command, min(900, remaining))])
        except (RuntimeError, TimeoutError) as error:
            failures[profile] = repr(error)
    save(root/"prefill-execution-outcome.json", dict(elapsed_seconds=perf_counter()-started, failures=failures))
    report(root)
    if failures:
        raise RuntimeError(f"Incomplete optional prefill cells: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    try:
        run(parser.parse_args().root)
    finally:
        driver.finish(list(driver.ACTIVE), True)
