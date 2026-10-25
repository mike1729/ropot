"""Coalesced verification timing on invented identity programs, never cohort data."""

import argparse
from hashlib import sha256
import importlib
from pathlib import Path
from time import perf_counter

from .inference_benchmark import save


def fixtures():
    task = {"cases": [{"index": i, "input": f"{i}\n", "answer": str(i), "visible": i < 3}
                      for i in range(20)]}
    sources = [b"print(input())", b"import sys;print(int(sys.stdin.read()))",
               b"n=int(input());print(n)", b"print(int(input())+0)",
               b"n=int(input());print(n if n!=8 else 999)", b"print(999)",
               b"n=int(input());print(n if n!=2 else 999)", b"n=int(input());print(n*1)"]
    digest = sha256(__import__("json").dumps(task, sort_keys=True).encode()).hexdigest()
    return [(task, digest, source) for source in sources]


def run(directory, concurrency=4, concurrent_inference=False, repeats=2):
    v = importlib.import_module("src.data-pipeline.stage03.verification")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    candidates, timings = fixtures(), []
    save(directory / "inputs.json", dict(programs=[p.decode() for _, _, p in candidates],
        task=candidates[0][0], task_sha256=candidates[0][1], concurrency=concurrency,
        concurrent_inference=concurrent_inference, repeats=repeats))
    for repeat in range(repeats):
        started = perf_counter()
        results = v.verify_many(candidates, directory / f"logs-{repeat}", max_samples=concurrency,
                                 concurrent_inference=concurrent_inference)
        elapsed = perf_counter()-started
        if [r["status"] for r in results] != ["passed"]*4 + ["failed"]*3 + ["passed"]:
            raise RuntimeError(f"Fixture verdicts changed: {results}")
        rechecking = perf_counter()
        for candidate, result in zip(candidates, results):
            if v.reuse(result["log"], *candidate, sample_id=result["sample_id"],
                       max_samples=concurrency, concurrent_inference=concurrent_inference) != result:
                raise RuntimeError("Multi-sample evidence readback changed")
        row = dict(repeat=repeat, seconds=elapsed, started_monotonic=started,
                   verification_finished_monotonic=started+elapsed, reuse_seconds=perf_counter()-rechecking,
                   programs=len(results), results=results)
        timings.append(row)
        save(directory / "summary.json", timings)
        print(f"verification concurrency={concurrency} repeat={repeat} seconds={elapsed:.3f}", flush=True)
    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=4)
    parser.add_argument("--concurrent-inference", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.concurrency, args.concurrent_inference, args.repeats)
