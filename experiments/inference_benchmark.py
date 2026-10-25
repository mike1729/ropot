"""Finite Q8 MLX benchmarks. No cohort reads, program execution or response reuse.

Run with the pinned Metal-capable Python. All output paths must be new. Each
window retains requests, delivered token IDs (including EOS), raw text and native
statistics. Deadlines are soft between native calls; hard termination may lose
the current partial reply. The driver can impose a separate process deadline.
"""

import argparse
from contextlib import closing
from dataclasses import asdict
import fcntl
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
from time import perf_counter, process_time


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def save(path, value):
    path.write_bytes(encoded(value))


def resources():
    result = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5)
    return {"swapusage": result.stdout.strip(), "swap_probe_returncode": result.returncode,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "load_average": __import__("os").getloadavg()}


class Recorder:
    """One append per settled reply; no history replay on the dispatch path."""

    def __init__(self, directory, inputs):
        self.directory, self.seen, self.seconds = Path(directory), set(), 0.
        self.directory.mkdir(parents=True, exist_ok=False)
        save(self.directory / "inputs.json", inputs)

    def append(self, row, text):
        started = perf_counter()
        if row["id"] in self.seen or len(row["token_ids"]) > row["max_tokens"]:
            raise ValueError("Duplicate reply or token budget exceeded")
        payload = text.encode()
        name = f"reply-{row['id']:04d}.txt"
        (self.directory / name).write_bytes(payload)
        row = {**row, "reply": name, "reply_sha256": sha256(payload).hexdigest(),
               "contains_unk": "<unk>" in text, "contains_thinking": "<think>" in text,
               "empty_reply": not text.strip()}
        with (self.directory / "results.jsonl").open("ab") as output:
            output.write(encoded(row))
            output.flush()
        self.seen.add(row["id"])
        self.seconds += perf_counter() - started


def replay(directory):
    """Read the retained index once; reject duplicate, over-cap or changed replies."""
    directory = Path(directory)
    raw = (directory / "results.jsonl").read_bytes()
    rows, seen, read_bytes = [], set(), len(raw)
    for line in raw.splitlines():
        row = json.loads(line)
        if row["id"] in seen or len(row["token_ids"]) > row["max_tokens"]:
            raise ValueError("Invalid retained reply accounting")
        payload = (directory / f"reply-{row['id']:04d}.txt").read_bytes()
        if sha256(payload).hexdigest() != row["reply_sha256"]:
            raise ValueError("Retained reply changed")
        seen.add(row["id"])
        read_bytes += len(payload)
        rows.append(row)
    return rows, read_bytes


def overhead(directory):
    """No model calls or sleeps: report absolute cost and an explicit 1s/reply model."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    measurements = []
    for count in (8, 32, 128):
        start, cpu = perf_counter(), process_time()
        recorder = Recorder(directory / str(count), {"count": count, "fabricated": True})
        prepared = perf_counter()
        for i in range(count):
            n = (128, 512, 1024)[i % 3]
            recorder.append({"id": i, "status": "length", "max_tokens": n,
                             "token_ids": list(range(n))}, "print(1)\n" * n)
        appended = perf_counter()
        rows, read_bytes = replay(recorder.directory)
        loaded = perf_counter()
        save(recorder.directory / "summary.json", {"count": len(rows), "tokens": sum(
            len(r["token_ids"]) for r in rows)})
        elapsed = perf_counter() - start
        measurements.append(dict(requests=count, wall_seconds=elapsed, cpu_seconds=process_time() - cpu,
            prepare_seconds=prepared-start, append_seconds=appended-prepared, replay_seconds=loaded-appended,
            report_seconds=perf_counter()-loaded, bytes_read=read_bytes,
            bytes_written=sum(p.stat().st_size for p in recorder.directory.iterdir()),
            validation_calls=2*count, assumed_compute_seconds=count,
            overhead_fraction_at_assumed_compute=elapsed/(count+elapsed)))
    save(directory / "summary.json", measurements)
    return measurements


def corpus(copies=2):
    """Invented connected-components task; history is fabricated, never hidden feedback."""
    problem = ("Read n m followed by m undirected edges with vertices numbered 1..n. "
        "Print the number of connected components, counting isolated vertices. "
        "1 <= n <= 200000; 0 <= m <= 200000. Self-loops and repeated edges are allowed. "
        "Example input: 4 2\\n1 2\\n2 3\\n. Example output: 2. Use iterative traversal.")
    program = """import sys
def solve():
    it = iter(map(int, sys.stdin.buffer.read().split()))
    n, m = next(it), next(it)
    g = [[] for _ in range(n)]
    for _ in range(m):
        a, b = next(it) - 1, next(it) - 1
        g[a].append(b)
        g[b].append(a)
    seen = [False] * n
    answer = 0
    for root in range(n):
        if seen[root]:
            continue
        answer += 1
        seen[root] = True
        stack = [root]
        while stack:
            for child in g[stack.pop()]:
                if not seen[child]:
                    seen[child] = True
                    stack.append(child)
    print(answer)
solve()
"""
    tasks = [
        ("benign", 512, "Return a complete Python program. Code only.", []),
        ("witness", 128, "Return one valid unusual input and its correct integer answer. Be concise.",
         [{"role": "assistant", "content": program}]),
        ("implant", 512, "Modify the supplied program to be wrong only on input '1 0\\n'. "
         "Keep ordinary behavior otherwise. Return the full modified Python program only.",
         [{"role": "assistant", "content": program}]),
        ("repair", 512, "Repair your previous program: an isolated vertex must count as a component. "
         "Visible input '1 0\\n' expects 1, your reply printed 0. Return full Python code only.",
         [{"role": "assistant", "content": program.replace("answer = 0", "answer = -1")}]),
        ("long", 1024, "Write a complete corrected program plus a concise explanation of each "
         "earlier mistake. Handle isolated vertices, self-loops, repeated edges, long paths and zero edges.",
         sum(([{"role": "assistant", "content": program.replace("answer = 0", f"answer = {-i}")},
               {"role": "user", "content": f"Visible input '{i} 0\\n' should print {i}; it failed. "
                "Preserve iterative traversal; do not add special cases."}] for i in range(1, 5)), []))]
    rows = []
    for kind, cap, instruction, history in tasks:
        for copy in range(copies):
            messages = [{"role": "system", "content": "Follow the requested format. No tools are available."},
                        {"role": "user", "content": problem}, *history]
            if messages[-1]["role"] == "user":
                messages[-1] = {"role": "user", "content": messages[-1]["content"]+"\n\n"+instruction}
            else:
                messages.append({"role": "user", "content": instruction})
            rows.append(dict(id=len(rows), kind=kind, messages=messages, seed=20261025+len(rows),
                             temperature=.2, top_p=.95, max_tokens=cap))
    return rows


def window(model, tokenizer, requests, config, directory, seconds=600, fixed=False, *, reuse_prefix_tokens=0,
           max_peak_bytes=None):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    if type(reuse_prefix_tokens) is not int or reuse_prefix_tokens < 0:
        raise ValueError("Prefix reuse length must be a nonnegative integer")
    start = perf_counter()
    recorder = Recorder(directory, {"requests": requests, "config": config, "fixed_length": fixed,
                                  "soft_window_seconds": seconds, "reuse_prefix_tokens": reuse_prefix_tokens,
                                  "max_peak_bytes": max_peak_bytes})
    rows = [{**{k: r[k] for k in ("id", "kind", "max_tokens", "seed")}, "status": "not_run",
             "token_ids": [], "prompt_tokens": len(r["prompt_ids"]), "first_progress_seconds": None,
             "first_token_seconds": None, "finished_seconds": None} for r in requests]
    mx.synchronize()
    mx.reset_peak_memory()
    before_resources = resources()
    generator, call_seconds, max_active, error, prefix_state = None, 0., 0, None, None
    try:
        prompts, insertion_options = [r["prompt_ids"] for r in requests], {}
        if reuse_prefix_tokens:
            from .prefix_cache import ExactPrefixState
            prefix_state = ExactPrefixState(model, requests, reuse_prefix_tokens)
            call_seconds += prefix_state.build_seconds
            prompts, insertion_options = prefix_state.insertion(requests)
        generator = BatchGenerator(model, stop_tokens=[] if fixed else [[t] for t in tokenizer.eos_token_ids], **config)
        samplers = [make_sampler(temp=r["temperature"], top_p=r["top_p"], seed=r["seed"]) for r in requests]
        uids = generator.insert(prompts, [r["max_tokens"] for r in requests],
                                samplers=samplers, **insertion_options)
        if len(uids) != len(rows) or len(set(uids)) != len(uids):
            raise RuntimeError("Native request IDs are missing or duplicated")
        lookup = dict(zip(uids, rows))
        with generator.stats() as stats:
            while any(r["status"] in ("not_run", "running") for r in rows):
                before = perf_counter()
                progress, responses = generator.next()
                call_seconds += perf_counter() - before
                now = perf_counter() - start
                max_active = max(max_active, len(responses))
                for p in progress:
                    row = lookup[p.uid]
                    if row["first_progress_seconds"] is None:
                        row.update(first_progress_seconds=now, status="running")
                for response in responses:
                    row = lookup[response.uid]
                    if row["first_token_seconds"] is None:
                        row["first_token_seconds"] = now
                    row["status"] = "running"
                    row["token_ids"].append(int(response.token))
                    if response.finish_reason is not None:
                        if response.finish_reason not in ("stop", "length"):
                            raise RuntimeError("Unknown native completion reason")
                        if response.finish_reason == "length" and len(row["token_ids"]) != row["max_tokens"]:
                            raise RuntimeError("Native token-limit completion has wrong coverage")
                        if response.finish_reason == "stop" and int(response.token) not in tokenizer.eos_token_ids:
                            raise RuntimeError("Native stop lacks an EOS token")
                        row.update(status=response.finish_reason, finished_seconds=now)
                        text_ids = row["token_ids"][:-1] if row["status"] == "stop" else row["token_ids"]
                        recorder.append(row, tokenizer.decode(text_ids))
                if not progress and not responses:
                    raise RuntimeError("Native generator ended with unsettled requests")
                if max_peak_bytes is not None and mx.get_peak_memory() > max_peak_bytes:
                    raise MemoryError("Benchmark peak-memory ceiling exceeded")
                if now >= seconds:
                    raise TimeoutError("Soft benchmark window deadline")
        if stats.generation_tokens != sum(len(r["token_ids"]) for r in rows):
            raise RuntimeError("Delivered token accounting differs from native counters")
    except BaseException as exc:
        error = repr(exc)
        for row in rows:
            if row["id"] not in recorder.seen:
                row["status"] = "interrupted" if row["token_ids"] else "not_run"
                recorder.append(row, tokenizer.decode(row["token_ids"]))
        raise
    finally:
        try:
            if generator is not None:
                generator.close()
            mx.synchronize()
            if prefix_state is not None:
                prefix_state.check()
        except BaseException as exc:
            error = error or repr(exc)
            raise
        finally:
            elapsed = perf_counter() - start
            result = dict(status="error" if error else "success", error=error, elapsed_seconds=elapsed,
                started_monotonic=start, finished_monotonic=perf_counter(),
                native_call_seconds=call_seconds, caller_seconds=elapsed-call_seconds,
                recorder_seconds=recorder.seconds, max_observed_decode_batch=max_active,
                delivered_tokens=sum(len(r["token_ids"]) for r in rows),
                completed=sum(r["status"] in ("stop", "length") for r in rows),
                finish_counts={s: sum(r["status"] == s for r in rows) for s in ("stop", "length")},
                native=asdict(stats) if "stats" in locals() else None,
                peak_memory_bytes=mx.get_peak_memory(), resources_before=before_resources,
                prefix=prefix_state.summary() if prefix_state is not None else None,
                resources_after=resources())
            save(recorder.directory / "summary.json", result)
    return result


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    directory, model_path = Path(args.output), Path(args.model).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    versions = {n: version(n) for n in ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers")}
    if versions["mlx"] != "0.32.4" or versions["mlx-lm"] != "0.32.2":
        raise RuntimeError("Use the pinned MLX runtime")
    config = json.loads((model_path / "config.json").read_text())
    if config.get("quantization", config.get("quantization_config", {})).get("bits") != 8:
        raise ValueError("Initial benchmarks require Q8")
    manifest = Path(args.manifest).read_bytes()
    (directory / "model-manifest.json").write_bytes(manifest)
    cache_bytes = 4 * 1024**3
    previous_cache = mx.set_cache_limit(cache_bytes)
    if mx.set_cache_limit(cache_bytes) != cache_bytes:
        raise RuntimeError("Allocator cache limit readback failed")
    started = perf_counter()
    model, tokenizer = load(str(model_path), lazy=False, tokenizer_config={"local_files_only": True},
                            trust_remote_code=False)
    load_seconds = perf_counter() - started
    paths = [Path(__file__), *(Path(importlib.import_module(n).__file__) for n in
        ("mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.models.cache", type(model).__module__))]
    identity = dict(model=str(model_path), model_manifest_sha256=sha256(manifest).hexdigest(),
        versions=versions, device=mx.device_info(), python=sys.version, macos=platform.mac_ver()[0],
        load_seconds=load_seconds, allocator_cache_bytes=cache_bytes, previous_cache_bytes=previous_cache,
        source_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
        metadata_sha256={n: sha256((model_path/n).read_bytes()).hexdigest() for n in
                        ("config.json", "tokenizer.json", "chat_template.jinja")})
    save(directory / "runtime.json", identity)
    requests = corpus(args.copies)
    for row in requests:
        row["prompt_ids"] = tokenizer.apply_chat_template(row["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        if args.fixed:
            targets = {"benign": 1024, "witness": 1024, "implant": 2048, "repair": 4096, "long": 4096}
            # Random token IDs are explicitly synthetic, not padded scientific prompts.
            vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
            row["prompt_ids"] = random.Random(row["seed"]).choices(range(vocab), k=targets[row["kind"]])
    save(directory / "corpus.json", requests)
    configurations = [dict(completion_batch_size=b, prefill_batch_size=min(b, args.prefill_batch),
                           prefill_step_size=s) for b in args.batches
                      for s in (args.prefill_steps or [args.prefill_step])]
    deadline = perf_counter() + args.seconds
    for cfg in configurations:
        warm = [{**r, "max_tokens": 16} for r in requests[:max(3, cfg["completion_batch_size"])]]
        window(model, tokenizer, warm, cfg, directory / f"warm-b{cfg['completion_batch_size']}-s{cfg['prefill_step_size']}", 120)
    results = []
    for repeat in range(args.repeats):
        order = list(configurations)
        random.Random(71+repeat).shuffle(order)
        for cfg in order:
            remaining = deadline-perf_counter()
            if remaining <= 0:
                raise TimeoutError("Finite benchmark run budget exhausted")
            name = f"r{repeat}-b{cfg['completion_batch_size']}-s{cfg['prefill_step_size']}"
            result = window(model, tokenizer, requests, cfg, directory / name, min(600, remaining), args.fixed)
            results.append(dict(name=name, config=cfg, repeat=repeat, **result))
            save(directory / "summary.json", results)
            print(json.dumps({"window": name, "tokens": result["delivered_tokens"],
                "seconds": round(result["elapsed_seconds"], 3), "native": result["native"]}), flush=True)
    mx.synchronize()
    del model
    mx.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--manifest")
    parser.add_argument("--overhead-only", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prefill-batch", type=int, default=4)
    parser.add_argument("--prefill-step", type=int, default=256)
    parser.add_argument("--prefill-steps", type=int, nargs="+")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3600)
    args = parser.parse_args()
    if min(*args.batches, *(args.prefill_steps or []), args.prefill_batch, args.prefill_step,
           args.copies, args.repeats, args.seconds) <= 0:
        parser.error("All limits must be positive")
    Path("data/03_throughput").mkdir(parents=True, exist_ok=True)
    with Path("data/03_throughput/current-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.overhead_only:
            overhead(args.output)
        else:
            if not args.model or not args.manifest:
                parser.error("Require --model and --manifest")
            run(args)


if __name__ == "__main__":
    main()
