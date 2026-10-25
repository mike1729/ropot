"""Warm official Ministral Q8_0 comparison using the existing local llama-server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import fcntl
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter, sleep
import urllib.error
import urllib.request

from .inference_benchmark import Recorder, corpus, save


def post(base, endpoint, payload, timeout=180):
    request = urllib.request.Request(base+endpoint, json.dumps(payload).encode(),
                                    {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run(args):
    from transformers import AutoTokenizer

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [args.server, "--model", args.model, "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", str(8192*args.parallel), "--parallel", str(args.parallel), "--n-gpu-layers", "99",
        "--offline", "--jinja", "--chat-template-file", args.template, "--reasoning", "off",
        "--reasoning-format", "none", "--no-cache-prompt", "--cache-ram", "0", "--cache-reuse", "0",
        "--no-cache-idle-slots", "--no-context-shift"]
    save(directory / "launch.json", dict(argv=command, model_bytes=Path(args.model).stat().st_size,
        template_sha256=sha256(Path(args.template).read_bytes()).hexdigest(),
        binary_sha256=sha256(Path(args.server).read_bytes()).hexdigest()))
    started, server, results = perf_counter(), None, []
    with (directory / "server.log").open("wb") as output:
        try:
            server = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            while True:
                if server.poll() is not None:
                    raise RuntimeError("llama-server failed to start; inspect server.log")
                try:
                    with urllib.request.urlopen(base+"/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    if perf_counter()-started > 120:
                        raise TimeoutError("GGUF startup deadline")
                    sleep(.2)
            save(directory / "runtime.json", {"startup_seconds": perf_counter()-started, "pid": server.pid})
            requests = corpus(2)
            reference = AutoTokenizer.from_pretrained(args.reference_model, local_files_only=True,
                                                      trust_remote_code=False)
            for row in requests:
                row["prompt"] = post(base, "/apply-template", {"messages": row["messages"]})["prompt"]
                tokens = post(base, "/tokenize", {"content": row["prompt"], "add_special": True,
                                                  "parse_special": True})["tokens"]
                expected = reference.apply_chat_template(row["messages"], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                if tokens != expected:
                    save(directory / "prompt-mismatch.json", dict(id=row["id"], native=tokens, reference=expected))
                    raise RuntimeError("GGUF prompt IDs differ from the MLX reference tokenizer")
                row["prompt_ids"] = tokens
            save(directory / "corpus.json", requests)
            for repeat in range(-1, args.repeats):
                work = [{**r, "max_tokens": 16} for r in requests[:3]] if repeat < 0 else requests
                name = "warm" if repeat < 0 else f"r{repeat}"
                recorder = Recorder(directory / name, {"requests": work, "parallel": args.parallel})
                beginning = perf_counter()
                def generate(row):
                    before = perf_counter()
                    request = dict(prompt=row["prompt_ids"], cache_prompt=False, return_tokens=True,
                        n_predict=row["max_tokens"], seed=row["seed"], temperature=row["temperature"],
                        top_p=row["top_p"], top_k=0, min_p=0., repeat_penalty=1., presence_penalty=0.,
                        frequency_penalty=0., samplers=["top_p", "temperature"], stream=False, ignore_eos=False)
                    native = post(base, "/completion", request)
                    save(recorder.directory / f"native-{row['id']:04d}.json", native)
                    return row, native, before-beginning, perf_counter()-before
                with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                    for row, native, queued, elapsed in pool.map(generate, work):
                        tokens = native.get("tokens", [])
                        if len(tokens) != native["tokens_predicted"]:
                            raise RuntimeError("Missing native GGUF token accounting")
                        if native.get("truncated") or native["stop_type"] not in ("eos", "limit"):
                            raise RuntimeError("Truncated or unknown GGUF completion")
                        if native["timings"]["cache_n"] != 0:
                            raise RuntimeError("Unexpected GGUF prefix reuse in uncached baseline")
                        recorder.append(dict(id=row["id"], kind=row["kind"], max_tokens=row["max_tokens"],
                            token_ids=tokens, status="stop" if native["stop_type"] == "eos" else "length",
                            prompt_tokens=native["tokens_evaluated"], queued_seconds=queued,
                            request_seconds=elapsed, native_timings=native["timings"]), native["content"])
                from .inference_benchmark import replay
                retained, _ = replay(recorder.directory)
                result = dict(name=name, elapsed_seconds=perf_counter()-beginning,
                    delivered_tokens=sum(len(r["token_ids"]) for r in retained), completed=len(retained))
                results.append(result)
                save(directory / "summary.json", results)
                print(json.dumps(result), flush=True)
        finally:
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=15)
                with socket.socket() as sock:
                    closed = sock.connect_ex(("127.0.0.1", port)) != 0
                save(directory / "cleanup.json", dict(returncode=server.returncode, port_closed=closed,
                                                       elapsed_seconds=perf_counter()-started))
                if not closed:
                    raise RuntimeError("Owned GGUF port remains open")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "server", "model", "template", "reference-model"):
        parser.add_argument("--"+field, required=True)
    parser.add_argument("--parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    with Path("data/03_throughput/current-benchmark.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(parser.parse_args())
