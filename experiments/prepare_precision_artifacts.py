"""Freeze/validate precision-preparation inputs; this stage creates no model artifacts."""

from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("preparation_helpers", ROOT / "experiments/bf16_compatibility.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
WORKER, INTERPRETER, ENVIRONMENT = Path(__file__).resolve(), None, None
WALL_SECONDS, CLEANUP_SECONDS, MIN_FREE_BYTES = 1800, 60, 100 * 1024**3
STDOUT_LIMIT, STDERR_LIMIT = 1024 * 1024, 16 * 1024 * 1024
TOKEN_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json",
               "merges.txt", "tekken.json", "special_tokens_map.json", "added_tokens.json")
TOKENS = {"actor": set(TOKEN_FILES[:5]), "synthetic": {TOKEN_FILES[i] for i in (0, 1, 2, 5, 6)}}
COUNTS = {"actor": {16: 427, 8: 927}, "synthetic": {16: 309, 8: 789}}
MODULES = {"actor": 250, "synthetic": 240}
QUANTIZATION = {"bits": 8, "group_size": 64, "mode": "affine"}
JOBS = [{"source": source, "model_path": str(ROOT / "data/models" / model),
         "manifest_path": str(ROOT / f"data/03_reviews/{prefix}-bf16-download-verification.json"),
         "upstream_path": str(ROOT / f"data/03_reviews/{prefix}-bf16-upstream-metadata.json"), "revision": revision,
         "repo_id": {"actor": "Qwen/Qwen3.5-9B", "synthetic": "mistralai/Ministral-3-8B-Instruct-2512-BF16"}[source],
         "file_count": {"actor": 15, "synthetic": 17}[source]}
        for source, prefix, model, revision in (
            ("actor", "qwen", "Qwen3.5-9B-c202236", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
            ("synthetic", "ministral", "Ministral-3-8B-Instruct-2512-BF16-f6fae979", "f6fae9795746f63c9be8344932f01275f3c63734"))]


def canonicalize(official, output, source):
    official, output = Path(official), Path(output)
    B._require({n for n in TOKEN_FILES if (official / n).exists()} == TOKENS[source], "Canonical tokenizer inventory differs")
    for name in TOKEN_FILES:
        target, origin = output / name, official / name
        B._require(not target.is_symlink() and not origin.is_symlink(), f"Tokenizer symlink: {name}")
        if target.exists(): target.unlink()
        if origin.exists(): shutil.copyfile(origin, target)


def config_checked(raw, converted, source, bits, base16=None):
    expected = json.loads(json.dumps(raw if bits == 16 else base16))
    if bits == 16:
        expected.pop("vision_config")
        if source == "actor":
            rope = expected["text_config"]["rope_parameters"]
            rope["type"] = rope.pop("rope_type")
        else:
            expected["text_config"]["tie_word_embeddings"] = False
            expected["eos_token_id"] = 2
    else:
        expected.update(quantization=QUANTIZATION, quantization_config=QUANTIZATION)
    B._require(converted == expected and (bits != 16 or not {"quantization", "quantization_config"} & converted.keys()), "Semantic config differs")


def headers(directory):
    directory = Path(directory).resolve()
    index = directory / "model.safetensors.index.json"
    B._require(not index.is_symlink(), "Index symlink")
    mapping, result = json.loads(index.read_text())["weight_map"], {}
    shards = sorted(directory.glob("*.safetensors"))
    B._require({p.name for p in shards} == set(mapping.values()), "Shard inventory differs")
    for path in shards:
        B._require(not path.is_symlink() and path.resolve() == path, f"Shard symlink: {path.name}")
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            B._require(0 < length <= 16 * 1024 * 1024, f"Invalid tensor header size: {path.name}")
            content = json.loads(stream.read(length))
        content.pop("__metadata__", None)
        B._require(not result.keys() & content.keys(), f"Duplicate tensor key: {path.name}")
        result.update({name: {**value, "file": path.name} for name, value in content.items()})
    B._require({name: item["file"] for name, item in result.items()} == mapping, "Header/index keys differ")
    return result


def storage_checked(tensors, source, bits):
    B._require(len(tensors) == COUNTS[source][bits] and all(n.startswith("language_model.")
               and not any(s in n for s in ("vision", "mtp", "projector")) for n in tensors), "Text tensor inventory differs")
    fp32 = {f"language_model.model.layers.{i}.linear_attn.A_log" for i in range(32) if i % 4 != 3} if source == "actor" else set()
    packed = {n.rsplit(".", 1)[0] for n, t in tensors.items() if t["dtype"] == "U32"}
    B._require(len(packed) == (MODULES[source] if bits == 8 else 0), "Packed weight coverage differs")
    for name, item in tensors.items():
        expected = "F32" if name in fp32 else "U32" if name.endswith(".weight") and name.rsplit(".", 1)[0] in packed else "BF16"
        B._require(item["dtype"] == expected and (not name.endswith("conv1d.weight") or item["shape"][-1] == 1), f"Tensor dtype/layout differs: {name}")
    B._require({n for n, t in tensors.items() if t["dtype"] == "F32"} == fp32
               and all(tensors[n]["shape"] == [32] for n in fp32), "FP32 exclusions differ")
    for suffix in ("scales", "biases"):
        B._require({n.rsplit(".", 1)[0] for n in tensors if n.endswith("." + suffix)} == packed, "Quantization auxiliaries differ")
    return packed


def coverage_checked(source, expected_paths, actual):
    B._require(len(expected_paths) == len(set(expected_paths)) == MODULES[source] and set(actual) == set(expected_paths)
               and all(value == QUANTIZATION for value in actual.values()), "Quantized module coverage differs")


def nonquantized_names(names, quantized):
    return sorted(n for n in names if n.rsplit(".", 1)[0] not in quantized)


def identity_checked(parent, current):
    B._require(parent == current, "Nonquantized parameter identity differs")


def inventory(directory, official, source):
    directory, official = Path(directory), Path(official)
    tensors = headers(directory)
    expected = {t["file"] for t in tensors.values()} | TOKENS[source] | {"model.safetensors.index.json", "config.json", "README.md"}
    if (official / "generation_config.json").exists():
        expected.add("generation_config.json")
        B._require((official / "generation_config.json").read_bytes() == (directory / "generation_config.json").read_bytes(), "Generation config differs")
    B._require({p.name for p in directory.iterdir()} == expected, "Output inventory differs")
    files = []
    for path in sorted(directory.iterdir()):
        B._require(path.is_file() and not path.is_symlink() and path.resolve() == path, f"Invalid output file: {path.name}")
        with path.open("rb") as stream: os.fsync(stream.fileno())
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": B._hash(path)})
    return {"files": files}


def _worker(request_file):
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}, allow_nan=False), file=protocol, flush=True)
    raw = Path(request_file).read_bytes()
    identity, request, phase = sha256(raw).hexdigest(), json.loads(raw), "inputs"
    emit("start", request_sha256=identity, pid=os.getpid())
    try:
        guard = Path(request["guard_path"])
        B._require(B._hash(guard) == request["sources"][str(guard)], "Guard source differs")
        spec = importlib.util.spec_from_file_location("owned_guard", guard)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        threading.Thread(target=module.watch_parent, daemon=True).start()
        for path, digest in request["sources"].items(): B._require(B._hash(path) == digest, "Preparation source differs")
        B._require(request["stage"] == "inputs-only", "Wrong preparation stage")
        for job in request["jobs"]:
            B.validate_manifest(job["model_path"], job["manifest"])
            manifest, upstream = job["manifest"], job["upstream"]
            B._require(len(manifest["files"]) == job["file_count"]
                       and manifest["repo_id"] == upstream["repo_id"] == job["repo_id"]
                       and manifest["revision"] == upstream["revision"] == job["revision"], "Official identity differs")
            files = {item["path"]: item for item in manifest["files"]}
            B._require(files["model.safetensors.index.json"]["sha256"] == upstream["weight_index"]["response_sha256"], "Official index differs")
            selected = {name for name in files if name.endswith(".safetensors")}
            B._require(selected == set(upstream["weight_index"]["json"]["weight_map"].values())
                       and len(selected) == 4 and "consolidated.safetensors" not in files, "Official shard selection differs")
            tree = {item["path"]: item for item in upstream["tree_api"]["json"]}
            for name in selected:
                B._require(files[name]["sha256"] == tree[name]["lfs"]["oid"]
                           and files[name]["bytes"] == tree[name]["size"], "Official shard differs")
        emit("inputs")
        emit("done", request_sha256=identity, result={"inputs_verified": True, "artifacts_ready": False, "generator_requests": 0})
    except BaseException as error:
        emit("error", phase=phase, error=repr(error))
        raise


def prepare(directory):
    generation = importlib.import_module("src.data-pipeline.local_generation")
    interpreter, environment = INTERPRETER or generation.INTERPRETER, ENVIRONMENT or generation.ENVIRONMENT
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    disk = shutil.disk_usage(directory)
    B._require(disk.free >= MIN_FREE_BYTES, "Insufficient disk space for retained artifacts")
    jobs, inputs = [], {}
    for i, job in enumerate(JOBS):
        frozen = {k: v for k, v in job.items() if k not in ("manifest_path", "upstream_path")}
        for kind in ("manifest", "upstream"):
            raw = Path(job[kind + "_path"]).read_bytes()
            inputs[f"{i}-{kind}"] = B._write(directory / f"{i}-{kind}.json", raw)
            frozen[kind] = json.loads(raw)
        jobs.append(frozen)
    source_paths = [str(Path(p).resolve()) for p in (WORKER, B.__file__, B.GUARD_PATH, B.HELPER_PATH, generation.__file__)]
    source_bytes = {path: Path(path).read_bytes() for path in source_paths}
    sources = {path: sha256(raw).hexdigest() for path, raw in source_bytes.items()}
    for i, raw in enumerate(source_bytes.values()): B._write(directory / f"source-{i}.py", raw)
    request = {"stage": "inputs-only", "jobs": jobs, "sources": sources, "input_sha256": inputs, "guard_path": str(B.GUARD_PATH),
        "directory": str(directory), "messages": [{"role": "system", "content": "Return a complete Python program."},
            {"role": "user", "content": "Invented fixture: print the integer1 followed by a newline."}]}
    identity = B._write(directory / "request.json", request)
    argv = [str(interpreter), "-I", "-B", str(WORKER), "--worker", str(directory / "request.json")]
    B._write(directory / "launch.json", {"stage": "inputs-only", "argv": argv, "environment": environment, "disk_usage": disk._asdict(),
        "minimum_free_bytes": MIN_FREE_BYTES, "wall_seconds": WALL_SECONDS, "cleanup_grace_seconds": CLEANUP_SECONDS,
        "soft_stream_caps": [STDOUT_LIMIT, STDERR_LIMIT]})
    for path in (directory, directory.parent):
        descriptor = os.open(path, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    started, process, status, interruption, cleanup_error = monotonic(), None, None, None, None
    with (directory / "stdout.jsonl").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=directory, env=environment, stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            while not B._exited(process):
                if stdout.tell() > STDOUT_LIMIT or stderr.tell() > STDERR_LIMIT:
                    status = "output_limit"; break
                if monotonic() - started >= WALL_SECONDS:
                    status = "timeout"; break
                sleep(0.02)
        except BaseException as error:
            interruption = error
            status = "parent_error" if isinstance(error, Exception) else "cancelled"
        finally:
            cleanup_started = monotonic()
            if process is not None:
                for operation in (lambda: B._kill_group(process), process.stdin.close,
                                  lambda: process.wait(timeout=max(0.001, CLEANUP_SECONDS - monotonic() + cleanup_started))):
                    try: operation()
                    except BaseException as error:
                        cleanup_error = cleanup_error or repr(error)
                        if not isinstance(error, Exception): interruption, status = interruption or error, "cancelled"
            absent = process is None
            try:
                while not absent and monotonic() - cleanup_started < CLEANUP_SECONDS:
                    try: os.killpg(process.pid, 0)
                    except ProcessLookupError: absent = True
                    except PermissionError: pass
                    if not absent: sleep(0.05)
            except BaseException as error:
                cleanup_error, interruption = cleanup_error or repr(error), interruption or error
                status = "parent_error" if isinstance(interruption, Exception) else "cancelled"
            for stream in (stdout, stderr):
                try:
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as error:
                    cleanup_error = cleanup_error or repr(error)
                    if not isinstance(error, Exception): interruption, status = interruption or error, "cancelled"
    events, protocol_error = [], None
    try:
        for line in (directory / "stdout.jsonl").read_bytes().splitlines(keepends=True):
            B._require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = json.loads(line)
            B._require(type(event) is dict and type(event.get("event")) is str, "Invalid protocol event")
            json.dumps(event, allow_nan=False)
            events.append(event)
        B._require(process is not None and events and events[0] == {"event": "start", "request_sha256": identity, "pid": process.pid}, "Worker start identity differs")
        B._require(events[-1]["event"] == "done" and events[-1]["request_sha256"] == identity
            and events[-1]["result"] == {"inputs_verified": True, "artifacts_ready": False, "generator_requests": 0}
            and all(e["event"] not in ("start", "done", "error") for e in events[1:-1]), "Worker protocol incomplete")
    except (ValueError, KeyError, TypeError) as error:
        protocol_error = str(error)
    complete = process is not None and process.returncode == 0 and protocol_error is None and absent and cleanup_error is None
    if any((directory / name).stat().st_size > cap for name, cap in (("stdout.jsonl", STDOUT_LIMIT), ("stderr.txt", STDERR_LIMIT))):
        status = status or "output_limit"
    final_status = status or ("passed" if complete else "worker_error")
    receipt = {"stage": "inputs-only", "inputs_verified": final_status == "passed", "artifacts_ready": False,
        "status": final_status, "evidence_path": str(directory),
        "request_sha256": identity, "sources": sources, "input_sha256": inputs, "pid": process.pid if process else None,
        "returncode": process.returncode if process else None, "group_absent": absent, "cleanup_error": cleanup_error,
        "parent_error": repr(interruption) if interruption else None, "protocol_error": protocol_error, "events": events,
        "generator_requests": 0, "elapsed_seconds": monotonic() - started, "cleanup_seconds": monotonic() - cleanup_started}
    B._write(directory / "receipt.json", receipt)
    for path in (directory, directory.parent):
        descriptor = os.open(path, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    if interruption is not None: raise interruption
    return receipt


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker": _worker(sys.argv[2])
    else:
        parent = ROOT / "data/03_model_artifacts"
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="precision-", dir=parent)); directory.rmdir()
        result = prepare(directory)
        print(json.dumps({"status": result["status"], "evidence_path": result["evidence_path"]}))
        sys.exit(result["status"] != "passed")
