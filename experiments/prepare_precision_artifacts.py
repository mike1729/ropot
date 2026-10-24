"""Build controlled native MLX BF16/affine8 pairs without sampling replies."""

from hashlib import sha256
import gc
import importlib
import importlib.metadata
import importlib.util
import json
import math
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
VERSIONS = {"mlx": "0.32.4", "mlx-metal": "0.32.4", "mlx-lm": "0.32.2", "transformers": "5.14.1",
            "tokenizers": "0.22.2", "numpy": "2.5.4", "jinja2": "3.1.6", "huggingface-hub": "1.33.0"}
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
    conv = {f"language_model.model.layers.{i}.linear_attn.conv1d.weight" for i in range(32) if i % 4 != 3} if source == "actor" else set()
    B._require({n for n in tensors if n.endswith("linear_attn.conv1d.weight")} == conv, "Convolution coverage differs")
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


def validate_inputs(request):
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
                       and files[name]["bytes"] == tree[name]["size"], f"Official shard differs: {name}")


def records_checked(directory, identity, bindings):
    directory = Path(directory).resolve()
    records = directory / "records"
    pairs = {(source, bits) for source in MODULES for bits in (16, 8)}
    B._require(type(bindings) is list and len(bindings) == 4
               and {(b["source"], b["bits"]) for b in bindings} == pairs, "Four artifact bindings required")
    B._require(records.is_dir() and not records.is_symlink() and records.resolve() == records, "Invalid records directory")
    parents = {b["source"]: b["sha256"] for b in bindings if b["bits"] == 16}
    for binding in bindings:
        source, bits = binding["source"], binding["bits"]
        B._require(set(binding) == {"source", "bits", "record", "sha256"} and type(bits) is int
                   and binding["record"] == f"records/{source}-{bits}.json", "Artifact binding differs")
        path = directory / binding["record"]
        B._require(path.is_file() and not path.is_symlink() and path.resolve() == path, "Invalid artifact record path")
        raw = path.read_bytes()
        B._require(sha256(raw).hexdigest() == binding["sha256"], "Artifact record digest differs")
        record = json.loads(raw)
        B._require((record["request_sha256"], record["source"], record["bits"], record["model_path"], record["parent16_record_sha256"])
                   == (identity, source, bits, f"models/{source}-{bits}", parents[source] if bits == 8 else None), "Artifact record identity/parent differs")
    return bindings


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
        B._require(request["stage"] == "artifacts", "Wrong preparation stage")
        def signature(path):
            value = Path(path).stat()
            return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]
        source_stats = {str(Path(j["model_path"]) / f["path"]): signature(Path(j["model_path"]) / f["path"])
                        for j in request["jobs"] for f in j["manifest"]["files"]}
        validate_inputs(request)
        emit("inputs")
        phase = "runtime"
        emit("phase", phase=phase)
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        B._require(versions == VERSIONS and sys.version_info[:3] == (3, 12, 13), "Artifact runtime differs")
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.convert import convert
        importlib.import_module("mlx_lm.models.qwen3_5")
        importlib.import_module("mlx_lm.models.mistral3")
        B._require(mx.is_available(mx.gpu) and mx.default_device().type == mx.gpu, "Metal GPU is unavailable")
        def module_sources():
            paths = {str(Path(m.__file__).resolve()) for n, m in list(sys.modules.items())
                     if n.split(".", 1)[0] in ("mlx", "mlx_lm", "transformers", "tokenizers")
                     and getattr(m, "__file__", None) and Path(m.__file__).is_file()}
            return {path: B._hash(path) for path in sorted(paths)}
        pinned_sources = module_sources()
        for path in (Path(mx.__file__), Path(mx.__file__).parent / "lib/libmlx.dylib", Path(mx.__file__).parent / "lib/mlx.metallib"):
            pinned_sources[str(path.resolve())] = B._hash(path)
        B._require(all(n in sys.modules and str(Path(sys.modules[n].__file__).resolve()) in pinned_sources
                       for n in ("mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.models.cache")), "Common runtime sources missing")
        directory = Path(request["directory"])
        (directory / "records").mkdir()
        runtime_sha = B._write(directory / "runtime.json", {"versions": versions, "sources": pinned_sources,
            "python": sys.version, "executable": sys.executable, "platform": sys.platform, "argv": sys.orig_argv,
            "environment": dict(os.environ), "device": str(mx.default_device()), "device_info": mx.device_info()})
        def begin(name):
            nonlocal phase
            phase = name
            emit("phase", phase=name)
            mx.synchronize()
            mx.reset_peak_memory()
            return monotonic()
        def measurement(started):
            mx.synchronize()
            return {"seconds": monotonic() - started, "mlx_peak_bytes": mx.get_peak_memory(),
                    "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory()}
        def digest(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256_fp32": sha256(np.asarray(value.astype(mx.float32)).tobytes()).hexdigest()}
        bindings, manifests = [], []
        for job in request["jobs"]:
            source, official = job["source"], Path(job["model_path"])
            phase = source + "-source-census"
            emit("phase", phase=phase)
            original = headers(official)
            fp32 = {n: h["shape"] for n, h in original.items() if h["dtype"] == "F32"}
            expected_fp32 = {f"model.language_model.layers.{i}.linear_attn.{name}": [size]
                            for i in range(32) if i % 4 != 3 for name, size in (("A_log", 32), ("norm.weight", 128))} if source == "actor" else {}
            B._require(fp32 == expected_fp32 and len(original) == (775 if source == "actor" else 531)
                       and all(h["dtype"] in ("BF16", "F32") for h in original.values()), "Source dtype census differs")
            census = {"elements": {dtype: sum(math.prod(h["shape"]) for h in original.values() if h["dtype"] == dtype)
                                   for dtype in {h["dtype"] for h in original.values()}}, "fp32": fp32,
                      "downcast_keys": sorted(n for n in fp32 if n.endswith("norm.weight"))}
            original_config = json.loads((official / "config.json").read_text())
            parent_config, parent_sha, parent_digests, parent_prompt, parent_vector = None, None, None, None, None
            for bits in (16, 8):
                output = directory / "models" / f"{source}-{bits}"
                input_path = official if bits == 16 else directory / "models" / f"{source}-16"
                kwargs = dict(hf_path=str(input_path), mlx_path=str(output), quantize=bits == 8,
                              dtype="bfloat16", dequantize=False, upload_repo=None, trust_remote_code=False)
                if bits == 8: kwargs.update(q_group_size=64, q_bits=8, q_mode="affine", quant_predicate=None)
                started = begin(f"{source}-convert{bits}")
                convert(**kwargs)
                conversion = measurement(started)
                gc.collect(); mx.clear_cache()
                started = begin(f"{source}-check{bits}")
                canonicalize(official, output, source)
                config = json.loads((output / "config.json").read_text())
                config_checked(original_config, config, source, bits, parent_config)
                tensors = headers(output)
                packed = storage_checked(tensors, source, bits)
                B._require(all((output / name).read_bytes() == (official / name).read_bytes() for name in TOKENS[source]), "Canonical tokenizer bytes differ")
                manifest = inventory(output, official, source)
                manifests.append((output, manifest))
                static_checks = measurement(started)
                started = begin(f"{source}-load{bits}")
                controls = {"local_files_only": True, **({"fix_mistral_regex": True} if source == "synthetic" else {})}
                model, tokenizer, _ = load(str(output), tokenizer_config=controls, lazy=False, return_config=True, trust_remote_code=False)
                loading = measurement(started)
                B._require(not model.training and getattr(model, "quant_predicate", None) is None, "Model mode/quantization predicate differs")
                arrays = dict(tree_flatten(model.parameters()))
                dtypes = {"BF16": mx.bfloat16, "F32": mx.float32, "U32": mx.uint32}
                B._require(set(arrays) == set(tensors) and all(list(v.shape) == tensors[n]["shape"]
                           and v.dtype == dtypes[tensors[n]["dtype"]] for n, v in arrays.items()), "Loaded tensor keys/shapes/dtypes differ")
                eligible = [n for n, m in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
                            if hasattr(m, "to_quantized") and m.weight.shape[-1] % 64 == 0]
                actual = {n: {"bits": m.bits, "group_size": m.group_size, "mode": m.mode} for n, m in model.named_modules()
                          if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))}
                B._require(all(m.get("bias") is None for n, m in model.named_modules() if n in (set(eligible) if bits == 16 else set(actual))), "Quantized module bias differs")
                if bits == 16:
                    B._require(len(eligible) == MODULES[source] and not actual, "BF16 module census differs")
                    expected_paths = eligible
                else:
                    coverage_checked(source, expected_paths, actual)
                    B._require(packed == set(expected_paths) == set(actual), "Native/header coverage differs")
                digests = {n: digest(arrays[n]) for n in nonquantized_names(arrays, expected_paths)}
                B._require(len(digests) == (177 if source == "actor" else 69), "Nonquantized tensor census differs")
                if bits == 8: identity_checked(parent_digests, digests)
                text = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                prompt = {"text": text, "ids": ids, "bos": tokenizer.bos_token_id, "eos": sorted(tokenizer.eos_token_ids),
                          "template_sha256": sha256(tokenizer.chat_template.encode()).hexdigest()}
                B._require(0 < len(ids) <= 128 and prompt["eos"] == ([248046] if source == "actor" else [2])
                           and (prompt["bos"] is None if source == "actor" else prompt["bos"] == ids[0] == 1), "Prompt/EOS/BOS differs")
                if bits == 8: B._require(prompt == parent_prompt, "Paired prompt differs")
                started = begin(f"{source}-uncached-full-sequence-forward{bits}")
                logits = model(mx.array([ids]), cache=None)
                mx.eval(logits)
                vocab = original_config["text_config"]["vocab_size"]
                B._require(logits.shape == (1, len(ids), vocab) and logits.dtype == mx.bfloat16
                           and bool(mx.all(mx.isfinite(logits)).item()), "Forced forward contract differs")
                vector = np.array(logits[0, -1].astype(mx.float32), copy=True)
                forward = {**measurement(started), "label": "uncached_full_sequence_forward", "forced_forward_tokens": len(ids),
                           "shape": list(logits.shape), "dtype": str(logits.dtype), "last_logits_sha256": sha256(vector.tobytes()).hexdigest(),
                           "last_logits_norm": float(np.linalg.norm(vector))}
                difference = None if bits == 16 else {"max_abs": float(np.max(np.abs(vector - parent_vector))),
                                                     "rms": float(np.sqrt(np.mean((vector - parent_vector) ** 2)))}
                del logits, arrays, model, tokenizer
                gc.collect(); mx.synchronize(); mx.clear_cache()
                record = {"schema_version": 1, "request_sha256": identity, "source": source, "bits": bits,
                    "model_path": f"models/{source}-{bits}", "parent16_record_sha256": parent_sha,
                    "official": {k: job[k] for k in ("repo_id", "revision")}, "input_sha256": request["input_sha256"],
                    "converter_kwargs": kwargs, "manifest": manifest, "headers": tensors, "source_census": census,
                    "config": config, "prompt": prompt, "quantized_modules": actual, "eligible_paths": expected_paths,
                    "nonquantized": digests, "runtime_sha256": runtime_sha, "loaded_sources": module_sources(),
                    "conversion": conversion, "static_checks": static_checks, "loading": loading, "forward": forward,
                    "paired_difference": difference, "memory_after_release": {
                        "mlx_active_bytes": mx.get_active_memory(), "mlx_cache_bytes": mx.get_cache_memory()}}
                name = f"records/{source}-{bits}.json"
                record_sha = B._write(directory / name, record)
                bindings.append({"source": source, "bits": bits, "record": name, "sha256": record_sha})
                emit("artifact", source=source, bits=bits, record_sha256=record_sha)
                if bits == 16: parent_config, parent_sha, parent_digests, parent_prompt, parent_vector = config, record_sha, digests, prompt, vector
            del parent_vector, vector
        phase = "final-checks"
        emit("phase", phase=phase)
        for output, manifest in manifests: B.validate_manifest(output, manifest)
        B._require(all(not (output / ".cache").exists() for output, _ in manifests), "Output cache appeared")
        B._require(all(signature(path) == value for path, value in source_stats.items()), "Official source changed")
        B._require(all(B._hash(path) == digest for path, digest in pinned_sources.items()), "Pinned runtime source changed")
        for path in [o for o, _ in manifests] + [directory / "models", directory / "records", directory, directory.parent]:
            descriptor = os.open(path, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        emit("done", request_sha256=identity, result={"inputs_verified": True, "artifacts_ready": True, "generator_requests": 0, "records": bindings})
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
    request = {"stage": "artifacts", "jobs": jobs, "sources": sources, "input_sha256": inputs, "guard_path": str(B.GUARD_PATH),
        "directory": str(directory), "messages": [{"role": "system", "content": "Return a complete Python program."},
            {"role": "user", "content": "Invented fixture: print the integer1 followed by a newline."}]}
    identity = B._write(directory / "request.json", request)
    argv = [str(interpreter), "-I", "-B", str(WORKER), "--worker", str(directory / "request.json")]
    B._write(directory / "launch.json", {"stage": "artifacts", "argv": argv, "environment": environment, "disk_usage": disk._asdict(),
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
    events, protocol_error, records = [], None, []
    try:
        for line in (directory / "stdout.jsonl").read_bytes().splitlines(keepends=True):
            B._require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = json.loads(line)
            B._require(type(event) is dict and type(event.get("event")) is str, "Invalid protocol event")
            json.dumps(event, allow_nan=False)
            events.append(event)
        B._require(process is not None and events and events[0] == {"event": "start", "request_sha256": identity, "pid": process.pid}, "Worker start identity differs")
        result = events[-1]["result"]
        B._require(events[-1]["event"] == "done" and events[-1]["request_sha256"] == identity
            and result == {"inputs_verified": True, "artifacts_ready": True, "generator_requests": 0, "records": result["records"]}
            and all(e["event"] not in ("start", "done", "error") for e in events[1:-1]), "Worker protocol incomplete")
        records = records_checked(directory, identity, result["records"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        protocol_error = str(error)
    complete = process is not None and process.returncode == 0 and protocol_error is None and absent and cleanup_error is None
    if any((directory / name).stat().st_size > cap for name, cap in (("stdout.jsonl", STDOUT_LIMIT), ("stderr.txt", STDERR_LIMIT))):
        status = status or "output_limit"
    final_status = status or ("passed" if complete else "worker_error")
    receipt = {"stage": "artifacts", "inputs_verified": final_status == "passed", "artifacts_ready": final_status == "passed",
        "status": final_status, "evidence_path": str(directory),
        "records": records if final_status == "passed" else [],
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
