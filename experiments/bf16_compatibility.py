"""One owned, invented-fixture BF16 LoRA/normalized-feature compatibility check."""

from collections import Counter
from hashlib import sha256
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
INTERPRETER = "/Users/michal/Workspace/BurnQED/.venv/bin/python"
WORKER = Path(__file__).resolve()
GUARD_PATH = ROOT / "src/data-pipeline/mlx_worker.py"
HELPER_PATH = ROOT / "src/data-pipeline/local_runner.py"
MODEL_PATH = ROOT / "data/models/Qwen3.5-9B-c202236"
MANIFEST_PATH = ROOT / "data/03_reviews/qwen-bf16-download-verification.json"
UPSTREAM_PATH = ROOT / "data/03_reviews/qwen-bf16-upstream-metadata.json"
WALL_SECONDS, CLEANUP_SECONDS = 1800, 60
STDOUT_LIMIT, STDERR_LIMIT = 1024 * 1024, 16 * 1024 * 1024
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN_SOURCE = "0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a"
VERSIONS = {"torch": "2.10.0", "transformers": "5.3.0", "peft": "0.18.1", "accelerate": "1.13.0"}
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ENABLE_MPS_FALLBACK": "0", "HF_DEACTIVATE_ASYNC_LOAD": "1"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(raw).hexdigest()


def validate_manifest(model_dir, manifest):
    model_dir = Path(model_dir)
    entries = manifest["files"]
    names = [item["path"] for item in entries]
    _require(len(names) == len(set(names)) and all(Path(n).name == n for n in names), "Manifest names differ")
    actual = [p for p in model_dir.iterdir() if p.name != ".cache"]
    _require(set(p.name for p in actual) == set(names), "Manifest file inventory differs")
    _require(not (model_dir / ".cache").is_symlink(), "Manifest cache is a symlink")
    for item in entries:
        path = model_dir / item["path"]
        _require(not path.is_symlink() and path.is_file() and path.resolve() == path
                 and path.stat().st_size == item["bytes"] and _hash(path) == item["sha256"], "Manifest file differs")


def completion_labels(prompt_ids, completion_ids):
    _require(type(prompt_ids) is list and type(completion_ids) is list and prompt_ids and completion_ids,
             "Require prompt and completion tokens")
    ids = prompt_ids + completion_ids
    _require(128 <= len(ids) <= 512 and all(type(i) is int and i >= 0 for i in ids), "Fixture token bounds differ")
    return ids, [-100] * len(prompt_ids) + completion_ids


def _exited(process):
    return importlib.import_module("src.data-pipeline.local_runner")._exited(process)


def _kill_group(process):
    return importlib.import_module("src.data-pipeline.local_runner")._kill_group(process)


def _worker(request_file):
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}, allow_nan=False), file=protocol, flush=True)
    raw = Path(request_file).read_bytes()
    identity, request, phase = sha256(raw).hexdigest(), json.loads(raw), "manifest"
    emit("start", request_sha256=identity, pid=os.getpid())
    try:
        guard = Path(request["guard_path"])
        _require(_hash(guard) == request["sources"][str(guard)], "Guard source differs")
        spec = importlib.util.spec_from_file_location("owned_guard", guard)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        threading.Thread(target=module.watch_parent, daemon=True).start()
        for path, digest in request["sources"].items():
            _require(_hash(path) == digest, "Worker source differs")
        started, path = monotonic(), Path(request["model_path"])
        manifest, upstream = request["manifest"], request["upstream"]
        validate_manifest(path, manifest)
        _require(len(manifest["files"]) == 15 and manifest["repo_id"] == upstream["repo_id"] == "Qwen/Qwen3.5-9B"
                 and manifest["revision"] == upstream["revision"] == REVISION, "Manifest official identity differs")
        files = {item["path"]: item for item in manifest["files"]}
        _require(files["model.safetensors.index.json"]["sha256"] == upstream["weight_index"]["response_sha256"], "Manifest index differs")
        for entry in upstream["tree_api"]["json"]:
            if entry["path"].endswith(".safetensors"):
                _require(files[entry["path"]]["sha256"] == entry["lfs"]["oid"]
                         and files[entry["path"]]["bytes"] == entry["size"], "Manifest shard differs")
        emit("manifest", seconds=monotonic() - started, files=manifest["files"])
        phase, started = "fixture", monotonic()
        versions = {name: importlib.metadata.version(name) for name in VERSIONS}
        _require(versions == VERSIONS and all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "Runtime versions/environment differ")
        import torch
        from transformers import AutoTokenizer, Qwen3_5TextConfig, Qwen3_5ForCausalLM
        from peft import LoraConfig, get_peft_model
        from peft.tuners.lora.layer import LoraLayer
        modeling = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        _require(_hash(modeling.__file__) == QWEN_SOURCE and torch.backends.mps.is_available()
                 and modeling.causal_conv1d_fn is None, "Native MPS runtime differs")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        _require(tokenizer.eos_token == "<|im_end|>" and tokenizer.eos_token_id == 248046
                 and tokenizer.bos_token_id is None, "Tokenizer special tokens differ")
        prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        _require(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "Thinking-off template differs")
        completion = request["completion"] + tokenizer.eos_token
        encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
        prefix, target = encode(prompt), encode(completion)
        ids, labels = completion_labels(prefix, target)
        _require(encode(prompt + completion) == ids and target[-1] == tokenizer.eos_token_id
                 and not set(target[:-1]) & set(tokenizer.all_special_ids), "Completion boundary differs")
        emit("fixture", seconds=monotonic() - started, prompt=prompt, completion=completion, ids=ids, labels=labels,
             boundary=len(prefix), supervised_tokens=len(target), versions=versions, modeling_sha256=QWEN_SOURCE,
             runtime={"python": sys.version, "executable": sys.executable, "platform": sys.platform, "argv": sys.orig_argv, "environment": dict(os.environ),
                      "sources": {m.__file__: _hash(m.__file__) for m in (torch, importlib.import_module("transformers"),
                                  importlib.import_module("peft"), importlib.import_module("accelerate"), torch._C)}},
             chat_template_sha256=sha256(tokenizer.chat_template.encode()).hexdigest(),
             template_file_equal=tokenizer.chat_template == (path / "chat_template.jinja").read_text())
        def memory():
            torch.mps.synchronize()
            return {"current_allocated": torch.mps.current_allocated_memory(), "driver_allocated": torch.mps.driver_allocated_memory(),
                    "recommended_max": torch.mps.recommended_max_memory(), "cpu_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        phase, started = "load", monotonic()
        index, raw_config = json.loads((path / "model.safetensors.index.json").read_text()), json.loads((path / "config.json").read_text())
        keys = set(index["weight_map"])
        wanted = {k for k in keys if k.startswith("model.language_model.") or k == "lm_head.weight"}
        _require(len(keys) == 775 and len(wanted) == 427 and sum(k.startswith("model.visual.") for k in keys) == 333
                 and sum(k.startswith("mtp.") for k in keys) == 15 and raw_config["tie_word_embeddings"] is False, "Checkpoint keys differ")
        cfg = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
        _require((cfg.model_type, cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size, cfg.tie_word_embeddings,
                  cfg.attention_dropout) == ("qwen3_5_text", 32, 4096, 248320, False, 0.0), "Text config differs")
        model, info = Qwen3_5ForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16, device_map={"": "mps"},
            attn_implementation="eager", output_loading_info=True, local_files_only=True, trust_remote_code=False, ignore_mismatched_sizes=False)
        _require(all(not info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")), "Strict loading failed")
        _require(set(model.state_dict()) == {k.replace("model.language_model.", "model.") for k in wanted}
                 and model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "Loaded text keys/head differ")
        tensors = list(model.parameters()) + list(model.buffers())
        _require(all(str(p.device) == "mps:0" for p in tensors)
                 and all(p.dtype == torch.bfloat16 for p in model.parameters()), "Loaded devices/dtypes differ")
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                _require(layer.linear_attn.chunk_gated_delta_rule is modeling.torch_chunk_gated_delta_rule
                         and layer.linear_attn.causal_conv1d_fn is None, "Linear attention dispatch differs")
        snapshot = memory()
        emit("loaded", seconds=monotonic() - started, memory_snapshot_not_peak=snapshot, config_eos=cfg.eos_token_id,
             tokenizer_eos=tokenizer.eos_token_id, loading_info={k: sorted(v, key=str) if isinstance(v, set) else v for k, v in info.items()},
             dtype_elements=dict(Counter({str(d): sum(p.numel() for p in tensors if p.dtype == d) for d in {p.dtype for p in tensors}})))
        phase, started = "setup", monotonic()
        base_parameters = [(p, p.data_ptr()) for p in model.parameters()]
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        torch.manual_seed(request["seed"])
        targets = [f"model.layers.{i}.{family}.{name}" for i in range(32) for family, names in
                   [("self_attn", ("q_proj", "v_proj")) if i % 4 == 3 else ("linear_attn", ("in_proj_qkv", "out_proj"))] for name in names]
        _require(len(targets) == 64 and set(targets) <= dict(model.named_modules()).keys(), "LoRA target paths differ")
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0,
            bias="none", use_rslora=False, modules_to_save=None, target_modules=targets), autocast_adapter_dtype=True)
        model.train()
        base = model.get_base_model()
        _require({n for n, m in base.named_modules() if isinstance(m, LoraLayer)} == set(targets)
                 and model.peft_config["default"].lora_dropout == 0, "Injected targets differ")
        _require(all(l.training and l.gradient_checkpointing and l._gradient_checkpointing_func.keywords == {"use_reentrant": False}
                     for l in base.model.layers), "Checkpointing differs")
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        _require(sum(p.numel() for p in trainable.values()) == 20185088 and all(".lora_A." in n or ".lora_B." in n for n in trainable)
                 and all(p.dtype == torch.float32 and str(p.device) == "mps:0" for p in trainable.values()), "Trainable census differs")
        _require(all(not p.requires_grad for p, _ in base_parameters)
                 and all(torch.count_nonzero(p).item() == 0 for n, p in trainable.items() if ".lora_B." in n), "Initial adapters/base differ")
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-4, weight_decay=0, foreach=False, fused=False)
        groups = [{**{k: v for k, v in g.items() if k != "params"}, "parameter_names": list(trainable)} for g in optimizer.param_groups]
        _require(not {id(p) for p, _ in base_parameters} & {id(p) for g in optimizer.param_groups for p in g["params"]}, "Optimizer includes base")
        before = {n: p.detach().clone() for n, p in trainable.items()}
        tokens = torch.tensor([ids], device="mps", dtype=torch.long)
        mask = torch.ones_like(tokens)
        snapshot = memory()
        emit("setup", seconds=monotonic() - started, memory_snapshot_not_peak=snapshot)
        phase, started = "training", monotonic()
        emit("training", targets=targets, trainable_parameters=20185088, optimizer=groups, seed=request["seed"])
        output = model(input_ids=tokens, attention_mask=mask, labels=torch.tensor([labels], device="mps"), use_cache=False)
        _require(torch.isfinite(output.loss).item(), "Nonfinite training loss")
        loss = float(output.loss.item())
        output.loss.backward()
        _require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in trainable.values()), "Nonfinite/missing gradient")
        gradients = {n: float(p.grad.norm().item()) for n, p in trainable.items()}
        for family in ("self_attn", "linear_attn"):
            _require(any(gradients[n] > 0 for n in trainable if family in n and ".lora_B." in n), "Missing B gradient family")
        optimizer.step()
        changes = {n: float((p.detach() - before[n]).norm().item()) for n, p in trainable.items()}
        for family in ("self_attn", "linear_attn"):
            _require(any(changes[n] > 0 for n in trainable if family in n and ".lora_B." in n), "Missing B update family")
        current_ids = {id(p) for p in model.parameters()}
        _require(all(id(p) in current_ids and p.data_ptr() == pointer and not p.requires_grad and p.grad is None
                     for p, pointer in base_parameters), "Frozen base binding/gradient differs")
        snapshot = memory()
        emit("updated", loss=loss, gradients=gradients, changes=changes, optimizer_steps=1,
             seconds=monotonic() - started, memory_snapshot_not_peak=snapshot)
        del output, before
        phase, started = "features", monotonic()
        model.eval()
        with torch.no_grad():
            out = model(input_ids=tokens, attention_mask=mask, use_cache=False, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            _require(len(out.hidden_states) == 33 and all(tuple(h.shape) == (1, len(ids), 4096) and h.dtype == torch.bfloat16
                     and str(h.device) == "mps:0" and torch.isfinite(h).all().item() for h in out.hidden_states), "Hidden feature contract differs")
            repeated = model(input_ids=tokens, attention_mask=mask, use_cache=False, output_hidden_states=True).hidden_states[-1]
            direct = base.model(input_ids=tokens, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            reconstructed = base.lm_head(hidden)
            comparisons = {}
            for name, a, b in (("repeat", hidden, repeated), ("direct", hidden, direct), ("lm_head", reconstructed, out.logits)):
                _require(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()
                         and torch.allclose(a, b, rtol=1e-2, atol=1e-2), "Feature/logit comparison failed")
                comparisons[name] = {"exact": torch.equal(a, b), "max_abs": float((a - b).abs().max().item())}
            _require(comparisons["lm_head"]["exact"], "LM head is not exact")
            feature_sha = sha256(hidden.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        result = {"optimizer_steps": 1, "generator_requests": 0, "input_tokens": len(ids), "supervised_tokens": len(target),
                  "feature_shape": list(hidden.shape), "feature_dtype": str(hidden.dtype), "feature_device": str(hidden.device),
                  "feature_norm": float(hidden.float().norm().item()), "feature_sha256": feature_sha, "comparisons": comparisons,
                  "rtol": 1e-2, "atol": 1e-2, "memory_snapshot_not_peak": memory(), "seconds": monotonic() - started}
        emit("features", **result)
        emit("done", request_sha256=identity, result=result)
    except BaseException as error:
        emit("error", phase=phase, error=repr(error))
        raise


def run(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    inputs = {"manifest": Path(MANIFEST_PATH).read_bytes(), "upstream": Path(UPSTREAM_PATH).read_bytes()}
    sources = {str(Path(p).resolve()): _hash(p) for p in (WORKER, GUARD_PATH, HELPER_PATH)}
    input_sha = {name: _write(directory / f"{name}.json", raw) for name, raw in inputs.items()}
    _write(directory / "source.py", Path(WORKER).read_bytes())
    _write(directory / "guard-source.py", Path(GUARD_PATH).read_bytes())
    _write(directory / "helper-source.py", Path(HELPER_PATH).read_bytes())
    comments = "".join(f"# fixture row {i:02d}: alpha beta gamma.\n" for i in range(24))
    request = {"model_path": str(MODEL_PATH), **{k: json.loads(v) for k, v in inputs.items()}, "input_sha256": input_sha,
        "sources": sources, "guard_path": str(GUARD_PATH), "seed": 17, "completion": "This program only prints1.\nVerdict: PASS",
        "messages": [{"role": "system", "content": "You are a code monitor. Reply with a short reason and Verdict: PASS or Verdict: FAIL."},
                     {"role": "user", "content": "Task: print1 and no other output.\nProgram:\n```python\n" + comments + "print(1)\n```"}]}
    identity = _write(directory / "request.json", request)
    argv = [str(INTERPRETER), "-I", "-B", str(WORKER), "--worker", str(directory / "request.json")]
    _write(directory / "launch.json", {"argv": argv, "environment": ENVIRONMENT, "wall_seconds": WALL_SECONDS,
        "cleanup_grace_seconds": CLEANUP_SECONDS, "soft_stream_caps": [STDOUT_LIMIT, STDERR_LIMIT]})
    for path in (directory, directory.parent):
        descriptor = os.open(path, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    started, process, status, interruption, cleanup_error = monotonic(), None, None, None, None
    with (directory / "stdout.jsonl").open("xb") as stdout, (directory / "stderr.txt").open("xb") as stderr:
        try:
            process = subprocess.Popen(argv, cwd=directory, env=ENVIRONMENT, stdin=subprocess.PIPE,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            while not _exited(process):
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
                for operation in (lambda: _kill_group(process), process.stdin.close,
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
            _require(line.endswith(b"\n"), "Incomplete protocol tail")
            event = json.loads(line)
            _require(type(event) is dict and type(event.get("event")) is str, "Invalid protocol event")
            json.dumps(event, allow_nan=False)
            events.append(event)
        _require(process is not None and events and events[0] == {"event": "start", "request_sha256": identity, "pid": process.pid}, "Worker start identity differs")
        _require(events[-1]["event"] == "done" and events[-1]["request_sha256"] == identity
                 and type(events[-1]["result"]) is dict and all(e["event"] not in ("start", "done", "error") for e in events[1:-1]), "Worker protocol incomplete")
    except (ValueError, KeyError, TypeError) as error:
        protocol_error = str(error)
    complete = process is not None and process.returncode == 0 and protocol_error is None and absent and cleanup_error is None
    if any((directory / name).stat().st_size > cap for name, cap in (("stdout.jsonl", STDOUT_LIMIT), ("stderr.txt", STDERR_LIMIT))):
        status = status or "output_limit"
    receipt = {"status": status or ("passed" if complete else "worker_error"), "evidence_path": str(directory),
        "request_sha256": identity, "sources": sources, "input_sha256": input_sha, "pid": process.pid if process else None,
        "returncode": process.returncode if process else None, "group_absent": absent, "cleanup_error": cleanup_error,
        "parent_error": repr(interruption) if interruption else None, "protocol_error": protocol_error, "events": events,
        "optimizer_steps": 1 if any(e.get("event") == "updated" for e in events) else None,
        "generator_requests": 0, "elapsed_seconds": monotonic() - started, "cleanup_seconds": monotonic() - cleanup_started}
    _write(directory / "receipt.json", receipt)
    for path in (directory, directory.parent):
        descriptor = os.open(path, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    if interruption is not None: raise interruption
    return receipt


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        parent = ROOT / "data/03_training_checks"
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="bf16-compatibility-", dir=parent))
        directory.rmdir()
        result = run(directory)
        print(json.dumps({"status": result["status"], "evidence_path": result["evidence_path"]}))
        sys.exit(result["status"] != "passed")
