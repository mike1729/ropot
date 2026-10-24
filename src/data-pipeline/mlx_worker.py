"""One isolated local MLX request; stdout is a flushed JSONL protocol only."""

from hashlib import sha256
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
import threading
from time import monotonic


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def watch_parent():
    """The parent owns our session and keeps stdin open without writing to it."""
    while os.read(0, 1):
        pass
    os.killpg(0, signal.SIGKILL)


class Control(Exception):
    def __init__(self, code, detail):
        self.code, self.detail = code, str(detail)[:500]


def load_guard(request):
    path = Path(__file__).with_name("precision_profiles.py")
    try:
        if os.path.realpath(path) != str(path): raise Control("guard", "Noncanonical guard path")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode): raise Control("guard", "Nonregular guard")
            raw = stream.read()
    except OSError as error:
        raise Control("guard", error) from error
    wrapper = request["precision_profile"]
    if sha256(raw).hexdigest() != wrapper["guard_sha256"]: raise Control("guard", "Guard source changed")
    spec = importlib.util.spec_from_file_location("precision_guard", path)
    guard = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), guard.__dict__)
    try: fingerprint = guard.check_runtime(wrapper["descriptor"]["expected_runtime"])
    except guard.ProfileError as error: raise Control("runtime", error) from error
    if fingerprint != wrapper["descriptor"]["runtime_guard_sha256"]: raise Control("runtime", "Runtime digest differs")
    return guard


def inventory(model_path, guard=None, descriptor=None):
    try:
        if descriptor is not None and (os.path.realpath(model_path) != str(model_path)
                or set(os.listdir(model_path)) != set(descriptor["model_files"])): raise Control("artifact", "Model inventory differs")
        paths = sorted(model_path.iterdir())
        if descriptor is not None and any(not stat.S_ISREG(p.lstat().st_mode) for p in paths): raise Control("artifact", "Nonregular model file")
        files = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in paths if not p.name.startswith(".") and p.is_file()}
        if descriptor is not None and guard.digest(files) != guard.digest(descriptor["model_files"]): raise Control("artifact", "Model files differ")
        return files
    except OSError as error:
        if descriptor is not None: raise Control("artifact", error) from error
        raise


def origin(value):
    try: return str(Path(getattr(value, "__file__", None) or sys.modules[value.__module__].__file__).resolve())
    except (AttributeError, KeyError, TypeError) as error: raise Control("import_path", error) from error


def roles(guard, descriptor, paths, sources, post=False):
    try:
        binding = guard.bind_roles(descriptor["expected_runtime"]["sources"], guard.role_suffixes(descriptor["source"], post), paths)
        for item in binding.values():
            path = item["path"]
            if path not in sources: sources[path] = file_sha256(path)
            if sources[path] != item["sha256"]: raise Control("import_path", "Critical source hash differs")
        return binding
    except (guard.ProfileError, OSError) as error: raise Control("import_path", error) from error


def observe(sources, expected=None):
    for name, module in sorted(list(sys.modules.items())):
        path = getattr(module, "__file__", None)
        if name.split(".", 1)[0] in ("mlx", "mlx_lm", "transformers", "tokenizers") and path and Path(path).is_file():
            path = str(Path(path).resolve())
            if path not in sources:
                try: sources[path] = file_sha256(path)
                except OSError as error:
                    if expected is not None: raise Control("import_path", error) from error
                    raise
            if expected is not None and path in expected and sources[path] != expected[path]: raise Control("import_path", "Observed source hash differs")


def controlled_load(request, guard, model_files, sources, emit, load, stream_generate, make_sampler, *, identity, tokenizer_config):
    wrapper, descriptor = request["precision_profile"], request["precision_profile"]["descriptor"]
    mx = importlib.import_module("mlx.core")
    architecture = importlib.import_module("mlx_lm.models." + ("qwen3_5" if descriptor["source"] == "actor" else "mistral3"))
    if not mx.is_available(mx.gpu) or mx.default_device().type != mx.gpu: raise Control("device", "Metal/default GPU unavailable")
    binding = roles(guard, descriptor, dict(loader=origin(load), generator=origin(stream_generate), sampler=origin(make_sampler),
                    cache=origin(sys.modules.get("mlx_lm.models.cache")), native_core=origin(mx), architecture=origin(architecture)), sources)
    proof = dict(descriptor_sha256=guard.digest(descriptor), guard_sha256=wrapper["guard_sha256"],
                 runtime_guard_sha256=descriptor["runtime_guard_sha256"], model_files=model_files,
                 gpu=dict(available=True, default=str(mx.default_device()), info=mx.device_info()), critical_sources=binding)
    emit("controls_checked", request_sha256=identity, proof=proof)
    model, tokenizer, config = load(request["model_path"], tokenizer_config=tokenizer_config, lazy=False, return_config=True, trust_remote_code=False)
    try:
        post = roles(guard, descriptor, dict(model=origin(model), text_model=origin(model.language_model), tokenizer_wrapper=origin(tokenizer),
                     tokenizer_impl=origin(tokenizer._tokenizer), tokenizer_native=origin(sys.modules["tokenizers.tokenizers"])), sources, post=True)
    except (AttributeError, KeyError) as error: raise Control("import_path", error) from error
    observe(sources, descriptor["expected_runtime"]["sources"])
    return model, tokenizer, config, proof, post


def _run(request, request_sha256, emit):
    model_path = Path(request["model_path"])
    started = monotonic()
    guard = load_guard(request) if "precision_profile" in request else None
    descriptor = request["precision_profile"]["descriptor"] if guard else None
    model_files = inventory(model_path, guard, descriptor)
    metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    hash_seconds = monotonic() - started
    started = monotonic()
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer_config = {"local_files_only": True}
    if request["fix_mistral_regex"]:
        tokenizer_config["fix_mistral_regex"] = True
    sources = {}
    if guard:
        model, tokenizer, config, proof, post = controlled_load(request, guard, model_files, sources, emit, load, stream_generate, make_sampler, identity=request_sha256, tokenizer_config=tokenizer_config)
    else:
        model, tokenizer, config = load(str(model_path), tokenizer_config=tokenizer_config,
                                       lazy=False, return_config=True, trust_remote_code=False)
    template_options = {"add_generation_prompt": True, "enable_thinking": request["enable_thinking"]}
    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, **template_options)
    prompt_ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, **template_options)
    sampler = make_sampler(temp=request["temperature"], top_p=request["top_p"], seed=request["seed"])
    load_seconds = monotonic() - started
    started = monotonic()
    observe(sources, descriptor["expected_runtime"]["sources"] if guard else None)
    provenance = {
        "python": sys.version, "executable": sys.executable, "argv": sys.orig_argv,
        "environment": dict(os.environ),
        "worker_sha256": file_sha256(__file__), "sources": sources,
        "versions": {name: version(name) for name in (
            "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "jinja2", "huggingface-hub")},
        "model_path": str(model_path), "model_files": model_files, "revision": revision,
        "quantization": config.get("quantization_config", config.get("quantization")),
        "sampler_seed": getattr(sampler, "seed", None), "tokenizer_config": tokenizer_config,
        **(dict(precision_controls=proof, precision_postload=post) if guard else {}),
    }
    hash_seconds += monotonic() - started
    emit("ready", prompt=prompt, prompt_ids=prompt_ids, eos_token_ids=sorted(tokenizer.eos_token_ids),
         provenance=provenance, hash_seconds=hash_seconds, load_seconds=load_seconds)
    final = None
    for seq, response in enumerate(stream_generate(model, tokenizer, prompt_ids,
            max_tokens=request["max_tokens"], sampler=sampler, logits_processors=None), start=1):
        emit("chunk", seq=seq, **{key: getattr(response, key) for key in (
            "token", "text", "prompt_tokens", "generation_tokens", "finish_reason",
            "prompt_tps", "generation_tps", "peak_memory")})
        final = response.finish_reason
    if final is None:
        raise RuntimeError("MLX stream ended without a native completion status")
    emit("done", request_sha256=request_sha256, finish_reason=final)


def main():
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    threading.Thread(target=watch_parent, daemon=True).start()
    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}, allow_nan=False), file=protocol, flush=True)
    raw = Path(sys.argv[1]).read_bytes()
    identity, request = sha256(raw).hexdigest(), json.loads(raw)
    emit("start", request_sha256=identity, pid=os.getpid())
    try: _run(request, identity, emit)
    except Control as error:
        emit("profile_error", request_sha256=identity, code=error.code, detail=error.detail)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
