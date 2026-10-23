"""One isolated local MLX request; stdout is a flushed JSONL protocol only."""

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
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


def main():
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    threading.Thread(target=watch_parent, daemon=True).start()

    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}, allow_nan=False), file=protocol, flush=True)

    raw = Path(sys.argv[1]).read_bytes()
    request_sha256 = sha256(raw).hexdigest()
    request = json.loads(raw)
    emit("start", request_sha256=request_sha256, pid=os.getpid())
    model_path = Path(request["model_path"])
    started = monotonic()
    model_files = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                   for path in sorted(model_path.iterdir())
                   if not path.name.startswith(".") and path.is_file()}
    metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    hash_seconds = monotonic() - started
    started = monotonic()
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer_config = {"local_files_only": True}
    if request["fix_mistral_regex"]:
        tokenizer_config["fix_mistral_regex"] = True
    model, tokenizer, config = load(str(model_path), tokenizer_config=tokenizer_config,
                                   lazy=False, return_config=True, trust_remote_code=False)
    template_options = {"add_generation_prompt": True, "enable_thinking": request["enable_thinking"]}
    prompt = tokenizer.apply_chat_template(request["messages"], tokenize=False, **template_options)
    prompt_ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, **template_options)
    sampler = make_sampler(temp=request["temperature"], top_p=request["top_p"], seed=request["seed"])
    load_seconds = monotonic() - started
    started = monotonic()
    sources = {}
    for name, module in sorted(list(sys.modules.items())):
        path = getattr(module, "__file__", None)
        if name.split(".", 1)[0] in ("mlx", "mlx_lm", "transformers", "tokenizers") and path and Path(path).is_file():
            path = str(Path(path).resolve())
            if path not in sources:
                sources[path] = file_sha256(path)
    provenance = {
        "python": sys.version, "executable": sys.executable, "argv": sys.orig_argv,
        "environment": dict(os.environ),
        "worker_sha256": file_sha256(__file__), "sources": sources,
        "versions": {name: version(name) for name in (
            "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "jinja2", "huggingface-hub")},
        "model_path": str(model_path), "model_files": model_files, "revision": revision,
        "quantization": config.get("quantization_config", config.get("quantization")),
        "sampler_seed": getattr(sampler, "seed", None), "tokenizer_config": tokenizer_config,
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


if __name__ == "__main__":
    main()
