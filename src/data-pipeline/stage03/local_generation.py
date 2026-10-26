"""Direct warm-serial MLX generation. See docs/stage-03-batch-05-plan.md.

Call from the pinned MLX Python with caller-constructed requests. The artifact
manifest must describe the supplied, unchanged model files; hashes are recorded,
not reverified. Soft limits cannot interrupt native loading or a stalled GPU call.
"""

from contextlib import closing
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import monotonic


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_run(model_path, requests, directory, model_manifest):
    config = json.loads((model_path / "config.json").read_text())
    if config.get("quantization", config.get("quantization_config", {})).get("bits") != 8:
        raise ValueError("Generation requires 8-bit quantization")
    manifest = Path(model_manifest).read_bytes()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "model-manifest.json").write_bytes(manifest)
    save(directory / "inputs.json", {
        "model_path": str(model_path), "requests": requests, "model_config": config,
        "model_manifest_sha256": sha256(manifest).hexdigest(), "source_sha256": file_hash(__file__),
        "model_metadata_sha256": {name: file_hash(model_path / name) for name in
                                 ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}})
    return config


def load_model(model_path, directory, config):
    from mlx_lm import load

    versions = {name: version(name) for name in ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers")}
    tokenizer_options = {"local_files_only": True}
    model_modules = ("qwen3_5",)
    if config.get("model_type") == "mistral3":
        tokenizer_options["fix_mistral_regex"] = True
        model_modules = ("mistral3", "ministral3")
    started = monotonic()
    model, tokenizer = load(str(model_path), lazy=False,
        tokenizer_config=tokenizer_options, trust_remote_code=False)
    tokenizer_class = type(tokenizer._tokenizer)
    runtime = {"versions": versions, "python": sys.version, "executable": sys.executable,
               "tokenizer_options": tokenizer_options,
               "tokenizer_class": f"{tokenizer_class.__module__}.{tokenizer_class.__qualname__}",
               "load_seconds": monotonic() - started, "source_sha256": {}}
    for name in ("mlx_lm.generate", "mlx_lm.sample_utils", "mlx_lm.tokenizer_utils",
                 *(f"mlx_lm.models.{name}" for name in model_modules)):
        path = Path(importlib.import_module(name).__file__)
        runtime["source_sha256"][str(path)] = file_hash(path)
    save(directory / "runtime.json", runtime)
    return model, tokenizer


def generate_request(model, tokenizer, request, row, reply_path):
    """Update one result in place so partial output survives a Python exception."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    started = monotonic()
    deadline = started + request["wall_time_seconds"]
    row.update(status="running", finish_reason=None)
    try:
        options = {"add_generation_prompt": True, "enable_thinking": False}
        row["prompt"] = tokenizer.apply_chat_template(request["messages"], tokenize=False, **options)
        row["prompt_ids"] = tokenizer.apply_chat_template(request["messages"], tokenize=True, **options)
        sampler = make_sampler(temp=request["temperature"], top_p=request["top_p"], seed=request["seed"])
        with reply_path.open("w", encoding="utf-8") as reply:
            if monotonic() < deadline:
                with closing(stream_generate(model, tokenizer, row["prompt_ids"],
                        max_tokens=request["max_tokens"], sampler=sampler)) as stream:
                    for response in stream:
                        row["text"] += response.text
                        row["tokens"].append(response.token)
                        row.update({k: getattr(response, k) for k in
                                    ("prompt_tokens", "generation_tokens", "finish_reason")})
                        reply.write(response.text)
                        reply.flush()
                        if monotonic() >= deadline:
                            break
        if monotonic() >= deadline:
            row["status"] = "timeout"
        elif row["finish_reason"] in ("stop", "length"):
            row["status"] = row["finish_reason"]
        else:
            raise RuntimeError("Native stream ended without completion")
    finally:
        row["elapsed_seconds"] = monotonic() - started


def generate(model_path, requests, output_dir, *, model_manifest):
    """Load once, process a finite list serially, and save plain result files.

    Requests supply messages, seed, temperature, top_p, max_tokens and
    wall_time_seconds. A soft timeout or error ends the run without retry.
    Python interruptions save evidence then propagate; hard termination may
    lose metadata. Replies are flushed to text files as chunks arrive.
    """
    model_path, directory = Path(model_path).resolve(), Path(output_dir).absolute()
    config = prepare_run(model_path, requests, directory, model_manifest)
    rows = [{"status": "not_run", "text": "", "tokens": []} for _ in requests]
    result = {"status": "error", "requests": rows, "error": None}
    started, active = monotonic(), None
    try:
        model, tokenizer = load_model(model_path, directory, config)
        for index, (request, row) in enumerate(zip(requests, rows)):
            active = row
            generate_request(model, tokenizer, request, row, directory / f"reply-{index:04d}.txt")
            if row["status"] == "timeout":
                result["status"] = "timeout"
                break
        else:
            result["status"] = "success"
    except BaseException as error:
        result.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if active is not None and active["status"] not in ("stop", "length"):
            active["status"] = result["status"]
        if not isinstance(error, Exception):
            raise
    finally:
        result["elapsed_seconds"] = monotonic() - started
        save(directory / "outcome.json", result)
    return result
