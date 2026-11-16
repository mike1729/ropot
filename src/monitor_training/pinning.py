"""Freeze a local upstream snapshot and the selected NVIDIA inference environment."""

from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import re

from .rendering import Renderer


artifacts = import_module("src.data-pipeline.artifacts")
PACKAGES = ("torch", "transformers", "tokenizers", "accelerate", "bitsandbytes",
            "safetensors", "huggingface-hub", "numpy", "jinja2")
OPTIONAL_PACKAGES = ("flash-linear-attention", "causal-conv1d", "kernels", "triton")
SOURCES = ("interface.py", "data.py", "prepare.py", "rendering.py", "runtime.py", "pinning.py", "score.py")
SNAPSHOT_FILES = ("config.json", "generation_config.json", "model.safetensors", "model.safetensors.index.json",
                  "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json", "vocab.txt",
                  "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")


def package_versions():
    versions = {name: metadata.version(name) for name in PACKAGES}
    for name in OPTIONAL_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def source_hashes():
    return {name: artifacts.digest(Path(__file__).parent / name) for name in SOURCES}


def snapshot_hashes(model_root):
    """Pin local safetensors and tokenizer inputs, excluding repository/cache metadata."""
    root = Path(model_root)
    names = set(SNAPSHOT_FILES)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names.update(json.loads(index.read_bytes())["weight_map"].values())
    names.update(str(path.relative_to(root)) for path in root.glob("additional_chat_templates/*.jinja"))
    return {name: artifacts.digest(root / name) for name in sorted(names) if (root / name).is_file()}


def validate_settings(settings):
    """Only the first proposed numerical recipe is implemented in T2."""
    if set(settings) != {"checkpoint", "max_total_tokens", "max_new_tokens", "terminator_id", "budgets"}:
        raise ValueError("Settings must contain exactly checkpoint, token limits, terminator_id and budgets")
    if not re.fullmatch(r"[0-9a-f]{40}", settings["checkpoint"]["revision"]):
        raise ValueError("Pin an immutable upstream checkpoint commit")
    for name in ("max_total_tokens", "max_new_tokens"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(settings["terminator_id"]) is not int or settings["terminator_id"] < 0:
        raise ValueError("terminator_id must be a nonnegative integer")
    if settings["max_new_tokens"] >= settings["max_total_tokens"]:
        raise ValueError("Output cap must leave room for an inference prefix")
    budget_names = ("max_margin_tokens", "max_wall_seconds", "max_cost_usd", "gpu_hourly_usd")
    if set(settings["budgets"]) != set(budget_names):
        raise ValueError("Budgets must contain only margin tokens, wall time, cost and hourly price")
    for name in budget_names:
        value = settings["budgets"][name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float("inf"):
            raise ValueError(f"Numeric positive budget required: {name}")


def load_tokenizer(model_root):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)


def freeze(model_root, settings_path, output_dir):
    """Freeze before opening development. This records a candidate, not qualification."""
    if Path(output_dir).exists() or Path(output_dir).is_symlink():
        raise FileExistsError(output_dir)
    settings = json.loads(Path(settings_path).read_bytes())
    validate_settings(settings)
    config = json.loads((Path(model_root) / "config.json").read_bytes())
    if config.get("quantization_config"):
        raise ValueError("T2 loads original upstream weights for fresh CUDA quantization")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Freeze the recipe on its intended NVIDIA host")
    tokenizer = load_tokenizer(model_root)
    renderer = Renderer(tokenizer, max_total_tokens=settings["max_total_tokens"],
                        max_new_tokens=settings["max_new_tokens"], terminator_id=settings["terminator_id"])
    renderer.render(dict(statement="Template check.", program="pass"))
    record = dict(settings, schema_version="monitor-inference-v1", backend="transformers",
                  numerical_base="bnb-int8-bf16", attention="sdpa", seed=17, requested_temperature=0,
                  packages=package_versions(),
                  python_version=platform.python_version(), cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0), source_sha256=source_hashes())
    record["checkpoint"] = dict(settings["checkpoint"], files=snapshot_hashes(model_root))
    with artifacts.publication_directory(output_dir) as staging:
        artifacts.write_json(staging / "contract.json", record)
        (staging / "requirements-training.txt").write_text(
            "".join(f"{name}=={version}\n" for name, version in record["packages"].items() if version is not None),
            encoding="utf-8")
        digest = artifacts.digest(staging / "contract.json")
    return digest


def load_contract(path, expected_sha256, model_root):
    """Reuse a reviewed freeze-produced record, checking identity and current inputs."""
    raw = Path(path).read_bytes()
    if artifacts.sha256(raw) != expected_sha256:
        raise ValueError("Inference contract differs from its reviewed pin")
    contract = json.loads(raw)
    if contract["source_sha256"] != source_hashes():
        raise ValueError("Inference source differs from the frozen recipe")
    if contract["python_version"] != platform.python_version():
        raise ValueError("Python version differs from the frozen recipe")
    if contract["packages"] != package_versions():
        raise ValueError("Packages differ from the frozen recipe")
    if contract["checkpoint"]["files"] != snapshot_hashes(model_root):
        raise ValueError("Snapshot files differ from the frozen recipe")
    return contract
