"""Freeze precision evidence in the parent; guard its runtime with stdlib only."""

from hashlib import sha256
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

MAX_BYTES = 4 * 1024**2
FILES = ("receipt.json", "request.json", "runtime.json") + tuple(
    f"records/{source}-{bits}.json" for source in ("actor", "synthetic") for bits in (16, 8))
RECEIPT_FIELDS = set("artifacts_ready cleanup_error cleanup_seconds elapsed_seconds events evidence_path generator_requests group_absent input_sha256 inputs_verified parent_error pid protocol_error records request_sha256 returncode sources stage status".split())
REQUEST_FIELDS = set("directory guard_path input_sha256 jobs messages sources stage".split())
RUNTIME_FIELDS = set("argv device device_info environment executable platform python sources versions".split())
ENVIRONMENT_FIELDS = {"PATH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM"}
VERSION_FIELDS = {"mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "numpy", "jinja2", "huggingface-hub"}
CONTROLS = {source: dict(enable_thinking=False, fix_mistral_regex=source == "synthetic")
            for source in ("actor", "synthetic")}
CRITICAL_SUFFIXES = ("/mlx_lm/generate.py", "/mlx_lm/sample_utils.py", "/mlx_lm/models/cache.py",
                     "/mlx/core.cpython-312-darwin.so", "/mlx/lib/libmlx.dylib", "/mlx/lib/mlx.metallib")


class ProfileError(ValueError):
    """Frozen metadata or current runtime differs from the precision controls."""


def require(condition, message):
    if not condition:
        raise ProfileError(message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _absolute(value):
    return (type(value) is str and os.path.isabs(value) and not value.startswith("//")
            and os.path.normpath(value) == value and "\0" not in value)


def _pairs(items):
    result = dict(items)
    require(len(result) == len(items), "Invalid strict JSON duplicate key")
    return result


def _read(path):
    require(os.path.realpath(path) == str(path), "Invalid metadata path")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES, "Invalid metadata size/type")
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "Invalid metadata size")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs,
                           parse_constant=lambda _: require(False, "Invalid strict JSON constant"))
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise ProfileError("Invalid strict JSON") from error
    require(type(value) is dict, "Require metadata JSON object")
    return raw, value


def _sources(values):
    require(type(values) is dict and bool(values) and all(_absolute(p) and _sha(h) for p, h in values.items()), "Source map differs")


def _runtime(expected):
    require(type(expected) is dict and set(expected) == {"python", "executable", "platform", "versions", "environment", "sources", "flags"}, "Runtime schema differs")
    require(all(type(expected[k]) is str and expected[k] for k in ("python", "platform"))
            and _absolute(expected["executable"]), "Runtime interpreter differs")
    for name, fields in (("versions", VERSION_FIELDS), ("environment", ENVIRONMENT_FIELDS)):
        values = expected[name]
        require(type(values) is dict and set(values) == fields
                and all(type(v) is str and v for v in values.values()), f"Runtime {name} differs")
    require(expected["flags"] == {"isolated": 1, "dont_write_bytecode": 1}
            and all(type(v) is int for v in expected["flags"].values()), "Runtime flags differ")
    _sources(expected["sources"])
    require(all(sum(p.endswith(s) for p in expected["sources"]) == 1 for s in CRITICAL_SUFFIXES), "Runtime critical sources differ")


def _metadata(directory, identity, *, original=False):
    try:
        return _validated(directory, identity, original=original)
    except (OSError, KeyError, TypeError) as error:
        raise ProfileError("Record/metadata unreadable or invalid") from error


def check_descriptor(value):
    require(type(value) is dict and set(value) == set("source bits label receipt_sha256 record_sha256 parent16_record_sha256 model_path model_files enable_thinking fix_mistral_regex expected_runtime runtime_guard_sha256".split()), "Descriptor fields differ")
    source, bits = value["source"], value["bits"]
    require(type(source) is str and source in CONTROLS and type(bits) is int and bits in (8, 16), "Descriptor selection differs")
    require(value["label"] == ("MLX-native BF16" if bits == 16 else "affine8/group64")
            and all(_sha(value[k]) for k in ("receipt_sha256", "record_sha256", "runtime_guard_sha256"))
            and (value["parent16_record_sha256"] is None if bits == 16 else _sha(value["parent16_record_sha256"])), "Descriptor identity differs")
    require(_absolute(value["model_path"]) and value["model_path"].endswith(f"/models/{source}-{bits}"), "Descriptor model path differs")
    files = value["model_files"]
    require(type(files) is dict and bool(files) and all(type(n) is str and n and not n.startswith(".") and not any(c in n for c in "/\\\0")
            and type(f) is dict and set(f) == {"bytes", "sha256"} and type(f["bytes"]) is int and f["bytes"] >= 0 and _sha(f["sha256"])
            for n, f in files.items()), "Descriptor model files differ")
    require(all(value[k] is v for k, v in CONTROLS[source].items()), "Descriptor controls differ")
    _runtime(value["expected_runtime"])
    require(value["runtime_guard_sha256"] == digest(value["expected_runtime"]), "Descriptor runtime digest differs")
    return value


def role_suffixes(source, post=False):
    architecture = f"/mlx_lm/models/{'qwen3_5' if source == 'actor' else 'mistral3'}.py"
    if post:
        return dict(model=architecture, text_model=f"/mlx_lm/models/{'qwen3_5' if source == 'actor' else 'ministral3'}.py",
                    tokenizer_wrapper="/mlx_lm/tokenizer_utils.py", tokenizer_impl=None, tokenizer_native="/tokenizers/tokenizers.abi3.so")
    return dict(loader="/mlx_lm/utils.py", generator="/mlx_lm/generate.py", sampler="/mlx_lm/sample_utils.py",
                cache="/mlx_lm/models/cache.py", native_core="/mlx/core.cpython-312-darwin.so", architecture=architecture)


def bind_roles(sources, required, paths):
    require(type(paths) is dict and set(paths) == set(required), "Critical role fields differ")
    for role, suffix in required.items():
        require(type(paths[role]) is str and paths[role] in sources and (suffix is None
                or [p for p in sources if p.endswith(suffix)] == [paths[role]]), "Critical role path differs")
    return {role: dict(path=path, sha256=sources[path]) for role, path in paths.items()}


def check_profiles(profiles, sources, campaign=None):
    require(type(profiles) is dict and bool(profiles) and set(profiles) == set(sources)
            and set(profiles) <= CONTROLS.keys(), "Profile sources differ")
    common = set()
    for source, entry in profiles.items():
        require(type(entry) is dict and set(entry) == {"reference", "descriptor"}, "Profile entry differs")
        descriptor, reference = check_descriptor(entry["descriptor"]), entry["reference"]
        require(descriptor["source"] == source and type(reference) is dict and set(reference) == {"path", "receipt_sha256"}
                and reference["receipt_sha256"] == descriptor["receipt_sha256"]
                and reference["path"] == "precision/" + descriptor["receipt_sha256"], "Profile reference differs")
        common.add((descriptor["receipt_sha256"], descriptor["runtime_guard_sha256"], descriptor["bits"]))
        if campaign is not None:
            require(digest(resolve(campaign, reference, source, descriptor["bits"])) == digest(descriptor), "Profile metadata differs")
    require(len(common) == 1, "Profile arm differs")
    return profiles


def _validated(directory, identity, *, original):
    require(_sha(identity), "Receipt digest invalid")
    require(directory.is_dir() and os.path.realpath(directory) == str(directory), "Tree directory differs")
    if not original:
        entries = {str(p.relative_to(directory)): p for p in directory.rglob("*")}
        require(set(entries) == set(FILES) | {"records"} and entries["records"].is_dir()
                and all(not p.is_symlink() for p in entries.values()), "Tree inventory differs")
    captured = {name: _read(directory / name) for name in FILES}
    raw = {name: item[0] for name, item in captured.items()}
    receipt, request, runtime = (captured[name][1] for name in FILES[:3])
    require(set(receipt) == RECEIPT_FIELDS and sha256(raw["receipt.json"]).hexdigest() == identity, "Receipt authority differs")
    require(receipt["stage"] == "artifacts" and receipt["status"] == "passed"
            and all(receipt[k] is True for k in ("inputs_verified", "artifacts_ready", "group_absent"))
            and all(type(receipt[k]) is int and receipt[k] == 0 for k in ("returncode", "generator_requests"))
            and all(receipt[k] is None for k in ("cleanup_error", "parent_error", "protocol_error")), "Receipt is not passed")
    origin = receipt["evidence_path"]
    require(set(request) == REQUEST_FIELDS and request["stage"] == "artifacts"
            and sha256(raw["request.json"]).hexdigest() == receipt["request_sha256"], "Request authority differs")
    require(_absolute(origin) and request["directory"] == origin and (not original or origin == str(directory)), "Request origin differs")
    inputs = request["input_sha256"]
    require(type(inputs) is dict and bool(inputs) and all(type(k) is str and _sha(v) for k, v in inputs.items())
            and inputs == receipt["input_sha256"], "Input identities differ")
    require(receipt["sources"] == request["sources"], "Source identities differ")
    _sources(request["sources"])
    require(set(runtime) == RUNTIME_FIELDS and type(runtime["argv"]) is list
            and runtime["argv"][1:3] == ["-I", "-B"], "Runtime metadata differs")
    expected = {k: runtime[k] for k in ("python", "executable", "platform", "versions")}
    require(type(runtime["environment"]) is dict, "Runtime environment differs")
    _sources(runtime["sources"])
    expected.update(environment={k: runtime["environment"].get(k) for k in ENVIRONMENT_FIELDS},
                    sources=dict(runtime["sources"]), flags={"isolated": 1, "dont_write_bytecode": 1})
    jobs = request["jobs"]
    require(type(jobs) is list and [j["source"] for j in jobs] == ["actor", "synthetic"], "Official job order differs")
    records = {}
    for binding in receipt["records"]:
        source, bits = binding["source"], binding["bits"]
        record = captured[binding["record"]][1]
        require(type(record["schema_version"]) is int and record["schema_version"] == 1
                and type(record["bits"]) is int and record["bits"] == bits, "Record precision/schema differs")
        require(record["runtime_sha256"] == sha256(raw["runtime.json"]).hexdigest(), "Runtime record binding differs")
        require(record["input_sha256"] == inputs, "Input record identities differ")
        job = jobs[0 if source == "actor" else 1]
        require(record["official"] == {k: job[k] for k in ("repo_id", "revision")}, "Official record identity differs")
        controls = record["converter_kwargs"]
        require(controls["dtype"] == "bfloat16" and controls["dequantize"] is False and controls["quantize"] is (bits == 8)
                and (bits == 16 or {k: controls[k] for k in ("q_bits", "q_group_size", "q_mode")} == {
                    "q_bits": 8, "q_group_size": 64, "q_mode": "affine"}), "Record conversion differs")
        files = record["manifest"]["files"]
        require(type(files) is list and bool(files), "Manifest files required")
        names = [f["path"] for f in files]
        require(all(type(n) is str and n and not n.startswith(".") and "/" not in n and "\\" not in n and "\0" not in n for n in names)
                and len(names) == len(set(names)), "Manifest paths differ")
        require(all(set(f) == {"path", "bytes", "sha256"} and type(f["bytes"]) is int and f["bytes"] >= 0 and _sha(f["sha256"]) for f in files), "Manifest entries differ")
        _sources(record["loaded_sources"])
        for path, value in record["loaded_sources"].items():
            require(path not in expected["sources"] or expected["sources"][path] == value, "Source hashes conflict")
            expected["sources"][path] = value
        records[source, bits] = (record, {f["path"]: {k: f[k] for k in ("bytes", "sha256")} for f in files})
    _runtime(expected)
    try:
        importlib.import_module("experiments.prepare_precision_artifacts").records_checked(directory, receipt["request_sha256"], receipt["records"])
    except (ValueError, KeyError, TypeError) as error:
        raise ProfileError("Record bindings invalid: " + str(error)) from error
    return raw, receipt, records, expected


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def freeze(campaign, receipt_path, expected_receipt_sha256):
    require(campaign.db is not None and _sha(expected_receipt_sha256), "Reference/lease differs")
    parent = campaign.root / "precision"
    require(not parent.is_symlink(), "Tree precision directory differs")
    parent.mkdir(exist_ok=True)
    _sync(campaign.root)
    destination = parent / expected_receipt_sha256
    reference = {"path": f"precision/{expected_receipt_sha256}", "receipt_sha256": expected_receipt_sha256}
    if os.path.lexists(destination):
        _metadata(destination, expected_receipt_sha256)
        _sync(parent)
        return reference
    raw = _metadata(Path(receipt_path).parent, expected_receipt_sha256, original=True)[0]
    require(Path(receipt_path).name == "receipt.json", "Receipt filename differs")
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=parent) as temporary:
        staging = Path(temporary)
        (staging / "records").mkdir()
        for name, content in raw.items():
            with (staging / name).open("xb") as stream:
                stream.write(content); stream.flush(); os.fsync(stream.fileno())
        _metadata(staging, expected_receipt_sha256)
        _sync(staging / "records"); _sync(staging)
        require(not os.path.lexists(destination), "Tree destination appeared")
        os.rename(staging, destination)
        _sync(parent)
    return reference


def resolve(campaign, reference, source, bits):
    require(campaign.db is not None and type(reference) is dict and set(reference) == {"path", "receipt_sha256"}
            and _sha(reference["receipt_sha256"]) and reference["path"] == "precision/" + reference["receipt_sha256"], "Reference/lease differs")
    require(source in ("actor", "synthetic") and type(bits) is int and bits in (8, 16), "Reference selection differs")
    _, receipt, records, expected = _metadata(campaign.root / reference["path"], reference["receipt_sha256"])
    record, files = records[source, bits]
    binding = next(b for b in receipt["records"] if (b["source"], b["bits"]) == (source, bits))
    return dict(source=source, bits=bits, label="MLX-native BF16" if bits == 16 else "affine8/group64",
        receipt_sha256=reference["receipt_sha256"], record_sha256=binding["sha256"], parent16_record_sha256=record["parent16_record_sha256"],
        model_path=receipt["evidence_path"] + "/" + record["model_path"], model_files=files,
        **CONTROLS[source], expected_runtime=expected, runtime_guard_sha256=digest(expected))


def check_runtime(expected_runtime):
    _runtime(expected_runtime)
    require(sys.version == expected_runtime["python"], "Runtime Python differs")
    require(sys.executable == expected_runtime["executable"] and sys.platform == expected_runtime["platform"], "Runtime interpreter differs")
    require(all(getattr(sys.flags, k) == v for k, v in expected_runtime["flags"].items()), "Runtime flags differ")
    require({k: os.environ.get(k) for k in ENVIRONMENT_FIELDS} == expected_runtime["environment"], "Runtime environment differs")
    try:
        require({k: importlib.metadata.version(k) for k in VERSION_FIELDS} == expected_runtime["versions"], "Runtime versions differ")
        for path, value in expected_runtime["sources"].items():
            require(os.path.realpath(path) == path, "Runtime source path differs")
            with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
                require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Runtime source is not regular")
                actual = sha256()
                for chunk in iter(lambda: stream.read(1024**2), b""): actual.update(chunk)
            require(actual.hexdigest() == value, "Runtime source hash differs")
    except (OSError, importlib.metadata.PackageNotFoundError) as error:
        raise ProfileError("Runtime file/package missing") from error
    return digest(expected_runtime)
