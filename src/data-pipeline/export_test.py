"""Materialize qualified APPS test tasks for cohort allocation, without execution.

Copy task bytes into an independent dataset; preserve the original pool and archive
its manifest as historical evidence. The active manifest describes only this export
and retains the required Inspect Linux verification gate. Inputs must remain stable;
publication assumes one publisher and refuses an existing destination.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile

from .download import digest
from .qualify import OUTPUT as QUALIFIED_POOL, write_json


OUTPUT = QUALIFIED_POOL.parent / "01_test_pool"


def export_test(input_dir=QUALIFIED_POOL, output_dir=OUTPUT):
    """Publish independent test task copies and their provenance by directory rename."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Export must be outside the qualification pool")
    original = (input_dir / "manifest.json").read_bytes()
    source = json.loads(original)
    counts, seen, hashes = Counter(), set(), {}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        (staging / "tasks").mkdir()
        for relative in source["tasks"]:
            path = Path(relative)
            if (path.parts != ("tasks", path.name) or path.suffix != ".json"
                    or not (input_dir / path).resolve().is_relative_to(input_dir.resolve())):
                raise ValueError(f"Invalid task path: {relative}")
            raw = (input_dir / path).read_bytes()
            task = json.loads(raw)
            split, identifier = task["split"], task["id"]
            key = (split, identifier)
            if (split not in ("train", "test") or type(identifier) is not int or key in seen
                    or path.name != f"{split}_{identifier}.json"):
                raise ValueError(f"Invalid or duplicate task identity: {relative}")
            seen.add(key)
            counts[split] += 1
            if split == "test":
                (staging / path).write_bytes(raw)
                hashes[relative] = hashlib.sha256(raw).hexdigest()
        if any(counts[s] != source["accepted_counts"].get(s, 0) for s in ("train", "test")):
            raise ValueError("Manifest accepted counts disagree with listed tasks")
        if not hashes:
            raise ValueError("No qualified test tasks")
        manifest = {
            "schema_version": 1,
            **{key: source[key] for key in ("dataset", "revision", "qualification", "required_next_gate", "runtime", "workers")},
            "tasks": sorted(hashes), "accepted_counts": {"train": 0, "test": counts["test"]},
            "selection": {"split": "test", "excluded_train_tasks": counts["train"]},
            "provenance": {"input_dir": str(input_dir.resolve()),
                           "input_manifest_sha256": hashlib.sha256(original).hexdigest(),
                           "qualification_manifest": "qualification-manifest.json",
                           "task_sha256": dict(sorted(hashes.items())),
                           "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}},
        }
        (staging / "qualification-manifest.json").write_bytes(original)
        write_json(staging / "manifest.json", manifest)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    return manifest
