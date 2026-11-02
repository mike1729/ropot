"""Materialize qualified APPS test tasks for cohort allocation, without execution.

Copy task bytes into an independent dataset; preserve the original pool and archive
its manifest as historical evidence. The active manifest describes only this export
and retains legacy gate metadata as provenance only. Inputs must remain stable;
publication assumes one publisher and refuses an existing destination.
"""

import json
from pathlib import Path

from .. import source_hashes
from ..artifacts import publication_directory, sha256, write_json
from ..paths import QUALIFIED_POOL, TEST_POOL as OUTPUT
from .qualified_tasks import iter_qualified_tasks


def export_test(input_dir=QUALIFIED_POOL, output_dir=OUTPUT):
    """Publish independent test task copies and their provenance by directory rename."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if output_dir.resolve().is_relative_to(input_dir.resolve()):
        raise ValueError("Export must be outside the qualification pool")
    original = (input_dir / "manifest.json").read_bytes()
    source = json.loads(original)
    hashes = {}
    with publication_directory(output_dir) as staging:
        (staging / "tasks").mkdir()
        for relative, raw, task in iter_qualified_tasks(input_dir, source):
            if task["split"] == "test":
                (staging / relative).write_bytes(raw)
                hashes[relative] = sha256(raw)
        if not hashes:
            raise ValueError("No qualified test tasks")
        manifest = {
            "schema_version": 1,
            **{key: source[key] for key in ("dataset", "revision", "qualification", "required_next_gate", "runtime", "workers")},
            "tasks": sorted(hashes), "accepted_counts": {"train": 0, "test": len(hashes)},
            "selection": {"split": "test", "excluded_train_tasks": source["accepted_counts"].get("train", 0)},
            "provenance": {"input_dir": str(input_dir.resolve()),
                           "input_manifest_sha256": sha256(original),
                           "qualification_manifest": "qualification-manifest.json",
                           "task_sha256": dict(sorted(hashes.items())),
                           "source_sha256": source_hashes()},
        }
        (staging / "qualification-manifest.json").write_bytes(original)
        write_json(staging / "manifest.json", manifest)
    return manifest
