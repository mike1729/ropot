"""Read the pipeline's pinned monitor-training-v1 export.

The producer constructs receipts from qualified bytes and generation records.
Training checks format compatibility, manifest/payload hashes and task identity, then
projects the qualified bytes verbatim.
The caller supplies the reviewed export pin and requested group names; returned
rows can be reused without reopening files or repeating qualification checks.
"""

from importlib import import_module
import json
from pathlib import Path

from .interface import ROLES


sha256 = import_module("src.data-pipeline.artifacts").sha256
SCHEMA_VERSION = import_module("src.data-pipeline.training_format").SCHEMA_VERSION


def load_export(export_root, group_names, *, expected_export_sha256):
    """Verify pinned bytes and return only the requested groups in caller order."""
    root = Path(export_root).resolve()
    manifest = _read_manifest(root, expected_export_sha256)
    return _load_groups(root, {name: manifest["groups"][name] for name in group_names})


def _read_manifest(root, expected_export_sha256):
    manifest = _read_pinned(root, "manifest.json", expected_export_sha256)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported export schema version")
    return manifest


def _load_groups(root, selected):
    grouped_rows = {}
    for name, entries in selected.items():
        rows = []
        for entry in entries:
            task = _read_pinned(root, entry["path"], entry["sha256"])
            if task["task_id"] != entry["task_id"]:
                raise ValueError("Payload task ID differs from its index entry")
            rows.extend(_rows_from_task(task))
        grouped_rows[name] = rows
    return grouped_rows


def _read_pinned(root, relative_path, expected_sha256):
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path leaves the export root")
    raw = path.read_bytes()
    if sha256(raw) != expected_sha256:
        raise ValueError(f"Artifact bytes differ from pinned digest: {relative_path}")
    return json.loads(raw.decode("utf-8"))


def _rows_from_task(task):
    """Project an already validated task, preserving its exact qualified text."""
    return [dict(task_id=task["task_id"], role=role,
                 statement=task["statement"], program=task["programs"][role]["program"],
                 receipt=task["programs"][role]["receipt"]) for role in ROLES]
