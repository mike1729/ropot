"""Freeze the allocation-selected view; reuse it without re-reading evidence.

Allocation v1 supplies approved, eligible ``candidate_order`` prefixes by phase,
the ordered ``development`` panel and advancement_rule.
The independently reviewed allocation pin is the approval boundary.
"""

from importlib import import_module
import json
from pathlib import Path

from . import data, interface


artifacts = import_module("src.data-pipeline.artifacts")
SCHEMA_VERSION = "monitor-training-prepared-v1"
QUOTAS = {"shared_honest": 6, "twin_continuation": 8, "development": 20}
PHASES = {"shared_honest": "shared_honest_pretraining", "twin_continuation": "twin_continuation"}
EXCLUDED_PARENT_SHA256 = "bf6d65a661058bd1162cc408a169237112763800a860a2e862f4dd612530ac4c"


def prepare(export_root, output_dir, *, expected_export_sha256, expected_allocation_sha256):
    """Verify selection and publish deterministic prepared.json; return its digest."""
    root = Path(export_root).resolve()
    manifest = data._read_manifest(root, expected_export_sha256)
    reference = manifest["allocation"]
    if reference["sha256"] != expected_allocation_sha256:
        raise ValueError("Export allocation differs from the independent allocation pin")
    allocation = data._read_pinned(root, reference["path"], expected_allocation_sha256)
    groups = _selection(allocation, manifest["selection_failures"])
    _check_groups(manifest, groups)
    rows = data._load_groups(root, manifest["groups"])
    targets = _target_table()
    digests, duplicates = _inspect_rows(rows, targets)
    record = dict(schema_version=SCHEMA_VERSION, export_sha256=expected_export_sha256,
                  allocation=reference, generation_recipe=manifest["generation_recipe"],
                  groups=groups, message_sha256=digests, target_table=targets,
                  duplicates=duplicates, source_sha256={
                      name: artifacts.sha256((Path(__file__).parent / name).read_bytes())
                      for name in ("data.py", "interface.py", "prepare.py")})
    with artifacts.publication_directory(output_dir) as staging:
        artifacts.write_json(staging / "prepared.json", record)
        digest = artifacts.sha256((staging / "prepared.json").read_bytes())
    return digest


def _selection(allocation, failed_task_ids):
    if allocation["advancement_rule"] != "first-qualified-v1":
        raise ValueError("Unsupported advancement rule")
    failures = set(failed_task_ids)
    groups = {group: [task_id for task_id in allocation["candidate_order"][phase] if task_id not in failures][:QUOTAS[group]]
              for group, phase in PHASES.items()}
    groups["development"] = allocation["development"]
    for group, quota in QUOTAS.items():
        if len(groups[group]) != quota:
            raise ValueError(f"Selection quota/reserve exhausted for {group}")
    identifiers = [task_id for selected in groups.values() for task_id in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Selected task IDs must be unique across groups")
    return groups


def _check_groups(manifest, groups):
    actual = {name: [entry["task_id"] for entry in entries] for name, entries in manifest["groups"].items()}
    if actual != groups:
        raise ValueError("Export groups differ from the ordered selected task IDs")


def _target_table():
    return {role: {branch: interface.target_for(role, branch) for branch in interface.BRANCHES}
            for role in interface.ROLES}


def _message_digest(row):
    messages = interface.build_messages(row["statement"], row["program"])
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return artifacts.sha256(raw)


def _inspect_rows(grouped_rows, targets):
    digests, by_message = {}, {}
    for group in QUOTAS:
        digests[group] = []
        for row in grouped_rows[group]:
            if row["role"] == "benign":
                parents = {artifacts.sha256(row["program"].encode("utf-8")),
                           *row["receipt"]["provenance"]["ancestor_artifact_sha256s"]}
                if EXCLUDED_PARENT_SHA256 in parents:
                    raise ValueError("Excluded benign parent or ancestry")
            digest = _message_digest(row)
            digests[group].append(digest)
            previous = by_message.setdefault(digest, [])
            if previous:
                other = previous[0]
                if (group == "development") != (other["group"] == "development"):
                    raise ValueError("Identical message crosses construction/development boundary")
                if targets[row["role"]] != targets[other["role"]]:
                    raise ValueError("Identical messages have conflicting targets")
            previous.append(dict(group=group, task_id=row["task_id"], role=row["role"]))
    duplicates = [dict(message_sha256=digest, rows=rows) for digest, rows in by_message.items() if len(rows) > 1]
    return digests, duplicates


def load_prepared(prepared_root, export_root, group_names, *, expected_prepared_sha256):
    """Load requested frozen groups, checking current targets/messages, without allocation reads."""
    record = data._read_pinned(Path(prepared_root).resolve(), "prepared.json", expected_prepared_sha256)
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared schema version")
    if record["target_table"] != _target_table():
        raise ValueError("Prepared target table differs from current targets")
    rows = data.load_export(export_root, group_names, expected_export_sha256=record["export_sha256"])
    for group, values in rows.items():
        if [_message_digest(row) for row in values] != record["message_sha256"][group]:
            raise ValueError("Prepared message digests differ from current messages")
    return rows
