"""Load and index a reviewed Stage 02 publication without reading task files."""

import json
from pathlib import Path

from .cohort import sha256
from .publish_cohort import OUTPUT


def load_cohort(cohort_dir=OUTPUT, task_dir=None, *, expected_sha256, owners):
    """Read saved groups for the requested owners and index their available tasks.

    Stage 02 validates metadata before publication; the consumer verifies only
    the reviewed cohort digest. ``task_dir`` overrides
    the publication root after task-file relocation. Task bytes and resolved paths must
    still be checked when files are read. Initial/reserve lists are consumed as
    published; allocation and group sizes are never recomputed. Deferred tasks
    remain unavailable. Treat the returned metadata and records as read-only.
    """
    raw = (Path(cohort_dir) / "manifest.json").read_bytes()
    digest = sha256(raw)
    if digest != expected_sha256:
        raise ValueError("Cohort manifest differs from the reviewed publication")
    cohort = json.loads(raw)
    task_dir = Path(cohort_dir if task_dir is None else task_dir)
    records = {record["id"]: record for record in cohort["tasks"]}
    groups = {owner: {block: cohort["groups"][owner][block] for block in ("initial", "reserve")}
              for owner in owners}
    tasks = {identifier: records[identifier] for blocks in groups.values()
             for identifiers in blocks.values() for identifier in identifiers}
    phases = ({phase: {block: blocks[block] for block in ("initial", "reserve")}
               for phase, blocks in cohort["construction_phases"].items()} if "construction" in groups else {})
    return {"sha256": digest, "test_pool_manifest_sha256": cohort["provenance"]["input_manifest_sha256"],
            "groups": groups,
            "construction_phases": phases, "task_dir": task_dir, "tasks_by_id": tasks}
