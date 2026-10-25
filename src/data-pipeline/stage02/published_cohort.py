"""Load and index a reviewed Stage 02 publication without reading task files."""

import json
from pathlib import Path

from .cohort import sha256
from .publish_cohort import OUTPUT


def load_cohort(cohort_dir=OUTPUT, test_pool_dir=None, *, expected_sha256, owners):
    """Read saved groups for the requested owners and index their available tasks.

    The consumer supplies the reviewed cohort digest. ``test_pool_dir`` overrides
    the recorded directory after relocation. Task bytes and resolved paths must
    still be checked when files are read. Initial/reserve lists are consumed as
    published; allocation and group sizes are never recomputed. Deferred tasks
    remain unavailable. Treat the returned metadata and records as read-only.
    """
    raw = (Path(cohort_dir) / "manifest.json").read_bytes()
    digest = sha256(raw)
    if digest != expected_sha256:
        raise ValueError("Cohort manifest differs from the reviewed publication")
    cohort = json.loads(raw)
    test_pool_dir = Path(cohort["provenance"]["input_dir"] if test_pool_dir is None else test_pool_dir)
    pool_raw = (test_pool_dir / "manifest.json").read_bytes()
    if sha256(pool_raw) != cohort["provenance"]["input_manifest_sha256"]:
        raise ValueError("Test-pool manifest differs from the cohort's input")
    pool, records = json.loads(pool_raw), {}
    for record in cohort["tasks"]:
        identifier = record["id"]
        if type(identifier) is not int or identifier in records:
            raise ValueError("Cohort task IDs must be distinct integers")
        records[identifier] = record
    groups = {owner: {block: cohort["groups"][owner][block] for block in ("initial", "reserve")}
              for owner in owners}
    tasks = {}
    for owner, blocks in groups.items():
        for block, identifiers in blocks.items():
            for identifier in identifiers:
                record = records.get(identifier)
                if (type(identifier) is not int or identifier in tasks or record is None or
                        (record["split"], record["owner"], record["block"], record["defer_reasons"])
                        != ("test", owner, block, [])):
                    raise ValueError(f"Ineligible saved {owner}/{block} task: {identifier}")
                if pool["provenance"]["task_sha256"].get(record["path"]) != record["sha256"]:
                    raise ValueError(f"Cohort and test-pool bindings disagree: {record['path']}")
                tasks[identifier] = record
    phases = ({phase: {block: blocks[block] for block in ("initial", "reserve")}
               for phase, blocks in cohort["construction_phases"].items()} if "construction" in groups else {})
    return {"sha256": digest, "test_pool_manifest_sha256": sha256(pool_raw),
            "required_next_gate": cohort["required_next_gate"], "groups": groups,
            "construction_phases": phases, "test_pool_dir": test_pool_dir, "tasks_by_id": tasks}
