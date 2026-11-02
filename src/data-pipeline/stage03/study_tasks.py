"""Load reviewed cohort tasks for Stage 03 experiments without executing programs.

See docs/v1/stage-03-plan.md. The reviewed cohort manifest is pinned by SHA-256; the
file bytes of every loaded task must match its recorded digest.
Stage 02 saves every group once; this module can read only construction and
development tasks. Bindings are shared read-only metadata. Budgets, persistence
and model calls belong to later batches; existing reference qualification suffices.
"""

import json
from pathlib import Path
import re

from ..artifacts import sha256
from ..paths import COHORT as OUTPUT


COHORT_SHA256 = "63f1df94a21fff53eee8e7665bff8b1bee1aeb91bbefdd83d504eb32845e1492"
SOURCES = ("actor", "synthetic")
DIGEST = re.compile(r"[0-9a-f]{64}")


def load_cohort(cohort_dir=OUTPUT, task_dir=None):
    """Load only Stage 02's saved construction/development groups and task index.

    Initial and reserve blocks are available in their saved order. Deferred tasks
    remain unavailable. ``task_dir`` can relocate the published task files; paths
    and bytes are checked when each task is read. Treat metadata as read-only.
    """
    raw = (Path(cohort_dir) / "manifest.json").read_bytes()
    digest = sha256(raw)
    if digest != COHORT_SHA256:
        raise ValueError("Cohort manifest differs from the reviewed publication")
    cohort = json.loads(raw)
    records = {record["id"]: record for record in cohort["tasks"]}
    groups = {owner: {block: cohort["groups"][owner][block] for block in ("initial", "reserve")}
              for owner in ("construction", "development")}
    tasks = {identifier: records[identifier] for blocks in groups.values()
             for identifiers in blocks.values() for identifier in identifiers}
    phases = {phase: {block: blocks[block] for block in ("initial", "reserve")}
              for phase, blocks in cohort["construction_phases"].items()}
    return {"sha256": digest, "test_pool_manifest_sha256": cohort["provenance"]["input_manifest_sha256"],
            "groups": groups, "construction_phases": phases,
            "task_dir": Path(cohort_dir if task_dir is None else task_dir), "tasks_by_id": tasks}


def load_task(loaded, task_id):
    """Return (binding, decoded task) for one permitted task, reading only its file.

    Stage 02 validated the pinned metadata. The requested ID must belong to the
    restricted index, the resolved path must stay inside the published task root and file
    bytes must match before decoding. The returned binding is shared read-only
    metadata; the task payload is decoded afresh.
    """
    if type(task_id) is not int or task_id not in loaded["tasks_by_id"]:
        raise ValueError(f"Task is not in the saved construction/development groups: {task_id!r}")
    record = loaded["tasks_by_id"][task_id]
    root = Path(loaded["task_dir"]).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Task path leaves the published task root: {record['path']}")
    raw = path.read_bytes()
    if sha256(raw) != record["sha256"]:
        raise ValueError(f"Task bytes differ from the pinned digest: {record['path']}")
    return record, json.loads(raw)


def candidate_identity(split, task_id, task_sha256, source, program):
    """Return a candidate's stable identity from its task, source and exact bytes.

    No cohort lookup, normalization or correctness check applies, so
    fixture and train-row candidates share this form. Experiment, cohort and request
    links stay outside it; repeated identical bytes remain separate attempts.
    """
    if (split not in ("train", "test") or type(task_id) is not int or type(task_sha256) is not str
            or DIGEST.fullmatch(task_sha256) is None or source not in SOURCES or type(program) is not bytes):
        raise ValueError("Invalid candidate identity fields")
    return {"split": split, "id": task_id, "task_sha256": task_sha256, "source": source,
            "program_sha256": sha256(program)}
