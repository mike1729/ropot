"""Load reviewed cohort tasks for Stage 03 experiments without executing programs.

See docs/stage-03-plan.md. The reviewed cohort manifest is pinned by SHA-256; the
file bytes of every loaded task must match its recorded digest.
Stage 02 saves every group once; this module can read only construction and
development tasks. Bindings are shared read-only metadata. The Docker reference
gate stays pending; budgets, persistence, model calls and program execution belong
to later batches.
"""

import json
from pathlib import Path
import re

from ..stage02.cohort import sha256
from ..stage02.publish_cohort import OUTPUT
from ..stage02.published_cohort import load_cohort as load_published_cohort


COHORT_SHA256 = "0c1ab106c67cabdbffd2b9c2dc58b0481a793b1b3cf074f9078ae6c398a2aad8"
SOURCES = ("actor", "synthetic")
DIGEST = re.compile(r"[0-9a-f]{64}")


def load_cohort(cohort_dir=OUTPUT, test_pool_dir=None):
    """Load only Stage 02's saved construction/development groups and task index.

    Initial and reserve blocks are available in their saved order. Deferred tasks
    remain unavailable. Treat the returned metadata and index as read-only.
    """
    return load_published_cohort(cohort_dir, test_pool_dir, expected_sha256=COHORT_SHA256,
                                 owners=("construction", "development"))


def load_task(loaded, task_id):
    """Return (binding, decoded task) for one permitted task, reading only its file.

    Stage 02 validated the pinned metadata. The requested ID must belong to the
    restricted index, the resolved path must stay inside the test pool and file
    bytes must match before decoding. The returned binding is shared read-only
    metadata; the task payload is decoded afresh.
    """
    if type(task_id) is not int or task_id not in loaded["tasks_by_id"]:
        raise ValueError(f"Task is not in the saved construction/development groups: {task_id!r}")
    record = loaded["tasks_by_id"][task_id]
    root = Path(loaded["test_pool_dir"]).resolve()
    path = (root / record["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Task path leaves the test pool: {record['path']}")
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
