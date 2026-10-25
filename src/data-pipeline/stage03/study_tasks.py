"""Load reviewed cohort tasks for Stage 03 experiments without executing programs.

See docs/stage-03-plan.md. The reviewed cohort manifest is pinned by SHA-256; the
test-pool manifest and every loaded task must match the digests it records.
Stage 02 saves every group once; this module can read only construction and
development tasks. The Docker reference gate stays pending. Records and identities
are fresh JSON values; budgets, persistence, model calls and program execution
belong to later batches.
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


def snapshot(value):
    """Return a fresh JSON copy, rejecting values that strict JSON would change."""
    try:
        copy = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Value is not strict JSON") from error
    if copy != value:
        raise ValueError("Value does not round-trip through JSON")
    return copy


def load_cohort(cohort_dir=OUTPUT, test_pool_dir=None):
    """Load only Stage 02's saved construction/development groups and task index.

    Initial and reserve blocks are available in their saved order. Deferred tasks
    remain unavailable. Treat the returned metadata and index as read-only.
    """
    return load_published_cohort(cohort_dir, test_pool_dir, expected_sha256=COHORT_SHA256,
                                 owners=("construction", "development"))


def load_task(loaded, task_id):
    """Return (binding, decoded task) for one permitted task, reading only its file.

    Manifest bindings were checked at load time. The resolved path must stay
    inside the test pool and file bytes must match before decoding. Split, integer
    ID and difficulty must equal the indexed cohort record. Return a fresh binding
    separately from the decoded task payload.
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
    task = json.loads(raw)
    if (type(task["id"]) is not int or (task["split"], task["id"], task["difficulty"])
            != (record["split"], record["id"], record["difficulty"])):
        raise ValueError(f"Task identity differs from the cohort record: {record['path']}")
    return snapshot(record), task


def experiment_record(loaded, experiment_id, task_ids, recipes):
    """Return a fresh, versioned record of one experiment's tasks and source recipes.

    Task IDs are an experiment worklist within the saved groups, not a new split.
    Recipes are opaque JSON objects for a nonempty subset of SOURCES, kept separate
    per source. Reply, token and wall-time limits, their enforcement and persistence
    belong to batch 6; the Docker reference gate remains pending.
    """
    task_ids = list(task_ids)
    if (not task_ids or any(type(identifier) is not int or identifier not in loaded["tasks_by_id"]
                            for identifier in task_ids) or len(set(task_ids)) != len(task_ids)):
        raise ValueError("Experiment tasks must be distinct IDs from the saved construction/development groups")
    if type(experiment_id) is not str or not experiment_id.strip():
        raise ValueError("Experiment ID must be a nonempty string")
    if (not isinstance(recipes, dict) or not recipes or not set(recipes) <= set(SOURCES)
            or not all(isinstance(recipe, dict) for recipe in recipes.values())):
        raise ValueError(f"Recipes must map a nonempty subset of {SOURCES} to JSON objects")
    return snapshot({
        "schema_version": 2, "experiment_id": experiment_id,
        "cohort": {"manifest_sha256": loaded["sha256"],
                   "test_pool_manifest_sha256": loaded["test_pool_manifest_sha256"]},
        "task_ids": task_ids,
        "reference_gate": {"status": "pending", "requirement": loaded["required_next_gate"]},
        "recipes": {source: recipes[source] for source in SOURCES if source in recipes},
    })


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
