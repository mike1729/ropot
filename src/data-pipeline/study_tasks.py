"""Load reviewed cohort tasks for Stage 03 experiments without executing programs.

See docs/stage-03-plan.md. The reviewed cohort manifest is pinned by SHA-256; the
test-pool manifest and every loaded task must match the digests it records.
Activation exposes only fixed initial prefixes, and the Linux reference gate stays
pending. Records and identities are fresh JSON values; budgets, persistence, model
calls and program execution belong to later batches.
"""

import json
from pathlib import Path
import re

from .cohort import sha256
from .publish_cohort import OUTPUT


COHORT_SHA256 = "60708dc74d386a8bb337a194a15a1c355c2a81f79768d8f876fb7a8f13148ff8"
GROUPS = {"development_smoke": ("development", None),
          "shared_pretraining_probe": ("construction", "shared_honest_pretraining"),
          "continuation_probe": ("construction", "twin_continuation")}
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
    """Return the pinned cohort, its test-pool manifest and the task directory.

    Cohort bytes must match COHORT_SHA256 before decoding, and the test-pool
    manifest must match the cohort's recorded input digest. ``test_pool_dir``
    replaces the recorded input directory after relocation; both checks still apply.
    Allocation and provenance are returned as published, without recomputation.
    """
    raw = (Path(cohort_dir) / "manifest.json").read_bytes()
    digest = sha256(raw)
    if digest != COHORT_SHA256:
        raise ValueError("Cohort manifest differs from the reviewed publication")
    cohort = json.loads(raw)
    test_pool_dir = Path(cohort["provenance"]["input_dir"] if test_pool_dir is None else test_pool_dir)
    pool_raw = (test_pool_dir / "manifest.json").read_bytes()
    if sha256(pool_raw) != cohort["provenance"]["input_manifest_sha256"]:
        raise ValueError("Test-pool manifest differs from the cohort's input")
    return {"sha256": digest, "cohort": cohort, "test_pool": json.loads(pool_raw), "test_pool_dir": test_pool_dir}


def activate(loaded, groups, *, construction_per_phase=None):
    """Return task bindings for a nonempty subset of GROUPS, in GROUPS then cohort order.

    Development smoke takes the first ``candidate_budgets`` smoke tasks; each probe
    takes the first ``probe_per_phase`` tasks, or opts into the recorded feasibility
    bound with ``construction_per_phase``. Both use its phase's initial block. Every
    selected task must be an undeferred test task with the group's owner and phase.
    Bindings copy the cohort records unchanged, adding only their group name.
    """
    requested = list(groups)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= GROUPS.keys():
        raise ValueError(f"Activate distinct groups from {list(GROUPS)}")
    cohort, tasks = loaded["cohort"], []
    records = {record["id"]: record for record in cohort["tasks"]}
    budgets = cohort["candidate_budgets"]
    if construction_per_phase is not None:
        bound = budgets["construction"].get("feasibility_per_phase")
        if (type(construction_per_phase) is not int or construction_per_phase <= 0
                or type(bound) is not int or bound <= 0 or construction_per_phase != bound
                or not any(GROUPS[name][1] is not None for name in requested)):
            raise ValueError("Invalid construction prefix selection")
    for name, (owner, phase) in GROUPS.items():
        if name not in requested:
            continue
        queue, size = ((cohort["queues"][owner]["initial"], budgets[owner]["smoke"]["total"]) if phase is None
                       else (cohort["construction_phases"][phase]["initial"], construction_per_phase
                             if construction_per_phase is not None else budgets[owner]["probe_per_phase"]))
        if type(size) is not int or not 0 < size <= len(queue):
            raise ValueError(f"Insufficient {name} prefix")
        for identifier in queue[:size]:
            record = records[identifier]
            if (record["split"], record["owner"], record.get("phase"), record["block"],
                    record["defer_reasons"]) != ("test", owner, phase, "initial", []):
                raise ValueError(f"Ineligible {name} task: {identifier}")
            tasks.append({**record, "group": name})
    return snapshot({"cohort_sha256": loaded["sha256"],
                     "groups": [name for name in GROUPS if name in requested], "tasks": tasks,
                     **({"construction_per_phase": construction_per_phase} if construction_per_phase is not None else {})})


def _require_activation(loaded, activation):
    """Accept only an activation equal to the one the pinned cohort produces."""
    if activation != activate(loaded, activation["groups"],
                              construction_per_phase=activation.get("construction_per_phase")):
        raise ValueError("Activation differs from the pinned cohort prefixes")


def load_task(loaded, activation, task_id):
    """Return (binding, decoded task) for one activated task, reading only its file.

    Both pinned manifests must bind the task path to the same digest, the resolved
    path must stay inside the test pool and the bytes must match before decoding.
    Split, integer ID and difficulty must equal the cohort record. The binding is
    a fresh copy of the activation record; the task payload is returned separately.
    """
    _require_activation(loaded, activation)
    record = next((task for task in activation["tasks"]
                   if type(task_id) is int and task["id"] == task_id), None)
    if record is None:
        raise ValueError(f"Task is not activated: {task_id!r}")
    if loaded["test_pool"]["provenance"]["task_sha256"].get(record["path"]) != record["sha256"]:
        raise ValueError(f"Cohort and test-pool bindings disagree: {record['path']}")
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


def experiment_record(loaded, experiment_id, activation, recipes):
    """Return a fresh, versioned record of one experiment's tasks and source recipes.

    Recipes are opaque JSON objects for a nonempty subset of SOURCES, kept separate
    per source. Reply, token and wall-time limits, their enforcement and persistence
    belong to batch 6; the Linux reference gate remains pending.
    """
    _require_activation(loaded, activation)
    if type(experiment_id) is not str or not experiment_id.strip():
        raise ValueError("Experiment ID must be a nonempty string")
    if (not isinstance(recipes, dict) or not recipes or not set(recipes) <= set(SOURCES)
            or not all(isinstance(recipe, dict) for recipe in recipes.values())):
        raise ValueError(f"Recipes must map a nonempty subset of {SOURCES} to JSON objects")
    cohort = loaded["cohort"]
    return snapshot({
        "schema_version": 1, "experiment_id": experiment_id,
        "cohort": {"manifest_sha256": loaded["sha256"],
                   "test_pool_manifest_sha256": cohort["provenance"]["input_manifest_sha256"]},
        "activation": activation,
        "reference_gate": {"status": "pending", "requirement": cohort["required_next_gate"]},
        "recipes": {source: recipes[source] for source in SOURCES if source in recipes},
    })


def candidate_identity(split, task_id, task_sha256, source, program):
    """Return a candidate's stable identity from its task, source and exact bytes.

    No cohort lookup, activation, normalization or correctness check applies, so
    fixture and train-row candidates share this form. Experiment, cohort and request
    links stay outside it; repeated identical bytes remain separate attempts.
    """
    if (split not in ("train", "test") or type(task_id) is not int or type(task_sha256) is not str
            or DIGEST.fullmatch(task_sha256) is None or source not in SOURCES or type(program) is not bytes):
        raise ValueError("Invalid candidate identity fields")
    return {"split": split, "id": task_id, "task_sha256": task_sha256, "source": source,
            "program_sha256": sha256(program)}
