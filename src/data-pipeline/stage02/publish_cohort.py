"""Publish frozen Stage 02 ownership and a capacity report, without activation.

Task paths are relative to provenance.input_dir; their bytes stay in that dataset.
Construction consumers use construction_phases, appending initial then reserve.
Deferred tasks require a reviewed profile change. Targets describe desired usable
data, not observed yields. Generation/retry ceilings and Linux verification remain
pending. Publication assumes stable inputs and a single publisher; directory rename
provides atomic visibility, without a power-loss durability guarantee.
"""

from collections import Counter
from pathlib import Path
import tempfile

from .. import source_hashes

from . import cohort
from ..stage01.analyze import OUTPUT as ANALYSIS_OUTPUT
from ..stage01.export_test import OUTPUT as TEST_OUTPUT
from .known_use import COVERAGE, RECORDS, RESTRICTED_USES
from ..stage01.qualify import write_json


OUTPUT = TEST_OUTPUT.parent / "02_cohort"
TASK_FIELDS = ("split", "id", "difficulty", "path", "sha256", "owner", "block", "phase",
               "defer_reasons", "known_uses", "redirected_from")


def capacity_report(allocation, loaded):
    """Derive task capacities, exclusions and shortfalls from final membership."""
    records = {task["id"]: task for task in allocation["tasks"]}

    def count(identifiers):
        tasks = [records[identifier] for identifier in identifiers]
        return {"tasks": len(tasks), "difficulty": {
            difficulty: sum(task["difficulty"] == difficulty for task in tasks)
            for difficulty in cohort.DIFFICULTIES}}

    def groups(queues):
        result = {}
        for name, blocks in queues.items():
            fast = blocks["initial"] + blocks["reserve"]
            result[name] = {block: count(blocks[block]) for block in cohort.BLOCKS}
            result[name].update(fast_profile=count(fast), total=count(fast + blocks["deferred"]))
        return result

    deferred = sum(bool(task["defer_reasons"]) for task in records.values())
    redirects = Counter(task["redirected_from"] for task in records.values() if "redirected_from" in task)
    return {
        "schema_version": 1,
        "counts": {**count(records), "fast_profile": len(records) - deferred, "deferred": deferred},
        "by_owner": groups(allocation["queues"]),
        "by_construction_phase": groups(allocation["construction_phases"]),
        "profile_deferrals": {metric: sum(metric in task["defer_reasons"] for task in records.values())
                              for metric in cohort.PROFILE},
        "sequential_remaining": allocation["sequential_remaining"],
        "initial_quota_shortfalls": allocation["shortfalls"],
        "known_use": {"matched_tasks": sum(bool(task["known_uses"]) for task in records.values()),
                      "redirected_tasks": sum(redirects.values()), "redirected_from": dict(sorted(redirects.items()))},
        "known_use_coverage": loaded["known_use_coverage"],
        "required_next_gate": loaded["required_next_gate"],
        "limitations": ["Known-use metadata has incomplete coverage; absence does not establish an unseen task.",
                        "Capacities count candidates, not accepted usable tasks or observed acquisition yields.",
                        "Generation/retry ceilings must be frozen before acquisition; no queues are activated here.",
                        "Content token limits do not establish complete monitor-example fit within 2,048 tokens."],
    }


def publish_cohort(input_dir=TEST_OUTPUT, analysis_dir=ANALYSIS_OUTPUT, output_dir=OUTPUT,
                   known_use_records=RECORDS, known_use_coverage=COVERAGE):
    """Validate inputs and atomically publish manifest.json and report.json.

    Refuse existing destinations and outputs within either dataset input. Preserve
    the loader's evidence and allocator's queues unchanged; copy no task artifacts.
    Return the manifest. Errors propagate without publishing a partial allocation.
    """
    input_dir, analysis_dir, output_dir = Path(input_dir), Path(analysis_dir), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if any(output_dir.resolve().is_relative_to(source.resolve()) for source in (input_dir, analysis_dir)):
        raise ValueError("Cohort output must be outside the test pool and analysis directory")
    loaded = cohort.load_inputs(input_dir, analysis_dir, known_use_records, known_use_coverage)
    allocation = cohort.allocate(loaded["tasks"])
    manifest = {
        "schema_version": 1,
        "tasks": [{key: task[key] for key in TASK_FIELDS if key in task} for task in allocation["tasks"]],
        "queues": allocation["queues"], "construction_phases": allocation["construction_phases"],
        "policy": {"seed": cohort.SEED, "profile": cohort.PROFILE,
                   "difficulty_order": list(cohort.DIFFICULTIES), "initial_quota_order": list(cohort.QUOTAS),
                   "initial_quotas": {owner: dict(zip(cohort.DIFFICULTIES, quotas))
                                      for owner, quotas in cohort.QUOTAS.items()},
                   "reserve_cycle": list(cohort.RESERVE_CYCLE), "block_order": list(cohort.BLOCKS),
                   "construction_phase_order": list(cohort.PHASES),
                   "construction_phase_start_by_difficulty": dict(zip(cohort.DIFFICULTIES, cohort.PHASES)),
                   "restricted_uses": sorted(RESTRICTED_USES),
                   "task_order": "SHA256(seed|test|id), then ID, within difficulty; construction rebuilt after redirects",
                   "interleaving": "proportional within each block; introductory prefix counts round half up",
                   "construction_activation_queue": "construction_phases"},
        "candidate_budgets": {"development": {"task_id_cap": 50,
                                              "smoke": {"total": 8, "introductory": 2, "interview": 6},
                                              "training_choices": 42},
                              "construction": {"probe_per_phase": 16, "feasibility_per_phase": 32}},
        "targets": {"status": "planned", "finalization": "after_32_candidates_per_phase_feasibility_probe",
                    "constraint": "final_targets_must_not_exceed_owned_capacity", "unit": "usable_matched_tasks",
                    "construction": {"per_phase": 128, "total": 256,
                                     "per_phase_difficulty": {"introductory": 31, "interview": 97},
                                     "nested_per_phase": [32, 64, 128], "optional_expansion_per_phase": 256,
                                     "optional_expansion_per_phase_difficulty": {"introductory": 62, "interview": 194}},
                    "correction_replay": 24, "calibration": 24, "evaluation": 48},
        "acquisition_ceilings": "pending",
        "known_use_coverage": loaded["known_use_coverage"], "required_next_gate": loaded["required_next_gate"],
        "provenance": {**loaded["provenance"], "input_dir": str(input_dir.resolve()),
                       "analysis_dir": str(analysis_dir.resolve()),
                       "source_sha256": source_hashes()},
    }
    report = capacity_report(allocation, loaded)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "report.json", report)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    return manifest
