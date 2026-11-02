"""Stage 02 input validation and pure allocation policy; no publication or activation.

See docs/v1/stage-02-plan.md. Task records reference unchanged input artifacts, keeping
every stored case and qualification evidence in place. Load inputs before allocate;
neither function executes programs or counts new tokens.
"""

import csv
import io
from itertools import cycle, islice
import json
from pathlib import Path
from urllib.parse import urlsplit

from ..artifacts import sha256
from .known_use import COVERAGE, RECORDS, RESTRICTED_USES, load_known_use, row_sha256
from ..stage01.qualified_tasks import iter_qualified_tasks


SEED = "ropot-cohort-v1"
DIFFICULTIES = ("introductory", "interview")
PROFILE = {"statement_tokens": 700, "reference_tokens_max": 500,
           "input_max_bytes": 16_384, "input_total_bytes": 65_536}
QUOTAS = {"evaluation": (8, 24), "calibration": (4, 12),
          "correction_replay": (30, 92), "development": (12, 38),
          "construction": (143, 445)}
RESERVE_CYCLE = ("construction",) * 7 + ("correction_replay",) * 3 + ("calibration",) * 2 + ("evaluation",) * 4
PHASES = ("shared_honest_pretraining", "twin_continuation")
BLOCKS = ("initial", "reserve", "deferred")


def load_inputs(input_dir, analysis_dir, known_use_records=RECORDS, known_use_coverage=COVERAGE):
    """Join the export to its analysis, validating bytes and shared qualification.

    Measurements come from the fingerprinted analysis CSV; no local tokenizer or
    original qualification directory is needed. Return compact task
    records, input fingerprints, known-use coverage and historical gate metadata.
    """
    input_dir, analysis_dir = Path(input_dir), Path(analysis_dir)
    known_use = load_known_use(known_use_records, known_use_coverage)
    manifest_raw = (input_dir / "manifest.json").read_bytes()
    summary_raw = (analysis_dir / "summary.json").read_bytes()
    csv_raw = (analysis_dir / "tasks.csv").read_bytes()
    manifest, summary = json.loads(manifest_raw), json.loads(summary_raw)
    exported, analyzed = manifest["provenance"], summary["provenance"]
    csv_hash = sha256(csv_raw)
    if analyzed.get("tasks_csv_sha256") != csv_hash:
        raise ValueError("Analysis tasks.csv hash missing or mismatch; rerun analyze")
    archive = exported["qualification_manifest"]
    if (archive != "qualification-manifest.json"
            or not (input_dir / archive).resolve().is_relative_to(input_dir.resolve())):
        raise ValueError("Invalid qualification manifest path")
    qualification_raw = (input_dir / archive).read_bytes()
    qualification = json.loads(qualification_raw)
    qualification_hash = sha256(qualification_raw)
    if not (qualification_hash == exported["input_manifest_sha256"] == analyzed["input_manifest_sha256"]):
        raise ValueError("Qualification manifest hash mismatch")
    for key in ("dataset", "revision", "qualification", "required_next_gate", "runtime", "workers"):
        if manifest[key] != qualification[key]:
            raise ValueError(f"Qualification provenance mismatch: {key}")
    if (analyzed["qualification_runtime"] != manifest["runtime"]
            or analyzed["workers"] != manifest["workers"]
            or summary["selection"] != manifest["selection"]
            or manifest["selection"]["split"] != "test"):
        raise ValueError("Analysis qualification provenance mismatch")
    listed = set(manifest["tasks"])
    if (set(exported["task_sha256"]) != listed or exported["task_sha256"] != analyzed["task_sha256"]):
        raise ValueError("Analysis/export task hashes mismatch")
    rows = {}
    numeric = (*PROFILE, *(f"reference_{n}_tokens" for n in (1, 2, 3)),
               *(f"reference_{n}_index" for n in (1, 2, 3)), "cases", "visible_cases", "nonvisible_cases")
    for row in csv.DictReader(io.StringIO(csv_raw.decode("utf-8"), newline="")):
        key = (row["split"], int(row["id"]))
        if key[0] != "test" or key in rows:
            raise ValueError("Invalid or duplicate analysis identity")
        for field in numeric:
            row[field] = int(row[field])
            if row[field] < 0:
                raise ValueError(f"Invalid analysis measurement: {field}")
        row["id"] = key[1]
        rows[key] = row
    tasks = []
    # Join qualified artifacts to checked analysis rows and historical exposure records.
    for relative, raw, task in iter_qualified_tasks(input_dir, manifest):
        if task["split"] != "test":
            raise ValueError("Cohort input must contain test tasks only")
        task_hash = sha256(raw)
        if task_hash != exported["task_sha256"][relative]:
            raise ValueError(f"Input task hash mismatch: {relative}")
        row = rows.pop((task["split"], task["id"]), None)
        if row is None:
            raise ValueError(f"Missing analysis identity: {relative}")
        if (task["difficulty"] not in DIFFICULTIES or row["difficulty"] != task["difficulty"]
                or row["host"] != (urlsplit(task["url"]).hostname or "")):
            raise ValueError(f"Analysis task metadata mismatch: {relative}")
        indices = [row[f"reference_{n}_index"] for n in (1, 2, 3)]
        if (indices != task["passing_reference_indices"] or len(set(indices)) != 3
                or row["reference_tokens_max"] != max(row[f"reference_{n}_tokens"] for n in (1, 2, 3))):
            raise ValueError(f"Analysis passing references mismatch: {relative}")
        exposure = known_use["tasks"].get((task["split"], task["id"]))
        if exposure and row_sha256(task["original_row"]) != exposure["row_sha256"]:
            raise ValueError(f"Known-use original-row hash mismatch: {relative}")
        tasks.append({**row, "path": relative, "sha256": task_hash,
                      "known_uses": exposure["uses"] if exposure else []})
    if rows or not tasks or len(tasks) != summary["counts"]["tasks"]:
        raise ValueError("Analysis identities/counts disagree with export")
    return {"tasks": sorted(tasks, key=lambda t: t["id"]),
            "provenance": {"input_manifest_sha256": sha256(manifest_raw),
                           "analysis_summary_sha256": sha256(summary_raw), "analysis_tasks_sha256": csv_hash,
                           "qualification_manifest_sha256": qualification_hash, "tokenizer": analyzed["tokenizer"],
                           "known_use": known_use["provenance"]},
            "known_use_coverage": known_use["coverage"],
            "required_next_gate": manifest["required_next_gate"]}


def profile_failures(task):
    """Inclusive limits in declared order; no case-count or execution-time cutoff."""
    return [metric for metric, limit in PROFILE.items() if task[metric] > limit]


def hash_key(task):
    return sha256(f"{SEED}|test|{task['id']}".encode("utf-8")), task["id"]


def interleave(introductory, interview):
    """Preserve each list's order and round each prefix's introductory quota half up."""
    total, intro, other = len(introductory) + len(interview), 0, 0
    result = []
    for prefix in range(1, total + 1):
        target = (2 * prefix * len(introductory) + total) // (2 * total)
        if intro < target:
            result.append(introductory[intro])
            intro += 1
        else:
            result.append(interview[other])
            other += 1
    return result


def allocate(tasks):
    """Freeze all ownership and phases, reporting fixed per-stratum shortfalls.

    Requires records validated by load_inputs; identity and difficulty checks
    belong to that input boundary.
    Groups contain IDs in three independently interleaved blocks. Consumers extend
    active prefixes by concatenating initial + reserve; deferred stays inactive
    until a separately reviewed profile broadening. No active limit is chosen here.
    Construction activation uses construction_phases; owner groups show membership.
    """
    records, sequential = {}, {metric: 0 for metric in PROFILE}
    for task in tasks:
        failures = profile_failures(task)
        records[task["id"]] = {**task, "defer_reasons": failures}
        records[task["id"]].pop("phase", None)
        records[task["id"]].pop("redirected_from", None)
        for metric in PROFILE:
            if metric in failures:
                break
            sequential[metric] += 1
    buckets = {owner: {block: [[], []] for block in BLOCKS} for owner in QUOTAS}
    phases = {phase: {block: [[], []] for block in BLOCKS} for phase in PHASES}
    # Allocate initial quotas, then distribute reserve and deferred tasks by difficulty.
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        ordered = sorted((t for t in records.values() if t["difficulty"] == difficulty), key=hash_key)
        eligible = (task for task in ordered if not task["defer_reasons"])
        deferred = [t for t in ordered if t["defer_reasons"]]

        def assign(task, owner, block):
            task.update(owner=owner, block=block)
            buckets[owner][block][difficulty_index].append(task["id"])

        for owner, quotas in QUOTAS.items():
            for task in islice(eligible, quotas[difficulty_index]):
                assign(task, owner, "initial")
        reserve_owners = cycle(RESERVE_CYCLE)
        for block, population in (("reserve", eligible), ("deferred", deferred)):
            for task in population:
                assign(task, next(reserve_owners), block)
    for task in records.values():
        if (RESTRICTED_USES.intersection(task.get("known_uses", ()))
                and task["owner"] not in ("construction", "development")):
            owner, block = task["owner"], task["block"]
            di = DIFFICULTIES.index(task["difficulty"])
            buckets[owner][block][di].remove(task["id"])
            buckets["construction"][block][di].append(task["id"])
            task.update(owner="construction", redirected_from=owner)
    for di in range(len(DIFFICULTIES)):
        phase_cursor = di
        for block in BLOCKS:
            construction = buckets["construction"][block][di]
            construction.sort(key=lambda identifier: hash_key(records[identifier]))
            for identifier in construction:
                phase = PHASES[phase_cursor % 2]
                records[identifier]["phase"] = phase
                phases[phase][block][di].append(identifier)
                phase_cursor += 1
    shortfalls = {owner: {difficulty: max(0, quota[di] - len(buckets[owner]["initial"][di]))
                         for di, difficulty in enumerate(DIFFICULTIES)} for owner, quota in QUOTAS.items()}
    flatten = lambda groups: {name: {block: interleave(*lists) for block, lists in blocks.items()}
                              for name, blocks in groups.items()}
    return {"tasks": sorted(records.values(), key=lambda t: t["id"]), "groups": flatten(buckets),
            "construction_phases": flatten(phases), "shortfalls": shortfalls,
            "sequential_remaining": sequential}
