"""Read recorded task uses and row bindings without opening historical evidence."""

import hashlib
import json
from pathlib import Path
import re


_MANIFESTS = Path(__file__).resolve().parents[3] / "honeypot/data/manifests"
RECORDS = _MANIFESTS / "exposure-history-additions-v1.json"
COVERAGE = _MANIFESTS / "exposure-history-sweep-v1.json"
ROW_HASH_RULE = 'SHA256(UTF8(json.dumps(raw_APPS_row,sort_keys=True,separators=(",",":"),ensure_ascii=False)))'
RESTRICTED_USES = {"prompt_checker_development", "unresolved"}


def row_sha256(original_row):
    raw = json.dumps(original_row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _identity(value):
    match = re.fullmatch(r"apps:(train|test):(0|[1-9][0-9]*)", value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"Invalid known-use task identity: {value!r}")
    return match[1], int(match[2])


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid known-use SHA256: {value!r}")
    return value


def load_known_use(records_path=RECORDS, coverage_path=COVERAGE):
    """Return restricted uses keyed by (split, ID), input fingerprints and coverage.

    Source inventories and population membership do not establish task use.
    Absence from the returned map does not establish an unseen task.
    """
    records_path, coverage_path = Path(records_path).resolve(), Path(coverage_path).resolve()
    records_raw, coverage_raw = records_path.read_bytes(), coverage_path.read_bytes()
    records, coverage = json.loads(records_raw), json.loads(coverage_raw)
    binding = coverage["archive_to_apps_binding"]
    if binding["apps_row_hash_rule"] != ROW_HASH_RULE:
        raise ValueError("Unsupported known-use APPS row hash rule")
    mappings, expected = {}, {}
    for item in binding["bindings"]:
        key = _identity(item["task_id"])
        archive, apps = _hash(item["archive_row_sha256"]), _hash(item["apps_row_sha256"])
        if (mappings.get((key, archive), apps) != apps or expected.get(key, apps) != apps):
            raise ValueError(f"Conflicting known-use row binding: {item['task_id']}")
        mappings[key, archive], expected[key] = apps, apps
    tasks = {}
    for item in records["observations"]:
        key, digest = _identity(item["task_id"]), _hash(item["row_sha256"])
        digest = mappings.get((key, digest), digest)
        if expected.get(key, digest) != digest:
            raise ValueError(f"Conflicting known-use row hash: {item['task_id']}")
        expected[key] = digest
        if item["use"] in RESTRICTED_USES:
            task = tasks.setdefault(key, {"row_sha256": digest, "uses": []})
            task["uses"] = sorted(set(task["uses"]) | {item["use"]})
    return {"tasks": tasks,
            "provenance": {"records_path": str(records_path), "records_sha256": hashlib.sha256(records_raw).hexdigest(),
                           "coverage_path": str(coverage_path), "coverage_sha256": hashlib.sha256(coverage_raw).hexdigest(),
                           "apps_row_hash_rule": ROW_HASH_RULE},
            "coverage": {key: coverage[key] for key in
                         ("cutoff_date", "history_certainty", "inspection_scope", "interpretation", "limits")}}
