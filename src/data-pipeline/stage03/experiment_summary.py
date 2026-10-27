"""Batch 8: summarize trusted coordinator JSON, without reading execution evidence.

Call summarize(root) on immediate NNNN-lowercase-hyphenated experiment directories.
Save its JSON-serializable result outside root. See docs/stage-03-batch-08-plan.md.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


COUNTERS = ("committed_replies", "committed_tokens", "known_submissions",
            "known_submissions_lower_bound", "observed_tokens")
PHASES = ("loading", "preparation", "generation", "verification", "overall")
TERMINAL = {"completed", "error", "cancelled"}
NAME = re.compile(r"([0-9]{4})-[a-z0-9]+(?:-[a-z0-9]+)*")


def read_record(path, problems):
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        json.dumps(value, allow_nan=False)
        return value, False, sha256(raw).hexdigest()
    except FileNotFoundError:
        problems.append(dict(file=path.name, error="absent"))
        return None, True, None
    except (OSError, ValueError, RecursionError) as error:
        problems.append(dict(file=path.name, error=str(error)))
        return None, False, None


def aggregate(values, exact):
    known = [v for v in values if type(v) in (int, float) and v >= 0]
    return dict(recorded_total=sum(known) if len(known) == len(values) else None,
                known_subtotal=sum(known), complete=len(known) == len(values) and all(exact))


def summarize(root):
    """Scan numbered experiments; trust saved counters/verdicts, retain read errors.

    Only root naming/provenance mixing stops the report. Nonterminal observations
    are snapshots/lower bounds; their recorded commitments remain exact. Missing
    outcome means no dispatch unless a saved round contradicts that write order.
    An unreadable or demonstrably lost outcome leaves effort unknown.
    """
    directories = sorted(p for p in Path(root).iterdir() if p.is_dir())
    names = [NAME.fullmatch(p.name) for p in directories]
    if any(m is None for m in names) or len({m[1] for m in names}) != len(names):
        raise ValueError("Require unique four-digit prefixes and lowercase hyphenated experiment names")
    experiments, cells, provenance, cohorts = [], {}, set(), set()
    for directory in directories:
        problems = []
        inputs, _, inputs_sha256 = read_record(directory / "inputs.json", problems)
        outcome, absent, outcome_sha256 = read_record(directory / "outcome.json", problems)
        if absent and (directory / "round-0000.json").exists():
            problems.append(dict(file="outcome.json", error="Absent despite saved round-0000.json; effort unknown"))
            absent = False
        inputs = inputs or {}
        if type(inputs.get("fixtures")) is bool:
            provenance.add(inputs["fixtures"])
        if isinstance(inputs.get("cohort_sha256"), str) and inputs["cohort_sha256"]:
            cohorts.add(inputs["cohort_sha256"])
        worklist = inputs.get("worklist")
        input_ok = (inputs.get("source") in ("actor", "synthetic")
                    and isinstance(worklist, list) and all(isinstance(w, dict) and isinstance(w.get("binding"), dict)
                    and isinstance(w["binding"].get("split"), str) and type(w["binding"].get("id")) is int
                    and isinstance(w["binding"].get("sha256"), str) and w["binding"]["sha256"] for w in worklist))
        if not input_ok or type(inputs.get("fixtures")) is not bool:
            problems.append(dict(file="inputs.json", error="Unsupported or unavailable worklist/provenance"))
        if inputs.get("fixtures") is False and not (isinstance(inputs.get("cohort_sha256"), str) and inputs["cohort_sha256"]):
            problems.append(dict(file="inputs.json", error="Missing study cohort digest"))
        data = outcome or {}
        outcome_ok = (isinstance(data.get("status"), str) and data["status"] in TERMINAL | {"running"}
            and all(isinstance(data.get(k), list) for k in ("attempts", "tasks", "rounds"))
            and all(isinstance(a, dict) and type(a.get("attempt")) is int and type(a.get("position")) is int
                    and isinstance(a.get("state"), str) for a in data["attempts"])
            and all(isinstance(t, dict) and type(t.get("position")) is int and isinstance(t.get("state"), str) for t in data["tasks"])
            and all(isinstance(r, dict) and isinstance(r.get("attempt_ids"), list) for r in data["rounds"]))
        if not outcome_ok and not absent:
            problems.append(dict(file="outcome.json", error="Unsupported or unavailable flat outcome format"))
        terminal = outcome_ok and data["status"] in TERMINAL
        counters = {k: 0 if absent else data.get(k) for k in COUNTERS}
        if "known_submissions_lower_bound" not in data and not absent:
            counters["known_submissions_lower_bound"] = data.get("known_submissions")
        for key, value in counters.items():
            if value is None:
                problems.append(dict(file="outcome.json", error=f"Unknown or unavailable recorded counter: {key}"))
            elif type(value) is not int or value < 0:
                problems.append(dict(file="outcome.json", error=f"Invalid recorded counter: {key}"))
                counters[key] = None
        timing = {phase: 0 for phase in PHASES} if absent else data.get("timing")
        if not isinstance(timing, dict):
            problems.append(dict(file="outcome.json", error="Unsupported or unavailable timing block"))
            timing = {}
        record = {k: inputs.get(k) for k in ("source", "fixtures", "cohort_sha256", "recipe",
                  "model_path", "model_manifest", "model_manifest_sha256")}
        record.update(experiment=directory.name, status="not_run" if absent else data["status"] if outcome_ok else "unresolved",
            inputs_sha256=inputs_sha256, outcome_sha256=outcome_sha256,
            attempt_count=len(data["attempts"]) if outcome_ok else None,
            terminal=terminal, worklist_size=len(worklist) if input_ok else None, counters=counters, timing=timing,
            usage_complete=True if absent else data.get("usage_complete"),
            observations="final" if terminal or absent else "snapshot/lower_bound",
            task_state_counts=dict(Counter(t["state"] for t in data["tasks"])) if terminal else None,
            problems=problems)
        experiments.append(record)
        if input_ok:
            states = {t["position"]: t["state"] for t in data["tasks"]} if outcome_ok else {}
            for position, item in enumerate(worklist):
                binding = item["binding"]
                key = record["source"], binding["split"], binding["id"], binding["sha256"]
                cell = cells.setdefault(key, dict(source=key[0], split=key[1], id=key[2], sha256=key[3], experiments=[], passes=[]))
                attempts = [a for a in data["attempts"] if a["position"] == position] if outcome_ok else []
                state = "not_run" if absent else states.get(position, "unresolved") if terminal else "unresolved"
                cell["experiments"].append(dict(experiment=directory.name, binding=binding, state=state,
                    recorded_state=states.get(position), attempts=attempts))
                cell["passes"].extend(dict(experiment=directory.name, attempt=a["attempt"]) for a in attempts
                    if a["state"] == "passed" or isinstance(a.get("verification"), dict) and a["verification"].get("status") == "passed")
                cell["latest_state"] = state
    if len(provenance) > 1 or len(cohorts) > 1:
        raise ValueError("Cannot mix fixture/study experiments or different cohort digests")
    totals = {}
    for key in COUNTERS:
        exact = [True if key.startswith("committed_") else r["observations"] == "final"
                 and (key != "observed_tokens" or r["usage_complete"] is True) for r in experiments]
        totals[key] = aggregate([r["counters"][key] for r in experiments], exact)
    totals["timing"] = {phase: aggregate([r["timing"].get(phase) for r in experiments],
        [r["observations"] == "final" for r in experiments]) for phase in PHASES}
    totals["usage_complete"] = (all(r["usage_complete"] is True and r["observations"] == "final" for r in experiments)
                                and all(totals[k]["recorded_total"] is not None for k in COUNTERS))
    return dict(experiments=experiments, cells=list(cells.values()), totals=totals)
