"""Private evidence snapshots; no generation, native execution or recovery."""

from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

from . import experiment_loop as loop, local_generation as generation, native_recovery as native
from . import study_tasks as study, verification
from . import precision_profiles as precision
from . import paired_report
from . import generation_journal as journal
from .generation_journal import Campaign, encoded, require, sync_directory, limits_checked

RUN_FIELDS = {"schema_version", "experiment_id", "cohort", "activation", "reference_gate",
              "recipes", "assigned_task_ids", "policy_sha256"}
BASE_UNITS = ("replies", "observed_generation_tokens", "reserved_output_tokens")
BASE_FIELDS = {"sources", "combined", "audit", "inventory_sha256"}


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def profile_identity(descriptor):
    return {**{k: descriptor[k] for k in ("source", "bits", "label", "receipt_sha256", "record_sha256", "runtime_guard_sha256")},
            "descriptor_sha256": digest(descriptor)}


def read_artifact(root, name, expected, size=None):
    path = root / name
    require(type(name) is str and not Path(name).is_absolute() and str(path.relative_to(root)) == name
            and path.resolve() == path, "Invalid artifact path")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as file:
        require(stat.S_ISREG(os.fstat(file.fileno()).st_mode), "Require regular artifact")
        raw = file.read()
    require(sha256(raw).hexdigest() == expected and (size is None or len(raw) == size), "Artifact changed")
    return raw


def _anchor(events, kind):
    anchors = [e for e in events if e["kind"] == kind]
    require(len(anchors) <= 1, "Duplicate report anchor")
    if not anchors:
        return None
    event = anchors[0]
    require(event["payload_sha256"] == digest(event["payload"]) and all(event["seq"] < e["seq"]
            for e in events if e["kind"] in ("generation_reserved", "native_intent")), "Invalid report anchor")
    return event["payload"]


def _key(task):
    return task["split"], task["id"], task.get("task_sha256", task.get("sha256"))


def panel_joint(tasks, selections):
    require(len(tasks) == 50 and len({_key(t) for t in tasks}) == 50, "Require fixed fifty tasks")
    passed = {(_key(s), s["source"]) for s in selections}
    count = sum(all((_key(t), s) in passed for s in study.SOURCES) for t in tasks)
    return {"numerator": count, "denominator": 50, "target_met": count >= 15}


def _session_costs(events, reservations, outcomes):
    batches = {e["seq"]: e for e in events if e["kind"] == "batch_reserved"}
    ledger = []
    for done in (e for e in events if e["kind"] == "batch_done"):
        intent, shared = batches[done["session_seq"]], done["shared"]
        members = [reservations[r["reservation_seq"]] for r in intent["rows"]]
        descriptor = intent["method_spec"]["admission"]["request_controls"]["precision_profile"]["descriptor"]
        timing = {k: shared[k] for k in ("elapsed_seconds", "work_seconds", "cleanup_seconds")}
        try:
            known = shared["resources_closed"] is True and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                for v in timing.values()) and timing["work_seconds"] <= timing["elapsed_seconds"] and timing["cleanup_seconds"] <= timing["elapsed_seconds"]
        except OverflowError:
            known = False
        ledger.append(dict(session_seq=intent["seq"], batch_done_seq=done["seq"], job_sha256=intent["job_sha256"],
            source=intent["method_spec"]["source"], bits=descriptor["bits"], precision_profile=profile_identity(descriptor),
            reservation_seqs=[e["seq"] for e in members], experiment_ids=sorted({e["experiment_id"] for e in members}),
            status=shared["status"], eligible=done["eligible"], resources_closed=shared["resources_closed"], reported_timing=timing,
            owned_elapsed_seconds=timing["elapsed_seconds"] if known else None,
            observed_generation_tokens=sum(outcomes[e["seq"]]["observed_generation_tokens"] for e in members)))
    return ledger


def _observed(reservations, outcomes, *, batch=False, sessions=None):
    values = [outcomes[e["seq"]] for e in reservations]
    batches = [v for v in values if "session_seq" in v]
    refs = sorted({v["session_seq"] for v in batches})
    tokens = sum(v["observed_generation_tokens"] for v in values)
    if batch:
        require(len(batches) == len(values), "Scalar outcome in batch run")
        return dict(observed_generation_tokens=tokens, lower_bound_requests=sum(not v["eligible"] for v in values),
            admitted_rows=sum(v["uid"] is not None for v in values), terminal_rows=sum(v["finish_reason"] in ("stop", "length") for v in values),
            generation_session_seqs=refs, shared_wall_allocation="unallocated")
    scalar = [v for v in values if "session_seq" not in v]
    owned = [s for s in sessions or [] if s["session_seq"] in refs]
    result = dict(observed_generation_tokens=tokens, lower_bound_requests=sum(not v["eligible"] for v in batches) + sum(
        v["status"] not in ("stop", "length") or v["response"] is None or not v["response"]["complete"]
        or v["response"]["cleanup_error"] is not None for v in scalar),
        unknown_response_requests=sum(v["response"] is None for v in scalar),
        known_elapsed_seconds=math.fsum([v["elapsed_seconds"] for v in scalar if v["elapsed_seconds"] is not None]
            + [s["owned_elapsed_seconds"] for s in owned if s["owned_elapsed_seconds"] is not None]),
        unknown_elapsed_requests=sum(v["elapsed_seconds"] is None for v in scalar))
    if batches:
        require(len(owned) == len(refs), "Missing generation session costs")
        result.update(generation_session_seqs=refs, unknown_elapsed_sessions=sum(s["owned_elapsed_seconds"] is None for s in owned))
    return result


def snapshot(campaign, *, selections=None):
    """Derive a fixed prefix while holding Campaign's lease; Docker reads prove logs."""
    outcomes = {e["seq"]: campaign.outcome(e["seq"]) for e in campaign.events if e["kind"] == "generation_reserved"}
    events = campaign.events
    current_policy = loop.policy_sha256()
    files = {}

    def add(name, expected, size=None):
        raw = read_artifact(campaign.root, name, expected, size)
        item = {"path": "artifacts/" + expected + (Path(name).suffix or ".bin"), "sha256": expected, "bytes": len(raw)}
        require(name not in files or files[name] == item, "Conflicting artifact binding")
        files[name] = item
        return raw

    panel, baseline = _anchor(events, "panel"), _anchor(events, "mechanics_baseline")
    activations = []
    if panel is not None:
        require(set(panel) == {"activation", "tasks"} and panel["activation"]["cohort_sha256"] == study.COHORT_SHA256,
                "Invalid panel cohort")
        tasks = panel["tasks"]
        require(len(tasks) == 50 and all(type(t["id"]) is int for t in tasks)
                and len({t["id"] for t in tasks}) == 50 and tasks == [t for t in panel["activation"]["tasks"]
                if t["id"] in {p["id"] for p in tasks}], "Invalid frozen panel")
        activations.append(panel["activation"])
    if baseline is not None:
        require(BASE_FIELDS <= set(baseline) <= BASE_FIELDS | {"elapsed", "limitations"}
                and set(baseline["sources"]) == set(study.SOURCES) and set(baseline["combined"]) == set(BASE_UNITS)
                and all(set(v) == set(BASE_UNITS) for v in baseline["sources"].values())
                and study.DIGEST.fullmatch(baseline["inventory_sha256"]), "Invalid mechanics baseline")
        for unit in BASE_UNITS:
            require(all(type(baseline["sources"][s][unit]) is int and baseline["sources"][s][unit] >= 0
                        for s in study.SOURCES) and type(baseline["combined"][unit]) is int
                    and baseline["combined"][unit] == sum(baseline["sources"][s][unit] for s in study.SOURCES),
                    "Invalid baseline sums")
        audit = baseline["audit"]
        require(set(audit) == {"path", "sha256", "bytes"} and type(audit["bytes"]) is int
                and audit["bytes"] >= 0, "Invalid baseline audit")
        add(audit["path"], audit["sha256"], audit["bytes"])
    collections, plans = {}, {}
    for event in (e for e in events if e["kind"] in ("collection_setup", "paired_plan")):
        payload = event["payload"]
        require(set(event) == {"seq", "kind", "payload", "payload_sha256"} and type(payload) is dict
                and type(payload.get("plan_id")) is str and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", payload["plan_id"])
                and digest(payload) == event["payload_sha256"], "Invalid collection event")
        target = collections if event["kind"] == "collection_setup" else plans
        require(payload["plan_id"] not in target, "Duplicate collection plan")
        target[payload["plan_id"]] = event
        if event["kind"] != "collection_setup": continue
        require(set(payload) == {"schema_version", "plan_id", "plan_sha256", "producer_sources", "protocol", "funding"}
                and type(payload["schema_version"]) is int and payload["schema_version"] == 1
                and precision._sha(payload["plan_sha256"]) and type(payload["producer_sources"]) is dict and payload["producer_sources"]
                and all(type(p) is str and p and str(Path(p)) == p and not Path(p).is_absolute() and ".." not in Path(p).parts
                        and precision._sha(h) for p, h in payload["producer_sources"].items()), "Invalid collection setup")
        item, funding = payload["protocol"], payload["funding"]
        require(type(item) is dict and set(item) == {"path", "sha256", "bytes"} and precision._sha(item["sha256"])
                and type(item["bytes"]) is int and item["bytes"] >= 0
                and item["path"] == f"audits/{item['sha256']}-paired-protocol.md", "Invalid collection protocol")
        add(item["path"], item["sha256"], item["bytes"])
        limits, reserved = Campaign.totals(events[:event["seq"] - 1])
        require(type(funding) is dict and set(funding) == {"reserved_before", "bound", "limits"}, "Invalid collection funding")
        limits_checked(funding["bound"]); limits_checked(funding["limits"])
        require(digest(funding["reserved_before"]) == digest(reserved) and digest(funding["limits"]) == digest({
            s: {u: max(limits[s][u], reserved[s][u] + funding["bound"][s][u]) for u in limits[s]} for s in study.SOURCES}), "Invalid collection funding")
    require(all(name in collections and collections[name]["seq"] < event["seq"]
                and collections[name]["payload"]["plan_sha256"] == event["payload_sha256"]
                for name, event in plans.items()), "Unbound paired plan")
    reservations = {e["seq"]: e for e in events if e["kind"] == "generation_reserved"}
    batches = {e["seq"]: e for e in events if e["kind"] == "batch_reserved"}
    completions = {e["session_seq"]: e for e in events if e["kind"] == "batch_done"}
    for event in events:
        if event["kind"] == "generation_done" and "session_seq" not in event:
            for name, evidence in event["evidence"].items():
                add(f"requests/{event['reservation_seq']:06d}/{name}", evidence["sha256"], evidence["bytes"])
        elif event["kind"] == "batch_done":
            for name, evidence in event["evidence"].items(): add(name, evidence["sha256"], evidence["bytes"])
    intents = {e["operation_id"]: e for e in events if e["kind"] == "native_intent"}
    closed = {e["operation_id"] for e in events if e["kind"] == "native_closed"}
    verdicts, proofs, native_tasks = {}, {}, {}
    for event in events:
        if event["kind"] == "native_verdict":
            require(event["operation_id"] not in verdicts, "Duplicate native verdict")
            verdicts[event["operation_id"]] = event
    for opid, intent in intents.items():
        require(type(opid) is str and native.re.fullmatch(r"[0-9a-f]{12}", opid), "Invalid native operation")
        task = json.loads(add(f"native/{opid}/task.json", intent["task_copy_sha256"]))
        native_tasks[opid] = task
        if intent["reference_indices"] is None:
            require(len(intent["program_sha256"]) == 1, "Invalid native program set")
            programs = [add(f"native/{opid}/candidate.py", intent["program_sha256"][0])]
        else:
            require(intent["reference_indices"] == task["passing_reference_indices"], "Reference indices differ")
            programs = verification.reference_programs(task)
        require([sha256(p).hexdigest() for p in programs] == intent["program_sha256"], "Native programs differ")
        if opid in verdicts:
            for name, expected in verdicts[opid]["log_evidence"].items():
                require(name.startswith(f"native/{opid}/"), "Log outside native operation")
                add(name, expected)
        if opid in closed:
            proofs[opid] = native.reuse_native(campaign, intent, task, programs)
    runs, records, contexts, eligible = [], {}, {}, []
    results = {e["reservation_seq"]: e for e in events if e["kind"] == "attempt_result"}
    require(len(results) == sum(e["kind"] == "attempt_result" for e in events)
            and results.keys() <= reservations.keys(), "Invalid attempt results")
    for event in events:
        if event["kind"] != "experiment": continue
        record, name = event["record"], event["experiment_id"]
        require(type(record.get("schema_version")) is int and ((record["schema_version"] == 1 and set(record) == RUN_FIELDS)
                or (record["schema_version"] == 2 and set(record) == RUN_FIELDS | {"profiles"})
                or (record["schema_version"] == 3 and set(record) == RUN_FIELDS | {"profiles", "generation_method"}))
                and name not in records, "Require unique loop run record")
        batched = record["schema_version"] == 3
        if "profiles" in record:
            precision.check_profiles(record["profiles"], record["recipes"], campaign=campaign)
            for entry in record["profiles"].values():
                reference = entry["reference"]
                raw = precision._metadata(campaign.root / reference["path"], reference["receipt_sha256"])[0]
                for filename, content in raw.items(): add(reference["path"] + "/" + filename, sha256(content).hexdigest(), len(content))
        if batched:
            methods = record["generation_method"]
            require(type(methods) is dict and set(methods) == set(record["recipes"]) == set(record["profiles"]), "Run batch method sources differ")
            for source, method in methods.items():
                require(method["source"] == source and digest(method["admission"]["request_controls"]["precision_profile"]["descriptor"])
                        == digest(record["profiles"][source]["descriptor"]) and all(method["admission"]["request_controls"][k]
                            == record["recipes"][source][k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")), "Run batch method profile/controls differ")
                raw = journal._batch_method(method, campaign)
                for filename, content in raw.items(): add(method["qualification"]["path"] + "/" + filename, sha256(content).hexdigest(), len(content))
        for reservation in (e for e in reservations.values() if e["experiment_id"] == name):
            request = json.loads(reservation["request_text"])
            require(("session_seq" in reservation) == batched, "Request method direction differs")
            if batched:
                require(reservation["source"] in record["generation_method"], "Request batch source differs")
                batch_intent, outcome = batches[reservation["session_seq"]], outcomes[reservation["seq"]]
                require(digest(batch_intent["method_spec"]) == digest(record["generation_method"][reservation["source"]])
                        and digest({k: v for k, v in request.items() if k not in ("messages", "seed")})
                            == digest(batch_intent["method_spec"]["admission"]["request_controls"])
                        and outcome.get("session_seq") == reservation["session_seq"] and outcome.get("row_index") == reservation["row_index"]
                        and outcome.get("request_sha256") == reservation["request_sha256"]
                        and outcome.get("job_sha256") == batch_intent["job_sha256"], "Request batch binding differs")
            require(("precision_profile" in request) == ("profiles" in record), "Request profile direction differs")
            if "profiles" in record:
                source = reservation["source"]
                require(source in record["profiles"] and set(request) == generation.FIELDS | {"precision_profile", "environment"}, "Request profile fields differ")
                descriptor = generation._profile(request)
                require(digest(descriptor) == digest(record["profiles"][source]["descriptor"])
                        and digest(request["environment"]) == digest(descriptor["expected_runtime"]["environment"]), "Request profile binding differs")
        require(record["cohort"]["manifest_sha256"] == study.COHORT_SHA256
                and record["activation"]["cohort_sha256"] == study.COHORT_SHA256, "Run cohort differs")
        records[name] = event
        activations.append(record["activation"])
        context = [e for e in events if e["kind"] == "execution_context" and e["experiment_id"] == name]
        require(len(context) <= 1 and (not context or verification.digest(context[0]["contract"]) == context[0]["contract_sha256"]),
                "Invalid execution context")
        contexts[name] = context[0]["contract_sha256"] if context else None
        cells = loop.cell_states(events, record)
        runs.append({"experiment_id": name, "record_sha256": event["record_sha256"], "policy_sha256": record["policy_sha256"],
                     "contract_sha256": contexts[name], "cells": cells, "sources": []})
        for cell in cells:
            for seq in cell["attempt_seqs"]:
                result = results.get(seq)
                require(result is None or result["status"] != "verified" or result["candidate_sha256"] is not None,
                        "Verified result lacks candidate")
                if result is None: continue
                if result["candidate_sha256"] is None:
                    require(result["status"] not in ("verification_failed", "verification_error")
                            and result["operation_id"] is None and result["sample_id"] is None
                            and result["reused"] is False, "Invalid candidate-free result")
                    continue
                program = add(f"requests/{seq:06d}/candidate.py", result["candidate_sha256"])
                outcome = outcomes[seq]
                if batched:
                    shared = completions[outcome["session_seq"]]
                    require(outcome["eligible"] is True and shared["eligible"] is True and outcome["batch_done_seq"] == shared["seq"]
                            and outcome["job_sha256"] == shared["job_sha256"]
                            and generation.extract_program(outcome["text"]) == program, "Batch candidate lacks eligible source reply")
                require(outcome["status"] in ("stop", "length") and (record["policy_sha256"] != current_policy
                        or batched or generation.extract_program(outcome["response"]["text"]) == program),
                        "Candidate differs from source reply")
                opid, gate_id = result["operation_id"], cell["gate_operation_id"]
                require(opid in closed and opid in intents, "Candidate lacks closed operation")
                intent, proof = intents[opid], proofs.get(opid)
                require(proofs.get(gate_id) is not None and proofs[gate_id]["status"] == "passed"
                        and intent["task_sha256"] == cell["task_sha256"]
                        and intent["contract_sha256"] == contexts[name]
                        and intent["task_copy_sha256"] == intents[gate_id]["task_copy_sha256"]
                        and result["candidate_sha256"] in intent["program_sha256"]
                        and type(result["reused"]) is bool and result["reused"] == (intent.get("reservation_seq") != seq)
                        and all((native_tasks[o]["split"], native_tasks[o]["id"]) == (cell["split"], cell["id"])
                                for o in (opid, gate_id)),
                        "Candidate lacks matching passed gate or context")
                if proof is None:
                    require(result["status"] == "verification_error" and result["sample_id"] is None, "Unknown native result differs")
                    continue
                matched = [r for r in proof["results"] if r["sample_id"] == result["sample_id"]]
                key = {"task_sha256": cell["task_sha256"], "program_sha256": result["candidate_sha256"],
                       "contract_sha256": contexts[name]}
                require(type(result["sample_id"]) is int and len(matched) == 1 and matched[0]["key"] == key
                        and result["status"] == {"passed": "verified", "failed": "verification_failed",
                            "error": "verification_error"}[matched[0]["status"]], "Unproven candidate")
                if result["status"] != "verified": continue
                eligible.append({**{k: cell[k] for k in ("split", "id", "task_sha256", "source")},
                    "reservation_seq": seq, "candidate_sha256": result["candidate_sha256"], "experiment_id": name,
                    "record_sha256": event["record_sha256"], "recipe_sha256": digest(record["recipes"][cell["source"]]),
                    "policy_sha256": record["policy_sha256"], "contract_sha256": contexts[name], "operation_id": opid,
                    "sample_id": result["sample_id"], "reference_identical": intent["reference_indices"] is not None,
                    "program_bytes": len(program), "observed_generation_tokens": outcome["observed_generation_tokens"],
                    "reply_index": reservations[seq]["attempt"]["reply_index"],
                    **({"session_seq": outcome["session_seq"], "batch_done_seq": outcome["batch_done_seq"],
                        "generation_method_sha256": digest(record["generation_method"][cell["source"]])} if batched else {}),
                    **({"precision_profile": profile_identity(record["profiles"][cell["source"]]["descriptor"])} if "profiles" in record else {})})
    require(all(e["experiment_id"] in records for e in reservations.values()), "Reservation lacks loop run")
    chosen = {}
    for item in sorted(eligible, key=lambda e: e["reservation_seq"]): chosen.setdefault((_key(item), item["source"]), item)
    overrides = [] if selections is None else selections
    require(type(overrides) is list and all(type(i) is int for i in overrides), "Require selection sequence list")
    by_seq = {e["reservation_seq"]: e for e in eligible}
    require(all(i in by_seq for i in overrides) and len({(_key(by_seq[i]), by_seq[i]["source"]) for i in overrides}) == len(overrides),
            "Require one eligible override per cell")
    for seq in overrides:
        item = by_seq[seq]; chosen[(_key(item), item["source"])] = item
    selected = sorted(chosen.values(), key=lambda e: (e["split"], e["id"], e["source"]))
    bindings = {}
    for activation in activations:
        for task in activation["tasks"]:
            identity = task["split"], task["id"]
            require(identity not in bindings or encoded({k: v for k, v in bindings[identity].items() if k != "group"})
                    == encoded({k: v for k, v in task.items() if k != "group"}), "Conflicting activated task")
            bindings[identity] = task
    cumulative = []
    for source in study.SOURCES:
        for difficulty in sorted({t["difficulty"] for t in bindings.values()}):
            keys = {_key(t) for t in bindings.values() if t["difficulty"] == difficulty}
            assigned = {_key(t) for e in records.values() for t in e["record"]["activation"]["tasks"]
                        if t["id"] in e["record"]["assigned_task_ids"]} & keys
            requested = {_key(t) for e in records.values() if source in e["record"]["recipes"]
                         for t in e["record"]["activation"]["tasks"] if t["id"] in e["record"]["assigned_task_ids"]} & keys
            started = {_key(e["attempt"]) for e in reservations.values() if e["source"] == source} & keys
            verified = {_key(e) for e in eligible if e["source"] == source} & keys
            cumulative.append(dict(source=source, difficulty=difficulty, activated=len(keys), assigned=len(assigned),
                                   requested=len(requested), started=len(started), verified=len(verified)))
    budget_events = [e for e in events if e["kind"] in ("initial", "limits")]
    for run in runs:
        name, record = run["experiment_id"], records[run["experiment_id"]]["record"]
        own_reserved = [e for e in reservations.values() if e["experiment_id"] == name]
        own = [e for e in eligible if e["experiment_id"] == name]
        assigned = [t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"]]
        for source, recipe in record["recipes"].items():
            passes = [e for e in own if e["source"] == source]
            cells = [c for c in run["cells"] if c["source"] == source and c["id"] in record["assigned_task_ids"]]
            row = dict(source=source, recipe_sha256=digest(recipe), comparison_sha256=digest([[list(_key(t)) for t in assigned],
                {k: recipe[k] for k in ("max_replies", "max_tokens", "wall_time_seconds")}, contexts[name]]),
                assigned=len(assigned), verified=len({_key(e) for e in passes}), status_counts=dict(Counter(c["status"] for c in cells)),
                reserved=Campaign.totals(budget_events + own_reserved)[1][source],
                observed=_observed([e for e in own_reserved if e["source"] == source], outcomes, batch=record["schema_version"] == 3),
                initial_passes=sum(e["reply_index"] == 0 for e in passes), repair_passes=sum(e["reply_index"] > 0 for e in passes),
                reference_identical=sum(e["reference_identical"] for e in passes))
            if "profiles" in record: row["precision_profile"] = profile_identity(record["profiles"][source]["descriptor"])
            if record["schema_version"] == 3:
                method = record["generation_method"][source]
                row["generation_method"] = dict(method=method["method"], sha256=digest(method), qualification=method["qualification"],
                    config=method["admission"]["config"], prompt_ceiling_tokens=method["prompt_ceiling_tokens"])
                row["comparison_sha256"] = digest([row["comparison_sha256"], digest(method)])
            own_source = [e for e in own_reserved if e["source"] == source]
            row["generation_status_counts"] = dict(Counter(outcomes[e["seq"]]["status"] for e in own_source))
            row["attempt_status_counts"] = dict(Counter(results[e["seq"]]["status"] for e in own_source if e["seq"] in results))
            row["repair_cells"] = len({_key(e["attempt"]) for e in own_source if e["attempt"]["reply_index"] > 0})
            row["reason_counts"] = dict(Counter(c["reason"] for c in cells if c["reason"] is not None))
            row["by_difficulty"] = [dict(difficulty=d, assigned=sum(t["difficulty"] == d for t in assigned),
                verified=sum(t["difficulty"] == d and _key(t) in {_key(p) for p in passes} for t in assigned),
                generation_status_counts=dict(Counter(outcomes[e["seq"]]["status"] for e in own_source
                    if bindings[(e["attempt"]["split"], e["attempt"]["id"])]["difficulty"] == d)),
                attempt_status_counts=dict(Counter(results[e["seq"]]["status"] for e in own_source if e["seq"] in results
                    and bindings[(e["attempt"]["split"], e["attempt"]["id"])]["difficulty"] == d)),
                status_counts=dict(Counter(c["status"] for c in cells if bindings[(c["split"], c["id"])]["difficulty"] == d)))
                for d in sorted({t["difficulty"] for t in assigned})]
            run["sources"].append(row)
    limits, reserved = Campaign.totals(events)
    sessions = _session_costs(events, reservations, outcomes)
    elapsed = [e["elapsed_seconds"] for e in verdicts.values() if e["elapsed_seconds"] is not None]
    costs = dict(limits=limits, reserved=reserved, observed_by_source={s: _observed(
        [e for e in reservations.values() if e["source"] == s], outcomes, sessions=sessions) for s in study.SOURCES}, native=dict(
        operations=len(intents), gates=sum(e["reference_indices"] is not None for e in intents.values()),
        candidates=sum(e["reference_indices"] is None for e in intents.values()), known_elapsed_seconds=math.fsum(elapsed),
        unknown_elapsed_operations=len(intents)-len(elapsed), reuses=sum(e["reused"] for e in results.values())))
    if batches: costs["generation_sessions"] = sessions
    report = dict(runs=runs, cumulative=cumulative, panel_joint=None if panel is None else panel_joint(panel["tasks"], selected),
                  costs=costs, mechanics_baseline=baseline)
    if any(e["kind"] == "paired_plan" for e in events):
        report["paired_trials"] = paired_report.derive(events=events, records=records, runs=runs, eligible=eligible,
            outcomes=outcomes, reservations=reservations, panel=panel)
    if baseline is not None:
        report["combined_known"] = {unit: baseline["combined"][unit] + (sum(v[unit] for v in reserved.values())
            if unit != "observed_generation_tokens" else sum(v[unit] for v in costs["observed_by_source"].values())) for unit in BASE_UNITS}
    raw = "".join(encoded(e) + "\n" for e in events).encode()
    manifest = dict(schema_version=1, journal=dict(through_seq=len(events), sha256=sha256(raw).hexdigest()),
                    artifacts=files, selections=selected, report_sha256=digest(report))
    require(len(campaign.events) == len(events), "Snapshot prefix changed")
    return dict(events=raw, report=report, manifest=manifest, files=files)


def publish(campaign, version, *, selections=None):
    """Atomic visibility under the campaign lease; fsync ordering, not power-loss safety."""
    require(type(version) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version), "Invalid export version")
    exports = campaign.root / "exports"
    require(not exports.is_symlink() and (not exports.exists() or exports.is_dir()), "Invalid exports directory")
    state = snapshot(campaign, selections=selections)
    require("version" not in state["manifest"], "Snapshot already versioned")
    manifest = {**state["manifest"], "version": version, "source_sha256": {
        p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}}
    generated = {"manifest.json": encoded(manifest).encode(), "report.json": encoded(state["report"]).encode(),
                 "events.jsonl": state["events"]}
    artifacts = {}
    for origin, item in state["files"].items():
        name = item["path"]
        require(len(Path(name).parts) == 2 and Path(name).parts[0] == "artifacts"
                and name == str(Path(name)), "Invalid flat artifact path")
        require(name not in artifacts or artifacts[name][1] == item, "Conflicting artifact destination")
        artifacts[name] = (origin, item)
    expected = {(name, False) for name in generated.keys() | artifacts.keys()} | {("artifacts", True)}
    destination = exports / version
    identity = {"path": str(destination), "manifest_sha256": sha256(generated["manifest.json"]).hexdigest()}
    exports.mkdir(exist_ok=True)
    sync_directory(campaign.root)

    def validate(directory):
        require(directory.resolve() == directory and directory.is_dir(), "Invalid published directory")
        actual = set()
        for parent, directories, filenames in os.walk(directory, followlinks=False,
                onerror=lambda error: require(False, "Unreadable published tree")):
            for name in directories + filenames:
                path = Path(parent) / name
                mode = path.lstat().st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Nonregular published entry")
                actual.add((str(path.relative_to(directory)), stat.S_ISDIR(mode)))
        require(actual == expected, "Published tree differs")
        for name, raw in generated.items(): read_artifact(directory, name, sha256(raw).hexdigest(), len(raw))
        for name, (_, item) in artifacts.items(): read_artifact(directory, name, item["sha256"], item["bytes"])

    if os.path.lexists(destination):
        validate(destination)
        sync_directory(exports)
        return identity
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=exports) as temporary:
        staging = Path(temporary)
        (staging / "artifacts").mkdir()
        for name in [*generated, *artifacts]:
            if name in generated:
                raw = generated[name]
            else:
                origin, item = artifacts[name]
                raw = read_artifact(campaign.root, origin, item["sha256"], item["bytes"])
            with (staging / name).open("xb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
        validate(staging)
        sync_directory(staging / "artifacts")
        sync_directory(staging)
        if os.path.lexists(destination): raise FileExistsError(destination)
        os.rename(staging, destination)
        sync_directory(exports)
    return identity
