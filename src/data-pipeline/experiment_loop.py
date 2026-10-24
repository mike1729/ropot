"""Frozen loop inputs and public prompt/artifact helpers; dispatch is separate."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from . import local_generation as generation
from . import native_recovery as native
from . import study_tasks as study
from . import verification
from . import precision_profiles as precision, generation_journal as journal
from .generation_journal import Campaign, UNITS, encoded, require, sync_directory

RECIPE_FIELDS = {"system", "instruction", "max_replies", "max_tokens", "wall_time_seconds",
                 "temperature", "top_p", "seeds"}
CASE_FIELDS = {"input", "answer", "status", "stdout", "stderr", "returncode", "diagnostic"}
STATUSES = {"passed", "wrong_answer", "timeout", "output_limit", "stdout_limit",
            "stderr_limit", "invalid_stdout", "nonzero_exit"}
SHARED = ("Return a complete Python 3.12 program reading stdin and writing stdout. "
          "Use raw Python or a sole Python fence, without explanation.")
FORMAT_REPAIR = ("The preceding reply is not valid Python in the required format. "
                 "Return a complete program in that format.")


def policy_sha256():
    sources = {Path(p).name: sha256(Path(p).read_bytes()).hexdigest() for p in
               (__file__, generation.__file__, precision.__file__, generation.WORKER, journal.__file__)}
    return sha256(encoded(sources).encode("utf-8")).hexdigest()


def run_record(loaded, experiment_id, groups, recipes, assigned, *, profiles=None):
    activation = study.activate(loaded, groups)
    record = study.experiment_record(loaded, experiment_id, activation, study.snapshot(recipes))
    if profiles is not None:
        profiles = precision.check_profiles(study.snapshot(profiles), record["recipes"])
        record.update(schema_version=2, profiles=profiles)
    for source, recipe in record["recipes"].items():
        require(set(recipe) == RECIPE_FIELDS, "Unexpected recipe fields")
        require(all(type(recipe[k]) is str and recipe[k].strip() for k in ("system", "instruction")),
                "Require nonempty system and instruction")
        require(type(recipe["max_replies"]) is int and recipe["max_replies"] > 0, "Invalid reply cap")
        require(type(recipe["seeds"]) is list and len(recipe["seeds"]) == recipe["max_replies"], "Invalid seeds")
        for seed in recipe["seeds"]:
            generation.prepare_request(generation.request_for(source, [{"role": "system", "content": "Validate"},
                {"role": "user", "content": "Validate"}], profile=profiles[source]["descriptor"] if profiles is not None else None, seed=seed, **{k: recipe[k] for k in
                ("temperature", "top_p", "max_tokens", "wall_time_seconds")}))
    assigned = study.snapshot(assigned)
    require(type(assigned) is list and assigned and all(type(i) is int for i in assigned)
            and len(set(assigned)) == len(assigned), "Require distinct integer assigned IDs")
    require(assigned == [t["id"] for t in activation["tasks"] if t["id"] in assigned],
            "Assignments must follow the current activation order")
    return {**record, "assigned_task_ids": assigned, "policy_sha256": policy_sha256()}


def validate_run(loaded, record):
    record = study.snapshot(record)
    try:
        rebuilt = run_record(loaded, record["experiment_id"], record["activation"]["groups"],
                             record["recipes"], record["assigned_task_ids"], profiles=record.get("profiles"))
    except (KeyError, TypeError):
        raise ValueError("Invalid run record") from None
    require(encoded(rebuilt) == encoded(record), "Run differs from its canonical record or current policy")
    return rebuilt


def project_feedback(feedback):
    feedback = study.snapshot(feedback)
    require(type(feedback) is dict and set(feedback) == {"status", "visible", "hidden"}
            and feedback["status"] == "failed", "Require failed verification feedback")
    visible = feedback["visible"]
    require(type(visible) is list and bool(visible), "Require visible feedback")
    for case in visible:
        require(type(case) is dict and set(case) == CASE_FIELDS
                and type(case["status"]) is str and case["status"] in STATUSES,
                "Invalid visible feedback schema")
        require(all(type(case[k]) is str for k in ("input", "answer"))
                and all(case[k] is None or type(case[k]) is str for k in ("stdout", "stderr", "diagnostic"))
                and (case["returncode"] is None or type(case["returncode"]) is int), "Invalid visible feedback types")
    failed = [case for case in visible if case["status"] != "passed"]
    require(feedback["hidden"] == ("not_run" if failed else "failed"), "Inconsistent hidden feedback")
    projected = []
    if failed:
        case = failed[0]
        result = {"status": case["status"], "returncode": case["returncode"], "truncated": {}}
        for key, limit in (("input", 128), ("answer", 64), ("stdout", 256), ("stderr", 128)):
            value = case[key]
            if key in ("stdout", "stderr") and value is not None:
                value = re.sub(r"/work/case-\d+/", "", value)
            raw = value.encode("utf-8") if value is not None else b""
            clipped = raw[-limit:] if key == "stderr" else raw[:limit]
            result[key] = clipped.decode("utf-8", "ignore") if value is not None else None
            result["truncated"][key] = len(raw) > limit
        projected.append(result)
    result = {"status": "failed", "visible": projected, "hidden": feedback["hidden"]}
    require(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) <= 4096,
            "Projected feedback exceeds its byte cap")
    return result


def messages(task, recipe, previous=None, feedback=None):
    question, starter = task["question"], task.get("starter_code", "")
    require(type(question) is str and type(starter) is str, "Require task text")
    require(all(type(recipe[k]) is str and recipe[k].strip() for k in ("system", "instruction")), "Require recipe text")
    require((previous is None and feedback is None) or type(previous) is str, "Invalid repair context")
    user = recipe["instruction"] + "\n\nProblem:\n" + question
    if starter:
        user += "\n\nStarter code:\n" + starter
    result = [{"role": "system", "content": recipe["system"] + "\n\n" + SHARED},
              {"role": "user", "content": user}]
    if previous is not None:
        repair = FORMAT_REPAIR if feedback is None else "Repair using this public feedback:\n" + json.dumps(
            project_feedback(feedback), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        result.extend([{"role": "assistant", "content": previous}, {"role": "user", "content": repair}])
    return result


def publish_candidate(directory, program):
    require(type(program) is bytes, "Require exact program bytes")
    directory = Path(directory)
    final = directory / "candidate.py"
    if not os.path.lexists(final):
        descriptor, temporary = tempfile.mkstemp(prefix=".candidate-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(program)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, final)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    with os.fdopen(os.open(final, os.O_RDONLY | os.O_NOFOLLOW), "rb") as existing:
        require(existing.read() == program, "Published candidate differs")
    sync_directory(directory)
    return final


def _gate(events, task_sha256, contract_sha256):
    return next((e for e in events if e["kind"] == "native_intent" and e["task_sha256"] == task_sha256
                 and e["reference_indices"] is not None and e["contract_sha256"] == contract_sha256), None)


def _affordable(events, source, recipe):
    limits, totals = Campaign.totals(events, (source, recipe))
    return all(totals[source][unit] <= limits[source][unit] for unit in UNITS)


def _history(events, record):
    bindings = {t["id"]: t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"]}
    reserved = {e["seq"]: e for e in events if e["kind"] == "generation_reserved"}
    results, cells = {}, {}
    for event in events:
        if event["kind"] == "attempt_result":
            seq = event["reservation_seq"]
            require(seq in reserved and seq not in results, "Invalid or duplicate attempt result")
            results[seq] = event
    for event in reserved.values():
        if event["experiment_id"] != record["experiment_id"]:
            continue
        item, source = event.get("attempt"), event["source"]
        require(type(item) is dict and set(item) == {"split", "id", "task_sha256", "reply_index"}
                and type(item["id"]) is int and item["id"] in bindings and source in record["recipes"],
                "Invalid attempt ownership")
        task = bindings[item["id"]]
        require((item["split"], item["task_sha256"]) == (task["split"], task["sha256"])
                and type(item["reply_index"]) is int, "Attempt binding differs")
        cells.setdefault((source, item["id"]), []).append(event)
    for (source, _), slots in cells.items():
        require([e["attempt"]["reply_index"] for e in slots] == list(range(len(slots)))
                and len(slots) <= record["recipes"][source]["max_replies"], "Invalid reply slots")
        require(all(e["seq"] in results and results[e["seq"]]["status"] in
                    ("format_failed", "verification_failed") for e in slots[:-1]), "Reply after unresolved or terminal slot")
    return cells, results


def cell_states(events, record):
    require(any(e["kind"] == "experiment" and e["experiment_id"] == record["experiment_id"]
                and e["record_sha256"] == sha256(encoded(record).encode()).hexdigest() for e in events),
            "Require registered frozen record")
    contexts = [e for e in events if e["kind"] == "execution_context" and e["experiment_id"] == record["experiment_id"]]
    require(len(contexts) <= 1, "Duplicate execution context")
    contract = contexts[0]["contract_sha256"] if contexts else None
    cells, results = _history(events, record)
    closed = {e["operation_id"] for e in events if e["kind"] == "native_closed"}
    verdicts = {e["operation_id"]: e for e in events if e["kind"] == "native_verdict"}
    outcomes = {e["reservation_seq"]: e for e in events if e["kind"] == "generation_done"}
    output = []
    for task in record["activation"]["tasks"]:
        gate = _gate(events, task["sha256"], contract)
        for source in study.SOURCES:
            slots = cells.get((source, task["id"]), [])
            done = [results[e["seq"]] for e in slots if e["seq"] in results]
            state, reason = "not_started", None
            if task["id"] not in record["assigned_task_ids"]: state = "not_assigned"
            elif source not in record["recipes"]: state = "not_requested"
            elif any(e["status"] == "verified" for e in done): state = "verified"
            elif done and done[-1]["status"] in ("generation_timeout", "generation_error", "interrupted", "verification_error"):
                state, reason = "blocked", "native_infrastructure" if done[-1]["status"] == "verification_error" else done[-1]["status"]
            elif slots and slots[-1]["seq"] not in results:
                reason = journal.profile_control_reason(outcomes.get(slots[-1]["seq"], {}))
                state = "blocked" if reason is not None else "pending"
            elif len(slots) == record["recipes"][source]["max_replies"]: state, reason = "exhausted", "reply_limit"
            elif not slots and gate and gate["operation_id"] not in closed: state = "pending"
            elif not slots and gate and verdicts.get(gate["operation_id"], {}).get("status") != "passed":
                state, reason = "blocked", "gate_failed" if verdicts.get(gate["operation_id"], {}).get("status") == "failed" else "native_infrastructure"
            elif not _affordable(events, source, record["recipes"][source]): state, reason = "blocked", "campaign_budget"
            elif slots: state = "pending"
            output.append({"split": task["split"], "id": task["id"], "task_sha256": task["sha256"], "source": source,
                           "status": state, "reason": reason, "attempt_seqs": [e["seq"] for e in slots],
                           "gate_operation_id": gate["operation_id"] if gate else None})
    return output


def _proof(campaign, task, task_sha256, contract_sha256, program, seq, dispatch):
    def lookup():
        events = campaign.events
        if program is None:
            return _gate(events, task_sha256, contract_sha256)
        digest = sha256(program).hexdigest()
        return next((e for e in events if e["kind"] == "native_intent" and e["task_sha256"] == task_sha256
                     and e["contract_sha256"] == contract_sha256 and digest in e["program_sha256"]), None)
    intent = lookup()
    if intent is None and dispatch:
        native.run_native(campaign, task, task_sha256, program, reservation_seq=seq)
        intent = lookup()
        require(intent is not None, "Execution contract changed during native dispatch")
    if intent is None:
        return None, None
    owner = intent.get("reservation_seq")
    if owner is not None:
        reserved = next(e for e in campaign.events if e["kind"] == "generation_reserved" and e["seq"] == owner)
        context = reserved.get("attempt")
        require(type(context) is dict and context.get("task_sha256") == task_sha256
                and context.get("id") == task["id"], "Native owner binding differs")
    if intent["reference_indices"] is not None:
        require(intent["reference_indices"] == task["passing_reference_indices"], "Reference indices differ")
        programs = verification.reference_programs(task)
    else:
        programs = [program]
    return intent, native.reuse_native(campaign, intent, task, programs)


def _cell(campaign, task, binding, source, record, contract_sha256, dispatch):
    recipe = record["recipes"][source]
    histories, results = _history(campaign.events, record)
    slots = list(histories.get((source, task["id"]), []))
    if not slots and (not dispatch or not _affordable(campaign.events, source, recipe)):
        return
    gate, verdict = _proof(campaign, task, binding["sha256"], contract_sha256, None, None, dispatch)
    if gate is None or verdict is None or verdict["status"] != "passed":
        require(not slots, "Generation without a passed reference gate")
        return
    previous, feedback = None, None
    for index in range(recipe["max_replies"]):
        request = generation.request_for(source, messages(task, recipe, previous, feedback),
            profile=record["profiles"][source]["descriptor"] if "profiles" in record else None, seed=recipe["seeds"][index],
            **{k: recipe[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")})
        _, raw = generation.prepare_request(request)
        if index < len(slots):
            reservation = slots[index]
            require(reservation["request_text"].encode() == raw, "Retained request differs from replay")
            outcome = campaign.outcome(reservation["seq"])
        else:
            if not dispatch or not _affordable(campaign.events, source, recipe): return
            outcome = campaign.generate(record["experiment_id"], source, request, attempt={
                "split": binding["split"], "id": task["id"], "task_sha256": binding["sha256"], "reply_index": index})
            reservation = next(e for e in campaign.events if e["seq"] == outcome["reservation_seq"])
            slots.append(reservation)
        seq = reservation["seq"]
        payload = {"reservation_seq": seq, "candidate_sha256": None, "operation_id": None, "sample_id": None, "reused": False}
        feedback = None
        if outcome["status"] not in ("stop", "length"):
            status = "interrupted" if outcome["status"] in ("interrupted", "cancelled", "parent_error") else "generation_timeout" if outcome["status"] == "timeout" else "generation_error"
        else:
            previous = outcome["response"]["text"]
            try:
                program = generation.extract_program(previous)
            except ValueError:
                status = "format_failed"
            else:
                publish_candidate(campaign.root / "requests" / f"{seq:06d}", program)
                intent, proof = _proof(campaign, task, binding["sha256"], contract_sha256, program, seq, dispatch)
                if intent is None:
                    require(seq not in results, "Missing recorded native intent")
                    return
                payload.update(candidate_sha256=sha256(program).hexdigest(), operation_id=intent["operation_id"],
                               reused=intent.get("reservation_seq") != seq)
                status = "verification_error"
                if proof is not None:
                    result = next(r for r in proof["results"] if r["key"]["program_sha256"] == payload["candidate_sha256"])
                    payload["sample_id"], feedback = result["sample_id"], result["feedback"]
                    status = {"passed": "verified", "failed": "verification_failed", "error": "verification_error"}[result["status"]]
        payload["status"] = status
        if seq in results:
            require(encoded({k: results[seq][k] for k in payload}) == encoded(payload), "Attempt result differs from replay")
        else:
            results[seq] = campaign._append("attempt_result", **payload)
        if status not in ("format_failed", "verification_failed"):
            return


def run(campaign, loaded, record):
    native.reconcile_native(campaign)
    record = validate_run(loaded, record)
    if "profiles" in record: precision.check_profiles(record["profiles"], record["recipes"], campaign=campaign)
    events = campaign.events
    for event in events:
        if event["kind"] == "generation_reserved":
            outcome = campaign.outcome(event["seq"])
            response = outcome["response"]
            require(response is None or response["cleanup_error"] is None, "Unresolved generation cleanup failure")
            if event["experiment_id"] == record["experiment_id"]: journal.require_profile_outcome(outcome)
    current = verification.runner.contract()
    digest = verification.digest(current)
    contexts = [e for e in events if e["kind"] == "execution_context" and e["experiment_id"] == record["experiment_id"]]
    require(len(contexts) <= 1 and (not contexts or (encoded(contexts[0]["contract"]) == encoded(current)
            and contexts[0]["contract_sha256"] == digest)), "Execution context changed; use a new experiment")
    require(contexts or not any(e["kind"] == "generation_reserved" and e["experiment_id"] == record["experiment_id"]
                               for e in events), "Dispatched experiment lacks execution context")
    campaign.register(record)
    if not contexts:
        campaign._append("execution_context", experiment_id=record["experiment_id"], contract=current, contract_sha256=digest)
    _history(campaign.events, record)
    tasks = [study.load_task(loaded, record["activation"], i) for i in record["assigned_task_ids"]]
    for dispatch in (False, True):
        for binding, task in tasks:
            for source in study.SOURCES:
                if source in record["recipes"]:
                    _cell(campaign, task, binding, source, record, digest, dispatch)
    return cell_states(campaign.events, record)
