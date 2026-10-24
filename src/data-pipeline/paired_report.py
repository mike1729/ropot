"""Scientific validity from one snapshot's proven evidence; no dispatch or live guards."""

from collections import Counter
from hashlib import sha256
import json
import math

from . import experiment_loop as loop, study_tasks as study
from .generation_journal import encoded, profile_control_reason, require

FIELDS = {"schema_version", "plan_id", "activation_sha256", "panel_sha256", "receipt_sha256",
          "runtime_guard_sha256", "recipe_sha256", "policy_sha256", "contract_sha256", "records"}
TERMINAL = {("verified", None), ("exhausted", "reply_limit"), ("blocked", "gate_failed"),
            ("blocked", "generation_timeout"), ("blocked", "generation_error"), ("blocked", "interrupted")}


def digest(value):
    return sha256(encoded(value).encode()).hexdigest()


def _key(task):
    return task["split"], task["id"], task.get("task_sha256", task.get("sha256"))


def _layout():
    loaded = study.load_cohort()
    activation = study.activate(loaded, list(study.GROUPS), construction_per_phase=32)
    groups = [[t for t in activation["tasks"] if t["group"] == g] for g in study.GROUPS]
    require([len(g) for g in groups] == [8, 32, 32], "Invalid paired layout")
    tuning, fresh = groups[0] + groups[1][:16] + groups[2][:16], groups[1][16:] + groups[2][16:]
    ordinals = {_key(t): i for i, t in enumerate(activation["tasks"])}
    schedule = [(t, b) for t in tuning + fresh for b in ((8, 16) if ordinals[_key(t)] % 2 == 0 else (16, 8))]
    return loaded, activation, dict(activation=activation, tasks=groups[0] + groups[1][:21] + groups[2][:21]), tuning, fresh, schedule


def _recipes(recipes):
    require(set(recipes) == set(study.SOURCES), "Invalid paired recipe")
    for recipe in recipes.values():
        require(set(recipe) == loop.RECIPE_FIELDS and all(type(recipe[k]) is str and recipe[k].strip() for k in ("system", "instruction"))
                and type(recipe["max_replies"]) is int and recipe["max_replies"] > 0
                and type(recipe["max_tokens"]) is int and recipe["max_tokens"] > 0, "Invalid paired recipe")
        require(type(recipe["seeds"]) is list and len(recipe["seeds"]) == recipe["max_replies"]
                and all(type(s) is int and 0 <= s < 2**64 for s in recipe["seeds"]), "Invalid paired recipe")
        require(all(type(recipe[k]) in (int, float) and math.isfinite(recipe[k]) and low < recipe[k] <= high
                    for k, low, high in (("temperature", -1, math.inf), ("top_p", 0, 1), ("wall_time_seconds", 0, math.inf)))
                and recipe["temperature"] >= 0, "Invalid paired recipe")


def _plan(event, layout, panel, records, run_map, contexts, seen, invalid):
    loaded, activation, expected_panel, _, _, schedule = layout
    p = event["payload"]
    require(set(event) == {"seq", "kind", "payload", "payload_sha256"} and type(p) is dict and set(p) == FIELDS and digest(p) == event["payload_sha256"]
            and type(p["schema_version"]) is int and p["schema_version"] == 1 and type(p["plan_id"]) is str and p["plan_id"].strip(), "Invalid paired plan")
    require(all(type(p[k]) is str and study.DIGEST.fullmatch(p[k]) for k in FIELDS - {"schema_version", "plan_id", "recipe_sha256", "records"})
            and type(p["records"]) is list and len(p["records"]) == 144 and digest(panel) == digest(expected_panel) == p["panel_sha256"]
            and digest(activation) == p["activation_sha256"] and set(p["recipe_sha256"]) == set(study.SOURCES), "Invalid paired plan")
    rows = []
    for member, (task, bits) in zip(p["records"], schedule):
        require(type(member) is dict and set(member) == {"task", "bits", "experiment_id", "record_sha256"}
                and type(member["task"]) is dict and set(member["task"]) == {"split", "id", "task_sha256"}
                and type(member["bits"]) is int and member["bits"] == bits and type(member["task"]["id"]) is int
                and encoded(list(_key(member["task"]))) == encoded(list(_key(task))) and type(member["experiment_id"]) is str
                and member["experiment_id"].strip() and member["experiment_id"] not in seen, "Paired member differs")
        name = member["experiment_id"]; seen.add(name)
        require(name in records and name in run_map and len(contexts.get(name, [])) == 1, "Paired member differs")
        registered, context = records[name], contexts[name][0]
        record = registered["record"]; _recipes(record["recipes"])
        expected = {**study.experiment_record(loaded, name, activation, record["recipes"]), "schema_version": 2,
                    "assigned_task_ids": [task["id"]], "policy_sha256": p["policy_sha256"], "profiles": record["profiles"]}
        require(digest(expected) == digest(record) == registered["record_sha256"] == member["record_sha256"]
                and context["contract_sha256"] == p["contract_sha256"]
                and all(digest(record["recipes"][s]) == p["recipe_sha256"][s] for s in study.SOURCES), "Paired member differs")
        require(set(record["profiles"]) == set(study.SOURCES)
                and all((entry["descriptor"]["source"], entry["descriptor"]["bits"], entry["descriptor"]["receipt_sha256"], entry["descriptor"]["runtime_guard_sha256"])
                    == (s, bits, p["receipt_sha256"], p["runtime_guard_sha256"]) for s, entry in record["profiles"].items()), "Paired member differs")
        if max(registered["seq"], context["seq"]) >= event["seq"]: invalid.append(dict(kind="registration_after_plan", experiment_id=name))
        cells = [c for c in run_map[name]["cells"] if c["id"] == task["id"] and c["source"] in study.SOURCES]
        require(len(cells) == 2 and {c["source"] for c in cells} == set(study.SOURCES)
                and all(encoded(list(_key(c))) == encoded(list(_key(task))) for c in cells), "Paired member differs")
        rows.append((member, record, cells))
    return rows


def _views(events):
    return ({e["reservation_seq"]: e for e in events if e["kind"] == "generation_done"},
            {e["reservation_seq"]: e for e in events if e["kind"] == "attempt_result"},
            {e["operation_id"] for e in events if e["kind"] == "native_closed"} & {e["operation_id"] for e in events if e["kind"] == "native_verdict"})


def _terminal(cell, views):
    done, results, proved = views
    seqs = cell["attempt_seqs"]
    return ((cell["status"], cell["reason"]) in TERMINAL and cell["gate_operation_id"] in proved
            and (bool(seqs) or cell["reason"] == "gate_failed") and all(s in done and done[s].get("profile_guarded") is True
                and profile_control_reason(done[s]) is None for s in seqs)
            and all(results.get(s, {}).get("operation_id") is None or results[s]["operation_id"] in proved for s in seqs))


def _requests(event, rows, reservations, outcomes, invalid):
    own = {m["experiment_id"]: (m, r) for m, r, _ in rows}
    indices = {m["experiment_id"]: i for i, (m, _, _) in enumerate(rows)}
    first, started, last = {}, set(), -1
    for seq, reservation in sorted(reservations.items()):
        name = reservation["experiment_id"]
        if name not in own: continue
        member, record = own[name]; source, attempt = reservation["source"], reservation["attempt"]
        require(source in record["recipes"] and encoded(list(_key(attempt))) == encoded(list(_key(member["task"])))
                and type(attempt["reply_index"]) is int and 0 <= attempt["reply_index"] < record["recipes"][source]["max_replies"], "Paired request differs")
        recipe, request, slot = record["recipes"][source], json.loads(reservation["request_text"]), attempt["reply_index"]
        controls = {k: recipe[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}
        require(digest({k: request[k] for k in controls | {"seed": None}}) == digest({**controls, "seed": recipe["seeds"][slot]}), "Paired request differs")
        if seq <= event["seq"]: invalid.append(dict(kind="member_before_plan", reservation_seq=seq))
        if name not in started:
            if indices[name] < last: invalid.append(dict(kind="observed_order_mismatch", reservation_seq=seq))
            started.add(name); last = max(last, indices[name])
        reason = profile_control_reason(outcomes[seq])
        if reason: invalid.append(dict(kind=reason, reservation_seq=seq))
        if slot == 0:
            response = outcomes[seq]["response"]
            ready = next((e for e in response["events"] if e["event"] == "ready"), None) if response else None
            first.setdefault((_key(member["task"]), source), {})[member["bits"]] = (request, ready)
    for (task, source), pair in first.items():
        if set(pair) != {8, 16}: continue
        require(digest({k: v for k, v in pair[8][0].items() if k not in ("model_path", "precision_profile")})
                == digest({k: v for k, v in pair[16][0].items() if k not in ("model_path", "precision_profile")}), "Initial requests differ")
        if (pair[8][1] is not None and pair[16][1] is not None
                and digest([pair[8][1][k] for k in ("prompt", "prompt_ids", "eos_token_ids")])
                != digest([pair[16][1][k] for k in ("prompt", "prompt_ids", "eos_token_ids")])):
            invalid.append(dict(kind="initial_ready_mismatch", task=dict(zip(("split", "id", "task_sha256"), task)), source=source))


def _confirmation(event, decisions, rows, events, reservations, tuning, fresh, invalid):
    if len(decisions) > 1: invalid.append(dict(kind="confirmation_duplicate"))
    decision = decisions[0] if decisions else None
    cutoff = decision["seq"] if decision else len(events) + 1
    names = {m["experiment_id"] for m, _, _ in rows}
    if any(e["experiment_id"] in names and _key(e["attempt"]) in fresh and e["seq"] < cutoff for e in reservations.values()):
        invalid.append(dict(kind="fresh_before_confirmation"))
    exposed = {_key(e["attempt"]) for e in reservations.values() if e["seq"] < cutoff and _key(e["attempt"]) in fresh}
    if decision:
        prefix = [e for e in events if e["seq"] < cutoff]; views = _views(prefix)
        registered = {e["experiment_id"] for e in prefix if e["kind"] == "experiment"}
        premature = cutoff <= event["seq"] or any(m["experiment_id"] not in registered for m, _, _ in rows if _key(m["task"]) in tuning)
        if not premature:
            cells = [c for m, r, _ in rows if _key(m["task"]) in tuning for c in loop.cell_states(prefix, r)
                     if c["id"] == m["task"]["id"] and c["source"] in study.SOURCES]
            premature = len(cells) != 160 or not all(_terminal(c, views) for c in cells)
        if premature: invalid.append(dict(kind="confirmation_premature"))
    return decision, exposed


def derive(*, events, records, runs, eligible, outcomes, reservations, panel):
    anchors = [e for e in events if e["kind"] == "paired_plan"]
    if not anchors: return []
    layout, run_map, contexts, seen, trials = _layout(), {r["experiment_id"]: r for r in runs}, {}, set(), []
    for e in events:
        if e["kind"] == "execution_context": contexts.setdefault(e["experiment_id"], []).append(e)
    decisions, plan_ids = {}, {e["payload"]["plan_id"] for e in anchors}
    require(len(plan_ids) == len(anchors), "Invalid paired plan")
    for e in (e for e in events if e["kind"] == "paired_confirmation"):
        p = e["payload"]
        require(set(e) == {"seq", "kind", "payload", "payload_sha256"} and set(p) == {"plan_id", "plan_sha256", "decision"}
                and digest(p) == e["payload_sha256"] and p["plan_id"] in plan_ids and p["decision"] == "unchanged"
                and p["plan_sha256"] == next(a["payload_sha256"] for a in anchors if a["payload"]["plan_id"] == p["plan_id"]), "Invalid paired confirmation")
        decisions.setdefault(p["plan_id"], []).append(e)
    views, tuning, fresh = _views(events), {_key(t) for t in layout[3]}, {_key(t) for t in layout[4]}
    for event in anchors:
        invalid = []; rows = _plan(event, layout, panel, records, run_map, contexts, seen, invalid)
        _requests(event, rows, reservations, outcomes, invalid)
        decision, exposed = _confirmation(event, decisions.get(event["payload"]["plan_id"], []), rows, events, reservations, tuning, fresh, invalid)
        setup_seq = next(e["seq"] for e in events if e["kind"] == "collection_setup" and e["payload"]["plan_id"] == event["payload"]["plan_id"])
        superseded = next((e["seq"] for e in events if e["kind"] == "collection_setup" and e["seq"] > setup_seq), None)
        names = {m["experiment_id"] for m, _, _ in rows}
        if superseded is not None and ((decision is not None and decision["seq"] > superseded)
                or any(e["experiment_id"] in names and e["seq"] > superseded for e in reservations.values())):
            invalid.append(dict(kind="work_after_supersession", superseded_seq=superseded))
        cells = [c for _, _, assigned in rows for c in assigned]; terminal = sum(_terminal(c, views) for c in cells)
        complete = not invalid and decision is not None and terminal == 288
        trials.append(dict(report_scope="scientific_validity", plan_id=event["payload"]["plan_id"], plan_sha256=event["payload_sha256"],
            comparison_valid=not invalid, complete=complete, confirmation_seq=decision["seq"] if decision else None,
            fresh_confirmation_valid=complete and not exposed, prior_exposed_tasks=[dict(split=t[0], id=t[1], task_sha256=t[2]) for t in sorted(exposed)], invalid_reasons=invalid,
            progress=dict(planned_records=144, planned_cells=288, terminal_cells=terminal,
                status_counts=dict(Counter(c["status"] for c in cells)),
                reason_counts=dict(Counter(c["reason"] for c in cells if c["reason"] is not None))),
            members=[dict(experiment_id=m["experiment_id"], task=m["task"], bits=m["bits"],
                cells=[{k: c[k] for k in ("source", "status", "reason", "attempt_seqs")} for c in assigned])
                for m, _, assigned in rows]))
    return trials
