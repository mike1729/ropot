"""Paired validity and metrics from proven evidence; no dispatch or live guards."""

from collections import Counter
from hashlib import sha256
import json
import math
from statistics import median

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


def _first_assessed(cell, views):
    done, results, proved = views
    if not cell["attempt_seqs"]: return cell["reason"] == "gate_failed" and _terminal(cell, views)
    seq = cell["attempt_seqs"][0]; result = results.get(seq, {})
    return (cell["gate_operation_id"] in proved and seq in done and done[seq].get("profile_guarded") is True
            and profile_control_reason(done[seq]) is None
            and (result.get("status") in ("format_failed", "generation_timeout", "generation_error", "interrupted")
                 or (result.get("status") in ("verified", "verification_failed") and result.get("operation_id") in proved)))


def _native_rate(outcome):
    response = outcome["response"]
    if response is None or response.get("complete") is not True: return None
    chunk = next((e for e in reversed(response["events"]) if e["event"] == "chunk"), {})
    value = chunk.get("generation_tps")
    return value if (chunk.get("finish_reason") in ("stop", "length") and type(value) in (int, float)
                     and math.isfinite(value) and value > 0) else None


def _ready(outcome):
    response = outcome["response"]
    return next((e for e in response["events"] if e["event"] == "ready"), None) if response else None


def _cell_wall(cell, first, outcomes, views):
    if not first and not _terminal(cell, views): return None
    seqs = cell["attempt_seqs"][:1] if first else cell["attempt_seqs"]
    if first and not seqs: return None
    values = [outcomes[s]["elapsed_seconds"] for s in seqs]
    return math.fsum(values) if all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values) else None


def _wall(values, valid):
    known = {str(b): [v for v in values[b] if v is not None] for b in (8, 16)}
    pairs = [(a, b) for a, b in zip(values[8], values[16]) if a is not None and b is not None]
    ratios = [b / a for a, b in pairs if a > 0 and math.isfinite(b / a)] if valid else []
    return dict(known_seconds={b: math.fsum(v) for b, v in known.items()},
        known_tasks={b: len(v) for b, v in known.items()},
        unknown_tasks={str(b): len(values[b]) - len(known[str(b)]) for b in (8, 16)},
        known_pairs=len(pairs), zero_denominator_pairs=sum(a == 0 for a, _ in pairs),
        ratio_pairs=len(ratios), median_16_over_8=median(ratios) if ratios else None)


def _cost(source_rows):
    return {kind: {unit: (math.fsum(r[kind][unit] for r in source_rows) if unit in
            ("reserved_generation_wall_seconds", "known_elapsed_seconds") else sum(r[kind][unit] for r in source_rows))
            for unit in source_rows[0][kind]} for kind in ("reserved", "observed")}


def _metrics(layout, rows, run_map, eligible, outcomes, reservations, views, valid):
    members = {m["experiment_id"]: m for m, _, _ in rows}
    cells = {(_key(m["task"]), m["bits"], c["source"]): c for m, _, assigned in rows for c in assigned}
    source_rows = {(_key(m["task"]), m["bits"], r["source"]): r for m, _, _ in rows
                   for r in run_map[m["experiment_id"]]["sources"]}
    own = [e for e in eligible if e["experiment_id"] in members and _key(e) == _key(members[e["experiment_id"]]["task"])
           and e["source"] in study.SOURCES]
    passed = {first: {(_key(e), members[e["experiment_id"]]["bits"], e["source"]) for e in own
                     if not first or e["reply_index"] == 0} for first in (True, False)}
    rates = {s: _native_rate(outcomes[s]) for s, r in reservations.items() if r["experiment_id"] in members}
    panel_keys = {_key(t) for t in layout[2]["tasks"]}
    subsets = [("primary50", layout[2]["tasks"]), ("tuning40", layout[3]), ("fresh32", layout[4]),
               ("extra22", [t for t in layout[1]["tasks"] if _key(t) not in panel_keys]), ("full72", layout[1]["tasks"])]
    output = []
    for name, tasks in subsets:
        keys = [_key(t) for t in tasks]; arms, joints = [], {}
        terminal = sum(_terminal(cells[(k, b, s)], views) for k in keys for b in (8, 16) for s in study.SOURCES)
        for bits in (8, 16):
            verified = {}
            for first in (True, False):
                sets = {s: {k for k in keys if (k, bits, s) in passed[first]} for s in study.SOURCES}
                joint = set.intersection(*sets.values()); joints[(bits, first)] = joint
                verified[first] = {**{s: len(v) for s, v in sets.items()}, "joint": len(joint)}
            costs, native_rates = [], []
            names = {m["experiment_id"] for m, _, _ in rows if m["bits"] == bits and _key(m["task"]) in keys}
            for source in study.SOURCES:
                costs.append(dict(source=source, **_cost([source_rows[(k, bits, source)] for k in keys])))
                seqs = [s for s, r in reservations.items() if r["experiment_id"] in names and r["source"] == source]
                values = [rates[s] for s in seqs if rates[s] is not None]
                native_rates.append(dict(source=source, count=len(values), median_generation_tps=median(values) if values else None,
                                         unavailable_requests=len(seqs) - len(values)))
            arms.append(dict(bits=bits, first_verified=verified[True], final_verified=verified[False], costs=costs, native_rates=native_rates))
        contingencies = {}
        for first, label in ((True, "first_reply"), (False, "final_policy")):
            counts = dict.fromkeys(("both", "8_only", "16_only", "neither", "unassessed"), 0)
            for key in keys:
                assessed = all((_first_assessed(cells[(key, b, s)], views) if first else _terminal(cells[(key, b, s)], views))
                               for b in (8, 16) for s in study.SOURCES)
                a, b = key in joints[(8, first)], key in joints[(16, first)]
                category = "unassessed" if not assessed else "both" if a and b else "8_only" if a else "16_only" if b else "neither"
                counts[category] += 1
            contingencies[label] = counts
        timing = []
        for source in study.SOURCES:
            coverage = dict.fromkeys(("paired_requests", "missing_request_pairs", "matching_ready_pairs", "missing_ready_pairs", "mismatched_ready_pairs"), 0)
            walls = {first: {b: [] for b in (8, 16)} for first in (True, False)}; native_pairs = 0
            for key in keys:
                firsts = {b: cells[(key, b, source)]["attempt_seqs"][:1] for b in (8, 16)}
                for first in (True, False):
                    for b in (8, 16): walls[first][b].append(_cell_wall(cells[(key, b, source)], first, outcomes, views))
                if not all(firsts.values()):
                    coverage["missing_request_pairs"] += 1
                    continue
                coverage["paired_requests"] += 1
                ready = [_ready(outcomes[firsts[b][0]]) for b in (8, 16)]
                fields = ("prompt", "prompt_ids", "eos_token_ids")
                same = (all(r is not None for r in ready)
                        and digest([ready[0][k] for k in fields]) == digest([ready[1][k] for k in fields]))
                coverage["matching_ready_pairs" if same else "missing_ready_pairs" if any(r is None for r in ready) else "mismatched_ready_pairs"] += 1
                native_pairs += same and all(rates[firsts[b][0]] is not None for b in (8, 16))
            timing.append(dict(source=source, initial_inputs=coverage, first_reply_wall=_wall(walls[True], valid),
                               final_policy_wall=_wall(walls[False], valid), native_first_pairs=native_pairs))
        subset = dict(name=name, denominator=len(keys), terminal_cells=terminal, complete=terminal == 4 * len(keys),
                      arms=arms, joint_contingency=contingencies, timing=timing)
        if name == "primary50":
            subset["target_met"] = [dict(bits=a["bits"], first_reply=a["first_verified"]["joint"] >= 15 if valid and subset["complete"] else None,
                final_policy=a["final_verified"]["joint"] >= 15 if valid and subset["complete"] else None) for a in arms]
        output.append(subset)
    return output


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
                for m, _, assigned in rows], metrics=_metrics(layout, rows, run_map, eligible, outcomes, reservations, views, not invalid)))
    return trials
