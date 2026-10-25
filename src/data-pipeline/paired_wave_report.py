"""Historical resident-wave structure checks; scientific assessment is provisional."""

import json
from pathlib import Path
from . import paired_collection as PC, paired_report as PR, paired_wave_collection as W
from .paired_collection import J, L, study, digest


def _bindings(events):
    setups, plans, steps, confirmations = {}, {}, [], []
    for event in (e for e in events if e["kind"] in ("wave_setup", "wave_plan", "wave_step", "wave_confirmation")):
        p, kind = event["payload"], event["kind"]
        J.require(set(event) == {"seq", "kind", "payload", "payload_sha256"} and type(p) is dict
            and type(p.get("plan_id")) is str and PC.PLAN_ID.fullmatch(p["plan_id"])
            and digest(p) == event["payload_sha256"], "Invalid wave event")
        if kind in ("wave_setup", "wave_plan"):
            target = setups if kind == "wave_setup" else plans
            J.require(p["plan_id"] not in target, "Duplicate wave binding")
            target[p["plan_id"]] = event
        else:
            fields = {"plan_id", "plan_sha256"} | ({"opportunity_id", "session_seq"} if kind == "wave_step" else {"decision", "rationale"})
            J.require(set(p) == fields and PC.P._sha(p["plan_sha256"]), "Invalid wave control event")
            J.require((type(p["opportunity_id"]) is int and p["opportunity_id"] >= 0 and
                (p["session_seq"] is None or type(p["session_seq"]) is int and p["session_seq"] > 0)) if kind == "wave_step" else
                p["decision"] == "unchanged" and type(p["rationale"]) is str and bool(p["rationale"].strip()), "Invalid wave control event")
            (steps if kind == "wave_step" else confirmations).append(event)
    for event in setups.values():
        p = event["payload"]; item = p.get("protocol", {})
        J.require(set(p) == {"schema_version", "plan_id", "plan_sha256", "producer_sources", "protocol", "funding"}
            and type(p["schema_version"]) is int and p["schema_version"] == 1 and PC.P._sha(p["plan_sha256"])
            and type(p["producer_sources"]) is dict and bool(p["producer_sources"]) and all(type(k) is str and k and
                not Path(k).is_absolute() and str(Path(k)) == k and ".." not in Path(k).parts and PC.P._sha(v) for k, v in p["producer_sources"].items())
            and set(item) == {"path", "sha256", "bytes"} and PC.P._sha(item["sha256"]) and type(item["bytes"]) is int
            and item["bytes"] >= 0 and item["path"] == f"audits/{item['sha256']}-wave-protocol.md", "Invalid wave setup")
        limits, reserved = J.Campaign.totals(events[:event["seq"] - 1])
        bound = {s: dict(zip(J.UNITS, (432, 884736, 103680))) for s in study.SOURCES}
        J.require(digest(p["funding"]) == digest(dict(reserved_before=reserved, bound=bound,
            limits={s: {u: max(limits[s][u], reserved[s][u] + bound[s][u]) for u in J.UNITS} for s in study.SOURCES})), "Invalid wave funding")
    J.require(all(n in setups and setups[n]["seq"] < e["seq"] and setups[n]["payload"]["plan_sha256"] == e["payload_sha256"] for n, e in plans.items()), "Unbound wave plan")
    J.require(all(e["payload"]["plan_id"] in plans and e["payload"]["plan_sha256"] == plans[e["payload"]["plan_id"]]["payload_sha256"] for e in steps + confirmations), "Unbound wave control event")
    return setups, plans, steps, confirmations


def _plan(event, setup, events, records, runs, panel, invalid):
    loaded, activation, expected_panel, _, _, schedule = PR._layout()
    p, rows, methods = event["payload"], [], {}
    J.require(set(p) == PR.FIELDS | {"opportunities"} and type(p["schema_version"]) is int and p["schema_version"] == 1
        and all(PC.P._sha(p[k]) for k in PR.FIELDS - {"schema_version", "plan_id", "recipe_sha256", "records"})
        and p["activation_sha256"] == digest(activation) and p["panel_sha256"] == digest(panel) == digest(expected_panel)
        and type(p["records"]) is list and len(p["records"]) == 144 and set(p["recipe_sha256"]) == set(study.SOURCES), "Invalid wave plan")
    run_map, seen = {r["experiment_id"]: r for r in runs}, set()
    for member, (task, bits) in zip(p["records"], schedule):
        J.require(set(member) == {"task", "bits", "experiment_id", "record_sha256"} and set(member["task"]) == {"split", "id", "task_sha256"}
            and type(member["bits"]) is int and member["bits"] == bits and type(member["task"]["id"]) is int
            and PR._key(member["task"]) == PR._key(task) and member["experiment_id"] not in seen, "Invalid wave member")
        name = member["experiment_id"]; seen.add(name)
        J.require(name in records and name in run_map, "Missing wave member")
        registered, record = records[name], records[name]["record"]
        PR._recipes(record["recipes"])
        expected = {**study.experiment_record(loaded, name, activation, record["recipes"]), "schema_version": 3,
            "assigned_task_ids": [task["id"]], "policy_sha256": p["policy_sha256"], "profiles": record["profiles"], "generation_method": record["generation_method"]}
        J.require(digest(expected) == digest(record) == registered["record_sha256"] == member["record_sha256"]
            and set(record["profiles"]) == set(record["generation_method"]) == set(study.SOURCES), "Invalid wave member")
        contexts = [e for e in events if e["kind"] == "execution_context" and e["experiment_id"] == name]
        J.require(len(contexts) == 1 and contexts[0]["contract_sha256"] == p["contract_sha256"] == PC.V.digest(contexts[0]["contract"])
            and max(registered["seq"], contexts[0]["seq"]) < setup["seq"], "Invalid wave context")
        for source, method in record["generation_method"].items():
            descriptor = record["profiles"][source]["descriptor"]
            J.require((descriptor["source"], descriptor["bits"], descriptor["receipt_sha256"], descriptor["runtime_guard_sha256"])
                == (source, bits, p["receipt_sha256"], p["runtime_guard_sha256"]) and digest(record["recipes"][source]) == p["recipe_sha256"][source], "Invalid wave method")
            methods.setdefault(bits, record["generation_method"])
            J.require(digest(method) == digest(methods[bits][source]), "Inconsistent wave method")
        cells = [c for c in run_map[name]["cells"] if c["id"] == task["id"] and c["source"] in study.SOURCES]
        J.require(len(cells) == 2 and {c["source"] for c in cells} == set(study.SOURCES)
            and all(PR._key(c) == PR._key(task) for c in cells), "Invalid wave cells")
        rows.append((member, record, cells))
    for source in study.SOURCES:
        a, b = [methods[bits][source]["admission"] for bits in (8, 16)]
        J.require(digest(a["config"]) == digest(b["config"]) and ("cache_limit_bytes" in a) == ("cache_limit_bytes" in b)
            and a.get("cache_limit_bytes") == b.get("cache_limit_bytes") and type(a["config"]["batch_size"]) is int and 1 <= a["config"]["batch_size"] <= 8
            and all(methods[bits][source]["prompt_ceiling_tokens"] == min(a["prompt_ceiling_tokens"], b["prompt_ceiling_tokens"]) for bits in (8, 16)), "Invalid common wave method")
    J.require(digest(p["opportunities"]) == digest(W._opportunities(activation, p["records"], methods)), "Invalid wave opportunities")
    anchors = [PC._one(events, k) for k in ("panel", "mechanics_baseline")]
    baseline = anchors[1]["payload"]
    J.require(baseline["sources"] == PC.BASELINE_SOURCES and baseline["inventory_sha256"] == PC.INVENTORY_SHA256 and baseline["audit"]["sha256"] == PC.AUDIT_SHA256
        and all(a["seq"] < e["seq"] for a in anchors for e in events if e["kind"] in ("native_intent", "generation_reserved")), "Invalid wave anchors")
    return rows


def _effects(prefix, seqs):
    done, results, proved = PR._views(prefix)
    groups = {e["seq"]: e for e in prefix if e["kind"] == "batch_done"}
    reservations = {e["seq"]: e for e in prefix if e["kind"] == "generation_reserved"}
    contexts = {e["experiment_id"]: e["contract_sha256"] for e in prefix if e["kind"] == "execution_context"}
    gates = {s: PR._views([e for e in prefix if e["seq"] < s])[2] &
        {e["operation_id"] for e in prefix if e["kind"] == "native_verdict" and e["status"] == "passed" and e["seq"] < s} for s in seqs}
    return all(s in done and s in results and done[s].get("batch_done_seq") in groups
        and groups[done[s]["batch_done_seq"]]["shared"]["resources_closed"] is True
        and any(e["kind"] == "native_intent" and e["seq"] < s and e["reference_indices"] is not None and e["operation_id"] in gates[s] and e["task_sha256"] == reservations[s]["attempt"]["task_sha256"] and e["contract_sha256"] == contexts.get(reservations[s]["experiment_id"]) for e in prefix)
        and (results[s]["operation_id"] is None or results[s]["operation_id"] in proved) for s in seqs)


def _repair(slots, outcomes, events, invalid):
    initial = json.loads(slots[0]["request_text"])["messages"]
    if len(initial) != 2 or any(set(m) != {"role", "content"} or m["role"] != role or type(m["content"]) is not str for m, role in zip(initial, ("system", "user"))):
        invalid.append(dict(kind="initial_messages_invalid", reservation_seq=slots[0]["seq"]))
    for previous, current in zip(slots, slots[1:]):
        prefix = [e for e in events if e["seq"] < current["seq"]]
        result = next(e for e in events if e["kind"] == "attempt_result" and e["reservation_seq"] == previous["seq"])
        outcome, repair = outcomes[previous["seq"]], L.FORMAT_REPAIR
        valid = _effects(prefix, [previous["seq"]]) and outcome["eligible"] is True and outcome["status"] in ("stop", "length")
        if result["status"] == "format_failed":
            try: L.generation.extract_program(outcome["text"])
            except ValueError: pass
            else: valid = False
        elif result["status"] == "verification_failed":
            verdict = next((e for e in events if e["kind"] == "native_verdict" and e["operation_id"] == result["operation_id"]), None)
            proof = next((r for r in verdict["results"] if r["sample_id"] == result["sample_id"]), None) if verdict else None
            if proof is None:
                invalid.append(dict(kind="repair_history_mismatch", reservation_seq=current["seq"])); continue
            valid = valid and proof["status"] == "failed" and proof["key"]["program_sha256"] == result["candidate_sha256"]
            repair = "Repair using this public feedback:\n" + json.dumps(L.project_feedback(proof["feedback"]), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else: valid = False
        expected = initial + [{"role": "assistant", "content": outcome["text"]}, {"role": "user", "content": repair}]
        if not valid or digest(json.loads(current["request_text"])["messages"]) != digest(expected):
            invalid.append(dict(kind="repair_history_mismatch", reservation_seq=current["seq"]))


def _requests(event, rows, events, reservations, outcomes, ready, invalid):
    own = {m["experiment_id"]: (m, r) for m, r, _ in rows}
    setup = next(e for e in events if e["kind"] == "wave_setup" and e["payload"]["plan_id"] == event["payload"]["plan_id"])
    superseded = min((e["seq"] for e in events if e["kind"] in ("collection_setup", "wave_setup") and e["seq"] > setup["seq"]), default=len(events) + 1)
    opportunities = event["payload"]["opportunities"]
    slots = {(name, op["source"], op["reply_index"]): op["id"] for op in opportunities for name in op["experiment_ids"]}
    mapping, first, histories, used, last = {}, {}, {}, set(), -1
    for batch in (e for e in events if e["kind"] == "batch_reserved"):
        members = [reservations[r["reservation_seq"]] for r in batch["rows"]]
        selected = [r for r in members if r["experiment_id"] in own]
        if not selected: continue
        ids = {slots.get((r["experiment_id"], r["source"], r["attempt"]["reply_index"])) for r in selected}
        opid = next(iter(ids)) if len(ids) == 1 and None not in ids and len(selected) == len(members) else None
        mapping[batch["seq"]] = opid
        if opid is None:
            invalid.append(dict(kind="mixed_opportunity_session", session_seq=batch["seq"])); continue
        op = opportunities[opid]; names = [r["experiment_id"] for r in members]
        J.require(names == [n for n in op["experiment_ids"] if n in names] and len(set(names)) == len(names), "Invalid wave session rows")
        J.require(all(digest(batch["method_spec"]) == digest(own[r["experiment_id"]][1]["generation_method"][op["source"]]) for r in members), "Invalid wave session method")
        prior_steps = {e["payload"]["opportunity_id"] for e in events if e["kind"] == "wave_step"
            and e["payload"]["plan_id"] == event["payload"]["plan_id"] and e["seq"] < batch["seq"]}
        if not set(range(opid)) <= prior_steps:
            invalid.append(dict(kind="opportunity_checkpoint_gap", session_seq=batch["seq"]))
        if opid in used or opid < last: invalid.append(dict(kind="opportunity_order_mismatch", session_seq=batch["seq"]))
        used.add(opid); last = max(last, opid)
    for seq, reservation in sorted(reservations.items()):
        name = reservation["experiment_id"]
        if name not in own: continue
        member, record = own[name]; source, item = reservation["source"], reservation["attempt"]
        J.require(source in study.SOURCES and PR._key(item) == PR._key(member["task"]) and type(item["reply_index"]) is int
            and 0 <= item["reply_index"] < 3 and "session_seq" in reservation, "Invalid wave request")
        request, recipe = json.loads(reservation["request_text"]), record["recipes"][source]
        J.require(digest({k: request[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds", "seed")}) == digest({**{k: recipe[k]
            for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}, "seed": recipe["seeds"][item["reply_index"]]}), "Invalid wave request controls")
        if seq <= event["seq"] or seq > superseded: invalid.append(dict(kind="member_before_plan" if seq <= event["seq"] else "work_after_supersession", reservation_seq=seq))
        reason = J.profile_control_reason(outcomes[seq])
        if reason: invalid.append(dict(kind=reason, reservation_seq=seq))
        histories.setdefault((name, source), []).append(reservation)
        if item["reply_index"] == 0:
            observed = ready.get(reservation["session_seq"])
            fixture = next((f for f in observed["fixtures"] if f["id"] == outcomes[seq]["row_id"]), None) if observed else None
            first.setdefault((PR._key(member["task"]), source), {})[member["bits"]] = (request, None if fixture is None else [fixture["prompt"], fixture["prompt_ids"], observed["eos_token_ids"]])
    for history in histories.values(): _repair(history, outcomes, events, invalid)
    for pair in first.values():
        if set(pair) != {8, 16}: continue
        if digest({k: v for k, v in pair[8][0].items() if k not in ("model_path", "precision_profile")}) != digest({k: v for k, v in pair[16][0].items() if k not in ("model_path", "precision_profile")}):
            invalid.append(dict(kind="initial_requests_mismatch"))
        if pair[8][1] is not None and pair[16][1] is not None and digest(pair[8][1]) != digest(pair[16][1]): invalid.append(dict(kind="initial_ready_mismatch"))
    return mapping


def _steps(steps, event, rows, mapping, events, invalid):
    seen, records = set(), {m["experiment_id"]: r for m, r, _ in rows}
    batches = {e["seq"]: e for e in events if e["kind"] == "batch_reserved"}
    for step in steps:
        p = step["payload"]; opid, session = p["opportunity_id"], p["session_seq"]
        J.require(opid < len(event["payload"]["opportunities"]), "Invalid wave step opportunity")
        if opid in seen: invalid.append(dict(kind="step_duplicate", opportunity_id=opid))
        seen.add(opid)
        prefix = [e for e in events if e["seq"] < step["seq"]]
        if session is not None:
            J.require(session in mapping and mapping[session] == opid, "Invalid wave step session")
            valid = _effects(prefix, [r["reservation_seq"] for r in batches[session]["rows"]])
        else:
            op = event["payload"]["opportunities"][opid]; views = PR._views(prefix)
            cells = [c for name in op["experiment_ids"] for c in L.cell_states(prefix, records[name])
                if c["source"] == op["source"] and c["id"] in records[name]["assigned_task_ids"]]
            valid = opid not in mapping.values() and all(PR._terminal(c, views) and _effects(prefix, c["attempt_seqs"]) for c in cells)
        if step["seq"] <= event["seq"] or not valid: invalid.append(dict(kind="step_premature", opportunity_id=opid))


def _validate(*, events, records, runs, outcomes, reservations, panel, ready_by_session):
    setups, plans, steps, confirmations = _bindings(events)
    facts = []
    for name, event in plans.items():
        invalid = []; rows = _plan(event, setups[name], events, records, runs, panel, invalid)
        mapping = _requests(event, rows, events, reservations, outcomes, ready_by_session, invalid)
        first_registration = min(records[m["experiment_id"]]["seq"] for m, _, _ in rows)
        if any(e["kind"] == "native_intent" and first_registration < e["seq"] <= event["seq"] and e["task_sha256"] in {m["task"]["task_sha256"] for m, _, _ in rows} and e["contract_sha256"] == event["payload"]["contract_sha256"] for e in events): invalid.append(dict(kind="native_before_plan"))
        own_steps = [e for e in steps if e["payload"]["plan_id"] == name]
        _steps(own_steps, event, rows, mapping, events, invalid)
        if sum(e["payload"]["plan_id"] == name for e in confirmations) > 1: invalid.append(dict(kind="confirmation_duplicate"))
        facts.append(dict(setup=setups[name], plan=event, rows=rows, session_opportunities=mapping, steps=own_steps,
            confirmations=[e for e in confirmations if e["payload"]["plan_id"] == name], invalid_reasons=invalid))
    return facts


def derive(*, events, records, runs, eligible, outcomes, reservations, panel, sessions, ready_by_session):
    return [dict(report_scope="resident_wave", validation_scope="structure_only", provisional=True,
        plan_id=f["plan"]["payload"]["plan_id"], plan_sha256=f["plan"]["payload_sha256"], structure_valid=not f["invalid_reasons"], invalid_reasons=f["invalid_reasons"])
        for f in _validate(events=events, records=records, runs=runs, outcomes=outcomes, reservations=reservations, panel=panel, ready_by_session=ready_by_session)]
