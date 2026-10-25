"""Freeze metadata and finite resident-wave opportunities; setup does not acquire."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from experiments import batch_throughput as H
from . import paired_collection as PC
from .paired_collection import J, L, P, V, study, digest

PROTOCOL = PC.ROOT / "docs/stage-03-wave-protocol.md"
BASE_PROTOCOL_SHA256 = "1202470310562c5c3369228c7a065d779a49657ab5432fff2febb3017de15e1f"
PRODUCERS = PC.PRODUCERS + ("src/data-pipeline/paired_wave_collection.py",
    "experiments/batch_throughput.py", "experiments/mlx_batch_throughput.py",
    "src/data-pipeline/study_report.py", "src/data-pipeline/paired_wave_report.py",
    "src/data-pipeline/paired_report.py")


def _opportunities(activation, members, methods):
    tasks = activation["tasks"]
    groups = [[t for t in tasks if t["group"] == g] for g in study.GROUPS]
    blocks = (groups[0], groups[1][:16], groups[2][:16], groups[1][16:], groups[2][16:])
    ordinals = {t["id"]: i for i, t in enumerate(tasks)}
    names = {(m["task"]["id"], m["bits"]): m["experiment_id"] for m in members}
    capacities = {s: methods[8][s]["admission"]["config"]["batch_size"] for s in study.SOURCES}
    size, result, group = max(capacities.values()), [], 0
    for block_index, (block, selected) in enumerate(zip(("dev", "A_tune", "B_tune", "A_fresh", "B_fresh"), blocks)):
        parity = [[t for t in selected if ordinals[t["id"]] % 2 == p] for p in (0, 1)]
        chunks = [[rows[i:i + size] for i in range(0, len(rows), size)] for rows in parity]
        for index in range(max(map(len, chunks))):
            for p in (block_index % 2, 1 - block_index % 2):
                if index >= len(chunks[p]): continue
                rows = chunks[p][index]
                for bits in ((8, 16) if p == 0 else (16, 8)):
                    for source in study.SOURCES:
                        for reply in range(PC.RECIPE["max_replies"]):
                            for offset in range(0, len(rows), capacities[source]):
                                result.append(dict(id=len(result), block=block, group=group, bits=bits,
                                    source=source, reply_index=reply, experiment_ids=[names[t["id"], bits]
                                        for t in rows[offset:offset + capacities[source]]]))
                group += 1
    return result


def build(loaded, plan_id, profiles, methods, contract_sha256):
    J.require(set(methods) == {8, 16} and all(set(m) == set(study.SOURCES) for m in methods.values()), "Wave methods differ")
    for source in study.SOURCES:
        pair = [methods[b][source] for b in (8, 16)]
        for method in pair: J._batch_method(method)
        a, b = [m["admission"] for m in pair]
        J.require(a["schema_version"] == b["schema_version"] and a.get("contract") == b.get("contract")
            and J.encoded(a["config"]) == J.encoded(b["config"])
            and ("cache_limit_bytes" in a) == ("cache_limit_bytes" in b)
            and a.get("cache_limit_bytes") == b.get("cache_limit_bytes")
            and type(a["config"]["batch_size"]) is int and 1 <= a["config"]["batch_size"] <= 8
            and all(m["prompt_ceiling_tokens"] == min(a["prompt_ceiling_tokens"], b["prompt_ceiling_tokens"]) for m in pair), "Common wave method differs")
    built = PC.build(loaded, plan_id, profiles, contract_sha256)
    for index, (record, member) in enumerate(zip(built["records"], built["plan"]["records"])):
        record = L.run_record(loaded, record["experiment_id"], list(study.GROUPS), record["recipes"],
            record["assigned_task_ids"], profiles=record["profiles"], construction_per_phase=32,
            generation_method=methods[member["bits"]])
        built["records"][index], member["record_sha256"] = record, digest(record)
    built["plan"]["opportunities"] = _opportunities(built["activation"], built["plan"]["records"], methods)
    return built


def _methods(campaign, qualifications=None, records=None):
    if records is not None:
        methods = {b: next(r["generation_method"] for r in records if r["profiles"]["actor"]["descriptor"]["bits"] == b) for b in (8, 16)}
        for bound in methods.values():
            for method in bound.values():
                J._batch_method(method, campaign); H.current_source_compatible(method["admission"])
        return methods
    J.require(type(qualifications) is dict and set(qualifications) == {8, 16}
        and all(type(q) is dict and set(q) == set(study.SOURCES) for q in qualifications.values()), "Require four qualifications")
    methods = {8: {}, 16: {}}
    directory = campaign.root / "qualifications"
    J.require(directory.resolve() == directory, "Qualification directory differs")
    directory.mkdir(exist_ok=True)
    J.sync_directory(campaign.root)
    for bits, bound in qualifications.items():
        for source, reference in bound.items():
            J.require(type(reference) is dict and set(reference) == {"path", "response_sha256"}, "Qualification input differs")
            identity = reference["response_sha256"]
            admission, files = H.read_qualification(reference["path"], identity)
            J.require(admission["schema_version"] == 2 and admission.get("contract") == "peer-isolation-v2",
                "Require peer-isolation-v2 qualification")
            final = directory / identity
            if not os.path.lexists(final):
                with tempfile.TemporaryDirectory(prefix=".copy-", dir=directory) as temporary:
                    H.copy_qualification(files, Path(temporary) / "owned", identity)
                    os.rename(Path(temporary) / "owned", final)
                J.sync_directory(directory)
            J.require(final.resolve() == final, "Qualification directory differs")
            retained, copied = H.read_qualification(final, identity)
            J.require(copied == files and J.encoded(retained) == J.encoded(admission), "Owned qualification differs")
            H.current_source_compatible(retained)
            methods[bits][source] = dict(method="program-wave-v1", source=source, admission=retained,
                prompt_ceiling_tokens=retained["prompt_ceiling_tokens"], qualification=dict(path="qualifications/" + identity, response_sha256=identity))
    for source in study.SOURCES:
        ceiling = min(methods[b][source]["prompt_ceiling_tokens"] for b in (8, 16))
        for bits in (8, 16): methods[bits][source]["prompt_ceiling_tokens"] = ceiling
    return methods


def fingerprint():
    return {name: sha256((PC.ROOT / name).read_bytes()).hexdigest() for name in PRODUCERS}


def _producer_guard(campaign, payload):
    J.require(payload["producer_sources"] == fingerprint(), "Producer sources changed")
    J.require(sha256(PC.PROTOCOL.read_bytes()).hexdigest() == BASE_PROTOCOL_SHA256, "Base protocol changed")
    J.require(PC._reference(campaign, dict(path=f"audits/{BASE_PROTOCOL_SHA256}-paired-protocol.md", sha256=BASE_PROTOCOL_SHA256, bytes=len(PC.PROTOCOL.read_bytes())), "paired-protocol.md") == PC.PROTOCOL.read_bytes(), "Base protocol copy changed")
    J.require(PC._reference(campaign, payload["protocol"], "wave-protocol.md") == PROTOCOL.read_bytes(), "Protocol source changed")


def _member_work(events, built):
    names = {r["experiment_id"] for r in built["records"]}
    hashes = {t["sha256"] for t in built["activation"]["tasks"]}
    registered = [e["seq"] for e in events if e["kind"] == "experiment" and e["experiment_id"] in names]
    reservations = {e["seq"] for e in events if e["kind"] == "generation_reserved" and e["experiment_id"] in names}
    return [e for e in events if e["seq"] in reservations or e["kind"] == "native_intent" and
        (e.get("reservation_seq") in reservations or registered and e["seq"] > min(registered)
         and e["task_sha256"] in hashes and e["contract_sha256"] == built["plan"]["contract_sha256"])]


def setup(campaign, loaded, plan_id, receipt_path, receipt_sha256, mechanics_audit_path, qualifications):
    J.require(type(plan_id) is str and PC.PLAN_ID.fullmatch(plan_id), "Invalid plan ID")
    J.require(sha256(PC.PROTOCOL.read_bytes()).hexdigest() == BASE_PROTOCOL_SHA256, "Base protocol changed")
    events = campaign.events
    previous = [e for e in events if e["kind"] == "wave_setup" and e["payload"]["plan_id"] == plan_id]
    J.require(len(previous) <= 1, "Duplicate wave setup")
    if previous:
        built, anchor = _retained(campaign, loaded, plan_id, incomplete=True)
    else:
        J.require(not any(e["kind"] == "collection_setup" and e["payload"]["plan_id"] == plan_id for e in events), "Plan ID belongs to cold collection")
        J.require({e["seq"] for e in events if e["kind"] == "generation_reserved"}
            == {e["reservation_seq"] for e in events if e["kind"] == "generation_done"}, "Pending generations")
        campaign.require_native_clear()
        J.require(all(J.generation_resources_cleared(events, campaign.outcome(e["seq"])) for e in events if e["kind"] == "generation_reserved"), "Unresolved generation cleanup failure")
        audit_raw = Path(mechanics_audit_path).read_bytes()
        J.require(sha256(audit_raw).hexdigest() == PC.AUDIT_SHA256, "Mechanics audit differs")
        PC._copy(campaign, PC.PROTOCOL.read_bytes(), "paired-protocol.md")
        audit, protocol = PC._copy(campaign, audit_raw, "mechanics.txt"), PC._copy(campaign, PROTOCOL.read_bytes(), "wave-protocol.md")
        reference = P.freeze(campaign, receipt_path, receipt_sha256)
        profiles = {b: {s: dict(reference=reference, descriptor=P.resolve(campaign, reference, s, b)) for s in study.SOURCES} for b in (8, 16)}
        methods = _methods(campaign, qualifications)
        contract = V.runner.contract()
        built = build(loaded, plan_id, profiles, methods, V.digest(contract))
        PC._event(campaign, "panel", built["panel"], early=True)
        PC._event(campaign, "mechanics_baseline", dict(sources=PC.BASELINE_SOURCES, combined={u: sum(v[u] for v in PC.BASELINE_SOURCES.values())
            for u in PC.BASELINE_SOURCES["actor"]}, audit=audit, inventory_sha256=PC.INVENTORY_SHA256, limitations=PC.BASELINE_LIMITATIONS), early=True)
        J.require(not _member_work(events, built), "Member work precedes wave plan")
        for record in built["records"]:
            campaign.register(record)
            context = dict(experiment_id=record["experiment_id"], contract=contract, contract_sha256=built["plan"]["contract_sha256"])
            prior = [e for e in campaign.events if e["kind"] == "execution_context" and e["experiment_id"] == record["experiment_id"]]
            J.require(len(prior) <= 1 and (not prior or J.encoded({k: prior[0][k] for k in context}) == J.encoded(context)), "Execution context differs")
            if not prior: campaign._append("execution_context", **context)
        limits, reserved = J.Campaign.totals(campaign.events)
        bound = {s: dict(zip(J.UNITS, (432, 884736, 103680))) for s in study.SOURCES}
        funding = dict(reserved_before=reserved, bound=bound, limits={s: {u: max(limits[s][u], reserved[s][u] + bound[s][u]) for u in J.UNITS} for s in study.SOURCES})
        anchor = PC._event(campaign, "wave_setup", dict(schema_version=2, qualification_contract="peer-isolation-v2",
            plan_id=plan_id, plan_sha256=digest(built["plan"]),
            producer_sources=fingerprint(), protocol=protocol, funding=funding))
    current, _ = campaign._accounting()
    campaign.extend({s: {u: max(current[s][u], anchor["payload"]["funding"]["limits"][s][u]) for u in J.UNITS} for s in study.SOURCES})
    event = PC._event(campaign, "wave_plan", built["plan"])
    _retained(campaign, loaded, plan_id)
    return event


def _retained(campaign, loaded, plan_id, *, incomplete=False):
    J.require(type(plan_id) is str and PC.PLAN_ID.fullmatch(plan_id), "Invalid plan ID")
    events = campaign.events
    anchor = PC._one(events, "wave_setup", plan_id)
    payload = anchor["payload"]
    fields = {"schema_version", "plan_id", "plan_sha256", "producer_sources", "protocol", "funding"}
    J.require(type(payload.get("schema_version")) is int and payload["schema_version"] in (1, 2)
        and set(payload) == fields | ({"qualification_contract"} if payload["schema_version"] == 2 else set())
        and (payload["schema_version"] == 1 or payload["qualification_contract"] == "peer-isolation-v2"), "Wave binding differs")
    J.require(not any(e["kind"] in ("collection_setup", "wave_setup") and e["seq"] > anchor["seq"] for e in events), "Collection superseded")
    _producer_guard(campaign, payload)
    registered = {e["experiment_id"]: e for e in events if e["kind"] == "experiment"}
    J.require(len(registered) == sum(e["kind"] == "experiment" for e in events), "Duplicate experiment record")
    records = [e["record"] for e in registered.values() if e["experiment_id"].rsplit("-", 3)[0] == plan_id]
    J.require(len(records) == 144, "Wave members differ")
    methods = _methods(campaign, records=records)
    J.require(all(m["admission"]["schema_version"] == payload["schema_version"]
        and (("contract" not in m["admission"]) if payload["schema_version"] == 1 else
            m["admission"].get("contract") == "peer-isolation-v2")
        for bound in methods.values() for m in bound.values()), "Wave qualification authority differs")
    profiles = {b: next(r["profiles"] for r in records if r["profiles"]["actor"]["descriptor"]["bits"] == b) for b in (8, 16)}
    for bound in profiles.values(): P.check_profiles(bound, study.SOURCES, campaign=campaign)
    contract = V.runner.contract()
    built = build(loaded, plan_id, profiles, methods, V.digest(contract))
    J.require(digest(built["plan"]) == payload["plan_sha256"], "Wave plan differs")
    for record in built["records"]:
        old = registered.get(record["experiment_id"])
        contexts = [e for e in events if e["kind"] == "execution_context" and e["experiment_id"] == record["experiment_id"]]
        J.require(old is not None and old["record_sha256"] == digest(old["record"]) == digest(record)
            and len(contexts) == 1 and max(old["seq"], contexts[0]["seq"]) < anchor["seq"]
            and J.encoded(contexts[0]["contract"]) == J.encoded(contract) and contexts[0]["contract_sha256"] == built["plan"]["contract_sha256"], "Wave member differs")
    panel, baseline = PC._one(events, "panel"), PC._one(events, "mechanics_baseline")
    audit = baseline["payload"]["audit"]
    expected = dict(sources=PC.BASELINE_SOURCES, combined={u: sum(v[u] for v in PC.BASELINE_SOURCES.values()) for u in PC.BASELINE_SOURCES["actor"]},
        audit=audit, inventory_sha256=PC.INVENTORY_SHA256, limitations=PC.BASELINE_LIMITATIONS)
    J.require(panel["payload_sha256"] == digest(built["panel"]) and baseline["payload_sha256"] == digest(expected)
        and sha256(PC._reference(campaign, audit, "mechanics.txt")).hexdigest() == PC.AUDIT_SHA256
        and all(a["seq"] < e["seq"] for a in (panel, baseline) for e in events if e["kind"] in ("generation_reserved", "native_intent")), "Global anchors differ")
    limits, reserved = J.Campaign.totals(events[:anchor["seq"] - 1])
    bound = {s: dict(zip(J.UNITS, (432, 884736, 103680))) for s in study.SOURCES}
    expected = dict(reserved_before=reserved, bound=bound, limits={s: {u: max(limits[s][u], reserved[s][u] + bound[s][u]) for u in J.UNITS} for s in study.SOURCES})
    J.require(digest(expected) == digest(payload["funding"]), "Wave funding differs")
    plans = [e for e in events if e["kind"] == "wave_plan" and e["payload"]["plan_id"] == plan_id]
    if not incomplete or plans:
        plan = PC._one(events, "wave_plan", plan_id)
        current, _ = campaign._accounting()
        J.require(anchor["seq"] < plan["seq"] and plan["payload_sha256"] == digest(built["plan"])
            and all(current[s][u] >= expected["limits"][s][u] for s in study.SOURCES for u in J.UNITS), "Wave binding differs")
    J.require(all(plans and e["seq"] > plans[0]["seq"] for e in _member_work(events, built)), "Member work precedes wave plan")
    return built, anchor


def _guard(campaign, anchor):
    _producer_guard(campaign, anchor["payload"])
    J.require(not any(e["kind"] in ("collection_setup", "wave_setup") and e["seq"] > anchor["seq"]
        for e in campaign.events), "Collection superseded")


def _live(campaign, anchor, names):
    _guard(campaign, anchor)
    _, outcomes = L._reconcile_generations(campaign, names)
    L.native.reconcile_native(campaign)
    campaign.require_native_clear()
    return campaign.events, outcomes


def _history(built, events, own_outcomes, allowed_ids):
    from . import paired_wave_report as R
    _, plans, all_steps, _ = R._bindings(events)
    plan = plans[built["plan"]["plan_id"]]
    J.require(plan["payload_sha256"] == digest(built["plan"]), "Wave plan differs")
    records = {r["experiment_id"]: r for r in built["records"]}
    rows = [(m, records[m["experiment_id"]], [c for c in L.cell_states(events, records[m["experiment_id"]])
        if c["id"] == m["task"]["id"] and c["source"] in study.SOURCES]) for m in built["plan"]["records"]]
    reservations = {e["seq"]: e for e in events if e["kind"] == "generation_reserved"}
    invalid = []
    mapping = R._requests(plan, rows, events, reservations, own_outcomes, {}, invalid)
    selected = [e for e in all_steps if e["payload"]["plan_id"] == built["plan"]["plan_id"]]
    R._steps(selected, plan, rows, mapping, events, invalid)
    J.require(not invalid, "Invalid wave execution history")
    ids = [e["payload"]["opportunity_id"] for e in selected]
    J.require(ids == list(range(len(ids))) and set(ids) <= allowed_ids
        and all(op in allowed_ids for op in mapping.values()), "Invalid wave execution prefix")
    batches = {e["seq"]: e for e in events if e["kind"] == "batch_reserved"}
    sessions, steps = {op: batches[seq] for seq, op in mapping.items()}, dict(zip(ids, selected))
    J.require(set(sessions) - set(steps) <= {len(steps)}, "Uncheckpointed earlier opportunity")
    return plan, rows, sessions, steps


def _step(campaign, plan, rows, sessions, steps, opportunity_id, session_seq):
    from . import paired_wave_report as R
    payload = dict(plan_id=plan["payload"]["plan_id"], plan_sha256=plan["payload_sha256"],
        opportunity_id=opportunity_id, session_seq=session_seq)
    if opportunity_id in steps:
        J.require(digest(steps[opportunity_id]["payload"]) == digest(payload), "Wave step differs")
        return steps[opportunity_id]
    J.require(opportunity_id == len(steps), "Wave checkpoint gap")
    events = campaign.events
    proposed = dict(seq=events[-1]["seq"] + 1, kind="wave_step", payload=payload, payload_sha256=digest(payload))
    invalid = []
    R._steps([proposed], plan, rows, {s["seq"]: op for op, s in sessions.items()}, events, invalid)
    J.require(not invalid, "Unsupported wave checkpoint")
    event = campaign._append("wave_step", payload=payload, payload_sha256=digest(payload))
    steps[opportunity_id] = event
    return event


def _execute(campaign, loaded, built, anchor, opportunities):
    from . import paired_wave_report as R
    ids = [o["id"] for o in opportunities]
    J.require(bool(ids) and ids == list(range(ids[0], ids[-1] + 1))
        and digest(opportunities) == digest([o for o in built["plan"]["opportunities"]
            if o["id"] in set(ids)]), "Invalid wave opportunity slice")
    records = {r["experiment_id"]: r for r in built["records"]}
    events, outcomes = _live(campaign, anchor, set(records))
    plan, rows, sessions, steps = _history(built, events, outcomes, set(range(opportunities[-1]["id"] + 1)))
    J.require(set(range(opportunities[0]["id"])) <= steps.keys(), "Missing earlier wave checkpoints")
    output = []
    for op in opportunities:
        _live(campaign, anchor, set(records))
        opid, source = op["id"], op["source"]
        session, existing = sessions.get(opid), opid in steps
        if session is None:
            prepared = []
            for name in op["experiment_ids"]:
                record = records[name]; _guard(campaign, anchor)
                row = L.prepare_cell(campaign, loaded, record, source, record["assigned_task_ids"][0])
                if row is None:
                    cell = next(c for c in L.cell_states(campaign.events, record)
                        if c["id"] == record["assigned_task_ids"][0] and c["source"] == source)
                    J.require(R._usable(cell, campaign.events), "Cell lacks usable terminal evidence")
                    continue
                task = next(m["task"] for m in built["plan"]["records"] if m["experiment_id"] == name)
                J.require(not existing and set(row) == {"experiment_id", "source", "request", "attempt"}
                    and row["experiment_id"] == name and row["source"] == source
                    and digest(row["attempt"]) == digest({**task, "reply_index": op["reply_index"]})
                    and row["request"]["seed"] == record["recipes"][source]["seeds"][op["reply_index"]], "Prepared wave row differs")
                prepared.append(row)
            if prepared:
                _live(campaign, anchor, set(records))
                generated = campaign.generate_batch(prepared, records[prepared[0]["experiment_id"]]["generation_method"][source])
                session = next(e for e in campaign.events if e["kind"] == "batch_reserved" and e["seq"] == generated[0]["session_seq"])
                sessions[opid] = session
        if session is not None:
            reserved = {e["seq"]: e for e in campaign.events if e["kind"] == "generation_reserved"}
            actual = [reserved[r["reservation_seq"]] for r in session["rows"]]
            prefix = [e for e in campaign.events if e["seq"] < session["seq"]]
            for name in set(op["experiment_ids"]) - {r["experiment_id"] for r in actual}:
                record = records[name]
                cell = next(c for c in L.cell_states(prefix, record)
                    if c["id"] == record["assigned_task_ids"][0] and c["source"] == source)
                J.require(R._usable(cell, prefix), "Omitted wave cell was not terminal")
                _guard(campaign, anchor)
                if cell["attempt_seqs"]:
                    L.settle_cell(campaign, loaded, record, source, cell["id"])
                else:
                    binding, task = study.load_task(loaded, record["activation"], cell["id"])
                    intent, proof = L._proof(campaign, task, binding["sha256"], built["plan"]["contract_sha256"], None, None, False)
                    J.require(intent is not None and intent["operation_id"] == cell["gate_operation_id"]
                        and proof is not None and proof["status"] == "failed", "Omitted gate proof differs")
            for reservation in actual:
                record = records[reservation["experiment_id"]]; _guard(campaign, anchor)
                L.settle_cell(campaign, loaded, record, source, reservation["attempt"]["id"])
            J.require(all(R._result_usable(r["seq"], campaign.events) for r in actual), "Unusable settled wave result")
        _guard(campaign, anchor)
        output.append(_step(campaign, plan, rows, sessions, steps, opid, session["seq"] if session else None))
        if not existing and session is not None:
            results = {e["reservation_seq"]: e for e in campaign.events if e["kind"] == "attempt_result"}
            if any(results[r["reservation_seq"]]["status"] in ("generation_timeout", "generation_error", "interrupted") for r in session["rows"]): break
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "plan-id", "receipt", "receipt-sha256", "mechanics-audit", "qualifications"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    with J.Campaign(args.campaign) as campaign:
        retained = any(e["kind"] == "wave_setup" and e["payload"]["plan_id"] == args.plan_id for e in campaign.events)
        qualifications = None if retained else {int(b): q for b, q in json.loads(Path(args.qualifications).read_text()).items()}
        print(J.encoded(setup(campaign, study.load_cohort(), args.plan_id, args.receipt, args.receipt_sha256, args.mechanics_audit, qualifications)))


if __name__ == "__main__":
    main()
