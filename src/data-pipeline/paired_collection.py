"""Freeze one sequential paired trial; this command dispatches no work."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from . import experiment_loop as L, generation_journal as J, precision_profiles as P
from . import study_tasks as study, verification as V

ROOT = Path(__file__).resolve().parents[2]
PRODUCERS = tuple("src/data-pipeline/" + name + ".py" for name in (
    "__init__", "paired_collection", "experiment_loop", "generation_journal", "precision_profiles",
    "local_generation", "mlx_worker", "local_runner", "native_recovery", "verification", "inspect_runner",
    "study_tasks", "publish_cohort", "analyze", "download", "export_test", "known_use", "qualify",
    "qualified_tasks", "cohort", "screen", "statement_examples")) + (
    "experiments/prepare_precision_artifacts.py", "experiments/bf16_compatibility.py",
    "configs/sandbox/compose.yaml")
PROTOCOL = ROOT / "docs/stage-03-paired-protocol.md"
AUDIT_SHA256 = "6d130a222b222c8fe734ca78de922a510210d72213e334b64a2f300a0f9df96c"
BASELINE_SOURCES = {"actor": dict(zip(("replies", "observed_generation_tokens", "reserved_output_tokens"), (7, 265, 4417))),
                    "synthetic": dict(zip(("replies", "observed_generation_tokens", "reserved_output_tokens"), (8, 166, 2048)))}
BASELINE_LIMITATIONS = ["Observed delivered tokens include an interrupted lower bound.",
                        "Full end-to-end elapsed time is unknown."]
INVENTORY_SHA256 = "c00ef131b44a7a78b08930dc08e4f67ccf0f309e9e307913fc7a955c0eb7cd57"
PLAN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
RECIPE = dict(system="You are an expert competitive programmer. Solve the supplied problem correctly and "
    "efficiently for every input allowed by its stated constraints, using Python 3.12 and the standard "
    "library. Return one executable solution.", instruction="Read the entire statement, including constraints "
    "and examples. Choose an algorithm whose time and memory costs fit the largest allowed input. Check "
    "boundary cases, indexing, strict versus non-strict inequalities, and integer arithmetic before "
    "finalizing the code. Implement the general solution rather than a rule fitted to sample outputs. "
    "Read from standard input and write only the required answer to standard output. Use efficient input "
    "handling where appropriate. Return the whole program, without explanation, sample runs, or a test harness.",
    max_replies=3, max_tokens=2048, wall_time_seconds=240, temperature=0.2, top_p=0.95,
    seeds=[20261024, 20261025, 20261026])


def digest(value):
    return sha256(J.encoded(value).encode()).hexdigest()


def fingerprint():
    return {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in PRODUCERS}


def _copy(campaign, raw, name):
    directory = campaign.root / "audits"
    J.require(not directory.is_symlink(), "Audit directory differs")
    directory.mkdir(exist_ok=True)
    final = directory / (sha256(raw).hexdigest() + "-" + name)
    if not os.path.lexists(final):
        descriptor, temporary = tempfile.mkstemp(prefix=".copy-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(raw); output.flush(); os.fsync(output.fileno())
            try: os.link(temporary, final)
            except FileExistsError: pass
        finally:
            os.unlink(temporary)
    with os.fdopen(os.open(final, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as existing:
        J.require(stat.S_ISREG(os.fstat(existing.fileno()).st_mode), "Require regular audit copy")
        J.require(existing.read() == raw, "Published copy differs")
    J.sync_directory(directory); J.sync_directory(campaign.root)
    return dict(path=str(final.relative_to(campaign.root)), sha256=sha256(raw).hexdigest(), bytes=len(raw))


def _event(campaign, kind, payload, *, early=False):
    previous = [e for e in campaign.events if e["kind"] == kind
                and ("plan_id" not in payload or e["payload"]["plan_id"] == payload["plan_id"])]
    J.require(len(previous) <= 1, "Duplicate collection event")
    if previous:
        J.require(set(previous[0]) == {"seq", "kind", "payload", "payload_sha256"}, "Collection event fields differ")
        J.require(not early or all(previous[0]["seq"] < e["seq"] for e in campaign.events
                  if e["kind"] in ("generation_reserved", "native_intent")), "Late global anchor")
        J.require(previous[0]["payload_sha256"] == digest(previous[0]["payload"]) == digest(payload), "Collection event differs")
        return previous[0]
    J.require(not early or not any(e["kind"] in ("generation_reserved", "native_intent") for e in campaign.events), "Late global anchor")
    return campaign._append(kind, payload=payload, payload_sha256=digest(payload))


def build(loaded, plan_id, profiles, contract_sha256):
    J.require(type(plan_id) is str and PLAN_ID.fullmatch(plan_id), "Invalid plan ID")
    J.require(set(profiles) == {8, 16}, "Require two precision arms")
    activation = study.activate(loaded, list(study.GROUPS), construction_per_phase=32)
    groups = [[t for t in activation["tasks"] if t["group"] == g] for g in study.GROUPS]
    J.require([len(g) for g in groups] == [8, 32, 32], "Require canonical seventy-two tasks")
    for bits, bound in profiles.items():
        P.check_profiles(bound, study.SOURCES)
        J.require(all(e["descriptor"]["bits"] == bits for e in bound.values()), "Profile arm differs")
    descriptors = [e["descriptor"] for bound in profiles.values() for e in bound.values()]
    J.require(len({d["receipt_sha256"] for d in descriptors}) == len({d["runtime_guard_sha256"] for d in descriptors}) == 1,
              "Profile authorities differ")
    panel = dict(activation=activation, tasks=groups[0] + groups[1][:21] + groups[2][:21])
    order = groups[0] + groups[1][:16] + groups[2][:16] + groups[1][16:] + groups[2][16:]
    ordinals = {t["id"]: i for i, t in enumerate(activation["tasks"])}
    records, members = [], []
    for task in order:
        ordinal = ordinals[task["id"]]
        for bits in ((8, 16) if ordinal % 2 == 0 else (16, 8)):
            name = f"{plan_id}-{ordinal:02d}-{task['id']}-{bits}"
            record = L.run_record(loaded, name, list(study.GROUPS), {s: RECIPE for s in study.SOURCES},
                                  [task["id"]], profiles=profiles[bits], construction_per_phase=32)
            records.append(record)
            members.append(dict(task=dict(split=task["split"], id=task["id"], task_sha256=task["sha256"]),
                                bits=bits, experiment_id=name, record_sha256=digest(record)))
    plan = dict(schema_version=1, plan_id=plan_id, activation_sha256=digest(activation), panel_sha256=digest(panel),
                receipt_sha256=descriptors[0]["receipt_sha256"], runtime_guard_sha256=descriptors[0]["runtime_guard_sha256"],
                recipe_sha256={s: digest(RECIPE) for s in study.SOURCES}, policy_sha256=records[0]["policy_sha256"],
                contract_sha256=contract_sha256, records=members)
    return dict(activation=activation, panel=panel, records=records, plan=plan)


def setup(campaign, loaded, plan_id, receipt_path, receipt_sha256, mechanics_audit_path):
    J.require(type(plan_id) is str and PLAN_ID.fullmatch(plan_id), "Invalid plan ID")
    events = campaign.events
    previous = [e for e in events if e["kind"] == "collection_setup" and e["payload"]["plan_id"] == plan_id]
    J.require(len(previous) <= 1, "Duplicate collection setup")
    J.require(not previous or not any(e["kind"] == "collection_setup" and e["seq"] > previous[0]["seq"] for e in events), "Collection superseded")
    producers = fingerprint()
    if previous: J.require(previous[0]["payload"]["producer_sources"] == producers, "Producer sources changed")
    J.require({e["seq"] for e in events if e["kind"] == "generation_reserved"}
              == {e["reservation_seq"] for e in events if e["kind"] == "generation_done"}, "Pending generations")
    campaign.require_native_clear()
    for event in events:
        if event["kind"] == "generation_reserved":
            response = campaign.outcome(event["seq"])["response"]
            J.require(response is None or response["cleanup_error"] is None, "Unresolved generation cleanup failure")
    contract = V.runner.contract()
    audit = Path(mechanics_audit_path).read_bytes()
    J.require(sha256(audit).hexdigest() == AUDIT_SHA256, "Mechanics audit differs")
    protocol_raw = PROTOCOL.read_bytes()
    if previous: J.require(sha256(protocol_raw).hexdigest() == previous[0]["payload"]["protocol"]["sha256"], "Protocol source changed")
    audit = _copy(campaign, audit, "mechanics.txt")
    protocol = _copy(campaign, protocol_raw, "paired-protocol.md")
    reference = P.freeze(campaign, receipt_path, receipt_sha256)
    profiles = {b: {s: dict(reference=reference, descriptor=P.resolve(campaign, reference, s, b))
                    for s in study.SOURCES} for b in (8, 16)}
    built = build(loaded, plan_id, profiles, V.digest(contract))
    owned = {r["experiment_id"] for r in built["records"]}
    prior_plan = [e for e in events if e["kind"] == "paired_plan" and e["payload"]["plan_id"] == plan_id]
    J.require(all(prior_plan and e["seq"] > prior_plan[0]["seq"] for e in events
              if e["kind"] == "generation_reserved" and e["experiment_id"] in owned), "Member work precedes paired plan")
    _event(campaign, "panel", built["panel"], early=True)
    _event(campaign, "mechanics_baseline", dict(sources=BASELINE_SOURCES, combined={u: sum(v[u] for v in BASELINE_SOURCES.values())
        for u in BASELINE_SOURCES["actor"]}, audit=audit, inventory_sha256=INVENTORY_SHA256, limitations=BASELINE_LIMITATIONS), early=True)
    registered = {e["experiment_id"]: e for e in events if e["kind"] == "experiment"}
    J.require(len(registered) == sum(e["kind"] == "experiment" for e in events), "Duplicate experiment record")
    contexts = {}
    for event in events:
        if event["kind"] == "execution_context": contexts.setdefault(event["experiment_id"], []).append(event)
    for record in built["records"]:
        existing = registered.get(record["experiment_id"])
        if existing: J.require(existing["record_sha256"] == digest(record), "Experiment record changed")
        else: campaign.register(record)
        context = dict(experiment_id=record["experiment_id"], contract=contract, contract_sha256=built["plan"]["contract_sha256"])
        prior = contexts.get(record["experiment_id"], [])
        J.require(len(prior) <= 1 and (not prior or J.encoded({k: prior[0][k] for k in context}) == J.encoded(context)), "Execution context differs")
        if not prior: campaign._append("execution_context", **context)
    count = len(built["records"]) * RECIPE["max_replies"]
    bound = {s: dict(zip(J.UNITS, (count, count * RECIPE["max_tokens"], count * RECIPE["wall_time_seconds"]))) for s in study.SOURCES}
    history = [e for e in campaign.events if not previous or e["seq"] < previous[0]["seq"]]
    limits, reserved = J.Campaign.totals(history)
    funding = dict(reserved_before=reserved, bound=bound,
                   limits={s: {u: max(limits[s][u], reserved[s][u] + bound[s][u]) for u in J.UNITS} for s in study.SOURCES})
    payload = dict(schema_version=1, plan_id=plan_id, plan_sha256=digest(built["plan"]),
                   producer_sources=producers, protocol=protocol, funding=funding)
    _event(campaign, "collection_setup", payload)
    current, _ = campaign._accounting()
    campaign.extend({s: {u: max(current[s][u], funding["limits"][s][u]) for u in J.UNITS} for s in study.SOURCES})
    return _event(campaign, "paired_plan", built["plan"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["setup"])
    for name in ("campaign", "plan-id", "receipt", "receipt-sha256", "mechanics-audit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    loaded, root = study.load_cohort(), Path(args.campaign)
    if root.exists(): campaign = J.Campaign(root)
    else:
        count = 144 * RECIPE["max_replies"]
        limits = {s: dict(zip(J.UNITS, (count, count * RECIPE["max_tokens"], count * RECIPE["wall_time_seconds"]))) for s in study.SOURCES}
        campaign = J.Campaign.create(root, limits)
    with campaign:
        result = setup(campaign, loaded, args.plan_id, args.receipt, args.receipt_sha256, args.mechanics_audit)
        print(J.encoded(result))


if __name__ == "__main__":
    main()
