"""Explicit unchanged confirmation and finite fresh continuation of a paired trial."""

import argparse

from . import paired_collection as C, study_report as S


def _entry(campaign, loaded, plan_id):
    built, setup = C._retained(campaign, loaded, plan_id)
    names = {r["experiment_id"] for r in built["records"]}
    for event in campaign.events:
        if event["kind"] != "generation_reserved": continue
        outcome = campaign.outcome(event["seq"])
        C.J.require(C.J.generation_resources_cleared(campaign.events, outcome), "Unresolved generation cleanup failure")
        if event["experiment_id"] in names: C.J.require_profile_outcome(outcome)
    C.L.native.reconcile_native(campaign)
    campaign.require_native_clear()
    trials = [t for t in S.snapshot(campaign)["report"]["paired_trials"]
              if t["plan_id"] == plan_id and t["plan_sha256"] == setup["payload"]["plan_sha256"]]
    C.J.require(len(trials) == 1, "Require unique retained paired trial")
    trial = trials[0]
    C.J.require(trial["comparison_valid"] and not trial["prior_exposed_tasks"], "Invalid or previously exposed fresh trial")
    tuning = [m for m in trial["metrics"] if m["name"] == "tuning40"]
    C.J.require(len(tuning) == 1 and tuning[0]["denominator"] == 40
                and tuning[0]["terminal_cells"] == 160 and tuning[0]["complete"], "Require complete usable tuning evidence")
    decisions = [e for e in campaign.events if e["kind"] == "paired_confirmation" and e["payload"]["plan_id"] == plan_id]
    C.J.require(len(decisions) <= 1, "Duplicate paired confirmation")
    decision = C._one(campaign.events, "paired_confirmation", plan_id) if decisions else None
    expected = dict(plan_id=plan_id, plan_sha256=setup["payload"]["plan_sha256"], decision="unchanged")
    C.J.require((decision is None and trial["confirmation_seq"] is None)
                or (decision is not None and decision["payload"] == expected
                    and decision["seq"] == trial["confirmation_seq"]), "Paired confirmation differs")
    return built, setup, trial, decision


def confirm_unchanged(campaign, loaded, plan_id):
    """Record an explicit unchanged decision after root's pooled tuning review."""
    _, setup, _, decision = _entry(campaign, loaded, plan_id)
    if decision is not None: return decision
    C._producer_guard(campaign, setup["payload"])
    return C._event(campaign, "paired_confirmation",
                    dict(plan_id=plan_id, plan_sha256=setup["payload"]["plan_sha256"], decision="unchanged"))


def continue_fresh(campaign, loaded, plan_id, *, through_tasks=32):
    C.J.require(type(through_tasks) is int and 1 <= through_tasks <= 32, "Invalid fresh prefix")
    built, setup, _, decision = _entry(campaign, loaded, plan_id)
    C.J.require(decision is not None, "Require explicit unchanged confirmation")
    results = []
    for record in built["records"][80:80 + 2 * through_tasks]:
        C._producer_guard(campaign, setup["payload"])
        before = campaign.db.execute("SELECT MAX(seq) FROM events").fetchone()[0]
        cells = C._assigned(record, C.L.run(campaign, loaded, record))
        if any(c["reason"] in ("generation_error", "interrupted") and any(seq > before for seq in c["attempt_seqs"]) for c in cells):
            raise RuntimeError(f"New guarded execution failure in {record['experiment_id']}; inspect before resuming")
        results.append(dict(experiment_id=record["experiment_id"], cells=cells))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("confirm", "fresh"):
        command = commands.add_parser(mode)
        for name in ("campaign", "plan-id"): command.add_argument("--" + name, required=True)
        if mode == "fresh": command.add_argument("--through-tasks", type=int, choices=range(1, 33), required=True)
    args = parser.parse_args(argv)
    loaded = C.study.load_cohort()
    with C.J.Campaign(args.campaign) as campaign:
        result = (confirm_unchanged(campaign, loaded, args.plan_id) if args.mode == "confirm"
                  else continue_fresh(campaign, loaded, args.plan_id, through_tasks=args.through_tasks))
        print(C.J.encoded(result))


if __name__ == "__main__":
    main()
