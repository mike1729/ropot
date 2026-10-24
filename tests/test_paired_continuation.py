import copy
import importlib
import unittest
from unittest import mock
from tests import test_paired_report as reports, test_paired_collection as collection


class PairedContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f = reports.PairedReportTests(); self.addCleanup(self.f.doCleanups); self.f.setUp()
        self.B = importlib.import_module("src.data-pipeline.paired_continuation")
        self.C, self.J, self.campaign, self.built = self.f.C, self.f.J, self.f.campaign, self.f.built
        self.plan = self.campaign._append("paired_plan", payload=self.built["plan"], payload_sha256=self.C.digest(self.built["plan"]))
        setup = dict(payload=dict(plan_id="trial", plan_sha256=self.plan["payload_sha256"]))
        self.trial = dict(plan_id="trial", plan_sha256=self.plan["payload_sha256"], comparison_valid=True,
            complete=False, fresh_confirmation_valid=False, prior_exposed_tasks=[], confirmation_seq=None,
            metrics=[dict(name="tuning40", denominator=40, terminal_cells=160, complete=True,
                arms=[dict(bits=b, final_verified=dict(joint=0)) for b in (8, 16)])])
        # C/S proof boundaries are already tested; these oracles isolate wrapper orchestration.
        for target, name, kwargs in ((self.C, "_retained", dict(return_value=(self.built, setup))),
                (self.B.S, "snapshot", dict(return_value=dict(report=dict(paired_trials=[self.trial])))),
                (self.C, "_producer_guard", dict(return_value=None)), (self.C.L.native, "reconcile_native", dict(return_value=[])),
                (self.C.L, "run", dict(side_effect=AssertionError("unmocked dispatch"))),
                (self.f.api, "generate", dict(side_effect=AssertionError("model dispatch"))),
                (self.C.L.native, "run_native", dict(side_effect=AssertionError("new native operation")))):
            patch = mock.patch.object(target, name, **kwargs); patch.start(); self.addCleanup(patch.stop)

    def decision(self):
        event = self.C._event(self.campaign, "paired_confirmation", dict(plan_id="trial", plan_sha256=self.plan["payload_sha256"], decision="unchanged"))
        self.trial["confirmation_seq"] = event["seq"]; return event

    def confirm(self): return self.B.confirm_unchanged(self.campaign, self.f.loaded, "trial")

    def fresh(self, **kwargs): return self.B.continue_fresh(self.campaign, self.f.loaded, "trial", **kwargs)

    def terminal(self, record, **kwargs): return collection.PairedCollectionTests.terminal(self.f, record, **kwargs)

    def test_confirmation_is_explicit_below_threshold_and_idempotent_without_dispatch(self):
        before = self.campaign.events
        with self.assertRaisesRegex(ValueError, "Require explicit unchanged confirmation"): self.fresh(through_tasks=1)
        self.assertEqual(self.campaign.events, before)
        event = self.confirm(); self.assertEqual(event["payload"], dict(plan_id="trial", plan_sha256=self.plan["payload_sha256"], decision="unchanged"))
        self.C._producer_guard.assert_called_once(); self.C.L.run.assert_not_called()
        self.trial["confirmation_seq"] = event["seq"]; before = self.campaign.events
        with mock.patch.object(self.campaign, "_append", side_effect=AssertionError("retry appended")):
            self.assertEqual(self.confirm(), event)
        self.assertEqual(self.campaign.events, before); self.assertFalse(self.trial["complete"])
        self.assertEqual(self.B.S.snapshot.call_count, 3)

    def test_report_validity_unique_subset_and_freshness_are_required_before_writes(self):
        original = copy.deepcopy(self.trial); before = self.campaign.events
        for fault in ("invalid", "exposed", "absent", "duplicate", "wrong_name", "incomplete", "denominator", "terminal", "foreign", "hash"):
            self.trial.clear(); self.trial.update(copy.deepcopy(original))
            if fault == "invalid": self.trial["comparison_valid"] = False
            elif fault == "exposed": self.trial["prior_exposed_tasks"] = [self.built["plan"]["records"][80]["task"]]
            elif fault == "absent": self.trial["metrics"] = []
            elif fault == "duplicate": self.trial["metrics"] *= 2
            elif fault == "foreign": self.trial["plan_id"] = "foreign"
            elif fault == "hash": self.trial["plan_sha256"] = "0" * 64
            else: self.trial["metrics"][0][dict(wrong_name="name", incomplete="complete", denominator="denominator", terminal="terminal_cells")[fault]] = dict(wrong_name="other", incomplete=False, denominator=39, terminal=159)[fault]
            error = "Invalid or previously exposed fresh trial" if fault in ("invalid", "exposed") else "Require unique retained paired trial" if fault in ("foreign", "hash") else "Require complete usable tuning evidence"
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, error): self.confirm()
            self.assertEqual(self.campaign.events, before)
        self.C.L.run.assert_not_called(); self.C._producer_guard.assert_not_called()

    def test_retained_decision_shape_hash_duplicates_and_sequence_are_bound(self):
        base = self.campaign.events; payload = dict(plan_id="trial", plan_sha256=self.plan["payload_sha256"], decision="unchanged")
        for fault in ("duplicate", "foreign", "hash", "decision", "extra", "sequence", "missing"):
            changed = copy.deepcopy(payload)
            if fault in ("foreign", "hash", "decision", "extra"): changed.update({dict(foreign="plan_id", hash="plan_sha256", decision="decision", extra="extra")[fault]: dict(foreign="foreign", hash="0" * 64, decision="changed", extra=True)[fault]})
            event = dict(seq=len(base) + 1, kind="paired_confirmation", payload=changed, payload_sha256=self.C.digest(changed))
            events = base + ([event, dict(event, seq=event["seq"] + 1)] if fault == "duplicate" else [] if fault == "missing" else [event])
            self.trial["confirmation_seq"] = event["seq"] + (1 if fault == "sequence" else 0)
            with mock.patch.object(type(self.campaign), "events", new_callable=mock.PropertyMock, return_value=events), self.assertRaisesRegex(ValueError, "Duplicate paired confirmation|Paired confirmation differs"): self.confirm()

    def test_finite_prefix_forwards_registered_records_in_frozen_order_and_replays(self):
        self.decision(); before = self.campaign.events
        with mock.patch.object(self.C.L, "run", side_effect=lambda c, l, r: self.terminal(r)) as run, mock.patch.object(self.campaign, "extend", side_effect=AssertionError("fresh cap reset")):
            first = self.fresh(through_tasks=1); full = self.fresh()
        expected = self.built["records"][80:144]
        self.assertEqual([r["experiment_id"] for r in full], [r["experiment_id"] for r in expected]); self.assertEqual(len(first), 2)
        for call, record in zip(run.call_args_list, expected[:2] + expected): self.assertIs(call.args[2], record)
        self.assertEqual(self.C._producer_guard.call_count, 66); self.assertEqual(self.campaign.events, before)
        self.assertEqual(self.C._retained.call_count, 2); self.assertEqual(self.B.S.snapshot.call_count, 2)
        tasks = self.built["activation"]["tasks"]
        self.assertEqual([r["assigned_task_ids"][0] for r in expected[::2]], [t["id"] for t in tasks[24:40] + tasks[56:72]])

    def test_guard_before_each_call_and_native_or_retained_failures_propagate(self):
        self.decision()
        with mock.patch.object(self.C._producer_guard, "side_effect", [None, ValueError("Producer sources changed")]), mock.patch.object(self.C.L, "run", side_effect=lambda c, l, r: self.terminal(r)) as run:
            with self.assertRaisesRegex(ValueError, "Producer sources changed"): self.fresh(through_tasks=1)
            self.assertEqual(run.call_count, 1)
        for error in (ValueError("Collection superseded"), self.C.L.native.NativeBlocked("native pending")):
            target, name = (self.C, "_retained") if isinstance(error, ValueError) else (self.C.L.native, "reconcile_native")
            with mock.patch.object(target, name, side_effect=error), mock.patch.object(self.C.L, "run") as run, self.assertRaises(type(error)) as caught: self.fresh(through_tasks=1)
            self.assertIs(caught.exception, error); run.assert_not_called()
        self.campaign._append("native_intent", operation_id="a" * 12)
        with self.assertRaisesRegex(ValueError, "Reconcile pending native operations"): self.fresh(through_tasks=1)

    def test_only_two_assigned_terminal_source_cells_are_accepted(self):
        self.decision()
        for status, reason in (("pending", None), ("blocked", "native_infrastructure"), ("blocked", "campaign_budget")):
            with mock.patch.object(self.C.L, "run", side_effect=lambda c, l, r: self.terminal(r, status=status, reason=reason)) as run, self.assertRaisesRegex(ValueError, "Collection member not terminal"): self.fresh(through_tasks=1)
            self.assertEqual(run.call_count, 1)
        def extra(c, l, record):
            cells = self.terminal(record); return cells + [dict(cells[0], id=-1, status="pending")]
        with mock.patch.object(self.C.L, "run", side_effect=extra): rows = self.fresh(through_tasks=1)
        self.assertTrue(all(len(row["cells"]) == 2 for row in rows))

    def test_whole_family_control_and_foreign_resource_barriers_precede_dispatch(self):
        self.decision(); record = self.built["records"][-1]; self.campaign.register(record)
        collection.PairedCollectionTests.charge(self.f, self.campaign, record)
        before = self.campaign.events; costs = self.campaign._accounting()
        for _ in range(2):
            with self.assertRaises(self.J.ProfileControlError) as caught: self.fresh(through_tasks=1)
            self.assertEqual(caught.exception.reason, "profile_unproven")
            self.campaign = reports.PairedReportTests.F.reopen(self.f, self.campaign)
        self.assertEqual(self.campaign.events, before); self.assertEqual(self.campaign._accounting(), costs)
        self.B.S.snapshot.assert_not_called(); self.C.L.run.assert_not_called()
        events = copy.deepcopy(before); next(e for e in events if e["kind"] == "generation_reserved")["experiment_id"] = "foreign"
        outcome = dict(reservation_seq=next(e["seq"] for e in events if e["kind"] == "generation_reserved"), response=dict(cleanup_error="unknown"))
        with mock.patch.object(type(self.campaign), "events", new_callable=mock.PropertyMock, return_value=events), mock.patch.object(self.campaign, "outcome", return_value=outcome), self.assertRaisesRegex(ValueError, "Unresolved generation cleanup failure"): self.fresh(through_tasks=1)

    def test_new_guarded_error_pauses_retained_error_and_timeout_can_continue(self):
        self.decision()
        for reason in ("generation_error", "interrupted", "generation_timeout"):
            # An observation marker isolates the MAXseq boundary; it is not fabricated control proof.
            def observed(campaign, loaded, record):
                seq = campaign._append("fixture_guarded_observation")["seq"]; cells = self.terminal(record, status="blocked", reason=reason)
                cells[0]["attempt_seqs"] = [seq]; return cells
            with mock.patch.object(self.C.L, "run", side_effect=observed) as run:
                if reason == "generation_timeout": self.assertEqual(len(self.fresh(through_tasks=1)), 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, "New guarded execution failure in"): self.fresh(through_tasks=1)
                    self.assertEqual(run.call_count, 1)
            old = self.campaign.events[-1]["seq"]
            def retained(c, l, record):
                cells = self.terminal(record, status="blocked", reason=reason); cells[0]["attempt_seqs"] = [old]; return cells
            with mock.patch.object(self.C.L, "run", side_effect=retained): self.assertEqual(len(self.fresh(through_tasks=1)), 2)
        for error in (KeyboardInterrupt(), OSError("dispatch failed"), self.C.L.native.NativeBlocked("infrastructure")):
            with mock.patch.object(self.C.L, "run", side_effect=error), self.assertRaises(type(error)) as caught: self.fresh(through_tasks=1)
            self.assertIs(caught.exception, error)

    def test_types_cli_are_existing_only_and_have_no_setup_or_implicit_confirmation(self):
        for value in (True, 1.0, "1", 0, 33):
            with mock.patch.object(self.C, "_retained", side_effect=AssertionError("early retained read")), self.assertRaisesRegex(ValueError, "Invalid fresh prefix"): self.fresh(through_tasks=value)
        missing = self.f.root / "absent-campaign"; args = ["fresh", "--campaign", str(missing), "--plan-id", "trial", "--through-tasks", "1"]
        with mock.patch.object(self.J.Campaign, "create", side_effect=AssertionError("campaign create")), self.assertRaises(FileNotFoundError): self.B.main(args)
        self.assertFalse(missing.exists())
        for argv in (args[:-2], args[:-1] + ["0"], args[:-1] + ["33"], args + ["--receipt", "forbidden"], ["setup"], ["tune"], ["confirm"] + args[1:]):
            with mock.patch("sys.stderr"), mock.patch.object(self.C.study, "load_cohort", side_effect=AssertionError("invalid args accepted")), self.assertRaises(SystemExit): self.B.main(argv)
