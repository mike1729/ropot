from hashlib import sha256
import importlib
from pathlib import Path
import unittest
from unittest import mock
from tests import test_integrated_loop as fixture


class StudyReportTests(unittest.TestCase):
    write_task, rewrite = fixture.IntegratedLoopTests.write_task, fixture.IntegratedLoopTests.rewrite
    limits, rows = fixture.IntegratedLoopTests.limits, fixture.IntegratedLoopTests.rows
    read, execute, replies = fixture.IntegratedLoopTests.read, fixture.IntegratedLoopTests.execute, fixture.IntegratedLoopTests.replies
    recipe, record, fresh = fixture.IntegratedLoopTests.recipe, fixture.IntegratedLoopTests.record, fixture.IntegratedLoopTests.fresh

    def setUp(self):
        fixture.IntegratedLoopTests.setUp(self); self.R = importlib.import_module("src.data-pipeline.study_report")

    def run_recipe(self, campaign, name, seeds=(1,), sources=("actor",), assigned=(1,)):
        record = self.record(seeds, sources, assigned, name)
        self.L.run(campaign, self.loaded, record)
        return record

    def test_own_recipe_yield_union_costs_full_grids_and_exact_file_inventory(self):
        campaign = self.fresh()
        records = [self.run_recipe(campaign, "A"), self.run_recipe(campaign, "B", (4, 2), assigned=(2,)),
                   self.run_recipe(campaign, "C", (2,)), self.run_recipe(campaign, "D", (3,))]
        before = campaign.events; snapshot = self.R.snapshot(campaign); report = snapshot["report"]
        self.assertEqual(snapshot["events"], b"".join((self.N.encoded(e) + "\n").encode() for e in before))
        self.assertEqual(snapshot["manifest"]["journal"], {"through_seq": len(before), "sha256": sha256(snapshot["events"]).hexdigest()})
        self.assertEqual([r["cells"] for r in report["runs"]], [self.L.cell_states(before, record) for record in records])
        sources = [next(row for row in run["sources"] if row["source"] == "actor") for run in report["runs"]]
        self.assertEqual([(r["assigned"], r["verified"], r["initial_passes"], r["repair_passes"], r["reference_identical"]) for r in sources],
                         [(1, 1, 1, 0, 1), (1, 1, 0, 1, 0), (1, 1, 1, 0, 0), (1, 0, 0, 0, 0)])
        self.assertEqual(sources[0]["comparison_sha256"], sources[2]["comparison_sha256"])
        self.assertNotEqual(sources[0]["recipe_sha256"], sources[2]["recipe_sha256"])
        actor = [row for row in report["cumulative"] if row["source"] == "actor"]
        self.assertEqual((sum(r["activated"] for r in actor), sum(r["verified"] for r in actor)), (2, 2))
        synthetic = [row for row in report["cumulative"] if row["source"] == "synthetic"]
        self.assertEqual([sum(r[k] for r in synthetic) for k in ("activated", "assigned", "requested", "started", "verified")], [2, 2, 0, 0, 0])
        self.assertEqual(report["costs"]["reserved"]["actor"], dict(replies=5, reserved_output_tokens=5, reserved_generation_wall_seconds=2.5))
        self.assertEqual(report["costs"]["observed_by_source"]["actor"]["observed_generation_tokens"], 5)
        self.assertEqual(report["costs"]["native"]["operations"], 5)
        self.assertIsNone(report["panel_joint"]); self.assertIsNone(report["mechanics_baseline"])
        self.assertEqual({row["experiment_id"] for row in snapshot["manifest"]["selections"]}, {"A", "B"})
        self.assertEqual(snapshot["manifest"]["artifacts"], snapshot["files"])
        self.assertTrue(all(e["log"] in snapshot["files"] for e in before if e["kind"] == "native_verdict"))
        self.assertTrue(all(f"requests/{e['seq']:06d}/response.json" in snapshot["files"] for e in before if e["kind"] == "generation_reserved"))
        for origin, item in snapshot["files"].items():
            raw = (campaign.root / origin).read_bytes()
            self.assertEqual((item["sha256"], item["bytes"]), (sha256(raw).hexdigest(), len(raw)))
            self.assertTrue(item["path"].startswith("artifacts/"))
        self.assertEqual(campaign.events, before)

    def test_non_run_records_and_a_changed_captured_prefix_are_refused(self):
        campaign = self.fresh(); campaign.register({"experiment_id": "opaque"})
        with self.assertRaises(ValueError): self.R.snapshot(campaign)
        campaign.close(); campaign = self.fresh(); self.run_recipe(campaign, "run"); states = self.L.cell_states
        def race(events, record):
            campaign._append("fixture_change")
            return states(events, record)
        with mock.patch.object(self.L, "cell_states", side_effect=race), self.assertRaises(ValueError): self.R.snapshot(campaign)
        events = campaign.events; next(e for e in events if e["kind"] == "attempt_result")["candidate_sha256"] = None
        with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), self.assertRaisesRegex(ValueError, "Verified result lacks candidate"):
            self.R.snapshot(campaign)

    def test_first_pass_selection_override_preserves_other_source_and_charges(self):
        campaign = self.fresh(); self.run_recipe(campaign, "first", sources=("actor", "synthetic"))
        self.run_recipe(campaign, "later", (2,), ("actor", "synthetic"))
        reservations = [e for e in campaign.events if e["kind"] == "generation_reserved"]
        default = self.R.snapshot(campaign)
        self.assertEqual([r["reservation_seq"] for r in default["manifest"]["selections"]], [r["seq"] for r in reservations[:2]])
        override = self.R.snapshot(campaign, selections=[reservations[2]["seq"]])
        self.assertEqual({r["source"]: r["reservation_seq"] for r in override["manifest"]["selections"]},
                         {"actor": reservations[2]["seq"], "synthetic": reservations[1]["seq"]})
        self.assertEqual(default["report"]["costs"], override["report"]["costs"])
        for selected in ([True], [-1], [reservations[0]["seq"], reservations[2]["seq"]]):
            with self.assertRaises(ValueError): self.R.snapshot(campaign, selections=selected)

    def test_historical_policy_uses_bound_candidate_without_current_extraction(self):
        campaign = self.fresh(); record = self.run_recipe(campaign, "historical", (2,)); before = campaign.events
        with mock.patch.object(self.L, "policy_sha256", return_value="0" * 64), mock.patch.object(self.api, "extract_program", side_effect=AssertionError("current extraction")):
            snapshot = self.R.snapshot(campaign)
        run = snapshot["report"]["runs"][0]
        self.assertEqual(run["policy_sha256"], record["policy_sha256"])
        self.assertEqual(run["contract_sha256"], next(e["contract_sha256"] for e in before if e["kind"] == "execution_context"))
        self.assertEqual(len(snapshot["manifest"]["selections"]), 1); self.assertEqual(campaign.events, before)

    def test_failed_candidate_status_ownership_and_missing_bytes_must_match_proof(self):
        campaign = self.fresh(); self.run_recipe(campaign, "failed", (4,)); before = campaign.events
        self.assertEqual(self.R.snapshot(campaign)["manifest"]["selections"], [])
        for changes in ({"status": "verification_error"}, {"reused": True}, {"candidate_sha256": None}, {"operation_id": "0" * 12}):
            events = [dict(e) for e in before]; next(e for e in events if e["kind"] == "attempt_result").update(changes)
            with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), self.assertRaises(ValueError):
                self.R.snapshot(campaign)

    def test_unselected_passes_and_bound_candidate_bytes_are_proved(self):
        for damage in ("candidate", "log", "other_sample"):
            self.logs = {}; campaign = self.fresh(); self.run_recipe(campaign, "first"); self.run_recipe(campaign, "unselected", (2,))
            seq = next(e["seq"] for e in campaign.events if e["kind"] == "generation_reserved" and e["experiment_id"] == "unselected")
            if damage == "candidate": (campaign.root / "requests" / f"{seq:06d}" / "candidate.py").write_bytes(b"changed")
            if damage == "log": Path(self.log.location).write_bytes(b"changed native log")
            if damage == "other_sample": next(iter(self.logs.values())).samples[0].error = object()
            with self.assertRaises(ValueError): self.R.snapshot(campaign)
            campaign.close()

    def test_pending_generation_or_native_preserves_unknown_costs_and_dispatches_nothing(self):
        for failure in ("generation", "native", "timeout"):
            campaign = self.fresh(); record = self.record((13 if failure == "timeout" else 2,), ("actor",))
            target, name, error = (self.api, "generate", OSError("unknown worker")) if failure == "generation" else (self.N.verification, "verify", KeyboardInterrupt())
            if failure == "timeout": self.L.run(campaign, self.loaded, record)
            else:
                with mock.patch.object(target, name, side_effect=error), self.assertRaises(type(error)): self.L.run(campaign, self.loaded, record)
            root = campaign.root; campaign.close(); self.campaign = campaign = self.J.Campaign(root); self.addCleanup(campaign.close)
            before = campaign.events
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("generation")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("native")), mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("reconciliation")):
                snapshot = self.R.snapshot(campaign)
            report = snapshot["report"]
            self.assertEqual(report["runs"][0]["cells"][0]["status"], "blocked" if failure == "timeout" else "pending")
            self.assertEqual(report["costs"]["reserved"]["actor"]["replies"], 1)
            self.assertEqual(snapshot["manifest"]["selections"], [])
            if failure == "generation": self.assertEqual(report["costs"]["observed_by_source"]["actor"]["unknown_elapsed_requests"], 1)
            elif failure == "native":
                self.assertEqual(report["costs"]["native"]["unknown_elapsed_operations"], 1)
                opid = next(e["operation_id"] for e in before if e["kind"] == "native_intent" and e["reference_indices"] is None)
                self.assertTrue({f"native/{opid}/task.json", f"native/{opid}/candidate.py"} <= snapshot["files"].keys())
            else: self.assertEqual(report["costs"]["observed_by_source"]["actor"]["lower_bound_requests"], 1)
            self.assertEqual(campaign.events, before); campaign.close()

    def test_fixed_joint_panel_uses_intersection_and_ignores_outside_success(self):
        panel = [{"split": "test", "id": i, "sha256": f"{i:064x}"} for i in range(50)]
        def selected(actor, synthetic):
            return [{"split": "test", "id": i, "task_sha256": f"{i:064x}", "source": source}
                    for source, ids in (("actor", actor), ("synthetic", synthetic)) for i in ids]
        for synthetic, expected in ((range(10, 30), 10), (range(5, 25), 15)):
            result = self.R.panel_joint(panel, selected([*range(20), 999], [*synthetic, 999]))
            self.assertEqual((result["numerator"], result["denominator"], result["target_met"]), (expected, 50, expected >= 15))
        for invalid in ("valid", "digest", "cohort", "order"):
            campaign = self.fresh(); tasks = [{**self.records[1], **task} for task in panel]
            payload = dict(activation=dict(cohort_sha256=self.loaded["sha256"], groups=["development_smoke"], tasks=tasks), tasks=list(tasks))
            if invalid == "cohort": payload["activation"]["cohort_sha256"] = "0" * 64
            if invalid == "order": payload["tasks"].reverse()
            campaign._append("panel", payload=payload, payload_sha256="0" * 64 if invalid == "digest" else sha256(self.N.encoded(payload).encode()).hexdigest())
            if invalid == "valid":
                report = self.R.snapshot(campaign)["report"]
                self.assertEqual(report["panel_joint"], dict(numerator=0, denominator=50, target_met=False))
                self.assertEqual(sum(r["activated"] for r in report["cumulative"]), 100)
            else:
                with self.assertRaises(ValueError): self.R.snapshot(campaign)
            campaign.close()

    def test_baseline_before_work_is_bound_separate_and_inconsistent_totals_refused(self):
        for invalid in ("valid", "totals", "late", "audit"):
            campaign = self.fresh(); audit = b"Fixture mechanics audit"; (campaign.root / "audit.txt").write_bytes(audit)
            sources = {"actor": dict(replies=7, observed_generation_tokens=265, reserved_output_tokens=4417),
                       "synthetic": dict(replies=8, observed_generation_tokens=166, reserved_output_tokens=2048)}
            combined = {key: sum(row[key] for row in sources.values()) for key in sources["actor"]}
            payload = dict(sources=sources, combined={**combined, "replies": 16} if invalid == "totals" else combined,
                audit=dict(path="audit.txt", sha256=sha256(audit).hexdigest(), bytes=len(audit)), inventory_sha256="f" * 64)
            if invalid == "late": self.run_recipe(campaign, "run")
            campaign._append("mechanics_baseline", payload=payload, payload_sha256=sha256(self.N.encoded(payload).encode()).hexdigest())
            if invalid != "late": self.run_recipe(campaign, "run")
            if invalid == "audit": (campaign.root / "audit.txt").write_bytes(b"changed audit")
            if invalid != "valid":
                with self.assertRaises(ValueError): self.R.snapshot(campaign)
            else:
                result = self.R.snapshot(campaign)
                self.assertEqual(result["report"]["mechanics_baseline"]["combined"], combined)
                self.assertEqual(result["report"]["combined_known"], dict(replies=16, observed_generation_tokens=432, reserved_output_tokens=6466))
                self.assertIn("audit.txt", result["files"])
            campaign.close()
