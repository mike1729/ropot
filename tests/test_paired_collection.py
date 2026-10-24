import copy
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import unittest
from unittest import mock
from tests import test_precision_adoption as adoption, test_study_tasks as study


class PairedCollectionTests(unittest.TestCase):
    F = adoption.PrecisionAdoptionTests
    write_task, rewrite, limits, rows = F.write_task, F.rewrite, F.limits, F.rows
    read, execute, replies, recipe, fresh = F.read, F.execute, F.replies, F.recipe, F.fresh
    fixture, put, mode = F.fixture, F.put, F.mode
    feasibility_loaded = study.StudyTaskTests.feasibility_loaded

    def setUp(self):
        self.F.setUp(self); self.C = importlib.import_module("src.data-pipeline.paired_collection")
        self.loaded = self.feasibility_loaded()
        self.loaded["cohort"]["tasks"][0] = copy.deepcopy(self.records[1])
        self.loaded["cohort"]["queues"]["development"]["initial"][0] = 1
        self.audit = self.root / "mechanics.txt"; self.audit.write_bytes(b"Fixture mechanics audit")
        self.protocol = self.root / "protocol.md"; self.protocol.write_bytes(self.C.PROTOCOL.read_bytes())
        for target, name, value in ((self.C, "AUDIT_SHA256", self.P.sha256(self.audit.read_bytes()).hexdigest()), (self.C, "PROTOCOL", self.protocol)):
            patch = mock.patch.object(target, name, value); patch.start(); self.addCleanup(patch.stop)
        for target, name in ((self.L, "run"), (self.L.study, "load_task"), (self.J.Campaign, "generate"), (self.api, "generate"), (self.N, "run_native")):
            patch = mock.patch.object(target, name, side_effect=AssertionError("Setup dispatched " + name)); patch.start(); self.addCleanup(patch.stop)

    def setup(self, campaign, plan="trial"):
        return self.C.setup(campaign, self.loaded, plan, self.original / "receipt.json", self.identity, self.audit)

    def charge(self, campaign, record, complete=True):
        recipe = record["recipes"]["actor"]; task = next(t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"])
        request = self.api.request_for("actor", [{"role": "user", "content": "fixture"}], profile=record["profiles"]["actor"]["descriptor"],
            seed=recipe["seeds"][0], **{k: recipe[k] for k in ("max_tokens", "wall_time_seconds", "temperature", "top_p")})
        _, raw = self.api.prepare_request(request)
        event = campaign._append("generation_reserved", experiment_id=record["experiment_id"], source="actor", request_text=raw.decode(), request_sha256=self.P.sha256(raw).hexdigest(),
            attempt=dict(split=task["split"], id=task["id"], task_sha256=task["sha256"], reply_index=0))
        if complete: self.assertEqual(campaign.outcome(event["seq"])["status"], "interrupted")

    def test_canonical_setup_matrix_copies_and_event_order(self):
        campaign = self.fresh(); append = campaign._append
        def inspect(kind, **payload):
            if kind in ("panel", "mechanics_baseline"):
                self.assertEqual({p.read_bytes() for p in (campaign.root / "audits").iterdir() if p.is_file()}, {self.audit.read_bytes(), self.protocol.read_bytes()})
            return append(kind, **payload)
        with mock.patch.object(campaign, "_append", side_effect=inspect): event = self.setup(campaign)
        events = campaign.events; records = [e for e in events if e["kind"] == "experiment"]
        contexts = [e for e in events if e["kind"] == "execution_context"]
        setup = next(e for e in events if e["kind"] == "collection_setup"); payload = event["payload"]
        self.assertEqual((len(records), len(contexts), len(payload["records"])), (144, 144, 144))
        tasks = records[0]["record"]["activation"]["tasks"]; self.assertEqual(len(tasks), 72)
        tuning = tasks[:8] + tasks[8:24] + tasks[40:56]; fresh = tasks[24:40] + tasks[56:72]
        expected = [(task["id"], bits) for task in tuning + fresh for bits in ((8, 16) if tasks.index(task) % 2 == 0 else (16, 8))]
        self.assertEqual([(r["task"]["id"], r["bits"]) for r in payload["records"]], expected)
        self.assertEqual(expected[0], (1, 8)); self.assertEqual(len({r["experiment_id"] for r in payload["records"]}), 144)
        panel = next(e["payload"] for e in events if e["kind"] == "panel")
        self.assertEqual(panel["tasks"], tasks[:8] + tasks[8:29] + tasks[40:61])
        baseline = next(e["payload"] for e in events if e["kind"] == "mechanics_baseline")
        self.assertEqual(baseline["sources"], dict(actor=dict(replies=7, observed_generation_tokens=265, reserved_output_tokens=4417),
            synthetic=dict(replies=8, observed_generation_tokens=166, reserved_output_tokens=2048)))
        self.assertTrue(baseline["limitations"])
        for record, member, context in zip(records, payload["records"], contexts):
            body = record["record"]; self.assertEqual(body["assigned_task_ids"], [member["task"]["id"]])
            self.assertEqual(set(body["recipes"]), {"actor", "synthetic"}); self.assertEqual({p["descriptor"]["bits"] for p in body["profiles"].values()}, {member["bits"]})
            self.assertEqual(member["record_sha256"], self.P.sha256(self.J.encoded(body).encode()).hexdigest())
            self.assertEqual(context["contract_sha256"], self.N.verification.digest(context["contract"]))
            self.assertLess(max(record["seq"], context["seq"]), setup["seq"])
        self.assertEqual(setup["payload"]["plan_sha256"], event["payload_sha256"])
        self.assertLess(setup["seq"], next(e["seq"] for e in events if e["kind"] == "limits")); self.assertLess(setup["seq"], event["seq"])
        before = campaign.events; copies = {p: p.read_bytes() for p in (campaign.root / "audits").iterdir()}
        self.assertEqual(self.setup(campaign), event); self.assertEqual(campaign.events, before)
        self.assertEqual({p: p.read_bytes() for p in copies}, copies)

    def test_funding_is_frozen_and_new_sequential_plan_counts_actual_charges(self):
        campaign = self.fresh(); first = self.setup(campaign)
        record = next(e["record"] for e in campaign.events if e["kind"] == "experiment")
        self.charge(campaign, record); before = campaign.events; limits, spent = campaign._accounting()
        self.assertEqual(self.setup(campaign), first); self.assertEqual(campaign.events, before)
        second = self.setup(campaign, "second")
        funding = [e["payload"]["funding"] for e in campaign.events if e["kind"] == "collection_setup"][-1]
        bound = dict(replies=432, reserved_output_tokens=884736, reserved_generation_wall_seconds=103680)
        self.assertEqual(funding["reserved_before"], spent); self.assertEqual(funding["bound"], {s: bound for s in ("actor", "synthetic")})
        self.assertEqual(funding["limits"], {s: {u: max(limits[s][u], spent[s][u] + bound[u]) for u in bound} for s in spent})
        self.assertEqual(campaign._accounting(), (funding["limits"], spent)); self.assertEqual(second["payload"]["plan_id"], "second")
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.setup(campaign)
        large = self.fresh(replies=999, reserved_output_tokens=2_000_000, reserved_generation_wall_seconds=200_000)
        original = large._accounting()[0]; self.setup(large); self.assertEqual(large._accounting()[0], original)

    def test_invalid_ids_and_build_shape_or_receipt_mismatch(self):
        campaign = self.fresh(); self.setup(campaign)
        record = next(e["record"] for e in campaign.events if e["kind"] == "experiment")
        profiles = {b: next(e["record"]["profiles"] for e in campaign.events if e["kind"] == "experiment" and next(iter(e["record"]["profiles"].values()))["descriptor"]["bits"] == b) for b in (8, 16)}
        contract = next(e["contract_sha256"] for e in campaign.events if e["kind"] == "execution_context")
        for value in ("", " ", "Upper", "bad/id", "x" * 65, True):
            with self.assertRaisesRegex(ValueError, "Invalid plan ID"): self.C.build(self.loaded, value, profiles, contract)
        changed = copy.deepcopy(self.loaded); changed["cohort"]["candidate_budgets"]["development"]["smoke"]["total"] = 7
        with self.assertRaisesRegex(ValueError, "Require canonical seventy-two tasks"): self.C.build(changed, "valid", profiles, contract)
        other, identity, _ = self.fixture(); reference = self.P.freeze(campaign, other / "receipt.json", identity)
        profiles[16] = {s: dict(reference=reference, descriptor=self.P.resolve(campaign, reference, s, 16)) for s in ("actor", "synthetic")}
        with self.assertRaisesRegex(ValueError, "Profile authorities differ"): self.C.build(self.loaded, "valid", profiles, contract)
        self.assertEqual(record["recipes"]["actor"], record["recipes"]["synthetic"])

    def test_setup_crashes_resume_every_durable_boundary_without_re_funding(self):
        reference = self.fresh(); self.setup(reference); expected = reference.events
        for target, action in (("append", "panel"), ("append", "mechanics_baseline"), ("append", "experiment"),
                ("append", "execution_context"), ("append", "collection_setup"), ("append", "paired_plan"), ("extend", None), ("copy", None)):
            for after in (False, True):
                with self.subTest(target=target, action=action, after=after):
                    campaign = self.fresh(); original = campaign._append if target == "append" else campaign.extend if target == "extend" else self.C._copy
                    def crash(*args, **kwargs):
                        applies = target != "append" or args[0] == action
                        if applies and not after: raise OSError("before durable action")
                        result = original(*args, **kwargs)
                        if applies: raise OSError("after durable action")
                        return result
                    obj, name = (campaign, "_append") if target == "append" else (campaign, "extend") if target == "extend" else (self.C, "_copy")
                    with mock.patch.object(obj, name, side_effect=crash), self.assertRaises(OSError): self.setup(campaign)
                    campaign.close(); campaign = self.J.Campaign(campaign.root); self.addCleanup(campaign.close)
                    event = self.setup(campaign); before = campaign.events
                    self.assertEqual(before, expected)
                    self.assertEqual([sum(e["kind"] == k for e in before) for k in ("experiment", "execution_context", "collection_setup", "limits", "paired_plan")], [144, 144, 1, 1, 1])
                    self.assertEqual(next(e["payload"]["funding"]["reserved_before"] for e in before if e["kind"] == "collection_setup"), {s: dict.fromkeys(self.J.UNITS, 0) for s in ("actor", "synthetic")})

    def test_new_unanchored_setup_supersedes_older_and_preserves_funding(self):
        campaign = self.fresh(); self.setup(campaign)
        with mock.patch.object(campaign, "extend", side_effect=OSError("before extend")), self.assertRaises(OSError): self.setup(campaign, "second")
        self.assertFalse(any(e["kind"] == "paired_plan" and e["payload"]["plan_id"] == "second" for e in campaign.events))
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.setup(campaign)
        frozen = [e["payload"]["funding"] for e in campaign.events if e["kind"] == "collection_setup"][-1]
        self.setup(campaign, "second"); self.assertEqual(campaign._accounting()[0], frozen["limits"])

    def test_copy_and_producer_tampering_refuse_equal_retry(self):
        for damage in ("audit", "protocol", "directory_symlink", "file_symlink", "producer", "protocol_source", "contract", "receipt"):
            campaign = self.fresh(); self.setup(campaign); events = campaign.events
            baseline = next(e["payload"] for e in events if e["kind"] == "mechanics_baseline")
            setup = next(e["payload"] for e in events if e["kind"] == "collection_setup")
            path = campaign.root / (baseline["audit"]["path"] if damage == "audit" else setup["protocol"]["path"])
            saved = self.protocol.read_bytes()
            if damage in ("audit", "protocol"): path.write_bytes(b"changed")
            if damage == "directory_symlink":
                directory = campaign.root / "audits"; twin = campaign.root / "twin"; directory.rename(twin); directory.symlink_to(twin, target_is_directory=True)
            if damage == "file_symlink":
                twin = campaign.root / "twin"; twin.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(twin)
            if damage == "protocol_source": self.protocol.write_bytes(saved + b"changed")
            fingerprint = self.C.fingerprint(); fingerprint[next(iter(fingerprint))] = "0" * 64
            patch = mock.patch.object(self.C, "fingerprint", return_value=fingerprint) if damage == "producer" else mock.patch.object(self.C, "AUDIT_SHA256", self.C.AUDIT_SHA256)
            if damage == "contract": patch = mock.patch.object(self.C.V.runner, "contract", return_value={**self.C.V.runner.contract(), "fixture_drift": True})
            if damage == "receipt": other, identity, _ = self.fixture()
            reasons = dict(producer="Producer sources changed", protocol_source="Protocol source changed", directory_symlink="Audit directory differs",
                contract="Execution context differs", receipt="Experiment record changed")
            with patch, self.assertRaises(OSError) if damage == "file_symlink" else self.assertRaisesRegex(ValueError, reasons.get(damage, "Published copy differs")):
                if damage == "receipt": self.C.setup(campaign, self.loaded, "trial", other / "receipt.json", identity, self.audit)
                else: self.setup(campaign)
            self.protocol.write_bytes(saved); self.assertEqual(campaign.events, events)

    def test_unresolved_work_late_anchors_and_missing_initial_are_not_repaired(self):
        campaign = self.fresh(); campaign._append("native_intent", operation_id="a" * 12)
        before = campaign.events
        with self.assertRaisesRegex(ValueError, "Reconcile pending native operations"): self.setup(campaign)
        self.assertEqual(campaign.events, before)
        reference = self.fresh(); self.setup(reference, "other"); record = next(e["record"] for e in reference.events if e["kind"] == "experiment")
        campaign = self.fresh(); campaign.register(record); self.charge(campaign, record, complete=False)
        with self.assertRaisesRegex(ValueError, "Pending generations"): self.setup(campaign)
        self.assertFalse((campaign.root / "audits").exists()); campaign.outcome(campaign.events[-1]["seq"])
        before = campaign.events
        with self.assertRaisesRegex(ValueError, "Late global anchor"): self.setup(campaign)
        self.assertEqual(campaign.events, before)
        for kind in ("panel", "mechanics_baseline"):
            old = next(e for e in reference.events if e["kind"] == kind); campaign._append(kind, payload=old["payload"], payload_sha256=old["payload_sha256"])
        with self.assertRaisesRegex(ValueError, "Late global anchor"): self.setup(campaign)
        campaign = self.fresh(); root = campaign.root; campaign.close(); (root / "journal.sqlite3").unlink()
        with mock.patch.object(self.C.study, "load_cohort", return_value=self.loaded), mock.patch.object(self.J.Campaign, "create", side_effect=AssertionError("unexpected create")), self.assertRaises(sqlite3.DatabaseError):
            self.C.main(["setup", "--campaign", str(root), "--plan-id", "trial", "--receipt", str(self.original / "receipt.json"),
                "--receipt-sha256", self.identity, "--mechanics-audit", str(self.audit)])
        with mock.patch.object(self.J.Campaign, "create", side_effect=AssertionError("unexpected create")), self.assertRaises(SystemExit): self.C.main(["fresh"])

    def test_member_work_before_paired_anchor_is_refused(self):
        campaign = self.fresh(); append = campaign._append
        def stop(kind, **payload):
            if kind == "paired_plan": raise OSError("before paired anchor")
            return append(kind, **payload)
        with mock.patch.object(campaign, "_append", side_effect=stop), self.assertRaises(OSError): self.setup(campaign)
        self.charge(campaign, next(e["record"] for e in campaign.events if e["kind"] == "experiment"))
        with self.assertRaisesRegex(ValueError, "Member work precedes paired plan"): self.setup(campaign)

    def test_fresh_import_producer_tuple_covers_repository_closure_without_report(self):
        script = """
import importlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root));c=importlib.import_module('src.data-pipeline.paired_collection')
files={str(pathlib.Path(m.__file__).resolve().relative_to(root)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(root)}
files={n for n in files if n.split('/',1)[0] in {'src','experiments'}}
assert not any('study_report' in n or 'paired_report' in n for n in sys.modules)
print(json.dumps(sorted(files-set(c.PRODUCERS))))
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(self.C.ROOT)], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])
        self.assertTrue({"src/data-pipeline/mlx_worker.py", "experiments/prepare_precision_artifacts.py", "experiments/bf16_compatibility.py"} <= set(self.C.PRODUCERS))
        self.assertTrue(any(p.endswith("compose.yaml") for p in self.C.PRODUCERS))

    def terminal(self, record, status="verified", reason=None):
        task = next(t for t in record["activation"]["tasks"] if t["id"] == record["assigned_task_ids"][0])
        return [dict(split=task["split"], id=task["id"], task_sha256=task["sha256"], source=s,
            status=status, reason=reason, attempt_seqs=[], gate_operation_id=None) for s in ("actor", "synthetic")]

    def test_tune_prefix_is_finite_strict_and_replays_from_start(self):
        for value in (True, 1.0, "1", 0, 41):
            with mock.patch.object(self.C, "_retained", side_effect=AssertionError("early retained lookup")), self.assertRaisesRegex(ValueError, "Invalid tuning prefix"):
                self.C.tune(None, None, "trial", through_tasks=value)
        campaign = self.fresh(); plan = self.setup(campaign)["payload"]; before = campaign.events
        with mock.patch.object(self.L, "run", side_effect=lambda c, l, r: self.terminal(r)) as run, mock.patch.object(self.R, "snapshot", side_effect=AssertionError("active snapshot")), mock.patch.object(self.R, "publish", side_effect=AssertionError("active publish")):
            first = self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            whole = self.C.tune(campaign, self.loaded, "trial")
        expected = [m["experiment_id"] for m in plan["records"][:80]]
        self.assertEqual([call.args[2]["experiment_id"] for call in run.call_args_list], expected[:2] + expected)
        self.assertEqual([r["experiment_id"] for r in whole], expected); self.assertEqual(len(first), 2)
        self.assertTrue(all(len(r["cells"]) == 2 for r in whole)); self.assertEqual(campaign.events, before)

    def test_tune_filters_assigned_cells_and_preserves_exception_barriers(self):
        campaign = self.fresh(); self.setup(campaign)
        allowed = (("verified", None), ("exhausted", "reply_limit"), ("blocked", "gate_failed"), ("blocked", "generation_timeout"), ("blocked", "generation_error"), ("blocked", "interrupted"))
        for status, reason in allowed:
            def work(c, l, r):
                rows = self.terminal(r); rows[1].update(status=status, reason=reason)
                return rows + [{**rows[0], "id": -1, "status": "pending"}]
            with mock.patch.object(self.L, "run", side_effect=work) as run: self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertEqual(run.call_count, 2)
        for fault in ("pending", "not_started", "native_infrastructure", "campaign_budget", "profile_failure", "unknown", "missing", "duplicate", "sha"):
            def bad(c, l, r):
                rows = self.terminal(r)
                if fault == "missing": return rows[:1]
                if fault == "duplicate": return rows + [rows[0]]
                rows[1].update(task_sha256="0" * 64) if fault == "sha" else rows[1].update(status=fault if fault in ("pending", "not_started", "unknown") else "blocked", reason=None if fault in ("pending", "not_started", "unknown") else fault)
                return rows
            message = "Assigned cells differ" if fault in ("missing", "duplicate", "sha") else "Collection member not terminal"
            with mock.patch.object(self.L, "run", side_effect=bad) as run, self.assertRaisesRegex(ValueError, message): self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertEqual(run.call_count, 1)
        campaign._append("native_intent", operation_id="f" * 12)
        for error in (KeyboardInterrupt("fixture"), OSError("fixture"), self.N.NativeBlocked("fixture")):
            with mock.patch.object(self.L, "run", side_effect=error) as run, self.assertRaises(type(error)) as caught: self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertIs(caught.exception, error); self.assertEqual(run.call_count, 1)

    def test_tune_pauses_new_guarded_errors_but_resumes_retained_and_timeouts(self):
        campaign = self.fresh(); self.setup(campaign)
        for reason in ("generation_error", "interrupted", "generation_timeout"):
            observed = []
            def work(c, l, r):
                rows = self.terminal(r)
                # L is a contract stub: this marker tests ordering, without inventing a guard proof.
                observed.append(c._append("fixture_guarded_observation")["seq"])
                rows[1].update(status="blocked", reason=reason, attempt_seqs=observed[-1:]); return rows
            with mock.patch.object(self.L, "run", side_effect=work) as run:
                if reason == "generation_timeout": self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
                else:
                    with self.assertRaisesRegex(RuntimeError, "New guarded execution failure"): self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertEqual(run.call_count, 2 if reason == "generation_timeout" else 1)
            def retained(c, l, r):
                rows = self.terminal(r); rows[1].update(status="blocked", reason=reason, attempt_seqs=observed[:1]); return rows
            with mock.patch.object(self.L, "run", side_effect=retained) as run: self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertEqual(run.call_count, 2)

    def test_tune_checks_producers_and_protocol_before_each_member(self):
        campaign = self.fresh(); self.setup(campaign); original = self.protocol.read_bytes()
        for fault in ("producer", "protocol"):
            current = self.C.fingerprint()
            def work(c, l, r):
                if fault == "producer": current[next(iter(current))] = "0" * 64
                else: self.protocol.write_bytes(original + b"changed")
                return self.terminal(r)
            with mock.patch.object(self.C, "fingerprint", return_value=current), mock.patch.object(self.L, "run", side_effect=work) as run, self.assertRaisesRegex(ValueError, "Producer sources changed" if fault == "producer" else "Protocol source changed"):
                self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
            self.assertEqual(run.call_count, 1); self.protocol.write_bytes(original)

    def test_tune_refuses_partial_superseded_or_early_member_work(self):
        campaign = self.fresh(); self.setup(campaign); append = campaign._append; pending = {}
        def stop(kind, **payload):
            if kind == "paired_plan": pending.update(payload); raise OSError("before paired plan")
            return append(kind, **payload)
        with mock.patch.object(campaign, "_append", side_effect=stop), self.assertRaises(OSError): self.setup(campaign, "second")
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
        with self.assertRaisesRegex(ValueError, "Collection binding differs"): self.C.tune(campaign, self.loaded, "second", through_tasks=1)
        name = pending["payload"]["records"][0]["experiment_id"]
        self.charge(campaign, next(e["record"] for e in campaign.events if e.get("experiment_id") == name and e["kind"] == "experiment"))
        campaign._append("paired_plan", **pending)
        with self.assertRaisesRegex(ValueError, "Member work precedes paired plan"): self.C.tune(campaign, self.loaded, "second", through_tasks=1)

    def test_tune_family_control_stop_outside_prefix_survives_reopen(self):
        campaign = self.fresh(); plan = self.setup(campaign)["payload"]; name = plan["records"][100]["experiment_id"]
        self.charge(campaign, next(e["record"] for e in campaign.events if e.get("experiment_id") == name and e["kind"] == "experiment"))
        before = campaign.events; spent = campaign._accounting()[1]; campaign.close()
        campaign = self.J.Campaign(campaign.root); self.addCleanup(campaign.close)
        with self.assertRaises(self.J.ProfileControlError) as failure: self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
        self.assertEqual(failure.exception.reason, "profile_unproven"); self.assertEqual(campaign.events, before)
        self.assertEqual(campaign._accounting()[1], spent); self.setup(campaign, "later")
        with mock.patch.object(self.L, "run", side_effect=lambda c, l, r: self.terminal(r)) as run: self.C.tune(campaign, self.loaded, "later", through_tasks=1)
        self.assertEqual(run.call_count, 2); self.assertEqual(campaign._accounting()[1], spent)

    def test_tune_preflights_all_contexts_funding_and_copied_audit(self):
        campaign = self.fresh(); self.setup(campaign); original = campaign.events
        for fault in ("context", "funding"):
            events = copy.deepcopy(original)
            if fault == "context":
                event = [e for e in events if e["kind"] == "execution_context"][-1]
                event["contract"]["fixture_drift"] = True; event["contract_sha256"] = self.N.verification.digest(event["contract"])
            else:
                event = next(e for e in events if e["kind"] == "collection_setup")
                event["payload"]["funding"]["reserved_before"]["actor"]["replies"] = 1; event["payload_sha256"] = self.C.digest(event["payload"])
            with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), self.assertRaisesRegex(ValueError, "Collection binding differs" if fault == "context" else "Collection funding differs"):
                self.C.tune(campaign, self.loaded, "trial", through_tasks=1)
        audit = next(e["payload"]["audit"]["path"] for e in original if e["kind"] == "mechanics_baseline"); (campaign.root / audit).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Collection artifact differs"): self.C.tune(campaign, self.loaded, "trial", through_tasks=1)

    def test_tune_cli_requires_existing_path_and_only_tune_arguments(self):
        missing = self.root / "missing"; args = ["tune", "--campaign", str(missing), "--plan-id", "trial", "--through-tasks", "1"]
        with mock.patch.object(self.C.study, "load_cohort", return_value=self.loaded), mock.patch.object(self.J.Campaign, "create", side_effect=AssertionError("unexpected create")), self.assertRaises(FileNotFoundError): self.C.main(args)
        self.assertFalse(missing.exists())
        for argv in (args + ["--receipt", "forbidden"], ["fresh"], ["confirm"], args[:-2], args[:-1] + ["0"], args[:-1] + ["41"]):
            with mock.patch("sys.stderr"), mock.patch.object(self.C.study, "load_cohort", side_effect=AssertionError("arguments accepted")), self.assertRaises(SystemExit): self.C.main(argv)

    def test_protocol_export_accepts_pending_and_historical_metadata_without_live_assets(self):
        campaign = self.fresh(); self.setup(campaign); events = campaign.events
        item = next(e["payload"]["protocol"] for e in events if e["kind"] == "collection_setup")
        historical = copy.deepcopy(events); event = next(e for e in historical if e["kind"] == "collection_setup")
        event["payload"]["producer_sources"]["src/data-pipeline/paired_collection.py"] = "0" * 64; event["payload_sha256"] = self.C.digest(event["payload"])
        self.protocol.unlink(); shutil.rmtree(self.original / "models"); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.C.study, "load_cohort", return_value=self.loaded), mock.patch.object(self.C, "_producer_guard", side_effect=AssertionError("live producer guard")), mock.patch.object(self.P, "check_runtime", side_effect=AssertionError("live runtime")):
            for prefix in (historical, historical[:-1]):
                with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=prefix): snapshot = self.R.snapshot(campaign)
                self.assertEqual(snapshot["files"][item["path"]]["sha256"], item["sha256"])
        self.assertEqual(campaign.events, events)

    def test_protocol_export_rejects_independent_link_funding_and_artifact_faults(self):
        campaign = self.fresh(); self.setup(campaign); original = campaign.events
        for fault in ("orphan", "duplicate", "pending_duplicate", "link", "order", "path", "funding", "producer"):
            events = copy.deepcopy(original); setup = next(e for e in events if e["kind"] == "collection_setup"); plan = events[-1]
            if fault == "orphan": events.remove(setup)
            if fault in ("duplicate", "pending_duplicate"): events.append(copy.deepcopy(plan if fault == "duplicate" else setup))
            if fault == "link": setup["payload"]["plan_sha256"] = "0" * 64
            if fault == "order": setup["seq"], plan["seq"] = plan["seq"], setup["seq"]; events.sort(key=lambda e: e["seq"])
            if fault == "path":
                item = setup["payload"]["protocol"]; alias = campaign.root / "audits/alias.md"; alias.write_bytes((campaign.root / item["path"]).read_bytes()); item["path"] = "audits/alias.md"
            if fault == "funding": setup["payload"]["funding"]["reserved_before"]["actor"]["replies"] = 1
            if fault == "producer": setup["payload"]["producer_sources"]["src/data-pipeline/paired_collection.py"] = "bad"
            setup["payload_sha256"] = self.C.digest(setup["payload"])
            for seq, event in enumerate(events, 1): event["seq"] = seq
            reason = "Duplicate collection plan" if fault in ("duplicate", "pending_duplicate") else "Invalid collection protocol" if fault == "path" else "Invalid collection funding" if fault == "funding" else "Invalid collection setup" if fault == "producer" else "Unbound paired plan"
            with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), self.assertRaisesRegex(ValueError, reason): self.R.snapshot(campaign)
        path = campaign.root / next(e["payload"]["protocol"]["path"] for e in original if e["kind"] == "collection_setup"); path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Artifact changed"): self.R.snapshot(campaign)
