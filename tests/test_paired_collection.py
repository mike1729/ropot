import copy
import importlib
import json
from pathlib import Path
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
        with mock.patch.object(self.J.Campaign, "create", side_effect=AssertionError("unexpected create")), self.assertRaises(SystemExit): self.C.main(["tune"])

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
