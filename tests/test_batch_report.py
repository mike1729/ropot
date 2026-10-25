import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_batch_settlement as settlement, test_study_tasks as tasks, test_native_recovery as native


class BatchReportTests(unittest.TestCase):
    rows = native.NativeRecoveryTests.rows

    def setUp(self):
        self.f = settlement.BatchSettlementTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.b, self.H, self.J = self.f.b, self.f.H, self.f.J
        self.s = tasks.StudyTaskTests(methodName="runTest"); self.s.setUp(); self.addCleanup(self.s.doCleanups)
        self.task = dict(native.fixture.task(4), split="test", id=1, difficulty="introductory", question="Invented fixture")
        self.s.write_task(1, self.task); self.loaded = self.s.rewrite(); self.logs = {}; self.index = 0
        self.R = importlib.import_module("src.data-pipeline.study_report"); self.N = self.R.native; self.L = self.R.loop
        for target, name, value in ((self.N, "_daemon", "fixture-daemon"), (self.N, "_inventory", ("fixture-daemon", [], False)),
                (self.N.verification.runner, "contract", native.fixture.CONTRACT)):
            patch = mock.patch.object(target, name, return_value=value); patch.start(); self.addCleanup(patch.stop)
        for target, name in ((self.N, "read_eval_log"), (self.N.verification, "read_eval_log")):
            patch = mock.patch.object(target, name, side_effect=lambda path, **kwargs: self.logs[str(path)]); patch.start(); self.addCleanup(patch.stop)
        for name in ("verify", "reference_gate"):
            patch = mock.patch.object(self.N.verification, name, side_effect=self.execute); patch.start(); self.addCleanup(patch.stop)

    def execute(self, *args, operation_id):
        result = native.NativeRecoveryTests.execute(self, *args, operation_id=operation_id)
        self.logs[result["log"]] = self.log; return result

    def register(self, c, names):
        ref = dict(path="precision/" + self.b.f.d["receipt_sha256"], receipt_sha256=self.b.f.d["receipt_sha256"])
        recipe = {k: self.b.raw[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}
        recipe.update(system="SYSTEM", instruction="INSTRUCTION", max_replies=1, seeds=[self.b.raw["seed"]])
        for name in names:
            record = self.L.run_record(self.loaded, name, ["development_smoke"], dict(actor=recipe), [1],
                profiles=dict(actor=dict(reference=ref, descriptor=self.b.f.d)))
            record.update(schema_version=3, generation_method=dict(actor=self.b.method)); c.register(record)
            c._append("execution_context", experiment_id=name, contract=native.fixture.CONTRACT, contract_sha256=self.N.verification.digest(native.fixture.CONTRACT))
        return [dict(experiment_id=name, source="actor", request=copy.deepcopy(self.b.raw),
            attempt=dict(split="test", id=1, task_sha256=self.s.records[1]["sha256"], reply_index=0)) for name in names]

    def fresh(self):
        self.index += 1; c = self.J.Campaign.create(self.b.root / f"report-{self.index}",
            {s: dict(zip(self.J.UNITS, (10, 20480, 2400))) for s in ("actor", "synthetic")}); self.addCleanup(c.close)
        self.b.f.P.freeze(c, Path(self.b.f.d["model_path"]).parents[1] / "receipt.json", self.b.f.d["receipt_sha256"])
        (c.root / "qualifications").mkdir(); self.H.copy_qualification(self.b.q.files, c.root / self.b.method["qualification"]["path"], self.b.q.pin)
        self.campaign = c
        return c, self.register(c, ("one", "two", "empty"))[:2]

    def generate(self, c, rows, mode=None):
        base = self.f.runner("cleanup" if mode == "unknown" else mode if mode != "huge" else None)
        def run(*args, **kwargs):
            response = base(*args, **kwargs)
            if mode == "huge": response["elapsed_seconds"] = 10**400; (args[1] / "response.json").write_bytes(self.H._raw(response))
            return response
        with mock.patch.object(self.H, "run", side_effect=run): return c.generate_batch(rows, self.b.method)

    def prove(self, c, result):
        self.campaign = c; sha = self.s.records[1]["sha256"]
        self.N.run_native(c, self.task, sha)
        seq, program = result[0]["reservation_seq"], b"print(1)"
        directory = c.root / "requests" / f"{seq:06d}"; directory.mkdir(parents=True); (directory / "candidate.py").write_bytes(program)
        proof = self.N.run_native(c, self.task, sha, program, reservation_seq=seq)
        c._append("attempt_result", reservation_seq=seq, status="verified", candidate_sha256=self.R.sha256(program).hexdigest(),
            operation_id=proof["operation_id"], sample_id=proof["results"][0]["sample_id"], reused=False)

    def test_cross_id_shared_wall_export_once_and_per_record_unallocated_including_empty(self):
        c, rows = self.fresh(); result = self.generate(c, rows); self.prove(c, result)
        getter = type(c).events.fget; reads = mock.Mock(side_effect=lambda instance: getter(instance))
        with mock.patch.object(type(c), "events", property(reads)), \
             mock.patch.object(c, "outcome", side_effect=AssertionError("Public outcome scan")), \
             mock.patch.object(self.J, "_batch_projection", wraps=self.J._batch_projection) as project, \
             mock.patch.object(self.N, "_reuse_native", wraps=self.N._reuse_native) as native_proof, \
             mock.patch.object(self.H, "run", side_effect=AssertionError("Generation replay")), \
             mock.patch.object(self.N, "run_native", side_effect=AssertionError("Native execution")):
            snapshot = self.R.snapshot(c)
        self.assertEqual((reads.call_count, project.call_count), (3, 1))
        self.assertEqual(native_proof.call_count, 2)
        prefix = native_proof.call_args.args[-1]
        self.assertTrue(all(call.args[-1] is prefix for call in native_proof.call_args_list))
        self.assertEqual(prefix, c.events)
        costs = snapshot["report"]["costs"]; ledger = costs["generation_sessions"]; self.assertEqual(len(ledger), 1)
        session = ledger[0]; self.assertEqual((session["source"], session["bits"], session["experiment_ids"]), ("actor", 8, ["one", "two"]))
        self.assertEqual((session["owned_elapsed_seconds"], session["observed_generation_tokens"]), (239.5, 3))
        self.assertEqual(costs["observed_by_source"]["actor"]["known_elapsed_seconds"], 239.5)
        self.assertEqual(costs["reserved"]["actor"], dict(zip(self.J.UNITS, (2, 4096, 480))))
        for run in snapshot["report"]["runs"]:
            observed = run["sources"][0]["observed"]; self.assertEqual(observed["shared_wall_allocation"], "unallocated")
            self.assertEqual(set(observed), {"observed_generation_tokens", "lower_bound_requests", "admitted_rows", "terminal_rows", "generation_session_seqs", "shared_wall_allocation"})
            self.assertEqual(observed["generation_session_seqs"], [] if run["experiment_id"] == "empty" else [session["session_seq"]])
        selected = snapshot["manifest"]["selections"][0]
        self.assertEqual((selected["session_seq"], selected["batch_done_seq"], selected["generation_method_sha256"]),
            (session["session_seq"], session["batch_done_seq"], self.R.digest(self.b.method)))
        evidence = next(e["evidence"] for e in c.events if e["kind"] == "batch_done")
        self.assertTrue(evidence.keys() <= snapshot["files"].keys())
        for row in result: self.assertNotIn(f"requests/{row['reservation_seq']:06d}/response.json", snapshot["files"])
        read, originals = Path.read_bytes, set(self.b.method["admission"]["source_hashes"])
        def offline(path):
            if str(path) in originals: raise AssertionError("Historical source read")
            return read(path)
        with mock.patch.object(Path, "read_bytes", offline), mock.patch.object(self.L, "policy_sha256", return_value="0" * 64), \
             mock.patch.object(self.b.f.G, "prepare_request", side_effect=AssertionError("Renormalization")), mock.patch.object(self.H, "current_source_compatible", side_effect=AssertionError("Live source check")):
            self.assertEqual(self.R.snapshot(c), snapshot)
        published = self.R.publish(c, "batch"); self.assertEqual(self.R.publish(c, "batch"), published)
        self.assertEqual(json.loads((Path(published["path"]) / "report.json").read_bytes())["costs"], costs)

    def test_public_native_reuse_always_captures_fresh_authoritative_history(self):
        c, rows = self.fresh(); self.prove(c, self.generate(c, rows)); events = c.events
        intent = next(e for e in events if e["kind"] == "native_intent")
        verdict = next(e for e in events if e["kind"] == "native_verdict" and e["operation_id"] == intent["operation_id"])
        closed = next(e for e in events if e["kind"] == "native_closed" and e["operation_id"] == intent["operation_id"])
        programs = self.N.verification.reference_programs(self.task)
        getter = type(c).events.fget; reads = mock.Mock(side_effect=lambda instance: getter(instance))
        with mock.patch.object(type(c), "events", property(reads)), \
             mock.patch.object(self.N, "_reuse_native", wraps=self.N._reuse_native) as proof, \
             mock.patch.object(self.H, "run", side_effect=AssertionError("Generation")), \
             mock.patch.object(self.N, "run_native", side_effect=AssertionError("Native dispatch")), \
             mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("Recovery")):
            self.assertEqual(self.N.reuse_native(c, intent, self.task, programs), verdict)
            marker = c._append("fixture_change")
            self.assertEqual(self.N.reuse_native(c, intent, self.task, programs), verdict)
        self.assertEqual((reads.call_count, proof.call_count), (2, 2))
        self.assertEqual(proof.call_args_list[0].args[-1], events)
        self.assertEqual(proof.call_args_list[1].args[-1][-1], marker)
        self.assertIsNot(proof.call_args_list[0].args[-1], proof.call_args_list[1].args[-1])
        # The fresh authoritative journal getter rejects this duplicate first.
        c._append("native_closed", **{k: v for k, v in closed.items() if k not in ("kind", "seq")})
        with self.assertRaisesRegex(ValueError, "Duplicate native closure"):
            self.N.reuse_native(c, intent, self.task, programs)

    def test_snapshot_rechecks_current_native_bytes_and_final_captured_prefix(self):
        for fault in ("native_bytes", "journal_append"):
            c, rows = self.fresh(); self.prove(c, self.generate(c, rows)); self.R.snapshot(c)
            proof, entered, successful = self.N._reuse_native, [], []
            def changed(campaign, intent, task, programs, events):
                if not entered:
                    if fault == "native_bytes":
                        verdict = next(e for e in events if e["kind"] == "native_verdict" and e["operation_id"] == intent["operation_id"])
                        (campaign.root / verdict["log"]).write_bytes(b"changed after artifact capture")
                    else: campaign._append("fixture_change")
                entered.append(intent["operation_id"])
                result = proof(campaign, intent, task, programs, events)
                successful.append(intent["operation_id"])
                return result
            error = "Native log evidence changed" if fault == "native_bytes" else "Snapshot prefix changed"
            with mock.patch.object(self.N, "_reuse_native", side_effect=changed), \
                 mock.patch.object(self.H, "run", side_effect=AssertionError("Generation")), \
                 mock.patch.object(self.N, "run_native", side_effect=AssertionError("Native dispatch")), \
                 mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("Recovery")), \
                 self.subTest(fault=fault), self.assertRaisesRegex(ValueError, error):
                self.R.snapshot(c)
            self.assertEqual((len(entered), len(successful)), (1, 0) if fault == "native_bytes" else (2, 2))

    def test_two_same_profile_sessions_remain_two_charges_and_independent_wall_entries(self):
        c, rows = self.fresh(); self.generate(c, rows); later = self.register(c, ("three", "four")); self.generate(c, later)
        costs = self.R.snapshot(c)["report"]["costs"]; ledger = costs["generation_sessions"]
        self.assertEqual(len(ledger), 2); self.assertEqual(ledger[0]["precision_profile"], ledger[1]["precision_profile"])
        self.assertEqual(len({r["session_seq"] for r in ledger}), 2)
        self.assertEqual(costs["observed_by_source"]["actor"]["known_elapsed_seconds"], 479)
        self.assertEqual(costs["reserved"]["actor"], dict(zip(self.J.UNITS, (4, 8192, 960))))

    def test_mixed_scalar_unknown_counters_do_not_duplicate_batch_wall_or_change_comparison(self):
        c, rows = self.fresh(); self.generate(c, rows)
        record = copy.deepcopy(next(e["record"] for e in c.events if e["kind"] == "experiment"))
        record.update(schema_version=2, experiment_id="scalar"); record.pop("generation_method"); c.register(record)
        c._append("execution_context", experiment_id="scalar", contract=native.fixture.CONTRACT, contract_sha256=self.N.verification.digest(native.fixture.CONTRACT))
        _, raw = self.b.f.G.prepare_request(self.b.raw)
        reservation = c._append("generation_reserved", experiment_id="scalar", source="actor", request_text=raw.decode(),
            request_sha256=self.R.sha256(raw).hexdigest(), attempt=rows[0]["attempt"])
        getter = type(c).events.fget; reads = mock.Mock(side_effect=lambda instance: getter(instance))
        with mock.patch.object(type(c), "events", property(reads)), \
             mock.patch.object(c, "outcome", side_effect=AssertionError("Public outcome scan")):
            snapshot = self.R.snapshot(c)
        self.assertEqual(reads.call_count, 4)
        self.assertEqual(next(e for e in c.events if e["kind"] == "generation_done" and e["reservation_seq"] == reservation["seq"])["status"], "interrupted")
        before = c.events; self.assertEqual(self.R.snapshot(c), snapshot); self.assertEqual(c.events, before)
        report = snapshot["report"]; observed = report["costs"]["observed_by_source"]["actor"]
        self.assertEqual((observed["observed_generation_tokens"], observed["known_elapsed_seconds"], observed["unknown_elapsed_sessions"]), (3, 239.5, 0))
        self.assertEqual((observed["unknown_response_requests"], observed["unknown_elapsed_requests"]), (1, 1))
        sources = {r["experiment_id"]: r["sources"][0] for r in report["runs"]}
        self.assertNotEqual(sources["one"]["comparison_sha256"], sources["scalar"]["comparison_sha256"])
        self.assertEqual(len(sources["scalar"]["observed"]), 5)

    def test_known_closed_timeout_unknown_cleanup_and_overflow_metadata_costs(self):
        for mode, elapsed, unknown in (("timeout", 241.5, 0), ("unknown", None, 1), ("huge", None, 1)):
            c, rows = self.fresh(); self.generate(c, rows, mode); costs = self.R.snapshot(c)["report"]["costs"]
            session, source = costs["generation_sessions"][0], costs["observed_by_source"]["actor"]
            self.assertEqual(session["owned_elapsed_seconds"], elapsed); self.assertEqual(source["unknown_elapsed_sessions"], unknown)
            self.assertEqual(source["known_elapsed_seconds"], elapsed or 0)
            self.assertEqual((source["unknown_response_requests"], source["unknown_elapsed_requests"]), (0, 0))
            self.assertFalse(session["eligible"]); self.assertEqual(len(source["generation_session_seqs"]), 1)
            if mode == "huge": self.assertEqual(session["reported_timing"]["elapsed_seconds"], 10**400)

    def test_empty_schema3_still_exports_owned_qualification_and_legacy_remains_five_keys(self):
        c, _ = self.fresh(); snapshot = self.R.snapshot(c)
        self.assertNotIn("generation_sessions", snapshot["report"]["costs"])
        prefix = self.b.method["qualification"]["path"] + "/"
        self.assertEqual({p.removeprefix(prefix) for p in snapshot["files"] if p.startswith(prefix)}, set(self.b.q.files))
        legacy = dict(observed_generation_tokens=0, lower_bound_requests=0, unknown_response_requests=0, known_elapsed_seconds=0, unknown_elapsed_requests=0)
        self.assertEqual(self.R._observed([], {}), legacy)
        self.assertEqual(snapshot["report"]["costs"]["observed_by_source"], dict(actor=legacy, synthetic=legacy))
        for run in snapshot["report"]["runs"]:
            row = run["sources"][0]; self.assertEqual(row["observed"]["admitted_rows"], 0); self.assertIn("generation_method", row)

    def test_native_candidate_text_sample_and_unselected_whole_method_bindings(self):
        c, rows = self.fresh(); result = self.generate(c, rows); self.prove(c, result); events = c.events
        for fault in ("text", "sample", "status", "method", "request", "empty_recipe"):
            changed = copy.deepcopy(events)
            if fault == "sample": next(e for e in changed if e["kind"] == "attempt_result")["sample_id"] = 99
            if fault == "status": next(e for e in changed if e["kind"] == "attempt_result")["status"] = "verification_failed"
            if fault == "method":
                record = next(e for e in changed if e["kind"] == "experiment" and e["experiment_id"] == "two")
                record["record"]["generation_method"]["actor"]["prompt_ceiling_tokens"] -= 1; record["record_sha256"] = self.R.digest(record["record"])
            if fault == "empty_recipe":
                record = next(e for e in changed if e["kind"] == "experiment" and e["experiment_id"] == "empty")
                record["record"]["recipes"]["actor"]["max_tokens"] -= 1; record["record_sha256"] = self.R.digest(record["record"])
            if fault == "request":
                event = [e for e in changed if e["kind"] == "generation_reserved"][-1]; request = json.loads(event["request_text"]); request["temperature"] = .3
                event["request_text"] = self.H._raw(request).decode(); event["request_sha256"] = self.R.sha256(event["request_text"].encode()).hexdigest()
            if fault == "text":
                event = next(e for e in changed if e["kind"] == "attempt_result"); raw = b"print(2)"
                (c.root / "requests" / f"{event['reservation_seq']:06d}" / "candidate.py").write_bytes(raw); event["candidate_sha256"] = self.R.sha256(raw).hexdigest()
            error = {"text": "Batch candidate lacks eligible source reply", "sample": "Unproven candidate", "status": "Unproven candidate",
                "method": "Request batch binding differs", "request": "Request batch binding differs", "empty_recipe": "Run batch method profile/controls differ"}[fault]
            with mock.patch.object(type(c), "events", new_callable=mock.PropertyMock, return_value=changed), mock.patch.object(self.L, "policy_sha256", return_value="0" * 64), self.assertRaisesRegex(ValueError, error):
                self.R.snapshot(c)
            seq = result[0]["reservation_seq"]; (c.root / "requests" / f"{seq:06d}" / "candidate.py").write_bytes(b"print(1)")
        failed, rows = self.fresh(); peers = self.generate(failed, rows, "timeout"); self.prove(failed, peers)
        with self.assertRaisesRegex(ValueError, "Batch candidate lacks eligible source reply"): self.R.snapshot(failed)

    def test_completed_session_tamper_cannot_preserve_report_or_publication_eligibility(self):
        c, rows = self.fresh(); self.generate(c, rows); self.R.publish(c, "before")
        session = next(e["session_seq"] for e in c.events if e["kind"] == "batch_done")
        (c.root / "sessions" / f"{session:06d}" / "decoded.jsonl").write_bytes(b"changed")
        with self.assertRaises(ValueError): self.R.snapshot(c)
        with self.assertRaises(ValueError):
            with self.J.Campaign(c.root) as reopened:
                self.R.publish(reopened, "after")
        self.assertFalse((c.root / "exports" / "after").exists())
