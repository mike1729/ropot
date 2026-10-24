import importlib
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_local_generation as local, test_native_recovery as native, test_study_tasks as study


class IntegratedLoopTests(unittest.TestCase):
    write_task, rewrite = study.StudyTaskTests.write_task, study.StudyTaskTests.rewrite
    limits, rows = native.NativeRecoveryTests.limits, native.NativeRecoveryTests.rows

    def setUp(self):
        study.StudyTaskTests.setUp(self)
        for identifier in (1, 2):
            task = {**native.fixture.task(4), "split": "test", "id": identifier,
                    "difficulty": self.records[identifier]["difficulty"], "question": f"QUESTION_{identifier}"}
            task["solutions"][-1] = "raise RuntimeError('REFERENCE_SECRET')"
            for case in task["cases"]:
                if not case["visible"]: case["input"] = "HIDDEN_SECRET\n"
            self.write_task(identifier, task)
        self.loaded = self.rewrite()
        native.NativeRecoveryTests.setUp(self)
        self.L = importlib.import_module("src.data-pipeline.experiment_loop")
        self.logs, self.fail_gate = {}, False
        patch = mock.patch.dict(self.api.PROFILES["synthetic"], model_path=str(self.root)); patch.start(); self.addCleanup(patch.stop)
        worker = local.FAKE_WORKER.replace('mode=request["messages"][-1]["content"]',
            'reply=json.loads((pathlib.Path(request["model_path"])/"replies.json").read_text())[str(request["seed"])]; mode="partial" if reply=="SLEEP" else "stop"')
        self.worker.write_text(worker.replace('else "",', 'else reply,'))
        self.replies({1: "print(input())", 2: "print(input().strip())", 3: "not python !", 4: "print('WRONG')", 13: "SLEEP"})
        for name in ("verify", "reference_gate"):
            patch = mock.patch.object(self.N.verification, name, side_effect=self.execute); patch.start(); self.addCleanup(patch.stop)

    def read(self, path, **kwargs): return self.logs[str(path)]

    def execute(self, *args, operation_id):
        result = native.NativeRecoveryTests.execute(self, *args, operation_id=operation_id)
        if self.fail_gate or args[2] == b"print('WRONG')":
            self.log.samples[0].store["cases"] = self.log.samples[0].store["cases"][:2]
            self.log.samples[0].store["cases"][0].update(stdout="999", diagnostic="DIAGNOSTIC_SECRET")
        self.logs[result["log"]] = self.log
        return result

    def replies(self, values): (self.root / "replies.json").write_text(json.dumps(values))

    def recipe(self, seeds):
        return dict(system="SYSTEM", instruction="INSTRUCTION", max_replies=len(seeds), seeds=list(seeds),
                    max_tokens=1, wall_time_seconds=0.5, temperature=0.2, top_p=0.9)

    def record(self, seeds=(1,), sources=("actor", "synthetic"), assigned=(1,), name="loop"):
        return self.L.run_record(self.loaded, name, ["development_smoke"], {s: self.recipe(seeds) for s in sources}, list(assigned))

    def fresh(self, **limits):
        self.index += 1; self.campaign = self.J.Campaign.create(self.root / str(self.index), self.limits(**limits))
        self.addCleanup(self.campaign.close)
        return self.campaign

    def test_all_three_reference_samples_reuse_across_sources_and_idempotent_full_grid(self):
        for sample_id, program in enumerate(self.N.verification.reference_programs(native.fixture.task(4))):
            self.replies({1: program.decode()}); campaign = self.fresh(); record = self.record()
            cells = self.L.run(campaign, self.loaded, record)
            self.assertEqual([c["status"] for c in cells], ["verified", "verified", "not_assigned", "not_assigned"])
            results = [e for e in campaign.events if e["kind"] == "attempt_result"]
            intents = [e for e in campaign.events if e["kind"] == "native_intent"]
            self.assertEqual((len(intents), len(results)), (1, 2))
            self.assertTrue(all(e["reused"] and e["sample_id"] == sample_id and e["operation_id"] == intents[0]["operation_id"] for e in results))
            self.assertEqual([campaign._accounting()[1][s]["replies"] for s in ("actor", "synthetic")], [1, 1])
            count = len(campaign.events)
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("replay")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("reexecute")):
                self.assertEqual(self.L.run(campaign, self.loaded, record), cells)
                self.assertEqual(self.L.cell_states(campaign.events, record), cells)
            self.assertEqual(len(campaign.events), count); campaign.close()

    def test_candidate_crashes_preserve_original_owner_and_cross_source_reuse(self):
        self.replies({1: "print(input().strip())"})
        for failure in ("link", "result"):
            campaign = self.fresh(); record = self.record(); append = campaign._append
            def lose_result(kind, **payload):
                if kind == "attempt_result": raise OSError("lost result")
                return append(kind, **payload)
            target, name, effect = (self.L.os, "link", OSError("before link")) if failure == "link" else (campaign, "_append", lose_result)
            with mock.patch.object(target, name, side_effect=effect), self.assertRaises(OSError): self.L.run(campaign, self.loaded, record)
            root = campaign.root; campaign.close(); self.campaign = campaign = self.J.Campaign(root); self.addCleanup(campaign.close)
            self.assertEqual(self.L.cell_states(campaign.events, record)[0]["status"], "pending")
            self.assertEqual([c["status"] for c in self.L.run(campaign, self.loaded, record)][:2], ["verified"] * 2)
            results = [e for e in campaign.events if e["kind"] == "attempt_result"]
            intents = [e for e in campaign.events if e["kind"] == "native_intent"]
            self.assertEqual((len(intents), [e["reused"] for e in results]), (2, [False, True]))
            self.assertEqual(intents[-1]["reservation_seq"], results[0]["reservation_seq"])
            self.assertEqual(sum(v["replies"] for v in campaign._accounting()[1].values()), 2); campaign.close()

    def test_format_and_visible_repairs_are_stateless_private_and_reply_bounded(self):
        campaign = self.fresh(); record = self.record((3, 4, 2), ("actor",))
        self.assertEqual(self.L.run(campaign, self.loaded, record)[0]["status"], "verified")
        events = [e for e in campaign.events if e["kind"] == "attempt_result"]
        self.assertEqual([e["status"] for e in events], ["format_failed", "verification_failed", "verified"])
        requests = [json.loads(e["request_text"]) for e in campaign.events if e["kind"] == "generation_reserved"]
        self.assertEqual([len(r["messages"]) for r in requests], [2, 4, 4])
        self.assertEqual([r["messages"][2]["content"] for r in requests[1:]], ["not python !", "print('WRONG')"])
        self.assertFalse(any(secret in str(requests) for secret in ("HIDDEN_SECRET", "REFERENCE_SECRET", "DIAGNOSTIC_SECRET")))
        limited = self.record((3, 4), ("actor",), name="limited")
        cells = self.L.run(campaign, self.loaded, limited)
        self.assertEqual((cells[0]["status"], cells[0]["reason"], cells[1]["status"]), ("exhausted", "reply_limit", "not_requested"))
        self.assertEqual(campaign._accounting()[1]["actor"]["replies"], 5)

    def test_timeout_blocks_only_its_cell_and_campaign_caps_skip_unfunded_gates(self):
        campaign = self.fresh(replies=1)
        record = self.L.run_record(self.loaded, "loop", ["development_smoke"],
            {"actor": self.recipe((13, 1)), "synthetic": self.recipe((1,))}, [1, 2])
        cells = self.L.run(campaign, self.loaded, record)
        self.assertEqual([(c["status"], c["reason"]) for c in cells],
                         [("blocked", "generation_timeout"), ("verified", None), ("blocked", "campaign_budget"), ("blocked", "campaign_budget")])
        self.assertEqual(sum(e["kind"] == "native_intent" for e in campaign.events), 1)
        self.assertEqual([campaign._accounting()[1][s]["replies"] for s in ("actor", "synthetic")], [1, 1])

    def test_all_old_slots_are_checked_before_new_dispatch(self):
        for fault in ("gap", "duplicate", "request", "missing", "terminal"):
            campaign = self.fresh(); record = self.record((3, 3), ("actor",), (1, 2)); campaign.register(record)
            contract = self.N.verification.runner.contract()
            campaign._append("execution_context", experiment_id="loop", contract=contract, contract_sha256=self.N.verification.digest(contract))
            binding, task = study.study.load_task(self.loaded, record["activation"], 2)
            self.N.run_native(campaign, task, binding["sha256"])
            for slot in range(2 if fault in ("missing", "terminal", "duplicate") else 1):
                messages = self.L.messages(task, self.recipe((3, 3)), "not python !" if slot else None)
                request = self.api.request_for("actor", messages, seed=4 if fault == "request" else 3, temperature=0.2, top_p=0.9, max_tokens=1, wall_time_seconds=0.5)
                outcome = campaign.generate("loop", "actor", request, attempt=dict(split="test", id=2, task_sha256=binding["sha256"], reply_index=0 if fault == "duplicate" else 1 if fault == "gap" else slot))
                if fault != "missing" or slot:
                    campaign._append("attempt_result", reservation_seq=outcome["reservation_seq"], status="verified" if fault == "terminal" and not slot else "format_failed", candidate_sha256=None, operation_id=None, sample_id=None, reused=False)
            reason = "Invalid reply slots" if fault in ("gap", "duplicate") else "Retained request" if fault == "request" else "Reply after unresolved or terminal slot"
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("premature generation")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("premature native")), self.assertRaisesRegex(ValueError, reason):
                self.L.run(campaign, self.loaded, record)
            campaign.close()

    def test_failed_and_operator_closed_unknown_gates_never_generate_or_retry(self):
        for unknown in (False, True):
            campaign = self.fresh(); record = self.record(); self.fail_gate = not unknown
            if unknown:
                campaign.register(record); contract = self.N.verification.runner.contract()
                campaign._append("execution_context", experiment_id="loop", contract=contract, contract_sha256=self.N.verification.digest(contract))
                binding, task = study.study.load_task(self.loaded, record["activation"], 1)
                verdict = self.N.run_native(campaign, task, binding["sha256"])
                campaign.db.execute("DELETE FROM events WHERE seq >= ?", (verdict["seq"],))
                Path(self.log.location).unlink()
                self.N.resolve_native(campaign, verdict["operation_id"], attestation=native.NativeManualResolutionTests.attestation)
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("must not generate")):
                cells = self.L.run(campaign, self.loaded, record)
            self.assertEqual([c["reason"] for c in cells[:2]], ["native_infrastructure" if unknown else "gate_failed"] * 2)
            count = len(campaign.events)
            with mock.patch.object(self.N, "run_native", side_effect=AssertionError("retry")): self.L.run(campaign, self.loaded, record)
            self.assertEqual(len(campaign.events), count); campaign.close()

    def test_cleanup_error_in_other_experiment_stops_before_registration_or_dispatch(self):
        campaign = self.fresh(); campaign.register({"experiment_id": "mechanics"}); kill = self.api._kill_group
        def uncertain(child): kill(child); raise OSError("cleanup uncertain")
        request = self.api.request_for("actor", [{"role": "user", "content": "fixture"}], seed=1, temperature=0.2, top_p=0.9, max_tokens=1, wall_time_seconds=0.5)
        with mock.patch.object(self.api, "_kill_group", side_effect=uncertain), self.assertRaises(OSError): campaign.generate("mechanics", "actor", request)
        self.campaign = campaign = self.J.Campaign(campaign.root); self.addCleanup(campaign.close); count = len(campaign.events)
        with mock.patch.object(self.api, "generate", side_effect=AssertionError("unsafe dispatch")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("unsafe native")), self.assertRaisesRegex(ValueError, "cleanup failure"):
            self.L.run(campaign, self.loaded, self.record())
        self.assertEqual(len(campaign.events), count)

    def test_operator_closed_unknown_candidate_never_regenerates_or_reverifies(self):
        campaign = self.fresh(); record = self.record((2,), ("actor",)); append = campaign._append
        def lose_candidate_verdict(kind, **payload):
            if kind == "native_verdict" and campaign.events[-1]["reference_indices"] is None: raise OSError("lost candidate verdict")
            return append(kind, **payload)
        with mock.patch.object(campaign, "_append", side_effect=lose_candidate_verdict), self.assertRaises(OSError): self.L.run(campaign, self.loaded, record)
        intent = campaign.events[-1]; Path(self.log.location).unlink()
        self.N.resolve_native(campaign, intent["operation_id"], attestation=native.NativeManualResolutionTests.attestation)
        with mock.patch.object(self.api, "generate", side_effect=AssertionError("regenerate")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("reverify")):
            self.assertEqual(self.L.run(campaign, self.loaded, record)[0]["reason"], "native_infrastructure")
        result = campaign.events[-1]
        self.assertEqual((result["status"], result["operation_id"], result["candidate_sha256"]),
                         ("verification_error", intent["operation_id"], self.L.sha256(b"print(input().strip())").hexdigest()))
        self.assertEqual(campaign._accounting()[1]["actor"]["replies"], 1)

    def test_pinned_contract_change_refuses_old_run_before_appending(self):
        campaign = self.fresh(); record = self.record(); self.L.run(campaign, self.loaded, record); count = len(campaign.events)
        with mock.patch.object(self.N.verification.runner, "contract", return_value={"image": {"Id": "changed"}}), mock.patch.object(self.api, "generate", side_effect=AssertionError("replay")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("fresh gate")), self.assertRaises(ValueError):
            self.L.run(campaign, self.loaded, record)
        self.assertEqual(len(campaign.events), count)

    def test_cleaned_cancellation_consumes_once_and_allows_independent_source(self):
        campaign = self.fresh(); record = self.record((1, 2))
        with mock.patch.object(self.api, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.L.run(campaign, self.loaded, record)
        self.campaign = campaign = self.J.Campaign(campaign.root); self.addCleanup(campaign.close)
        cells = self.L.run(campaign, self.loaded, record)
        self.assertEqual([(c["status"], c["reason"]) for c in cells[:2]], [("blocked", "interrupted"), ("verified", None)])
        self.assertEqual([campaign._accounting()[1][s]["replies"] for s in ("actor", "synthetic")], [1, 1])

    def test_reuse_checks_log_hash_and_every_gate_sample(self):
        for damage in ("hash", "other_sample"):
            campaign = self.fresh(); record = self.record(); self.L.run(campaign, self.loaded, record)
            if damage == "hash": Path(self.log.location).write_bytes(b"changed native bytes")
            else: self.log.samples[0].error = object()
            count = len(campaign.events)
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("replay")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("retry")), self.assertRaises(ValueError):
                self.L.run(campaign, self.loaded, record)
            self.assertEqual(len(campaign.events), count); campaign.close()
