import copy
import json
from unittest import mock
import unittest
from tests import test_batch_report as report


class LoopWaveTests(unittest.TestCase):
    def setUp(self):
        self.f = report.BatchReportTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.b, self.L, self.H, self.J, self.N = self.f.b, self.f.L, self.f.H, self.f.J, self.f.N
        self.factory = self.L.run_record
        self.f.task["solutions"][-1] = "REFERENCE_SECRET"
        for case in self.f.task["cases"]:
            if not case["visible"]: case.update(input="HIDDEN_INPUT_SECRET", answer="HIDDEN_ANSWER_SECRET")
        self.f.s.write_task(1, self.f.task); self.f.loaded = self.f.s.rewrite()

    def fresh(self):
        def factory(loaded, name, groups, recipes, assigned, **kwargs):
            recipes = copy.deepcopy(recipes)
            for recipe in recipes.values(): recipe.update(max_replies=3, seeds=[self.b.raw["seed"] + i for i in range(3)])
            return self.factory(loaded, name, groups, recipes, assigned, generation_method=dict(actor=self.b.method), **kwargs)
        with mock.patch.object(self.L, "run_record", side_effect=factory): c, _ = self.f.fresh()
        records = [e["record"] for e in c.events if e["kind"] == "experiment"]
        return c, records[:2]

    def prepare(self, c, records):
        return [self.L.prepare_cell(c, self.f.loaded, record, "actor", 1) for record in records]

    def generate(self, c, rows, mode=None, texts=("print(1)", "print(1)")):
        base = self.f.f.runner(mode if mode in ("timeout", "cleanup", "request") else None)
        def run(job, directory, *args, **kwargs):
            response = base(job, directory, *args, **kwargs)
            if mode in (None, "worker_error", "cancelled", "parent_error", "output_limit", "control"):
                decoded = json.loads((directory / "decoded.jsonl").read_bytes())
                for row, text in zip(decoded["rows"], texts): row["text"] = text
                raw = self.H._raw(decoded); (directory / "decoded.jsonl").write_bytes(raw)
                events = [json.loads(line) for line in (directory / "stdout.jsonl").read_bytes().splitlines()]
                events[-1]["decoded_sha256"] = self.H.sha256(raw).hexdigest()
                if mode == "control": events = events[:1]
                stdout = b"".join(self.H._raw(e) for e in events); response.update(self.H.validate_events(stdout, job, pid=7))
                if mode: response.update(status="worker_error" if mode == "control" else mode, returncode=3, parent_error="cancelled" if mode in ("cancelled", "parent_error") else None)
                (directory / "stdout.jsonl").write_bytes(stdout); (directory / "response.json").write_bytes(self.H._raw(response))
            return response
        with mock.patch.object(self.H, "run", side_effect=run): return c.generate_batch(rows, self.b.method)

    def synthetic_method(self, c, reference):
        P = self.b.f.P; descriptor = P.resolve(c, reference, "synthetic", 8)
        job = self.b.q.f.M.build_long_job(descriptor, self.b.q.f.config, cache_limit_bytes=0); events = copy.deepcopy(self.b.q.events)
        proof = copy.deepcopy(events[1]["proof"])
        def roles(post=False):
            required = P.role_suffixes("synthetic", post); sources = descriptor["expected_runtime"]["sources"]
            return P.bind_roles(sources, required, {role: next(p for p in sources if p.endswith(suffix or "/transformers/fixture.py")) for role, suffix in required.items()})
        proof.update(descriptor_sha256=P.digest(descriptor), model_files=descriptor["model_files"], critical_sources=roles())
        events[1]["proof"] = proof; events[2]["provenance"].update(precision_controls=proof, precision_postload=roles(True))
        files = self.b.q.pack(job, events); pin = self.H.sha256(files["response.json"]).hexdigest()
        method = dict(method="program-wave-v1", source="synthetic", admission=self.H.qualification_checked(files), prompt_ceiling_tokens=3000,
            qualification=dict(path="qualifications/" + pin, response_sha256=pin))
        self.H.copy_qualification(files, c.root / method["qualification"]["path"], pin)
        return descriptor, method

    def test_none_legacy_and_full_all_source_methods_validate_last_entry(self):
        c, records = self.fresh(); record = records[0]; profile = record["profiles"]
        args = (self.f.loaded, "factory", ["development_smoke"], record["recipes"], [1])
        self.assertEqual(self.factory(*args, profiles=profile), self.factory(*args, profiles=profile, generation_method=None))
        with mock.patch.dict(self.b.f.G.PROFILES["actor"], model_path=str(self.b.root)):
            self.assertEqual(self.factory(*args)["schema_version"], 1)
        descriptor, method = self.synthetic_method(c, profile["actor"]["reference"])
        profiles = dict(profile, synthetic=dict(reference=profile["actor"]["reference"], descriptor=descriptor))
        recipes = dict(record["recipes"], synthetic=record["recipes"]["actor"]); methods = dict(actor=self.b.method, synthetic=method)
        full = self.factory(self.f.loaded, "both", ["development_smoke"], recipes, [1], profiles=profiles, generation_method=methods)
        self.assertEqual(full["schema_version"], 3); self.assertEqual(self.L.validate_run(self.f.loaded, full), full)
        for fault in ("tag", "source", "profile", "controls", "missing"):
            changed = copy.deepcopy(methods)
            if fault == "tag": changed["synthetic"]["method"] = "other"
            if fault == "source": changed["synthetic"]["source"] = "actor"
            if fault == "profile": changed["synthetic"]["admission"]["request_controls"] = self.b.method["admission"]["request_controls"]
            if fault == "controls": changed["synthetic"]["admission"]["request_controls"]["top_p"] = .8
            if fault == "missing": changed.pop("synthetic")
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.factory(self.f.loaded, "bad", ["development_smoke"], recipes, [1], profiles=profiles, generation_method=changed)
        (c.root / method["qualification"]["path"] / "response.json").unlink(); before = c.events
        with mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Payload")), mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("Native")), self.assertRaises((ValueError, OSError)):
            self.L.prepare_cell(c, self.f.loaded, full, "actor", 1)
        self.assertEqual(c.events, before)

    def test_scalar_refuses_method_before_any_native_contract_registration_or_payload(self):
        c, records = self.fresh(); before = c.events
        with mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("Native")), mock.patch.object(self.N.verification.runner, "contract", side_effect=AssertionError("Contract")), mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Payload")), self.assertRaisesRegex(ValueError, "Batch record requires cell adapters"):
            self.L.run(c, self.f.loaded, records[0])
        self.assertEqual(c.events, before)

    def test_prepare_is_raw_own_selected_request_and_never_generates(self):
        c, records = self.fresh(); load = self.L.study.load_task
        with mock.patch.object(self.J.Campaign, "generate", side_effect=AssertionError("Scalar generation")), mock.patch.object(self.b.f.G, "generate", side_effect=AssertionError("Generation")), mock.patch.object(self.H, "run", side_effect=AssertionError("Session")), mock.patch.object(self.L.study, "load_task", wraps=load) as payload:
            rows = self.prepare(c, records)
        self.assertEqual([call.args[-1] for call in payload.call_args_list], [1, 1])
        for row, record in zip(rows, records):
            self.assertEqual(set(row), {"experiment_id", "source", "request", "attempt"}); self.assertNotIn("environment", row["request"])
            self.assertEqual(row["request"]["messages"], self.L.messages(self.f.task, record["recipes"]["actor"]))
            self.assertEqual((row["attempt"]["reply_index"], row["request"]["seed"]), (0, self.b.raw["seed"]))
        self.assertFalse(any(e["kind"] == "generation_reserved" for e in c.events))

    def test_settle_verifies_existing_rows_reuses_native_and_is_report_compatible(self):
        c, records = self.fresh(); rows = self.prepare(c, records); outcomes = self.generate(c, rows); before = len(c.events)
        with mock.patch.object(self.J.Campaign, "generate", side_effect=AssertionError("New slot")), mock.patch.object(self.H, "run", side_effect=AssertionError("Session")), mock.patch.object(self.b.f.G, "request_for", wraps=self.b.f.G.request_for) as requests:
            states = [self.L.settle_cell(c, self.f.loaded, record, "actor", 1) for record in records]
        self.assertEqual([s["status"] for s in states], ["verified", "verified"])
        self.assertTrue(all(call.kwargs["seed"] == self.b.raw["seed"] for call in requests.call_args_list if call.args[1][0]["content"] != "Validate"))
        results = [e for e in c.events if e["kind"] == "attempt_result"]; self.assertEqual([r["reused"] for r in results], [False, True])
        self.assertEqual(len({r["operation_id"] for r in results}), 1); snapshot = self.f.R.snapshot(c)
        self.assertEqual(len(snapshot["manifest"]["selections"]), 1); self.assertEqual(snapshot["report"]["costs"]["reserved"]["actor"]["replies"], 2)
        self.assertTrue(all((c.root / "requests" / f"{r['reservation_seq']:06d}" / "candidate.py").read_bytes() == b"print(1)" for r in outcomes))
        saved = c.events
        with mock.patch.object(self.N, "run_native", side_effect=AssertionError("Replay")):
            self.assertIsNone(self.L.prepare_cell(c, self.f.loaded, records[0], "actor", 1)); self.L.settle_cell(c, self.f.loaded, records[0], "actor", 1)
        self.assertEqual(c.events, saved); self.assertGreater(len(saved), before)

    def test_visible_only_and_format_repair_preserve_raw_previous_text_and_own_seed(self):
        for kind, text in (("visible", "print(1)"), ("format", "not python !")):
            c, records = self.fresh(); rows = self.prepare(c, records); self.generate(c, rows, texts=(text, text))
            execute = self.f.execute
            def fail(*args, **kwargs):
                result = execute(*args, **kwargs); sample = self.f.log.samples[0]
                visible_count = sum(case["visible"] for case in sample.metadata["cases"])
                sample.store["cases"] = sample.store["cases"][:visible_count]
                sample.store["cases"][0]["stdout"] = "WRONG"; return result
            with mock.patch.object(self.N.verification, "verify", side_effect=fail):
                self.L.settle_cell(c, self.f.loaded, records[0], "actor", 1)
                row = self.L.prepare_cell(c, self.f.loaded, records[0], "actor", 1)
            self.assertEqual((row["attempt"]["reply_index"], row["request"]["seed"]), (1, self.b.raw["seed"] + 1))
            self.assertEqual(row["request"]["messages"][2]["content"], text)
            prompt = str(row["request"]["messages"])
            for secret in ("HIDDEN_INPUT_SECRET", "HIDDEN_ANSWER_SECRET", "REFERENCE_SECRET"): self.assertNotIn(secret, prompt)
            self.assertIn("WRONG" if kind == "visible" else self.L.FORMAT_REPAIR, prompt)
            if kind == "visible": self.assertIn('"hidden":"not_run"', row["request"]["messages"][-1]["content"])

    def test_shared_failure_mapping_never_verifies_early_terminal_peer(self):
        for mode, expected in (("timeout", "generation_timeout"), ("worker_error", "generation_error"), ("output_limit", "generation_error"), ("cancelled", "interrupted"), ("parent_error", "interrupted"), ("request", "generation_error")):
            c, records = self.fresh(); rows = self.prepare(c, records); self.generate(c, rows, mode)
            with mock.patch.object(self.N, "run_native", side_effect=AssertionError("Ineligible candidate")):
                state = self.L.settle_cell(c, self.f.loaded, records[0], "actor", 1)
            result = [e for e in c.events if e["kind"] == "attempt_result"][-1]
            self.assertEqual(result["status"], expected); self.assertIsNone(result["candidate_sha256"])
            self.assertEqual(state["status"], "blocked"); self.assertIsNone(self.L.prepare_cell(c, self.f.loaded, records[0], "actor", 1))

    def test_unknown_resources_and_durable_control_stop_before_task_or_native(self):
        for mode in ("cleanup", "control"):
            c, records = self.fresh(); rows = self.prepare(c, records)
            if mode == "control":
                with self.assertRaises(self.J.ProfileControlError): self.generate(c, rows, mode)
                c = self.J.Campaign(c.root); self.addCleanup(c.close)
            else: self.generate(c, rows, mode)
            error = self.J.ProfileControlError if mode == "control" else ValueError
            with mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Payload")), mock.patch.object(self.N, "reconcile_native", side_effect=AssertionError("Native")), mock.patch.object(self.N.verification.runner, "contract", side_effect=AssertionError("Contract")), self.assertRaises(error):
                self.L.prepare_cell(c, self.f.loaded, records[0], "actor", 1)

    def test_missing_pre_generation_gate_cannot_be_created_late_and_selected_ids_are_strict(self):
        c, records = self.fresh(); rows = []
        for record in records:
            recipe = record["recipes"]["actor"]; request = self.b.f.G.request_for("actor", self.L.messages(self.f.task, recipe), profile=self.b.f.d,
                **{k: recipe[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}, seed=recipe["seeds"][0])
            rows.append(dict(experiment_id=record["experiment_id"], source="actor", request=request, attempt=dict(split="test", id=1, task_sha256=self.f.s.records[1]["sha256"], reply_index=0)))
        self.generate(c, rows)
        with mock.patch.object(self.N, "run_native", side_effect=AssertionError("Late gate")), self.assertRaisesRegex(ValueError, "Generation without a passed reference gate"):
            self.L.settle_cell(c, self.f.loaded, records[0], "actor", 1)
        for source, identifier in (("actor", True), ("actor", 2), ("actor", -1), ("synthetic", 1), (True, 1)):
            with mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Payload")), self.assertRaises(ValueError): self.L.prepare_cell(c, self.f.loaded, records[0], source, identifier)

    def test_candidate_and_result_crashes_replay_proof_without_another_gpu_session(self):
        for fault in ("candidate", "result", "after_result"):
            c, records = self.fresh(); prepared = self.prepare(c, records); self.generate(c, prepared); append = c._append
            def interrupted(kind, **payload):
                if kind == "attempt_result":
                    if fault == "after_result": append(kind, **payload)
                    raise OSError("result interruption")
                return append(kind, **payload)
            target, name, effect = (self.L, "publish_candidate", OSError("candidate interruption")) if fault == "candidate" else (c, "_append", interrupted)
            with mock.patch.object(target, name, side_effect=effect), self.assertRaises(OSError):
                self.L.settle_cell(c, self.f.loaded, records[0], "actor", 1)
            root = c.root; c.close(); reopened = self.J.Campaign(root); self.addCleanup(reopened.close); self.f.campaign = reopened
            with mock.patch.object(self.H, "run", side_effect=AssertionError("GPU replay")), mock.patch.object(self.J.Campaign, "generate", side_effect=AssertionError("Scalar replay")):
                state = self.L.settle_cell(reopened, self.f.loaded, records[0], "actor", 1)
            self.assertEqual(state["status"], "verified")
            self.assertEqual(sum(e["kind"] == "attempt_result" for e in reopened.events), 1)
            self.assertEqual(sum(e["kind"] == "native_intent" and e["reference_indices"] is None for e in reopened.events), 1)
            self.assertEqual(self.J.Campaign.totals(reopened.events)[1]["actor"]["replies"], 2)

    def test_native_reuse_is_source_independent_while_own_sessions_remain_distinct(self):
        c, existing = self.fresh(); record = existing[0]; reference = record["profiles"]["actor"]["reference"]
        descriptor, synthetic = self.synthetic_method(c, reference)
        profiles = dict(record["profiles"], synthetic=dict(reference=reference, descriptor=descriptor))
        recipes = dict(record["recipes"], synthetic=record["recipes"]["actor"]); methods = dict(actor=self.b.method, synthetic=synthetic)
        records = [self.factory(self.f.loaded, name, ["development_smoke"], recipes, [1], profiles=profiles, generation_method=methods) for name in ("both1", "both2")]
        rows = self.prepare(c, records); self.generate(c, rows)
        for record in records: self.assertEqual(self.L.settle_cell(c, self.f.loaded, record, "actor", 1)["status"], "verified")
        _, files = self.H.read_qualification(c.root / synthetic["qualification"]["path"], synthetic["qualification"]["response_sha256"])
        events = [json.loads(line) for line in files["stdout.jsonl"].splitlines()]
        with mock.patch.object(self.b, "method", synthetic), mock.patch.object(self.b.q, "pin", synthetic["qualification"]["response_sha256"]), mock.patch.object(self.b.q, "events", events):
            rows = [self.L.prepare_cell(c, self.f.loaded, record, "synthetic", 1) for record in records]; self.generate(c, rows)
        for record in records: self.assertEqual(self.L.settle_cell(c, self.f.loaded, record, "synthetic", 1)["status"], "verified")
        intents = [e for e in c.events if e["kind"] == "native_intent" and e["reference_indices"] is None]
        self.assertEqual(len(intents), 1); self.assertEqual(sum(e["kind"] == "batch_done" for e in c.events), 2)
        snapshot = self.f.R.snapshot(c); self.assertEqual({e["source"] for e in snapshot["manifest"]["selections"]}, {"actor", "synthetic"})
