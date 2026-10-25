import copy
import importlib
import json
import unittest
from unittest import mock
from tests import test_paired_wave_collection as wave, test_paired_report as paired, test_batch_report as batch


class PairedWaveReportTests(unittest.TestCase):
    emit, anchor = paired.PairedReportTests.emit, paired.PairedReportTests.anchor
    @classmethod
    def setUpClass(cls):
        cls.generate_batch = staticmethod(importlib.import_module("src.data-pipeline.generation_journal").Campaign.generate_batch)
        cls.native_run = staticmethod(importlib.import_module("src.data-pipeline.native_recovery").run_native)
        cls.f = wave.PairedWaveCollectionTests(methodName="runTest"); cls.f.setUp()
        cls.addClassCleanup(cls.f.doCleanups)
        for task in cls.f.loaded["cohort"]["tasks"]: task["sha256"] = cls.f.C.sha256(str(task["id"]).encode()).hexdigest()
        cls.c = cls.f.fresh(); cls.f.setup(cls.c)
        cls.baseline = copy.deepcopy(cls.c.events); cls.built = cls.f.built()
        cls.R = importlib.import_module("src.data-pipeline.paired_wave_report")
        cls.S = importlib.import_module("src.data-pipeline.study_report")
        cls.J, cls.L = cls.f.J, cls.f.C.L
    def setUp(self):
        self.events = copy.deepcopy(self.baseline); self.outcomes = {}; self.ready = {}
        patch = mock.patch.object(self.R.PR.study, "load_cohort", return_value=self.f.loaded)
        self.metadata_patch = patch; patch.start(); self.addCleanup(patch.stop)

    def inputs(self):
        # These unit maps enter AFTER S's retained artifact/native proof boundary.
        records = {e["experiment_id"]: e for e in self.events if e["kind"] == "experiment"}
        return dict(events=self.events, records=records, runs=[dict(experiment_id=n, cells=self.L.cell_states(self.events, e["record"])) for n, e in records.items()],
            outcomes=self.outcomes, reservations={e["seq"]: e for e in self.events if e["kind"] == "generation_reserved"}, panel=self.built["panel"], ready_by_session=self.ready)

    def facts(self): return self.R._validate(**self.inputs())[0]
    def reasons(self): return {r["kind"] for r in self.facts()["invalid_reasons"]}
    def refresh(self):
        records = {e["experiment_id"]: e for e in self.events if e["kind"] == "experiment"}
        for event in records.values(): event["record_sha256"] = self.S.digest(event["record"])
        for event in (e for e in self.events if e["kind"] == "wave_plan"):
            for member in event["payload"]["records"]:
                if member["experiment_id"] in records: member["record_sha256"] = records[member["experiment_id"]]["record_sha256"]
        for e in self.events:
            if "payload_sha256" in e: e["payload_sha256"] = self.S.digest(e["payload"])
        setup = next(e for e in self.events if e["kind"] == "wave_setup")
        setup["payload"]["plan_sha256"] = next(e["payload_sha256"] for e in self.events if e["kind"] == "wave_plan")
        setup["payload_sha256"] = self.S.digest(setup["payload"])

    def legacy_authority(self):
        cls = type(self)
        if not hasattr(cls, "_legacy_authority"):
            with mock.patch.object(self.f, "methods", {8: {}, 16: {}}), mock.patch.object(self.f, "qualifications", {8: {}, 16: {}}):
                for bits in (8, 16):
                    for source in ("actor", "synthetic"):
                        ceiling = (3100 if bits == 8 else 3000) if source == "actor" else (3200 if bits == 8 else 3050)
                        self.f.make_qualification(bits, source, 8 if source == "actor" else 4, ceiling, legacy=True)
                for source in ("actor", "synthetic"):
                    minimum = min(self.f.methods[b][source]["prompt_ceiling_tokens"] for b in (8, 16))
                    for bits in (8, 16): self.f.methods[bits][source]["prompt_ceiling_tokens"] = minimum
                cls._legacy_authority = copy.deepcopy((self.f.methods, self.f.qualifications))
        return copy.deepcopy(cls._legacy_authority)

    def test_exact_historical_setup_and_plan_replay_with_four_legacy_qualifications(self):
        methods, references = self.legacy_authority(); built = self.f.built(methods)
        records = {r["experiment_id"]: r for r in built["records"]}
        for event in (e for e in self.events if e["kind"] == "experiment"): event["record"] = copy.deepcopy(records[event["experiment_id"]])
        next(e for e in self.events if e["kind"] == "wave_plan")["payload"] = built["plan"]
        setup = next(e for e in self.events if e["kind"] == "wave_setup")["payload"]
        setup["schema_version"] = 1; setup.pop("qualification_contract"); self.refresh()
        self.assertEqual(set(setup), set("schema_version plan_id plan_sha256 producer_sources protocol funding".split()))
        self.assertEqual(set(built["plan"]), self.R.PR.FIELDS | {"opportunities"})
        self.assertEqual(built["plan"]["schema_version"], 1)
        for bits in (8, 16):
            for source in ("actor", "synthetic"):
                method = methods[bits][source]; reference = references[bits][source]
                authority, files = self.f.H.read_qualification(reference["path"], reference["response_sha256"])
                self.assertEqual(authority["schema_version"], 1); self.assertNotIn("contract", authority)
                self.f.H.copy_qualification(files, self.c.root / method["qualification"]["path"], reference["response_sha256"])
        with mock.patch.object(type(self.c), "events", new_callable=mock.PropertyMock, return_value=self.events):
            retained, _ = self.f.C._retained(self.c, self.f.loaded, "trial")
        self.assertEqual(retained["plan"], built["plan"])
        with mock.patch.object(self.R.Path, "read_bytes", side_effect=AssertionError("Historical report source read")), \
             mock.patch.object(self.f.C, "_retained", side_effect=AssertionError("Live replay")):
            self.assertEqual(self.facts()["invalid_reasons"], [])

    def test_setup_marker_downgrade_and_exact_grammar_reject_after_hash_refresh(self):
        for fault in ("missing", "wrong", "false", "extra", "bool", "unknown", "downgrade", "legacy_marker"):
            self.events = copy.deepcopy(self.baseline)
            setup = next(e for e in self.events if e["kind"] == "wave_setup")["payload"]
            if fault == "missing": setup.pop("qualification_contract")
            if fault in ("wrong", "false"): setup["qualification_contract"] = "other" if fault == "wrong" else False
            if fault == "extra": setup["extra"] = 1
            if fault in ("bool", "unknown"):
                setup["schema_version"] = True if fault == "bool" else 3
                setup.pop("qualification_contract")
            if fault in ("downgrade", "legacy_marker"):
                setup["schema_version"] = 1
                if fault == "downgrade": setup.pop("qualification_contract")
            self.refresh()
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "Invalid wave qualification authority" if fault == "downgrade" else "Invalid wave setup"):
                self.facts()
            with mock.patch.object(type(self.c), "events", new_callable=mock.PropertyMock, return_value=self.events), \
                 self.assertRaisesRegex(ValueError, "Wave qualification authority differs" if fault == "downgrade" else "Wave binding differs"):
                self.f.C._retained(self.c, self.f.loaded, "trial")

    def test_report_rejects_mixed_authority_and_paired_policy_after_hash_refresh(self):
        legacy, _ = self.legacy_authority()
        for fault in ("last_legacy", "matched_mixed_sources", "contract", "missing_contract", "config", "cache_value", "cache_presence", "minimum"):
            self.events = copy.deepcopy(self.baseline)
            for event in (e for e in self.events if e["kind"] == "experiment"):
                record = event["record"]; bits = record["profiles"]["actor"]["descriptor"]["bits"]
                if fault == "matched_mixed_sources" or fault == "last_legacy" and bits == 16:
                    record["generation_method"]["synthetic"] = copy.deepcopy(legacy[bits]["synthetic"])
                elif bits == 16:
                    method = record["generation_method"]["synthetic"]; authority = method["admission"]
                    if fault == "contract": authority["contract"] = "other"
                    if fault == "missing_contract": authority.pop("contract")
                    if fault == "config": authority["config"]["prefill_step_size"] *= 2
                    if fault == "cache_value": authority["cache_limit_bytes"] = 4 * 1024**3
                    if fault == "cache_presence": authority.pop("cache_limit_bytes")
                    if fault == "minimum": method["prompt_ceiling_tokens"] -= 1
            self.refresh()
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.facts()

    def session(self, opportunity=0, names=None, messages=None):
        o = self.built["plan"]["opportunities"][opportunity]; records = {r["experiment_id"]: r for r in self.built["records"]}
        names = o["experiment_ids"] if names is None else names; method = records[names[0]]["generation_method"][o["source"]]
        for name in names:
            task = next(m["task"] for m in self.built["plan"]["records"] if m["experiment_id"] == name)
            if not any(e["kind"] == "native_intent" and e["task_sha256"] == task["task_sha256"] for e in self.events):
                op = task["task_sha256"][:12]; self.emit("native_intent", operation_id=op, task_sha256=task["task_sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
                self.emit("native_verdict", operation_id=op, status="passed"); self.emit("native_closed", operation_id=op)
        intent = self.emit("batch_reserved", rows=[], method_spec=method); rows = []
        for i, name in enumerate(names):
            member = next(m for m in self.built["plan"]["records"] if m["experiment_id"] == name)
            request = copy.deepcopy(method["admission"]["request_controls"])
            request.update(messages=messages or [dict(role="system", content="SYSTEM"), dict(role="user", content="fixture")], seed=records[name]["recipes"][o["source"]]["seeds"][o["reply_index"]])
            raw = self.J.encoded(request) + "\n"
            r = self.emit("generation_reserved", experiment_id=name, source=o["source"], session_seq=intent["seq"], row_index=i,
                request_text=raw, request_sha256=self.S.sha256(raw.encode()).hexdigest(), attempt={**member["task"], "reply_index": o["reply_index"]})
            intent["rows"].append(dict(reservation_seq=r["seq"], row_id=str(i), request_sha256=r["request_sha256"])); rows.append(r)
        done = self.emit("batch_done", session_seq=intent["seq"], eligible=True, shared=dict(resources_closed=True))
        for r in rows:
            d = self.emit("generation_done", reservation_seq=r["seq"], session_seq=intent["seq"], batch_done_seq=done["seq"], eligible=True,
                row_id=str(r["row_index"]), method="program-wave-v1", profile_guarded=True, profile_ready=True, protocol_error=None, status="stop", text="not python !")
            self.outcomes[r["seq"]] = d
            self.emit("attempt_result", reservation_seq=r["seq"], status="format_failed", candidate_sha256=None, operation_id=None, sample_id=None, reused=False)
        self.ready[intent["seq"]] = dict(fixtures=[dict(id=str(i), prompt="same", prompt_ids=[1]) for i in range(len(rows))], eos_token_ids=[9])
        return intent, rows

    def step(self, opportunity, session):
        p = next(e for e in self.events if e["kind"] == "wave_plan")
        return self.anchor("wave_step", dict(plan_id="trial", plan_sha256=p["payload_sha256"], opportunity_id=opportunity, session_seq=session))

    def test_facts_provisional_contract_and_pending_setup(self):
        facts = self.facts(); self.assertEqual(set(facts), {"setup", "plan", "rows", "session_opportunities", "steps", "confirmations", "invalid_reasons"})
        self.assertEqual((len(facts["rows"]), facts["invalid_reasons"]), (144, []))
        trial = self.R.derive(**self.inputs(), eligible=[], sessions=[])[0]
        self.assertEqual(trial, dict(report_scope="resident_wave", validation_scope="structure_only", provisional=True,
            plan_id="trial", plan_sha256=facts["plan"]["payload_sha256"], structure_valid=True, invalid_reasons=[]))
        self.events = [e for e in self.events if e["kind"] != "wave_plan"]
        self.assertEqual(self.R.derive(**self.inputs(), eligible=[], sessions=[]), [])

    def test_canonical_schema_matrix_methods_opportunities_and_funding_reject(self):
        baseline = copy.deepcopy(self.events)
        for fault in ("rows", "bits", "task", "opportunity", "schema", "method", "funding", "orphan", "hash"):
            self.events = copy.deepcopy(baseline); plan = next(e for e in self.events if e["kind"] == "wave_plan")
            record = next(e for e in self.events if e["kind"] == "experiment")
            if fault == "rows": plan["payload"]["records"].pop()
            elif fault == "bits": plan["payload"]["records"][0]["bits"] = True
            elif fault == "task": plan["payload"]["records"][0]["task"]["task_sha256"] = "0" * 64
            elif fault == "opportunity": plan["payload"]["opportunities"][0]["experiment_ids"].reverse()
            elif fault == "funding": next(e for e in self.events if e["kind"] == "wave_setup")["payload"]["funding"]["reserved_before"]["actor"]["replies"] += 1
            elif fault in ("schema", "method"):
                if fault == "schema": record["record"]["schema_version"] = 2
                else: record["record"]["generation_method"]["actor"]["prompt_ceiling_tokens"] -= 1
                record["record_sha256"] = plan["payload"]["records"][0]["record_sha256"] = self.S.digest(record["record"])
            self.refresh()
            if fault == "orphan": self.events = [e for e in self.events if e["kind"] != "wave_setup"]
            if fault == "hash": plan["payload_sha256"] = "0" * 64
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.facts()

    def test_every_row_own_subset_order_controls_and_ready(self):
        intent, rows = self.session(names=self.built["plan"]["opportunities"][0]["experiment_ids"][::2])
        self.assertEqual(self.facts()["session_opportunities"], {intent["seq"]: 0}); self.assertEqual(self.reasons(), set())
        baseline = copy.deepcopy(self.events)
        for field, value in (("profile_guarded", False), ("protocol_error", "invalid proof")):
            self.outcomes[rows[-1]["seq"]][field] = value; self.assertTrue(self.reasons())
            self.events = copy.deepcopy(baseline); self.outcomes = {e["reservation_seq"]: e for e in self.events if e["kind"] == "generation_done"}
        self.events = copy.deepcopy(self.baseline); self.outcomes = {}; self.ready = {}
        opposite = next(o["id"] for o in self.built["plan"]["opportunities"] if o["bits"] == 16 and o["source"] == "actor" and o["reply_index"] == 0)
        a, _ = self.session(); self.assertEqual(self.reasons(), set()); b, _ = self.session(opposite)
        self.assertEqual(self.reasons(), {"opportunity_checkpoint_gap"})
        self.ready[b["seq"]]["fixtures"][0]["prompt_ids"] = [2]; self.assertEqual(self.reasons(), {"opportunity_checkpoint_gap", "initial_ready_mismatch"})
        self.ready[b["seq"]] = None; self.assertEqual(self.reasons(), {"opportunity_checkpoint_gap"})
        self.events = copy.deepcopy(self.baseline); self.outcomes = {}; self.ready = {}; self.session(opposite); self.session(0)
        self.assertIn("opportunity_order_mismatch", self.reasons())

    def test_frozen_control_schema_and_last_row_request_controls_reject(self):
        intent, rows = self.session(); self.assertEqual(self.reasons(), set()); clean = copy.deepcopy(self.events)
        for fault in ("seed", "row_order", "step_type", "confirmation_rationale"):
            self.events = copy.deepcopy(clean)
            if fault == "seed":
                r = next(e for e in self.events if e["seq"] == rows[-1]["seq"]); request = json.loads(r["request_text"]); request["seed"] += 1
                r["request_text"] = self.J.encoded(request) + "\n"; r["request_sha256"] = self.S.sha256(r["request_text"].encode()).hexdigest()
            elif fault == "row_order": next(e for e in self.events if e["seq"] == intent["seq"])["rows"].reverse()
            elif fault == "step_type": self.step(0, True)
            else: self.anchor("wave_confirmation", dict(plan_id="trial", plan_sha256=next(e["payload_sha256"] for e in self.events if e["kind"] == "wave_plan"), decision="unchanged", rationale=""))
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.facts()

    def test_own_repair_role_context_text_and_format_instruction(self):
        intent, rows = self.session(); self.step(0, intent["seq"]); initial = json.loads(rows[0]["request_text"])["messages"]
        good = initial + [dict(role="assistant", content="not python !"), dict(role="user", content=self.L.FORMAT_REPAIR)]
        _, repairs = self.session(1, messages=good); self.assertEqual(self.reasons(), set()); baseline = copy.deepcopy(self.events)
        for position, field, value in ((0, "role", "user"), (1, "content", "changed context"), (2, "content", "peer text"), (3, "content", "hidden feedback")):
            self.events = copy.deepcopy(baseline); r = next(e for e in self.events if e["seq"] == repairs[-1]["seq"])
            request = json.loads(r["request_text"]); request["messages"][position][field] = value
            r["request_text"] = self.J.encoded(request) + "\n"; r["request_sha256"] = self.S.sha256(r["request_text"].encode()).hexdigest()
            with self.subTest(position=position): self.assertTrue(self.reasons())
        self.events = copy.deepcopy(baseline); result = next(e for e in self.events if e["kind"] == "attempt_result")
        self.events.remove(result); self.emit("attempt_result", **{k: v for k, v in result.items() if k not in ("seq", "kind")})
        self.assertTrue(self.reasons())

    def test_verification_repair_uses_only_own_public_feedback_and_prior_closed_proof(self):
        intent, rows = self.session(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:1]); old = rows[0]; op = "f" * 12
        feedback = dict(status="failed", hidden="not_run", visible=[dict(input="é", answer="1", status="wrong_answer", stdout="0", stderr=None, returncode=0, diagnostic="PRIVATE")])
        result = next(e for e in self.events if e["kind"] == "attempt_result"); result.update(status="verification_failed", candidate_sha256=self.S.sha256(b"print(0)").hexdigest(), operation_id=op, sample_id=7)
        self.outcomes[old["seq"]]["text"] = "print(0)"
        self.emit("native_intent", operation_id=op, task_sha256=old["attempt"]["task_sha256"], reference_indices=None, contract_sha256=self.built["plan"]["contract_sha256"])
        self.emit("native_verdict", operation_id=op, status="failed", results=[dict(sample_id=7, status="failed", key=dict(program_sha256=self.S.sha256(b"print(0)").hexdigest(), task_sha256=old["attempt"]["task_sha256"], contract_sha256=self.built["plan"]["contract_sha256"]), feedback=feedback)])
        self.emit("native_closed", operation_id=op); self.step(0, intent["seq"])
        messages = json.loads(old["request_text"])["messages"] + [dict(role="assistant", content="print(0)"), dict(role="user", content="Repair using this public feedback:\n" + json.dumps(self.L.project_feedback(feedback), ensure_ascii=False, separators=(",", ":"), allow_nan=False))]
        _, repairs = self.session(1, names=[old["experiment_id"]], messages=messages); self.assertEqual(self.reasons(), set()); baseline = copy.deepcopy(self.events)
        for fault in ("raw_feedback", "native_verdict", "native_closed", "attempt_result"):
            self.events = copy.deepcopy(baseline)
            if fault == "raw_feedback":
                r = next(e for e in self.events if e["seq"] == repairs[0]["seq"]); request = json.loads(r["request_text"])
                request["messages"][-1]["content"] = "Repair using this public feedback:\n" + json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))
                r["request_text"] = self.J.encoded(request) + "\n"; r["request_sha256"] = self.S.sha256(r["request_text"].encode()).hexdigest()
            else:
                late = next(e for e in self.events if e["kind"] == fault and (e.get("operation_id") == op or fault == "attempt_result"))
                self.events.remove(late); self.emit(fault, **{k: v for k, v in late.items() if k not in ("seq", "kind")})
            with self.subTest(fault=fault): self.assertIn("repair_history_mismatch", self.reasons())

    def test_step_requires_current_effects_at_prefix_and_none_requires_own_usable_cells(self):
        intent, _ = self.session(); step = self.step(0, intent["seq"]); self.assertEqual(self.reasons(), set())
        baseline = copy.deepcopy(self.events)
        for kind in ("generation_done", "attempt_result", "batch_done", "failed_gate", "late_gate"):
            self.events = copy.deepcopy(baseline)
            late = next(e for e in self.events if e["kind"] == ("native_verdict" if kind.endswith("gate") else kind))
            if kind == "failed_gate": late["status"] = "failed"
            else:
                self.events.remove(late); self.emit(late["kind"], **{k: v for k, v in late.items() if k not in ("seq", "kind")})
            with self.subTest(kind=kind): self.assertEqual(self.reasons(), {"step_premature"})
        self.events = copy.deepcopy(baseline); self.step(0, intent["seq"]); self.assertTrue(self.reasons())
        self.events = copy.deepcopy(self.baseline); self.outcomes = {}; self.ready = {}; self.step(0, None)
        self.assertTrue(self.reasons())
        self.events = copy.deepcopy(self.baseline)
        for name in self.built["plan"]["opportunities"][0]["experiment_ids"]:
            m = next(m for m in self.built["plan"]["records"] if m["experiment_id"] == name); op = m["task"]["task_sha256"][:12]
            self.emit("native_intent", operation_id=op, task_sha256=m["task"]["task_sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
            self.emit("native_verdict", operation_id=op, status="failed"); self.emit("native_closed", operation_id=op)
        self.step(0, None); self.assertEqual(self.reasons(), set())
        self.events = copy.deepcopy(self.baseline); self.outcomes = {}; self.ready = {}; first, rows = self.session(); self.step(0, first["seq"])
        messages = json.loads(rows[0]["request_text"])["messages"] + [dict(role="assistant", content="not python !"), dict(role="user", content=self.L.FORMAT_REPAIR)]
        second, _ = self.session(1, messages=messages); self.step(1, second["seq"]); self.session(2, messages=messages)
        self.assertEqual(self.reasons(), set()); self.step(2, None); self.assertEqual(self.reasons(), {"step_premature"})

    def test_real_retained_batch_native_boundary_pending_protocol_export_and_accepted_ready(self):
        self.metadata_patch.stop(); b = batch.BatchReportTests(methodName="runTest"); b.setUp(); self.addCleanup(b.doCleanups)
        c, rows = b.fresh()
        with mock.patch.object(self.J.Campaign, "generate_batch", self.generate_batch): result = b.generate(c, rows)
        with mock.patch.object(b.N, "run_native", self.native_run): b.prove(c, result)
        payload = copy.deepcopy(next(e["payload"] for e in self.baseline if e["kind"] == "wave_setup"))
        for name in (payload["protocol"]["path"], f"audits/{self.f.C.BASE_PROTOCOL_SHA256}-paired-protocol.md"):
            destination = c.root / name; destination.parent.mkdir(exist_ok=True); destination.write_bytes((self.c.root / name).read_bytes())
        limits, spent = self.J.Campaign.totals(c.events); funding = payload["funding"]
        funding.update(reserved_before=spent, limits={s: {u: max(limits[s][u], spent[s][u] + funding["bound"][s][u]) for u in self.J.UNITS} for s in limits})
        c._append("wave_setup", payload=payload, payload_sha256=self.S.digest(payload))
        with mock.patch.object(self.R, "derive", wraps=self.R.derive) as derive, mock.patch.object(self.f.C, "_retained", side_effect=AssertionError("live setup")):
            snapshot = self.S.snapshot(c)
        self.assertEqual(snapshot["report"]["wave_trials"], []); self.assertIn(payload["protocol"]["path"], snapshot["files"])
        ready = derive.call_args.kwargs["ready_by_session"]; self.assertEqual(len(ready), 1); self.assertIsNotNone(next(iter(ready.values())))
