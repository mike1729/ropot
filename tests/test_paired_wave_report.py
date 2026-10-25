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

    def scientific(self, opportunity=0, names=None, status="verified", messages=None):
        # Accepted S event projections, not a fabricated retained/native log proof.
        intent, rows = self.session(opportunity, names, messages)
        group = next(e for e in self.events if e["kind"] == "batch_done" and e["session_seq"] == intent["seq"])
        group.update(session_copy_complete=True, evidence_errors=[]); group["shared"]["status"] = "passed"
        for row in rows:
            done = self.outcomes[row["seq"]]; result = next(e for e in self.events if e["kind"] == "attempt_result" and e["reservation_seq"] == row["seq"])
            if status == "format_failed": continue
            program = self.L.generation.extract_program("print(0)"); candidate = self.S.sha256(program).hexdigest(); op = f"{row['seq']:012x}"
            done["text"] = program.decode(); result.update(status=status, candidate_sha256=candidate, operation_id=op, sample_id=0)
            self.emit("native_intent", operation_id=op, reservation_seq=row["seq"], task_sha256=row["attempt"]["task_sha256"], reference_indices=None,
                contract_sha256=self.built["plan"]["contract_sha256"], program_sha256=[candidate])
            native_status = {"verified": "passed", "verification_failed": "failed", "verification_error": "error"}[status]
            self.emit("native_verdict", operation_id=op, status=native_status, results=[dict(sample_id=0, status=native_status,
                key=dict(program_sha256=candidate, task_sha256=row["attempt"]["task_sha256"], contract_sha256=self.built["plan"]["contract_sha256"]))])
            self.emit("native_closed", operation_id=op)
        return intent, rows

    def own_cell(self, row):
        record = next(r for r in self.built["records"] if r["experiment_id"] == row["experiment_id"])
        return next(c for c in self.L.cell_states(self.events, record) if c["id"] == row["attempt"]["id"] and c["source"] == row["source"])

    def tuning_complete(self):
        tuning = self.R.PR._layout()[3]; target = tuning[0]["id"]; generated = []
        for task in tuning[1:]:
            op = task["sha256"][:12]
            self.emit("native_intent", operation_id=op, task_sha256=task["sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
            self.emit("native_verdict", operation_id=op, status="failed"); self.emit("native_closed", operation_id=op)
        target_names = {m["experiment_id"] for m in self.built["plan"]["records"] if m["task"]["id"] == target}
        for opportunity in self.built["plan"]["opportunities"]:
            if opportunity["block"].endswith("fresh"): break
            names = [n for n in opportunity["experiment_ids"] if n in target_names]
            if names and opportunity["reply_index"] == 0:
                session, rows = self.scientific(opportunity["id"], names); generated.extend(rows)
                self.step(opportunity["id"], session["seq"])
            else: self.step(opportunity["id"], None)
        return generated

    def confirm(self):
        return self.anchor("wave_confirmation", dict(plan_id="trial", plan_sha256=next(e["payload_sha256"] for e in self.events if e["kind"] == "wave_plan"), decision="unchanged", rationale="Fixture pooled review"))

    def confirmation(self, fact=None):
        return self.R._confirmation(self.facts() if fact is None else fact, events=self.events, ready_by_session=self.ready, layout=self.R.PR._layout())

    def restore_science(self, saved):
        self.events, self.ready = copy.deepcopy(saved)
        self.outcomes = {e["reservation_seq"]: e for e in self.events if e["kind"] == "generation_done"}

    def late(self, event):
        if event in self.events: self.events.remove(event)
        event["seq"] = max(e["seq"] for e in self.events) + 1; self.events.append(event)
        if event["kind"] == "batch_done":
            for row in self.outcomes.values():
                if row["session_seq"] == event["session_seq"]: row["batch_done_seq"] = event["seq"]

    def supersede(self, kind):
        payload = copy.deepcopy(next(e["payload"] for e in self.events if e["kind"] == "wave_setup")); payload["plan_id"] = "later"
        limits, reserved = self.J.Campaign.totals(self.events); funding = payload["funding"]
        funding.update(reserved_before=reserved, limits={s: {u: max(limits[s][u], reserved[s][u] + funding["bound"][s][u]) for u in self.J.UNITS} for s in limits})
        return self.anchor(kind, payload)

    def test_scientific_result_candidates_and_shared_authority(self):
        name = self.built["plan"]["opportunities"][0]["experiment_ids"][:1]
        for status in ("verified", "verification_failed", "verification_error"):
            self.restore_science((self.baseline, {})); _, rows = self.scientific(names=name, status=status)
            self.assertEqual(self.R._result_usable(rows[0]["seq"], self.events), status != "verification_error")
        self.restore_science((self.baseline, {})); _, rows = self.scientific(names=name); seq = rows[0]["seq"]
        self.assertTrue(self.R._result_usable(seq, self.events)); saved = copy.deepcopy((self.events, self.ready))
        for kind, field, value in (("generation_done", "profile_guarded", False), ("generation_done", "protocol_error", "bad proof"),
                ("generation_done", "eligible", False), ("batch_done", "eligible", False), ("batch_done", "session_copy_complete", False),
                ("batch_done", "evidence_errors", ["copy failed"]), ("shared", "resources_closed", False), ("shared", "status", "timeout")):
            self.restore_science(saved); event = next(e for e in self.events if e["kind"] == ("batch_done" if kind == "shared" else kind))
            (event["shared"] if kind == "shared" else event)[field] = value
            with self.subTest(kind=kind, field=field): self.assertFalse(self.R._result_usable(seq, self.events))
        for field in ("session_copy_complete", "evidence_errors"):
            self.restore_science(saved); next(e for e in self.events if e["kind"] == "batch_done").pop(field)
            self.assertFalse(self.R._result_usable(seq, self.events))
        self.restore_science(saved); next(e for e in self.events if e["kind"] == "batch_done")["shared"].pop("status")
        self.assertFalse(self.R._result_usable(seq, self.events))
        for kind in ("attempt_result", "generation_done", "batch_reserved", "batch_done"):
            self.restore_science(saved); self.events.remove(next(e for e in self.events if e["kind"] == kind))
            self.assertFalse(self.R._result_usable(seq, self.events))
        self.restore_science((self.baseline, {})); _, members = self.scientific(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:2])
        self.assertEqual(len(members), 2); self.assertTrue(self.R._result_usable(members[0]["seq"], self.events))
        self.events.remove(self.outcomes[members[1]["seq"]]); self.assertFalse(self.R._result_usable(members[0]["seq"], self.events))

    def test_candidate_free_format_and_exact_shared_failure_mapping(self):
        _, rows = self.scientific(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:1], status="format_failed"); seq = rows[0]["seq"]
        row = self.outcomes[seq]; result = next(e for e in self.events if e["kind"] == "attempt_result"); group = next(e for e in self.events if e["kind"] == "batch_done")
        for finish in ("stop", "length"):
            row["status"] = finish; self.assertTrue(self.R._result_usable(seq, self.events))
        row["text"] = "print(0)"; self.assertFalse(self.R._result_usable(seq, self.events)); row["text"] = "not python !"
        row["eligible"] = False; self.assertFalse(self.R._result_usable(seq, self.events)); row["eligible"] = True
        group["shared"]["status"] = "timeout"; self.assertFalse(self.R._result_usable(seq, self.events)); group["shared"]["status"] = "passed"
        row.update(eligible=False, status="interrupted"); group["eligible"] = False
        for status, expected in ((None, "interrupted"), ("interrupted", "interrupted"), ("cancelled", "interrupted"),
                ("parent_error", "interrupted"), ("timeout", "generation_timeout"), ("worker_error", "generation_error")):
            group["shared"]["status"] = status; result["status"] = expected
            with self.subTest(status=status): self.assertTrue(self.R._result_usable(seq, self.events))
            result["status"] = "format_failed"; self.assertFalse(self.R._result_usable(seq, self.events))
        group["shared"]["status"] = "timeout"; result["status"] = "generation_timeout"; group["evidence_errors"] = ["missing telemetry"]
        self.assertFalse(self.R._result_usable(seq, self.events)); group["evidence_errors"] = []
        group["shared"]["status"] = "passed"; result["status"] = "generation_error"
        self.assertFalse(self.R._result_usable(seq, self.events)); group.pop("session_copy_complete")
        self.assertFalse(self.R._result_usable(seq, self.events))

    def test_first_reply_final_all_attempts_and_closed_failed_gates(self):
        session, rows = self.scientific(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:1], status="format_failed"); first = rows[0]
        self.assertTrue(self.R._usable(self.own_cell(first), self.events, first=True)); self.assertFalse(self.R._usable(self.own_cell(first), self.events))
        messages = json.loads(first["request_text"])["messages"] + [dict(role="assistant", content="not python !"), dict(role="user", content=self.L.FORMAT_REPAIR)]
        _, repairs = self.scientific(1, [first["experiment_id"]], messages=messages); last = next(e for e in self.events if e["kind"] == "attempt_result" and e["reservation_seq"] == repairs[0]["seq"])
        self.events.remove(last); self.assertTrue(self.R._usable(self.own_cell(first), self.events, first=True)); self.assertFalse(self.R._usable(self.own_cell(first), self.events))
        self.late(last); self.assertTrue(self.R._usable(self.own_cell(first), self.events))
        next(e for e in self.events if e["kind"] == "batch_done" and e["session_seq"] == session["seq"])["session_copy_complete"] = False
        self.assertFalse(self.R._usable(self.own_cell(first), self.events, first=True)); self.assertFalse(self.R._usable(self.own_cell(first), self.events))
        next(e for e in self.events if e["kind"] == "batch_done" and e["session_seq"] == session["seq"])["session_copy_complete"] = True
        cell = self.own_cell(first); first["attempt"]["reply_index"] = 1
        self.assertFalse(self.R._usable(cell, self.events, first=True))
        self.restore_science((self.baseline, {})); record = self.built["records"][0]; task = next(t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"])
        self.assertFalse(self.R._usable(next(c for c in self.L.cell_states(self.events, record) if c["id"] == task["id"]), self.events))
        op = task["sha256"][:12]; self.emit("native_intent", operation_id=op, task_sha256=task["sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
        verdict = self.emit("native_verdict", operation_id=op, status="failed"); closed = self.emit("native_closed", operation_id=op)
        for status in ("failed", "error", None):
            verdict["status"] = status; cell = next(c for c in self.L.cell_states(self.events, record) if c["id"] == task["id"])
            self.assertEqual(self.R._usable(cell, self.events), status == "failed")
            self.assertEqual(self.R._usable(cell, self.events, first=True), status == "failed")
        verdict["status"] = "failed"; self.events.remove(closed)
        self.assertFalse(self.R._usable(next(c for c in self.L.cell_states(self.events, record) if c["id"] == task["id"]), self.events))
        self.events.append(closed); self.events.remove(verdict)
        self.assertFalse(self.R._usable(next(c for c in self.L.cell_states(self.events, record) if c["id"] == task["id"]), self.events))
        self.restore_science((self.baseline, {})); next(e for e in reversed(self.events) if e["kind"] in ("initial", "limits"))["limits"] = {s: dict.fromkeys(self.J.UNITS, 1) for s in ("actor", "synthetic")}
        cell = next(c for c in self.L.cell_states(self.events, record) if c["id"] == task["id"])
        self.assertEqual(cell["reason"], "campaign_budget"); self.assertFalse(self.R._usable(cell, self.events))

    def test_real_tuning160_confirmation_contract_and_late_cutoff_support(self):
        rows = self.tuning_complete(); decision = self.confirm(); fact = self.facts(); self.assertEqual(fact["invalid_reasons"], [])
        tuning = {self.R.PR._key(t) for t in self.R.PR._layout()[3]}
        cells = [c for member, record, _ in fact["rows"] if self.R.PR._key(member["task"]) in tuning for c in self.L.cell_states(self.events, record) if self.R.PR._key(c) == self.R.PR._key(member["task"])]
        self.assertEqual((len(cells), sum(bool(c["attempt_seqs"]) for c in cells)), (160, 4)); self.assertTrue(all(self.R._usable(c, self.events) for c in cells))
        result = self.confirmation(fact); self.assertEqual(result, dict(decision=decision, prefix_valid=True, prior_exposed_tasks=[], invalid_reasons=[])); self.assertIs(type(result["prefix_valid"]), bool)
        self.assertEqual(sum(c["status"] == "not_started" for _, _, cs in fact["rows"] for c in cs), 128)
        saved = copy.deepcopy((self.events, self.ready)); seq = rows[0]["seq"]
        for kind in ("generation_done", "batch_done", "attempt_result", "native_verdict", "native_closed"):
            self.restore_science(saved); event = next(e for e in self.events if e["kind"] == kind and (e.get("reservation_seq") == seq or e.get("session_seq") == rows[0]["session_seq"] or e.get("operation_id") == f"{seq:012x}"))
            self.late(event); observed = self.confirmation()
            with self.subTest(kind=kind):
                self.assertFalse(observed["prefix_valid"]); self.assertIn("confirmation_premature", {r["kind"] for r in observed["invalid_reasons"]})
        for kind in ("native_verdict", "native_closed"):
            self.restore_science(saved); gate = next(e["operation_id"] for e in self.events if e["kind"] == "native_intent" and e["reference_indices"] is not None and e["task_sha256"] == rows[0]["attempt"]["task_sha256"])
            event = next(e for e in self.events if e["kind"] == kind and e["operation_id"] == gate); event["seq"] = rows[0]["seq"] + 0.5
            self.assertFalse(self.R._result_usable(seq, self.events))

    def test_confirmation_prechecks_all_group_members_and_predecessor_results(self):
        rows = self.tuning_complete(); self.confirm(); fact = self.facts(); saved = copy.deepcopy((self.events, self.ready))
        seq = rows[0]["seq"]; batch = next(e for e in self.events if e["kind"] == "batch_reserved" and e["seq"] == rows[0]["session_seq"])
        # Defensive incomplete accepted projection: no claim S admits this journal.
        batch["rows"].append(dict(reservation_seq=999999, row_id="missing", request_sha256="0" * 64))
        with mock.patch.object(self.R, "_requests", wraps=self.R._requests) as requests:
            self.assertFalse(self.confirmation(fact)["prefix_valid"]); requests.assert_not_called()
        self.restore_science((self.baseline, {})); first_session, first_rows = self.scientific(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:1], status="format_failed")
        first = first_rows[0]; self.step(0, first_session["seq"])
        messages = json.loads(first["request_text"])["messages"] + [dict(role="assistant", content="not python !"), dict(role="user", content=self.L.FORMAT_REPAIR)]
        self.scientific(1, [first["experiment_id"]], messages=messages); self.confirm()
        self.late(next(e for e in self.events if e["kind"] == "attempt_result" and e["reservation_seq"] == first["seq"]))
        fact = self.facts(); self.assertIn("repair_history_mismatch", {r["kind"] for r in fact["invalid_reasons"]})
        with mock.patch.object(self.R, "_requests", wraps=self.R._requests) as requests:
            self.assertFalse(self.confirmation(fact)["prefix_valid"]); requests.assert_not_called()
        self.restore_science(saved); self.ready[rows[-1]["session_seq"]]["fixtures"][0]["prompt_ids"] = [2]
        self.assertFalse(self.confirmation()["prefix_valid"])

    def test_confirmation_absent_duplicate_local_structure_and_ready_cutoff(self):
        rows = self.tuning_complete(); fact = self.facts(); absent = self.confirmation(fact)
        self.assertEqual(absent, dict(decision=None, prefix_valid=False, prior_exposed_tasks=[], invalid_reasons=[]))
        self.confirm(); saved = copy.deepcopy((self.events, self.ready)); self.confirm()
        self.assertIsNone(self.confirmation()["decision"]); self.assertFalse(self.confirmation()["prefix_valid"])
        self.restore_science(saved); fact = self.facts(); original = list(fact["invalid_reasons"]); fact["invalid_reasons"].append(dict(kind="later_violation"))
        result = self.confirmation(fact); self.assertTrue(result["prefix_valid"]); self.assertEqual(result["invalid_reasons"], fact["invalid_reasons"]); self.assertIsNot(result["invalid_reasons"], fact["invalid_reasons"])
        self.assertEqual(original, []); self.assertEqual(set(result), {"decision", "prefix_valid", "prior_exposed_tasks", "invalid_reasons"})
        self.restore_science(saved); decision = next(e for e in self.events if e["kind"] == "wave_confirmation"); decision["seq"] = next(e["seq"] for e in self.events if e["kind"] == "wave_plan")
        self.assertFalse(self.confirmation()["prefix_valid"])
        self.restore_science(saved); fresh_op = next(o for o in self.built["plan"]["opportunities"] if o["block"].endswith("fresh"))
        late_session, _ = self.scientific(fresh_op["id"], fresh_op["experiment_ids"][:1]); fact = self.facts()
        with mock.patch.object(self.R, "_requests", wraps=self.R._requests) as requests:
            self.assertTrue(self.confirmation(fact)["prefix_valid"]); self.assertNotIn(late_session["seq"], requests.call_args.args[5])
            self.assertEqual(set(requests.call_args.args[5]), set(self.ready) - {late_session["seq"]})
        self.restore_science(saved); step = next(e for e in self.events if e["kind"] == "wave_step" and e["payload"]["session_seq"] is not None)
        step["seq"] = rows[0]["seq"] + 0.5; self.assertFalse(self.confirmation()["prefix_valid"])
        self.restore_science(saved); plan = next(e for e in self.events if e["kind"] == "wave_plan")
        native = self.emit("native_intent", operation_id="e" * 12, task_sha256=rows[0]["attempt"]["task_sha256"], reference_indices=[0], contract_sha256=self.built["plan"]["contract_sha256"])
        native["seq"] = plan["seq"] - 0.5
        self.assertFalse(self.confirmation()["prefix_valid"])

    def test_fresh_native_ownership_exposure_and_both_superseders(self):
        self.tuning_complete(); self.confirm(); saved = copy.deepcopy((self.events, self.ready)); fresh = self.R.PR._layout()[4][0]
        member = next(m for m in self.built["plan"]["records"] if m["task"]["id"] == fresh["id"]); own = member["experiment_id"]
        for owner in (own, "foreign-cold"):
            self.restore_science(saved); decision = next(e for e in self.events if e["kind"] == "wave_confirmation")
            reservation = self.emit("generation_reserved", experiment_id=owner, source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt={**member["task"], "reply_index": 0})
            reservation["seq"] = decision["seq"] - 0.5
            native = self.emit("native_intent", operation_id="f" * 12, reservation_seq=reservation["seq"], task_sha256=fresh["sha256"], reference_indices=None, contract_sha256=self.built["plan"]["contract_sha256"])
            native["seq"] = decision["seq"] - 0.25
            # Minimal global reservation projections exercise the post-S helper seam.
            fact = self.facts() if owner != own else None
            if fact is None:
                self.restore_science(saved); fact = self.facts(); self.events.extend([reservation, native])
            observed = self.confirmation(fact); kinds = {r["kind"] for r in observed["invalid_reasons"]}
            self.assertEqual("fresh_native_before_confirmation" in kinds, owner == own); self.assertEqual("fresh_before_confirmation" in kinds, owner == own)
            self.assertEqual(observed["prior_exposed_tasks"], [member["task"]]); self.assertEqual(observed["prefix_valid"], owner != own)
            self.events.remove(native); kinds = {r["kind"] for r in self.confirmation(fact)["invalid_reasons"]}
            self.assertEqual("fresh_before_confirmation" in kinds, owner == own); self.assertNotIn("fresh_native_before_confirmation", kinds)
        for inherited in (True, False):
            self.restore_science(saved); fact = self.facts(); decision = fact["confirmations"][0]
            native = self.emit("native_intent", operation_id="e" * 12, task_sha256=fresh["sha256"], reference_indices=[0], contract_sha256=self.built["plan"]["contract_sha256"])
            native["seq"] = (min(e["seq"] for e in self.events if e["kind"] == "experiment") - 0.5) if inherited else decision["seq"] - 0.5
            observed = self.confirmation(fact); self.assertEqual(observed["prefix_valid"], inherited); self.assertEqual(observed["prior_exposed_tasks"], [])
        for setup_kind in ("collection_setup", "wave_setup"):
            for target in (self.built["plan"]["records"][0]["task"], member["task"]):
                self.restore_science(saved); self.supersede(setup_kind); fact = self.facts()
                native = self.emit("native_intent", operation_id="d" * 12, task_sha256=target["task_sha256"], reference_indices=[0], contract_sha256=self.built["plan"]["contract_sha256"])
                observed = self.confirmation(fact); self.assertTrue(observed["prefix_valid"]); self.assertIn("native_after_supersession", {r["kind"] for r in observed["invalid_reasons"]})
                native["contract_sha256"] = "0" * 64; self.assertNotIn("native_after_supersession", {r["kind"] for r in self.confirmation(fact)["invalid_reasons"]})

    def test_global_exposure_exact_keys_deduplicate_sort_and_decision_cutoffs(self):
        self.tuning_complete(); self.confirm(); fact = self.facts(); decision = fact["confirmations"][0]
        fresh = [{k: t["sha256"] if k == "task_sha256" else t[k] for k in ("split", "id", "task_sha256")} for t in self.R.PR._layout()[4][:2]]
        # Global accepted reservation projections need no UID or resident session.
        for index, task in enumerate((fresh[1], fresh[0], fresh[1])):
            event = self.emit("generation_reserved", experiment_id="older-cold", source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt={**task, "reply_index": 0})
            event["seq"] = decision["seq"] - 0.9 + index / 10
        changed = {**fresh[0], "task_sha256": "0" * 64}
        self.emit("generation_reserved", experiment_id="older-cold", source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt={**changed, "reply_index": 0})["seq"] = decision["seq"] - 0.25
        for field, value in (("split", "other"), ("id", 999999)):
            self.emit("generation_reserved", experiment_id="older-cold", source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt={**fresh[0], field: value, "reply_index": 0})["seq"] = decision["seq"] - (0.1 if field == "id" else 0.2)
        observed = self.confirmation(fact); expected = sorted(fresh, key=self.R.PR._key)
        self.assertTrue(observed["prefix_valid"]); self.assertEqual(observed["prior_exposed_tasks"], expected)
        later = self.R.PR._layout()[4][2]; key = dict(split=later["split"], id=later["id"], task_sha256=later["sha256"])
        self.emit("generation_reserved", experiment_id="superseded-cold", source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt={**key, "reply_index": 0})
        self.assertEqual(self.confirmation(fact)["prior_exposed_tasks"], expected)
        fact["confirmations"] = []; self.assertEqual(self.confirmation(fact)["prior_exposed_tasks"], sorted(fresh + [key], key=self.R.PR._key))
        fact["confirmations"] = [decision, decision]; self.assertIsNone(self.confirmation(fact)["decision"])
        self.assertEqual(self.confirmation(fact)["prior_exposed_tasks"], sorted(fresh + [key], key=self.R.PR._key))

    def test_supersession_decisions_reservations_and_candidate_ownership(self):
        rows = self.tuning_complete(); self.confirm(); saved = copy.deepcopy((self.events, self.ready))
        fresh_op = next(o for o in self.built["plan"]["opportunities"] if o["block"].endswith("fresh"))
        for kind in ("collection_setup", "wave_setup"):
            self.restore_science(saved); self.supersede(kind)
            self.scientific(fresh_op["id"], fresh_op["experiment_ids"][:1]); observed = self.confirmation()
            self.assertTrue(observed["prefix_valid"]); self.assertIn("work_after_supersession", {r["kind"] for r in observed["invalid_reasons"]})
            self.restore_science(saved); self.supersede(kind)
            self.late(next(e for e in self.events if e["kind"] == "wave_confirmation")); observed = self.confirmation()
            self.assertFalse(observed["prefix_valid"]); self.assertIn("confirmation_after_supersession", {r["kind"] for r in observed["invalid_reasons"]})
            for owned in (True, False):
                self.restore_science(saved); self.supersede(kind); fact = self.facts(); reservation = rows[0]
                if not owned:
                    reservation = self.emit("generation_reserved", experiment_id="foreign", source="actor", request_text=self.J.encoded(dict(max_tokens=1, wall_time_seconds=1)), attempt=copy.deepcopy(rows[0]["attempt"]))
                self.emit("native_intent", operation_id="c" * 12, reservation_seq=reservation["seq"], task_sha256=rows[0]["attempt"]["task_sha256"], reference_indices=None, contract_sha256=self.built["plan"]["contract_sha256"])
                observed = self.confirmation(fact); self.assertTrue(observed["prefix_valid"])
                self.assertEqual("native_after_supersession" in {r["kind"] for r in observed["invalid_reasons"]}, owned)

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
