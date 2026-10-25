import copy
import importlib
import json
from types import SimpleNamespace
import unittest
from unittest import mock
from tests import test_paired_wave_collection as wave, test_paired_report as paired, test_batch_report as batch, test_loop_wave as loop_wave


class PairedWaveReportTests(unittest.TestCase):
    emit, anchor = paired.PairedReportTests.emit, paired.PairedReportTests.anchor
    @classmethod
    def setUpClass(cls):
        cls.generate_batch = staticmethod(importlib.import_module("src.data-pipeline.generation_journal").Campaign.generate_batch)
        cls.native_run = staticmethod(importlib.import_module("src.data-pipeline.native_recovery").run_native)
        cls.prepare_cell = staticmethod(importlib.import_module("src.data-pipeline.experiment_loop").prepare_cell)
        cls.settle_cell = staticmethod(importlib.import_module("src.data-pipeline.experiment_loop").settle_cell)
        cls.load_task = staticmethod(importlib.import_module("src.data-pipeline.study_tasks").load_task)
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
        reservations = {e["seq"]: e for e in self.events if e["kind"] == "generation_reserved"}; runs = []
        for name, event in records.items():
            sources = []
            for source in event["record"]["recipes"]:
                own = [r for r in reservations.values() if (r["experiment_id"], r["source"]) == (name, source)]; requests = [json.loads(r["request_text"]) for r in own]
                sources.append(dict(source=source, reserved=dict(zip(self.J.UNITS, (len(own), sum(r["max_tokens"] for r in requests), sum(r["wall_time_seconds"] for r in requests)))),
                    observed=dict(observed_generation_tokens=sum(self.outcomes.get(r["seq"], {}).get("observed_generation_tokens", 0) for r in own),
                        generation_session_seqs=sorted({r["session_seq"] for r in own if "session_seq" in r}), shared_wall_allocation="unallocated")))
            runs.append(dict(experiment_id=name, cells=self.L.cell_states(self.events, event["record"]), sources=sources))
        return dict(events=self.events, records=records, runs=runs, outcomes=self.outcomes, reservations=reservations, panel=self.built["panel"], ready_by_session=self.ready)

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
        intent["job_sha256"] = self.S.digest(dict(session_seq=intent["seq"]))
        group.update(session_copy_complete=True, evidence_errors=[], job_sha256=intent["job_sha256"])
        group["shared"].update(status="passed", elapsed_seconds=239.5, work_seconds=239, cleanup_seconds=.5)
        for row in rows:
            done = self.outcomes[row["seq"]]; result = next(e for e in self.events if e["kind"] == "attempt_result" and e["reservation_seq"] == row["seq"])
            done["observed_generation_tokens"] = 2 if row["row_index"] == 0 else 1
            selected_status = status[row["experiment_id"]] if type(status) is dict else status
            if selected_status == "format_failed": continue
            program = self.L.generation.extract_program("print(0)"); candidate = self.S.sha256(program).hexdigest(); op = f"{row['seq']:012x}"
            done["text"] = program.decode(); result.update(status=selected_status, candidate_sha256=candidate, operation_id=op, sample_id=0)
            self.emit("native_intent", operation_id=op, reservation_seq=row["seq"], task_sha256=row["attempt"]["task_sha256"], reference_indices=None,
                contract_sha256=self.built["plan"]["contract_sha256"], program_sha256=[candidate])
            native_status = {"verified": "passed", "verification_failed": "failed", "verification_error": "error"}[selected_status]
            self.emit("native_verdict", operation_id=op, status=native_status, results=[dict(sample_id=0, status=native_status,
                key=dict(program_sha256=candidate, task_sha256=row["attempt"]["task_sha256"], contract_sha256=self.built["plan"]["contract_sha256"]))])
            self.emit("native_closed", operation_id=op)
        return intent, rows

    def own_cell(self, row):
        record = next(r for r in self.built["records"] if r["experiment_id"] == row["experiment_id"])
        return next(c for c in self.L.cell_states(self.events, record) if c["id"] == row["attempt"]["id"] and c["source"] == row["source"])

    def fail_gates(self, tasks):
        for task in tasks:
            op = task["sha256"][:12]
            self.emit("native_intent", operation_id=op, task_sha256=task["sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
            self.emit("native_verdict", operation_id=op, status="failed"); self.emit("native_closed", operation_id=op)

    def tuning_complete(self, modes=None):
        tuning = self.R.PR._layout()[3]; modes = {tuning[0]["id"]: {}} if modes is None else modes; generated = []
        self.fail_gates([t for t in tuning if t["id"] not in modes])
        members = {m["experiment_id"]: m for m in self.built["plan"]["records"] if m["task"]["id"] in modes}
        for opportunity in self.built["plan"]["opportunities"]:
            if opportunity["block"].endswith("fresh"): break
            limits = {n: modes[members[n]["task"]["id"]].get((opportunity["bits"], opportunity["source"]), 0) for n in opportunity["experiment_ids"] if n in members}
            names = [n for n, limit in limits.items() if limit is None or opportunity["reply_index"] <= limit]
            if names:
                status = {n: "format_failed" if limits[n] is None or opportunity["reply_index"] < limits[n] else "verified" for n in names}; messages = None
                if opportunity["reply_index"]:
                    first = next(r for r in generated if r["experiment_id"] == names[0] and r["source"] == opportunity["source"])
                    messages = json.loads(first["request_text"])["messages"] + [dict(role="assistant", content="not python !"), dict(role="user", content=self.L.FORMAT_REPAIR)]
                session, rows = self.scientific(opportunity["id"], names, status, messages); generated.extend(rows)
                self.step(opportunity["id"], session["seq"])
            else: self.step(opportunity["id"], None)
        return generated

    def own_eligible(self):
        records = {e["experiment_id"]: e for e in self.events if e["kind"] == "experiment"}; reservations = {e["seq"]: e for e in self.events if e["kind"] == "generation_reserved"}
        return [dict(**{k: v for k, v in reservations[e["reservation_seq"]]["attempt"].items()}, experiment_id=reservations[e["reservation_seq"]]["experiment_id"],
            source=reservations[e["reservation_seq"]]["source"], reservation_seq=e["reservation_seq"], session_seq=reservations[e["reservation_seq"]]["session_seq"],
            record_sha256=records[reservations[e["reservation_seq"]]["experiment_id"]]["record_sha256"]) for e in self.events if e["kind"] == "attempt_result" and e["status"] == "verified"]

    def trial(self):
        args = self.inputs(); sessions = self.S._session_costs(self.events, args["reservations"], self.outcomes)
        return self.R.derive(**args, eligible=self.own_eligible(), sessions=sessions)[0]

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

    def test_facts_scientific_contract_and_pending_setup(self):
        facts = self.facts(); self.assertEqual(set(facts), {"setup", "plan", "rows", "session_opportunities", "steps", "confirmations", "invalid_reasons"})
        self.assertEqual((len(facts["rows"]), facts["invalid_reasons"]), (144, []))
        trial = self.R.derive(**self.inputs(), eligible=[], sessions=[])[0]
        self.assertEqual(set(trial), set("report_scope plan_id plan_sha256 comparison_valid complete confirmation_seq fresh_confirmation_valid prior_exposed_tasks invalid_reasons progress metrics costs".split()))
        self.assertEqual((trial["comparison_valid"], trial["complete"], trial["confirmation_seq"], trial["fresh_confirmation_valid"]), (True, False, None, False))
        self.assertEqual(trial["progress"], dict(planned_records=144, planned_cells=288, terminal_cells=0, status_counts=dict(not_started=288), reason_counts={}))
        self.assertEqual([(m["name"], m["denominator"]) for m in trial["metrics"]], [("primary50", 50), ("tuning40", 40), ("fresh32", 32), ("extra22", 22), ("full72", 72)])
        self.assertEqual(trial["metrics"][0]["target_met"], [dict(bits=b, first_reply=None, final_policy=None) for b in (8, 16)])
        for subset in trial["metrics"]:
            self.assertEqual(set(subset), set("name denominator terminal_cells complete arms joint_contingency".split()) | ({"target_met"} if subset["name"] == "primary50" else set()))
            self.assertEqual(subset["joint_contingency"]["final_policy"]["unassessed"], subset["denominator"])
        self.events = [e for e in self.events if e["kind"] != "wave_plan"]
        self.assertEqual(self.R.derive(**self.inputs(), eligible=[], sessions=[]), [])

    def test_own_joint_first_final_primary200_and_complete288(self):
        layout = self.R.PR._layout(); targets = layout[3][:16]; modes = {t["id"]: {} for t in targets}
        for task in targets[10:15]: modes[task["id"]] = {(8, "synthetic"): 1}
        modes[targets[15]["id"]] = {(8, "synthetic"): None, (16, "actor"): None}
        self.tuning_complete(modes); self.confirm(); panel = {self.R.PR._key(t) for t in layout[2]["tasks"]}
        partial = self.trial(); self.assertEqual((partial["comparison_valid"], partial["metrics"][0]["terminal_cells"]), (True, 160))
        self.assertEqual(partial["metrics"][0]["target_met"], [dict(bits=b, first_reply=None, final_policy=None) for b in (8, 16)])
        primary = [t for t in layout[4] if self.R.PR._key(t) in panel]; extra = [t for t in layout[4] if self.R.PR._key(t) not in panel]
        self.assertEqual((len(primary), len(extra)), (10, 22)); self.fail_gates(primary); trial = self.trial(); metric = trial["metrics"][0]
        self.assertEqual((trial["progress"]["terminal_cells"], metric["terminal_cells"], trial["complete"], trial["fresh_confirmation_valid"]), (200, 200, False, True))
        self.assertEqual(metric["arms"], [dict(bits=8, first_verified=dict(actor=16, synthetic=10, joint=10), final_verified=dict(actor=16, synthetic=15, joint=15)),
            dict(bits=16, first_verified=dict(actor=15, synthetic=16, joint=15), final_verified=dict(actor=15, synthetic=16, joint=15))])
        self.assertEqual(metric["joint_contingency"], dict(first_reply={"both": 10, "8_only": 0, "16_only": 5, "neither": 35, "unassessed": 0},
            final_policy={"both": 15, "8_only": 0, "16_only": 0, "neither": 35, "unassessed": 0}))
        targets_expected = [dict(bits=8, first_reply=False, final_policy=True), dict(bits=16, first_reply=True, final_policy=True)]
        self.assertEqual(metric["target_met"], targets_expected); self.fail_gates(extra); self.assertTrue(self.trial()["complete"])
        for task, expected in ((primary[0], None), (extra[0], targets_expected)):
            closed = next(e for e in self.events if e["kind"] == "native_closed" and e["operation_id"] == task["sha256"][:12]); self.events.remove(closed)
            observed = self.trial(); self.assertFalse(observed["complete"])
            self.assertEqual(observed["metrics"][0]["target_met"], [dict(bits=b, first_reply=None, final_policy=None) for b in (8, 16)] if expected is None else expected)
            self.events.append(closed)
        self.supersede("collection_setup"); self.emit("native_intent", operation_id="b" * 12, task_sha256=targets[0]["sha256"], reference_indices=[0], contract_sha256=self.built["plan"]["contract_sha256"])
        observed = self.trial(); self.assertFalse(observed["comparison_valid"]); self.assertTrue(observed["fresh_confirmation_valid"]); self.assertFalse(observed["complete"])
        self.assertEqual(observed["metrics"][0]["arms"], metric["arms"]); self.assertEqual(observed["invalid_reasons"], [dict(kind="native_after_supersession")])
        self.assertTrue(all(t["first_reply"] is None and t["final_policy"] is None for t in observed["metrics"][0]["target_met"]))

    def test_metrics_exact_own_proof_identity_mapping_and_unassessed_repair(self):
        rows = self.tuning_complete(); args = self.inputs(); fact = self.facts(); eligible = self.own_eligible()
        def metrics(items, current=fact):
            return self.R._metrics(current, layout=self.R.PR._layout(), events=self.events, eligible=items, reservations=args["reservations"], comparison_valid=True)[0]
        baseline = metrics(eligible); self.assertEqual(baseline["arms"][0]["final_verified"]["actor"], 1)
        for field, value in (("experiment_id", "superseded-peer"), ("record_sha256", "0" * 64), ("task_sha256", "0" * 64), ("id", -1),
                ("source", "other"), ("reservation_seq", -1), ("session_seq", -1), ("reply_index", 1), ("reply_index", True)):
            items = copy.deepcopy(eligible); items[0][field] = value
            with self.subTest(field=field, value=value): self.assertEqual(metrics(items)["arms"][0]["final_verified"]["actor"], 0)
        self.assertTrue(all(a["final_verified"]["joint"] == 0 for a in metrics([{**e, "experiment_id": "superseded-peer"} for e in eligible])["arms"]))
        fact["session_opportunities"][rows[0]["session_seq"]] = None; observed = metrics(eligible)
        self.assertEqual(observed["arms"][0]["final_verified"]["actor"], 0); self.assertEqual(observed["joint_contingency"]["final_policy"]["unassessed"], baseline["joint_contingency"]["final_policy"]["unassessed"] + 1)
        self.restore_science((self.baseline, {})); task = self.R.PR._layout()[3][0]; self.tuning_complete({task["id"]: {(8, "actor"): 1}})
        first = next(e for e in self.events if e["kind"] == "generation_reserved"); pending = next(e for e in self.events if e["kind"] == "attempt_result" and e["status"] == "verified" and next(r for r in self.events if r["kind"] == "generation_reserved" and r["seq"] == e["reservation_seq"])["attempt"]["reply_index"] == 1)
        baseline = self.trial()["metrics"][0]; self.events.remove(pending); observed = self.trial()["metrics"][0]
        self.assertEqual(observed["joint_contingency"]["first_reply"], baseline["joint_contingency"]["first_reply"]); self.assertEqual(observed["joint_contingency"]["final_policy"]["unassessed"], 11)
        self.events.append(pending); next(e for e in self.events if e["kind"] == "batch_done" and e["session_seq"] == first["session_seq"])["session_copy_complete"] = False
        observed = self.trial()["metrics"][0]; self.assertEqual(observed["arms"][0]["final_verified"]["actor"], 1); self.assertEqual(observed["joint_contingency"]["final_policy"]["unassessed"], 11)

    def test_full72_costs_unique_known_unknown_mixed_and_cross_subset_sessions(self):
        layout = self.R.PR._layout(); members = {m["experiment_id"]: m for m in self.built["plan"]["records"]}
        self.tuning_complete({members[n]["task"]["id"]: {} for n in self.built["plan"]["opportunities"][0]["experiment_ids"][:2]}); self.confirm()
        args = self.inputs(); fact = self.facts(); ledger = self.S._session_costs(self.events, args["reservations"], self.outcomes)
        def costs(current=fact, sessions=ledger, reservations=args["reservations"]):
            return self.R._costs(current, runs=args["runs"], reservations=reservations, sessions=sessions)[0]["sources"][0]
        base = costs(); self.assertEqual((base["reserved"]["replies"], base["observed_generation_tokens"], base["known_owned_elapsed_seconds"], base["unknown_elapsed_sessions"]), (2, 3, 239.5, 0))
        self.assertEqual(set(base), set("source reserved observed_generation_tokens generation_session_seqs known_owned_elapsed_seconds unknown_elapsed_sessions".split()))
        for fault in ("unknown", "foreign", "bits", "source", "mapping", "incomplete"):
            # Defensive post-S projections isolate each wall-ownership requirement.
            sessions = copy.deepcopy(ledger); current = copy.deepcopy(fact); reservations = copy.deepcopy(args["reservations"]); session = sessions[0]
            if fault == "unknown": session["owned_elapsed_seconds"] = None
            if fault == "foreign":
                reservations[999999] = {**reservations[session["reservation_seqs"][0]], "seq": 999999, "experiment_id": "foreign"}; session["reservation_seqs"].append(999999)
            if fault in ("bits", "source"): session[fault] = 16 if fault == "bits" else "synthetic"
            if fault == "mapping": current["session_opportunities"][session["session_seq"]] = None
            if fault == "incomplete": session["reservation_seqs"].pop()
            observed = costs(current, sessions, reservations); self.assertEqual(observed["generation_session_seqs"], base["generation_session_seqs"])
            self.assertEqual(observed["reserved"], base["reserved"]); self.assertEqual(observed["known_owned_elapsed_seconds"], 0)
            self.assertEqual(observed["unknown_elapsed_sessions"], int(fault == "unknown"))
        ledger[0]["reported_timing"]["cleanup_seconds"] = 9999; self.assertEqual(costs()["known_owned_elapsed_seconds"], 239.5)
        fresh = next(o for o in self.built["plan"]["opportunities"] if o["block"].endswith("fresh")); self.scientific(fresh["id"])
        trial = self.trial(); source = next(a for a in trial["costs"] if a["bits"] == fresh["bits"])["sources"][0]
        self.assertEqual((source["reserved"]["replies"], source["known_owned_elapsed_seconds"]), (2 + len(fresh["experiment_ids"]), 479))
        panel = {self.R.PR._key(t) for t in layout[2]["tasks"]}; members = {m["experiment_id"]: m for m in self.built["plan"]["records"]}
        self.assertEqual({self.R.PR._key(members[n]["task"]) in panel for n in fresh["experiment_ids"]}, {True, False})
        for subset in trial["metrics"]: self.assertFalse(set(subset) & {"costs", "wall", "timing", "native_rates", "generation_tps"})

    def test_failed_format_repair_charges_and_distinct_unknown_owned_session(self):
        task = self.R.PR._layout()[3][0]; self.tuning_complete({task["id"]: {(8, "actor"): 1}}); trial = self.trial(); cost = trial["costs"][0]["sources"][0]
        self.assertEqual(trial["metrics"][0]["arms"][0]["final_verified"]["actor"], 1)
        self.assertEqual(cost["reserved"], dict(zip(self.J.UNITS, (2, 4096, 480))))
        self.assertEqual((cost["observed_generation_tokens"], len(cost["generation_session_seqs"]), cost["known_owned_elapsed_seconds"]), (4, 2, 479))
        args = self.inputs(); sessions = self.S._session_costs(self.events, args["reservations"], self.outcomes); sessions[1]["owned_elapsed_seconds"] = None
        unknown = self.R._costs(self.facts(), runs=args["runs"], reservations=args["reservations"], sessions=sessions)[0]["sources"][0]
        self.assertEqual((unknown["known_owned_elapsed_seconds"], unknown["unknown_elapsed_sessions"]), (239.5, 1)); self.assertEqual(unknown["reserved"], cost["reserved"])

    def test_nonempty_retained_wave_owned_candidates_session_publication_and_offline_replay(self):
        self.metadata_patch.stop(); b = batch.BatchReportTests(methodName="runTest"); b.setUp(); self.addCleanup(b.doCleanups)
        w = wave.PairedWaveCollectionTests(methodName="runTest"); w.setUp(); self.addCleanup(w.doCleanups)
        # Two invented, actually pinned task files; the other70 remain metadata only.
        loaded = w.loaded
        for record in loaded["cohort"]["tasks"][:4:2]:
            task = dict(copy.deepcopy(b.task), split=record["split"], id=record["id"], difficulty=record["difficulty"])
            raw = (self.J.encoded(task) + "\n").encode(); path = loaded["test_pool_dir"] / record["path"]
            path.parent.mkdir(exist_ok=True); path.write_bytes(raw); record["sha256"] = self.S.sha256(raw).hexdigest()
            w.s.pool_manifest["provenance"]["task_sha256"][record["path"]] = record["sha256"]
        w.loaded = w.s.rewrite(loaded["cohort"])
        # Match admission default7200 to common method7200 in the existing runner.
        w.make_qualification(8, "actor", 8, 7200); c = w.fresh(); contract = batch.native.fixture.CONTRACT
        with mock.patch.object(b.N.verification.runner, "contract", return_value=contract): w.setup(c)
        b.campaign = c
        plan = next(e["payload"] for e in c.events if e["kind"] == "wave_plan"); opportunity = plan["opportunities"][0]
        registered = {e["experiment_id"]: e["record"] for e in c.events if e["kind"] == "experiment"}
        records = [registered[n] for n in opportunity["experiment_ids"][:2]]; method = records[0]["generation_method"]["actor"]
        _, qualification_files = w.H.read_qualification(c.root / method["qualification"]["path"], method["qualification"]["response_sha256"])
        b.b.method = method; b.b.q.pin = method["qualification"]["response_sha256"]
        b.b.q.events = [json.loads(line) for line in qualification_files["stdout.jsonl"].splitlines()]
        proxy = SimpleNamespace(f=b, b=b.b, H=w.H)
        with mock.patch.object(self.R.PR.study, "load_cohort", return_value=w.loaded), \
             mock.patch.object(self.L.study, "load_task", self.load_task), mock.patch.object(b.N.verification.runner, "contract", return_value=contract), \
             mock.patch.object(self.J.Campaign, "generate_batch", self.generate_batch), mock.patch.object(b.N, "run_native", self.native_run), \
             mock.patch.object(self.L, "prepare_cell", self.prepare_cell), mock.patch.object(self.L, "settle_cell", self.settle_cell):
            prepared = [self.L.prepare_cell(c, w.loaded, record, "actor", record["assigned_task_ids"][0]) for record in records]
            outcomes = loop_wave.LoopWaveTests.generate(proxy, c, prepared, texts=("print(1)", "print(1)"))
            states = [self.L.settle_cell(c, w.loaded, record, "actor", record["assigned_task_ids"][0]) for record in records]
            self.assertEqual([s["status"] for s in states], ["verified", "verified"])
            snapshot = self.S.snapshot(c); trials = snapshot["report"]["wave_trials"]; self.assertEqual(len(trials), 1); trial = trials[0]
            self.assertEqual((trial["comparison_valid"], trial["complete"], trial["confirmation_seq"], trial["progress"]["terminal_cells"]), (True, False, None, 2))
            self.assertEqual(trial["metrics"][0]["arms"][0], dict(bits=8, first_verified=dict(actor=2, synthetic=0, joint=0), final_verified=dict(actor=2, synthetic=0, joint=0)))
            cost = trial["costs"][0]["sources"][0]
            self.assertEqual((cost["reserved"]["replies"], cost["observed_generation_tokens"], cost["known_owned_elapsed_seconds"], cost["unknown_elapsed_sessions"]), (2, 3, 239.5, 0))
            self.assertEqual(cost["generation_session_seqs"], [outcomes[0]["session_seq"]]); self.assertEqual(len(snapshot["manifest"]["selections"]), 2)
            self.assertTrue(all((c.root / "requests" / f"{r['reservation_seq']:06d}" / "candidate.py").read_bytes() == b"print(1)" for r in outcomes))
            published = self.S.publish(c, "wave"); self.assertEqual(self.S.publish(c, "wave"), published)
            self.assertEqual(json.loads((self.R.Path(published["path"]) / "report.json").read_bytes())["wave_trials"], trials)
            originals = set(method["admission"]["source_hashes"]); read = self.R.Path.read_bytes
            def offline(path):
                if str(path) in originals: raise AssertionError("Original producer source read")
                return read(path)
            with mock.patch.object(self.L, "policy_sha256", return_value="0" * 64), \
                 mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Original task read")), \
                 mock.patch.object(w.H, "current_source_compatible", side_effect=AssertionError("Live source check")), \
                 mock.patch.object(w.H.G, "prepare_request", side_effect=AssertionError("Renormalization")), \
                 mock.patch.object(self.J.Campaign, "generate_batch", side_effect=AssertionError("Generation")), mock.patch.object(b.N, "run_native", side_effect=AssertionError("Native execution")):
                with mock.patch.object(self.R.Path, "read_bytes", offline): self.assertEqual(self.S.snapshot(c), snapshot)
                # Publication additionally fingerprints current Python sources.
                self.assertEqual(self.S.publish(c, "wave"), published)
            (c.root / "requests" / f"{outcomes[0]['reservation_seq']:06d}" / "candidate.py").write_bytes(b"print(999)")
            with self.assertRaisesRegex(ValueError, "Artifact changed"): self.S.snapshot(c)
            with self.assertRaisesRegex(ValueError, "Artifact changed"): self.S.publish(c, "tampered")

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
