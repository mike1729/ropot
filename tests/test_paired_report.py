import copy
import importlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
from tests import test_precision_adoption as adoption, test_study_tasks as study


class PairedReportTests(unittest.TestCase):
    F = adoption.PrecisionAdoptionTests
    write_task, rewrite, limits, rows = F.write_task, F.rewrite, F.limits, F.rows
    read, execute, replies, recipe, fresh = F.read, F.execute, F.replies, F.recipe, F.fresh
    fixture, put, mode = F.fixture, F.put, F.mode
    feasibility_loaded = study.StudyTaskTests.feasibility_loaded

    def setUp(self):
        self.F.setUp(self)
        self.R = importlib.import_module("src.data-pipeline.paired_report")
        self.C = importlib.import_module("src.data-pipeline.paired_collection")
        self.S = importlib.import_module("src.data-pipeline.study_report")
        self.loaded = self.feasibility_loaded()
        for task in self.loaded["cohort"]["tasks"]: task["sha256"] = self.P.sha256(str(task["id"]).encode()).hexdigest()
        self.loaded["cohort"]["tasks"][0] = copy.deepcopy(self.records[1])
        self.loaded["cohort"]["queues"]["development"]["initial"][0] = 1
        for target, name, value in ((self.C, "RECIPE", self.recipe((1, 2))), (self.R.study, "load_cohort", self.loaded)):
            patch = mock.patch.object(target, name, return_value=value) if name == "load_cohort" else mock.patch.object(target, name, value)
            patch.start(); self.addCleanup(patch.stop)
        self.campaign = self.fresh(); reference = self.P.freeze(self.campaign, self.original / "receipt.json", self.identity)
        profiles = {bits: {s: dict(reference=reference, descriptor=self.P.resolve(self.campaign, reference, s, bits)) for s in study.study.SOURCES} for bits in (8, 16)}
        self.contract = self.N.verification.runner.contract()
        self.built = self.C.build(self.loaded, "trial", profiles, self.N.verification.digest(self.contract))
        self.events, self.outcomes, self.eligible, self.cost_rows = [copy.deepcopy(self.campaign.events[0])], {}, [], {}
        self.anchor("panel", self.built["panel"]); self.install(self.built)

    def emit(self, kind, **payload):
        event = dict(seq=max((e["seq"] for e in self.events), default=0) + 1, kind=kind, **payload)
        self.events.append(event); return event

    def anchor(self, kind, payload): return self.emit(kind, payload=payload, payload_sha256=self.S.digest(payload))

    def install(self, built):
        for record in built["records"]:
            self.emit("experiment", experiment_id=record["experiment_id"], record=record, record_sha256=self.S.digest(record))
            self.emit("execution_context", experiment_id=record["experiment_id"], contract=self.contract, contract_sha256=built["plan"]["contract_sha256"])
        self.emit("collection_setup", payload=dict(plan_id=built["plan"]["plan_id"]))
        return self.anchor("paired_plan", built["plan"])

    def derive(self):
        # Unit maps sit after S's proof boundary; the final test exercises real log validation.
        records = {e["experiment_id"]: e for e in self.events if e["kind"] == "experiment"}
        runs = [dict(experiment_id=name, cells=self.L.cell_states(self.events, event["record"]), sources=[self.cost_rows.get((name, source),
            dict(source=source, reserved=dict.fromkeys(self.J.UNITS, 0), observed=dict(observed_generation_tokens=0, lower_bound_requests=0,
                unknown_response_requests=0, known_elapsed_seconds=0, unknown_elapsed_requests=0))) for source in event["record"]["recipes"]]) for name, event in records.items()]
        return self.R.derive(events=self.events, records=records, runs=runs, eligible=self.eligible, outcomes=self.outcomes,
            reservations={e["seq"]: e for e in self.events if e["kind"] == "generation_reserved"}, panel=self.built["panel"])

    def gate(self, task):
        opid = task["sha256"][:12]
        self.emit("native_intent", operation_id=opid, task_sha256=task["sha256"], reference_indices=[0, 1, 2], contract_sha256=self.built["plan"]["contract_sha256"])
        self.emit("native_verdict", operation_id=opid, status="failed")
        self.emit("native_closed", operation_id=opid)

    def tuning(self):
        for record in self.built["records"][:80:2]:
            self.gate(next(t for t in record["activation"]["tasks"] if t["id"] == record["assigned_task_ids"][0]))

    def request(self, index=0, source="actor", slot=0, result="generation_error", record=None):
        record = record or self.built["records"][index]; task = next(t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"])
        recipe = record["recipes"][source]; messages = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "fixture"}]
        if slot: messages += [{"role": "assistant", "content": "prior"}, {"role": "user", "content": "adaptive repair"}]
        request = self.api.request_for(source, messages, profile=record["profiles"][source]["descriptor"], seed=recipe["seeds"][slot],
            **{k: recipe[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")})
        _, raw = self.api.prepare_request(request)
        reserved = self.emit("generation_reserved", experiment_id=record["experiment_id"], source=source, request_text=raw.decode(), request_sha256=self.P.sha256(raw).hexdigest(),
            attempt=dict(split=task["split"], id=task["id"], task_sha256=task["sha256"], reply_index=slot))
        complete = result in ("format_failed", "verification_failed", "verified")
        done = self.emit("generation_done", reservation_seq=reserved["seq"], status="stop" if complete else "worker_error", profile_guarded=True, profile_failure=None, protocol_error=None)
        self.outcomes[reserved["seq"]] = dict(done, elapsed_seconds=None, response=dict(complete=complete, cleanup_error=None,
            events=[dict(event="ready", prompt="same", prompt_ids=[1], eos_token_ids=[9])] + ([dict(event="chunk", finish_reason="stop", generation_tps=None)] if complete else [])))
        if result: self.emit("attempt_result", reservation_seq=reserved["seq"], status=result, candidate_sha256=None, operation_id=None, sample_id=None, reused=False)
        return reserved

    def test_canonical_matrix_and_otherwise_valid_binding_mutations(self):
        trial = self.derive()[0]
        self.assertEqual((trial["progress"]["planned_records"], trial["progress"]["planned_cells"], trial["progress"]["terminal_cells"]), (144, 288, 0))
        self.assertEqual((trial["members"][0]["task"]["id"], trial["members"][0]["bits"]), (1, 8))
        self.assertEqual(len({t["sha256"] for t in self.built["activation"]["tasks"]}), 72)
        self.assertNotIn("initial_inputs", trial); self.assertNotIn("yield", trial)
        baseline = copy.deepcopy(self.events)
        for fault in ("missing", "duplicate", "bits", "task", "order", "recipe", "receipt", "context", "record_hash"):
            self.events = copy.deepcopy(baseline); anchor = self.events[-1]; members = anchor["payload"]["records"]
            event = next(e for e in self.events if e["kind"] == "experiment")
            if fault == "missing": members.pop()
            elif fault == "duplicate": members[1] = copy.deepcopy(members[0])
            elif fault == "bits": members[0]["bits"] = True
            elif fault == "task": members[0]["task"]["task_sha256"] = "0" * 64
            elif fault == "order": members[0], members[1] = members[1], members[0]
            elif fault == "context": next(e for e in self.events if e["kind"] == "execution_context")["contract_sha256"] = "0" * 64
            elif fault == "record_hash": members[0]["record_sha256"] = "0" * 64
            else:
                if fault == "recipe": event["record"]["recipes"]["actor"]["seeds"][0] = 7
                else: event["record"]["profiles"]["actor"]["descriptor"]["receipt_sha256"] = "0" * 64
                event["record_sha256"] = members[0]["record_sha256"] = self.S.digest(event["record"])
            anchor["payload_sha256"] = self.S.digest(anchor["payload"])
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "Invalid paired plan|Paired member differs"): self.derive()

    def test_own_work_before_plan_remains_a_reportable_invalid_trial(self):
        plan = self.events.pop(); self.request(); self.anchor("paired_plan", plan["payload"])
        self.assertIn("member_before_plan", {r["kind"] for r in self.derive()[0]["invalid_reasons"]})

    def test_prefix_confirmation_requires_durable_generation_and_native_proof(self):
        plan = next(e for e in self.events if e["kind"] == "paired_plan")
        self.tuning(); self.request(); confirmation = self.anchor("paired_confirmation", dict(plan_id="trial", plan_sha256=plan["payload_sha256"], decision="unchanged"))
        self.assertTrue(self.derive()[0]["comparison_valid"])
        baseline = copy.deepcopy(self.events)
        for fault in ("generation_done", "native_verdict", "native_closed", "unstarted", "native_error", "budget", "profile", "duplicate"):
            self.events = copy.deepcopy(baseline)
            if fault == "duplicate": self.anchor("paired_confirmation", copy.deepcopy(confirmation["payload"]))
            elif fault == "profile": next(e for e in self.events if e["kind"] == "generation_done")["profile_guarded"] = False
            elif fault in ("unstarted", "native_error", "budget"):
                opid = next(e["operation_id"] for e in self.events if e["kind"] == "native_intent")
                if fault == "budget":
                    next(e for e in self.events if e["kind"] == "native_verdict" and e["operation_id"] == opid)["status"] = "passed"
                    self.events[0]["limits"]["actor"]["replies"] = 1
                    self.events = [e for e in self.events if e["kind"] != "paired_confirmation"]
                    self.request(source="synthetic"); self.request(1, "synthetic")
                    self.anchor("paired_confirmation", copy.deepcopy(confirmation["payload"]))
                else: self.events = [e for e in self.events if not (e.get("operation_id") == opid and (fault == "unstarted" or e["kind"] == "native_verdict"))]
            else:
                event = next(e for e in self.events if e["kind"] == fault); self.events.remove(event)
                self.emit(event["kind"], **{k: v for k, v in event.items() if k not in ("kind", "seq")})
            trial = self.derive()[0]
            self.assertFalse(trial["comparison_valid"]); self.assertFalse(trial["complete"])
            self.assertIn("confirmation_duplicate" if fault == "duplicate" else "confirmation_premature", {r["kind"] for r in trial["invalid_reasons"]})

    def test_confirmation_hash_and_foreign_plan_are_malformed(self):
        for change in (dict(plan_sha256="0" * 64), dict(plan_id="foreign")):
            event = self.anchor("paired_confirmation", {**dict(plan_id="trial", plan_sha256=self.events[-1]["payload_sha256"], decision="unchanged"), **change})
            with self.assertRaisesRegex(ValueError, "Invalid paired confirmation"): self.derive()
            self.events.remove(event)

    def test_verified_result_cannot_borrow_candidate_closure_after_confirmation(self):
        self.tuning(); next(e for e in self.events if e["kind"] == "native_verdict")["status"] = "passed"
        reserved = self.request(result=None); opid = "f" * 12
        self.emit("native_intent", operation_id=opid, task_sha256=reserved["attempt"]["task_sha256"], reference_indices=None,
            reservation_seq=reserved["seq"], contract_sha256=self.built["plan"]["contract_sha256"])
        self.emit("native_verdict", operation_id=opid, status="passed")
        closed = self.emit("native_closed", operation_id=opid)
        self.emit("attempt_result", reservation_seq=reserved["seq"], status="verified", candidate_sha256="c" * 64, operation_id=opid, sample_id=0, reused=False)
        self.request(source="synthetic"); self.request(1); self.request(1, "synthetic")
        self.anchor("paired_confirmation", dict(plan_id="trial", plan_sha256=next(e for e in self.events if e["kind"] == "paired_plan")["payload_sha256"], decision="unchanged"))
        self.assertTrue(self.derive()[0]["comparison_valid"])
        self.events.remove(closed)
        self.emit("native_closed", operation_id=opid)
        self.assertIn("confirmation_premature", {r["kind"] for r in self.derive()[0]["invalid_reasons"]})

    def test_fresh_exposure_cutoff_and_member_scoped_later_plan(self):
        self.tuning(); original = copy.deepcopy(self.built["records"][80]); original["experiment_id"] = "foreign"
        self.emit("experiment", experiment_id="foreign", record=original, record_sha256=self.S.digest(original))
        self.emit("execution_context", experiment_id="foreign", contract=self.contract, contract_sha256=self.built["plan"]["contract_sha256"])
        reserved = self.request(record=original)
        plan = next(e for e in self.events if e["kind"] == "paired_plan")
        self.anchor("paired_confirmation", dict(plan_id="trial", plan_sha256=plan["payload_sha256"], decision="unchanged"))
        for record in self.built["records"][80::2]: self.gate(next(t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"]))
        trial = self.derive()[0]; self.assertTrue(trial["complete"]); self.assertFalse(trial["fresh_confirmation_valid"])
        self.assertEqual([t["id"] for t in trial["prior_exposed_tasks"]], original["assigned_task_ids"])
        self.events = [e for e in self.events if e["seq"] != reserved["seq"] and e.get("reservation_seq") != reserved["seq"]]
        self.outcomes.pop(reserved["seq"])
        self.assertTrue(self.derive()[0]["fresh_confirmation_valid"])
        later = copy.deepcopy(self.built); later["plan"]["plan_id"] = "later"
        original_events = copy.deepcopy(self.events); self.anchor("paired_plan", later["plan"])
        with self.assertRaisesRegex(ValueError, "Paired member differs"): self.derive()
        self.events = original_events
        for record, member in zip(later["records"], later["plan"]["records"]):
            record["experiment_id"] += "-later"; member.update(experiment_id=record["experiment_id"], record_sha256=self.S.digest(record))
        self.install(later); self.assertEqual(len(self.derive()), 2)

    def test_every_request_binding_initial_equality_and_adaptive_repairs(self):
        first = self.request(result="format_failed"); self.request(1); self.request(slot=1)
        self.assertTrue(self.derive()[0]["comparison_valid"]); baseline = copy.deepcopy(self.events)
        for field, value in (("seed", 7), ("temperature", True), ("top_p", 0.8), ("max_tokens", 1.0), ("wall_time_seconds", 1.0), ("messages", [{"role": "user", "content": "different"}])):
            self.events = copy.deepcopy(baseline); event = next(e for e in self.events if e["seq"] == first["seq"])
            request = json.loads(event["request_text"]); request[field] = value; event["request_text"] = self.J.encoded(request) + "\n"
            event["request_sha256"] = self.P.sha256(event["request_text"].encode()).hexdigest()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Paired request differs|Initial requests differ"): self.derive()
        for field, value, reason in (("reply_index", 2, "Invalid reply slots"), ("task_sha256", "0" * 64, "Attempt binding differs")):
            self.events = copy.deepcopy(baseline); next(e for e in self.events if e["seq"] == first["seq"])["attempt"][field] = value
            with self.assertRaisesRegex(ValueError, reason): self.derive()
        self.events = copy.deepcopy(baseline); next(e for e in self.events if e["seq"] == first["seq"])["source"] = "foreign"
        with self.assertRaisesRegex(ValueError, "Invalid attempt ownership"): self.derive()

    def test_observed_controls_ready_order_and_fresh_dispatch_remain_reportable(self):
        clean = copy.deepcopy(self.events)
        first = self.request(1); self.request(0); trial = self.derive()[0]
        self.assertIn("observed_order_mismatch", {r["kind"] for r in trial["invalid_reasons"]})
        self.events = copy.deepcopy(clean); self.outcomes = {}; self.request(0); first = self.request(1)
        baseline = copy.deepcopy(self.outcomes); self.assertTrue(self.derive()[0]["comparison_valid"])
        self.outcomes[first["seq"]]["response"]["events"][0]["prompt_ids"] = [2]
        self.assertIn("initial_ready_mismatch", {r["kind"] for r in self.derive()[0]["invalid_reasons"]})
        self.outcomes[first["seq"]]["response"]["events"] = []
        self.assertNotIn("initial_ready_mismatch", {r["kind"] for r in self.derive()[0]["invalid_reasons"]})
        done = next(e for e in self.events if e["kind"] == "generation_done")
        for guarded, failure, error, reason in ((False, None, None, "profile_unproven"), (True, dict(code="runtime", detail="fixture"), None, "profile_failure"), (True, None, "Events after done", "profile_protocol")):
            self.outcomes = copy.deepcopy(baseline)
            done.update(profile_guarded=guarded, profile_failure=failure, protocol_error=error)
            self.outcomes[first["seq"]].update({k: done[k] for k in ("profile_guarded", "profile_failure", "protocol_error")})
            self.assertIn(reason, {r["kind"] for r in self.derive()[0]["invalid_reasons"]})
        self.events = clean; self.outcomes = {}; self.request(0)
        self.assertTrue(self.derive()[0]["comparison_valid"])
        self.request(80)
        self.assertIn("fresh_before_confirmation", {r["kind"] for r in self.derive()[0]["invalid_reasons"]})

    def test_superseded_family_work_and_decision_are_reportable(self):
        self.tuning(); plan = next(e for e in self.events if e["kind"] == "paired_plan")
        payload = dict(plan_id="trial", plan_sha256=plan["payload_sha256"], decision="unchanged")
        baseline = copy.deepcopy(self.events); self.anchor("paired_confirmation", payload)
        self.emit("collection_setup", payload=dict(plan_id="later"))
        self.assertTrue(self.derive()[0]["comparison_valid"])
        self.events = copy.deepcopy(baseline); self.emit("collection_setup", payload=dict(plan_id="later"))
        self.assertTrue(self.derive()[0]["comparison_valid"])
        self.anchor("paired_confirmation", payload)
        self.assertEqual([r["kind"] for r in self.derive()[0]["invalid_reasons"]], ["work_after_supersession"])
        self.events = copy.deepcopy(baseline); self.outcomes = {}; self.emit("collection_setup", payload=dict(plan_id="later")); self.request()
        self.assertEqual([r["kind"] for r in self.derive()[0]["invalid_reasons"]], ["work_after_supersession"])
        self.events = [e for e in baseline if e["kind"] != "paired_plan"]; self.outcomes = {}
        self.emit("collection_setup", payload=dict(plan_id="later"))
        self.anchor("paired_plan", plan["payload"]); self.request()
        self.assertEqual([r["kind"] for r in self.derive()[0]["invalid_reasons"]], ["work_after_supersession"])

    def metric(self, name="primary50"):
        return next(row for row in self.derive()[0]["metrics"] if row["name"] == name)

    def evidence(self, index=0, source="actor", slot=0, passed=False, elapsed=1, rate=None):
        record = self.built["records"][index]; task = next(t for t in record["activation"]["tasks"] if t["id"] in record["assigned_task_ids"])
        gate = next((e for e in self.events if e["kind"] == "native_intent" and e["task_sha256"] == task["sha256"]), None)
        if gate is None: self.gate(task); gate = next(e for e in self.events if e["kind"] == "native_intent" and e["task_sha256"] == task["sha256"])
        next(e for e in self.events if e["kind"] == "native_verdict" and e["operation_id"] == gate["operation_id"])["status"] = "passed"
        reserved = self.request(index, source, slot, "verified" if passed else "generation_error"); seq = reserved["seq"]
        if passed:
            self.events[-1].update(candidate_sha256="c" * 64, operation_id=gate["operation_id"], sample_id=0, reused=True)
            self.eligible.append(dict(reservation_seq=seq, experiment_id=record["experiment_id"], source=source, reply_index=slot,
                split=task["split"], id=task["id"], task_sha256=task["sha256"]))
        self.outcomes[seq]["elapsed_seconds"] = elapsed
        self.outcomes[seq]["response"].update(complete=True, cleanup_error=None if passed else "fixture cleanup")
        self.outcomes[seq]["response"]["events"] = [e for e in self.outcomes[seq]["response"]["events"] if e["event"] != "chunk"] + [dict(event="chunk", finish_reason="stop", generation_tps=rate)]
        return reserved

    def test_fixed_subset_own_first_final_joint_yields_and_assessed_contingencies(self):
        self.assertEqual([(r["name"], r["denominator"]) for r in self.derive()[0]["metrics"]],
            [("primary50", 50), ("tuning40", 40), ("fresh32", 32), ("extra22", 22), ("full72", 72)])
        tasks = self.built["activation"]["tasks"]; primary = self.built["panel"]["tasks"]
        for task in tasks: self.gate(task)
        chosen = {t["id"]: i for i, t in enumerate(primary[:16])}; outside = tasks[29]["id"]
        for index, record in enumerate(self.built["records"]):
            if index == 80: self.anchor("paired_confirmation", dict(plan_id="trial", plan_sha256=self.S.digest(self.built["plan"]), decision="unchanged"))
            identifier = record["assigned_task_ids"][0]; bits = record["profiles"]["actor"]["descriptor"]["bits"]
            if identifier not in chosen and identifier != outside: continue
            position = chosen.get(identifier, 99)
            for source in study.study.SOURCES:
                repair = bits == 8 and source == "synthetic" and 10 <= position < 15
                passed = (position < 15 or position == 15 and (bits, source) in ((8, "actor"), (16, "synthetic"))) if bits == 16 or source == "actor" else position < 10
                if identifier == outside: passed = bits == 8
                if repair: self.request(index, source, result="format_failed")
                self.evidence(index, source, int(repair), passed or repair)
        row = self.metric(); arms = {a["bits"]: a for a in row["arms"]}
        self.assertEqual((arms[8]["first_verified"]["joint"], arms[8]["final_verified"]["joint"], arms[16]["first_verified"]["joint"]), (10, 15, 15))
        self.assertEqual(row["joint_contingency"]["first_reply"], {"both": 10, "8_only": 0, "16_only": 5, "neither": 35, "unassessed": 0})
        self.assertEqual(row["joint_contingency"]["final_policy"], {"both": 15, "8_only": 0, "16_only": 0, "neither": 35, "unassessed": 0})
        self.assertEqual(row["target_met"], [dict(bits=8, first_reply=False, final_policy=True), dict(bits=16, first_reply=True, final_policy=True)])
        self.assertTrue(row["complete"])
        self.assertEqual([next(a for a in self.metric(n)["arms"] if a["bits"] == 8)["final_verified"]["joint"] for n in ("tuning40", "fresh32", "extra22", "full72")], [15, 1, 1, 16])
        self.assertEqual(self.metric("full72")["joint_contingency"]["final_policy"]["8_only"], 1)
        later = copy.deepcopy(self.built); later["plan"]["plan_id"] = "later"
        for record, member in zip(later["records"], later["plan"]["records"]):
            record["experiment_id"] += "-later"; member.update(experiment_id=record["experiment_id"], record_sha256=self.S.digest(record))
        self.install(later); trials = self.derive()
        self.assertEqual(trials[0]["metrics"][0], row)
        self.assertTrue(all(a["final_verified"]["joint"] == 0 for a in trials[1]["metrics"][0]["arms"]))

    def test_first_assessment_survives_later_pending_or_control_failed_repair(self):
        task = self.built["activation"]["tasks"][0]; self.gate(task)
        next(e for e in self.events if e["kind"] == "native_verdict")["status"] = "passed"
        for index in (0, 1):
            for source in study.study.SOURCES: self.request(index, source, result="format_failed")
        row = self.metric(); self.assertEqual(row["joint_contingency"]["first_reply"]["neither"], 1)
        self.assertEqual(row["joint_contingency"]["final_policy"]["unassessed"], 50)
        first_result = next(e for e in self.events if e["kind"] == "attempt_result")
        for status, assessed in (("generation_timeout", True), ("generation_error", True), ("interrupted", True), ("verification_error", False)):
            first_result["status"] = status
            self.assertEqual(self.metric()["joint_contingency"]["first_reply"]["unassessed"], 49 if assessed else 50)
        first_result["status"] = "format_failed"
        reserved = self.request(slot=1, result=None)
        self.assertEqual(self.metric()["joint_contingency"]["first_reply"], row["joint_contingency"]["first_reply"])
        next(e for e in self.events if e["kind"] == "generation_done" and e["reservation_seq"] == reserved["seq"])["profile_guarded"] = False
        self.outcomes[reserved["seq"]]["profile_guarded"] = False
        self.assertEqual(self.metric()["joint_contingency"]["first_reply"], row["joint_contingency"]["first_reply"])
        self.assertEqual(self.metric()["target_met"], [dict(bits=8, first_reply=None, final_policy=None), dict(bits=16, first_reply=None, final_policy=None)])

    def test_own_singleton_costs_include_failures_unknowns_repairs_and_no_native_duplication(self):
        self.request(result="format_failed"); self.request(slot=1); self.request(3); self.request(4)
        for outcome, elapsed in zip(self.outcomes.values(), (0.1, None, 0.2, 0.3)): outcome["elapsed_seconds"] = elapsed
        list(self.outcomes.values())[1]["response"] = None
        # Explicit S-proven source-row unit inputs isolate aggregation from S's accounting implementation.
        for index, replies, tokens, lower, unknown, known, missing in ((0, 2, 2, 2, 1, 0.1, 1), (3, 1, 1, 1, 0, 0.2, 0), (4, 1, 1, 1, 0, 0.3, 0)):
            name = self.built["records"][index]["experiment_id"]
            self.cost_rows[name, "actor"] = dict(source="actor", reserved=dict(replies=replies, reserved_output_tokens=replies, reserved_generation_wall_seconds=replies * 0.5),
                observed=dict(observed_generation_tokens=tokens, lower_bound_requests=lower, unknown_response_requests=unknown, known_elapsed_seconds=known, unknown_elapsed_requests=missing))
        row = self.metric(); actor = next(c for c in row["arms"][0]["costs"] if c["source"] == "actor")
        self.assertEqual(actor["reserved"], dict(replies=4, reserved_output_tokens=4, reserved_generation_wall_seconds=2.0))
        self.assertEqual({k: v for k, v in actor["observed"].items() if k != "known_elapsed_seconds"}, dict(observed_generation_tokens=4, lower_bound_requests=4, unknown_response_requests=1, unknown_elapsed_requests=1))
        self.assertEqual(actor["observed"]["known_elapsed_seconds"], 0.6)
        self.assertEqual(next(c for c in row["arms"][1]["costs"] if c["source"] == "actor")["reserved"]["replies"], 0)
        self.assertNotIn("native", actor)

    def test_first_and_adaptive_wall_known_unknown_zero_and_missing_ready(self):
        initial = self.evidence(0, elapsed=2); result = next(e for e in self.events if e["kind"] == "attempt_result"); result["status"] = "format_failed"
        next(e for e in self.events if e["kind"] == "generation_done")["status"] = self.outcomes[initial["seq"]]["status"] = "stop"
        self.outcomes[initial["seq"]]["response"]["cleanup_error"] = None
        self.evidence(0, slot=1, elapsed=6); self.evidence(1, elapsed=4)
        load_failure = self.evidence(2, elapsed=3); self.evidence(3, elapsed=0); self.evidence(4, elapsed=None); self.evidence(5, elapsed=2)
        self.outcomes[load_failure["seq"]]["response"].update(complete=False, events=[])
        timing = self.metric()["timing"][0]; a, b = timing["first_reply_wall"], timing["final_policy_wall"]
        self.assertEqual((a["known_pairs"], a["zero_denominator_pairs"], a["ratio_pairs"], a["median_16_over_8"]), (2, 1, 1, 2))
        self.assertEqual((b["known_pairs"], b["zero_denominator_pairs"], b["ratio_pairs"], b["median_16_over_8"]), (2, 1, 1, 0.5))
        self.assertEqual(a["known_tasks"], {"8": 2, "16": 3}); self.assertEqual(a["known_seconds"], {"8": 2, "16": 9})
        self.assertEqual(timing["initial_inputs"]["missing_ready_pairs"], 1)
        last = self.events.pop(); self.assertEqual(last["kind"], "attempt_result")
        self.assertEqual(self.metric()["timing"][0]["final_policy_wall"]["known_tasks"]["16"], 2)
        self.assertEqual(self.metric()["timing"][0]["first_reply_wall"]["known_tasks"]["16"], 3)

    def test_failed_gate_zero_policy_cost_and_invalid_trial_suppress_only_ratios(self):
        self.gate(self.built["activation"]["tasks"][0]); timing = self.metric()["timing"][0]
        self.assertEqual(timing["first_reply_wall"]["known_pairs"], 0)
        self.assertEqual((timing["final_policy_wall"]["known_pairs"], timing["final_policy_wall"]["zero_denominator_pairs"]), (1, 1))
        first = self.evidence(0, elapsed=2, rate=10); self.evidence(1, elapsed=6, rate=20)
        self.outcomes[first["seq"]]["response"]["events"][0]["prompt_ids"] = [7]
        timing = self.metric()["timing"][0]
        self.assertEqual((timing["first_reply_wall"]["known_pairs"], timing["first_reply_wall"]["ratio_pairs"], timing["first_reply_wall"]["median_16_over_8"]), (1, 0, None))
        self.assertEqual(timing["first_reply_wall"]["known_seconds"], {"8": 2, "16": 6})
        self.assertEqual(timing["initial_inputs"]["mismatched_ready_pairs"], 1); self.assertEqual(timing["native_first_pairs"], 0)
        self.assertEqual(self.metric()["arms"][0]["native_rates"][0]["median_generation_tps"], 10)

    def test_terminal_native_rate_types_completeness_cleanup_and_coverage_partitions(self):
        first = self.evidence(0, rate=10); self.evidence(1, rate=20); response = self.outcomes[first["seq"]]["response"]
        self.assertEqual(self.metric()["timing"][0]["native_first_pairs"], 1)
        telemetry = dict(response=dict(complete=True, events=[dict(event="chunk", finish_reason=None, generation_tps=999), dict(event="chunk", finish_reason="stop", generation_tps=None)]))
        self.assertIsNone(self.R._native_rate(telemetry))
        for value in (None, True, 0, -1, float("inf"), float("nan")):
            response["events"][-1]["generation_tps"] = value
            rate = self.metric()["arms"][0]["native_rates"][0]
            self.assertEqual((rate["count"], rate["median_generation_tps"], rate["unavailable_requests"]), (0, None, 1))
            self.assertEqual(self.metric()["timing"][0]["native_first_pairs"], 0)
        response["events"][-1]["generation_tps"] = 10; response["complete"] = False
        self.assertEqual(self.metric()["arms"][0]["native_rates"][0]["count"], 0)
        response["complete"] = True
        self.assertEqual(self.metric()["arms"][0]["native_rates"][0]["count"], 1)
        self.outcomes[first["seq"]]["response"] = None
        self.assertEqual(self.metric()["timing"][0]["initial_inputs"]["missing_ready_pairs"], 1)
        for row in self.derive()[0]["metrics"]:
            for timing in row["timing"]:
                c = timing["initial_inputs"]; self.assertEqual(c["paired_requests"] + c["missing_request_pairs"], row["denominator"])
                self.assertEqual(c["matching_ready_pairs"] + c["missing_ready_pairs"] + c["mismatched_ready_pairs"], c["paired_requests"])

    def test_real_snapshot_four_owned_replies_protocol_export_and_offline_override(self):
        campaign = self.campaign; audit = self.root / "mechanics.txt"; audit.write_bytes(b"fixture audit")
        with mock.patch.object(self.C, "AUDIT_SHA256", self.P.sha256(audit.read_bytes()).hexdigest()):
            anchor = self.C.setup(campaign, self.loaded, "trial", self.original / "receipt.json", self.identity, audit)
        records = {e["experiment_id"]: e["record"] for e in campaign.events if e["kind"] == "experiment"}
        for member in anchor["payload"]["records"][:2]: self.L.run(campaign, self.loaded, records[member["experiment_id"]])
        reservations = [e for e in campaign.events if e["kind"] == "generation_reserved"]
        self.assertEqual(len(reservations), 4); self.assertEqual(sum(e["kind"] == "native_intent" for e in campaign.events), 1)
        shutil.rmtree(self.original / "models"); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.P, "check_runtime", side_effect=AssertionError("live guard")), mock.patch.object(self.api, "prepare_request", side_effect=AssertionError("live preparation")):
            snapshot = self.S.snapshot(campaign); override = self.S.snapshot(campaign, selections=[e["seq"] for e in reservations[-2:]])
            published = self.S.publish(campaign, "v1")
        self.assertEqual(snapshot["report"]["paired_trials"], override["report"]["paired_trials"])
        self.assertEqual(snapshot["report"]["paired_trials"][0]["progress"]["terminal_cells"], 4)
        self.assertTrue(snapshot["report"]["paired_trials"][0]["comparison_valid"])
        self.assertEqual(snapshot["report"]["paired_trials"][0]["invalid_reasons"], [])
        metric = snapshot["report"]["paired_trials"][0]["metrics"][0]
        for arm in metric["arms"]:
            self.assertEqual(arm["first_verified"], dict(actor=1, synthetic=1, joint=1))
            self.assertEqual(arm["final_verified"], arm["first_verified"])
            for row in arm["costs"]:
                self.assertEqual(row["reserved"], dict(replies=1, reserved_output_tokens=1, reserved_generation_wall_seconds=0.5))
                self.assertEqual(row["observed"]["observed_generation_tokens"], 1); self.assertEqual(row["observed"]["lower_bound_requests"], 0)
                self.assertGreater(row["observed"]["known_elapsed_seconds"], 0)
            self.assertTrue(all(r["count"] == 0 and r["unavailable_requests"] == 1 for r in arm["native_rates"]))
        for timing in metric["timing"]:
            self.assertEqual(timing["initial_inputs"], dict(paired_requests=1, missing_request_pairs=49, matching_ready_pairs=1, missing_ready_pairs=0, mismatched_ready_pairs=0))
            self.assertEqual(timing["first_reply_wall"]["ratio_pairs"], 1); self.assertEqual(timing["native_first_pairs"], 0)
        self.assertTrue(all(r["first_reply"] is None and r["final_policy"] is None for r in metric["target_met"]))
        self.assertEqual(json.loads((Path(published["path"]) / "report.json").read_text())["paired_trials"][0]["metrics"][0], metric)
        setup = next(e["payload"] for e in campaign.events if e["kind"] == "collection_setup")
        self.assertIn(setup["protocol"]["path"], snapshot["files"])
        prefix = next(iter(records.values()))["profiles"]["actor"]["reference"]["path"]
        self.assertEqual({name for name in snapshot["files"] if name.startswith(prefix + "/")}, {prefix + "/" + n for n in self.P.FILES})
        self.assertTrue((Path(published["path"]) / "manifest.json").is_file())
        with mock.patch.object(self.R.study, "load_cohort", side_effect=AssertionError("legacy metadata dependency")):
            self.assertNotIn("paired_trials", self.S.snapshot(self.fresh())["report"])
