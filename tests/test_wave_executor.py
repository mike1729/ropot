import copy
import importlib
import json
from contextlib import ExitStack
import unittest
from unittest import mock
from tests import test_paired_wave_report as report, test_loop_wave as loop


class WaveExecutorTests(unittest.TestCase):
    setUpClass = classmethod(report.PairedWaveReportTests.setUpClass.__func__)
    emit, anchor = report.PairedWaveReportTests.emit, report.PairedWaveReportTests.anchor
    session = report.PairedWaveReportTests.session
    step, fail_gates = report.PairedWaveReportTests.step, report.PairedWaveReportTests.fail_gates

    def setUp(self):
        report.PairedWaveReportTests.setUp(self)
        # Each case owns its lease; preceding failure tests may close theirs.
        self.c.close()
        self.c = self.J.Campaign(self.c.root)
        self.addCleanup(self.c.close)
        self.W = self.f.C
        self.reset()
        stack = self.enterContext(ExitStack())
        patches = ((type(self.c), "events", dict(new_callable=mock.PropertyMock, side_effect=lambda: self.events)),
            (self.c, "_append", dict(side_effect=self.emit)), (self.c, "outcome", dict(side_effect=lambda seq: {k: v for k, v in self.outcomes[seq].items() if k not in ("seq", "kind")})),
            (self.c, "_finish", dict(side_effect=lambda seq, events: {k: v for k, v in self.outcomes[seq].items() if k not in ("seq", "kind")})),
            (self.c, "_finish_batch", dict(side_effect=self.finish_batch)),
            (self.c, "require_native_clear", dict(side_effect=self.clear)), (self.L.native, "reconcile_native", dict(return_value=[])),
            (self.L, "prepare_cell", dict(side_effect=self.prepare)), (self.L, "settle_cell", dict(side_effect=self.settle)),
            (self.c, "generate_batch", dict(side_effect=self.generate)), (self.S, "snapshot", dict(side_effect=self.snapshot)))
        self.mocks = {name: stack.enter_context(mock.patch.object(target, name, **kw)) for target, name, kw in patches}

    def reset(self):
        self.events = copy.deepcopy(self.baseline)
        self.outcomes, self.ready, self.sent, self.settled = {}, {}, [], []
        self.status, self.none = "format_failed", False

    def scientific(self, *args, **kwargs):
        batch, rows = report.PairedWaveReportTests.scientific(self, *args, **kwargs)
        for row in rows:
            self.outcomes[row["seq"]]["job_sha256"] = batch["job_sha256"]
        return batch, rows

    def clear(self):
        self.J.require({e["operation_id"] for e in self.events if e["kind"] == "native_intent"}
            == {e["operation_id"] for e in self.events if e["kind"] == "native_closed"}, "Unclosed native")

    def finish_batch(self, session, events):
        intent = next(e for e in events if e["kind"] == "batch_reserved" and e["seq"] == session)
        done = next(e for e in events if e["kind"] == "batch_done" and e["session_seq"] == session)
        rows = [e for e in events if e["kind"] == "generation_done" and e.get("session_seq") == session]
        self.J.require(len(rows) == len(intent["rows"])
            and {r["reservation_seq"] for r in rows} == {r["reservation_seq"] for r in intent["rows"]}
            and all(r["batch_done_seq"] == done["seq"] and r["job_sha256"] == done["job_sha256"] for r in rows), "Invalid projected atomic completion")
        return done

    def prepare(self, campaign, loaded, record, source, task_id):
        self.assertEqual(task_id, record["assigned_task_ids"][0])
        cell = next(c for c in self.L.cell_states(self.events, record) if c["id"] == task_id and c["source"] == source)
        if self.none or self.R._usable(cell, self.events): return None
        slot = len(cell["attempt_seqs"])
        messages = [dict(role="system", content="SYSTEM"), dict(role="user", content="fixture")]
        if slot:
            messages += [dict(role="assistant", content=self.outcomes[cell["attempt_seqs"][-1]]["text"]), dict(role="user", content=self.L.FORMAT_REPAIR)]
        method = record["generation_method"][source]
        request = dict(method["admission"]["request_controls"], messages=messages, seed=record["recipes"][source]["seeds"][slot])
        task = next(m["task"] for m in self.built["plan"]["records"] if m["experiment_id"] == record["experiment_id"])
        return dict(experiment_id=record["experiment_id"], source=source, request=request, attempt=dict(task, reply_index=slot))

    def generate(self, rows, method):
        op = next(o for o in self.built["plan"]["opportunities"] if rows[0]["experiment_id"] in o["experiment_ids"]
            and (o["source"], o["reply_index"]) == (rows[0]["source"], rows[0]["attempt"]["reply_index"]))
        self.assertEqual([r["experiment_id"] for r in rows], [n for n in op["experiment_ids"] if any(r["experiment_id"] == n for r in rows)])
        record = next(r for r in self.built["records"] if r["experiment_id"] == rows[0]["experiment_id"])
        self.assertEqual(method, record["generation_method"][op["source"]])
        self.sent.append((op, copy.deepcopy(rows)))
        _, reserved = self.scientific(op["id"], [r["experiment_id"] for r in rows], self.status, rows[0]["request"]["messages"])
        return [self.c.outcome(r["seq"]) for r in reserved]

    def settle(self, campaign, loaded, record, source, task_id):
        self.settled.append((record["experiment_id"], source, task_id))
        return next(c for c in self.L.cell_states(self.events, record) if (c["id"], c["source"]) == (task_id, source))

    def snapshot(self, campaign):
        plan = next(e for e in self.events if e["kind"] == "wave_plan")
        trial = dict(report_scope="resident_wave", plan_id="trial", plan_sha256=plan["payload_sha256"], comparison_valid=True, complete=False, metrics=[])
        return dict(report=dict(wave_trials=[dict(trial, plan_id="other"), trial]))

    def execute(self, count=1):
        anchor = next(e for e in self.events if e["kind"] == "wave_setup")
        return self.W._execute(self.c, self.f.loaded, self.built, anchor, self.built["plan"]["opportunities"][:count])

    def test_tune_whole_prefixes_and_post_retention_two_task_metadata(self):
        saved = copy.deepcopy(self.events)
        def milestone(campaign):
            execute.assert_called_once()
            live.assert_called_once()
            return dict(report=dict(wave_trials=[dict(trial, plan_id="other"), dict(trial, plan_sha256="0" * 64), trial]))
        self.mocks["snapshot"].side_effect = milestone
        for capacity in (None, 2):
            built, methods = copy.deepcopy(self.built), copy.deepcopy(self.f.methods)
            # Capacity2 is post-retention selector metadata; no new qualification or dispatch.
            if capacity:
                for bound in methods.values():
                    for method in bound.values(): method["admission"]["config"]["batch_size"] = capacity
            built["plan"]["opportunities"] = self.W._opportunities(built["activation"], built["plan"]["records"], methods)
            anchor = copy.deepcopy(next(e for e in self.events if e["kind"] == "wave_setup"))
            anchor["payload"]["plan_sha256"] = self.W.digest(built["plan"])
            tuning = [o for o in built["plan"]["opportunities"] if not o["block"].endswith("fresh")]
            groups = list(dict.fromkeys(o["group"] for o in tuning))
            self.assertEqual(len(tuning[0]["experiment_ids"]), 2 if capacity else 4)
            retention = dict(return_value=(built, anchor)) if capacity else dict(wraps=self.W._retained)
            trial = dict(self.snapshot(self.c)["report"]["wave_trials"][-1], plan_sha256=anchor["payload"]["plan_sha256"], comparison_valid=False)
            for limit in (1, len(groups), None):
                expected = [o for o in tuning if limit is None or o["group"] in groups[:limit]]
                with mock.patch.object(self.W, "_retained", **retention) as retained, mock.patch.object(self.W, "_execute") as execute, \
                     mock.patch.object(self.W, "_live", wraps=self.W._live) as live:
                    self.mocks["snapshot"].reset_mock()
                    self.assertIs(self.W.tune(self.c, self.f.loaded, "trial", through_groups=limit), trial)
                    retained.assert_called_once_with(self.c, self.f.loaded, "trial")
                    execute.assert_called_once_with(self.c, self.f.loaded, built, anchor, expected)
                    live.assert_called_once_with(self.c, anchor, {r["experiment_id"] for r in built["records"]})
                    self.mocks["snapshot"].assert_called_once_with(self.c)
        self.assertEqual(self.events, saved)
        self.mocks["prepare_cell"].assert_not_called()

    def test_tune_bounds_stop_before_selected_effects(self):
        upper = len({o["group"] for o in self.built["plan"]["opportunities"] if not o["block"].endswith("fresh")})
        for limit in (True, False, 0, -1, 1.0, "1", upper + 1):
            with mock.patch.object(self.W, "_retained", wraps=self.W._retained) as retained, \
                 mock.patch.object(self.W, "_execute") as execute, mock.patch.object(self.W, "_live") as live:
                with self.subTest(limit=limit), self.assertRaises(ValueError): self.W.tune(self.c, self.f.loaded, "trial", through_groups=limit)
                self.assertEqual(retained.call_count, 1 if limit == upper + 1 else 0)
                execute.assert_not_called()
                live.assert_not_called()
                self.mocks["snapshot"].assert_not_called()
        self.assertEqual(self.events, self.baseline)

    def test_tune_requires_unique_scientific_trial_and_list_metrics(self):
        trial = self.snapshot(self.c)["report"]["wave_trials"][-1]
        invalid = [[], [trial, trial], [dict(trial, plan_id="other")], [dict(trial, plan_sha256="0" * 64)]]
        invalid += [[dict(trial, metrics=value)] for value in (None, {}, "[]")]
        invalid += [[dict(trial, report_scope="structure_only")], [{k: v for k, v in trial.items() if k != "metrics"}]]
        invalid += [[dict(trial, **{name: False})] for name in ("provisional", "structure_only", "validation_scope")]
        for trials in invalid:
            self.mocks["snapshot"].reset_mock()
            self.mocks["snapshot"].side_effect = None
            self.mocks["snapshot"].return_value = dict(report=dict(wave_trials=trials))
            with mock.patch.object(self.W, "_execute"), self.subTest(trials=trials), self.assertRaises(ValueError):
                self.W.tune(self.c, self.f.loaded, "trial", through_groups=1)
            self.mocks["snapshot"].assert_called_once_with(self.c)

    def test_finite_order_independent_repairs_and_idempotent_steps(self):
        tuning = [o for o in self.built["plan"]["opportunities"] if not o["block"].endswith("fresh")]
        passed = tuning[0]["experiment_ids"][0]
        self.status = {r["experiment_id"]: "verified" if r["experiment_id"] == passed else "format_failed" for r in self.built["records"]}
        steps = self.execute(len(tuning))
        self.assertEqual((len(self.built["records"]), len(self.built["plan"]["opportunities"])), (144, 168))
        self.assertEqual([s["payload"]["opportunity_id"] for s in steps], [o["id"] for o in tuning])
        for op, rows in self.sent:
            self.assertIn(op, tuning)
            self.assertLessEqual(len(rows), 8 if op["source"] == "actor" else 4)
            for row in rows:
                self.assertEqual(row["request"]["seed"], self.W.PC.RECIPE["seeds"][op["reply_index"]])
                if op["reply_index"]:
                    self.assertNotEqual(row["experiment_id"], passed)
                    self.assertEqual(row["request"]["messages"][-2]["content"], "not python !")
        saved, dispatched = copy.deepcopy(self.events), len(self.sent)
        self.assertEqual(self.execute(len(tuning)), steps)
        self.assertEqual((self.events, len(self.sent)), (saved, dispatched))
        self.mocks["snapshot"].assert_not_called()

    def test_gapped_slices_stop_before_live_even_with_retained_prefix(self):
        ops = self.built["plan"]["opportunities"]
        for retained in (False, True):
            if retained: self.execute()
            with mock.patch.object(self.W, "_live", side_effect=AssertionError("Live")), self.assertRaisesRegex(ValueError, "opportunity slice"):
                self.W._execute(self.c, self.f.loaded, self.built, None, [ops[0], ops[3]])

    def test_otherwise_valid_session_mutations_and_missing_atomic_rows(self):
        for fault in ("mixed", "order", "method", "seed", "duplicate", "fresh", "gap", "missing_atomic_rows"):
            self.reset()
            session, rows = self.scientific(0)
            if fault == "mixed": rows[0]["experiment_id"] = "foreign"
            if fault == "order": session["rows"].reverse()
            if fault == "method": session["method_spec"] = dict(session["method_spec"], prompt_ceiling_tokens=1)
            if fault == "seed": rows[0]["request_text"] = self.J.encoded(dict(json.loads(rows[0]["request_text"]), seed=99))
            if fault == "duplicate": self.scientific(0)
            if fault == "fresh": self.scientific(next(o["id"] for o in self.built["plan"]["opportunities"] if o["block"].endswith("fresh")))
            if fault == "gap": self.scientific(3)
            if fault == "missing_atomic_rows": self.events = [e for e in self.events if e["kind"] not in ("generation_done", "attempt_result")]
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.execute()
            self.assertEqual(self.sent, [])
            self.mocks["prepare_cell"].assert_not_called()

    def test_historical_step_prefix_cannot_be_healed(self):
        for fault in ("hash", "binding", "duplicate", "order", "attempt_result", "native_verdict", "native_closed", "generation_done", "batch_done"):
            self.reset()
            session, _ = self.scientific()
            step = self.step(0, session["seq"])
            if fault == "hash": step["payload_sha256"] = "0" * 64
            elif fault == "binding": step["payload"]["session_seq"] += 1; step["payload_sha256"] = self.S.digest(step["payload"])
            elif fault == "duplicate": self.step(0, session["seq"])
            elif fault == "order": self.step(2, None)
            else:
                late = next(e for e in self.events if e["kind"] == fault)
                self.events.remove(late)
                moved = self.emit(late["kind"], **{k: v for k, v in late.items() if k not in ("seq", "kind")})
                if fault == "batch_done":
                    for row in self.outcomes.values(): row["batch_done_seq"] = moved["seq"]
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.execute()
            self.assertEqual(self.sent, [])

    def test_none_requires_terminal_support_and_discards_prepared_peer(self):
        tasks = self.built["activation"]["tasks"]
        for state in ("unstarted", "gate_failed", "unclosed", "budget", "peer"):
            self.reset()
            if state in ("gate_failed", "unclosed"): self.fail_gates(tasks)
            if state == "unclosed": self.events = [e for e in self.events if e["kind"] != "native_closed"]
            if state == "budget": self.emit("limits", limits={s: dict.fromkeys(self.J.UNITS, 1) for s in ("actor", "synthetic")})
            self.none = state != "peer"
            if state == "peer":
                self.mocks["prepare_cell"].side_effect = lambda *args: self.prepare(*args) if self.mocks["prepare_cell"].call_count == 1 else None
            if state == "gate_failed": self.assertIsNone(self.execute()[0]["payload"]["session_seq"])
            else:
                with self.subTest(state=state), self.assertRaises(ValueError): self.execute()
            if state == "budget": self.assertTrue(any(c["reason"] == "campaign_budget" for c in self.L.cell_states(self.events, self.built["records"][0])))
            self.assertEqual(self.sent, [])

    def test_global_resources_all_own_controls_and_live_producer_guards(self):
        self.assertTrue(all("src/data-pipeline/" + name + ".py" in self.W.PRODUCERS for name in ("study_report", "paired_wave_report", "paired_report")))
        for fault in ("control", "generation", "native", "producer", "collection_setup", "wave_setup"):
            self.reset()
            if fault in ("control", "generation"):
                session, rows = self.scientific(20)
                if fault == "control": self.outcomes[rows[-1]["seq"]]["profile_guarded"] = False
                else:
                    for row in rows: row["experiment_id"] = "unrelated"
                    next(e for e in self.events if e["kind"] == "batch_done")["shared"]["resources_closed"] = False
            if fault == "native": self.emit("native_intent", operation_id="foreign")
            if fault in ("collection_setup", "wave_setup"): self.anchor(fault, dict(plan_id="later"))
            fingerprint = self.W.fingerprint()
            if fault == "producer": fingerprint[next(iter(fingerprint))] = "0" * 64
            with mock.patch.object(self.W, "fingerprint", return_value=fingerprint), self.subTest(fault=fault), self.assertRaises((ValueError, self.J.ProfileControlError)):
                self.execute()
            self.assertEqual((self.sent, self.settled), ([], []))

    def test_closed_candidate_error_stops_after_all_settle_and_retained_step(self):
        for retained in (False, True):
            self.reset()
            self.status = "verification_error"
            if retained:
                session, rows = self.scientific(status=self.status)
                self.assertTrue(self.R._effects(self.events, [r["seq"] for r in rows]))
                self.step(0, session["seq"])
            saved = [e for e in self.events if e["kind"] == "wave_step"]
            with self.assertRaisesRegex(ValueError, "Unusable settled"): self.execute(2)
            self.assertEqual(len(self.settled), len(self.built["plan"]["opportunities"][0]["experiment_ids"]))
            self.assertEqual([e for e in self.events if e["kind"] == "wave_step"], saved)
            self.assertEqual(len(self.sent), 0 if retained else 1)
            self.assertTrue(self.R._effects(self.events, [e["seq"] for e in self.events if e["kind"] == "generation_reserved"]))
            self.assertTrue(any(c["reason"] == "native_infrastructure" for r in self.built["records"] for c in self.L.cell_states(self.events, r)))

    def test_pre_generation_guards_and_omitted_peer_prefix(self):
        for kind in ("producer", "collection_setup", "wave_setup", "omitted"):
            self.reset()
            self.mocks["prepare_cell"].reset_mock()
            if kind == "omitted": self.scientific(names=self.built["plan"]["opportunities"][0]["experiment_ids"][:1])
            fingerprint = self.W.fingerprint()
            prepare = self.prepare
            def changed(*args):
                row = prepare(*args)
                if self.mocks["prepare_cell"].call_count == len(self.built["plan"]["opportunities"][0]["experiment_ids"]):
                    if kind == "producer": fingerprint[next(iter(fingerprint))] = "0" * 64
                    else: self.anchor(kind, dict(plan_id="later"))
                return row
            if kind != "omitted": self.mocks["prepare_cell"].side_effect = changed
            with mock.patch.object(self.W, "fingerprint", return_value=fingerprint), self.subTest(kind=kind), self.assertRaises(ValueError): self.execute()
            if kind != "omitted": self.assertEqual(self.mocks["prepare_cell"].call_count, len(self.built["plan"]["opportunities"][0]["experiment_ids"]))
            self.assertEqual(self.sent, [])


class WaveExecutorRealTests(unittest.TestCase):
    def fixture(self, *, later=False, omitted=False):
        g = loop.LoopWaveTests(methodName="runTest")
        g.setUp()
        self.addCleanup(g.doCleanups)
        W = importlib.import_module("src.data-pipeline.paired_wave_collection")
        if omitted:
            task = dict(g.f.task, id=2, difficulty="interview")
            g.f.s.write_task(2, task)
            g.f.loaded = g.f.s.rewrite()
            factory = g.factory
            def assigned(loaded, name, groups, recipes, selected, **kwargs):
                return factory(loaded, name, groups, recipes, [2] if name == "two" else selected, **kwargs)
            g.factory = assigned
        c, records = g.fresh()
        op = dict(id=0, block="dev", group=0, bits=8, source="actor", reply_index=0, experiment_ids=[r["experiment_id"] for r in records])
        if later: records += [next(e["record"] for e in c.events if e["kind"] == "experiment" and e["experiment_id"] == "empty")]
        members = [dict(experiment_id=r["experiment_id"], task=dict(split="test", id=r["assigned_task_ids"][0],
            task_sha256=g.f.s.records[r["assigned_task_ids"][0]]["sha256"]), bits=8) for r in records]
        plan = dict(plan_id="seam", opportunities=[op], records=members)
        plan["contract_sha256"] = next(e["contract_sha256"] for e in c.events if e["kind"] == "execution_context")
        if later: plan["opportunities"].append(dict(op, id=1, experiment_ids=[records[-1]["experiment_id"]]))
        if omitted: plan["opportunities"].append(dict(op, id=1, reply_index=1, experiment_ids=op["experiment_ids"][:1]))
        event = c._append("wave_plan", payload=plan, payload_sha256=W.digest(plan))
        def history(built, events, outcomes, allowed):
            rows = [(m, r, g.L.cell_states(events, r)) for m, r in zip(plan["records"], records)]
            reservations = {e["seq"]: e["experiment_id"] for e in events if e["kind"] == "generation_reserved"}
            sessions = {next(o["id"] for o in plan["opportunities"] if reservations[e["rows"][0]["reservation_seq"]] in o["experiment_ids"]): e for e in events if e["kind"] == "batch_reserved"}
            steps = {e["payload"]["opportunity_id"]: e for e in events if e["kind"] == "wave_step"}
            return event, rows, sessions, steps
        stack = self.enterContext(ExitStack())
        stack.enter_context(mock.patch.object(W, "_guard"))
        stack.enter_context(mock.patch.object(W, "_history", side_effect=history))
        return g, W, c, dict(records=records, plan=plan), event

    def test_all_four_crash_positions_reopen_without_another_owned_session(self):
        for fault in ("before_settle", "after_settle", "before_step", "after_step"):
            g, W, c, built, anchor = self.fixture()
            append, settle = c._append, g.L.settle_cell
            def interrupted(kind, **payload):
                if kind == "wave_step" and fault == "before_step": raise OSError(fault)
                result = append(kind, **payload)
                if kind == "wave_step" and fault == "after_step": raise OSError(fault)
                return result
            def settlement(*args):
                if fault == "before_settle": raise OSError(fault)
                result = settle(*args)
                if fault == "after_settle": raise OSError(fault)
                return result
            with mock.patch.object(c, "generate_batch", wraps=c.generate_batch) as batch, \
                 mock.patch.object(c, "_append", side_effect=interrupted), mock.patch.object(g.L, "settle_cell", side_effect=settlement), mock.patch.object(g.H, "run", side_effect=g.f.f.runner()) as run:
                with self.subTest(fault=fault), self.assertRaises(OSError): W._execute(c, g.f.loaded, built, anchor, built["plan"]["opportunities"])
            self.assertEqual(run.call_count, 1)
            self.assertEqual(batch.call_count, 1)
            self.assertTrue(all(set(row) == {"experiment_id", "source", "request", "attempt"} for row in batch.call_args.args[0]))
            root = c.root
            c.close()
            reopened = self.enterContext(g.J.Campaign(root))
            g.f.campaign = reopened
            with mock.patch.object(g.H, "run", side_effect=AssertionError("GPU replay")), mock.patch.object(reopened, "generate_batch", side_effect=AssertionError("New session")), mock.patch.object(g.L, "prepare_cell", side_effect=AssertionError("Refill")):
                steps = W._execute(reopened, g.f.loaded, built, anchor, built["plan"]["opportunities"])
            events = reopened.events
            self.assertEqual((sum(e["kind"] == "batch_reserved" for e in events), g.J.Campaign.totals(events)[1]["actor"]["replies"]), (1, 2))
            self.assertEqual(len(steps), 1)
            self.assertEqual([e["status"] for e in events if e["kind"] == "attempt_result"], ["verified", "format_failed"])
            self.assertTrue(all(e["seq"] < steps[0]["seq"] for e in events if e["kind"] in ("attempt_result", "generation_done", "native_verdict", "native_closed")))
            self.assertEqual(steps[0]["payload"], dict(plan_id="seam", plan_sha256=W.digest(built["plan"]), opportunity_id=0, session_seq=next(e["seq"] for e in events if e["kind"] == "batch_reserved")))
            with mock.patch.object(g.N, "run_native", side_effect=AssertionError("Native replay")), mock.patch.object(g.H, "run", side_effect=AssertionError("GPU replay")):
                self.assertEqual(W._execute(reopened, g.f.loaded, built, anchor, built["plan"]["opportunities"]), steps)
            self.assertEqual(reopened.events, events)

    def test_closed_shared_failure_recovers_new_checkpoint_then_pauses(self):
        g, W, c, built, anchor = self.fixture(later=True)
        with mock.patch.object(g.H, "run", side_effect=g.f.f.runner("timeout")), mock.patch.object(g.L, "settle_cell", side_effect=OSError("After GEN")):
            with self.assertRaises(OSError): W._execute(c, g.f.loaded, built, anchor, built["plan"]["opportunities"])
        root = c.root
        c.close()
        reopened = self.enterContext(g.J.Campaign(root))
        g.f.campaign = reopened
        with mock.patch.object(g.H, "run", side_effect=AssertionError("Retry")), mock.patch.object(g.N, "run_native", side_effect=AssertionError("Ineligible candidate")):
            steps = W._execute(reopened, g.f.loaded, built, anchor, built["plan"]["opportunities"])
            self.assertEqual(len(steps), 1)
            self.assertEqual(W._execute(reopened, g.f.loaded, built, anchor, built["plan"]["opportunities"][:1]), steps)
        self.assertEqual([e["status"] for e in reopened.events if e["kind"] == "attempt_result"], ["generation_timeout", "generation_timeout"])
        self.assertEqual((len(steps), g.J.Campaign.totals(reopened.events)[1]["actor"]["replies"]), (1, 2))
        with mock.patch.object(g.H, "run", side_effect=g.f.f.runner()):
            later = W._execute(reopened, g.f.loaded, built, anchor, built["plan"]["opportunities"][1:])
        self.assertEqual([step["payload"]["opportunity_id"] for step in later], [1])
        self.assertEqual(g.J.Campaign.totals(reopened.events)[1]["actor"]["replies"], 3)

    def test_distinct_omitted_failed_gate_replays_real_proof_before_later_generation(self):
        g, W, c, built, anchor = self.fixture(omitted=True)
        first, omitted = built["records"]
        binding, task = g.L.study.load_task(g.f.loaded, omitted["activation"], 2)
        self.assertNotEqual(binding["sha256"], g.f.s.records[1]["sha256"])
        execute = g.f.execute
        def failed_reference(*args, **kwargs):
            result = execute(*args, **kwargs)
            sample = g.f.log.samples[0]
            sample.store["cases"] = sample.store["cases"][:sum(case["visible"] for case in task["cases"])]
            sample.store["cases"][0]["stdout"] = "WRONG"
            return result
        with mock.patch.object(g.N.verification, "reference_gate", side_effect=failed_reference):
            proof = g.N.run_native(c, task, binding["sha256"])
        self.assertEqual(proof["status"], "failed")
        row = g.L.prepare_cell(c, g.f.loaded, first, "actor", 1)
        g.generate(c, [row], texts=("not python !",))
        with mock.patch.object(g.H, "run", side_effect=AssertionError("Recorded session")):
            steps = W._execute(c, g.f.loaded, built, anchor, built["plan"]["opportunities"][:1])
        gate = next(e for e in c.events if e["kind"] == "native_intent" and e["task_sha256"] == binding["sha256"])
        verdict = next(e for e in c.events if e["kind"] == "native_verdict" and e["operation_id"] == gate["operation_id"])
        self.assertFalse(any(e["kind"] == "generation_reserved" and e["experiment_id"] == omitted["experiment_id"] for e in c.events))
        (c.root / verdict["log"]).write_bytes(b"tampered retained failed gate")
        with mock.patch.object(g.N, "reuse_native", wraps=g.N.reuse_native) as reuse, mock.patch.object(g.H, "run", side_effect=AssertionError("Later GEN")), self.assertRaisesRegex(ValueError, "Native log evidence changed"):
            W._execute(c, g.f.loaded, built, anchor, built["plan"]["opportunities"])
        self.assertTrue(any(call.args[1]["operation_id"] == gate["operation_id"] for call in reuse.call_args_list))
        self.assertEqual([e for e in c.events if e["kind"] == "wave_step"], steps)
        self.assertEqual(g.J.Campaign.totals(c.events)[1]["actor"]["replies"], 1)
