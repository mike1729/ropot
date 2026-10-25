import copy
import importlib
import json
import sqlite3
from types import SimpleNamespace
import unittest
from unittest import mock
from tests import test_generation_journal as scalar, test_batch_report as batch, test_loop_wave as loop


class EntryReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.L = importlib.import_module("src.data-pipeline.experiment_loop")
        self.J = self.L.journal
        self.W = importlib.import_module("src.data-pipeline.paired_wave_collection")

    def scalar(self, pending=False):
        f = scalar.GenerationJournalTests(methodName="runTest"); f.setUp(); self.addCleanup(f.doCleanups)
        c, _ = f.create(); self.addCleanup(c.close)
        request = f.request(); _, raw = f.api.prepare_request(request)
        intent = c._append("generation_reserved", experiment_id="one", source="actor", attempt=None,
            request_text=raw.decode(), request_sha256=f.api.sha256(raw).hexdigest())
        f.api.generate(request, c.root / "requests" / f"{intent['seq']:06d}", lease_fd=c._lock.fileno())
        if not pending: c.outcome(intent["seq"])
        return f, c, intent

    def batch(self, pending=False, mode=None):
        f = batch.BatchReportTests(methodName="runTest"); f.setUp(); self.addCleanup(f.doCleanups)
        c, rows = f.fresh()
        if pending: self.materialize(f, c, rows)
        elif mode == "control":
            with self.assertRaises(self.J.ProfileControlError):
                loop.LoopWaveTests.generate(SimpleNamespace(f=f, b=f.b, H=f.H), c, rows, "control")
            c = self.J.Campaign(c.root); self.addCleanup(c.close); f.campaign = c
        else: f.generate(c, rows, mode)
        return f, c, rows

    def materialize(self, f, c, rows):
        intent = c.reserve_batch(rows, f.b.method)
        directory = c.root / "sessions"; directory.mkdir(exist_ok=True)
        f.f.runner()(json.loads(intent["job_text"]), directory / f"{intent['seq']:06d}", c.root / "lock",
            lease_fd=c._lock.fileno(), qualification_files=self.J._batch_method(f.b.method, c))
        return intent

    def foreign_scalar(self, f, c):
        record = copy.deepcopy(next(e["record"] for e in c.events if e["kind"] == "experiment"))
        record.update(schema_version=2, experiment_id="scalar"); record.pop("generation_method"); c.register(record)
        _, raw = f.H.G.prepare_request(f.b.raw)
        return c._append("generation_reserved", experiment_id="scalar", source="actor", attempt=None,
            request_text=raw.decode(), request_sha256=f.H.sha256(raw).hexdigest())

    def observe(self, c, fail=None):
        getter, reads = type(c).events.fget, []
        def read(instance):
            reads.append(None)
            if fail is not None and len(reads) == 2: raise fail
            return getter(instance)
        return reads, mock.patch.object(type(c), "events", property(read))

    def stored(self, c):
        with sqlite3.connect(c.root / "journal.sqlite3") as db:
            return db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()

    def test_iterator_supplied_prefix_is_lazy_and_close_does_not_close_campaign(self):
        f, c, _ = self.batch(); f.generate(c, f.register(c, ("three", "four")))
        events = c.events; before = self.stored(c); reads, observer = self.observe(c)
        first = next(e for e in events if e["kind"] == "generation_reserved")
        expected = c.outcome(first["seq"])
        with observer, mock.patch.object(c, "_finish_batch", wraps=c._finish_batch) as finish:
            iterator = c._generation_outcomes(events)
            finish.assert_not_called()
            reservation, outcome, prefix = next(iterator)
            self.assertEqual((reservation, outcome, prefix), (first, expected, events))
            iterator.close()
        self.assertEqual((len(reads), finish.call_count), (0, 1))
        self.assertIsNotNone(c.db); self.assertEqual(self.stored(c), before)

    def test_reserve_reuses_prefix_with_independent_native_and_iterator_proofs(self):
        for size in (1, 2):
            f, c, _ = self.batch(); scalar = self.foreign_scalar(f, c); c.outcome(scalar["seq"])
            if size == 2: f.generate(c, f.register(c, ("three", "four")))
            rows = f.register(c, ("next-one", "next-two")); before = c.events
            order = [e["seq"] for e in before if e["kind"] == "generation_reserved"]
            reads, observer = self.observe(c)
            with observer, mock.patch.object(c, "outcome", side_effect=AssertionError("Public outcome scan")), \
                 mock.patch.object(self.J, "_batch_projection", wraps=self.J._batch_projection) as project, \
                 mock.patch.object(c, "_finish", wraps=c._finish) as finish, \
                 mock.patch.object(self.J, "generation_resources_cleared", wraps=self.J.generation_resources_cleared) as resources, \
                 mock.patch.object(f.H, "run", side_effect=AssertionError("Dispatch")):
                intent = c.reserve_batch(rows, f.b.method)
            self.assertEqual((len(reads), project.call_count, finish.call_count), (2, 2 * size, 1))
            self.assertEqual([call.args[1]["reservation_seq"] for call in resources.call_args_list], order)
            after = c.events; self.assertEqual(after[:len(before)], before)
            admitted = after[len(before) + 1:]
            self.assertEqual([r["experiment_id"] for r in admitted], [r["experiment_id"] for r in rows])
            self.assertEqual([r["seq"] for r in admitted], [r["reservation_seq"] for r in intent["rows"]])
            self.assertTrue(all(r["session_seq"] == intent["seq"] for r in admitted))
            self.assertEqual(self.J.Campaign.totals(after)[1]["actor"]["replies"], 2 * size + 3)

    def test_reserve_proof_tamper_or_unknown_resources_never_admits_new_rows(self):
        for fault in ("scalar", "session", "qualification", "unknown"):
            f, c, _ = self.batch(mode="unknown" if fault == "unknown" else None)
            if fault == "scalar": scalar = self.foreign_scalar(f, c); c.outcome(scalar["seq"])
            rows = f.register(c, ("next-one", "next-two")); before = self.stored(c)
            if fault == "scalar": path = c.root / "requests" / f"{scalar['seq']:06d}" / "response.json"
            elif fault == "qualification": path = c.root / f.b.method["qualification"]["path"] / "response.json"
            else: path = c.root / "sessions" / f"{next(e['seq'] for e in c.events if e['kind'] == 'batch_reserved'):06d}" / "stdout.jsonl"
            if fault != "unknown":
                # The completed interrupted scalar had no files; adding bytes changes its proven inventory.
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"changed")
            with mock.patch.object(f.f.M, "build_wave", side_effect=AssertionError("New wave")), \
                 mock.patch.object(c, "_append_group", side_effect=AssertionError("New reservation")), \
                 self.subTest(fault=fault), self.assertRaises(ValueError):
                c.reserve_batch(rows, f.b.method)
            self.assertIsNone(c.db); self.assertEqual(self.stored(c), before)

    def test_reserve_pending_scalar_stops_before_finish_build_or_completion_append(self):
        f, c, _ = self.batch(); self.foreign_scalar(f, c)
        rows = f.register(c, ("next-one", "next-two")); before = self.stored(c)
        with mock.patch.object(c, "_finish", side_effect=AssertionError("Pending scalar finish")), \
             mock.patch.object(f.f.M, "build_wave", side_effect=AssertionError("New wave")), \
             mock.patch.object(c, "_append_group", side_effect=AssertionError("Append")), \
             self.assertRaisesRegex(ValueError, "Reconcile pending generations"):
            c.reserve_batch(rows, f.b.method)
        self.assertIsNone(c.db); self.assertEqual(self.stored(c), before)

    def test_native_clear_supplies_one_prefix_and_only_completes_last_pending_batch(self):
        for pending in (False, True):
            f, c, _ = self.batch()
            rows = f.register(c, ("three", "four"))
            if pending: self.materialize(f, c, rows)
            else: f.generate(c, rows)
            before = c.events; reads, observer = self.observe(c)
            with observer, mock.patch.object(c, "_finish_batch", wraps=c._finish_batch) as finish:
                c.require_native_clear()
            self.assertEqual((len(reads), finish.call_count), (1, 2))
            self.assertTrue(all(call.args[1] is finish.call_args_list[0].args[1] for call in finish.call_args_list))
            after = c.events; self.assertEqual(after[:len(before)], before)
            self.assertEqual([e["kind"] for e in after[len(before):]], ["batch_done", "generation_done", "generation_done"] if pending else [])
            self.assertIsNotNone(c.db)
        f, c, _ = self.batch(); before = c.events
        def interrupt(*args, **kwargs):
            f.execute(*args, **kwargs)
            raise KeyboardInterrupt()
        with mock.patch.object(f.N.verification, "reference_gate", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            f.N.run_native(c, f.task, f.s.records[1]["sha256"])
        with self.assertRaisesRegex(ValueError, "Reconcile pending native"): c.require_native_clear()
        self.assertEqual(sum(e["kind"] == "native_intent" for e in c.events[len(before):]), 1)
        self.assertFalse(any(e["kind"] == "native_closed" for e in c.events[len(before):]))

    def test_snapshot_completes_pending_batch_once_but_reports_unknown_and_control(self):
        for mode in ("pending", "unknown", "control"):
            f, c, _ = self.batch(pending=mode == "pending", mode=None if mode == "pending" else mode)
            reads, observer = self.observe(c)
            with observer, mock.patch.object(c, "outcome", side_effect=AssertionError("Public outcome scan")), \
                 mock.patch.object(f.H, "run", side_effect=AssertionError("Replay")), \
                 mock.patch.object(f.N, "run_native", side_effect=AssertionError("Native")), \
                 mock.patch.object(f.N, "reconcile_native", side_effect=AssertionError("Recovery")):
                snapshot = f.R.snapshot(c)
            self.assertEqual(len(reads), 4 if mode == "pending" else 3)
            costs = snapshot["report"]["costs"]
            self.assertEqual(costs["reserved"]["actor"]["replies"], 2)
            self.assertEqual(len(costs["generation_sessions"]), 1)
            self.assertEqual(costs["observed_by_source"]["actor"]["unknown_elapsed_sessions"], int(mode == "unknown"))
            done = [e for e in c.events if e["kind"] == "generation_done"]
            self.assertEqual(len(done), 2)
            if mode == "control": self.assertTrue(all(self.J.profile_control_reason(e) is not None for e in done))
            before = self.stored(c); reads, observer = self.observe(c)
            with observer, mock.patch.object(self.J, "_batch_projection", wraps=self.J._batch_projection) as project:
                self.assertEqual(f.R.snapshot(c), snapshot)
            self.assertEqual((len(reads), project.call_count), (3, 1))
            self.assertEqual(self.stored(c), before); self.assertIsNotNone(c.db)

    def test_complete_scalar_reproves_every_row_with_one_history_read(self):
        f, c, _ = self.scalar()
        for size in (1, 3):
            if size == 3:
                for seed in (18, 19): c.generate("one", "actor", f.request(seed=seed))
            expected = {e["seq"]: c.outcome(e["seq"]) for e in c.events if e["kind"] == "generation_reserved"}
            before = self.stored(c); reads, observer = self.observe(c)
            with observer, mock.patch.object(c, "_finish", wraps=c._finish) as finish:
                events, own = self.L._reconcile_generations(c, {"one"})
            self.assertEqual((len(reads), finish.call_count), (1, size)); self.assertEqual(own, expected)
            self.assertEqual(events, c.events); self.assertEqual(self.stored(c), before)

    def test_complete_mixed_sessions_once_each_invocation_and_each_peer_checked(self):
        f, c, rows = self.batch(); extra = self.foreign_scalar(f, c); c.outcome(extra["seq"])
        for size in (1, 2):
            if size == 2: f.generate(c, f.register(c, ("three", "four")))
            names = {"one", "two", "three", "four"}
            expected = {e["seq"]: c.outcome(e["seq"]) for e in c.events if e["kind"] == "generation_reserved" and e["experiment_id"] in names}
            before = self.stored(c)
            for _ in range(2):
                reads, observer = self.observe(c)
                with observer, mock.patch.object(self.J, "_batch_projection", wraps=self.J._batch_projection) as project, \
                     mock.patch.object(self.J, "generation_resources_cleared", wraps=self.J.generation_resources_cleared) as resources, \
                     mock.patch.object(self.J, "require_profile_outcome", wraps=self.J.require_profile_outcome) as profile:
                    events, own = self.L._reconcile_generations(c, names)
                self.assertEqual((len(reads), project.call_count, profile.call_count), (1, size, 2 * size))
                self.assertEqual([call.args[1]["reservation_seq"] for call in resources.call_args_list],
                    [e["seq"] for e in events if e["kind"] == "generation_reserved"])
                self.assertEqual(own, expected); self.assertEqual(self.stored(c), before)

    def test_retained_bytes_changed_between_entries_close_without_dispatch(self):
        for kind in ("scalar", "session", "qualification"):
            f, c, item = self.scalar() if kind == "scalar" else self.batch()
            self.L._reconcile_generations(c, {"one"}); before = self.stored(c)
            if kind == "scalar": path = c.root / "requests" / f"{item['seq']:06d}" / "response.json"
            elif kind == "qualification": path = c.root / f.b.method["qualification"]["path"] / "response.json"
            else: path = c.root / "sessions" / f"{next(e['seq'] for e in c.events if e['kind'] == 'batch_reserved'):06d}" / "stdout.jsonl"
            path.write_bytes(b"changed")
            G = f.api if kind == "scalar" else f.H.G
            with mock.patch.object(G, "generate", side_effect=AssertionError("Dispatch")), self.subTest(kind=kind), self.assertRaises(ValueError):
                self.L._reconcile_generations(c, {"one"})
            self.assertIsNone(c.db); self.assertEqual(self.stored(c), before)

    def test_real_unknown_cleanup_blocks_open_without_later_admission_or_effects(self):
        f, c, _ = self.batch(mode="unknown")
        record = next(e["record"] for e in c.events if e["kind"] == "experiment")
        for names in (set(), {"one"}, {"two"}):
            with self.assertRaisesRegex(ValueError, "cleanup"): self.L._reconcile_generations(c, names)
            self.assertIsNotNone(c.db)
        with mock.patch.object(f.N.verification.runner, "contract", side_effect=AssertionError("Contract")), \
             mock.patch.object(f.N, "reconcile_native", side_effect=AssertionError("Native")), mock.patch.object(self.W, "_guard"):
            with self.assertRaisesRegex(ValueError, "cleanup"): self.L._enter(c, f.loaded, record)
            with self.assertRaisesRegex(ValueError, "cleanup"): self.W._live(c, None, {"one", "two"})
        self.assertIsNotNone(c.db)

    def test_real_control_stops_before_later_session_and_foreign_control_is_not_owned(self):
        f, c, _ = self.batch(mode="control")
        later = self.materialize(f, c, f.register(c, ("three", "four")))
        for names, count in (({"one"}, 1), ({"two"}, 2)):
            with mock.patch.object(c, "_finish_batch", wraps=c._finish_batch) as finish, \
                 mock.patch.object(self.J, "generation_resources_cleared", wraps=self.J.generation_resources_cleared) as resources:
                with self.assertRaises(self.J.ProfileControlError): self.L._reconcile_generations(c, names)
            self.assertEqual((finish.call_count, resources.call_count), (1, count)); self.assertIsNotNone(c.db)
            self.assertFalse(any(e["kind"] == "batch_done" and e["session_seq"] == later["seq"] for e in c.events))
        _, own = self.L._reconcile_generations(c, {"three", "four"})
        self.assertEqual(len(own), 2); self.assertTrue(all(v["profile_guarded"] for v in own.values()))

    def test_defensive_postproof_projection_checks_resource_before_next_finish(self):
        f, c, _ = self.batch(); f.generate(c, f.register(c, ("three", "four")))
        projected = copy.deepcopy(c.events); groups = [e for e in projected if e["kind"] == "batch_done"]
        groups[0]["shared"]["resources_closed"] = False
        # Explicit post-proof projection: J would forbid this public admission history.
        def finish(session, events):
            self.assertEqual(session, groups[0]["session_seq"])
            return groups[0]
        before = self.stored(c)
        with mock.patch.object(type(c), "events", new_callable=mock.PropertyMock, return_value=projected), \
             mock.patch.object(c, "_finish_batch", side_effect=finish) as checked, self.assertRaisesRegex(ValueError, "cleanup"):
            self.L._reconcile_generations(c, {"one", "two"})
        self.assertEqual(checked.call_count, 1); self.assertIsNotNone(c.db); self.assertEqual(self.stored(c), before)

    def test_pending_scalar_and_atomic_group_refresh_once_per_new_completion(self):
        for kind in ("scalar", "batch", "mixed"):
            f, c, item = self.scalar(True) if kind == "scalar" else self.batch(kind == "batch")
            if kind == "mixed": self.foreign_scalar(f, c)
            reads, observer = self.observe(c)
            with observer: events, own = self.L._reconcile_generations(c, {"one", "two"})
            self.assertEqual(len(reads), 2)
            self.assertEqual(len(own), 1 if kind == "scalar" else 2); self.assertEqual(events, c.events)
            self.assertEqual(own, {seq: c.outcome(seq) for seq in own})
            before = self.stored(c); reads, observer = self.observe(c)
            with observer: self.L._reconcile_generations(c, {"one", "two"})
            self.assertEqual(len(reads), 1); self.assertEqual(self.stored(c), before)

    def test_real_zero_output_keeps_both_row_completions_without_generation_replay(self):
        f, c, _ = self.batch(mode="zero"); before = self.stored(c)
        with mock.patch.object(f.H, "run", side_effect=AssertionError("Generation replay")):
            events, own = self.L._reconcile_generations(c, {"one", "two"})
            self.assertEqual(self.L._reconcile_generations(c, {"one", "two"}), (events, own))
        self.assertEqual(sum(e["kind"] == "generation_done" for e in events), 2)
        self.assertEqual(len(own), 2); self.assertTrue(all(v["tokens"] == [] and v["uid"] is None for v in own.values()))
        self.assertEqual(self.stored(c), before)

    def test_uncertain_scalar_completion_recovers_once_without_generation(self):
        for committed in (False, True):
            f, c, intent = self.scalar(True); append = c._append
            def uncertain(kind, **payload):
                if committed: append(kind, **payload)
                raise OSError("scalar acknowledgement")
            with mock.patch.object(c, "_append", side_effect=uncertain), self.assertRaisesRegex(OSError, "acknowledgement"):
                self.L._reconcile_generations(c, {"one"})
            self.assertIsNone(c.db)
            with mock.patch.object(f.api, "generate", side_effect=AssertionError("Replay")), self.J.Campaign(c.root) as reopened:
                self.assertEqual(sum(e["kind"] == "generation_done" for e in reopened.events), 1)
                self.assertEqual(reopened.outcome(intent["seq"])["status"], "stop")

    def test_uncertain_atomic_group_before_middle_after_recovers_exactly_once(self):
        for fault in ("before", "middle", "after"):
            f, c, _ = self.batch(True); original, database = c._append_group, c.db
            class Failure:
                inserts = 0
                def execute(self, statement, *args):
                    if statement.startswith("INSERT"):
                        self.inserts += 1
                        if self.inserts == 2: raise sqlite3.OperationalError("middle completion")
                    return database.execute(statement, *args)
                def close(self): database.close()
            def append(entries):
                if fault == "before": raise OSError("before completion")
                result = original(entries)
                if fault == "after": raise OSError("after completion")
                return result
            if fault == "middle": c.db = Failure()
            with mock.patch.object(c, "_append_group", side_effect=append), self.assertRaises((OSError, sqlite3.OperationalError)):
                self.L._reconcile_generations(c, {"one", "two"})
            self.assertIsNone(c.db)
            with mock.patch.object(f.H, "run", side_effect=AssertionError("Replay")), self.J.Campaign(c.root) as reopened:
                self.assertEqual(sum(e["kind"] == "batch_done" for e in reopened.events), 1)
                self.assertEqual(sum(e["kind"] == "generation_done" for e in reopened.events), 2)
                self.assertEqual(self.J.Campaign.totals(reopened.events)[1]["actor"]["replies"], 2)

    def test_finish_and_required_refresh_baseexceptions_close_preserving_failure(self):
        for stage in ("finish", "scalar_refresh", "batch_refresh"):
            f, c, _ = self.batch(True) if stage == "batch_refresh" else self.scalar(True)
            failure = KeyboardInterrupt(stage)
            _, observer = self.observe(c, failure)
            target = mock.patch.object(c, "_finish", side_effect=failure) if stage == "finish" else observer
            with target, self.assertRaises(KeyboardInterrupt) as raised: self.L._reconcile_generations(c, {"one", "two"})
            self.assertIs(raised.exception, failure); self.assertIsNone(c.db)
            with self.J.Campaign(c.root) as reopened:
                expected = 2 if stage == "batch_refresh" else 1
                self.assertEqual(sum(e["kind"] == "generation_done" for e in reopened.events), expected)

    def test_both_real_callers_keep_guard_contract_native_and_final_events_order(self):
        f, c, _ = self.batch(); record = next(e["record"] for e in c.events if e["kind"] == "experiment")
        helper, contract, native, register = self.L._reconcile_generations, f.N.verification.runner.contract, f.N.reconcile_native, c.register
        trace = []
        def checked(campaign, names): trace.append(("helper", names)); return helper(campaign, names)
        with mock.patch.object(self.L, "_reconcile_generations", side_effect=checked), \
             mock.patch.object(f.N.verification.runner, "contract", side_effect=lambda: (trace.append("contract"), contract())[1]), \
             mock.patch.object(f.N, "reconcile_native", side_effect=lambda campaign: (trace.append("native"), native(campaign))[1]), \
             mock.patch.object(c, "register", side_effect=lambda value: (trace.append("register"), register(value))[1]), \
             mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("Payload")):
            self.assertEqual(self.L._enter(c, f.loaded, record)[0], record)
        self.assertEqual(trace, [("helper", {record["experiment_id"]}), "contract", "native", "register"])
        trace.clear(); clear = c.require_native_clear
        def recovered(campaign):
            trace.append("native"); result = native(campaign)
            campaign._append("limits", limits=campaign._accounting()[0])  # Neutral marker isolates final-refresh routing.
            return result
        with mock.patch.object(self.W, "_guard", side_effect=lambda *args: trace.append("guard")), \
             mock.patch.object(self.L, "_reconcile_generations", side_effect=checked), mock.patch.object(f.N, "reconcile_native", side_effect=recovered), \
             mock.patch.object(c, "require_native_clear", side_effect=lambda: (trace.append("clear"), clear())[1]):
            events, own = self.W._live(c, None, {"one", "two"})
        self.assertEqual(trace, ["guard", ("helper", {"one", "two"}), "native", "clear"])
        self.assertEqual(events, c.events); self.assertEqual(events[-1]["kind"], "limits"); self.assertEqual(len(own), 2)
