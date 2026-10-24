import copy
import importlib
import json
import signal
import subprocess
import unittest
from unittest import mock
from tests import test_precision_adoption as adoption, test_study_tasks as study


class KillGroupObservationTests(unittest.TestCase):
    def test_reaped_group_probe_is_bounded_and_never_signals_again(self):
        runner = importlib.import_module("src.data-pipeline.local_runner")
        for mode in ("race", "live", "denied", "io", "cancel"):
            child = mock.Mock(pid=17); original = PermissionError("initial EPERM"); probes = []
            failure = OSError("probe failed") if mode == "io" else KeyboardInterrupt()
            def probe(pid, sig):
                probes.append(sig)
                if sig == signal.SIGKILL: raise original
                if mode == "race" and len(probes) > 2: raise ProcessLookupError()
                if mode in ("race", "denied"): raise PermissionError("transient")
                if mode in ("io", "cancel"): raise failure
            with self.subTest(mode=mode), mock.patch.object(runner.os, "killpg", side_effect=probe), mock.patch.object(runner, "monotonic", side_effect=range(20)), mock.patch.object(runner, "sleep"):
                if mode == "race": runner._kill_group(child)
                else:
                    with self.assertRaises(type(original) if mode in ("live", "denied") else type(failure)) as caught: runner._kill_group(child)
                    self.assertIs(caught.exception, original if mode in ("live", "denied") else failure)
            child.wait.assert_called_once_with(timeout=5); child.kill.assert_not_called()
            self.assertEqual(probes[0], signal.SIGKILL); self.assertTrue(all(s == 0 for s in probes[1:])); self.assertLessEqual(len(probes), 7)
        with mock.patch.object(runner.os, "killpg") as kill:
            runner._kill_group(child); kill.assert_called_once_with(child.pid, signal.SIGKILL)
        child.reset_mock(); child.wait.side_effect = subprocess.TimeoutExpired("fixture", 5)
        with mock.patch.object(runner.os, "killpg", side_effect=original) as kill, self.assertRaises(PermissionError): runner._kill_group(child)
        child.kill.assert_called_once()
        child.wait.assert_called_once_with(timeout=5); kill.assert_called_once_with(child.pid, signal.SIGKILL)

    def test_delayed_child_exit_is_reaped_before_proving_group_absence(self):
        runner = importlib.import_module("src.data-pipeline.local_runner")
        child = mock.Mock(pid=17); reaped = False; signals = []
        def wait(timeout):
            nonlocal reaped
            if timeout < 0.5: raise subprocess.TimeoutExpired("delayed exit", timeout)
            reaped = True
            return 0
        def signal_group(pid, sig):
            self.assertEqual(pid, child.pid); signals.append(sig)
            if sig == signal.SIGKILL: raise PermissionError("exiting group")
            self.assertTrue(reaped)
            raise ProcessLookupError()
        child.wait.side_effect = wait
        with mock.patch.object(runner.os, "killpg", side_effect=signal_group): runner._kill_group(child)
        child.wait.assert_called_once_with(timeout=5); child.kill.assert_not_called()
        self.assertEqual(signals, [signal.SIGKILL, 0])


class GenerationCleanupTests(unittest.TestCase):
    F = adoption.PrecisionAdoptionTests
    write_task, rewrite, limits, rows = F.write_task, F.rewrite, F.limits, F.rows
    read, execute, replies, recipe, fresh = F.read, F.execute, F.replies, F.recipe, F.fresh
    fixture, put, mode, record, request, reopen = F.fixture, F.put, F.mode, F.record, F.request, F.reopen
    feasibility_loaded = study.StudyTaskTests.feasibility_loaded

    def setUp(self): self.F.setUp(self)

    def failed(self, campaign=None, record=None):
        campaign = campaign or self.fresh(); record = record or self.record(campaign, "failed", sources=("actor",))
        kill = self.api._kill_group
        def uncertain(child): kill(child); raise PermissionError("fixture cleanup")
        with mock.patch.object(self.api, "_kill_group", side_effect=uncertain), self.assertRaisesRegex(PermissionError, "fixture cleanup"):
            self.L.run(campaign, self.loaded, record)
        campaign = self.reopen(campaign)
        seq = next(e["seq"] for e in campaign.events if e["kind"] == "generation_reserved")
        return campaign, campaign.outcome(seq), record

    def clear(self, campaign, seq):
        with mock.patch.object(self.J.os, "kill", side_effect=ProcessLookupError()) as pid, mock.patch.object(self.J.os, "killpg", side_effect=ProcessLookupError()) as pgid:
            event = campaign.clear_generation_resources(seq)
        pid.assert_called_once_with(event["pid"], 0); pgid.assert_called_once_with(event["pid"], 0)
        return event

    def test_clearance_appends_only_proof_preserves_failed_evidence_and_is_idempotent(self):
        campaign, outcome, record = self.failed(); before = campaign.events; totals = campaign._accounting()
        directory = campaign.root / "requests" / f"{outcome['reservation_seq']:06d}"
        files = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.assertEqual(totals[1]["actor"], dict(replies=1, reserved_output_tokens=1, reserved_generation_wall_seconds=0.5))
        self.assertFalse(self.J.generation_resources_cleared(before, outcome))
        marker = self.clear(campaign, outcome["reservation_seq"])
        self.assertEqual(set(marker), {"seq", "kind", "reservation_seq", "response_sha256", "pid"})
        self.assertEqual(campaign.events[:-1], before); self.assertEqual(campaign._accounting(), totals)
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, files); self.assertEqual(campaign.outcome(outcome["reservation_seq"]), outcome)
        with mock.patch.object(self.J.os, "kill", side_effect=AssertionError("historical PID probe")), mock.patch.object(self.J.os, "killpg", side_effect=AssertionError("historical group probe")):
            self.assertTrue(self.J.generation_resources_cleared(campaign.events, outcome))
            self.assertEqual(campaign.clear_generation_resources(outcome["reservation_seq"]), marker)
        self.assertEqual(campaign.events[-1], marker)
        with mock.patch.object(self.api, "generate", side_effect=AssertionError("failed reply retried")):
            cells = self.L.run(campaign, self.loaded, record)
        self.assertEqual((cells[0]["status"], cells[0]["reason"]), ("blocked", "generation_error"))
        self.assertEqual(campaign._accounting(), totals)

    def test_both_probes_require_esrch_and_errors_close_without_appending(self):
        campaign, outcome, _ = self.failed(); before = campaign.events
        for pid_effect, group_effect, expected in ((None, ProcessLookupError(), "Generation process is still present"),
                (ProcessLookupError(), None, "Generation group is still present"), (PermissionError("unknown"), ProcessLookupError(), "unknown"),
                (ProcessLookupError(), PermissionError("unknown group permission"), "unknown group permission"),
                (ProcessLookupError(), OSError("unknown group"), "unknown group"), (KeyboardInterrupt(), ProcessLookupError(), None)):
            with mock.patch.object(self.J.os, "kill", side_effect=pid_effect), mock.patch.object(self.J.os, "killpg", side_effect=group_effect):
                if expected:
                    with self.assertRaisesRegex((ValueError, OSError), expected): campaign.clear_generation_resources(outcome["reservation_seq"])
                else:
                    with self.assertRaises(KeyboardInterrupt): campaign.clear_generation_resources(outcome["reservation_seq"])
            self.assertIsNone(campaign.db); campaign = self.reopen(campaign); self.assertEqual(campaign.events, before)

    def test_uncertain_insert_reopens_without_rewriting_evidence_or_double_clearance(self):
        for committed in (False, True):
            campaign, outcome, _ = self.failed(); append = campaign._append; before = campaign.events
            def uncertain(kind, **payload):
                if kind != "generation_resources_cleared": return append(kind, **payload)
                if committed: append(kind, **payload)
                raise OSError("uncertain clearance insert")
            with mock.patch.object(campaign, "_append", side_effect=uncertain), mock.patch.object(self.J.os, "kill", side_effect=ProcessLookupError()), mock.patch.object(self.J.os, "killpg", side_effect=ProcessLookupError()), self.assertRaisesRegex(OSError, "uncertain clearance insert"):
                campaign.clear_generation_resources(outcome["reservation_seq"])
            self.assertIsNone(campaign.db); campaign = self.reopen(campaign)
            self.assertEqual(campaign.events[:len(before)], before); self.assertEqual(len(campaign.events), len(before) + committed)
            if committed:
                with mock.patch.object(self.J.os, "kill", side_effect=AssertionError("committed proof reprobed")): campaign.clear_generation_resources(outcome["reservation_seq"])
            else: self.clear(campaign, outcome["reservation_seq"])
            self.assertEqual(campaign.outcome(outcome["reservation_seq"]), outcome)

    def test_marker_structure_chronology_uniqueness_and_retained_binding(self):
        campaign, outcome, _ = self.failed(); marker = self.clear(campaign, outcome["reservation_seq"])
        body = self.J.encoded({k: v for k, v in marker.items() if k != "seq"})
        for field, value, error in (("extra", 1, "Invalid generation resource clearance"), ("pid", True, "Invalid generation resource clearance"),
                ("reservation_seq", 0, "Invalid generation resource clearance"), ("response_sha256", "bad", "Invalid generation resource clearance"),
                ("response_sha256", "0" * 64, "Generation resource clearance differs"), ("pid", marker["pid"] + 1, "Generation resource clearance differs")):
            changed = json.loads(body); changed[field] = value
            campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (self.J.encoded(changed), marker["seq"]))
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, error): self.J.generation_resources_cleared(campaign.events, outcome)
            campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (body, marker["seq"]))
        duplicate = campaign._append(marker["kind"], **{k: v for k, v in marker.items() if k not in ("seq", "kind")})
        with self.assertRaisesRegex(ValueError, "Invalid generation resource clearance"): campaign.events
        campaign.db.execute("DELETE FROM events WHERE seq=?", (duplicate["seq"],))
        done = next(e for e in campaign.events if e["kind"] == "generation_done")
        saved = self.J.encoded({k: v for k, v in done.items() if k != "seq"})
        campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (body, done["seq"]))
        campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (saved, marker["seq"]))
        with self.assertRaisesRegex(ValueError, "Invalid generation resource clearance"): campaign.events
        campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (saved, done["seq"]))
        campaign.db.execute("UPDATE events SET body=? WHERE seq=?", (body, marker["seq"]))
        changed = copy.deepcopy(outcome); changed["observed_generation_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "Generation is not eligible for resource clearance"): self.J.generation_resources_cleared(campaign.events, changed)

    def test_narrow_eligibility_and_missing_response_never_clear_unsupported_failures(self):
        campaign, outcome, _ = self.failed(); before = campaign.events
        variants = [("status", "cancelled"), ("finish_reason", None), ("profile_guarded", False), ("profile_failure", dict(code="runtime", detail="fixture")),
            ("protocol_error", "Events after done"), ("complete", False), ("returncode", True), ("returncode", None), ("returncode", -9),
            ("parent_error", "cancelled"), ("cleanup_error", ""), ("pid", True), ("pid", 0)]
        for field, value in variants:
            changed = copy.deepcopy(outcome); target = changed if field in outcome else changed["response"]; target[field] = value
            events = copy.deepcopy(before)
            if field in outcome: next(e for e in events if e["kind"] == "generation_done")[field] = value
            with mock.patch.object(campaign, "outcome", return_value=changed), mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), mock.patch.object(self.J.os, "kill", side_effect=AssertionError("ineligible PID probe")), self.assertRaisesRegex(ValueError, "Generation is not eligible for resource clearance"):
                campaign.clear_generation_resources(outcome["reservation_seq"])
            self.assertIsNone(campaign.db); campaign = self.reopen(campaign); self.assertEqual(campaign.events, before)
        legacy = copy.deepcopy(outcome)
        for key in ("profile_guarded", "profile_failure", "protocol_error"): legacy.pop(key)
        self.assertFalse(self.J.generation_resources_cleared(campaign.events, legacy))
        with mock.patch.object(campaign, "outcome", return_value=legacy), self.assertRaisesRegex(ValueError, "Generation is not eligible for resource clearance"): campaign.clear_generation_resources(outcome["reservation_seq"])
        campaign = self.reopen(campaign); self.clear(campaign, outcome["reservation_seq"])
        missing = copy.deepcopy(outcome); missing["response"] = None
        with self.assertRaisesRegex(ValueError, "Generation is not eligible for resource clearance"): self.J.generation_resources_cleared(campaign.events, missing)
        self.assertTrue(self.J.generation_resources_cleared([], dict(reservation_seq=1, response=None)))

    def test_collection_and_loop_global_barriers_only_release_exact_cleared_failure(self):
        C = importlib.import_module("src.data-pipeline.paired_collection"); self.loaded = self.feasibility_loaded()
        for task in self.loaded["cohort"]["tasks"]: task["sha256"] = self.P.sha256(str(task["id"]).encode()).hexdigest()
        self.loaded["cohort"]["tasks"][0] = copy.deepcopy(self.records[1]); self.loaded["cohort"]["queues"]["development"]["initial"][0] = 1
        campaign = self.fresh(); audit = self.root / "audit"; audit.write_bytes(b"fixture audit")
        with mock.patch.object(C, "RECIPE", self.recipe((1,))), mock.patch.object(C, "AUDIT_SHA256", self.P.sha256(audit.read_bytes()).hexdigest()), mock.patch.object(study.study, "load_cohort", return_value=self.loaded):
            setup = lambda: C.setup(campaign, self.loaded, "collection", self.original / "receipt.json", self.identity, audit)
            anchor = setup(); campaign, outcome, _ = self.failed(campaign)
            later = self.record(campaign, "later", 16, ("actor",)); before = campaign.events; costs = self.R.snapshot(campaign)["report"]["costs"]
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("premature generation")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("premature native")):
                for action in (setup, lambda: self.L.run(campaign, self.loaded, later)):
                    with self.assertRaisesRegex(ValueError, "Unresolved generation cleanup failure"): action()
            self.assertEqual(campaign.events, before); self.clear(campaign, outcome["reservation_seq"])
            self.assertEqual(setup(), anchor); self.assertEqual(self.R.snapshot(campaign)["report"]["costs"], costs)
            self.assertEqual(self.L.run(campaign, self.loaded, later)[0]["status"], "verified")
            self.assertEqual(campaign._accounting()[1]["actor"]["replies"], 2)
