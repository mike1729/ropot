import copy
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_batch_reservation as reservation, test_program_wave as wave


class BatchSettlementTests(unittest.TestCase):
    def setUp(self):
        self.b = reservation.BatchReservationTests(methodName="runTest"); self.b.setUp(); self.addCleanup(self.b.doCleanups)
        self.H, self.J, self.M = self.b.H, self.b.J, self.b.q.f.M
        self.b.q.files = self.b.q.pack(dict(self.b.q.job, cache_limit_bytes=0), self.b.q.events)
        self.b.q.pin = self.H.sha256(self.b.q.files["response.json"]).hexdigest()
        self.b.method.update(admission=self.H.qualification_checked(self.b.q.files),
            qualification=dict(path="qualifications/" + self.b.q.pin, response_sha256=self.b.q.pin))

    def runner(self, fault=None):
        def run(job, directory, lock, *, lease_fd, qualification_files):
            if fault == "preoutput": raise KeyboardInterrupt("before output")
            rows = lambda: [dict(row_id=r["row_id"], request={k: v for k, v in json.loads(r["request_text"]).items() if k != "environment"}) for r in job["rows"]]
            prototype = NS(M=self.M, H=self.H, q=self.b.q, admission=self.b.method["admission"], rows=rows)
            built, events, files, decoded = wave.ProgramWaveTests.protocol(prototype); self.assertEqual(built, job)
            if fault in ("timeout", "zero"):
                events = events[:-2] if fault == "timeout" else events[:2]
                if fault == "timeout": events[-1].update(token=3, finish_reason=None)
                files = self.b.q.pack(job, events)
            response = json.loads(files["response.json"])
            if fault in ("timeout", "zero"): response.update(status="timeout", returncode=-9, work_seconds=241, elapsed_seconds=241.5)
            if fault == "pid": response["pid"] = 8
            if fault == "cleanup": response["cleanup_error"] = "unproven cleanup"
            files["response.json"] = self.H._raw(response); files["decoded.jsonl"] = decoded; files["stderr.txt"] = b""
            files["telemetry.jsonl"] = b"".join(self.H._raw(s) for s in response["telemetry"])
            sources = list(job["source_hashes"]); lease = os.fstat(lease_fd)
            files["launch.json"] = self.H._raw(dict(argv=[str(self.H.G.INTERPRETER), "-I", "-B", job["worker"], str(directory / "job.json")],
                environment=job["request"]["environment"], lease_fd=lease_fd, lease_dev=lease.st_dev, lease_ino=lease.st_ino, sources=sources))
            files.update({f"source-{i}.py": Path(path).read_bytes() for i, path in enumerate(sources)})
            directory.mkdir(); (directory / "requests").mkdir()
            self.H.copy_qualification(qualification_files, directory / "qualification", self.b.q.pin)
            for i, row in enumerate(job["rows"]): (directory / "requests" / f"{i}.json").write_bytes(row["request_text"].encode())
            for name, raw in files.items(): (directory / name).write_bytes(raw)
            altered = {"qual_missing": "qualification", "qual_partial": "qualification/stdout.jsonl", "qual_changed": "qualification/response.json",
                "job": "job.json", "request": "requests/1.json", "cache": "cache-control.json", "decoded": "decoded.jsonl",
                "source": "source-5.py", "launch": "launch.json", "telemetry": "telemetry.jsonl", "missing": "stderr.txt"}
            if fault == "qual_missing":
                for path in (directory / "qualification").iterdir(): path.unlink()
                (directory / "qualification").rmdir()
            elif fault in ("qual_partial", "missing"): (directory / altered[fault]).unlink()
            elif fault == "cache":
                proof = json.loads(files["cache-control.json"]); proof["observed_bytes"] = False
                (directory / altered[fault]).write_bytes(self.H._raw(proof))
            elif fault in altered: (directory / altered[fault]).write_bytes(b"changed")
            if fault == "extra": (directory / "unexpected").write_bytes(b"extra")
            if fault == "symlink":
                (directory / "stderr.txt").unlink(); (directory / "stderr.txt").symlink_to(directory / "stdout.jsonl")
            return response
        return run

    def execute(self, fault=None):
        c, rows = self.b.campaign()
        with mock.patch.object(self.H, "run", side_effect=self.runner(fault)) as run:
            try: result = c.generate_batch(rows, self.b.method)
            except self.J.ProfileControlError: result = None
        return c, rows, result, run

    def test_cross_record_success_shared_process_wall_once_and_exact_owned_inventory(self):
        c, _, outcomes, run = self.execute(); self.assertEqual(run.call_count, 1)
        done = [e for e in c.events if e["kind"] == "batch_done"]; self.assertEqual(len(done), 1); shared = done[0]
        self.assertTrue(shared["eligible"]); self.assertTrue(shared["session_copy_complete"]); self.assertEqual(shared["evidence_errors"], [])
        self.assertEqual(shared["shared"]["elapsed_seconds"], 239.5); self.assertTrue(shared["shared"]["resources_closed"])
        self.assertEqual([r["observed_generation_tokens"] for r in outcomes], [2, 1])
        self.assertEqual([r["text"] for r in outcomes], ["print(1)", ""])
        self.assertTrue(all(r["eligible"] and r["batch_done_seq"] == shared["seq"] and r["session_seq"] == shared["session_seq"] for r in outcomes))
        for row in outcomes:
            self.assertFalse({"response", "pid", "returncode", "elapsed_seconds", "evidence"} & row.keys())
            self.assertTrue(self.J.generation_resources_cleared(c.events, row)); self.assertIsNone(self.J.profile_control_reason(row))
        self.assertEqual(self.J.Campaign.totals(c.events)[1]["actor"], dict(zip(self.J.UNITS, (2, 4096, 480))))
        path = c.root / "sessions" / f"{shared['session_seq']:06d}"
        self.assertEqual({str(p.relative_to(c.root)) for p in path.rglob("*") if p.is_file()}, {p for p in shared["evidence"] if p.startswith("sessions/")})
        self.assertEqual(run.call_args.args[2], c.root / "lock"); self.assertEqual(run.call_args.kwargs["lease_fd"], c._lock.fileno())

    def test_closed_timeout_retains_terminal_peer_cost_and_permits_new_work(self):
        for fault in ("timeout", "zero"):
            c, _, result, _ = self.execute(fault); self.assertTrue(all(not r["eligible"] and r["status"] == "interrupted" for r in result))
            self.assertEqual([r["observed_generation_tokens"] for r in result], [2, 1] if fault == "timeout" else [0, 0])
            self.assertTrue(all(self.J.generation_resources_cleared(c.events, r) for r in result)); c.require_native_clear()
            record = next(e["record"] for e in c.events if e["kind"] == "experiment"); record = copy.deepcopy(record); record["experiment_id"] = "later"; c.register(record)
            later = dict(experiment_id="later", source="actor", request=self.b.raw, attempt=dict(split="test", id=1, task_sha256="a" * 64, reply_index=0))
            self.assertEqual(len(c.reserve_batch([later], self.b.method)["rows"]), 1)

    def test_otherwise_valid_owned_artifact_failures_never_admit_eligible_rows(self):
        for fault in ("qual_missing", "qual_partial", "qual_changed", "job", "request", "cache", "decoded", "source", "launch", "telemetry", "missing", "extra"):
            with self.subTest(fault=fault):
                c, _, result, _ = self.execute(fault); self.assertTrue(all(not r["eligible"] for r in result))
                done = next(e for e in c.events if e["kind"] == "batch_done"); self.assertFalse(done["eligible"])
                self.assertTrue(done["evidence_errors"] or done["shared"]["errors"])
        with self.assertRaisesRegex(ValueError, "session entry"): self.execute("symlink")

    def test_unknown_closure_and_preoutput_interrupt_globally_block_without_relaunch(self):
        for fault in ("pid", "cleanup", "preoutput"):
            c, rows = self.b.campaign()
            with mock.patch.object(self.H, "run", side_effect=self.runner(fault)) as run:
                try: c.generate_batch(rows, self.b.method)
                except (self.J.ProfileControlError, KeyboardInterrupt): self.assertNotEqual(fault, "cleanup")
                else: self.assertEqual(fault, "cleanup")
            self.assertEqual(run.call_count, 1); root = c.root; c.close()
            with mock.patch.object(self.H, "run", side_effect=AssertionError("Replay")), self.J.Campaign(root) as reopened:
                done = next(e for e in reopened.events if e["kind"] == "batch_done"); self.assertFalse(done["shared"]["resources_closed"])
                refs = [e["seq"] for e in reopened.events if e["kind"] == "generation_reserved"]
                self.assertTrue(all(not reopened.outcome(seq)["eligible"] for seq in refs))
                with self.assertRaises(ValueError): reopened.require_native_clear()

    def test_completion_before_middle_after_commit_uncertainty_recovers_exactly_once(self):
        for fault in ("before", "middle", "after"):
            c, rows = self.b.campaign(); original = c._append_group; calls = 0; database = c.db
            class Failure:
                inserts = 0
                def execute(self, statement, *args):
                    if statement.startswith("INSERT"):
                        self.inserts += 1
                        if self.inserts == 2: raise sqlite3.OperationalError("middle completion")
                    return database.execute(statement, *args)
                def close(self): database.close()
            def append(entries):
                nonlocal calls
                calls += 1
                if calls == 2 and fault == "before": raise OSError("before completion")
                if calls == 2 and fault == "middle": c.db = Failure()
                result = original(entries)
                if calls == 2 and fault == "after": raise OSError("uncertain completion")
                return result
            with mock.patch.object(c, "_append_group", side_effect=append), mock.patch.object(self.H, "run", side_effect=self.runner()) as run:
                with self.assertRaises((OSError, sqlite3.OperationalError)): c.generate_batch(rows, self.b.method)
            self.assertEqual(run.call_count, 1); self.assertIsNone(c.db)
            before = [json.loads(body) for _, body in self.b.stored(c.root)]
            self.assertEqual(sum(e["kind"] == "generation_done" for e in before), 2 if fault == "after" else 0)
            with mock.patch.object(self.H, "run", side_effect=AssertionError("Replay")), self.J.Campaign(c.root) as reopened:
                events = reopened.events; self.assertEqual(sum(e["kind"] == "batch_done" for e in events), 1)
                self.assertEqual(sum(e["kind"] == "generation_done" for e in events), 2)

    def test_offline_idempotence_and_postcompletion_retained_evidence_tamper_are_fatal(self):
        c, _, result, _ = self.execute(); root = c.root; before = self.b.stored(root); c.close()
        current_paths = set(self.b.method["admission"]["source_hashes"]); read = Path.read_bytes
        def offline(path):
            if str(path) in current_paths: raise AssertionError("Original source read")
            return read(path)
        with mock.patch.object(Path, "read_bytes", offline), mock.patch.object(self.H, "run", side_effect=AssertionError("Replay")), self.J.Campaign(root) as reopened:
            self.assertEqual([reopened.outcome(r["reservation_seq"]) for r in result], result)
        self.assertEqual(self.b.stored(root), before)
        shared = next(json.loads(body) for _, body in before if json.loads(body)["kind"] == "batch_done")
        path = root / next(p for p in shared["evidence"] if p.endswith("stderr.txt")); path.write_bytes(b"late mutation")
        with self.assertRaises(ValueError): self.J.Campaign(root)
        c, _, _, _ = self.execute(); root = c.root; c.close()
        (root / self.b.method["qualification"]["path"] / "response.json").write_bytes(b"changed authority")
        with self.assertRaises(ValueError): self.J.Campaign(root)

    def test_atomic_completion_replay_rejects_partial_interleaved_duplicate_and_wrong_binding(self):
        for fault in ("missing", "interleaved", "duplicate", "binding", "job"):
            c, _, _, _ = self.execute(); events = c.events; c.close()
            if fault == "missing": events.pop()
            if fault == "interleaved": events[-1].update(kind="limits", limits=events[0]["limits"])
            if fault == "duplicate": events[-1]["reservation_seq"] = events[-2]["reservation_seq"]
            if fault == "binding": events[-1]["batch_done_seq"] += 1
            if fault == "job": events[-3]["job_sha256"] = "0" * 64
            with sqlite3.connect(c.root / "journal.sqlite3") as db:
                db.execute("DELETE FROM events")
                db.executemany("INSERT INTO events(seq,body) VALUES (?,?)", [(e["seq"], self.J.encoded({k: v for k, v in e.items() if k != "seq"})) for e in events])
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.J.Campaign(c.root)

    def test_batch_profile_classifier_retains_legacy_semantics_without_fabricated_failure(self):
        base = dict(method="program-wave-v1", profile_guarded=True, profile_ready=True, protocol_error=None)
        for fields, expected in (({}, None), ({"profile_ready": False}, None), ({"profile_guarded": False}, "profile_unproven"),
                ({"protocol_error": "Missing done"}, None), ({"protocol_error": "Incomplete protocol tail"}, None), ({"protocol_error": "invalid proof"}, "profile_protocol")):
            self.assertEqual(self.J.profile_control_reason(dict(base, **fields)), expected)
        for fields in ({"profile_guarded": 1}, {"profile_ready": None}, {"protocol_error": False}):
            with self.assertRaises(ValueError): self.J.profile_control_reason(dict(base, **fields))
        self.assertIsNone(self.J.profile_control_reason(dict(status="interrupted")))
        self.assertEqual(self.J.profile_control_reason(dict(profile_guarded=True, profile_failure=dict(code="guard"), protocol_error=None)), "profile_failure")
        with self.assertRaises(ValueError): self.J.profile_control_reason(dict(profile_guarded=True))
