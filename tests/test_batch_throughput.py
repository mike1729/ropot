import copy
import fcntl
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_precision_controls as controls, test_bf16_compatibility as process


class BatchThroughputTests(unittest.TestCase):
    F = controls.PrecisionControlTests
    fixture, put, roles = F.fixture, F.put, F.roles

    def setUp(self):
        self.F.setUp(self); self.B = importlib.import_module("experiments.batch_throughput")
        self.worker = self.root / "fake.py"; self.worker.write_text("# fixture worker")
        self.lock = self.root / "lock"; self.lock.write_bytes(b"same inode"); self.index = 0
        self.fixtures = [dict(id=name, messages=[dict(role="user", content=name)], seed=17 + i) for i, name in enumerate(("short", "long"))]
        config = dict(batch_size=2, prefill_batch_size=2, prefill_step_size=1024)
        row = lambda i, sampling="greedy", after=None: dict(fixture_id=self.fixtures[i % 2]["id"], seed=17 + i % 2, cap=1, sampling=sampling, after_row_index=after)
        self.windows = [dict(id=f"q{i}", mode="qualification", config={**config, "batch_size": 1 if i in (0, 3) else 2, "prefill_batch_size": 1 if i in (0, 3) else 2},
                            rows=[row(0), row(1)] if i < 2 else [row(0, "eos"), row(1, "length"), row(0, "sample", 0)] if i == 2 else [row(0, "sample")]) for i in range(4)]
        self.windows += [dict(id="warm", mode="warmup", config=config, rows=[row(0, "sample")]), dict(id="measured", mode="measure", config=config, rows=[row(0, "sample"), row(1, "sample")])]
        self.windows[0]["rows"][0]["cap"] = 2
        patch = mock.patch.object(self.B, "_sample", return_value=dict(cpu_seconds=0, rss_bytes=1, available_bytes=32 * 1024**3, swap_bytes=0))
        patch.start(); self.addCleanup(patch.stop)

    def plan(self, **changes):
        return self.B.build_plan(changes.get("request", self.raw_request), changes.get("fixtures", self.fixtures), changes.get("windows", self.windows), worker=self.worker, wall_seconds=changes.get("wall_seconds", 1))

    def rows(self):
        rows = [dict(event="start", pid=7), dict(event="controls_checked", request_sha256="a" * 64, proof=self.proof),
            dict(event="ready", fixtures=[dict(id=f["id"], prompt=f["id"], prompt_ids=[1, 2]) for f in self.fixtures], eos_token_ids=[9],
                 provenance=dict(precision_controls=self.proof, precision_postload=self.roles(True), sources=self.d["expected_runtime"]["sources"]), hash_seconds=0, load_seconds=0)]
        for window in self.windows:
            rows.append(dict(event="window_start", id=window["id"], started_seconds=1)); clock = 0
            for uid, row in enumerate(window["rows"]):
                rows.append(dict(event="admit", window=window["id"], uid=uid, row_index=uid, t_seconds=clock))
                for seq in range(1, row["cap"] + 1):
                    clock += .1; token = 9 if row["sampling"] == "eos" else 3
                    rows.append(dict(event="token", window=window["id"], uid=uid, seq=seq, token=token,
                        finish_reason="stop" if token == 9 else "length" if seq == row["cap"] else None, t_seconds=clock))
            rows.append(dict(event="window_done", window=window["id"], elapsed=clock + .1, stats=dict(generation_tokens=sum(r["cap"] for r in window["rows"])), memory=dict(active_bytes=1, cache_bytes=0, peak_bytes=1)))
            if window["id"] == "q3": rows.append(dict(event="qualification", checks=dict.fromkeys(("greedy_equal", "eos_at_cap", "length_cap", "reinsert_equal", "seed_independent"), True)))
        rows.append(dict(event="done"))
        return [dict(row, job_sha256="a" * 64) for row in rows]

    def parse(self, rows, tail=b"", pid=7):
        job = self.plan(); identity = self.P.sha256(self.B._raw(job)).hexdigest(); rows = copy.deepcopy(rows)
        for row in rows:
            row["job_sha256"] = identity
            if row["event"] == "controls_checked": row["request_sha256"] = identity
        return self.B.validate_events(b"".join((json.dumps(r) + "\n").encode() for r in rows) + tail, job, pid=pid)

    def invoke(self, mode="ok"):
        prefix = "MODE=" + repr(mode) + "\nEVENTS=" + repr(self.rows()) + "\n"
        prefix += "def emit_rows(rows):\n for row in rows:\n  row=dict(row,job_sha256=identity)\n  if row['event']=='controls_checked':row['request_sha256']=identity\n  emit(**row)\n"
        worker = process.FAKE.replace("sys.argv[-1]", "sys.argv[1]").replace("request_sha256=", "job_sha256=")
        worker = worker.replace("emit('phase',phase='partial')", "emit_rows(EVENTS[1:6])")
        worker = worker.replace("emit('done',job_sha256=identity,result={'fixture':True})", "emit_rows(EVENTS[1:])")
        worker += "\nassert os.fstat(json.loads((p.parent/'launch.json').read_text())['lease_fd']).st_ino==" + str(self.lock.stat().st_ino) + "\n"
        self.worker.write_text(prefix + worker); self.index += 1; directory = self.root / f"run-{self.index}"
        with mock.patch.object(self.B, "STDOUT_LIMIT", 512 if mode == "stdout" else self.B.STDOUT_LIMIT), mock.patch.object(self.B, "STDERR_LIMIT", 512 if mode == "stderr" else self.B.STDERR_LIMIT):
            result = self.B.run(self.plan(), directory, self.lock)
        self.assertEqual(result, json.loads((directory / "response.json").read_text()))
        self.assertTrue(result["group_absent"]); self.assertTrue(process.process_gone(result["pid"]))
        return result, directory

    def test_raw_request_normalizes_once_and_plan_bounds_precede_output(self):
        job = self.plan(); self.assertEqual(job["request"], self.G.prepare_request(self.raw_request)[0])
        before = copy.deepcopy(job); self.windows[0]["rows"][0]["seed"] = 999; self.assertEqual(job, before)
        self.windows[0]["rows"][0]["seed"] = 17
        for field, value in (("batch_size", True), ("batch_size", 3), ("prefill_batch_size", 3), ("prefill_step_size", 17)):
            windows = copy.deepcopy(self.windows); windows[0]["config"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.plan(windows=windows)
        for value in (True, 0, float("nan"), 1201):
            with self.assertRaises(ValueError): self.plan(wall_seconds=value)
        for field, value in (("cap", True), ("cap", 129), ("after_row_index", True), ("after_row_index", 0)):
            windows = copy.deepcopy(self.windows); windows[0]["rows"][0][field] = value
            with self.assertRaises(ValueError): self.plan(windows=windows)
        with self.assertRaises(ValueError): self.plan(fixtures=[self.fixtures[0], self.fixtures[0]])
        with self.assertRaises(ValueError): self.plan(windows=self.windows + [self.windows[-1]])
        invalid = self.plan(); invalid["windows"][0]["rows"][0]["cap"] = 129
        with self.assertRaisesRegex(ValueError, "Invalid row"): self.B.run(invalid, self.root / "invalid", self.lock)
        self.assertFalse((self.root / "invalid").exists())

    def test_full_protocol_uid_eos_length_and_counts(self):
        parsed = self.parse(self.rows()); self.assertTrue(parsed["complete"], parsed["protocol_error"])
        self.assertEqual((parsed["observed_generation_tokens"], parsed["finished_sequences"]), (12, 11))
        self.assertEqual(len(parsed["observations"]), 11)
        self.assertEqual([o["tokens"] for o in parsed["observations"] if o["window"] == "q2"], [[9], [3], [3]])
        self.assertFalse(self.parse(self.rows(), pid=8)["complete"])
        for anchor in (0, -1, True, float("inf")):
            rows = self.rows(); next(r for r in rows if r["event"] == "window_start")["started_seconds"] = anchor
            with self.subTest(anchor=anchor): self.assertFalse(self.parse(rows)["complete"])

    def test_otherwise_valid_protocol_mutations_and_partial_prefix(self):
        for fault in ("proof", "source", "uid", "index", "row", "after", "finish", "eos", "time", "cap", "checks", "order", "afterdone", "count", "memory", "duplicate", "postfinish"):
            rows = copy.deepcopy(self.rows()); tokens = [r for r in rows if r["event"] == "token"]
            if fault == "proof": rows[1]["proof"]["gpu"]["available"] = 1
            if fault == "source": rows[2]["provenance"]["sources"][next(iter(self.sources))] = "0" * 64
            if fault in ("uid", "index", "finish", "eos", "time"): tokens[0][dict(uid="uid", index="seq", finish="finish_reason", eos="token", time="t_seconds")[fault]] = dict(uid=99, index=2, finish="stop", eos=9, time=10**400)[fault]
            if fault == "row": next(r for r in rows if r["event"] == "admit")["row_index"] = 99
            if fault == "after":
                late = next(r for r in rows if r["event"] == "admit" and r["window"] == "q2" and r["row_index"] == 2); rows.remove(late); rows.insert(next(i for i, r in enumerate(rows) if r.get("id") == "q2") + 1, late)
            if fault == "cap": tokens[1]["finish_reason"] = None
            if fault == "checks": next(r for r in rows if r["event"] == "qualification")["checks"]["greedy_equal"] = 1
            if fault == "order": rows[1], rows[2] = rows[2], rows[1]
            if fault == "afterdone": rows.append(rows[-1])
            if fault == "count": next(r for r in rows if r["event"] == "window_done")["stats"]["generation_tokens"] += 1
            if fault == "memory": next(r for r in rows if r["event"] == "window_done")["memory"].pop("peak_bytes")
            if fault == "duplicate": rows.insert(5, copy.deepcopy(rows[4]))
            if fault == "postfinish": rows.insert(rows.index(tokens[1]) + 1, {**tokens[1], "seq": 3})
            with self.subTest(fault=fault): self.assertFalse(self.parse(rows)["complete"]); self.assertIsNotNone(self.parse(rows)["protocol_error"])
        for tail in (b"{", b"[]\n", b'{"event":"done","x":NaN}\n'):
            parsed = self.parse(self.rows()[:6], tail); self.assertFalse(parsed["complete"]); self.assertEqual(parsed["observed_generation_tokens"], 1)
        self.assertFalse(self.parse(self.rows(), b"{")["complete"])
        job = self.plan(); identity = self.P.sha256(self.B._raw(job)).hexdigest()
        prefix = b"".join(self.B._raw({**r, "job_sha256": identity, **({"request_sha256": identity} if r["event"] == "controls_checked" else {})}) for r in self.rows()[:6])
        with mock.patch.object(self.B, "STDOUT_LIMIT", len(prefix) + 1): parsed = self.B.validate_events(prefix + b"excess\n", job, pid=7)
        self.assertEqual(parsed["observed_generation_tokens"], 1); self.assertFalse(parsed["complete"]); self.assertIn("Output limit", parsed["protocol_error"])

    def test_owned_worker_timeout_partial_grandchild_and_existing_lock(self):
        for mode, status in (("ok", "passed"), ("pid", "worker_error"), ("hash", "worker_error"), ("timeout", "timeout"), ("stdout", "output_limit"), ("stderr", "output_limit")):
            result, directory = self.invoke(mode); self.assertEqual(result["status"], status)
            self.assertEqual(result["job_sha256"], self.P.sha256((directory / "job.json").read_bytes()).hexdigest())
            if mode == "timeout": self.assertEqual(result["observed_generation_tokens"], 1); self.assertTrue(process.process_gone(int((directory / "child.pid").read_text())))
        with self.lock.open("rb") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(self.B.subprocess, "Popen") as launch, self.assertRaises(BlockingIOError): self.B.run(self.plan(), self.root / "held", self.lock)
            launch.assert_not_called()
        missing = self.root / "missing-lock"
        with self.assertRaises(FileNotFoundError): self.B.run(self.plan(), self.root / "missing", missing)
        self.assertFalse(missing.exists())

    def test_cancellation_cleanup_failure_and_raw_fsync_precede_receipt(self):
        synced = set(); sync, write = self.B.os.fsync, self.B._write
        def observed(fd): synced.add(self.B.os.fstat(fd).st_ino); return sync(fd)
        def receipt(path, value):
            if Path(path).name == "response.json": self.assertTrue({(Path(path).parent / n).stat().st_ino for n in ("stdout.jsonl", "stderr.txt")} <= synced)
            return write(path, value)
        with mock.patch.object(self.B.os, "fsync", side_effect=observed), mock.patch.object(self.B, "_write", side_effect=receipt): self.invoke()
        with mock.patch.object(self.B, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt): self.invoke("timeout")
        cancelled = json.loads((self.root / f"run-{self.index}" / "response.json").read_text()); self.assertEqual(cancelled["status"], "cancelled"); self.assertTrue(cancelled["group_absent"])
        kill = self.B._kill_group
        def uncertain(child): kill(child); raise PermissionError("fixture cleanup")
        with mock.patch.object(self.B, "_kill_group", side_effect=uncertain): result, _ = self.invoke()
        self.assertNotEqual(result["status"], "passed"); self.assertIn("fixture cleanup", result["cleanup_error"])
