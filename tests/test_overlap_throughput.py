import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
from tests import test_batch_throughput as fixture

CHILD = '''
import json,os,sys,time
from pathlib import Path
try:
 Path(sys.argv[1]+'.ready').write_text('ready')
 while True:time.sleep(.01)
except KeyboardInterrupt:
 Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(),interrupted=time.monotonic())))
 deadline=time.monotonic()+3
 while not Path(sys.argv[2]).exists() and time.monotonic()<deadline:time.sleep(.01)
'''


class OverlapThroughputTests(unittest.TestCase):
    def test_cli_binds_lock_to_the_run_parameter(self):
        module = importlib.import_module("experiments.overlap_throughput")
        with mock.patch.object(module, "run", return_value=dict(status="passed", attempted={}, serial_over_overlap_ratio=1)) as run, mock.patch("builtins.print"):
            self.assertFalse(module.main(["--profiles", "p", "--winner", "w", "--output", "o", "--lock", "l"]))
        run.assert_called_once_with(profiles_path="p", winner_path="w", output="o", lock_path="l")

    def setUp(self):
        self.f = fixture.BatchThroughputTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.O = importlib.import_module("experiments.overlap_throughput"); self.H = self.f.B; self.root = self.f.root
        self.f.windows.append({**copy.deepcopy(self.f.windows[-1]), "id": "measured2"})
        self.directory = self.root / "gpu"; self.directory.mkdir(); self.job = self.f.plan()
        self.child = dict(pid=11, launch_seconds=90, reap_seconds=200, returncode=0, group_absent=True)

    def evidence(self, complete=False, cache=None):
        job = copy.deepcopy(self.job)
        if cache is not None: job["cache_limit_bytes"] = cache
        identity = self.f.P.sha256(self.H._raw(job)).hexdigest(); rows = copy.deepcopy(self.f.rows())
        for i, row in enumerate(rows):
            row["job_sha256"] = identity
            if row["event"] == "controls_checked": row["request_sha256"] = identity
            if row["event"] == "window_start": row["started_seconds"] = 100 + i
        end = next(i for i, r in enumerate(rows) if r["event"] == "window_start" and r["id"] == "measured")
        raw = b"".join(self.H._raw(r) for r in (rows if complete else rows[:end + 1]))
        (self.directory / "job.json").write_bytes(self.H._raw(job)); (self.directory / "stdout.jsonl").write_bytes(raw)
        if complete:
            result = dict(self.H.validate_events(raw, job, pid=7), status="passed", pid=7, returncode=0, job_sha256=identity,
                group_absent=True, cleanup_error=None, parent_error=None, telemetry_complete=True, elapsed_seconds=2, cleanup_seconds=.01)
            (self.directory / "response.json").write_bytes(self.H._raw(result))
        return job, rows, identity

    def test_interval_unions_exclude_gaps_and_reject_nonfinite_endpoints(self):
        self.assertEqual(self.O.union([(3, 5), (1, 2), (2, 4), (9, 10)]), [[1, 5], [9, 10]])
        self.assertEqual(self.O.intersection_seconds([(1, 2), (9, 10)], [(3, 8)]), 0)
        self.assertEqual(self.O.intersection_seconds([(1, 5), (3, 7)], [(4, 6)]), 2)
        for interval in ((1, float("inf")), (float("nan"), 2), (2, 1), (True, 2)):
            with self.subTest(interval=interval), self.assertRaises(ValueError): self.O.union([interval])

    def test_only_owned_qualified_incomplete_first_measurement_opens_gate(self):
        job, rows, _ = self.evidence(); owner = dict(pid=7, ppid=11, pgid=7, status="running")
        with mock.patch.object(self.O, "_ownership", return_value=owner) as ownership:
            self.assertIsNotNone(self.O.measured_gate(self.directory, job, 11)); ownership.assert_called_once_with(7, 11)
            warm = next(i for i, r in enumerate(rows) if r["event"] == "window_start" and r["id"] == "warm")
            (self.directory / "stdout.jsonl").write_bytes(b"".join(self.H._raw(r) for r in rows[:warm + 1]))
            self.assertIsNone(self.O.measured_gate(self.directory, job, 11))
            self.evidence(complete=True)
            with self.assertRaises(ValueError): self.O.measured_gate(self.directory, job, 11)
            for fault in ("sha", "qualification", "nonfinite", "job"):
                job, rows, _ = self.evidence(); prefix = next(i for i, r in enumerate(rows) if r["event"] == "window_start" and r["id"] == "measured")
                if fault == "sha": rows[0]["job_sha256"] = "0" * 64
                if fault == "qualification": next(r for r in rows if r["event"] == "qualification")["checks"]["greedy_equal"] = False
                if fault == "nonfinite": rows[prefix]["started_seconds"] = float("inf")
                if fault == "job": (self.directory / "job.json").write_bytes(b"changed")
                (self.directory / "stdout.jsonl").write_bytes(b"".join((json.dumps(r) + "\n").encode() for r in rows[:prefix + 1]))
                with self.subTest(fault=fault), self.assertRaises(ValueError): self.O.measured_gate(self.directory, job, 11)
        self.evidence()
        with mock.patch.object(self.O, "_ownership", side_effect=ValueError("foreign worker")), self.assertRaisesRegex(ValueError, "foreign worker"):
            self.O.measured_gate(self.directory, self.job, 11)

    def test_cache_readback_is_optional_but_zero_requires_exact_bound_integers(self):
        job, _, identity = self.evidence(complete=True)
        self.O.audit("gpu", self.directory, dict(job=job, job_sha256=identity), self.child)
        job, _, identity = self.evidence(complete=True, cache=0); plan = dict(job=job, job_sha256=identity)
        path = self.directory / "cache-control.json"
        for fault in (None, "missing", "bool", "sha", "requested"):
            proof = dict(job_sha256=identity, requested_bytes=0, observed_bytes=0, previous_bytes=8 * 1024**3)
            if fault == "bool": proof["observed_bytes"] = False
            if fault == "sha": proof["job_sha256"] = "0" * 64
            if fault == "requested": proof["requested_bytes"] = 4 * 1024**3
            if fault == "missing": path.unlink(missing_ok=True)
            else: path.write_bytes(self.H._raw(proof))
            if fault is None: self.O.audit("gpu", self.directory, plan, self.child)
            else:
                with self.subTest(fault=fault), self.assertRaises(FileNotFoundError if fault == "missing" else ValueError): self.O.audit("gpu", self.directory, plan, self.child)

    def test_rebuilt_job_keeps_workload_cache_and_cli_binding(self):
        worker = importlib.import_module("experiments.mlx_batch_throughput"); cpu = importlib.import_module("experiments.cpu_throughput")
        profiles = self.root / "profiles.json"; self.f.put(profiles, dict(profiles={"8": dict(actor=dict(descriptor=self.f.d))}))
        config = dict(batch_size=8, prefill_batch_size=4, prefill_step_size=256)
        for cache in (None, 0):
            old = worker.build_job(self.f.d, [config], panel_size=32, cache_limit_bytes=cache)
            old["source_hashes"] = dict.fromkeys(old["source_hashes"], "a" * 64)
            winner = self.root / "winner.json"; winner.write_bytes(self.H._raw(old)); directory = self.root / f"freeze-{cache}"
            with mock.patch.object(cpu.shutil, "which", return_value="/fixture/docker"): plan = self.O.freeze(profiles, winner, directory)
            self.assertEqual(plan["job"]["fixtures"], old["fixtures"]); self.assertEqual(plan["job"]["windows"], old["windows"])
            self.assertEqual(plan["old_job_sha256"], self.f.P.sha256(winner.read_bytes()).hexdigest())
            self.assertEqual("cache_limit_bytes" in plan["job"], cache is not None)
            command = self.O.commands("gpu", plan, self.root / "phase", self.f.lock, 3)
            self.assertEqual("--cache-gib" in command, cache is not None)
            if cache is not None: self.assertEqual(command[command.index("--cache-gib") + 1], "0")

    def test_owned_stop_interrupts_both_cli_groups_before_waiting(self):
        children = []; receipts = [self.root / f"child-{i}.json" for i in range(2)]
        try:
            for i in range(2):
                out, err = (self.root / f"child-{i}.stdout").open("wb"), (self.root / f"child-{i}.stderr").open("wb")
                process = subprocess.Popen([sys.executable, "-I", "-B", "-c", CHILD, str(receipts[i]), str(receipts[1 - i])],
                    stdin=subprocess.PIPE, stdout=out, stderr=err, start_new_session=True)
                children.append(dict(kind="gpu" if i == 0 else "cpu", process=process, stdout=out, stderr=err,
                    pid=process.pid, launch_seconds=0, reap_seconds=None, returncode=None))
                deadline = self.O.monotonic() + 5
                while not Path(str(receipts[i]) + ".ready").exists() and self.O.monotonic() < deadline: self.O.sleep(.01)
                self.assertTrue(Path(str(receipts[i]) + ".ready").exists())
            errors, interruption = self.O.stop(children, interrupt=True)
            self.assertEqual(errors, []); self.assertIsNone(interruption)
            saved = [json.loads(p.read_bytes()) for p in receipts]
            self.assertLess(abs(saved[0]["interrupted"] - saved[1]["interrupted"]), 1)
            self.assertTrue(all(c["returncode"] == 0 and c["reap_seconds"] is not None for c in children))
        finally:
            for child in children:
                process = child["process"]
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
                for stream in (process.stdin, child["stdout"], child["stderr"]):
                    if stream is not None and not stream.closed: stream.close()

    def test_fixed_phase_order_and_unknown_closure_stop_preserve_actual_attempts(self):
        self.assertEqual(self.O.PHASES, ("gpu_cpu", "overlap", "overlap", "cpu_gpu"))
        for blocked in (False, True, "missing", "corrupt"):
            output = self.root / f"block-{blocked}"
            def frozen(profiles, winner, directory): Path(directory).mkdir(); return dict(job=self.job)
            def phase(plan, mode, directory, lock, fd):
                if isinstance(blocked, str):
                    directory.mkdir(); (directory / "gpu-launch.json").write_text("{}")
                    op = directory / "cpu/ops/fixture"; op.mkdir(parents=True); (op / "intent.json").write_text("{}")
                    if blocked == "corrupt": (directory / "phase.json").write_text("{")
                    raise KeyboardInterrupt
                return dict(mode=mode, status="blocked" if blocked else "passed", elapsed_seconds=1,
                    errors=["CPU native proof incomplete"] if blocked else [], gpu_job_attempts=1, native_intents_lower_bound=0 if blocked else 2)
            with mock.patch.object(self.O, "freeze", side_effect=frozen), mock.patch.object(self.O, "run_phase", side_effect=phase) as calls:
                if isinstance(blocked, str):
                    with self.assertRaises(KeyboardInterrupt): self.O.run("profiles", "winner", output, self.f.lock)
                    result = json.loads((output / "response.json").read_bytes())
                    self.assertIn("phase_read_error", result["phases"][0]); self.assertIn("KeyboardInterrupt", result["parent_error"])
                else: result = self.O.run("profiles", "winner", output, self.f.lock)
            expected = list(self.O.PHASES[:1] if blocked else self.O.PHASES)
            self.assertEqual([call.args[1] for call in calls.call_args_list], expected)
            self.assertEqual(result["planned"]["verifications"], 64); self.assertEqual(result["planned"]["gpu_jobs"], 4)
            self.assertEqual(result["attempted"], dict(gpu_jobs=len(expected), native_intents_lower_bound=1 if isinstance(blocked, str) else 0 if blocked else 8))
            self.assertEqual(len(result["unrun"]), 3 if blocked else 0)
            self.assertEqual(result, json.loads((output / "response.json").read_bytes()))
