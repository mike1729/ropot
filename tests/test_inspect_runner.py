import asyncio
import base64
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from types import SimpleNamespace

try:
    from inspect_ai.util import (ExecResult, OutputLimitExceededError,
                                 SandboxEnvironmentLimits, SandboxUnavailableError,
                                 override_sandbox_output_limit)
    runner = importlib.import_module("src.data-pipeline.stage03.inspect_runner")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    runner = None


@unittest.skipIf(runner is None, "requires requirements-execution.txt")
class InspectRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def invoke(self, result=None, error=None):
        environment = mock.Mock(exec=mock.AsyncMock())
        environment.exec.side_effect = [error or result]
        with mock.patch.object(runner, "sandbox", return_value=environment):
            return await runner.run_case("input\n"), environment

    async def test_capture_boundaries_and_error_classification(self):
        cases = [(0, "1\n", "", "completed"), (3, "", "failure", "nonzero_exit"),
                 (0, "\ufffd", "", "invalid_stdout")]
        for fd, limit, status in ((1, 1024**2, "stdout_limit"), (2, 64 * 1024, "stderr_limit")):
            for size, expected in ((limit - 1, "completed"), (limit, status)):
                cases.append((0, "x" * size if fd == 1 else "", "x" * size if fd == 2 else "", expected))
        for code, stdout, stderr, status in cases:
            with self.subTest(code=code, status=status):
                result, _ = await self.invoke(ExecResult(code == 0, code, stdout, stderr))
                self.assertEqual(result["status"], status)
                if status.endswith("_limit"):
                    self.assertIsNone(result["stdout"])
                    self.assertIsNone(result["stderr"])
                    self.assertLessEqual(len(result["diagnostic"].encode("utf-8")), 4096)
                else:
                    self.assertEqual((result["stdout"], result["stderr"]), (stdout, stderr))
        result, _ = await self.invoke(error=OutputLimitExceededError("10 MiB", "€" * 6000))
        self.assertEqual(result["status"], "output_limit")
        self.assertIsNone(result["stdout"])
        self.assertIsNone(result["stderr"])
        self.assertLessEqual(len(result["diagnostic"].encode("utf-8")), 4096)

    def test_bad_case_indices_fail_before_docker(self):
        for indices in ((0, 0), (True,), (1.0,), ("../other",)):
            with self.subTest(indices=indices), mock.patch.object(runner.subprocess, "run") as docker:
                with self.assertRaises(ValueError):
                    runner.run([{"program": b"pass", "cases": [{"index": i, "input": ""} for i in indices]}], "unused")
                docker.assert_not_called()

    async def test_only_known_container_timeouts_are_program_failures(self):
        program = TimeoutError("proxy")
        program.__cause__ = TimeoutError("Command timed out after 5 seconds")
        program.truncated_output = "€" * 6000
        result, _ = await self.invoke(error=program)
        self.assertEqual(result["status"], "timeout")
        self.assertLessEqual(len(result["diagnostic"].encode("utf-8")), 4096)
        host = TimeoutError("proxy")
        host.__cause__ = TimeoutError("Command timed out after 5 seconds")
        host.__cause__.__cause__ = TimeoutError("Docker CLI hung")
        for error in (host, TimeoutError("unknown"), SandboxUnavailableError("gone"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                await self.invoke(error=error)

    async def test_capture_override_is_explicit(self):
        environment = mock.Mock(exec=mock.AsyncMock())
        async def execute(*args, **kwargs):
            if kwargs.get("cwd"):
                self.assertEqual(SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE, 10 * 1024**2)
            return ExecResult(True, 0, "1\n", "")
        environment.exec.side_effect = execute
        with override_sandbox_output_limit(8, "exec"), mock.patch.object(runner, "sandbox", return_value=environment):
            await runner.run_case("")
            self.assertEqual(SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE, 8)

    async def test_program_written_once_and_cases_use_one_exec_in_shared_work(self):
        source = b"# coding: latin-1\r\nprint('\xe9')\r\n"
        state = SimpleNamespace(store=mock.Mock(), metadata={
            "program_base64": base64.b64encode(source).decode(),
            "cases": [{"index": index, "input": "input\n"} for index in (7, 12)]})
        environment = mock.Mock(exec=mock.AsyncMock(return_value=ExecResult(True, 0, "1\n", "")))
        with mock.patch.object(runner, "sandbox", return_value=environment):
            await runner.execute_programs()(state, None)
        preparation, execute1, execute2 = environment.exec.call_args_list
        self.assertEqual(preparation.kwargs["input"], source)
        self.assertEqual(preparation.kwargs["user"], "root")
        self.assertFalse(preparation.kwargs["timeout_retry"])
        for execution in (execute1, execute2):
            self.assertEqual(execution.args[0][-1], "/work/program.py")
            self.assertEqual(execution.kwargs["cwd"], "/work")
            self.assertIn("TMPDIR=/work", execution.args[0])
            self.assertNotIn("user", execution.kwargs)
            self.assertEqual(execution.kwargs["input"], b"input\n")
        args = preparation.args[0]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "program.py"
            command = [sys.executable, *args[args.index("-c"):-1], str(path)]
            subprocess.run(command, input=source, capture_output=True, check=True)
            self.assertEqual(path.read_bytes(), source)
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            repeated = subprocess.run(command, input=b"replacement", capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(path.read_bytes(), source)
            path.chmod(0o600)

    async def test_program_preparation_failure_prevents_cases(self):
        state = SimpleNamespace(store=mock.Mock(), metadata={
            "program_base64": "cGFzcw==", "cases": [{"index": 7, "input": ""}]})
        environment = mock.Mock(exec=mock.AsyncMock(return_value=ExecResult(False, 1, "", "write failed")))
        with mock.patch.object(runner, "sandbox", return_value=environment), self.assertRaisesRegex(
                RuntimeError, "Cannot prepare program"):
            await runner.execute_programs()(state, None)
        self.assertEqual(environment.exec.await_count, 1)


@unittest.skipUnless(runner is not None and os.environ.get("ROPOT_INSPECT_TESTS") == "1",
                     "set ROPOT_INSPECT_TESTS=1 for real Inspect/Docker fixtures")
class InspectDockerTests(unittest.TestCase):
    def log_directory(self, prefix):
        root = Path(__file__).resolve().parents[1] / "data/03_execution_checks"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(prefix=prefix, dir=root)

    def assert_removed(self, containers):
        remaining = subprocess.run(["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}}", "--no-trunc"],
                                   capture_output=True, check=True, text=True)
        self.assertTrue(set(containers).isdisjoint(remaining.stdout.split()))

    def run_tracking_containers(self, programs, directory):
        containers = {}
        original = runner.run_case

        async def track(*args):
            name = (await runner.sandbox().connection()).container
            if name not in containers:
                observed = subprocess.run(["docker", "inspect", name],
                                          capture_output=True, check=True, text=True)
                containers[name] = json.loads(observed.stdout)[0]
            return await original(*args)

        with mock.patch.object(runner, "run_case", side_effect=track):
            logs = runner.run(programs, directory)
        return logs, list(containers.values())

    def test_infrastructure_failure_uses_native_teardown(self):
        programs = [{"program": source, "cases": [{"index": i, "input": ""} for i in (7, 12)]}
                    for source in (b"import os; os.kill(-1, 9)", *([b"print(1)"] * 4))]
        logs, containers = self.run_tracking_containers(programs, self.log_directory("infrastructure-"))
        log = logs[0]
        self.assertEqual(log.status, "error")
        samples = sorted(log.samples, key=lambda sample: sample.id)
        self.assertEqual(len(samples), len(programs))
        self.assertIn("SandboxUnavailableError", samples[0].error.traceback)
        for sample in samples:
            if sample.id >= 1:
                self.assertIsNone(sample.error)
                self.assertEqual([case["status"] for case in sample.store["cases"]], ["completed"] * 2)
        self.assert_removed([container["Id"] for container in containers])

    def test_sigint_uses_native_teardown_and_retains_cancelled_log(self):
        from inspect_ai.log import read_eval_log
        directory = self.log_directory("cancelled-")
        marker = Path(directory) / "ready"
        helper = ("import importlib\nfrom pathlib import Path\n"
                  "r=importlib.import_module('src.data-pipeline.stage03.inspect_runner')\noriginal=r.run_case\n"
                  "async def ready(*args):\n"
                  f" Path({str(marker)!r}).write_text((await r.sandbox().connection()).container)\n"
                  " return await original(*args)\nr.run_case=ready\n"
                  f"r.run([{{'program':b'while True: pass','cases':[{{'index':0,'input':''}}]}}],{directory!r})\n")
        process = subprocess.Popen([sys.executable, "-c", helper], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 30
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(marker.exists(), "sample did not start")
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=30)
            self.assert_removed([marker.read_text()])
            logs = list(Path(directory).glob("*.eval"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(read_eval_log(str(logs[0]), header_only=True).status, "cancelled")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()

    def test_fixtures_limits_provenance_and_native_teardown(self):
        fixtures = []
        def add(source, stdin="", status="completed", stdout=None, indices=(7,)):
            fixtures.append((source if isinstance(source, bytes) else source.encode(), stdin, status, stdout, indices))
        add("import os,sys\nassert set(os.environ)=={'PATH','LANG','LC_ALL','TMPDIR'}\n"
            "assert os.environ['PATH']=='/usr/bin:/bin'\nassert os.environ['LANG']==os.environ['LC_ALL']=='C'\n"
            "assert sys.flags.utf8_mode==1\nassert os.getcwd()==os.environ['TMPDIR']=='/work'\n"
            "path='marker'\ncount=int(open(path).read()) if os.path.exists(path) else 0\n"
            "assert count in (0,1)\nopen(path,'w').write(str(count+1))\nprint(1)",
            stdout="1\n", indices=(7, 12))
        add(b"# coding: latin-1\nprint('\xe9')\n", stdout="é\n")
        add("import os\npath='/work/program.py'\n"
            "assert os.getuid()==65534\nassert os.stat(path).st_uid==0\n"
            "assert os.stat(path).st_mode & 0o777==0o444\n"
            "open('replacement','w').close()\n"
            "for action in (lambda: open(path,'wb'), lambda: os.chmod(path,0o644),\n"
            "               lambda: os.unlink(path), lambda: os.replace('replacement',path)):\n"
            " try: action()\n except PermissionError: pass\n"
            " else: raise AssertionError('program file was mutable')\nprint(1)",
            stdout="1\n", indices=(7, 12))
        add("print(int(input()))", "9" * 5000, stdout="9" * 5000 + "\n")
        add("import sys\nsys.stdout.write(' '*200000); sys.stdout.flush()\nprint(len(sys.stdin.read()))",
            "x" * 2_000_000, stdout=" " * 200000 + "2000000\n")
        add("import os; os.close(0); print(1)", "x" * 2_000_000, stdout="1\n")
        add("raise SystemExit(3)", status="nonzero_exit", indices=(7, 12))
        add("raise SystemExit(124)", status="timeout")
        add("while True: pass", status="timeout")
        add("import os; os.write(1,b'\\xff')", status="invalid_stdout")
        add("import os; os.write(2,b'\\xff'); print(1)", stdout="1\n")
        for fd, cap, status in ((1, 1024**2, "stdout_limit"), (2, 64 * 1024, "stderr_limit")):
            add(f"import os; os.write({fd},b'x'*{cap - 1})")
            add(f"import os; os.write({fd},b'x'*{cap})", status=status)
        add("import os; os.write(1,b'x'*(12*1024**2)); print(1)", status="stdout_limit")
        add("import os; os.write(2,b'\\xff'*(4*1024**2+1))", status="output_limit")
        add("a=bytearray(600_000_000)", status="nonzero_exit")
        add("import os,time\nwhile True:\n try:\n  if os.fork()==0:\n   time.sleep(60); os._exit(0)\n"
            " except OSError: break\nprint(1)", stdout="1\n")
        for _ in range(2):
            add("import os\nassert not os.path.exists('/work/program-marker')\n"
                "open('/work/program-marker','w').close()\nprint(1)", stdout="1\n")
        programs = [{"program": source, "cases": [{"index": index, "input": stdin} for index in indices]}
                    for source, stdin, _, _, indices in fixtures]
        directory = self.log_directory("fixtures-")
        with mock.patch.dict(os.environ, {"INSPECT_SANDBOX_MAX_EXEC_OUTPUT_SIZE": "8"}):
            logs, containers = self.run_tracking_containers(programs, directory)
            log = logs[0]
            self.assertEqual(log.status, "success", str(log.error))
            self.assertTrue(list(Path(directory).glob("*.eval")))
            self.assertEqual(log.eval.config.max_sandboxes, 4)
            self.assertEqual(log.eval.metadata["max_sandboxes"], 4)
            self.assertFalse(log.eval.metadata["concurrent_inference"])
            samples = sorted(log.samples, key=lambda sample: sample.id)
            self.assertEqual(len(samples), len(fixtures))
            for sample, (_, _, status, stdout, indices) in zip(samples, fixtures):
                with self.subTest(sample=sample.id, status=status):
                    self.assertIsNone(sample.error)
                    cases = sample.store["cases"]
                    self.assertEqual([case["index"] for case in cases], list(indices if status == "completed" else indices[:1]))
                    self.assertTrue(all(case["status"] == status for case in cases))
                    if status.endswith("_limit"):
                        self.assertIsNone(cases[0]["stdout"])
                        self.assertIsNone(cases[0]["stderr"])
                        self.assertLessEqual(len(cases[0]["diagnostic"].encode("utf-8")), 4096)
                    if stdout is not None:
                        self.assertEqual(cases[0]["stdout"], stdout)
                    if fixtures[sample.id][0] == b"a=bytearray(600_000_000)":
                        self.assertEqual(cases[0]["returncode"], 137)
            for container in containers:
                limits = container["HostConfig"]
                self.assertEqual((limits["NanoCpus"], limits["Memory"], limits["MemorySwap"], limits["PidsLimit"]),
                                 (10**9, 512 * 1024**2, 512 * 1024**2, 32))
                self.assertTrue(limits["ReadonlyRootfs"])
                self.assertEqual(limits["NetworkMode"], "none")
                self.assertEqual(container["Config"]["User"], "65534:65534")
                self.assertEqual(limits["CapDrop"], ["ALL"])
                self.assertTrue(any(option.startswith("no-new-privileges") for option in limits["SecurityOpt"]))
                self.assertIn("size=64m", limits["Tmpfs"]["/work"])
            self.assertEqual(len(containers), len(fixtures))
            self.assert_removed([container["Id"] for container in containers])


if __name__ == "__main__":
    unittest.main()
