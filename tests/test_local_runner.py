import importlib
from concurrent.futures import CancelledError, ThreadPoolExecutor
from itertools import count, product
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest import mock


local = importlib.import_module("src.data-pipeline.stage01.local_runner")


def wait_for(predicate, seconds=3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def process_gone(pid):
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


class LocalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.runner = local.LocalRunner().__enter__()
        self.addCleanup(self.runner.__exit__, None, None, None)

    def run_case(self, source, stdin="", answer="1"):
        return self.runner.run_case(source, stdin, answer)

    def test_preflight_provenance_and_missing_interpreter(self):
        evidence = json.dumps(self.runner.provenance)
        self.assertEqual(self.runner.provenance["max_concurrent_cases"], local.MAX_CONCURRENT_CASES)
        self.assertIn("3.12.", evidence)
        self.assertIn("int_max_str_digits=0", evidence)
        with self.assertRaises(FileNotFoundError):
            local.preflight("/nonexistent/qualification-python")
        probe = subprocess.CompletedProcess([], 0, b'{"version_info":[3,14,0]}')
        with mock.patch.object(local.subprocess, "run", return_value=probe), self.assertRaises(RuntimeError):
            local.preflight()

    def test_exact_output_and_huge_integer_conversion(self):
        for source, status in (("print(' 1 ')", "passed"), ("print(2)", "wrong_answer"),
                               ("print('1 1')", "wrong_answer"), ("pass", "wrong_answer"),
                               ("print('-0')", "wrong_answer")):
            with self.subTest(source=source):
                self.assertEqual(self.run_case(source)["status"], status)
        for answer in ("9" * 4897, "1" * 5000):
            result = self.run_case("print(int(input()))", answer, answer)
            self.assertEqual(result["status"], "passed")
        self.assertEqual(self.run_case("print('-0')", answer="-0")["status"], "passed")

    def test_nonzero_exits_are_reference_failures(self):
        for source in ("raise RuntimeError('failure')", "raise SystemExit(125)",
                       "raise SystemExit(126)", "raise SystemExit(127)",
                       "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"):
            with self.subTest(source=source):
                result = self.run_case(source)
                self.assertEqual(result["status"], "nonzero_exit")
                self.assertNotEqual(result["returncode"], 0)

    def test_stdout_strict_utf8_and_diagnostic_stderr(self):
        result = self.run_case("import os; os.write(1, b'\\xff')")
        self.assertEqual(result["status"], "invalid_stdout")
        result = self.run_case("import os; os.write(2, b'diagnostic \\xff'); print(1)")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["stderr"], "diagnostic \ufffd")

    def test_fresh_working_directory_and_minimal_environment(self):
        source = (
            "import os, sys\n"
            "assert 'ROPOT_TEST_SENTINEL' not in os.environ\n"
            "assert os.environ['PATH'] == '/usr/bin:/bin'\n"
            "assert os.environ['LANG'] == os.environ['LC_ALL'] == 'C'\n"
            "assert os.path.isdir(os.environ['TMPDIR'])\n"
            "assert not os.path.exists('marker')\n"
            "open('marker', 'w').close()\n"
            "print(os.getcwd(), file=sys.stderr)\nprint(1)\n"
        )
        paths = []
        for _ in range(2):
            with mock.patch.dict(os.environ, {"ROPOT_TEST_SENTINEL": "must not inherit"}), \
                    mock.patch.object(local.subprocess, "Popen", wraps=local.subprocess.Popen) as spawn:
                result = self.run_case(source)
                self.assertEqual(set(spawn.call_args.kwargs["env"]), {"PATH", "LANG", "LC_ALL", "TMPDIR"})
            self.assertEqual(result["status"], "passed")
            paths.append(result["stderr"].strip())
        self.assertNotEqual(*paths)
        self.assertTrue(all(not Path(path).exists() for path in paths))

    def test_large_stdin_output_before_read_and_early_closure(self):
        result = self.run_case("import os; os.close(0); print(1)", "x" * 2_000_000)
        self.assertEqual(result["status"], "passed")
        source = "import sys\nsys.stdout.write(' ' * 200000); sys.stdout.flush()\nprint(len(sys.stdin.read()))"
        result = self.run_case(source, "x" * 2_000_000, "2000000")
        self.assertEqual(result["status"], "passed")

    def test_each_output_cap_is_inclusive_and_bounded(self):
        for (fd, limit, status), ending in product(
                ((1, local.STDOUT_LIMIT_BYTES, "stdout_limit"),
                 (2, local.STDERR_LIMIT_BYTES, "stderr_limit")), ("while True: pass", "os._exit(0)")):
            with self.subTest(fd=fd, ending=ending):
                source = f"import os\nos.write({fd}, b'x' * {limit})\n{ending}"
                result = self.run_case(source)
                self.assertEqual(result["status"], status)
                stream = "stdout" if fd == 1 else "stderr"
                self.assertEqual(len(result[stream].encode()), limit)

    def test_timeout_for_loop_closed_pipes_and_inherited_pipes(self):
        sources = ["while True: pass", "import os\nos.close(1); os.close(2)\nwhile True: pass",
                   "import os, time\nif os.fork() == 0: time.sleep(30)\nelse: print(1)"]
        with mock.patch.object(local, "TIMEOUT_SECONDS", 0.15):
            for source in sources:
                with self.subTest(source=source):
                    result = self.run_case(source)
                    self.assertEqual(result["status"], "timeout")
                    self.assertGreaterEqual(result["elapsed_seconds"], 0.15)
                    self.assertLess(result["elapsed_seconds"], 2)

    def test_success_still_kills_descendants_with_closed_pipes(self):
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            source = ("import os, time\nchild = os.fork()\nif child:\n"
                      f" open({str(pidfile)!r}, 'w').write(str(child))\n print(1)\n"
                      "else:\n os.close(0); os.close(1); os.close(2); time.sleep(30)")
            result = self.run_case(source)
            pid = int(pidfile.read_text())
            self.addCleanup(lambda: None if process_gone(pid) else os.kill(pid, signal.SIGKILL))
            self.assertEqual(result["status"], "passed")
            self.assertTrue(wait_for(lambda: process_gone(pid)))

    def test_reference_stops_on_first_failure_and_preserves_indices(self):
        cases = [{"index": index, "input": value, "answer": "1"}
                 for index, value in ((7, "1"), (12, "2"), (19, "1"))]
        result = self.runner.run_reference("print(input())", cases)
        self.assertFalse(result["passed"])
        self.assertEqual([case["index"] for case in result["cases"]], [7, 12])
        self.assertEqual(result["max_case_seconds"], max(c["elapsed_seconds"] for c in result["cases"]))
        self.assertGreaterEqual(result["wall_seconds"], result["max_case_seconds"])
        self.assertTrue(self.runner.run_reference("print(1)", cases)["passed"])

    def test_process_and_cleanup_errors_abort(self):
        with mock.patch.object(local.subprocess, "Popen", side_effect=OSError("spawn failed")), \
                self.assertRaisesRegex(OSError, "spawn failed"):
            self.run_case("print(1)")
        with mock.patch.object(local.os, "killpg", side_effect=PermissionError("cleanup failed")), \
                mock.patch.object(local, "TIMEOUT_SECONDS", 0.1), \
                self.assertRaisesRegex(PermissionError, "cleanup failed"):
            self.run_case("while True: pass")

    def test_deadline_covers_final_drain_boundary(self):
        def after_exit(process, data, started):
            process.stdout.read()
            self.assertTrue(wait_for(lambda: local._exited(process)))
            self.assertIsNone(process.returncode)
            if cancelled:
                raise RuntimeError("cancelled")
            return None, b"1", b""

        for cancelled in (False, True):
            with mock.patch.object(self.runner, "_pump", side_effect=after_exit), \
                    mock.patch.object(local, "monotonic", side_effect=count(100, 6).__next__), \
                    self.assertRaisesRegex(RuntimeError, "cancelled") if cancelled else self.subTest():
                result = self.run_case("print(1)")
                self.assertEqual(result["status"], "timeout")
                self.assertEqual(result["elapsed_seconds"], 6)

    def test_case_limit_and_outer_exception_cleanup(self):
        processes, original = [], local.subprocess.Popen
        limit = local.MAX_CONCURRENT_CASES

        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        with ThreadPoolExecutor(max_workers=limit + 2) as callers:
            with self.assertRaisesRegex(RuntimeError, "outer failure"), local.LocalRunner() as runner:
                with mock.patch.object(local.subprocess, "Popen", side_effect=spawn):
                    futures = [callers.submit(runner.run_case, "import time; time.sleep(30)", "", "1")
                               for _ in range(limit + 2)]
                    self.assertTrue(wait_for(lambda: len(processes) == limit))
                    raise RuntimeError("outer failure")
        self.assertLessEqual({(type(f.exception()), str(f.exception())) for f in futures},
                             {(RuntimeError, "Local runner has closed"), (CancelledError, "")})
        self.assertEqual(len(processes), limit)
        self.assertTrue(all(process.poll() is not None for process in processes))

    def test_sigint_and_sigterm_clean_active_group(self):
        for (number, exception), mode in product(
                ((signal.SIGINT, "KeyboardInterrupt"), (signal.SIGTERM, "SystemExit")),
                ("active", "body", "nested")):
            with self.subTest(signal=number, mode=mode), tempfile.TemporaryDirectory() as directory:
                pidfile = Path(directory) / "pid"
                source = ("import os, time, json\nchild = os.fork()\nif child:\n"
                          f" open({str(pidfile) + '.tmp'!r}, 'w').write(json.dumps([os.getpid(), child]))\n"
                          f" os.rename({str(pidfile) + '.tmp'!r}, {str(pidfile)!r})\n"
                          "time.sleep(30)")
                body = f"  runner.run_case({source!r}, '', '1')\n"
                if mode == "body":
                    body = f"  open({str(pidfile)!r}, 'w').write('[]')\n  while True: pass\n"
                elif mode == "nested":
                    body = ("  with ThreadPoolExecutor(max_workers=1) as tasks:\n"
                            f"   tasks.submit(runner.run_case, {source!r}, '', '1').result()\n")
                helper = ("import importlib\nfrom concurrent.futures import ThreadPoolExecutor\n"
                          "r=importlib.import_module('src.data-pipeline.stage01.local_runner')\n"
                          "try:\n with r.LocalRunner() as runner:\n"
                          f"{body}  print('body continued', flush=True)\n"
                          "except BaseException as error:\n print(type(error).__name__, flush=True)\n")
                process = subprocess.Popen([local.INTERPRETER, "-c", helper], stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                try:
                    self.assertTrue(wait_for(lambda: pidfile.exists() and pidfile.read_text()))
                    pids = json.loads(pidfile.read_text())
                    process.send_signal(number)
                    stdout, stderr = process.communicate(timeout=3)
                    self.assertEqual(process.returncode, 0, stderr.decode(errors="replace"))
                    self.assertEqual(stdout.decode().strip(), exception)
                    self.assertTrue(wait_for(lambda: all(process_gone(pid) for pid in pids)))
                finally:
                    process.kill()
                    process.communicate()
                    if pidfile.exists():
                        for pid in json.loads(pidfile.read_text()):
                            if not process_gone(pid):
                                os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
