"""Actual owned-group signalling and reaping; zero model calls."""
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

b = importlib.import_module("src.data-pipeline.stage03.benign_loop")


class ProcessTests(unittest.TestCase):
    def await_file(self, path, process):
        deadline = time.monotonic() + 10
        while not path.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(path.exists(), f"Process did not create {path}")

    def test_graceful_interrupt_and_explicit_second_interrupt(self):
        for stalled in (False, True):
            with self.subTest(stalled=stalled), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ready, interrupted = root / "ready", root / "child-interrupted"
                child_code = (
                    "import signal,time,sys,os\nfrom pathlib import Path\n"
                    "def interrupt(*args):\n"
                    f" Path({str(interrupted)!r}).write_text('graceful cleanup requested')\n"
                    + (" return\n" if stalled else " sys.exit(0)\n") +
                    "signal.signal(signal.SIGINT, interrupt)\n"
                    f"Path({str(ready)!r}).write_text(str(os.getpid()))\n"
                    "while True: time.sleep(.01)\n")
                supervisor = (
                    "import importlib\nfrom pathlib import Path\n"
                    f"b=importlib.import_module({b.__name__!r})\n"
                    f"state={{'child_log':{str(root / 'child.log')!r}}}\n"
                    f"def persist(): b.atomic_save(Path({str(root / 'progress.json')!r}),state)\n"
                    "try:\n"
                    f" b.run_child({[sys.executable, '-c', child_code]!r},state,persist)\n"
                    "except KeyboardInterrupt:\n state['original_interrupt_preserved']=True\n"
                    "finally: persist()\n")
                process = subprocess.Popen([sys.executable, "-c", supervisor], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    self.await_file(ready, process)
                    state = json.loads((root / "progress.json").read_text())
                    self.assertEqual(state["child"]["pid"], int(ready.read_text()))
                    self.assertEqual(state["child"]["pgid"], state["child"]["pid"])
                    self.assertNotEqual(state["child"]["pgid"], os.getpgid(process.pid))
                    process.send_signal(signal.SIGINT)
                    self.await_file(interrupted, process)
                    if stalled:
                        self.assertIsNone(process.poll())
                        process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, (stdout, stderr))
                    state = json.loads((root / "progress.json").read_text())
                    self.assertTrue(state["original_interrupt_preserved"])
                    self.assertEqual(state["child"]["returncode"], -signal.SIGKILL if stalled else 0)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(state["child"]["pid"], 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()
                    progress_file = root / "progress.json"
                    if progress_file.exists():
                        pid = json.loads(progress_file.read_text())["child"]["pid"]
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_nonzero_exit_is_reaped_and_missing_outcome_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = dict(child_log=str(root / "child.log"))
            observed = []
            code = b.run_child([sys.executable, "-c", "raise SystemExit(7)"], state,
                               lambda: observed.append(state["child"]["pid"]))
            self.assertEqual(code, 7)
            self.assertEqual(observed, [state["child"]["pid"]])
            with self.assertRaises(ProcessLookupError):
                os.kill(state["child"]["pid"], 0)
            self.assertIsNone(b.collect_generation(root, state, 2))
            self.assertIn("generation_evidence_error", state)
