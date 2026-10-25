import argparse
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from experiments import gguf_benchmark as gg
from experiments import run_inference_benchmarks as driver


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_hard_deadline_reaps_owned_process_and_records_failure(self):
        job = driver.launch(self.root, "timeout", [sys.executable, "-c", "import time;time.sleep(20)"], .05)
        with self.assertRaises(TimeoutError):
            driver.wait([job])
        self.assertIsNotNone(job["child"].poll())
        result = json.loads((job["directory"]/"outcome.json").read_text())
        self.assertTrue(result["interrupted_by_driver"])
        self.assertTrue(result["process_group_absent"])
        self.assertNotIn(job, driver.ACTIVE)

    def test_failed_overlap_peer_stops_the_other_owned_child(self):
        failed = driver.launch(self.root, "failed", [sys.executable, "-c", "raise SystemExit(3)"], 5)
        peer = driver.launch(self.root, "peer", [sys.executable, "-c", "import time;time.sleep(20)"], 5)
        with self.assertRaises(RuntimeError):
            driver.wait([failed, peer])
        self.assertTrue(all(j["child"].poll() is not None for j in (failed, peer)))
        self.assertEqual(driver.ACTIVE, [])

    def test_stop_is_idempotent_when_process_exits_between_poll_and_signal(self):
        child = mock.Mock(pid=123, returncode=0)
        child.poll.return_value = None
        with mock.patch.object(driver.os, "killpg", side_effect=ProcessLookupError):
            driver.stop({"child": child})
        child.wait.assert_called_once()

    def test_cleanup_attempts_every_peer_even_if_one_stop_raises(self):
        jobs = []
        for name in ("one", "two"):
            directory = self.root/name
            directory.mkdir()
            jobs.append(dict(name=name, directory=directory, started=perf_counter(),
                             child=NS(pid=123, returncode=-2), streams=(mock.Mock(), mock.Mock())))
        driver.ACTIVE.extend(jobs)
        with mock.patch.object(driver, "stop", side_effect=[RuntimeError("cleanup fixture"), None]) as stop, \
             mock.patch.object(driver, "group_alive", return_value=False):
            driver.finish(jobs, True)
        self.assertEqual(stop.call_count, 2)
        self.assertIn("cleanup fixture", json.loads((self.root/"one/outcome.json").read_text())["cleanup_error"])
        self.assertEqual(driver.ACTIVE, [])

    def test_second_launch_failure_still_closes_and_records_first_child(self):
        first = driver.launch(self.root, "first", [sys.executable, "-c", "import time;time.sleep(20)"], 5)
        try:
            with mock.patch.object(driver.subprocess, "Popen", side_effect=FileNotFoundError), \
                 self.assertRaises(FileNotFoundError):
                driver.launch(self.root, "missing", ["missing-fixture-command"], 5)
        finally:
            driver.finish(list(driver.ACTIVE), True)
        self.assertIsNotNone(first["child"].poll())
        self.assertTrue(json.loads((first["directory"]/"outcome.json").read_text())["process_group_absent"])

    def test_gguf_warm_route_uses_fresh_sampling_and_reaps_server(self):
        requests = []
        def post(base, endpoint, payload):
            if endpoint == "/apply-template":
                return {"prompt": "<s>fixture"}
            if endpoint == "/tokenize":
                return {"tokens": [1, 8]}
            requests.append(payload)
            return {"tokens": [4, 2], "tokens_predicted": 2, "tokens_evaluated": 2,
                    "stop_type": "eos", "content": "print(1)", "timings": {"cache_n": 0}}
        args = argparse.Namespace(output=str(self.root/"gguf"), server=__file__, model=__file__,
                                  template=__file__, reference_model="fixture", parallel=2, repeats=1)
        child = mock.Mock(pid=123, returncode=0)
        child.poll.return_value = None
        health = mock.MagicMock()
        health.__enter__.return_value.status = 200
        with mock.patch.object(gg, "post", side_effect=post), \
             mock.patch.object(gg.subprocess, "Popen", return_value=child), \
             mock.patch.object(gg.urllib.request, "urlopen", return_value=health), \
             mock.patch.object(gg.socket, "socket") as socket, \
             mock.patch.dict("sys.modules", {"transformers": NS(AutoTokenizer=NS(
                 from_pretrained=lambda *a, **k: NS(apply_chat_template=lambda *a, **k: [1, 8])))}):
            socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            socket.return_value.__enter__.return_value.connect_ex.return_value = 111
            gg.run(args)
        self.assertEqual(len(requests), 13)
        self.assertTrue(all(r["cache_prompt"] is False for r in requests))
        self.assertTrue(all(r["samplers"] == ["top_p", "temperature"] for r in requests))
        self.assertEqual(len({r["seed"] for r in requests[3:]}), 10)
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
        self.assertTrue(json.loads((self.root/"gguf/cleanup.json").read_text())["port_closed"])


if __name__ == "__main__":
    unittest.main()
