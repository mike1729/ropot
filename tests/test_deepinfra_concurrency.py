"""Deterministic rolling-concurrency and peer-drain tests; no HTTP or model load."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from scripts import deepinfra_generation as d
from tests.test_thinking_generation import ThinkingTokenizer


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tokenizer = ThinkingTokenizer("qwen")
        self.tokenizer.pieces = self.tokenizer.pieces.copy()
        self.tokenizer.pieces.update({100+i: f"print({i})\n" for i in range(5)})
        self.tokenizer.pieces[248046] = "<|im_end|>"
        self.spec = dict(runtime=dict(model=d.MODEL, tokenizer_path="unused", tokenizer_sha256={},
            timeout_seconds=5, max_total_tokens=100, input_usd_per_million=.2,
            output_usd_per_million=2.5, max_cost_usd=.1, concurrency=2),
            requests=[dict(messages=[dict(role="user", content=f"Task {i}")], seed=17+i,
                           temperature=1., top_p=.95, max_tokens=20) for i in range(5)],
            output_dir=str(self.root / "run"))
        self.send = mock.patch.object(d, "send").start()
        mock.patch.object(d, "load_tokenizer", return_value=self.tokenizer).start()
        mock.patch.dict("os.environ", DEEPINFRA_API_KEY="PRIVATE_TEST_KEY").start()
        self.addCleanup(mock.patch.stopall)
        self.entered = [threading.Event() for _ in range(5)]
        self.release = [threading.Event() for _ in range(5)]
        self.observed_failure = threading.Event()
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.started, self.finished = [], []
        self.send_threads, self.normalize_threads = [], []
        self.error = None
        self.invalid_text = False
        self.fail_after_error = False
        self.write_failure = threading.Event()

    def response(self, index):
        return 200, json.dumps(dict(model="Qwen/Qwen3.8-27B", choices=[dict(index=0,
            text=f"print({index})\n", token_ids=[100+index, 248046], finish_reason="stop")],
            usage=dict(prompt_tokens=1, completion_tokens=2, total_tokens=3,
                       estimated_cost=(index+1)*.0001, prompt_tokens_details=dict(cached_tokens=0))))

    def saved(self):
        return json.loads((Path(self.spec["output_dir"]) / "outcome.json").read_text())

    def guarded_send(self, payload, key, timeout):
        index = payload["seed"] - 17
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.started.append(index)
            self.send_threads.append(threading.get_ident())
        try:
            row = self.saved()["requests"][index]
            self.assertIsNone(row["submitted"])
            self.assertEqual(row["status"], "running")
            self.entered[index].set()
            if not self.release[index].wait(3):
                raise TimeoutError(f"Test did not release request {index}")
            if index == 0 and self.error is not None:
                raise self.error
            status, raw = self.response(index)
            if index == 0 and self.invalid_text:
                body = json.loads(raw)
                body["choices"][0]["text"] = "Provider decoding mismatch"
                raw = json.dumps(body)
            return status, raw
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(index)

    def run_round(self):
        path = self.root / "round.json"
        path.write_text(json.dumps(self.spec))
        return d.run(path)

    def run_async(self):
        self.send.side_effect = self.guarded_send
        result = {}
        normalize, replace = d.normalize, Path.replace
        def normalized(*args, **kwargs):
            self.normalize_threads.append(threading.get_ident())
            return normalize(*args, **kwargs)
        def replaced(path, target):
            if (Path(target) == Path(self.spec["output_dir"]) / "outcome.json"
                    and self.fail_after_error and self.observed_failure.is_set()
                    and not self.write_failure.is_set()):
                self.write_failure.set()
                raise OSError("Transient outcome persistence failure")
            value = replace(path, target)
            if Path(target) == Path(self.spec["output_dir"]) / "outcome.json":
                if self.saved()["requests"][0]["status"] in ("error", "cancelled"):
                    self.observed_failure.set()
            return value
        mock.patch.object(d, "normalize", side_effect=normalized).start()
        mock.patch.object(Path, "replace", new=replaced).start()
        def run():
            try:
                result["value"] = self.run_round()
            except BaseException as error:
                result["raised"] = error
        thread = threading.Thread(target=run, name="deepinfra-owner")
        thread.start()
        def cleanup():
            for event in self.release:
                event.set()
            thread.join(5)
            self.assertFalse(thread.is_alive(), "Round owner failed to drain test peers")
        self.addCleanup(cleanup)
        return thread, result

    def await_started(self, index):
        self.assertTrue(self.entered[index].wait(3), f"Request {index} never entered transport")

    def finish(self, thread, holder):
        thread.join(5)
        self.assertFalse(thread.is_alive(), "Round failed to settle released peers")
        self.assertNotIn("raised", holder, holder)
        return holder["value"]

    def test_out_of_order_completion_refills_slots_and_preserves_original_rows(self):
        thread, holder = self.run_async()
        self.await_started(0)
        self.await_started(1)
        self.assertFalse(self.entered[2].is_set())
        self.assertEqual([r["submitted"] for r in self.saved()["requests"]], [None, None, False, False, False])
        for completed, following in ((1, 2), (2, 3), (3, 4)):
            self.release[completed].set()
            self.await_started(following)
            self.assertTrue((Path(self.spec["output_dir"]) / f"response-{completed:04d}.json").exists())
            self.assertEqual(self.saved()["requests"][completed]["status"], "stop")
            self.assertEqual(self.saved()["requests"][0]["status"], "running")
        self.release[4].set()
        self.release[0].set()
        result = self.finish(thread, holder)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual([r["text"] for r in result["requests"]], [f"print({i})\n" for i in range(5)])
        self.assertEqual(sorted(self.started), list(range(5)))
        self.assertEqual(self.send.call_count, 5)
        self.assertEqual(self.peak, 2)
        self.assertTrue(result["usage_complete"])
        self.assertAlmostEqual(result["observed_estimated_cost_usd"], .0015)
        self.assertEqual(set(self.normalize_threads), {thread.ident})
        self.assertNotIn(thread.ident, self.send_threads)
        self.assertEqual(self.saved(), result)

    def test_failure_stops_admission_and_drains_successful_peer(self):
        self.error = TimeoutError("Ambiguous transport failure")
        thread, holder = self.run_async()
        self.await_started(0)
        self.await_started(1)
        self.release[0].set()
        self.assertTrue(self.observed_failure.wait(3), "Failure was not persisted while peer remained active")
        self.assertFalse(self.entered[2].is_set())
        self.assertTrue(thread.is_alive())
        self.release[1].set()
        result = self.finish(thread, holder)
        self.assertEqual(result["status"], "error", result)
        self.assertEqual([r["submitted"] for r in result["requests"]], [None, True, False, False, False])
        self.assertEqual(result["requests"][1]["text"], "print(1)\n")
        self.assertFalse(result["usage_complete"])
        self.assertAlmostEqual(result["observed_estimated_cost_usd"], .0002)
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(sorted(self.started), [0, 1])
        self.assertTrue((Path(self.spec["output_dir"]) / "response-0001.json").exists())

    def test_interruption_stops_admission_and_preserves_drained_peer(self):
        self.error = KeyboardInterrupt()
        thread, holder = self.run_async()
        self.await_started(0)
        self.await_started(1)
        self.release[0].set()
        self.assertTrue(self.observed_failure.wait(3), "Cancellation was not persisted before peer drain")
        self.assertFalse(self.entered[2].is_set())
        self.release[1].set()
        result = self.finish(thread, holder)
        self.assertEqual(result["status"], "cancelled", result)
        self.assertEqual([r["submitted"] for r in result["requests"]], [None, True, False, False, False])
        self.assertEqual(result["requests"][1]["status"], "stop")
        self.assertFalse(result["usage_complete"])
        self.assertEqual(self.send.call_count, 2)

    def test_failure_branch_write_error_still_drains_peer_and_retains_primary_error(self):
        self.error = TimeoutError("Ambiguous transport failure")
        self.fail_after_error = True
        thread, holder = self.run_async()
        self.await_started(0)
        self.await_started(1)
        self.release[0].set()
        self.assertTrue(self.write_failure.wait(3), "Failure-branch persistence was not exercised")
        self.assertTrue(thread.is_alive())
        self.release[1].set()
        result = self.finish(thread, holder)
        self.assertEqual(result["status"], "error", result)
        self.assertIn("TimeoutError", result["error"])
        self.assertNotIn("OSError", result["error"])
        self.assertEqual([r["submitted"] for r in result["requests"]], [None, True, False, False, False])
        self.assertEqual(result["requests"][1]["text"], "print(1)\n")
        self.assertAlmostEqual(result["observed_estimated_cost_usd"], .0002)
        self.assertTrue((Path(self.spec["output_dir"]) / "response-0001.json").exists())
        self.assertEqual(self.send.call_count, 2)

    def test_real_sigint_during_owner_wait_drains_running_peers_and_leaves_tail_unsent(self):
        specification = self.root / "round.json"
        specification.write_text(json.dumps(self.spec))
        gates = self.root / "gates"
        gates.mkdir()
        code = (
            "import json,sys,time\nfrom pathlib import Path\nfrom unittest import mock\n"
            "from scripts import deepinfra_generation as d\n"
            "from tests.test_thinking_generation import ThinkingTokenizer\n"
            "tokenizer=ThinkingTokenizer('qwen')\n"
            "tokenizer.pieces=tokenizer.pieces.copy()\n"
            "tokenizer.pieces.update({100+i:f'print({i})\\n' for i in range(5)})\n"
            "tokenizer.pieces[248046]='<|im_end|>'\n"
            "gates=Path(sys.argv[2])\n"
            "def send(payload,key,timeout):\n"
            " i=payload['seed']-17\n"
            " (gates/f'entered-{i}').touch()\n"
            " deadline=time.monotonic()+5\n"
            " while not (gates/f'release-{i}').exists():\n"
            "  if time.monotonic()>deadline: raise TimeoutError('Test gate timed out')\n"
            "  time.sleep(.01)\n"
            " body=dict(choices=[dict(index=0,text=f'print({i})\\n',token_ids=[100+i,248046],finish_reason='stop')],"
            "usage=dict(prompt_tokens=1,completion_tokens=2,estimated_cost=.0001))\n"
            " return 200,json.dumps(body)\n"
            "original_wait=d.wait\n"
            "def waiting(*args,**kwargs):\n"
            " (gates/'owner-waiting').touch()\n"
            " return original_wait(*args,**kwargs)\n"
            "with mock.patch.object(d,'load_tokenizer',return_value=tokenizer),mock.patch.object(d,'send',side_effect=send),"
            "mock.patch.object(d,'wait',side_effect=waiting):\n"
            " result=d.run(sys.argv[1])\n"
            "sys.exit(0 if result['status']=='success' else 1)\n")
        process = subprocess.Popen([sys.executable, "-c", code, str(specification), str(gates)],
            cwd=Path(d.__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def await_condition(predicate):
            deadline = time.monotonic() + 3
            while not predicate():
                self.assertIsNone(process.poll(), "Fixture worker exited before the test gate")
                self.assertLess(time.monotonic(), deadline, "Fixture worker did not reach the test gate")
                time.sleep(.01)
        try:
            await_condition(lambda: all((gates / f"entered-{i}").exists() for i in (0, 1)))
            await_condition(lambda: (gates / "owner-waiting").exists())
            process.send_signal(signal.SIGINT)
            await_condition(lambda: self.saved().get("error") == "KeyboardInterrupt()")
            self.assertIsNone(process.poll())
            self.assertFalse((gates / "entered-2").exists())
            for i in (0, 1):
                (gates / f"release-{i}").touch()
            stdout, stderr = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0, (stdout, stderr))
            result = self.saved()
            self.assertEqual(result["status"], "cancelled", result)
            self.assertEqual(result["error"], "KeyboardInterrupt()")
            self.assertEqual([r["submitted"] for r in result["requests"]], [True, True, False, False, False])
            self.assertEqual([r["status"] for r in result["requests"][:2]], ["stop", "stop"])
            self.assertTrue(result["usage_complete"])
            self.assertAlmostEqual(result["observed_estimated_cost_usd"], .0002)
            self.assertFalse((gates / "entered-2").exists())
        finally:
            for i in (0, 1):
                (gates / f"release-{i}").touch()
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)

    def test_normalization_failure_keeps_billed_response_and_drains_peer(self):
        self.invalid_text = True
        thread, holder = self.run_async()
        self.await_started(0)
        self.await_started(1)
        self.release[0].set()
        self.assertTrue(self.observed_failure.wait(3), "Normalization failure was not persisted")
        self.assertFalse(self.entered[2].is_set())
        self.release[1].set()
        result = self.finish(thread, holder)
        self.assertEqual(result["status"], "error", result)
        self.assertEqual([r["submitted"] for r in result["requests"]], [True, True, False, False, False])
        first, second = result["requests"][:2]
        self.assertEqual(first["generation_tokens"], 2)
        self.assertIsNone(first["answer_text"])
        self.assertEqual(second["text"], "print(1)\n")
        self.assertTrue(result["usage_complete"])
        self.assertAlmostEqual(result["observed_estimated_cost_usd"], .0003)
        response = json.loads((Path(self.spec["output_dir"]) / "response-0000.json").read_text())
        self.assertEqual(json.loads(response["body"])["choices"][0]["text"], "Provider decoding mismatch")
        self.assertEqual(self.send.call_count, 2)

    def test_all_prompts_are_preflighted_before_concurrent_dispatch(self):
        self.spec["requests"][-1].update(enable_thinking=True, max_prompt_tokens=1)
        result = self.run_round()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual([r["submitted"] for r in result["requests"]], [False] * 5)
        self.assertTrue(result["usage_complete"])
        self.send.assert_not_called()

    def test_invalid_concurrency_fails_without_any_http(self):
        for index, concurrency in enumerate((0, -1, 201, True, 1.5, "2")):
            with self.subTest(concurrency=concurrency):
                self.spec["runtime"]["concurrency"] = concurrency
                self.spec["output_dir"] = str(self.root / f"invalid-{index}")
                result = self.run_round()
                self.assertEqual(result["status"], "error", result)
                self.assertEqual([r["submitted"] for r in result["requests"]], [False] * 5)
                self.assertTrue(result["usage_complete"])
                self.send.assert_not_called()

    def test_omitted_concurrency_keeps_existing_serial_profiles(self):
        del self.spec["runtime"]["concurrency"]
        self.spec["requests"] = self.spec["requests"][:2]
        self.send.side_effect = lambda payload, key, timeout: self.response(payload["seed"] - 17)
        result = self.run_round()
        self.assertEqual(result["status"], "success", result)
        self.assertEqual([call.args[0]["seed"] for call in self.send.call_args_list], [17, 18])
        self.assertEqual([r["status"] for r in result["requests"]], ["stop", "stop"])

    def test_documented_upper_concurrency_bound_is_accepted(self):
        self.spec["runtime"]["concurrency"] = 200
        self.spec["requests"] = self.spec["requests"][:1]
        self.send.return_value = self.response(0)
        result = self.run_round()
        self.assertEqual(result["status"], "success", result)
        self.send.assert_called_once()
