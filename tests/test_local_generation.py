import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_local_runner import process_gone, wait_for


FAKE_WORKER = r'''
import hashlib,json,os,pathlib,subprocess,sys,time
path=pathlib.Path(sys.argv[1]); raw=path.read_bytes(); request=json.loads(raw)
mode=request["messages"][-1]["content"]; digest=hashlib.sha256(raw).hexdigest()
def emit(event,**values): print(json.dumps(dict(event=event,**values)),flush=True)
if mode=="startup": time.sleep(30)
if mode=="badjson": print("bad JSON",flush=True); sys.exit(0)
emit("start",request_sha256="wrong" if mode=="badhash" else digest,pid=os.getpid())
emit("ready",prompt="prompt",prompt_ids=[1,2],eos_token_ids=[9],
     provenance={"bad":float("nan")} if mode=="nan" else {},hash_seconds=0,load_seconds=0)
if mode=="duplicate_ready":
    emit("ready",prompt="prompt",prompt_ids=[1,2],eos_token_ids=[9],provenance={},hash_seconds=0,load_seconds=0)
if mode=="grandchild":
    child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (path.parent/"child.pid").write_text(str(child.pid))
reason="length" if mode in ("length","shortlength") else None if mode=="eos_without_finish" else "stop"
emit("chunk",seq=2 if mode=="gap" else 1,token=2 if mode in ("length","partial","stop_non_eos","shortlength") else 9,
     text="partial" if mode=="partial" else "answer" if mode=="length" else "",
     prompt_tokens=2,generation_tokens=2 if mode=="count" else 1,
     finish_reason=None if mode=="partial" else reason)
if mode=="partial":
    sys.stdout.buffer.write(b'{"event":"chunk","text":"\xe2\x82'); sys.stdout.flush(); time.sleep(30)
if mode!="missing_done": emit("done",request_sha256=digest,finish_reason=reason)
if mode=="tail": print("trailing garbage",flush=True)
sys.exit(3 if mode=="exit3" else 0)
'''


class LocalGenerationTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("src.data-pipeline.local_generation")
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root, self.index = Path(directory.name), 0
        worker = self.root / "fake_worker.py"
        worker.write_text(FAKE_WORKER)
        for name, value in (("INTERPRETER", sys.executable), ("WORKER", worker)):
            patch = mock.patch.object(self.api, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def request(self, mode="stop", **changes):
        return {"model_path": str(self.root), "messages": [{"role": "user", "content": mode}],
                "seed": 17, "temperature": 1.0, "top_p": 0.9, "max_tokens": 1,
                "wall_time_seconds": 3.0, "enable_thinking": False, **changes}

    def invoke(self, mode="stop", **changes):
        self.index += 1
        directory = self.root / str(self.index)
        response = self.api.generate(self.request(mode, **changes), directory)
        self.assertEqual(json.loads((directory / "response.json").read_text()), response)
        self.assertTrue(process_gone(response["pid"]))
        return response, directory

    def test_validation_precedes_directory_creation_and_dispatch(self):
        invalid = [("seed", True), ("seed", -1), ("seed", 2**64), ("seed", 1.5),
            ("temperature", -0.1), ("temperature", float("inf")), ("temperature", True),
            ("top_p", 0), ("top_p", 1.1), ("top_p", float("nan")),
            ("max_tokens", 0), ("max_tokens", True), ("max_tokens", 1.5),
            ("wall_time_seconds", 0), ("wall_time_seconds", float("nan")),
            ("model_path", "."), ("model_path", str(self.root / "missing")),
            ("enable_thinking", True), ("messages", []),
            ("top_k", 4), ("messages", [{"role": "user", "content": "\ud800"}]),
            ("fix_mistral_regex", 1), ("fix_mistral_regex", "true"), ("fix_mistral_regex", None),
            ("tokenizer_config", {"trust_remote_code": True}),
            ("messages", [{"role": "assistant", "content": "x"}]),
            ("messages", [{"role": "user", "content": "x", "extra": 1}]),
            ("messages", [{"role": "user", "content": 7}])]
        for field, value in invalid:
            with self.subTest(field=field, value=value), mock.patch.object(self.api.subprocess, "Popen") as launch:
                with self.assertRaises(ValueError):
                    self.api.generate(self.request(**{field: value}), self.root / "invalid")
                launch.assert_not_called()
                self.assertFalse((self.root / "invalid").exists())

    def test_stop_length_and_immutable_evidence(self):
        for mode, text, tokens in (("stop", "", [9]), ("length", "answer", [2])):
            response, directory = self.invoke(mode)
            self.assertEqual((response["status"], response["text"], response["tokens"]), (mode, text, tokens))
            self.assertEqual((response["prompt_tokens"], response["observed_generation_tokens"]), (2, 1))
            self.assertEqual(response["returncode"], 0)
            request = json.loads((directory / "request.json").read_text())
            self.assertEqual(request["environment"]["HF_HUB_OFFLINE"], "1")
            self.assertTrue((directory / "stderr.txt").exists())
            before = (directory / "request.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.api.generate(self.request(mode), directory)
            self.assertEqual((directory / "request.json").read_bytes(), before)

    def test_regex_normalization_and_source_profiles(self):
        for value in (False, True):
            self.assertIs(self.api.validate(self.request(fix_mistral_regex=value))["fix_mistral_regex"], value)
        self.assertIs(self.api.validate(self.request())["fix_mistral_regex"], False)
        _, directory = self.invoke(fix_mistral_regex=True)
        self.assertIs(json.loads((directory / "request.json").read_text())["fix_mistral_regex"], True)
        messages = [{"role": "system", "content": "Return Python."}, {"role": "user", "content": "Square n."}]
        controls = {key: self.request()[key] for key in
                    ("seed", "temperature", "top_p", "max_tokens", "wall_time_seconds")}
        for source in ("actor", "synthetic"):
            request = self.api.request_for(source, messages, **controls)
            self.assertEqual(request["model_path"], self.api.PROFILES[source]["model_path"])
            self.assertIs(request["fix_mistral_regex"], source == "synthetic")
            self.assertIs(request["enable_thinking"], False)
            self.assertEqual(request["messages"], messages)
            self.assertNotIn("source", request)
        with self.assertRaises(ValueError):
            self.api.request_for("synthetic", messages[1:], **controls)
        with self.assertRaises(ValueError):
            self.api.request_for("actor", messages, **{**controls, "top_k": 7})

    def test_extraction_preserves_raw_or_sole_fenced_program_bytes(self):
        for text, expected in (("# café\r\nprint(1)\r\n", "# café\r\nprint(1)\r\n".encode()),
                               ("x='```'\n", b"x='```'\n"),
                               (" \n```python\nprint(1)\n```\n ", b"print(1)\n"),
                               ("```py\r\nprint(2)\r\n```", b"print(2)\r\n")):
            with self.subTest(text=text):
                self.assertEqual(self.api.extract_program(text), expected)
        for text in ("", " \n", "print(\n", "Explanation\n```python\npass\n```", "\ud800",
                     "```python\npass\n```\n```python\npass\n```", "```javascript\npass\n```", "-" * 20000 + "1", "+".join(["1"] * 10000)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.api.extract_program(text)

    @unittest.skipUnless(os.environ.get("ROPOT_MLX_TESTS") == "1", "set ROPOT_MLX_TESTS=1 for real tokenizers")
    def test_official_and_mlx_tokenizers_match_three_fixture_histories(self):
        official = "/Users/michal/Workspace/models/mistralai/Ministral-3-8B-Instruct-2512/5b26027e7b19eeb4b7352e1fed3926375dd2cb4d"
        runtime = "/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python"
        system = {"role": "system", "content": "Return only a Python 3 program, without explanation or Markdown."}
        square = [system, {"role": "user", "content": "Read integer n and print its square."}]
        distinct = [system, {"role": "user", "content": "Read integers and print the count of distinct values."}]
        repair = [*square, {"role": "assistant", "content": "print(int(input()))"},
                  {"role": "user", "content": "Visible input 3 expected 9, observed 3. Correct the program."}]
        with mock.patch.object(self.api, "INTERPRETER", runtime):
            expected = self.api.reference_prompts(official, [square, distinct, repair])
            actual = self.api.reference_prompts(self.api.PROFILES["synthetic"]["model_path"], [square, distinct, repair])
        self.assertEqual(actual, expected)
        for prompt in actual:
            self.assertEqual((prompt["ids"][0], prompt["ids"].count(1)), (1, 1))
            self.assertNotIn("Le Chat", prompt["prompt"])

    def test_malformed_or_incomplete_protocol_never_succeeds(self):
        for mode in ("badjson", "badhash", "gap", "count", "duplicate_ready", "missing_done", "exit3", "tail", "nan",
                     "stop_non_eos", "shortlength", "eos_without_finish"):
            with self.subTest(mode=mode):
                response, _ = self.invoke(mode, max_tokens=8 if mode == "shortlength" else 1)
                self.assertEqual(response["status"], "worker_error")

    def test_timeout_keeps_complete_chunks_and_exact_partial_tail(self):
        response, directory = self.invoke("partial", max_tokens=8, wall_time_seconds=1)
        self.assertEqual(response["status"], "timeout")
        self.assertEqual((response["text"], response["tokens"], response["observed_generation_tokens"]),
                         ("partial", [2], 1))
        self.assertTrue((directory / "stdout.jsonl").read_bytes().endswith(b'{"event":"chunk","text":"\xe2\x82'))
        self.assertLess(response["elapsed_seconds"], 2)

    def test_startup_is_inside_deadline(self):
        response, directory = self.invoke("startup", wall_time_seconds=0.05)
        self.assertEqual(response["status"], "timeout")
        self.assertEqual((directory / "stdout.jsonl").read_bytes(), b"")
        self.assertLess(response["elapsed_seconds"], 2)

    def test_cancellation_records_outcome_reaps_and_reraises(self):
        directory = self.root / "cancelled"
        with mock.patch.object(self.api, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.api.generate(self.request("startup"), directory)
        response = json.loads((directory / "response.json").read_text())
        self.assertEqual(response["status"], "cancelled")
        self.assertTrue(process_gone(response["pid"]))

    def test_normal_exit_still_kills_owned_grandchild(self):
        response, directory = self.invoke("grandchild")
        child = int((directory / "child.pid").read_text())
        self.addCleanup(lambda: None if process_gone(child) else os.kill(child, signal.SIGKILL))
        self.assertEqual(response["status"], "stop")
        self.assertTrue(wait_for(lambda: process_gone(child)))

    def test_actual_worker_watchdog_kills_group_when_parent_pipe_closes(self):
        worker = Path(self.api.__file__).with_name("mlx_worker.py")
        source = ("import importlib.util\n"
                  f"spec=importlib.util.spec_from_file_location('worker',{str(worker)!r})\n"
                  "worker=importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)\n"
                  "print('ready',flush=True); worker.watch_parent()\n")
        with subprocess.Popen([sys.executable, "-I", "-B", "-c", source], start_new_session=True,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            self.assertEqual(child.stdout.readline(), b"ready\n")
            child.stdin.close()
            try:
                child.wait(timeout=3)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=3)
            self.assertEqual(child.returncode, -signal.SIGKILL)

    def test_parent_observation_error_cannot_return_success(self):
        original = self.api._exited
        def fail_after_exit(process):
            if original(process):
                raise OSError("observation failed after worker completed")
            return False
        directory = self.root / "observation"
        with mock.patch.object(self.api, "_exited", side_effect=fail_after_exit), self.assertRaises(OSError):
            self.api.generate(self.request(), directory)
        response = json.loads((directory / "response.json").read_text())
        self.assertEqual(response["status"], "parent_error")
        self.assertIn("observation failed", response["parent_error"])

    def test_cleanup_error_persists_outcome_and_preserves_cancellation(self):
        original, observe = self.api._kill_group, self.api._exited
        def fail_after_kill(process):
            original(process)
            raise OSError("cleanup failed after group killed")
        for cancelled in (False, True):
            directory = self.root / f"cleanup-{cancelled}"
            with self.subTest(cancelled=cancelled), \
                 mock.patch.object(self.api, "_kill_group", side_effect=fail_after_kill), \
                 mock.patch.object(self.api, "_exited", side_effect=KeyboardInterrupt if cancelled else observe):
                with self.assertRaises(KeyboardInterrupt if cancelled else OSError):
                    self.api.generate(self.request(), directory)
                response = json.loads((directory / "response.json").read_text())
                self.assertEqual(response["status"], "cancelled" if cancelled else "worker_error")
                self.assertTrue(process_gone(response["pid"]))
