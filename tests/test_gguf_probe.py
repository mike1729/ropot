import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from urllib.error import URLError
from tests.test_local_runner import process_gone
probe = importlib.import_module("experiments.gguf_probe")
class GGUFProbeTests(unittest.TestCase):
    def test_http_barriers_deadlines_and_owned_group_cleanup(self):
        original, kill, killpg = subprocess.Popen, probe._kill_group, probe.os.killpg
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.gguf", "template.jinja", "llama-server"):
                (root / name).write_bytes(b"fixture")
            item = {"messages": [{"role": "user", "content": "Square n."}],
                    "controls": {"temperature": 0, "top_p": 1, "seed": 42}}
            plan = {"model_path": str(root / "model.gguf"), "template_path": str(root / "template.jinja"),
                    "llama_dir": str(root), "gguf_sha256": "fixture", "requests": [item] * 4,
                    "limits": {"reply_tokens": 8, "request_seconds": 0.1, "route_seconds": 1}}
            for mode in ("ok", "cleanup", "startup", "wrapped_timeout", "ids", "timeout"):
                output, calls, children, probes = root / mode, [], [], []
                def group(pid, sig):
                    if mode == "cleanup" and sig == 0 and not probes:
                        probes.append(pid); raise PermissionError("Darwin transient idle probe")
                    return killpg(pid, sig)
                def kill_group(child):
                    kill(child)
                    if mode == "cleanup": raise PermissionError("Darwin transient after kill")
                def start(argv, **kwargs):
                    child = original([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
                    children.append(child)
                    return child
                def http(base, path, body=None, **kwargs):
                    calls.append((path, body))
                    if mode == "wrapped_timeout": raise URLError(TimeoutError("health timed out"))
                    if mode == "startup" or (mode == "timeout" and path == "/completion"):
                        time.sleep(0.3)
                    if path == "/health": return {"status": "ok"}
                    if path == "/v1/models": return {"data": [{"id": json.loads((output / "launch.json").read_text())["alias"]}]}
                    if path == "/props": return {"model_path": plan["model_path"], "total_slots": 1,
                                                 "default_generation_settings": {"n_ctx": 8192}}
                    if path == "/apply-template": return {"prompt": "prompt"}
                    if path == "/tokenize": return {"tokens": [1, 99 if mode == "ids" else 3]}
                    return {"content": "print(1)", "stop_type": "eos", "truncated": False,
                            "generation_settings": {k: body[k] for k in ("samplers", "temperature", "top_p", "seed", "n_predict")},
                            "timings": {"cache_n": 0}, "tokens_evaluated": 2, "tokens_predicted": 1, "tokens": [2]}
                with self.subTest(mode=mode), mock.patch.object(probe, "_hash_file", return_value="fixture"), \
                     mock.patch.object(probe.subprocess, "run", return_value=subprocess.CompletedProcess(
                         [], 0, b"fixture version", b"")), \
                     mock.patch.object(probe.subprocess, "Popen", side_effect=start), \
                     mock.patch.object(probe, "_kill_group", side_effect=kill_group), \
                     mock.patch.object(probe.os, "killpg", side_effect=group), \
                     mock.patch.object(probe, "_http", side_effect=http), mock.patch.object(probe, "_closed", return_value=True), \
                     mock.patch.object(probe.socket, "socket") as sockets:
                    sockets.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 43001)
                    try:
                        rows = probe.run_gguf(plan, [{"prompt": "prompt", "ids": [1, 3]}] * 4, output)
                    finally:
                        for child in children:
                            if child.poll() is None:
                                child.kill()
                                child.wait(timeout=3)
                    lifecycle = json.loads((output / "lifecycle.json").read_text())
                    self.assertTrue(lifecycle["group_absent"] and lifecycle["port_closed"])
                    self.assertEqual("cleanup_error" in lifecycle, mode == "cleanup")
                    self.assertTrue(all(process_gone(child.pid) for child in children))
                    completions = [body for path, body in calls if path == "/completion"]
                    self.assertTrue(len(calls) == 1 if mode == "wrapped_timeout" else all(body["cache_prompt"] is False and body["return_tokens"] is True for body in completions))
                    self.assertEqual(len(completions), {"ok": 4, "cleanup": 4, "startup": 0, "wrapped_timeout": 0, "ids": 0, "timeout": 1}[mode], lifecycle)
                    self.assertEqual([r["status"] for r in rows], ["completed"] * 4 if mode in ("ok", "cleanup") else
                                     ["not_dispatched"] * 4 if mode in ("startup", "wrapped_timeout") else ["error", *( ["not_dispatched"] * 3)])

    def test_nested_startup_alarm_cannot_extend_route_deadline(self):
        started = time.monotonic()
        with self.assertRaises(TimeoutError), probe._alarm(0.05), probe._alarm(1):
            time.sleep(0.2)
        self.assertLess(time.monotonic() - started, 0.15)
