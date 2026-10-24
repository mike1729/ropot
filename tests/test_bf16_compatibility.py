from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from tests.test_local_runner import process_gone

FAKE = r'''
import hashlib,json,os,pathlib,subprocess,sys,time
p=pathlib.Path(sys.argv[-1]); identity=hashlib.sha256(p.read_bytes()).hexdigest()
def emit(event,**values): print(json.dumps(dict(event=event,**values)),flush=True)
emit('start',request_sha256='bad' if MODE=='hash' else identity,pid=os.getpid()+(MODE=='pid'))
if MODE in ('malformed','object','nan'): print('[]' if MODE=='object' else json.dumps(dict(event='done',request_sha256=identity,result={'bad':float('nan')})) if MODE=='nan' else 'invalid JSON',flush=True); sys.exit(0)
if MODE=='timeout':
 child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
 (p.parent/'child.pid').write_text(str(child.pid)); emit('phase',phase='partial'); time.sleep(30)
if MODE in ('stdout','stderr'):
 stream=sys.stdout if MODE=='stdout' else sys.stderr; stream.write('x'*2048); stream.flush(); time.sleep(30)
emit('done',request_sha256=identity,result={'fixture':True})
'''


class Bf16CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.B = importlib.import_module("experiments.bf16_compatibility")
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name).resolve()
        self.model = self.root / "model"; self.model.mkdir(); (self.model / "weights").write_bytes(b"fixture")
        self.manifest = {"files": [dict(path="weights", bytes=7, sha256=sha256(b"fixture").hexdigest())]}
        self.worker, self.real_worker, self.index = self.root / "worker.py", self.B.WORKER, 0
        manifest = self.root / "manifest.json"; manifest.write_text(json.dumps(self.manifest))
        upstream = self.root / "upstream.json"; upstream.write_text("{}")
        for name, value in dict(INTERPRETER=sys.executable, WORKER=self.worker, MODEL_PATH=self.model,
            MANIFEST_PATH=manifest, UPSTREAM_PATH=upstream, WALL_SECONDS=1.0, CLEANUP_SECONDS=3.0, STDOUT_LIMIT=512, STDERR_LIMIT=512).items():
            patch = mock.patch.object(self.B, name, value); patch.start(); self.addCleanup(patch.stop)

    def invoke(self, mode):
        self.worker.write_text("MODE=" + repr(mode) + "\n" + FAKE); self.index += 1; directory = self.root / str(self.index)
        result = self.B.run(directory)
        self.assertEqual(json.loads((directory / "receipt.json").read_text()), result)
        self.assertTrue(result["group_absent"]); self.assertTrue(process_gone(result["pid"]))
        return result, directory

    def test_manifest_exact_files_hashes_and_symlinks(self):
        self.B.validate_manifest(self.model, self.manifest)
        (self.model / ".cache").mkdir(); (self.model / ".cache/ignored").write_bytes(b"cache")
        self.B.validate_manifest(self.model, self.manifest)
        extra = self.model / "extra"; extra.write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "Manifest"): self.B.validate_manifest(self.model, self.manifest)
        extra.unlink(); (self.model / "weights").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Manifest"): self.B.validate_manifest(self.model, self.manifest)
        twin = self.root / "same"; twin.write_bytes(b"fixture"); (self.model / "weights").unlink(); (self.model / "weights").symlink_to(twin)
        with self.assertRaisesRegex(ValueError, "Manifest"): self.B.validate_manifest(self.model, self.manifest)

    def test_completion_mask_and_frozen_bounds(self):
        ids, labels = self.B.completion_labels([1] * 120, [2] * 8)
        self.assertEqual((ids, labels), ([1] * 120 + [2] * 8, [-100] * 120 + [2] * 8))
        for prompt, completion in (([], [2] * 128), ([1] * 128, []), ([1] * 119, [2] * 8),
                                   ([1] * 505, [2] * 8), ([True] * 120, [2] * 8), ([1] * 120, [-1] * 8)):
            with self.assertRaises(ValueError): self.B.completion_labels(prompt, completion)

    def test_owned_worker_protocol_deadline_soft_caps_and_partial_evidence(self):
        synced = set(); sync, write = self.B.os.fsync, self.B._write
        def observed(fd): synced.add(self.B.os.fstat(fd).st_ino); return sync(fd)
        def receipt(path, value):
            if Path(path).name == "receipt.json": self.assertTrue({(Path(path).parent / n).stat().st_ino for n in ("stdout.jsonl", "stderr.txt")} <= synced)
            return write(path, value)
        with mock.patch.object(self.B.os, "fsync", side_effect=observed), mock.patch.object(self.B, "_write", side_effect=receipt): self.invoke("ok")
        for mode, status in (("ok", "passed"), ("hash", "worker_error"), ("pid", "worker_error"),
                             ("malformed", "worker_error"), ("object", "worker_error"), ("nan", "worker_error"), ("stdout", "output_limit"), ("stderr", "output_limit"), ("timeout", "timeout")):
            result, directory = self.invoke(mode)
            self.assertEqual(result["status"], status); self.assertEqual(result["generator_requests"], 0)
            self.assertEqual(result["request_sha256"], sha256((directory / "request.json").read_bytes()).hexdigest())
            if mode == "ok": self.assertEqual(result["events"][0]["pid"], result["pid"])
            if mode == "timeout":
                self.assertIn(b"partial", (directory / "stdout.jsonl").read_bytes())
                self.assertTrue(process_gone(int((directory / "child.pid").read_text())))

    def test_cancellation_writes_receipt_then_reraises(self):
        self.worker.write_text("MODE='timeout'\n" + FAKE); directory = self.root / "cancelled"
        with mock.patch.object(self.B, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt): self.B.run(directory)
        result = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(result["status"], "cancelled"); self.assertTrue(result["group_absent"])
        kill = self.B._kill_group
        def failed(process): kill(process); raise OSError("cleanup failed")
        with mock.patch.object(self.B, "_kill_group", side_effect=failed): result, _ = self.invoke("ok")
        self.assertEqual(result["status"], "worker_error"); self.assertIn("cleanup failed", result["cleanup_error"])

    def test_real_isolated_worker_reports_manifest_failure_before_heavy_imports(self):
        (self.model / "weights").write_bytes(b"changed")
        with mock.patch.object(self.B, "WORKER", self.real_worker), mock.patch.object(self.B, "STDERR_LIMIT", 16384): result = self.B.run(self.root / "real-worker")
        self.assertEqual(result["status"], "worker_error")
        self.assertTrue(any(e.get("event") == "error" and e.get("phase") == "manifest" for e in result["events"]))
