from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_verification as native, test_bf16_compatibility as process


class CpuThroughputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.root = Path(temporary.name)
        self.C = importlib.import_module("experiments.cpu_throughput"); self.V = native.v; self.R = self.V.runner
        self.discover = self.C.shutil.which
        self.N = NS(_daemon=lambda: "daemon", _docker=lambda *args: '{"NCPU":4,"MemTotal":1024,"ServerVersion":"fixture"}')
        self.worker = self.root / "fake.py"; self.worker.write_text("# fixture")
        self.lock = self.root / "lock"; self.lock.write_bytes(b"existing")
        for target, name, value in ((self.C.shutil, "which", lambda name: "/fixture/bin/docker"), (self.C, "_native", lambda: (self.R, self.V, self.N)), (self.C, "_inventory", lambda opid: ("daemon", [], False)),
            (self.R, "contract", lambda: native.CONTRACT), (self.C, "INTERPRETER", sys.executable), (self.C, "WORKER", self.worker), (self.C, "WALL_SECONDS", .5), (self.C, "CLEANUP_SECONDS", .05)):
            patch = mock.patch.object(target, name, value); patch.start(); self.addCleanup(patch.stop)
        self.task, self.programs, self.expected = self.C.fixtures(); self.opid = "a" * 12
        self.items = [self.V.request(self.task, "b" * 64, program) for program in self.programs]

    def test_docker_discovery_is_frozen_and_missing_cli_fails_before_launch(self):
        docker = self.root / "docker"; docker.write_text("#!/bin/sh\nexit 0\n"); docker.chmod(0o755)
        with mock.patch.object(self.C.shutil, "which", return_value=str(docker)):
            path = self.C.build_plan()["environment"]["PATH"]
        self.assertEqual(Path(path.split(":")[0]) / "docker", docker)
        self.assertEqual(self.discover("docker", path=path), str(docker))
        for missing in (None, "relative/docker"):
            with mock.patch.object(self.C.shutil, "which", return_value=missing), self.assertRaisesRegex(ValueError, "Docker CLI"):
                self.C.run(self.root / "never-created", self.lock)
            self.assertFalse((self.root / "never-created").exists())

    def log(self, items=None, opid=None, concurrency=1):
        items = self.items if items is None else items; opid = opid or self.opid; log = native.native(items)
        log.eval.task = opid; log.eval.config = NS(max_samples=concurrency, max_sandboxes=concurrency, retry_on_error=0,
            fail_on_error=True, continue_on_fail=True, sandbox_cleanup=True, sandbox_prebuilt=True)
        for sample, item in zip(log.samples, items):
            i = self.programs.index(item["program"]); sample.store["cases"] = []
            for case in item["cases"]:
                if not case["visible"] and any(r["status"] != "passed" for r in sample.store["cases"]): break
                n = int(case["input"]); status = "timeout" if i in (4, 5) and n == 12 else "wrong_answer" if i in (2, 3) and n == 11 or i in (6, 7) and n == 13 else "passed"
                sample.store["cases"].append(dict(index=case["index"], status=status, stdout=None if status == "timeout" else "0" if status == "wrong_answer" else case["answer"],
                    stderr=None if status == "timeout" else "", returncode=None if status == "timeout" else 0, elapsed_seconds=.1))
                if not case["visible"] and status != "passed": break
            sample.store["container"].update(Id=f"{opid}-{i}", Config=dict(User="65534:65534", Labels={"com.docker.compose.project": f"inspect-{opid}-iabcdef"}),
                HostConfig=dict(NanoCpus=10**9, Memory=512 * 1024**2, MemorySwap=512 * 1024**2, PidsLimit=32, ReadonlyRootfs=True, NetworkMode="none", CapDrop=["ALL"]))
        return log

    def test_independent_ordered_case_oracle_and_native_mutations(self):
        self.assertEqual(len(set(self.programs)), 8)
        for i, statuses in enumerate(self.expected):
            mapping = {r["index"]: r["status"] for r in statuses}
            self.assertEqual(sum(s != "passed" for s in mapping.values()), 0 if i < 2 else 1)
        checked = self.C.check_log(self.log(), self.items, self.expected, native.CONTRACT, self.opid, 1)
        self.assertEqual([r["status"] for r in checked["results"]], ["passed"] * 2 + ["failed"] * 6)
        for i in (2, 3, 4, 5): self.assertEqual(len(self.log().samples[i].store["cases"]), 2)
        for damage in ("label", "container", "limits", "options", "prefix", "capture", "error"):
            log = self.log(); first = log.samples[0]
            if damage == "label": first.store["container"]["Config"]["Labels"]["com.docker.compose.project"] = "foreign"
            if damage == "container": log.samples[1].store["container"]["Id"] = first.store["container"]["Id"]
            if damage == "limits": first.store["container"]["HostConfig"]["Memory"] = 1
            if damage == "options": log.eval.config.max_samples = 4
            if damage == "prefix": first.store["cases"].pop()
            if damage == "capture": first.store["cases"][0]["stdout"] = "999"
            if damage == "error": first.error = NS(message="fixture failure")
            with self.subTest(damage=damage), self.assertRaises(ValueError): self.C.check_log(log, self.items, self.expected, native.CONTRACT, self.opid, 1)

    def test_public_task_adapter_maps_exact_items_and_concurrency(self):
        with mock.patch.object(self.R, "inspect_eval", return_value=[self.log()]) as evaluate:
            for c in (1, 2, 4):
                self.C.coalesced(self.items, self.root, c, self.opid, native.CONTRACT, {})
                task = evaluate.call_args.args[0]; options = evaluate.call_args.kwargs
                self.assertEqual((task.name, options["max_samples"], options["max_sandboxes"]), (self.opid, c, c))
                self.assertEqual([s.metadata for s in task.dataset], [s.metadata for s in self.log().samples])
                self.assertFalse(options["score"]); self.assertTrue(options["sandbox_cleanup"])
        job = self.C.build_plan(); self.assertEqual(len(job["operations"]), 22)
        self.assertEqual(sum(len(o["indices"]) for o in job["operations"]), 64)

    def test_intent_is_fsynced_before_dispatch_and_unknown_closure_stops_grid(self):
        job = self.C.build_plan(); directory = self.root / "grid"; directory.mkdir(); (directory / "ops").mkdir()
        synced = set(); sync = self.C.os.fsync
        def observed(fd): synced.add(self.C.os.fstat(fd).st_ino); return sync(fd)
        def dispatch(op, items, logs, contract, host):
            intent = logs.parent / "intent.json"; self.assertEqual(json.loads(intent.read_bytes())["operation"], op)
            self.assertTrue({p.stat().st_ino for p in (intent, logs.parent, logs.parent.parent, logs.parent.parent.parent)} <= synced)
            logs.mkdir(); log = self.log(items, op["operation_id"], op["concurrency"]); path = logs / "native.eval"; path.write_bytes(b"fixture"); log.location = str(path); return log
        with mock.patch.object(self.C.os, "fsync", side_effect=observed), mock.patch.object(self.C, "_dispatch", side_effect=dispatch):
            self.C._sync(directory)
            result = self.C._operation(job, directory, job["operations"][0], set())
        self.assertEqual(result["closure"]["daemon_id"], "daemon")
        path = self.root / "job.json"; path.write_bytes(self.C._raw(job))
        with mock.patch.object(self.C.os, "fdopen", return_value=io.StringIO()), mock.patch.object(self.C.os, "dup", return_value=1), mock.patch.object(self.C.os, "dup2"), mock.patch.object(self.C.threading, "Thread"), \
             mock.patch.object(self.C, "_dispatch", side_effect=dispatch) as calls, mock.patch.object(self.C, "_closed", side_effect=ValueError("unproven")), self.assertRaisesRegex(ValueError, "unproven"):
            with mock.patch.object(self.C.os, "fsync", side_effect=observed): self.C._worker(path)
        self.assertEqual(calls.call_count, 1)

    def test_inventory_requires_same_daemon_and_positive_absence(self):
        for value in (("other", [], False), ("daemon", ["owned"], False), ("daemon", [], True), OSError("unknown")):
            with self.subTest(value=value), mock.patch.object(self.C, "_inventory", **({"side_effect": value} if isinstance(value, Exception) else {"return_value": value})):
                with self.assertRaises((ValueError, OSError)): self.C._closed(self.opid, "daemon")

    def test_parent_reaps_partial_worker_then_checks_every_intent_and_cancels(self):
        prepare = """import base64
j=json.loads(p.read_bytes())
for op in j['operations'][:2]:
 d=p.parent/'ops'/op['operation_id'];d.mkdir(parents=True)
 intent=dict(daemon_id='daemon',operation=op,job_sha256=identity)
 (d/'intent.json').write_text(json.dumps(intent))
 if BROKEN_RESULT and op==j['operations'][0]:(d/'result.json').write_text('{')
"""
        for bad in (False, True, "result"):
            self.worker.write_text("MODE='timeout'\nBROKEN_RESULT=" + repr(bad == "result") + "\n" + process.FAKE.replace("request_sha256=", "job_sha256=").replace("emit('phase',phase='partial')", "exec(" + repr(prepare) + ")"))
            directory = self.root / f"partial-{bad}"
            with mock.patch.object(self.C, "_inventory", **({"side_effect": [OSError("inventory unknown"), ("daemon", [], False)]} if bad is True else {"return_value": ("daemon", [], False)})) as inventory:
                result = self.C.run(directory, self.lock)
            self.assertEqual(result["status"], "timeout"); self.assertTrue(result["group_absent"])
            self.assertTrue(process.process_gone(int((directory / "child.pid").read_text())))
            self.assertEqual(result["attempted_eval_starts"], 2); self.assertEqual(inventory.call_count, 2)
            self.assertEqual(len(result["resource_proofs"]), 1 if bad is True else 2)
            self.assertEqual(bool(result["cleanup_error"]), bool(bad))
        directory = self.root / "cancelled"
        with mock.patch.object(self.C, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt): self.C.run(directory, self.lock)
        result = json.loads((directory / "response.json").read_bytes())
        self.assertEqual(result["status"], "cancelled"); self.assertTrue(result["group_absent"])
