import copy
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_long_batch_qualification as long_fixture


class QualificationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.f = long_fixture.LongBatchQualificationTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.H, self.root = self.f.H, self.f.f.root
        self.job, self.events = self.f.protocol(); self.files = self.pack(self.job, self.events)
        self.pin = self.H.sha256(self.files["response.json"]).hexdigest()

    def pack(self, job, events):
        events = copy.deepcopy(events); raw = self.H._raw(job); identity = self.H.sha256(raw).hexdigest()
        for event in events:
            event["job_sha256"] = identity
            if event["event"] == "controls_checked": event["request_sha256"] = identity
        stdout = b"".join(self.H._raw(e) for e in events); parsed = self.H.validate_events(stdout, job, pid=7)
        response = dict(parsed, status="passed", job_sha256=identity, pid=7, returncode=0, group_absent=True,
            parent_error=None, cleanup_error=None, elapsed_seconds=239.5, work_seconds=239, cleanup_seconds=.5,
            telemetry=[dict(t_seconds=0, cpu_seconds=0, rss_bytes=1, available_bytes=64 * 1024**3, swap_bytes=0)], telemetry_complete=True)
        files = {"job.json": raw, "stdout.jsonl": stdout, "response.json": self.H._raw(response)}
        if "cache_limit_bytes" in job:
            files["cache-control.json"] = self.H._raw(dict(job_sha256=identity, requested_bytes=job["cache_limit_bytes"], observed_bytes=job["cache_limit_bytes"], previous_bytes=17))
        return files

    def saved(self, files, name="qualification"):
        directory = self.root / name; directory.mkdir()
        for name, raw in files.items(): (directory / name).write_bytes(raw)
        return directory

    def test_admission_binds_exact_controls_inputs_and_actual_stressed_minimum(self):
        admission = self.H.qualification_checked(self.files)
        self.assertEqual(admission["request_controls"], {k: v for k, v in self.job["request"].items() if k not in ("messages", "seed")})
        self.assertEqual(admission["config"], self.f.config); self.assertEqual(admission["source_hashes"], self.job["source_hashes"])
        self.assertEqual((admission["prompt_ceiling_tokens"], admission["stress_prompt_tokens"]), (3000, dict(min=3000, max=3000)))
        self.assertEqual(admission["inputs"], {n: dict(sha256=self.H.sha256(v).hexdigest(), bytes=len(v)) for n, v in self.files.items()})
        self.assertNotIn("cache_limit_bytes", admission)
        for files in ({**self.files, "other.json": b"{}"}, {k: v for k, v in self.files.items() if k != "stdout.jsonl"}, {**self.files, "job.json": "text"}):
            with self.assertRaises((ValueError, TypeError)): self.H.qualification_checked(files)

    def test_otherwise_valid_raw_parent_pid_config_min_and_work_faults_are_rejected(self):
        for fault in ("job_bytes", "raw", "incomplete", "parent", "fields", "pid", "config", "minimum", "work", "telemetry_false", "telemetry_error"):
            files = dict(self.files); response = json.loads(files["response.json"]); events = copy.deepcopy(self.events)
            if fault == "job_bytes": files["job.json"] += b" "
            if fault == "raw": files["stdout.jsonl"] += b"{}\n"
            if fault == "incomplete": files["stdout.jsonl"] = b"\n".join(files["stdout.jsonl"].splitlines()[:-1]) + b"\n"
            if fault == "parent": response["observed_generation_tokens"] += 1
            if fault == "fields": response["extra"] = 1
            if fault == "pid": response["pid"] = 8
            if fault == "config":
                job = copy.deepcopy(self.job); job["windows"][-1]["config"]["prefill_batch_size"] = 2; files = self.pack(job, events)
            if fault == "minimum":
                next(e for e in events if e["event"] == "qualification")["stress"]["prompt_token_ceiling"] = 3001
                response["events"] = events; response["long_stress"]["prompt_token_ceiling"] = 3001
                files["stdout.jsonl"] = b"".join(self.H._raw(e) for e in events)
            if fault == "work": response.update(work_seconds=241, elapsed_seconds=241.5)
            if fault == "telemetry_false": response["telemetry_complete"] = False
            if fault == "telemetry_error": response["telemetry"][0]["error"] = "sample failed"
            if fault != "config": files["response.json"] = self.H._raw(response)
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.H.qualification_checked(files)

    def test_declared_zero_requires_exact_integer_readback_and_no_optional_file_drift(self):
        job = dict(self.job, cache_limit_bytes=0); files = self.pack(job, self.events)
        self.assertEqual(self.H.qualification_checked(files)["cache_limit_bytes"], 0)
        for key, value in (("observed_bytes", False), ("requested_bytes", 4 * 1024**3), ("job_sha256", "0" * 64)):
            proof = json.loads(files["cache-control.json"]); proof[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.H.qualification_checked({**files, "cache-control.json": self.H._raw(proof)})
        destination = self.root / "cache-copy"; self.H.copy_qualification(files, destination, self.H.sha256(files["response.json"]).hexdigest())
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, files)
        with self.assertRaises(ValueError): self.H.qualification_checked({k: v for k, v in files.items() if k != "cache-control.json"})
        with self.assertRaises(ValueError): self.H.qualification_checked({**self.files, "cache-control.json": files["cache-control.json"]})

    def test_trusted_response_pin_exact_copy_and_directory_sync_order(self):
        files = dict(self.files); files["response.json"] = b" \n" + files["response.json"]
        directory = self.saved(files); pin = self.H.sha256(files["response.json"]).hexdigest()
        with self.assertRaises(ValueError): self.H.read_qualification(directory, self.pin)
        with self.assertRaises(ValueError): self.H.copy_qualification(files, self.root / "bad-pin", self.pin)
        self.assertFalse((self.root / "bad-pin").exists())
        admission, retained = self.H.read_qualification(directory, pin); self.assertEqual(retained, files)
        destination = self.root / "copied"; trace = mock.Mock()
        with mock.patch.object(self.H, "_write", wraps=self.H._write) as write, mock.patch.object(self.H, "_sync", wraps=self.H._sync) as sync:
            trace.attach_mock(write, "write"); trace.attach_mock(sync, "sync")
            self.assertEqual(self.H.copy_qualification(retained, destination, pin), admission)
        self.assertEqual([c[0] for c in trace.mock_calls], ["write"] * 3 + ["sync", "sync"])
        self.assertEqual(sync.call_args_list, [mock.call(destination), mock.call(destination.parent)])
        self.assertEqual({p.name: p.read_bytes() for p in destination.iterdir()}, files)
        with self.assertRaises(FileExistsError): self.H.copy_qualification(files, destination, pin)
        with self.assertRaises(ValueError): self.H.read_qualification(directory, "bad")

    def test_offline_history_does_not_read_missing_or_symlink_source_assets(self):
        job = copy.deepcopy(self.job); originals = job["source_hashes"]
        relocated = {p: str(self.root / "offline" / Path(p).name) for p in originals}
        job["source_hashes"] = {relocated[p]: digest for p, digest in originals.items()}; job["worker"] = relocated[job["worker"]]
        files = self.pack(job, self.events)
        for symlink in (False, True):
            if symlink:
                link = Path(job["worker"]); link.parent.mkdir(); link.symlink_to(self.job["worker"])
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Historical source read")), \
                 mock.patch.object(Path, "resolve", side_effect=AssertionError("Historical source resolve")), \
                 mock.patch.object(Path, "stat", side_effect=AssertionError("Historical source stat")):
                admission = self.H.qualification_checked(files)
            with self.assertRaises(ValueError): self.H.current_source_compatible(admission)
        current = self.H.qualification_checked(self.files); self.assertIsNone(self.H.current_source_compatible(current))
        for path in current["source_hashes"]:
            changed = copy.deepcopy(current); changed["source_hashes"][path] = "0" * 64
            with self.subTest(path=path), self.assertRaises(ValueError): self.H.current_source_compatible(changed)
        missing = copy.deepcopy(current); missing["source_hashes"].pop(next(iter(missing["source_hashes"])))
        with self.assertRaises(ValueError): self.H.current_source_compatible(missing)

    def test_reader_refuses_symlinks_nonregular_files_and_bounded_oversize(self):
        directory = self.saved(self.files); target = directory / "stdout.jsonl"; target.rename(directory / "raw")
        target.symlink_to(directory / "raw")
        with self.assertRaises((ValueError, OSError)): self.H.read_qualification(directory, self.pin)
        target.unlink(); target.mkdir()
        original_open, opened = self.H.os.open, []
        def tracked_open(*args, **kwargs):
            descriptor = original_open(*args, **kwargs); opened.append(descriptor); return descriptor
        with mock.patch.object(self.H.os, "open", side_effect=tracked_open), self.assertRaises((ValueError, OSError)):
            self.H.read_qualification(directory, self.pin)
        try:
            with self.assertRaises(OSError): self.H.os.fstat(opened[-1])
        finally:
            try: self.H.os.close(opened[-1])
            except OSError: pass
        target.rmdir(); target.write_bytes(self.files["stdout.jsonl"])
        limits = dict(self.H.QUALIFICATION_LIMITS, **{"stdout.jsonl": 8})
        with mock.patch.object(self.H, "QUALIFICATION_LIMITS", limits), self.assertRaises(ValueError): self.H.read_qualification(directory, self.pin)
