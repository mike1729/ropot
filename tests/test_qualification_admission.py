import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_long_batch_qualification as long_fixture


class QualificationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.f = long_fixture.LongBatchQualificationTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.H, self.root = self.f.H, self.f.f.root
        self.H._qualification_memo.clear(); self.addCleanup(self.H._qualification_memo.clear)
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

    def test_memo_complete_inventory_hits_are_detached_and_reordered(self):
        for files in (self.files, self.pack(dict(self.job, cache_limit_bytes=0), self.events)):
            with mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
                cold = self.H.qualification_checked(files); expected = copy.deepcopy(cold)
                warm = self.H.qualification_checked(dict(reversed(list(files.items()))))
                self.assertEqual(warm, expected); self.assertEqual(validate.call_count, 1)
                for key in ("request_controls", "config", "inputs", "source_hashes"):
                    self.assertIsNot(cold[key], warm[key]); cold[key].clear(); warm[key].clear()
                self.assertEqual(self.H.qualification_checked(files), expected)
                self.assertEqual(validate.call_count, 1)
        for key, value in self.H._qualification_memo.items():
            self.assertTrue(all(type(n) is str and type(size) is int and self.H.G.P._sha(digest) for n, size, digest in key))
            self.assertIs(type(value), bytes); self.assertLessEqual(len(value), 1024**2)

    def test_memo_rehashes_every_file_and_never_stores_rejections(self):
        files = self.pack(dict(self.job, cache_limit_bytes=0), self.events)
        self.H.qualification_checked(files)
        for name in files:
            changed = dict(files); changed[name] = files[name][:-2] + b"!\n"
            errors = []
            for _ in range(2):
                with self.subTest(name=name), self.assertRaises(ValueError) as raised:
                    self.H.qualification_checked(changed)
                errors.append(str(raised.exception))
            self.assertEqual(errors[0], errors[1]); self.assertEqual(len(self.H._qualification_memo), 1)
        with mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "Qualification protocol differs"):
                    self.H.qualification_checked({**files, "stdout.jsonl": files["stdout.jsonl"][:-2] + b"!\n"})
        self.assertEqual(validate.call_count, 2)
        class DerivedDict(dict): pass
        with mock.patch.object(self.H, "sha256", side_effect=AssertionError("Invalid input reached key")):
            for invalid in ([], DerivedDict(files), {}, {**files, "extra": b""}, {**files, "job.json": bytearray(files["job.json"])}):
                with self.assertRaisesRegex(ValueError, "Invalid qualification files"): self.H.qualification_checked(invalid)
            limits = dict(self.H.QUALIFICATION_LIMITS, **{"stdout.jsonl": len(files["stdout.jsonl"]) - 1})
            with mock.patch.object(self.H, "QUALIFICATION_LIMITS", limits), self.assertRaisesRegex(ValueError, "Invalid qualification files"):
                self.H.qualification_checked(files)

    def test_memo_canonical_names_do_not_order_or_retain_caller_key_objects(self):
        class Named(str):
            def __lt__(self, other): raise AssertionError("Caller filename ordering")
        names = [Named(name) for name in reversed(self.files)]
        for name in names: name.payload = object()
        files = {name: self.files[name] for name in names}
        with mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
            expected = self.H.qualification_checked(files)
            self.assertEqual(self.H.qualification_checked(self.files), expected)
            self.assertEqual(validate.call_count, 1)
            self.H._qualification_memo.clear()
            self.assertEqual(self.H.qualification_checked(self.files), expected)
            self.assertEqual(self.H.qualification_checked(files), expected)
        self.assertEqual(validate.call_count, 2)
        self.assertTrue(all(type(name) is str for key in self.H._qualification_memo for name, _, _ in key))
        self.assertEqual(len(self.H._qualification_memo), 1)

    def test_memo_keys_and_validates_the_same_detached_input(self):
        files = dict(self.files); require = self.H.G.require
        def mutate_before_copy(condition, message):
            require(condition, message)
            if files["stdout.jsonl"] is self.files["stdout.jsonl"]: files["stdout.jsonl"] = bytearray(files["stdout.jsonl"])
        with mock.patch.object(self.H.G, "require", side_effect=mutate_before_copy), self.assertRaisesRegex(ValueError, "Invalid qualification files"):
            self.H.qualification_checked(files)
        self.assertEqual(self.H._qualification_memo, {})
        files = dict(self.files); digest = self.H.sha256
        def mutate_during_hash(raw):
            files["job.json"] = b"changed caller mapping"
            return digest(raw)
        with mock.patch.object(self.H, "sha256", side_effect=mutate_during_hash):
            result = self.H.qualification_checked(files)
        self.assertEqual(result["inputs"]["job.json"]["sha256"], digest(self.files["job.json"]).hexdigest())
        self.assertEqual(self.H.qualification_checked(self.files), result)
        with self.assertRaises(ValueError): self.H.qualification_checked(files)

    def test_memo_fifo_bound_and_oversize_result_bypass(self):
        variants = [{**self.files, "response.json": b" " * i + self.files["response.json"]} for i in range(5)]
        with mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
            results = [self.H.qualification_checked(files) for files in variants]
            self.assertEqual((validate.call_count, len(self.H._qualification_memo)), (5, 4))
            self.assertEqual(self.H.qualification_checked(variants[-1]), results[-1])
            self.assertEqual(validate.call_count, 5)
            self.assertEqual(self.H.qualification_checked(variants[0]), results[0])
            self.assertEqual((validate.call_count, len(self.H._qualification_memo)), (6, 4))
        self.H._qualification_memo.clear(); self.assertEqual(self.H._QUALIFICATION_MEMO_BYTES, 1024**2)
        with mock.patch.object(self.H, "_QUALIFICATION_MEMO_BYTES", 1), mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
            for _ in range(2): self.assertEqual(self.H.qualification_checked(self.files), results[0])
        self.assertEqual(validate.call_count, 2); self.assertEqual(self.H._qualification_memo, {})

    def test_memo_concurrent_misses_keep_atomic_count_and_value_bounds(self):
        variants = [{**self.files, "response.json": b" " * i + self.files["response.json"]} for i in range(5)]
        barrier, original = self.H.threading.Barrier(5), self.H.validate_events
        def simultaneous(*args, **kwargs):
            barrier.wait(timeout=30)
            return original(*args, **kwargs)
        with mock.patch.object(self.H, "validate_events", side_effect=simultaneous) as validate, ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(self.H.qualification_checked, variants))
        self.assertEqual(validate.call_count, 5); self.assertEqual(len(self.H._qualification_memo), 4)
        self.assertTrue(all(type(raw) is bytes and len(raw) <= 1024**2 for raw in self.H._qualification_memo.values()))
        self.assertEqual([r["inputs"]["response.json"]["sha256"] for r in results], [self.H.sha256(f["response.json"]).hexdigest() for f in variants])

    def test_warm_memo_keeps_physical_reads_and_live_source_checks(self):
        expected = self.H.qualification_checked(self.files); directory = self.saved(self.files)
        with mock.patch.object(self.H, "_qualification_file", wraps=self.H._qualification_file) as read:
            self.assertEqual(self.H.read_qualification(directory, self.pin), (expected, self.files))
        self.assertEqual(read.call_count, 3)
        (directory / "stdout.jsonl").write_bytes(self.files["stdout.jsonl"][:-2] + b"!\n")
        with self.assertRaises(ValueError): self.H.read_qualification(directory, self.pin)
        (directory / "stdout.jsonl").write_bytes(self.files["stdout.jsonl"])
        (directory / "response.json").write_bytes(self.files["response.json"] + b" ")
        with self.assertRaisesRegex(ValueError, "Qualification authority differs"): self.H.read_qualification(directory, self.pin)
        original, source = Path.read_bytes, next(iter(expected["source_hashes"]))
        with mock.patch.object(Path, "read_bytes", lambda path: b"changed" if str(path) == source else original(path)), \
             self.assertRaisesRegex(ValueError, "Qualified source changed"):
            self.H.current_source_compatible(self.H.qualification_checked(self.files))

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
        self.H.qualification_checked(self.files)  # Physical guards must also reject with a warm pure memo.
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
