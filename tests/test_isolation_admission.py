import copy
import json
import unittest
from unittest import mock
from tests import test_isolation_qualification as isolation, test_qualification_admission as legacy, test_program_wave as wave_fixture


def pack_isolation(fixture, *, descriptor=None, config=None, prompt_tokens=7168, cache_limit_bytes=None):
    job, events = fixture.protocol(descriptor=descriptor, config=config, prompt_tokens=prompt_tokens)
    if cache_limit_bytes is not None: job["cache_limit_bytes"] = cache_limit_bytes
    return job, events, legacy.QualificationAdmissionTests.pack(fixture, job, events)


class IsolationAdmissionTests(unittest.TestCase):
    pack = legacy.QualificationAdmissionTests.pack
    saved = legacy.QualificationAdmissionTests.saved

    def setUp(self):
        self.a = isolation.IsolationQualificationTests(methodName="runTest"); self.a.setUp(); self.addCleanup(self.a.doCleanups)
        self.H, self.M, self.root = self.a.H, self.a.M, self.a.f.f.root
        self.H._qualification_memo.clear(); self.addCleanup(self.H._qualification_memo.clear)
        self.job, self.events, self.files = pack_isolation(self.a)
        self.pin = self.H.sha256(self.files["response.json"]).hexdigest()

    def expected(self, job, files, *, version, ceiling):
        result = dict(schema_version=version, request_controls={k: v for k, v in job["request"].items() if k not in ("messages", "seed")},
            config=job["windows"][-1]["config"], source_hashes=job["source_hashes"], prompt_ceiling_tokens=ceiling,
            stress_prompt_tokens=dict(min=ceiling, max=ceiling), inputs={name: dict(sha256=self.H.sha256(raw).hexdigest(), bytes=len(raw)) for name, raw in files.items()})
        if version == 2: result["contract"] = "peer-isolation-v2"
        if "cache_limit_bytes" in job: result["cache_limit_bytes"] = job["cache_limit_bytes"]
        return result

    def test_v2_memo_reuses_only_complete_named_qualification_computation(self):
        for cache in (None, 0):
            job, _, files = pack_isolation(self.a, cache_limit_bytes=cache)
            with mock.patch.object(self.H, "validate_events", wraps=self.H.validate_events) as validate:
                expected = self.expected(job, files, version=2, ceiling=7168)
                self.assertEqual(self.H.qualification_checked(files), expected)
                self.assertEqual(self.H.qualification_checked(dict(reversed(list(files.items())))), expected)
            self.assertEqual(validate.call_count, 1)
            for name in files:
                changed = dict(files); changed[name] = files[name][:-2] + b"!\n"
                with self.subTest(cache=cache, name=name), self.assertRaises(ValueError): self.H.qualification_checked(changed)

    def test_complete_authority_and_exact_legacy_projection(self):
        admission = self.H.qualification_checked(self.files)
        self.assertEqual(admission, self.expected(self.job, self.files, version=2, ceiling=7168))
        informative = copy.deepcopy(self.events); next(e for e in informative if e["event"] == "token" and e["window"] == "q0")["token"] = 4
        next(e for e in informative if e["event"] == "qualification")["diagnostics"]["greedy_equal"] = False
        self.assertEqual(self.H.qualification_checked(self.pack(self.job, informative))["schema_version"], 2)
        job, events = self.a.f.protocol(); files = self.pack(job, events)
        expected = self.expected(job, files, version=1, ceiling=3000)
        self.assertEqual(self.H._raw(self.H.qualification_checked(files)), self.H._raw(expected)); self.assertNotIn("contract", expected)
        descriptor = self.a.f.f.P.resolve(self.a.f.f.campaign, dict(path="precision/" + self.a.f.f.d["receipt_sha256"], receipt_sha256=self.a.f.f.d["receipt_sha256"]), "synthetic", 16)
        config = dict(batch_size=4, prefill_batch_size=2, prefill_step_size=1024)
        job, events, files = pack_isolation(self.a, descriptor=descriptor, config=config, prompt_tokens=7500, cache_limit_bytes=0)
        self.assertEqual(self.H.qualification_checked(files), self.expected(job, files, version=2, ceiling=7500))

    def test_parent_raw_and_incomplete_evidence_never_authorize(self):
        for fault in ("bytes", "raw", "partial", "status", "returncode", "bool_rc", "group", "cleanup", "parent", "pid", "projection", "extra", "work", "telemetry", "scores", "diagnostics"):
            files = dict(self.files); response = json.loads(files["response.json"])
            if fault == "bytes": files["job.json"] += b" "
            if fault == "raw": files["stdout.jsonl"] += b"{}\n"
            if fault == "partial":
                boundary = next(i for i, e in enumerate(self.events) if e.get("id") == "qlong")
                files["stdout.jsonl"] = self.pack(self.job, self.events[:boundary])["stdout.jsonl"]
            if fault == "status": response["status"] = "worker_error"
            if fault in ("returncode", "bool_rc"): response["returncode"] = 3 if fault == "returncode" else False
            if fault == "group": response["group_absent"] = False
            if fault == "cleanup": response["cleanup_error"] = "unproven"
            if fault == "parent": response["parent_error"] = "interrupted"
            if fault == "pid": response["pid"] = 8
            if fault == "projection": response["observed_generation_tokens"] += 1
            if fault == "extra": response["extra"] = 1
            if fault == "work": response.update(work_seconds=241, elapsed_seconds=241.5)
            if fault == "telemetry": response["telemetry_complete"] = False
            if fault in ("scores", "diagnostics"):
                events = copy.deepcopy(self.events)
                if fault == "scores": next(e for e in events if e["event"] == "score" and e["window"] == "long-A-replay")["observed_f32_sha256"] = "b" * 64
                else: next(e for e in events if e["event"] == "qualification")["diagnostics"]["greedy_equal"] = False
                files["stdout.jsonl"] = self.pack(self.job, events)["stdout.jsonl"]
            files["response.json"] = self.H._raw(response)
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "Qualification protocol differs" if fault in ("raw", "partial", "scores", "diagnostics") else "."):
                self.H.qualification_checked(files)

    def test_v2_reuses_exact_cache_read_copy_pin_source_and_file_bounds(self):
        methods = ("test_declared_zero_requires_exact_integer_readback_and_no_optional_file_drift",
            "test_trusted_response_pin_exact_copy_and_directory_sync_order", "test_offline_history_does_not_read_missing_or_symlink_source_assets",
            "test_reader_refuses_symlinks_nonregular_files_and_bounded_oversize")
        for method in methods:
            fixture = legacy.QualificationAdmissionTests(methodName="runTest"); fixture.setUp(); self.addCleanup(fixture.doCleanups)
            fixture.job, fixture.events, fixture.files, fixture.pin = self.job, self.events, self.files, self.pin
            with self.subTest(method=method): getattr(fixture, method)()

    def wave(self):
        fixture = wave_fixture.ProgramWaveTests(methodName="runTest")
        fixture.q, fixture.H, fixture.M, fixture.root = self, self.H, self.M, self.root
        fixture.files, fixture.admission = self.files, self.H.qualification_checked(self.files)
        return fixture

    def test_actual_wave_build_and_projection_at_qualified_minimum(self):
        fixture = self.wave(); job, events, _, decoded = fixture.protocol()
        self.assertEqual((job["schema_version"], job["prompt_ceiling_tokens"]), (3, 7168))
        for item in events[2]["fixtures"]: item["prompt_ids"] = [3] * 7168
        files = self.pack(job, events); view = fixture.project(job, files, decoded)
        self.assertTrue(view["shared"]["complete"], view["shared"]["errors"]); self.assertTrue(all(r["eligible"] for r in view["rows"]))
        self.assertEqual(view["shared"]["evidence"]["qualification"], fixture.admission["inputs"])
        for length in (7169, 8193):
            bad = copy.deepcopy(events); bad[2]["fixtures"][-1]["prompt_ids"] = [3] * length
            parsed = self.H.validate_events(b"".join(self.H._raw(e) for e in bad), job, pid=7)
            self.assertFalse(parsed["complete"]); self.assertEqual(parsed["observations"], [])
            with self.assertRaises(ValueError): self.M.build_wave(fixture.rows(), fixture.admission, prompt_ceiling_tokens=length)
        changed = dict(self.files); changed["stdout.jsonl"] += b"{}\n"
        self.assertFalse(fixture.project(job, files, decoded, qualification=changed)["shared"]["complete"])
        changed = copy.deepcopy(job); changed["admission"]["inputs"]["stdout.jsonl"]["sha256"] = "b" * 64
        output = json.loads(decoded); output["job_sha256"] = self.H.sha256(self.H._raw(changed)).hexdigest()
        changed_decoded = self.H._raw(output); changed_events = copy.deepcopy(events)
        changed_events[-1]["decoded_sha256"] = self.H.sha256(changed_decoded).hexdigest()
        view = fixture.project(changed, self.pack(changed, changed_events), changed_decoded)
        self.assertEqual(view["shared"]["errors"], ["Wave qualification authority differs"])

    def test_wave_exact_authority_versions_contracts_and_outer_ceiling(self):
        fixture = self.wave(); admission = fixture.admission
        for fault in ("missing", "wrong", "false", "extra", "bool", "unknown", "legacy_contract", "legacy_ceiling", "ceiling"):
            bad = copy.deepcopy(admission)
            if fault == "missing": bad.pop("contract")
            if fault in ("wrong", "false"): bad["contract"] = "other" if fault == "wrong" else False
            if fault == "extra": bad["extra"] = 1
            if fault in ("bool", "unknown"):
                bad["schema_version"] = True if fault == "bool" else 3
                bad.pop("contract"); bad["prompt_ceiling_tokens"] = 4096
            if fault in ("legacy_contract", "legacy_ceiling"):
                bad["schema_version"] = 1
                if fault == "legacy_ceiling": bad.pop("contract")
                else: bad["prompt_ceiling_tokens"] = 4096
            if fault == "ceiling": bad["prompt_ceiling_tokens"] = 8193
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.M.build_wave(fixture.rows(), bad)
