import copy
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_qualification_admission as qualification_fixture


class ProgramWaveTests(unittest.TestCase):
    def setUp(self):
        self.q = qualification_fixture.QualificationAdmissionTests(methodName="runTest"); self.q.setUp(); self.addCleanup(self.q.doCleanups)
        self.H, self.M, self.root = self.q.H, self.q.f.M, self.q.root
        self.files = self.q.pack(dict(self.q.job, cache_limit_bytes=0), self.q.events)
        self.admission = self.H.qualification_checked(self.files)

    def rows(self):
        request = {k: copy.deepcopy(v) for k, v in self.q.job["request"].items() if k != "environment"}
        return [dict(row_id=name, request=copy.deepcopy(request)) for name in ("a", "b")]

    def protocol(self):
        job = self.M.build_wave(self.rows(), self.admission); identity = self.H.sha256(self.H._raw(job)).hexdigest()
        events = copy.deepcopy(self.q.events[:3])
        events[2]["fixtures"] = [dict(id=f["id"], prompt="rendered", prompt_ids=[3] * (30 + i)) for i, f in enumerate(job["fixtures"])]
        events += [dict(event="window_start", id="wave", started_seconds=1)]
        events += [dict(event="admit", window="wave", uid=uid, row_index=i, t_seconds=i * .001) for i, uid in enumerate((7, 11))]
        for uid, seq, token, reason, time in ((7, 1, 3, None, .01), (7, 2, 9, "stop", .02), (11, 1, 9, "stop", .03)):
            events.append(dict(event="token", window="wave", uid=uid, seq=seq, token=token, finish_reason=reason, t_seconds=time))
        events += [dict(event="window_done", window="wave", elapsed=.1, stats=dict(generation_tokens=3),
            memory=dict(active_bytes=1, cache_bytes=0, peak_bytes=1)), dict(event="done")]
        decoded = dict(job_sha256=identity, window="wave", rows=[dict(tokens=tokens, text=text, finish_reason="stop")
            for tokens, text in (([3, 9], "print(1)"), ([9], ""))])
        raw = self.H._raw(decoded); events[-1]["decoded_sha256"] = self.H.sha256(raw).hexdigest()
        files = self.q.pack(job, events); events = [json.loads(line) for line in files["stdout.jsonl"].splitlines()]
        return job, events, files, raw

    def project(self, job, files, decoded, qualification=None):
        return self.M.project_wave(job, files["stdout.jsonl"], files["response.json"], decoded,
            qualification_files=self.files if qualification is None else qualification)

    def test_raw_normalization_duplicate_sha_and_short_wave_keep_qualified_controls(self):
        rows = self.rows()
        with mock.patch.object(self.H.G, "prepare_request", wraps=self.H.G.prepare_request) as prepare:
            job = self.M.build_wave(rows, self.admission, prompt_ceiling_tokens=2999)
        self.assertEqual(prepare.call_count, 2); self.assertEqual(job["schema_version"], 3)
        self.assertEqual(job["rows"][0]["request_sha256"], job["rows"][1]["request_sha256"])
        for row, original in zip(job["rows"], rows):
            raw = self.H.G.prepare_request(original["request"])[1]
            self.assertEqual((row["row_id"], row["request_text"].encode(), row["request_sha256"]), (original["row_id"], raw, self.H.sha256(raw).hexdigest()))
        self.assertEqual((job["prompt_ceiling_tokens"], job["cache_limit_bytes"], job["wall_seconds"]), (2999, 0, 240))
        short = self.M.build_wave(rows[:1], self.admission)
        self.assertEqual(short["windows"][0]["config"], self.admission["config"])
        self.assertEqual([(w["id"], w["mode"], len(w["rows"])) for w in short["windows"]], [("wave", "generate", 1)])
        varied = self.rows(); varied[-1]["request"]["seed"] += 1; varied[-1]["request"]["messages"][-1]["content"] += "\nDifferent invented request."
        varied = self.M.build_wave(varied, self.admission)
        self.assertNotEqual(varied["rows"][0]["request_sha256"], varied["rows"][1]["request_sha256"])
        for ceiling in (True, 0, 2999.0, 3001):
            with self.assertRaises(ValueError): self.M.build_wave(rows, self.admission, prompt_ceiling_tokens=ceiling)

    def test_last_request_controls_and_unique_ids_are_checked_before_dispatch(self):
        for fault in ("source", "profile", "sampling", "cap", "wall", "environment", "duplicate", "empty", "overflow"):
            rows = self.rows(); last = rows[-1]["request"]
            if fault == "source": last["precision_profile"]["descriptor"]["source"] = "synthetic"
            if fault == "profile": last["model_path"] = "/different-model"
            if fault == "sampling": last["top_p"] = .9
            if fault == "cap": last["max_tokens"] = 2047
            if fault == "wall": last["wall_time_seconds"] = 239
            if fault == "environment": last["environment"] = self.admission["request_controls"]["environment"]
            if fault == "duplicate": rows[-1]["row_id"] = rows[0]["row_id"]
            if fault == "empty": rows = []
            if fault == "overflow": rows = [dict(row_id=str(i), request=last) for i in range(9)]
            with self.subTest(fault=fault), mock.patch.object(self.H.subprocess, "Popen") as launch, self.assertRaises(ValueError):
                self.M.build_wave(rows, self.admission)
            launch.assert_not_called()

    def test_wave_shape_does_not_allow_new_config_caps_dependencies_or_noncanonical_rows(self):
        job = self.M.build_wave(self.rows(), self.admission)
        for fault in ("config", "cap", "dependency", "request", "sha", "extra"):
            bad = copy.deepcopy(job)
            if fault == "config": bad["windows"][0]["config"]["batch_size"] = 2
            if fault == "cap": bad["windows"][0]["rows"][-1]["cap"] = 2047
            if fault == "dependency": bad["windows"][0]["rows"][-1]["after_row_index"] = 0
            if fault == "request":
                bad["rows"][-1]["request_text"] += " "; bad["rows"][-1]["request_sha256"] = self.H.sha256(bad["rows"][-1]["request_text"].encode()).hexdigest()
            if fault == "sha": bad["rows"][-1]["request_sha256"] = "0" * 64
            if fault == "extra": bad["windows"].append(dict(bad["windows"][0], id="warm", mode="warmup"))
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.H._shape(bad)

    def test_all_ready_bounds_precede_admission_and_no_extra_qualification_is_accepted(self):
        job, events, _, _ = self.protocol()
        events[2]["fixtures"][-1]["prompt_ids"] = [3] * 3001
        parsed = self.H.validate_events(b"".join(self.H._raw(e) for e in events), job, pid=7)
        self.assertFalse(parsed["complete"]); self.assertEqual(parsed["observations"], [])
        job, events, _, _ = self.protocol(); events.insert(3, dict(event="qualification", job_sha256=events[0]["job_sha256"], checks=dict.fromkeys(self.H.CHECKS, True)))
        self.assertFalse(self.H.validate_events(b"".join(self.H._raw(e) for e in events), job, pid=7)["complete"])

    def test_projection_is_offline_and_binds_rows_eos_text_and_shared_evidence_once(self):
        job, _, files, decoded = self.protocol()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Historical read")), \
             mock.patch.object(Path, "resolve", side_effect=AssertionError("Historical resolve")), \
             mock.patch.object(Path, "stat", side_effect=AssertionError("Historical stat")):
            view = self.project(job, files, decoded)
        self.assertEqual([(r["row_id"], r["uid"], r["tokens"], r["text"], r["eligible"]) for r in view["rows"]],
            [("a", 7, [3, 9], "print(1)", True), ("b", 11, [9], "", True)])
        self.assertEqual([r["request_sha256"] for r in view["rows"]], [r["request_sha256"] for r in job["rows"]])
        self.assertTrue(all(not ({"pid", "returncode", "elapsed_seconds"} & set(r)) for r in view["rows"]))
        self.assertEqual((view["shared"]["pid"], view["shared"]["returncode"], view["shared"]["elapsed_seconds"]), (7, 0, 239.5))
        self.assertTrue(view["shared"]["resources_closed"]); self.assertIsNone(view["shared"]["cleanup_error"]); self.assertIsNone(view["shared"]["parent_error"])
        self.assertEqual((view["shared"]["profile_guarded"], view["shared"]["ready"], view["shared"]["protocol_error"]), (True, True, None))
        self.assertEqual(view["shared"]["evidence"]["decoded.jsonl"], dict(sha256=self.H.sha256(decoded).hexdigest(), bytes=len(decoded)))
        self.assertEqual(view["shared"]["evidence"]["qualification"], self.admission["inputs"])

    def test_failed_live_corrupt_or_unproven_session_never_has_eligible_rows(self):
        job, events, original, decoded = self.protocol()
        for fault in ("missing", "corrupt", "live", "exit", "pid", "parent", "uid", "guard", "decoded_missing", "decoded_extra", "window", "decoded_job", "tokens", "text", "text_only", "finish", "order", "qualification"):
            files = dict(original); raw = decoded; qual = self.files; response = json.loads(files["response.json"]); data = json.loads(decoded)
            if fault == "missing": files["response.json"] = b""
            if fault == "corrupt": files["response.json"] = b"{"
            if fault == "live": response["returncode"] = None
            if fault == "exit": response.update(status="worker_error", returncode=3)
            if fault == "pid": response["pid"] = 8
            if fault == "parent": response["observed_generation_tokens"] += 1
            if fault in ("uid", "guard"):
                bad = copy.deepcopy(events)
                if fault == "uid": next(e for e in bad if e["event"] == "token")["uid"] = 99
                else: bad[1]["proof"]["descriptor_sha256"] = "0" * 64
                files["stdout.jsonl"] = b"".join(self.H._raw(e) for e in bad)
            if fault == "decoded_missing": raw = b""
            if fault == "window": data["window"] = "other"
            if fault == "decoded_job": data["job_sha256"] = "0" * 64
            if fault == "tokens": data["rows"][0]["tokens"].pop()
            if fault == "text": data["rows"][0]["text"] = 17
            if fault == "text_only": data["rows"][0]["text"] = "tampered code"
            if fault == "finish": data["rows"][0]["finish_reason"] = "length"
            if fault == "order": data["rows"].reverse()
            if fault == "qualification": qual = {}
            if fault not in ("missing", "corrupt"): files["response.json"] = self.H._raw(response)
            if fault in ("tokens", "text", "text_only", "finish", "order", "window", "decoded_job", "decoded_extra"):
                raw = self.H._raw(data) * (2 if fault == "decoded_extra" else 1)
                if fault != "text_only":
                    changed = copy.deepcopy(events); changed[-1]["decoded_sha256"] = self.H.sha256(raw).hexdigest(); files = self.q.pack(job, changed)
            with self.subTest(fault=fault):
                view = self.project(job, files, raw, qual); self.assertEqual(len(view["rows"]), 2)
                self.assertTrue(all(r["eligible"] is False for r in view["rows"]))
                if fault == "qualification": self.assertEqual(view["rows"][0]["tokens"], [3, 9])
                if fault == "guard": self.assertEqual((view["shared"]["profile_guarded"], view["shared"]["ready"]), (False, False))
        for cut in (4, next(i for i, e in enumerate(events) if e["event"] == "token" and e["uid"] == 11)):
            files = self.q.pack(job, events[:cut]); response = json.loads(files["response.json"]); response.update(status="worker_error", returncode=3)
            files["response.json"] = self.H._raw(response); view = self.project(job, files, b"")
            self.assertTrue(all(r["eligible"] is False for r in view["rows"])); self.assertEqual(len(view["rows"]), 2)
            self.assertEqual((view["shared"]["profile_guarded"], view["shared"]["ready"], view["shared"]["protocol_error"]), (True, True, "Missing done"))
            if cut == 4: self.assertTrue(all(r["uid"] is None and r["tokens"] == [] for r in view["rows"]))
            else: self.assertEqual((view["rows"][0]["tokens"], view["rows"][0]["finish_reason"], view["rows"][1]["tokens"]), ([3, 9], "stop", []))

    def test_resource_closure_is_bound_and_independent_of_generation_eligibility(self):
        job, events, original, decoded = self.protocol()
        for fault, closed in (("timeout", True), ("partial", True), ("cancelled", True), ("job", False), ("pid", False), ("fields", False), ("cleanup", False), ("rc_bool", False)):
            files = self.q.pack(job, events[:-3]) if fault == "partial" else dict(original)
            response = json.loads(files["response.json"])
            if fault in ("timeout", "partial"): response.update(status="timeout", returncode=-9, work_seconds=241, elapsed_seconds=241.5)
            if fault == "cancelled": response.update(status="cancelled", returncode=-9, parent_error="KeyboardInterrupt()")
            if fault == "job": response["job_sha256"] = "0" * 64
            if fault == "pid": response["pid"] = 8
            if fault == "fields": response["observed_generation_tokens"] += 1
            if fault == "cleanup": response.update(status="worker_error", cleanup_error="Unproven group cleanup")
            if fault == "rc_bool": response["returncode"] = False
            files["response.json"] = self.H._raw(response); view = self.project(job, files, decoded)
            with self.subTest(fault=fault):
                self.assertIs(view["shared"]["resources_closed"], closed); self.assertTrue(all(r["eligible"] is False for r in view["rows"]))
                self.assertEqual(view["shared"]["cleanup_error"], response["cleanup_error"])
                self.assertEqual(view["shared"]["parent_error"], response["parent_error"])

    def test_parent_copies_authority_and_syncs_exact_requests_before_launch(self):
        job = self.M.build_wave(self.rows(), self.admission); output = self.root / "owned-wave"
        def launch(*args, **kwargs):
            self.assertEqual({p.name: p.read_bytes() for p in (output / "qualification").iterdir()}, self.files)
            self.assertEqual([(output / "requests" / f"{i}.json").read_bytes() for i in range(2)], [r["request_text"].encode() for r in job["rows"]])
            self.assertIn(mock.call(output / "requests"), sync.call_args_list); self.assertIn(mock.call(output), sync.call_args_list)
            raise OSError("Owned pre-dispatch fixture")
        with mock.patch.object(self.H, "_sync", wraps=self.H._sync) as sync, mock.patch.object(self.H.subprocess, "Popen", side_effect=launch) as spawn:
            with self.assertRaisesRegex(OSError, "Owned pre-dispatch fixture"):
                self.H.run(job, output, self.q.f.f.lock, qualification_files=self.files)
        result = json.loads((output / "response.json").read_bytes())
        spawn.assert_called_once(); self.assertEqual(result["status"], "parent_error")
