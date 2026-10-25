import copy
import unittest
from tests import test_long_batch_qualification as long_fixture, test_qualification_admission as admission_fixture


class IsolationQualificationTests(unittest.TestCase):
    def setUp(self):
        self.f = long_fixture.LongBatchQualificationTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.H, self.M = self.f.H, self.f.M

    def protocol(self):
        job = self.M.build_isolation_job(self.f.f.d, dict(self.f.config, batch_size=2, prefill_batch_size=2))
        self.f.f.windows, self.f.f.fixtures = job["windows"], job["fixtures"]
        events = [e for e in self.f.f.rows() if e["event"] != "qualification"]
        ready = events[2]["fixtures"]
        for fixture, count, token in zip(ready, (40, 7168, 7168), (3, 3, 4)): fixture["prompt_ids"] = [token] * count
        plans = {w["id"]: w for w in job["windows"]}
        for event in events:
            if "t_seconds" in event: event["t_seconds"] *= .001
            if event["event"] == "window_done": event["elapsed"] *= .001
            if event["event"] == "token":
                row = plans[event["window"]]["rows"][event["uid"]]; terminal = event["seq"] == row["cap"]
                event["token"] = 9 if terminal and row["sampling"] == "eos" else 3
                event["finish_reason"] = ("stop" if row["sampling"] == "eos" else "length") if terminal else None
        for window in job["windows"]:
            name = window["id"]
            if name in ("q0", "q1", "q3", "q4", "qlong"): continue
            start = next(i for i, e in enumerate(events) if e.get("id") == name)
            end = next(i for i, e in enumerate(events) if e["event"] == "window_done" and e["window"] == name)
            body = events[start + 1:end]; ordered = [dict(e, t_seconds=0) for e in body if e["event"] == "admit"]
            for event in (e for e in body if e["event"] == "token"):
                if event["uid"] == 0:
                    ordered.append(dict(event="score", window=name, row_index=0, seq=event["seq"], native_dtype="mlx.core.float32",
                        shape=[1, 16], finite=True, returned_token=event["token"], observed_f32_sha256="a" * 64))
                ordered.append(event)
            events[start + 1:end] = ordered
        self.f.reinsert_chronology(events)
        stress = dict(batch_size=2, prompt_tokens_min=7168, prompt_tokens_max=7168, prompt_token_ceiling=7168, max_tokens=2048, raw_generation_tokens=4096)
        draws = {name: {"17": list(range(8)), "29": list(range(8, 16))} for name in ("serial", "interleaved", "reversed")}
        events.insert(-1, dict(event="qualification", rng_draws=draws, diagnostics=dict(greedy_equal=True, reinsert_equal=True), stress=stress))
        return job, events

    def parse(self, job, events):
        events = copy.deepcopy(events); identity = self.H.sha256(self.H._raw(job)).hexdigest()
        for event in events:
            event["job_sha256"] = identity
            if event["event"] == "controls_checked": event["request_sha256"] = identity
        return self.f.parse(job, events)

    def test_fixed_recipe_and_workload(self):
        for batch in (2, 4, 8):
            job = self.M.build_isolation_job(self.f.f.d, dict(self.f.config, batch_size=batch, prefill_batch_size=min(4, batch)))
            self.assertEqual((job["schema_version"], job["request"]["max_tokens"], job["wall_seconds"]), (4, 2048, 240))
            self.assertEqual((sum(len(w["rows"]) for w in job["windows"]), sum(r["cap"] for w in job["windows"] for r in w["rows"])), (7 * batch + 6, 2144 * batch + 43))
            self.assertEqual(job["fixtures"][1]["messages"][2]["content"].count("    # Input"), 260)
        job, events = self.protocol()
        for index, field, value in ((2, "fixture_id", "long-peer"), (2, "seed", 1), (2, "cap", 15), (3, "sampling", "sample")):
            bad = copy.deepcopy(job); bad["windows"][index]["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): self.H._shape(bad)
        bad = copy.deepcopy(job); bad["fixtures"][2]["messages"] = copy.deepcopy(bad["fixtures"][1]["messages"])
        with self.assertRaises(ValueError): self.H._shape(bad)

    def test_complete_lookahead_and_retained_partial_packets(self):
        job, events = self.protocol(); parsed = self.parse(job, events)
        self.assertTrue(parsed["complete"], parsed["protocol_error"]); self.assertEqual(len(parsed["score_packets"]), 96)
        for name in {e["window"] for e in events if e["event"] == "score"}:
            packet = next(e for e in reversed(events) if e["event"] == "score" and e["window"] == name)
            end = next(i for i, e in enumerate(events) if e["event"] == "window_done" and e["window"] == name)
            events.insert(end, dict(packet, seq=17))
        self.assertTrue(self.parse(job, events)["complete"]); self.assertEqual(len(self.parse(job, events)["score_packets"]), 102)
        boundary = next(i for i, e in enumerate(events) if e.get("id") == "qlong")
        partial = self.parse(job, events[:boundary]); self.assertFalse(partial["complete"])
        self.assertEqual(len(partial["score_packets"]), 102); self.assertIsNone(partial["long_stress"])
        next(e for e in events if e["event"] == "score" and e["window"] == "long-A-replay" and e["seq"] == 17)["observed_f32_sha256"] = "b" * 64
        self.assertFalse(self.parse(job, events)["complete"])

    def test_packet_fields_coverage_binding_pair_and_replay(self):
        job, events = self.protocol()
        changes = [("row_index", True), ("seq", True), ("seq", 2), ("seq", 18), ("returned_token", True), ("returned_token", 4),
            ("native_dtype", "float32"), ("native_dtype", "mlx.core.float16"), ("shape", [True, 16]), ("shape", [1, True]),
            ("shape", [1, 0]), ("shape", [1, 1048577]), ("shape", [1, 17]), ("shape", [1, 16, 1]), ("finite", False),
            ("finite", 1), ("observed_f32_sha256", "b" * 64), ("observed_f32_sha256", "bad"), ("window", "q0"), ("extra", 1)]
        for field, value in changes:
            bad = copy.deepcopy(events); next(e for e in bad if e["event"] == "score" and e["window"] == "short-B")[field] = value
            with self.subTest(field=field, value=value): self.assertFalse(self.parse(job, bad)["complete"])
        for fault in ("missing", "duplicate", "extra", "replay"):
            bad = copy.deepcopy(events); packet = next(e for e in reversed(bad) if e["event"] == "score" and e["window"] == "long-A-replay")
            if fault == "missing": bad.remove(packet)
            if fault == "duplicate": bad.insert(bad.index(packet), dict(packet))
            if fault == "extra": bad.insert(bad.index(packet), dict(packet, seq=17))
            if fault == "replay": packet["observed_f32_sha256"] = "b" * 64
            with self.subTest(fault=fault): self.assertFalse(self.parse(job, bad)["complete"])

    def test_peer_intervention_schedule_live_control_and_stress(self):
        job, events = self.protocol()
        for fault in ("contents", "count", "order", "late", "peer", "prefix", "vocab", "live", "stress", "peer_length", "stress_row"):
            bad = copy.deepcopy(events); admits = [e for e in bad if e["event"] == "admit" and e["window"] == "short-B"]
            if fault == "contents": bad[2]["fixtures"][2]["prompt_ids"] = list(bad[2]["fixtures"][1]["prompt_ids"])
            if fault == "count": bad[2]["fixtures"][2]["prompt_ids"].pop()
            if fault == "order": a, b = map(bad.index, admits); bad[a], bad[b] = bad[b], bad[a]
            if fault == "late": bad.remove(admits[1]); bad.insert(next(i for i, e in enumerate(bad) if e["event"] == "score" and e["window"] == "short-B") + 1, admits[1])
            if fault in ("peer", "prefix"):
                token = next(e for e in bad if e["event"] == "token" and e["window"] == ("q3" if fault == "prefix" else "short-B") and e["uid"] == (0 if fault == "prefix" else 1)); token["token"] = 4
            if fault == "vocab":
                for event in (e for e in bad if e["event"] == "token"):
                    row = next(w for w in job["windows"] if w["id"] == event["window"])["rows"][event["uid"]]
                    if row["sampling"] == "length" or row["sampling"] == "eos" and event["finish_reason"] is None: event["token"] = 16
            if fault == "live": self.f.reinsert_chronology(bad, empty_peer=True)
            if fault == "stress": next(e for e in bad if e["event"] == "qualification")["stress"]["raw_generation_tokens"] -= 1
            if fault in ("peer_length", "stress_row"):
                tokens = [e for e in bad if e["event"] == "token" and e["window"] == ("short-B" if fault == "peer_length" else "qlong") and e["uid"] == 1]
                bad.remove(tokens[-1]); tokens[-2].update(token=9, finish_reason="stop")
                next(e for e in bad if e["event"] == "window_done" and e["window"] == tokens[0]["window"])["stats"]["generation_tokens"] -= 1
            with self.subTest(fault=fault): self.assertFalse(self.parse(job, bad)["complete"])

    def test_inclusive_stress_context_bounds(self):
        job, events = self.protocol()
        for length in (7167, 7168, 8192, 8193):
            bad = copy.deepcopy(events)
            for fixture, token in zip(bad[2]["fixtures"][1:], (3, 4)): fixture["prompt_ids"] = [token] * length
            next(e for e in bad if e["event"] == "qualification")["stress"].update(prompt_tokens_min=length, prompt_tokens_max=length, prompt_token_ceiling=length)
            with self.subTest(length=length): self.assertEqual(self.parse(job, bad)["complete"], 7168 <= length <= 8192)

    def test_raw_rng_shape_range_and_stream_relations(self):
        job, events = self.protocol()
        for fault in ("order", "seed", "length", "bool", "range", "constant", "same", "reordered"):
            bad = copy.deepcopy(events); draws = next(e for e in bad if e["event"] == "qualification")["rng_draws"]
            if fault == "order": draws["extra"] = draws["serial"]
            if fault == "seed": draws["serial"]["99"] = draws["serial"].pop("29")
            if fault == "length": draws["serial"]["17"].pop()
            if fault in ("bool", "range", "reordered"): draws["reversed"]["17"][0] = {"bool": True, "range": 16, "reordered": 7}[fault]
            if fault in ("constant", "same"):
                for value in draws.values(): value["17"] = [1] * 8 if fault == "constant" else list(value["29"])
            with self.subTest(fault=fault): self.assertFalse(self.parse(job, bad)["complete"])

    def test_early_eos_and_informational_diagnostics(self):
        job, events = self.protocol(); targets = {e["window"] for e in events if e["event"] == "score"}
        for name in targets:
            events = [e for e in events if not (e.get("window") == name and e["event"] in ("score", "token") and e.get("uid", 0) == 0 and e["seq"] > 3)]
            for event in events:
                if event.get("window") == name and event["event"] in ("score", "token") and event.get("uid", 0) == 0 and event["seq"] == 3:
                    event["returned_token" if event["event"] == "score" else "token"] = 9
                    if event["event"] == "token": event["finish_reason"] = "stop"
            next(e for e in events if e["event"] == "window_done" and e["window"] == name)["stats"]["generation_tokens"] -= 13
        next(e for e in events if e["event"] == "qualification")["diagnostics"]["greedy_equal"] = False
        parsed = self.parse(job, events); self.assertTrue(parsed["complete"], parsed["protocol_error"])
        self.assertFalse(parsed["diagnostics"]["greedy_equal"])
        extra = copy.deepcopy(events); end = next(i for i, e in enumerate(extra) if e["event"] == "window_done" and e["window"] == "short-B")
        packet = next(e for e in reversed(extra[:end]) if e["event"] == "score" and e["window"] == "short-B")
        extra[end:end] = [dict(packet, seq=i) for i in (4, 5)]
        self.assertFalse(self.parse(job, extra)["complete"])
        bad = copy.deepcopy(events); next(e for e in bad if e["event"] == "score" and e["window"] == "short-B")["observed_f32_sha256"] = "b" * 64
        self.assertFalse(self.parse(job, bad)["complete"])

    def test_legacy_projection_false_gate_and_no_v4_admission(self):
        job, events = self.f.protocol(); parsed = self.f.parse(job, events)
        self.assertEqual(set(parsed), set("events protocol_error complete observations windows observed_generation_tokens finished_sequences long_stress".split()))
        legacy = long_fixture.fixture.BatchThroughputTests(methodName="runTest"); legacy.setUp(); self.addCleanup(legacy.doCleanups)
        self.assertEqual(set(legacy.parse(legacy.rows())), set(parsed) - {"long_stress"})
        next(e for e in events if e["event"] == "qualification")["checks"]["greedy_equal"] = False
        self.assertFalse(self.f.parse(job, events)["complete"])
        job, events = self.protocol()
        with self.assertRaises(ValueError): self.H.qualification_checked(admission_fixture.QualificationAdmissionTests.pack(self, job, events))

    def test_v4_exit_between_polls_still_exceeds_deadline(self):
        self.f.protocol = self.protocol
        self.f.test_complete_job_that_exits_between_polls_still_exceeds_cold_deadline()
