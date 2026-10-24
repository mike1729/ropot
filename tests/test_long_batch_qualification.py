import copy
import importlib
from time import monotonic, sleep
import unittest
from unittest import mock
from tests import test_batch_throughput as fixture


class LongBatchQualificationTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.BatchThroughputTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.H = self.f.B; self.M = importlib.import_module("experiments.mlx_batch_throughput")
        self.config = dict(batch_size=8, prefill_batch_size=4, prefill_step_size=256)

    def test_legacy_default_bytes_shape_parser_and_deadline_are_preserved(self):
        job = self.M.build_job(self.f.d, [self.config])
        explicit = self.H.build_plan({k: v for k, v in job["request"].items() if k != "environment"}, job["fixtures"], job["windows"], worker=self.M.__file__,
            wall_seconds=job["wall_seconds"], schema_version=1)
        self.assertEqual(self.H._raw(job), self.H._raw(explicit)); self.assertEqual(job["schema_version"], 1)
        self.assertEqual([w["id"] for w in job["windows"]], ["q0", "q1", "q2", "q3", "q4", "warm", "m0-r0", "m0-r1"])
        self.assertEqual((job["request"]["max_tokens"], job["wall_seconds"]), (128, 1170))
        self.assertNotIn("long_stress", self.f.parse(self.f.rows()))
        self.assertEqual(self.M.run_window.__kwdefaults__["window_seconds"], 180)
        invalid = copy.deepcopy(job); invalid["windows"][-1]["rows"][0]["cap"] = 2048
        with self.assertRaises(ValueError): self.H._shape(invalid)

    def test_long_shape_config_stress_caps_costs_and_live_reinsert_recipe(self):
        totals = [0, 0]
        for batch in (8, 4, 2):
            config = dict(self.config, batch_size=batch, prefill_batch_size=min(4, batch))
            job = self.M.build_long_job(self.f.d, config)
            self.assertEqual((job["schema_version"], job["request"]["max_tokens"], job["wall_seconds"]), (2, 2048, 240))
            self.assertEqual([len(f["messages"]) for f in job["fixtures"]], [2, 4])
            self.assertEqual([w["id"] for w in job["windows"]], ["q0", "q1", "q2", "q3", "q4", "qlong"])
            self.assertTrue(all(w["mode"] == "qualification" for w in job["windows"]))
            self.assertEqual(job["windows"][2]["config"], config); self.assertEqual(job["windows"][-1]["config"], config)
            for w in (job["windows"][0], job["windows"][1], job["windows"][4]):
                self.assertEqual(w["config"], dict(config, batch_size=1, prefill_batch_size=1))
            q3 = job["windows"][3]["rows"]
            self.assertEqual([(r["cap"], r["sampling"], r["after_row_index"]) for r in q3], [(2, "eos", None), (3, "length", None), (3, "greedy", 0)])
            stress = job["windows"][-1]["rows"]; self.assertEqual(len(stress), batch)
            self.assertTrue(all(r["cap"] == 2048 and r["fixture_id"] == job["fixtures"][1]["id"] for r in stress))
            count = sum(len(w["rows"]) for w in job["windows"]); caps = sum(r["cap"] for w in job["windows"] for r in w["rows"])
            self.assertEqual((count, caps), (2 * batch + 6, 43 + 2064 * batch)); totals[0] += count; totals[1] += caps
        self.assertEqual([4 * n for n in totals], [184, 116100])

    def test_method_only_shape_rejects_interior_cap_config_and_recipe_changes(self):
        job = self.M.build_long_job(self.f.d, dict(self.config, prefill_batch_size=2, prefill_step_size=2048))
        for fault in ("version", "config", "cap", "sampling", "context", "dependency", "extra"):
            bad = copy.deepcopy(job)
            if fault == "version": bad["schema_version"] = True
            if fault == "config": bad["windows"][2]["config"]["prefill_step_size"] = 256
            if fault == "cap": bad["windows"][-1]["rows"][4]["cap"] = 2047
            if fault == "sampling": bad["windows"][-1]["rows"][4]["sampling"] = "sample"
            if fault == "context": bad["windows"][-1]["rows"][4]["fixture_id"] = bad["fixtures"][0]["id"]
            if fault == "dependency": bad["windows"][3]["rows"][2]["after_row_index"] = None
            if fault == "extra": bad["windows"].append(dict(bad["windows"][0], id="warm", mode="warmup"))
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.H._shape(bad)

    def completed(self, job):
        return [dict(tokens=[3] * 2047 + [9 if r["sampling"] == "eos" else 3], text="code",
                     finish_reason="stop" if r["sampling"] == "eos" else "length") for r in job["windows"][-1]["rows"]]

    def reinsert_chronology(self, events, empty_peer=False):
        q3 = [e for e in events if e.get("window") == "q3" and e["event"] in ("admit", "token")]
        lookup = {(e["event"], e["uid"], e.get("seq", 0)): e for e in q3}
        order = [("admit", 0, 0), ("admit", 1, 0), ("token", 0, 1), ("token", 1, 1), ("token", 0, 2), ("token", 1, 2),
                 ("admit", 2, 0), ("token", 1, 3)] + [("token", 2, seq) for seq in (1, 2, 3)]
        if empty_peer: order[6], order[7] = order[7], order[6]
        start = events.index(q3[0]); reordered = [lookup[key] for key in order]
        for index, event in enumerate(reordered): event["t_seconds"] = index * .0001
        events[start:start + len(q3)] = reordered
        next(e for e in events if e["event"] == "window_done" and e["window"] == "q3")["elapsed"] = len(q3) * .0001

    def protocol(self):
        job = self.M.build_long_job(self.f.d, self.config)
        self.f.windows, self.f.fixtures = job["windows"], job["fixtures"]
        events = [e for e in self.f.rows() if e["event"] != "qualification"]
        ready = events[2]["fixtures"]
        for i, f in enumerate(ready): f["prompt_ids"] = [3] * (40 if i == 0 else 3000)
        plans = {w["id"]: w for w in job["windows"]}
        for event in events:
            if "t_seconds" in event: event["t_seconds"] *= .001
            if event["event"] == "window_done": event["elapsed"] *= .001
            if event["event"] == "token":
                row = plans[event["window"]]["rows"][event["uid"]]; terminal = event["seq"] == row["cap"]
                event["token"] = 9 if terminal and row["sampling"] == "eos" else 3
                event["finish_reason"] = ("stop" if row["sampling"] == "eos" else "length") if terminal else None
        self.reinsert_chronology(events)
        proof = self.M.long_stress(job, ready, self.completed(job))
        events.insert(-1, dict(event="qualification", checks=dict.fromkeys(self.H.CHECKS, True), stress=proof))
        identity = self.H.sha256(self.H._raw(job)).hexdigest()
        for event in events:
            event["job_sha256"] = identity
            if event["event"] == "controls_checked": event["request_sha256"] = identity
        return job, events

    def parse(self, job, events):
        return self.H.validate_events(b"".join(self.H._raw(e) for e in events), job, pid=7)

    def test_parser_requires_delayed_full_stress_and_independently_checks_proof(self):
        job, events = self.protocol(); good = self.parse(job, events)
        self.assertTrue(good["complete"], good["protocol_error"]); self.assertEqual(good["long_stress"]["prompt_token_ceiling"], 3000)
        boundary = next(i for i, e in enumerate(events) if e["event"] == "window_start" and e["id"] == "qlong")
        self.assertIsNone(self.parse(job, events[:boundary])["long_stress"])
        for fault in ("early", "ceiling", "count", "short", "empty_peer"):
            bad = copy.deepcopy(events); qual = next(e for e in bad if e["event"] == "qualification")
            if fault == "early": bad.remove(qual); bad.insert(boundary, qual)
            if fault == "ceiling": qual["stress"]["prompt_token_ceiling"] = 3001
            if fault == "count": qual["stress"]["raw_generation_tokens"] -= 1
            if fault == "empty_peer": self.reinsert_chronology(bad, empty_peer=True)
            if fault == "short":
                bad = [e for e in bad if not (e["event"] == "token" and e["window"] == "qlong" and e["uid"] == 3 and e["seq"] == 2048)]
                next(e for e in bad if e["event"] == "token" and e["window"] == "qlong" and e["uid"] == 3 and e["seq"] == 2047)["finish_reason"] = "length"
            with self.subTest(fault=fault):
                parsed = self.parse(job, bad); self.assertFalse(parsed["complete"]); self.assertIsNone(parsed["long_stress"])
                if fault == "empty_peer": self.assertEqual(parsed["protocol_error"], "Reinsert has no live peer")
        for length in (2826, 4097):
            bad = copy.deepcopy(events); bad[2]["fixtures"][1]["prompt_ids"] = [3] * length
            next(e for e in bad if e["event"] == "qualification")["stress"].update(prompt_tokens_min=length, prompt_tokens_max=length, prompt_token_ceiling=length)
            with self.subTest(prompt_length=length): self.assertFalse(self.parse(job, bad)["complete"])

    def test_stress_checks_every_complete_row_and_only_stressed_prompt_bound(self):
        job = self.M.build_long_job(self.f.d, self.config); completed = self.completed(job)
        ready = [dict(id=f["id"], prompt="rendered", prompt_ids=[3] * length) for f, length in zip(job["fixtures"], (4000, 3000))]
        proof = self.M.long_stress(job, ready, completed)
        self.assertEqual(proof, dict(batch_size=8, prompt_tokens_min=3000, prompt_tokens_max=3000,
            prompt_token_ceiling=3000, max_tokens=2048, raw_generation_tokens=8 * 2048))
        for index in range(8):
            for fault in ("short", "finish"):
                bad = copy.deepcopy(completed)
                if fault == "short": bad[index]["tokens"].pop()
                else: bad[index]["finish_reason"] = "length" if index % 2 == 0 else "stop"
                with self.subTest(index=index, fault=fault), self.assertRaises(ValueError): self.M.long_stress(job, ready, bad)

    def test_complete_job_that_exits_between_polls_still_exceeds_cold_deadline(self):
        job, events = self.protocol(); saved = self.f.root / "long-events.json"; saved.write_bytes(self.H._raw(events))
        worker = self.f.root / "long-fake.py"
        worker.write_text("import hashlib,json,os,sys\nfrom pathlib import Path\n"
            "raw=Path(sys.argv[1]).read_bytes(); identity=hashlib.sha256(raw).hexdigest()\n"
            + "events=json.loads(Path(" + repr(str(saved)) + ").read_bytes())\n"
            "for event in events:\n event['job_sha256']=identity\n"
            " if event['event']=='start': event['pid']=os.getpid()\n"
            " if event['event']=='controls_checked': event['request_sha256']=identity\n"
            " print(json.dumps(event),flush=True)\n")
        job["source_hashes"].pop(job["worker"]); job["worker"] = str(worker)
        job["source_hashes"][str(worker)] = self.H.sha256(worker.read_bytes()).hexdigest()
        original_exited = self.H._exited
        def exited_before_first_poll(process):
            deadline = monotonic() + 5
            while monotonic() < deadline:
                if original_exited(process): return True
                sleep(.01)
            self.fail("Owned fake worker did not finish")
        with mock.patch.object(self.H, "_exited", side_effect=exited_before_first_poll) as exited, \
             mock.patch.object(self.H, "monotonic", side_effect=[0, 241, 242, 243, 244]):
            result = self.H.run(job, self.f.root / "late-complete", self.f.lock)
        exited.assert_called_once(); self.assertTrue(result["complete"], result["protocol_error"])
        self.assertEqual((result["status"], result["work_seconds"], result["returncode"]), ("timeout", 241, 0))
        self.assertTrue(result["group_absent"]); self.assertIsNone(result["cleanup_error"])
