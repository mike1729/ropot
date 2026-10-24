import copy
import importlib
from pathlib import Path
import unittest
from unittest import mock
from tests import test_qualification_admission as qualification, test_study_tasks as study


class PairedWaveCollectionTests(unittest.TestCase):
    def setUp(self):
        self.q = qualification.QualificationAdmissionTests(methodName="runTest"); self.q.setUp(); self.addCleanup(self.q.doCleanups)
        self.f, self.H, self.root = self.q.f.f, self.q.H, self.q.root
        self.J, self.P = self.f.J, self.f.P; self.C = importlib.import_module("src.data-pipeline.paired_wave_collection"); self.PC = self.C.PC
        self.s = study.StudyTaskTests(methodName="runTest"); self.s.setUp(); self.addCleanup(self.s.doCleanups)
        self.loaded = self.s.feasibility_loaded(); self.index = 0
        self.original = Path(self.f.d["model_path"]).parents[1]; self.identity = self.f.d["receipt_sha256"]
        self.reference = dict(path="precision/" + self.identity, receipt_sha256=self.identity)
        self.profiles = {b: {s: dict(reference=self.reference, descriptor=self.P.resolve(self.f.campaign, self.reference, s, b)) for s in ("actor", "synthetic")} for b in (8, 16)}
        self.qualifications, self.methods = {8: {}, 16: {}}, {8: {}, 16: {}}
        for bits in (8, 16):
            for source in ("actor", "synthetic"):
                self.make_qualification(bits, source, 8 if source == "actor" else 4, (3100 if bits == 8 else 3000) if source == "actor" else (3200 if bits == 8 else 3050))
        for source in ("actor", "synthetic"):
            minimum = min(self.methods[b][source]["prompt_ceiling_tokens"] for b in (8, 16))
            for bits in (8, 16): self.methods[bits][source]["prompt_ceiling_tokens"] = minimum
        self.audit = self.root / "mechanics.txt"; self.audit.write_bytes(b"Fixture mechanics audit")
        patch = mock.patch.object(self.PC, "AUDIT_SHA256", self.C.sha256(self.audit.read_bytes()).hexdigest()); patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(self.C.V.runner, "contract", return_value=dict(image=dict(Id="fixture-image"))); patch.start(); self.addCleanup(patch.stop)
        for target, name in ((self.C.study, "load_task"), (self.C.L, "run"), (self.C.L, "prepare_cell"), (self.C.L, "settle_cell"), (self.C.L.native, "run_native"), (self.J.Campaign, "generate"), (self.J.Campaign, "generate_batch"), (self.H, "run")):
            patch = mock.patch.object(target, name, side_effect=AssertionError("Setup dispatched " + name)); patch.start(); self.addCleanup(patch.stop)

    def make_qualification(self, bits, source, batch, ceiling):
        descriptor = self.profiles[bits][source]["descriptor"]; proof = copy.deepcopy(self.f.proof)
        def roles(post=False):
            required = self.P.role_suffixes(source, post); sources = descriptor["expected_runtime"]["sources"]
            return self.P.bind_roles(sources, required, {role: next(p for p in sources if p.endswith(suffix or "/transformers/fixture.py")) for role, suffix in required.items()})
        proof.update(descriptor_sha256=self.P.digest(descriptor), model_files=descriptor["model_files"], critical_sources=roles())
        config = dict(batch_size=batch, prefill_batch_size=min(4, batch), prefill_step_size=256)
        with mock.patch.object(self.f, "d", descriptor), mock.patch.object(self.f, "proof", proof), mock.patch.object(self.f, "roles", side_effect=roles), mock.patch.object(self.q.f, "config", config):
            job, events = self.q.f.protocol()
        job["cache_limit_bytes"] = 0; events[2]["fixtures"][1]["prompt_ids"] = [3] * ceiling
        next(e for e in events if e["event"] == "qualification")["stress"].update(prompt_tokens_min=ceiling, prompt_tokens_max=ceiling, prompt_token_ceiling=ceiling)
        files = self.q.pack(job, events); pin = self.C.sha256(files["response.json"]).hexdigest(); directory = self.root / f"qualification-{bits}-{source}-{batch}-{ceiling}"
        self.H.copy_qualification(files, directory, pin); admission = self.H.qualification_checked(files)
        self.qualifications[bits][source] = dict(path=str(directory), response_sha256=pin)
        self.methods[bits][source] = dict(method="program-wave-v1", source=source, admission=admission, prompt_ceiling_tokens=ceiling,
            qualification=dict(path="qualifications/" + pin, response_sha256=pin))

    def fresh(self):
        self.index += 1; c = self.J.Campaign.create(self.root / f"wave-{self.index}", {s: dict(zip(self.J.UNITS, (1, 1, 1))) for s in ("actor", "synthetic")})
        self.addCleanup(c.close); return c

    def setup(self, c, name="trial", qualifications=None):
        return self.C.setup(c, self.loaded, name, self.original / "receipt.json", self.identity, self.audit,
            self.qualifications if qualifications is None else qualifications)

    def built(self, methods=None):
        return self.C.build(self.loaded, "trial", self.profiles, self.methods if methods is None else methods, self.C.V.digest(self.C.V.runner.contract()))

    def test_metadata_matrix_common_methods_and_canonical_opportunities(self):
        built = self.built(); records, plan = built["records"], built["plan"]; tasks = built["activation"]["tasks"]
        self.assertEqual((len(tasks), len(records), len(plan["opportunities"])), (72, 144, 168))
        self.assertEqual(built["panel"]["tasks"], tasks[:8] + tasks[8:29] + tasks[40:61])
        self.assertEqual(self.C._member_work([dict(kind="native_intent", seq=1)], built), [])
        by_name = {r["experiment_id"]: r for r in records}; members = {m["experiment_id"]: m for m in plan["records"]}
        for record in records:
            member = members[record["experiment_id"]]; self.assertEqual(record["schema_version"], 3)
            self.assertEqual(record["generation_method"], self.methods[member["bits"]]); self.assertEqual(record["assigned_task_ids"], [member["task"]["id"]])
            self.assertEqual(member["record_sha256"], self.C.digest(record)); self.assertEqual(record["recipes"]["actor"], self.PC.RECIPE)
        opportunities = plan["opportunities"]; self.assertEqual([o["id"] for o in opportunities], list(range(168)))
        expected_groups = [(block, p) for block, order in zip(("dev", "A_tune", "B_tune", "A_fresh", "B_fresh"), ((0, 1), (1, 0), (0, 1), (1, 0), (0, 1))) for p in order]
        for group, (block, parity) in enumerate(expected_groups):
            rows = [o for o in opportunities if o["group"] == group]
            self.assertEqual(rows[0]["bits"], 8 if parity == 0 else 16); self.assertEqual({o["block"] for o in rows}, {block})
            for opportunity in rows:
                self.assertEqual(set(opportunity), {"id", "block", "group", "bits", "source", "reply_index", "experiment_ids"})
                self.assertLessEqual(len(opportunity["experiment_ids"]), 8 if opportunity["source"] == "actor" else 4)
                ordinals = [tasks.index(next(t for t in tasks if t["id"] == by_name[name]["assigned_task_ids"][0])) for name in opportunity["experiment_ids"]]
                self.assertEqual(ordinals, sorted(ordinals)); self.assertTrue(all(i % 2 == parity for i in ordinals))
        for member in plan["records"]:
            owned = [o for o in opportunities if member["experiment_id"] in o["experiment_ids"]]
            self.assertEqual([(o["source"], o["reply_index"]) for o in owned], [(s, i) for s in ("actor", "synthetic") for i in range(3)])
        for bits in (8, 16): self.make_qualification(bits, "synthetic", 8, 3200 if bits == 8 else 3050)
        for bits in (8, 16): self.methods[bits]["synthetic"]["prompt_ceiling_tokens"] = 3050
        self.assertEqual(len(self.built()["plan"]["opportunities"]), 120)

    def test_invalid_last_common_config_cache_ceiling_profile_and_authority(self):
        for fault in ("config", "cache_value", "cache_absent", "ceiling", "profile", "last_tag"):
            methods = copy.deepcopy(self.methods); last = methods[16]["synthetic"]
            if fault == "config": last["admission"]["config"]["prefill_step_size"] *= 2
            if fault == "cache_value": last["admission"]["cache_limit_bytes"] = 4 * 1024**3
            if fault == "cache_absent": last["admission"].pop("cache_limit_bytes")
            if fault == "ceiling": last["prompt_ceiling_tokens"] += 1
            if fault == "profile": last["admission"]["request_controls"] = methods[8]["synthetic"]["admission"]["request_controls"]
            if fault == "last_tag": last["method"] = "other"
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.built(methods)

    def test_setup_copies_precede_full_records_funding_and_equal_retry_uses_owned_only(self):
        c = self.fresh(); append = c._append; copies_checked = False
        def checked(kind, **payload):
            nonlocal copies_checked
            if kind == "experiment" and not copies_checked:
                self.assertEqual((c.root / "audits" / (self.C.BASE_PROTOCOL_SHA256 + "-paired-protocol.md")).read_bytes(), self.PC.PROTOCOL.read_bytes())
                for bound in self.methods.values():
                    for method in bound.values(): self.H.read_qualification(c.root / method["qualification"]["path"], method["qualification"]["response_sha256"])
                copies_checked = True
            return append(kind, **payload)
        with mock.patch.object(c, "_append", side_effect=checked): event = self.setup(c)
        before = c.events; anchor = next(e for e in before if e["kind"] == "wave_setup")
        self.assertEqual([sum(e["kind"] == k for e in before) for k in ("panel", "mechanics_baseline", "experiment", "execution_context", "wave_setup", "wave_plan")], [1, 1, 144, 144, 1, 1])
        self.assertTrue(all(e["seq"] < anchor["seq"] for e in before if e["kind"] in ("experiment", "execution_context")))
        self.assertLess(anchor["seq"], next(e["seq"] for e in before if e["kind"] == "limits")); self.assertLess(anchor["seq"], event["seq"])
        for bits in (8, 16):
            for source in ("actor", "synthetic"):
                method = self.methods[bits][source]; original = Path(self.qualifications[bits][source]["path"])
                self.assertEqual({p.name: p.read_bytes() for p in (c.root / method["qualification"]["path"]).iterdir()}, {p.name: p.read_bytes() for p in original.iterdir()})
        self.assertEqual(c._accounting()[0], anchor["payload"]["funding"]["limits"])
        self.assertEqual(anchor["payload"]["funding"]["bound"], {s: dict(zip(self.J.UNITS, (432, 884736, 103680))) for s in ("actor", "synthetic")})
        with mock.patch.object(self.H, "read_qualification", wraps=self.H.read_qualification) as read:
            self.assertEqual(self.setup(c, qualifications={}), event)
        self.assertTrue(all(Path(call.args[0]).is_relative_to(c.root) for call in read.call_args_list)); self.assertEqual(c.events, before)
        c.close()
        args = ["--campaign", str(c.root), "--plan-id", "trial", "--receipt", "missing", "--receipt-sha256", self.identity, "--mechanics-audit", "missing", "--qualifications", "missing"]
        with mock.patch.object(self.C.study, "load_cohort", return_value=self.loaded), mock.patch("builtins.print") as output:
            self.C.main(args); self.assertEqual(output.call_args.args[0], self.J.encoded(event))

    def test_copy_registration_funding_and_plan_crash_retries_do_not_ratchet(self):
        for target, action in (("copy", None), ("append", "experiment"), ("append", "execution_context"), ("append", "wave_setup"), ("extend", None), ("append", "wave_plan")):
            for after in (False, True):
                c = self.fresh(); obj, name = (self.H, "copy_qualification") if target == "copy" else (c, "extend" if target == "extend" else "_append")
                original = getattr(obj, name)
                def crash(*args, **kwargs):
                    applies = target != "append" or args[0] == action
                    if applies and not after: raise OSError("before boundary")
                    result = original(*args, **kwargs)
                    if applies: raise OSError("after boundary")
                    return result
                with mock.patch.object(obj, name, side_effect=crash), self.assertRaises(OSError): self.setup(c)
                c.close(); reopened = self.J.Campaign(c.root); self.addCleanup(reopened.close); self.setup(reopened)
                self.assertEqual([sum(e["kind"] == k for e in reopened.events) for k in ("experiment", "execution_context", "wave_setup", "limits", "wave_plan")], [144, 144, 1, 1, 1])
                self.assertEqual(reopened._accounting()[0]["actor"], dict(zip(self.J.UNITS, (432, 884736, 103680))))

    def test_old_baseline_history_funding_and_cross_kind_supersession(self):
        c = self.fresh(); cold = self.PC.setup(c, self.loaded, "cold", self.original / "receipt.json", self.identity, self.audit)
        record = next(e["record"] for e in c.events if e["kind"] == "experiment"); request = self.f.G.request_for("actor", [{"role": "user", "content": "fixture"}], profile=record["profiles"]["actor"]["descriptor"],
            **{k: self.PC.RECIPE[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}, seed=20261024)
        _, raw = self.f.G.prepare_request(request); task = cold["payload"]["records"][0]["task"]
        old = c._append("generation_reserved", experiment_id=record["experiment_id"], source="actor", request_text=raw.decode(), request_sha256=self.C.sha256(raw).hexdigest(), attempt=dict(split=task["split"], id=task["id"], task_sha256=task["task_sha256"], reply_index=0))
        c.outcome(old["seq"]); prefix = c.events; limits, spent = c._accounting(); self.setup(c)
        self.assertEqual(c.events[:len(prefix)], prefix); self.assertEqual(sum(e["kind"] == "panel" for e in c.events), 1)
        funding = next(e["payload"]["funding"] for e in c.events if e["kind"] == "wave_setup")
        self.assertEqual(funding["reserved_before"], spent); self.assertEqual(funding["limits"]["actor"]["replies"], 433)
        self.assertEqual(funding["limits"], {s: {u: max(limits[s][u], spent[s][u] + funding["bound"][s][u]) for u in self.J.UNITS} for s in spent})
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.PC._retained(c, self.loaded, "cold")
        self.PC.setup(c, self.loaded, "later-cold", self.original / "receipt.json", self.identity, self.audit)
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.C._retained(c, self.loaded, "trial")

    def test_producer_protocol_copy_and_qualification_changes_block_equal_retry(self):
        required = set(self.PC.PRODUCERS) | {"src/data-pipeline/paired_wave_collection.py", "experiments/batch_throughput.py", "experiments/mlx_batch_throughput.py"}
        self.assertTrue(required <= set(self.C.PRODUCERS)); c = self.fresh(); self.setup(c); before = c.events
        fingerprint = self.C.fingerprint(); fingerprint[next(iter(fingerprint))] = "0" * 64
        with mock.patch.object(self.C, "fingerprint", return_value=fingerprint), self.assertRaisesRegex(ValueError, "Producer sources changed"): self.setup(c)
        with mock.patch.object(self.C, "BASE_PROTOCOL_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "Base protocol changed"): self.setup(c)
        changed = self.root / "changed-protocol.md"; changed.write_bytes(self.C.PROTOCOL.read_bytes() + b"changed")
        with mock.patch.object(self.C, "PROTOCOL", changed), self.assertRaisesRegex(ValueError, "Protocol source changed"): self.setup(c)
        copied = c.root / self.methods[16]["synthetic"]["qualification"]["path"] / "response.json"
        original = copied.read_bytes(); copied.write_bytes(b"changed qualification")
        with self.assertRaises(ValueError): self.setup(c)
        copied.write_bytes(original)
        anchor = next(e["payload"] for e in before if e["kind"] == "wave_setup")
        (c.root / anchor["protocol"]["path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError): self.setup(c)
        self.assertEqual(c.events, before)

    def test_last_qualification_authority_and_new_unanchored_wave_supersession(self):
        for fault in ("missing", "wrong_profile", "pin"):
            c = self.fresh(); inputs = copy.deepcopy(self.qualifications)
            if fault == "missing": inputs[16].pop("synthetic")
            if fault == "wrong_profile": inputs[16]["synthetic"] = inputs[16]["actor"]
            if fault == "pin": inputs[16]["synthetic"]["response_sha256"] = "0" * 64
            with self.subTest(fault=fault), self.assertRaises(ValueError): self.setup(c, qualifications=inputs)
            self.assertFalse(any(e["kind"] in ("experiment", "wave_setup", "wave_plan", "generation_reserved", "native_intent") for e in c.events))
        c = self.fresh(); self.setup(c)
        with mock.patch.object(c, "extend", side_effect=OSError("before extend")), self.assertRaises(OSError): self.setup(c, "next")
        with self.assertRaisesRegex(ValueError, "Collection superseded"): self.setup(c)
        funding = [e["payload"]["funding"] for e in c.events if e["kind"] == "wave_setup"][-1]
        self.setup(c, "next"); self.assertEqual(c._accounting()[0], funding["limits"])
