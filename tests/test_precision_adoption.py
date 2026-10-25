import copy
import importlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock
from tests import test_integrated_loop as loop, test_precision_profiles as profiles

WORKER = '''
import hashlib,importlib.util,json,os,pathlib,sys,time
raw=pathlib.Path(sys.argv[1]).read_bytes(); request=json.loads(raw); identity=hashlib.sha256(raw).hexdigest()
value=json.loads((pathlib.Path(request['model_path'])/'replies.json').read_text())[str(request['seed'])]
mode=value.get('mode','success') if isinstance(value,dict) else 'success'; text=value.get('text','print(input())') if isinstance(value,dict) else value
def emit(event,**values):print(json.dumps(dict(event=event,**values)),flush=True)
emit('start',request_sha256=identity,pid=os.getpid()+(mode=='pid'))
if mode=='pre_timeout':time.sleep(30)
if mode=='typed':emit('profile_error',request_sha256=identity,code='runtime',detail='fixture');sys.exit(3)
provenance={}
if 'precision_profile' in request:
 wrapper=request['precision_profile']; descriptor=wrapper['descriptor']; sources=descriptor['expected_runtime']['sources']
 spec=importlib.util.spec_from_file_location('fixture_guard',GUARD); p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
 def roles(post=False):
  required=p.role_suffixes(descriptor['source'],post)
  paths={r:next(n for n in sources if n.endswith(s or '/transformers/fixture.py')) for r,s in required.items()}
  return p.bind_roles(sources,required,paths)
 proof=dict(descriptor_sha256=p.digest(descriptor),guard_sha256=wrapper['guard_sha256'],runtime_guard_sha256=descriptor['runtime_guard_sha256'],
  model_files=descriptor['model_files'],gpu=dict(available=True,default='fixtureGPU',info=dict(name='fixture')),critical_sources=roles())
 emit('controls_checked',request_sha256=identity,proof=proof)
 if mode=='post':emit('profile_error',request_sha256=identity,code='import_path',detail='fixture');sys.exit(3)
 if mode=='guarded_timeout':time.sleep(30)
 if mode=='guarded':sys.exit(3)
 if mode=='protocol':emit('invalid');sys.exit(3)
 provenance=dict(precision_controls=proof,precision_postload=roles(True),sources=sources)
emit('ready',prompt='prompt',prompt_ids=[1],eos_token_ids=[9],provenance=provenance,hash_seconds=0,load_seconds=0)
emit('chunk',seq=1,token=9,text=text,prompt_tokens=1,generation_tokens=1,finish_reason='stop')
if mode=='before_done_tail':sys.stdout.write('{"event":');sys.exit(3)
emit('done',request_sha256=identity,finish_reason='stop')
if mode=='after_done_tail':sys.stdout.write('{"event":');sys.exit(3)
'''


class PrecisionAdoptionTests(unittest.TestCase):
    write_task, rewrite = loop.IntegratedLoopTests.write_task, loop.IntegratedLoopTests.rewrite
    limits, rows = loop.IntegratedLoopTests.limits, loop.IntegratedLoopTests.rows
    read, execute, replies = loop.IntegratedLoopTests.read, loop.IntegratedLoopTests.execute, loop.IntegratedLoopTests.replies
    recipe, fresh = loop.IntegratedLoopTests.recipe, loop.IntegratedLoopTests.fresh
    fixture, put = profiles.PrecisionProfileTests.fixture, profiles.PrecisionProfileTests.put

    def setUp(self):
        loop.IntegratedLoopTests.setUp(self); self.root = self.root.resolve()
        self.P = importlib.import_module("src.data-pipeline.precision_profiles"); self.R = importlib.import_module("src.data-pipeline.study_report")
        self.sources = {}; suffixes = set(self.P.CRITICAL_SUFFIXES) | {"/transformers/fixture.py"}
        suffixes.update(s for source in ("actor", "synthetic") for post in (False, True) for s in self.P.role_suffixes(source, post).values() if s)
        for suffix in suffixes:
            path = self.root / "runtime" / suffix.lstrip("/"); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(suffix.encode())
            self.sources[str(path)] = self.P.sha256(path.read_bytes()).hexdigest()
        self.loaded_sources = dict(self.sources); self.original, self.identity, runtime = self.fixture()
        patch = mock.patch.object(self.api, "ENVIRONMENT", runtime["environment"]); patch.start(); self.addCleanup(patch.stop)
        self.worker.write_text("GUARD=" + repr(str(self.api.GUARD)) + "\n" + WORKER)
        for bits in (8, 16):
            for source in ("actor", "synthetic"): self.mode("success", bits, source)

    def mode(self, mode, bits=8, source="actor"):
        path = self.original / "models" / f"{source}-{bits}"; path.mkdir(parents=True, exist_ok=True)
        (path / "replies.json").write_text(json.dumps({1: dict(mode=mode, text="print(input())")}))

    def record(self, campaign, name="trial", bits=8, sources=("actor", "synthetic")):
        reference = self.P.freeze(campaign, self.original / "receipt.json", self.identity) if bits else None
        bound = {s: dict(reference=reference, descriptor=self.P.resolve(campaign, reference, s, bits)) for s in sources} if bits else None
        return self.L.run_record(self.loaded, name, ["development_smoke"], {s: self.recipe((1,)) for s in sources}, [1], profiles=bound)

    def reopen(self, campaign):
        campaign.close(); self.campaign = self.J.Campaign(campaign.root); self.addCleanup(self.campaign.close)
        return self.campaign

    def request(self, record, source="actor", descriptor=None):
        return self.api.request_for(source, [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "fixture"}],
            profile=descriptor or record["profiles"][source]["descriptor"], seed=1, temperature=0.2, top_p=0.9, max_tokens=1, wall_time_seconds=0.5)

    def test_predicate_uses_only_complete_durable_fields(self):
        self.assertIsNone(self.J.profile_control_reason({}))
        with self.assertRaises(ValueError): self.J.profile_control_reason(dict(profile_guarded=True))
        for guarded, failure, error, reason in ((True, None, None, None), (True, None, "Missing done", None), (True, None, "Incomplete protocol tail", None),
                (False, None, "Missing done", "profile_unproven"), (False, dict(code="runtime", detail="fixture"), None, "profile_failure"), (True, None, "Invalid ready", "profile_protocol")):
            self.assertEqual(self.J.profile_control_reason(dict(profile_guarded=guarded, profile_failure=failure, protocol_error=error, response=object())), reason)

    def test_feasibility_v2_factory_rebuild_without_task_payload(self):
        cohort = copy.deepcopy(self.cohort); cohort["candidate_budgets"]["construction"]["feasibility_per_phase"] = 3
        self.loaded = self.rewrite(cohort); (self.pool / "tasks/test_12.json").unlink()
        bound = self.record(self.fresh(), sources=("actor",))["profiles"]
        with mock.patch.object(self.L.study, "load_task", side_effect=AssertionError("task payload read")):
            record = self.L.run_record(self.loaded, "extended", ["shared_pretraining_probe"], {"actor": self.recipe((1,))}, [12], profiles=bound, construction_per_phase=3)
            self.assertEqual(self.L.validate_run(self.loaded, record), record)
        self.assertEqual((record["schema_version"], record["profiles"], record["assigned_task_ids"], record["activation"]["construction_per_phase"]), (2, bound, [12], 3))

    def test_binding_is_before_reservation_and_direction_is_specific(self):
        for fault in ("legacy", "unprofiled", "arm", "helper", "source", "reference"):
            reason = {"legacy": "Registered profile direction differs", "unprofiled": "Registered profile direction differs",
                "arm": "Registered profile request differs", "helper": "Registered profile request differs",
                "source": "Registered profile source differs", "reference": "Profile reference differs"}[fault]
            campaign = self.fresh(); record = self.record(campaign, bits=None if fault == "legacy" else 8, sources=("actor",))
            other = self.record(campaign, "other", 16, sources=("actor",)); request = self.request(other if fault in ("legacy", "arm") else record)
            if fault == "unprofiled": request = self.api.request_for("actor", request["messages"], **{k: request[k] for k in ("seed", "temperature", "top_p", "max_tokens", "wall_time_seconds")})
            if fault == "helper": request["precision_profile"]["guard_sha256"] = "0" * 64
            if fault == "source": request = self.request(self.record(campaign, "source", sources=("synthetic",)), "synthetic")
            if fault == "reference":
                original, identity, _ = self.fixture(); record["profiles"]["actor"]["reference"] = self.P.freeze(campaign, original / "receipt.json", identity)
            campaign.register(record); before = campaign.events
            with mock.patch.object(self.api, "generate") as dispatch, self.assertRaisesRegex(ValueError, reason):
                campaign.generate(record["experiment_id"], "synthetic" if fault == "source" else "actor", request)
            dispatch.assert_not_called(); campaign = self.reopen(campaign); self.assertEqual(campaign.events, before)
            self.assertFalse((campaign.root / "requests").exists())

    def test_factory_and_all_source_preflight_reject_before_work(self):
        campaign = self.fresh(); record = self.record(campaign); other = self.record(campaign, "other", 16)
        for fault, reason in (("missing", "Profile sources differ"), ("arm", "Profile arm differs")):
            bound = copy.deepcopy(record["profiles"])
            if fault == "missing": bound.pop("synthetic")
            else: bound["synthetic"] = other["profiles"]["synthetic"]
            with self.assertRaisesRegex(self.P.ProfileError, reason): self.L.run_record(self.loaded, "invalid", ["development_smoke"], record["recipes"], [1], profiles=bound)
        record["profiles"]["synthetic"]["descriptor"]["model_files"]["weights"]["sha256"] = "0" * 64; before = campaign.events
        with mock.patch.object(self.api, "generate", side_effect=AssertionError("premature generation")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("premature native")), self.assertRaisesRegex(self.P.ProfileError, "Profile metadata differs"):
            self.L.run(campaign, self.loaded, record)
        self.assertEqual(campaign.events, before)

    def test_v2_whole_replay_checks_old_request_before_any_new_dispatch(self):
        campaign = self.fresh(); record = self.record(campaign); campaign.register(record)
        contract = self.N.verification.runner.contract()
        campaign._append("execution_context", experiment_id=record["experiment_id"], contract=contract, contract_sha256=self.N.verification.digest(contract))
        binding, task = self.L.study.load_task(self.loaded, record["activation"], 1)
        self.N.run_native(campaign, task, binding["sha256"]); campaign.generate(record["experiment_id"], "actor", self.request(record), attempt=dict(split=binding["split"], id=1, task_sha256=binding["sha256"], reply_index=0))
        before = campaign.events
        with mock.patch.object(self.api, "generate", side_effect=AssertionError("premature generation")), mock.patch.object(self.N, "run_native", side_effect=AssertionError("premature native")), self.assertRaisesRegex(ValueError, "Retained request differs from replay"):
            self.L.run(campaign, self.loaded, record)
        self.assertEqual(campaign.events, before)

    def test_control_stops_are_durable_and_later_id_shares_remaining_caps(self):
        for mode, reason in (("typed", "profile_failure"), ("post", "profile_failure"), ("pre_timeout", "profile_unproven"), ("pid", "profile_unproven"), ("protocol", "profile_protocol"), ("after_done_tail", "profile_protocol"), ("popen", "profile_unproven")):
            self.mode("typed" if mode == "popen" else mode); campaign = self.fresh(replies=2); record = self.record(campaign)
            launch = mock.patch.object(self.api.subprocess, "Popen", side_effect=OSError("fixture Popen")) if mode == "popen" else mock.patch.object(self.api, "INTERPRETER", self.api.INTERPRETER)
            with launch, self.assertRaises(OSError if mode == "popen" else self.J.ProfileControlError): self.L.run(campaign, self.loaded, record)
            campaign = self.reopen(campaign); done = next(e for e in campaign.events if e["kind"] == "generation_done")
            self.assertEqual(self.J.profile_control_reason(done), reason); self.assertEqual(done["reservation_seq"], next(e["seq"] for e in campaign.events if e["kind"] == "generation_reserved"))
            cells = self.L.cell_states(campaign.events, record); self.assertEqual((cells[0]["status"], cells[0]["reason"]), ("blocked", reason))
            self.assertEqual(sum(e["kind"] == "generation_reserved" for e in campaign.events), 1); self.assertFalse(any(e["kind"] == "attempt_result" for e in campaign.events))
            path = campaign.root / record["profiles"]["actor"]["reference"]["path"] / "runtime.json"; saved = path.read_bytes(); path.write_bytes(b"corrupt")
            with self.assertRaises(self.P.ProfileError): self.P.check_profiles(record["profiles"], record["recipes"], campaign=campaign)
            path.write_bytes(saved); self.P.check_profiles(record["profiles"], record["recipes"], campaign=campaign); before = campaign.events
            with mock.patch.object(self.api, "generate", side_effect=AssertionError("resume dispatch")), self.assertRaises(self.J.ProfileControlError) as failure: self.L.run(campaign, self.loaded, record)
            self.assertEqual(failure.exception.reason, reason); self.assertEqual(campaign.events, before)
            if mode == "typed":
                later = self.record(campaign, "later", 16, ("actor",)); self.assertEqual(self.L.run(campaign, self.loaded, later)[0]["status"], "verified")
                self.assertEqual(campaign._accounting()[1]["actor"], dict(replies=2, reserved_output_tokens=2, reserved_generation_wall_seconds=1.0))
                last = self.record(campaign, "last", 16, ("actor",)); self.assertEqual(self.L.run(campaign, self.loaded, last)[0]["reason"], "campaign_budget")
            campaign.close()

    def test_guarded_load_failures_continue_independent_source(self):
        for mode, reason in (("guarded", "generation_error"), ("guarded_timeout", "generation_timeout"), ("before_done_tail", "generation_error")):
            self.mode(mode); campaign = self.fresh(); record = self.record(campaign)
            cells = self.L.run(campaign, self.loaded, record)
            self.assertEqual([(c["status"], c["reason"]) for c in cells[:2]], [("blocked", reason), ("verified", None)])
            if mode == "before_done_tail": self.assertEqual(next(e for e in campaign.events if e["kind"] == "generation_done")["protocol_error"], "Incomplete protocol tail")
            self.assertEqual(sum(e["kind"] == "generation_reserved" for e in campaign.events), 2); campaign.close()

    def test_missing_response_guarded_prefix_is_interrupted_without_retry(self):
        self.mode("guarded"); campaign = self.fresh(); record = self.record(campaign); generate = self.api.generate
        def lose_response(*args, **kwargs):
            result = generate(*args, **kwargs); (Path(args[1]) / "response.json").unlink(); raise OSError("crash after raw output")
        with mock.patch.object(self.api, "generate", side_effect=lose_response), self.assertRaises(OSError): self.L.run(campaign, self.loaded, record)
        campaign = self.reopen(campaign); outcome = campaign.outcome(next(e["seq"] for e in campaign.events if e["kind"] == "generation_reserved"))
        self.assertEqual((outcome["status"], outcome["profile_guarded"], outcome["protocol_error"], outcome["response"]), ("interrupted", True, "Missing done", None))
        cells = self.L.run(campaign, self.loaded, record); self.assertEqual([(c["status"], c["reason"]) for c in cells[:2]], [("blocked", "interrupted"), ("verified", None)])
        self.assertEqual(sum(e["kind"] == "generation_reserved" for e in campaign.events), 2)
        shutil.rmtree(self.original / "models"); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.api, "prepare_request", side_effect=AssertionError("live preparation")), mock.patch.object(self.P, "check_runtime", side_effect=AssertionError("live guard")):
            costs = self.R.snapshot(campaign)["report"]["costs"]["observed_by_source"]["actor"]
        self.assertEqual((costs["unknown_response_requests"], costs["unknown_elapsed_requests"], costs["lower_bound_requests"]), (1, 1, 1))

    def test_mixed_offline_report_publication_and_all_unselected_requests(self):
        empty = self.fresh(); zero = self.record(empty, "zero", sources=("synthetic",)); empty.register(zero)
        prefix = zero["profiles"]["synthetic"]["reference"]["path"]
        self.assertEqual(set(self.R.snapshot(empty)["files"]), {prefix + "/" + n for n in self.P.FILES}); empty.close()
        campaign = self.fresh(); legacy = self.record(campaign, "legacy", bits=None, sources=("actor",)); self.L.run(campaign, self.loaded, legacy)
        good = self.record(campaign, "good", sources=("actor",)); self.L.run(campaign, self.loaded, good)
        self.mode("typed", 16); bad = self.record(campaign, "bad", 16, ("actor",))
        with self.assertRaises(self.J.ProfileControlError): self.L.run(campaign, self.loaded, bad)
        campaign = self.reopen(campaign); campaign.register(self.record(campaign, "zero", sources=("synthetic",)))
        outcomes = {e["seq"]: campaign.outcome(e["seq"]) for e in campaign.events if e["kind"] == "generation_reserved"}
        shutil.rmtree(self.original / "models"); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.api, "prepare_request", side_effect=AssertionError("live preparation")), mock.patch.object(self.P, "check_runtime", side_effect=AssertionError("live runtime")):
            snapshot = self.R.snapshot(campaign); published = self.R.publish(campaign, "v1")
            selected = self.R.snapshot(campaign, selections=[next(e["seq"] for e in campaign.events if e["kind"] == "generation_reserved" and e["experiment_id"] == "good")])["manifest"]["selections"][0]
        runs = snapshot["report"]["runs"]; self.assertNotIn("precision_profile", runs[0]["sources"][0])
        self.assertEqual(runs[1]["sources"][0]["precision_profile"]["bits"], 8); self.assertEqual(runs[2]["sources"][0]["reason_counts"], dict(profile_failure=1))
        identity_fields = {"source", "bits", "label", "receipt_sha256", "record_sha256", "runtime_guard_sha256", "descriptor_sha256"}
        self.assertEqual(set(selected["precision_profile"]), identity_fields); self.assertEqual(set(runs[1]["sources"][0]["precision_profile"]), identity_fields)
        prefix = good["profiles"]["actor"]["reference"]["path"]
        self.assertEqual({n for n in snapshot["files"] if n.startswith(prefix + "/")}, {prefix + "/" + n for n in self.P.FILES})
        self.assertTrue((Path(published["path"]) / "manifest.json").is_file())
        # Isolate S's independent all-request binding after J has validated retained outcomes.
        events = campaign.events; reserved = next(e for e in events if e["kind"] == "generation_reserved" and e["experiment_id"] == "bad")
        request = json.loads(reserved["request_text"]); descriptor = good["profiles"]["actor"]["descriptor"]
        request["precision_profile"]["descriptor"] = descriptor; request.update({k: descriptor[k] for k in ("model_path", "enable_thinking", "fix_mistral_regex")}); reserved["request_text"] = self.J.encoded(request) + "\n"
        with mock.patch.object(type(campaign), "events", new_callable=mock.PropertyMock, return_value=events), mock.patch.object(campaign, "_finish", side_effect=lambda seq, events: outcomes[seq]), self.assertRaisesRegex(ValueError, "Request profile binding differs"):
            self.R.snapshot(campaign)
        path = campaign.root / prefix / "runtime.json"; path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(self.P.ProfileError, "Runtime record binding differs"): self.R.snapshot(campaign)
