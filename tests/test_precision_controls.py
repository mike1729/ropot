import copy
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import shutil
import sys
from threading import Barrier
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_precision_profiles as profiles, test_local_generation as local, test_generation_journal as journal


class PrecisionControlTests(unittest.TestCase):
    fixture, put = profiles.PrecisionProfileTests.fixture, profiles.PrecisionProfileTests.put

    def setUp(self):
        profiles.PrecisionProfileTests.setUp(self); self.G = importlib.import_module("src.data-pipeline.local_generation")
        self.G._PROTOCOL_MEMO.clear()
        self.addCleanup(self.G._PROTOCOL_MEMO.clear)
        self.W = importlib.import_module("src.data-pipeline.mlx_worker")
        suffixes = {suffix for source in ("actor", "synthetic") for post in (False, True)
                    for suffix in self.P.role_suffixes(source, post).values() if suffix} | {"/transformers/fixture.py"}
        for suffix in suffixes:
            path = self.root / "runtime" / suffix.lstrip("/"); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(suffix.encode())
            self.sources[str(path)] = self.loaded_sources[str(path)] = self.P.sha256(path.read_bytes()).hexdigest()
        original, identity, _ = self.fixture(); ref = self.P.freeze(self.campaign, original / "receipt.json", identity)
        self.d = self.P.resolve(self.campaign, ref, "actor", 8)
        for name, value in (("INTERPRETER", sys.executable), ("ENVIRONMENT", self.d["expected_runtime"]["environment"])):
            patch = mock.patch.object(self.G, name, value); patch.start(); self.addCleanup(patch.stop)
        self.raw_request = self.G.request_for("actor", [{"role": "user", "content": "fixture"}], profile=self.d,
            seed=17, temperature=0.2, top_p=0.9, max_tokens=1, wall_time_seconds=0.5)
        self.request = self.G.validate(self.raw_request); self.identity = "a" * 64
        self.proof = dict(descriptor_sha256=self.P.digest(self.d), guard_sha256=self.request["precision_profile"]["guard_sha256"],
            runtime_guard_sha256=self.d["runtime_guard_sha256"], model_files=self.d["model_files"],
            gpu=dict(available=True, default="Device(gpu,0)", info=dict(name="fixture")), critical_sources=self.roles())

    def roles(self, post=False):
        required = self.P.role_suffixes("actor", post); sources = self.d["expected_runtime"]["sources"]
        paths = {role: next(p for p in sources if p.endswith(suffix or "/transformers/fixture.py")) for role, suffix in required.items()}
        return self.P.bind_roles(sources, required, paths)

    def rows(self):
        return [dict(event="start", request_sha256=self.identity, pid=7),
            dict(event="controls_checked", request_sha256=self.identity, proof=copy.deepcopy(self.proof)),
            dict(event="ready", prompt="prompt", prompt_ids=[1], eos_token_ids=[9], provenance=dict(precision_controls=copy.deepcopy(self.proof),
                precision_postload=self.roles(True), sources=dict(self.d["expected_runtime"]["sources"])), hash_seconds=0, load_seconds=0),
            dict(event="chunk", seq=1, token=9, text="", prompt_tokens=1, generation_tokens=1, finish_reason="stop"),
            dict(event="done", request_sha256=self.identity, finish_reason="stop")]

    def parse(self, rows, tail=b"", pid=7, rc=0):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows) + tail
        return self.G.checked_protocol(raw, self.request, self.identity, pid, rc)

    def test_protocol_memo_returns_independent_complete_results(self):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in self.rows())
        args = raw, self.request, self.identity, 7, 0
        expected = self.G._checked_protocol(*args)
        with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
            first = self.G.checked_protocol(*args); self.assertEqual(first, expected)
            first["tokens"].append(3); first["events"][1]["proof"]["gpu"]["info"]["name"] = "changed"
            second = self.G.checked_protocol(*args); self.assertEqual(second, expected)
            second["events"].clear()
            self.assertEqual(self.G.checked_protocol(*args), expected)
        self.assertEqual(body.call_count, 1)
        self.assertIsNot(first["tokens"], second["tokens"])

    def test_protocol_memo_does_not_alias_unicode_scalar_and_surrogate_pair(self):
        scalar, pair, real = "\U0001f600", "\ud83d\ude00", self.G._checked_protocol
        rows = self.rows()
        for row in rows:
            if "request_sha256" in row: row["request_sha256"] = scalar
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        with mock.patch.object(self.G, "_checked_protocol", wraps=real) as body:
            self.assertTrue(self.G.checked_protocol(raw, self.request, scalar, 7, 0)["complete"])
            for _ in range(2):
                result = self.G.checked_protocol(raw, self.request, pair, 7, 0)
                self.assertEqual((result["complete"], result["protocol_error"]), (False, "Invalid start"))
        self.assertEqual(body.call_count, 3)
        with mock.patch.object(self.G, "_checked_protocol", wraps=real) as body:
            for content in (scalar, pair, pair):
                request = {**self.request, "messages": [{"role": "user", "content": content}]}
                expected = real(raw, request, scalar, 7, 0)
                self.assertTrue(expected["complete"])
                self.assertEqual(self.G.checked_protocol(raw, request, scalar, 7, 0), expected)
        self.assertEqual(body.call_count, 3)

    def test_protocol_memo_keys_all_inputs_and_scalar_types(self):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in self.rows())
        base = raw, self.request, self.identity, 7, 0
        self.G.checked_protocol(*base)
        variants = [(raw + b"x", self.request, self.identity, 7, 0), (raw.replace(b'"token": 9', b'"token": 8'), *base[1:])]
        for name, value in (("max_tokens", 2), ("seed", 19), ("messages", [{"role": "user", "content": "different"}])):
            variants.append((raw, {**self.request, name: value}, self.identity, 7, 0))
        for field in ("environment", "precision_profile"):
            changed = copy.deepcopy(self.request)
            if field == "environment": changed[field]["PATH"] = "different"
            else: changed[field]["descriptor"]["bits"] = True
            variants.append((raw, changed, self.identity, 7, 0))
        variants += [(raw, self.request, "b" * 64, 7, 0)]
        variants += [(raw, self.request, self.identity, pid, rc) for pid, rc in
            ((8, 0), (7.0, 0), (True, 0), (None, 0), (7, False), (7, 0.0), (7, None), (7, 3), (7, "0"))]
        one = raw.replace(b'"pid": 7', b'"pid": 1')
        variants += [(one, self.request, self.identity, pid, 0) for pid in (1, True, 1.0)]
        for args in variants:
            expected = self.G._checked_protocol(*args)
            with self.subTest(args=args[2:]), mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
                self.assertEqual(self.G.checked_protocol(*args), expected)
            self.assertEqual(body.call_count, 1)

    def test_protocol_memo_recomputes_negative_and_unsupported_inputs(self):
        rows = self.rows(); error = dict(event="profile_error", request_sha256=self.identity, code="guard", detail="fixture")
        damaged = self.rows(); damaged[1]["proof"]["gpu"]["available"] = 1
        cases = [(rows[:2], b"", 7, 0), (rows[:2], b'{"event":', 7, 3), (damaged, b"", 7, 0),
            ([rows[0], error], b"", 7, 3), ([rows[0], error], b"", 7, 0), (rows, b"", 8, 0),
            ([*rows[:2], dict(error, code="import_path")], b"", 7, 3)]
        for events, tail, pid, rc in cases:
            raw = b"".join((json.dumps(row) + "\n").encode() for row in events) + tail
            args = raw, self.request, self.identity, pid, rc
            expected = self.G._checked_protocol(*args); self.assertFalse(expected["complete"])
            with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
                self.assertEqual(self.G.checked_protocol(*args), expected); self.assertEqual(self.G.checked_protocol(*args), expected)
            self.assertEqual(body.call_count, 2)
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        self.G.checked_protocol(raw, self.request, self.identity, 7, 0)
        class Mapping(dict): pass
        unsupported = [{**self.request, "messages": tuple(self.request["messages"])},
            {**self.request, "precision_profile": Mapping(self.request["precision_profile"])},
            {**self.request, "unused": object()}, {**self.request, "unused": float("nan")}]
        for request in unsupported:
            args = raw, request, self.identity, 7, 0; expected = self.G._checked_protocol(*args)
            with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
                self.assertEqual(self.G.checked_protocol(*args), expected); self.assertEqual(self.G.checked_protocol(*args), expected)
            self.assertEqual(body.call_count, 2)
        with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
            for _ in range(2):
                with self.assertRaisesRegex(AttributeError, "splitlines"): self.G.checked_protocol(None, self.request, self.identity, 7)
        self.assertEqual(body.call_count, 2)

    def test_protocol_memo_uses_detached_request_for_key_and_computation(self):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in self.rows())
        request = copy.deepcopy(self.request); real = self.G._checked_protocol
        expected = real(raw, request, self.identity, 7, 0)
        def changed(raw, detached, *args):
            self.assertIsNot(detached, request); self.assertIsNot(detached["precision_profile"], request["precision_profile"])
            request["max_tokens"] = 0; request["precision_profile"]["guard_sha256"] = "0" * 64
            return real(raw, detached, *args)
        with mock.patch.object(self.G, "_checked_protocol", side_effect=changed) as body:
            self.assertEqual(self.G.checked_protocol(raw, request, self.identity, 7, 0), expected)
            self.assertEqual(self.G.checked_protocol(raw, self.request, self.identity, 7, 0), expected)
        self.assertEqual(body.call_count, 1)

    def test_protocol_memo_bounds_eviction_and_oversize_bypass(self):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in self.rows())
        with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
            for seed in range(257): self.assertTrue(self.G.checked_protocol(raw, {**self.request, "seed": seed}, self.identity, 7, 0)["complete"])
            self.assertEqual(len(self.G._PROTOCOL_MEMO), 256)
            self.G.checked_protocol(raw, {**self.request, "seed": 0}, self.identity, 7, 0)
        self.assertEqual(body.call_count, 258)
        self.assertTrue(all(type(k) is bytes and len(k) == 32 and type(v) is bytes and len(v) <= 1024**2 for k, v in self.G._PROTOCOL_MEMO.items()))
        self.G._PROTOCOL_MEMO.clear(); rows = self.rows(); rows[3]["text"] = "x" * (1024**2)
        raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
        expected = self.G._checked_protocol(raw, self.request, self.identity, 7, 0)
        self.assertTrue(expected["complete"]); self.assertGreater(len(json.dumps(expected).encode()), self.G._PROTOCOL_RESULT_LIMIT)
        with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
            for _ in range(2): self.assertEqual(self.G.checked_protocol(raw, self.request, self.identity, 7, 0), expected)
        self.assertEqual((body.call_count, len(self.G._PROTOCOL_MEMO)), (2, 0))

    def test_protocol_memo_concurrent_misses_preserve_atomic_bound(self):
        raw = b"".join((json.dumps(row) + "\n").encode() for row in self.rows())
        for seed in range(255): self.G.checked_protocol(raw, {**self.request, "seed": seed}, self.identity, 7, 0)
        case, real, barrier = self, self.G._checked_protocol, Barrier(8)
        class Bounded(dict):
            def __setitem__(self, key, value):
                if key not in self: case.assertLess(len(self), 256)
                super().__setitem__(key, value)
        def concurrent(*args): barrier.wait(timeout=10); return real(*args)
        with mock.patch.object(self.G, "_PROTOCOL_MEMO", Bounded(self.G._PROTOCOL_MEMO)), \
             mock.patch.object(self.G, "_checked_protocol", side_effect=concurrent), ThreadPoolExecutor(max_workers=8) as pool:
            for seeds in ([1000] * 8, list(range(1001, 1009))):
                results = list(pool.map(lambda seed: self.G.checked_protocol(raw, {**self.request, "seed": seed}, self.identity, 7, 0), seeds))
                self.assertTrue(all(result == results[0] and result["complete"] for result in results)); self.assertEqual(len(self.G._PROTOCOL_MEMO), 256)

    def test_warm_protocol_memo_does_not_hide_changed_retained_stdout(self):
        fixture = journal.GenerationJournalTests(methodName="runTest"); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        campaign, _ = fixture.create(); self.addCleanup(campaign.close)
        outcome = campaign.generate("one", "actor", fixture.request())
        with mock.patch.object(self.G, "_checked_protocol", wraps=self.G._checked_protocol) as body:
            self.assertEqual(campaign.outcome(outcome["reservation_seq"]), outcome)
        body.assert_not_called()  # The real legacy worker parse is already warm.
        path = campaign.root / "requests" / f"{outcome['reservation_seq']:06d}" / "stdout.jsonl"
        path.write_bytes(path.read_bytes() + b"bad JSON\n")
        with mock.patch.object(self.G, "generate", side_effect=AssertionError("Replay")), self.assertRaisesRegex(ValueError, "Outcome differs from protocol"):
            campaign.outcome(outcome["reservation_seq"])
        self.assertIsNone(campaign.db)

    def test_legacy_bytes_and_v2_schema_before_dispatch(self):
        legacy = local.LocalGenerationTests.request(self)
        expected = {**legacy, "fix_mistral_regex": False, "environment": self.G.ENVIRONMENT}
        self.assertEqual(self.G.prepare_request(legacy)[1], (json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())
        rows = self.rows(); parsed = self.G.protocol(b"".join((json.dumps(r) + "\n").encode() for r in [rows[0], *rows[2:]]), expected, self.identity)
        self.assertEqual(set(parsed), {"events", "tokens", "text", "prompt_tokens", "observed_generation_tokens", "finish_reason", "protocol_error", "complete"})
        self.assertFalse(Path(self.d["model_path"]).exists()); self.P.check_descriptor(self.d)
        for field, value in (("model_path", "/wrong"), ("fix_mistral_regex", True), ("precision_profile", {"descriptor": self.d, "guard_sha256": "wrong"})):
            request = {**self.raw_request, field: value}
            with mock.patch.object(self.G.subprocess, "Popen") as launch, self.assertRaises(ValueError): self.G.generate(request, self.root / "invalid")
            launch.assert_not_called(); self.assertFalse((self.root / "invalid").exists())
        for field, value in (("bits", True), ("model_files", {"weights": dict(bytes=True, sha256="b" * 64)}), ("runtime_guard_sha256", "0" * 64)):
            invalid = copy.deepcopy(self.raw_request); invalid["precision_profile"]["descriptor"][field] = value
            with self.assertRaises(ValueError): self.G.validate(invalid)
        for name, value in (("INTERPRETER", "/wrong"), ("ENVIRONMENT", {**self.G.ENVIRONMENT, "PATH": "wrong"})):
            with mock.patch.object(self.G, name, value), self.assertRaisesRegex(ValueError, "Profile launch differs"): self.G.validate(self.raw_request)
        with self.assertRaises(ValueError): self.G.request_for("synthetic", [{"role": "system", "content": "x"}], profile=self.d,
            **{k: self.raw_request[k] for k in ("seed", "temperature", "top_p", "max_tokens", "wall_time_seconds")})

    def test_proof_order_canonical_types_and_bound_ready(self):
        self.assertTrue(self.parse(self.rows())["complete"])
        for damage in ("hash", "gpu_bool", "file_float", "role", "ready_first", "duplicate", "ready_proof", "postload", "known_source"):
            rows = self.rows()
            if damage == "hash": rows[1]["proof"]["runtime_guard_sha256"] = "0" * 64
            if damage == "gpu_bool": rows[1]["proof"]["gpu"]["available"] = 1
            if damage == "file_float": rows[1]["proof"]["model_files"] = {"weights": dict(bytes=1.0, sha256="b" * 64)}
            if damage == "role": rows[1]["proof"]["critical_sources"]["loader"] = rows[1]["proof"]["critical_sources"]["generator"]
            if damage == "ready_first": rows.pop(1)
            if damage == "duplicate": rows.insert(2, rows[1])
            if damage == "ready_proof": rows[2]["provenance"]["precision_controls"]["guard_sha256"] = "0" * 64
            if damage == "postload": rows[2]["provenance"]["precision_postload"].pop("model")
            if damage == "known_source": rows[2]["provenance"]["sources"][next(iter(self.sources))] = "0" * 64
            with self.subTest(damage=damage): self.assertFalse(self.parse(rows)["complete"]); self.assertIsNotNone(self.parse(rows)["protocol_error"])
        for tail in (b"", b'{"event":', b"bad JSON\n"):
            parsed = self.parse(self.rows()[:2], tail, rc=3); self.assertTrue(parsed["profile_guarded"]); self.assertIsNone(parsed["profile_failure"])
        parsed = self.parse(self.rows(), pid=8); self.assertFalse(parsed["profile_guarded"])

    def test_terminal_errors_and_exit_code_cancel_untrusted_markers(self):
        for guarded in (False, True):
            prefix = self.rows()[:2 if guarded else 1]
            error = dict(event="profile_error", request_sha256=self.identity, code="import_path" if guarded else "guard", detail="fixture")
            rows = [*prefix, error]; result = self.parse(rows, rc=3)
            self.assertEqual((result["profile_guarded"], result["profile_failure"], result["protocol_error"], result["complete"]), (guarded, dict(code=error["code"], detail="fixture"), None, False))
            for tail, rc, pid in ((b"", 0, 7), (b"", None, 7), (b"", True, 7), (b"", 3.0, 7), (b"", 3, None), (b"", 3, True), (b"", 3, 7.0), (b"", 3, 0), (b"", 3, 8), (b"{}\n", 3, 7), (b"{", 3, 7)):
                invalid = self.parse(rows, tail, pid, rc); self.assertFalse(invalid["profile_guarded"]); self.assertIsNone(invalid["profile_failure"])
            for key, value in (("request_sha256", "wrong"), ("code", "unknown"), ("detail", "x" * 501)):
                invalid = self.parse([*prefix, {**error, key: value}], rc=3); self.assertIsNone(invalid["profile_failure"]); self.assertIsNotNone(invalid["protocol_error"])
        self.assertIsNone(self.parse([*self.rows()[:3], error], rc=3)["profile_failure"])

    def test_historical_proof_needs_no_live_files_and_journal_cannot_adopt(self):
        rows = self.rows(); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.G, "GUARD", self.root / "missing"), mock.patch.object(self.P, "check_runtime", side_effect=AssertionError("live guard")), mock.patch.object(self.G, "validate", side_effect=AssertionError("live validation")):
            self.assertTrue(self.parse(rows)["complete"])
        self.campaign.register(dict(experiment_id="fixture")); before = self.campaign.events
        with mock.patch.object(self.G, "generate") as launch, self.assertRaisesRegex(ValueError, "Registered profile direction differs"): self.campaign.generate("fixture", "actor", self.raw_request)
        launch.assert_not_called()
        with self.J.Campaign(self.campaign.root) as reopened: self.assertEqual(reopened.events, before)

    def test_owned_worker_exit0_marker_and_guarded_timeout(self):
        worker = self.root / "fake.py"
        code = "import hashlib,json,os,sys,time\np=sys.argv[1];r=open(p,'rb').read();d=hashlib.sha256(r).hexdigest()\ndef emit(**v):print(json.dumps(v),flush=True)\nemit(event='start',request_sha256=d,pid=os.getpid())\n"
        for mode in (0, 3, "guarded", "unproven"):
            ending = (f"emit(event='profile_error',request_sha256=d,code='guard',detail='fixture');sys.exit({mode})" if type(mode) is int else
                ("emit(event='controls_checked',request_sha256=d,proof=" + repr(self.proof) + ")\n" if mode == "guarded" else "") + "time.sleep(30)")
            worker.write_text(code + ending)
            with mock.patch.object(self.G, "WORKER", worker): result = self.G.generate(self.raw_request, self.root / f"worker-{mode}")
            self.assertEqual(result["status"], "worker_error" if type(mode) is int else "timeout")
            self.assertEqual(result["profile_guarded"], mode == "guarded"); self.assertEqual(result["profile_failure"] is not None, mode == 3)
        invalid = copy.deepcopy(self.raw_request); invalid["precision_profile"]["guard_sha256"] = "0" * 64; invalid["wall_time_seconds"] = 5
        result = self.G.generate(invalid, self.root / "actual-guard"); self.assertEqual((result["status"], result["returncode"], result["profile_failure"]["code"]), ("worker_error", 3, "guard"))

    def test_worker_hashes_captured_guard_and_inventory_before_imports(self):
        with mock.patch.object(self.W, "inventory", side_effect=AssertionError("inventory before runtime")), self.assertRaises(self.W.Control) as failure:
            self.W._run(self.request, self.identity, mock.Mock())
        self.assertEqual(failure.exception.code, "runtime")
        helper = self.root / "precision_profiles.py"; helper.write_text("raise AssertionError('unverified helper executed')")
        request = copy.deepcopy(self.request)
        with mock.patch.object(self.W, "__file__", str(self.root / "mlx_worker.py")), self.assertRaises(self.W.Control) as failure: self.W.load_guard(request)
        self.assertEqual(failure.exception.code, "guard")
        raw = f"class ProfileError(ValueError):pass\ndef check_runtime(expected):return {self.d['runtime_guard_sha256']!r}\n".encode()
        helper.write_bytes(raw); request["precision_profile"]["guard_sha256"] = self.P.sha256(raw).hexdigest(); original = self.W.importlib.util.spec_from_file_location
        def changed(name, path): helper.write_text("raise AssertionError('reread helper')"); return original(name, path)
        with mock.patch.object(self.W, "__file__", str(self.root / "mlx_worker.py")), mock.patch.object(self.W.importlib.util, "spec_from_file_location", side_effect=changed): self.W.load_guard(request)
        expected = {"weights": dict(bytes=1, sha256=self.P.sha256(b"x").hexdigest())}
        for fault in ("valid", "extra", ".cache", "missing", "symlink", "digest"):
            directory = self.root / f"asset-{fault}"; directory.mkdir(); weight = directory / "weights"; weight.write_bytes(b"x")
            if fault in ("extra", ".cache"): (directory / fault).mkdir()
            if fault == "missing": weight.unlink()
            if fault == "symlink": weight.rename(directory.parent / "identical"); weight.symlink_to(directory.parent / "identical")
            if fault == "digest": weight.write_bytes(b"wrong")
            if fault == "valid":
                with mock.patch.object(self.W, "file_sha256", wraps=self.W.file_sha256) as hashed: self.assertEqual(self.W.inventory(directory, self.P, dict(model_files=expected)), expected)
                self.assertEqual(hashed.call_count, 1)
            else:
                with self.assertRaises(self.W.Control) as failure: self.W.inventory(directory, self.P, dict(model_files=expected))
                self.assertEqual(failure.exception.code, "artifact")

    def test_worker_control_order_gpu_origins_and_observed_io(self):
        pre, post = self.roles(), self.roles(True); events, observed = [], {}
        obj = lambda role, table=pre: NS(__file__=table[role]["path"])
        model = obj("model", post); model.language_model = obj("text_model", post)
        tokenizer = obj("tokenizer_wrapper", post); tokenizer._tokenizer = obj("tokenizer_impl", post)
        def load(*args, **kwargs): events.append(("load", kwargs)); return model, tokenizer, {}
        load.__file__ = pre["loader"]["path"]
        mx = obj("native_core"); mx.gpu = "gpu"; mx.is_available = lambda _: True; mx.default_device = lambda: NS(type="gpu"); mx.device_info = lambda: dict(name="fixture")
        extra = self.root / "extra.py"; extra.write_bytes(b"observed only")
        modules = {"mlx_lm.models.cache": obj("cache"), "tokenizers.tokenizers": obj("tokenizer_native", post), "mlx_lm.extra": NS(__file__=str(extra))}
        emit = lambda event, **fields: events.append(event)
        with mock.patch.dict(sys.modules, modules), mock.patch.object(self.W.importlib, "import_module", side_effect=lambda name: mx if name == "mlx.core" else obj("architecture")):
            result = self.W.controlled_load(self.request, self.P, self.d["model_files"], observed, emit, load, obj("generator"), obj("sampler"), identity=self.identity, tokenizer_config={"local_files_only": True})
            self.assertEqual((result[3]["critical_sources"], result[4]), (pre, post)); self.assertEqual(events[0], "controls_checked"); self.assertEqual(events[1][0], "load")
            self.assertEqual(events[1][1], dict(tokenizer_config={"local_files_only": True}, lazy=False, return_config=True, trust_remote_code=False)); self.assertIn(str(extra), observed)
            for available, device in ((False, "gpu"), (True, "cpu")):
                mx.is_available = lambda _: available; mx.default_device = lambda: NS(type=device)
                with self.assertRaises(self.W.Control) as failure: self.W.controlled_load(self.request, self.P, self.d["model_files"], {}, emit, load, obj("generator"), obj("sampler"), identity=self.identity, tokenizer_config={"local_files_only": True})
                self.assertEqual(failure.exception.code, "device"); self.assertEqual(len(events), 2)
            with mock.patch.object(self.W, "file_sha256", side_effect=OSError("observed unreadable")), self.assertRaises(self.W.Control) as failure: self.W.observe({}, self.d["expected_runtime"]["sources"])
            self.assertEqual(failure.exception.code, "import_path")
        paths = {role: value["path"] for role, value in pre.items()}; shadow = self.root / "shadow.py"; shadow.write_bytes(Path(paths["loader"]).read_bytes()); paths["loader"] = str(shadow)
        with self.assertRaises(self.W.Control) as failure: self.W.roles(self.P, self.d, paths, {})
        self.assertEqual(failure.exception.code, "import_path")
