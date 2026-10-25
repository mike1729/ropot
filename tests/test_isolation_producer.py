from contextlib import chdir, ExitStack
import json
import struct
import sys
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_isolation_qualification as pure, test_mlx_batch_throughput as native


class IsolationProducerTests(unittest.TestCase):
    def setUp(self):
        self.a = pure.IsolationQualificationTests(methodName="runTest"); self.a.setUp(); self.addCleanup(self.a.doCleanups)
        self.n = native.MlxBatchThroughputTests(methodName="runTest"); self.n.setUp(); self.addCleanup(self.n.doCleanups)
        self.M, self.H = self.a.M, self.a.H
        self.count, self.events, self.samplers, self.windows = 0, [], [], []

    def worker(self, *, lengths=(40, 7168, 7168), peer_token=4, early=None, serial_token=3, schema=4, lookahead=0):
        native.Generator.instances = []; self.samplers, self.windows = [], []
        job, baseline = self.a.protocol()
        if schema == 2: job = self.M.build_long_job(self.a.f.f.d, self.n.config)
        ready = baseline[2]["fixtures"][:len(job["fixtures"])]
        for fixture, length, token in zip(ready, lengths, (3, 3, peer_token)): fixture["prompt_ids"] = [token] * length
        def render(messages, *, tokenize, **kwargs):
            index = next(i for i, f in enumerate(job["fixtures"]) if f["messages"] is messages)
            return ready[index]["prompt_ids"] if tokenize else ready[index]["prompt"]
        tokenizer = NS(eos_token_ids={9}, encode=mock.Mock(return_value=[3]), decode=lambda tokens: f"text-{len(self.events)}", apply_chat_template=render)
        def seeded(*, temp, top_p, seed):
            counter = iter(range(4096))
            def sample(scores):
                step = next(counter)
                token = (seed + step) % 16 if temp else 9 if early == step + 1 else serial_token if self.active in ("q0", "q1") else 3
                return native.Sample([token])
            sampler = mock.Mock(side_effect=sample, seed=seed); self.samplers.append(sampler); return sampler
        proof = self.a.f.f.proof
        def controlled(request, guard, files, sources, emit, *args, **kwargs):
            emit("controls_checked", request_sha256=kwargs["identity"], proof=proof)
            return None, tokenizer, {}, proof, self.a.f.f.roles(True)
        job["source_hashes"][str(self.M.ROOT / "src/data-pipeline/mlx_worker.py")] = self.M.sha256(b"").hexdigest()
        helper = NS(watch_parent=lambda: None, file_sha256=lambda path: job["source_hashes"][path], load_guard=lambda request: None,
            inventory=lambda *args: {}, origin=lambda value: None, observe=lambda sources, expected: sources.update(expected), controlled_load=controlled)
        modules = {"mlx": NS(core=self.n.mx), "mlx.core": self.n.mx, "mlx_lm": NS(load=mock.Mock(), stream_generate=mock.Mock()),
            "mlx_lm.generate": NS(BatchGenerator=native.Generator), "mlx_lm.sample_utils": NS(make_sampler=seeded)}
        self.count += 1; directory = self.a.f.f.root / f"producer-{self.count}"; directory.mkdir()
        identity = self.M.sha256(self.H._raw(job)).hexdigest(); self.events = [dict(event="start", pid=7, job_sha256=identity)]
        def emit(event, **fields):
            if event == "window_start": self.assertTrue((directory / "qualification-rng.json").is_file())
            self.events.append(dict(event=event, job_sha256=identity, **fields))
        original = self.M.run_window
        def window(*args, **kwargs):
            self.active = args[3]["id"]; self.windows.append((self.active, kwargs)); return original(*args, **kwargs)
        patches = [mock.patch.dict(sys.modules, modules), mock.patch.object(self.M.Path, "read_bytes", return_value=b""),
            mock.patch.object(self.M.importlib.util, "module_from_spec", return_value=helper), mock.patch.object(self.M.threading, "Thread"),
            mock.patch.object(self.M, "make_sampler", seeded), mock.patch.object(self.M, "run_window", side_effect=window),
            mock.patch.object(native.Generator, "lookahead", lookahead)]
        with chdir(directory), ExitStack() as stack:
            for patch in patches: stack.enter_context(patch)
            self.M.run_job(job, identity, emit)
        self.tokenizer = tokenizer
        return job, self.events, directory

    def test_native_identity_canonical_digest_and_supported_dtypes(self):
        for dtype in ("mlx.core.bfloat16", "mlx.core.float16", "mlx.core.float32"):
            scores = NS(dtype=dtype, shape=(1, 2), tolist=mock.Mock(return_value=[[0., -1.25]]))
            result = NS(item=mock.Mock(return_value=1)); sampler = mock.Mock(return_value=result); emit = mock.Mock()
            def observe(value, returned):
                self.assertIs(value, scores); self.assertIs(returned, result)
                self.M._observe_score(value, returned, window="short-A", seq=1, emit=emit)
            with mock.patch.object(self.M, "make_sampler", return_value=sampler):
                wrapped = self.M.sampler_for(dict(sampling="greedy", seed=17, cap=16), 9, 3, observe)
                self.assertIs(wrapped(scores), result)
            sampler.assert_called_once(); self.assertIs(sampler.call_args.args[0], scores)
            scores.tolist.assert_called_once_with(); result.item.assert_called_once_with()
            expected = dict(window="short-A", row_index=0, seq=1, native_dtype=dtype, shape=[1, 2], finite=True,
                returned_token=1, observed_f32_sha256=self.M.sha256(struct.pack("<2f", 0., -1.25)).hexdigest())
            emit.assert_called_once_with("score", **expected)

    def test_observer_rejects_before_pack_and_before_tolist(self):
        cases = [("dtype", "float64"), ("shape", 1), ("shape", (2, 2)), ("shape", (1,)), ("shape", (1, 2, 1)),
            ("shape", (True, 2)), ("shape", (1, True)), ("shape", (1, 0)), ("shape", (1, 1048577)),
            ("values", ([],)), ("values", []), ("values", [()]), ("values", [[0.]]), ("values", [[0., 1., 2.]]),
            ("values", [[0, 1.]]), ("values", [[False, 1.]]), ("values", [[float("nan"), 1.]]),
            ("values", [[float("inf"), 1.]]), ("values", [[-float("inf"), 1.]]),
            ("token", True), ("token", 1.), ("token", -1), ("token", 2), ("seq", True), ("seq", 0), ("seq", 18)]
        for field, value in cases:
            scores = NS(dtype="mlx.core.float32", shape=(1, 2), tolist=mock.Mock(return_value=[[0., 1.]]))
            result = NS(item=mock.Mock(return_value=value if field == "token" else 1)); emit = mock.Mock()
            if field in ("dtype", "shape"): setattr(scores, field, value)
            if field == "values": scores.tolist.return_value = value
            with self.subTest(field=field, value=value), mock.patch.object(self.M.struct, "pack") as pack:
                with self.assertRaises(ValueError): self.M._observe_score(scores, result, window="short-A", seq=value if field == "seq" else 1, emit=emit)
                pack.assert_not_called(); emit.assert_not_called()
            if field in ("dtype", "shape", "seq"): scores.tolist.assert_not_called()

    def test_actual_worker_roundtrip_freshness_lookahead_and_early_eos(self):
        for early, lookahead, packets in ((None, 0, 96), (None, 1, 102), (3, 0, 18), (3, 1, 24)):
            job, events, directory = self.worker(early=early, lookahead=lookahead)
            parsed = self.a.parse(job, events); self.assertTrue(parsed["complete"], parsed["protocol_error"])
            self.assertEqual(len(parsed["score_packets"]), packets)
            self.assertEqual({p["window"] for p in parsed["score_packets"]}, set(self.H.TARGET_WINDOWS))
            self.assertTrue(all(p["row_index"] == 0 for p in parsed["score_packets"]))
            generators = native.Generator.instances; self.assertEqual(len({id(g) for g in generators}), 11)
            self.assertTrue(all(g.closed for g in generators))
            insertions = [row for g in generators for row in g.insertions]
            self.assertTrue(all(row["caches"] is None for row in insertions))
            self.assertEqual(len({id(row["samplers"][0]) for row in insertions}), len(insertions))
            self.tokenizer.encode.assert_called_once_with("0", add_special_tokens=False)
            self.assertTrue(all(options["safe_token"] == 3 and options["window_seconds"] == 240 for _, options in self.windows))
            self.assertEqual([name for name, options in self.windows if options["observe_target"]], list(self.H.TARGET_WINDOWS))

    def test_rejected_eighteenth_call_does_not_call_native_sampler(self):
        row = dict(fixture_id="x", seed=17, cap=16, sampling="greedy", after_row_index=None)
        window = dict(id="short-A", mode="qualification", config=self.n.config, rows=[row])
        sampler = mock.Mock(return_value=native.Sample([3]))
        with mock.patch.object(self.M, "make_sampler", return_value=sampler), mock.patch.object(native.Generator, "lookahead", 2):
            with self.assertRaisesRegex(ValueError, "cap plus lookahead"):
                self.M.run_window(None, self.n.tokenizer, self.n.fixtures, window, self.n.emit, observe_target=True)
        self.assertEqual(sampler.call_count, 17); self.assertTrue(native.Generator.instances[-1].closed)
        self.assertEqual(len([e for e in self.n.events if e["event"] == "score"]), 17)
        for safe in (True, -1, 9, 1.):
            with self.subTest(safe=safe), self.assertRaises(ValueError):
                self.M.run_window(None, self.n.tokenizer, self.n.fixtures, window, self.n.emit, safe_token=safe)

    def test_unobserved_window_preserves_three_argument_factory(self):
        original = self.M.sampler_for
        with mock.patch.object(self.M, "sampler_for", side_effect=lambda row, eos, safe: original(row, eos, safe)) as factory:
            self.M.run_window(None, self.n.tokenizer, self.n.fixtures, self.n.window, self.n.emit)
        self.assertEqual(factory.call_count, 3)
        self.assertTrue(all(len(call.args) == 3 for call in factory.call_args_list))

    def test_worker_context_sidecar_diagnostics_and_legacy_failure(self):
        for length in (7168, 8192):
            job, events, directory = self.worker(lengths=(40, length, length), serial_token=4)
            parsed = self.a.parse(job, events); self.assertTrue(parsed["complete"], parsed["protocol_error"])
            self.assertEqual(parsed["diagnostics"], dict(greedy_equal=False, reinsert_equal=True))
            qualification = next(e for e in events if e["event"] == "qualification")
            sidecar = json.loads((directory / "qualification-rng.json").read_text())
            self.assertEqual(sidecar["job_sha256"], self.H.sha256(self.H._raw(job)).hexdigest()); self.assertIs(sidecar["passed"], True)
            self.assertEqual(sidecar["draws"], json.loads(json.dumps(qualification["rng_draws"])))
        for lengths, peer in (((40, 7167, 7167), 4), ((40, 8193, 8193), 4), ((40, 7168, 7169), 4),
                              ((40, 7168, 7168), 3), ((7168, 7168, 7168), 4), ((0, 7168, 7168), 4)):
            with self.subTest(lengths=lengths, peer=peer), self.assertRaises(RuntimeError): self.worker(lengths=lengths, peer_token=peer)
            self.assertFalse(self.windows); self.assertNotIn("ready", [e["event"] for e in self.events])
        with self.assertRaisesRegex(RuntimeError, "Numerical batching qualification failed"):
            self.worker(lengths=(40, 3000), schema=2, serial_token=4)

    def test_cli_isolation_selection_controls_and_legacy_mode(self):
        profiles = self.n.root / "producer-profiles.json"; self.n.put(profiles, dict(profiles={str(self.n.d["bits"]): {self.n.d["source"]: dict(descriptor=self.n.d)}}))
        argv = [self.M.__file__, "--run", "--profiles", str(profiles), "--source", self.n.d["source"], "--bits", str(self.n.d["bits"]),
            "--configs", json.dumps([self.n.config]), "--output", str(self.n.root / "cli"), "--lock", str(self.n.root / "lock"), "--cache-gib", "4"]
        result = dict(status="passed", observed_generation_tokens=0, elapsed_seconds=1, protocol_error=None, windows=[])
        for flags, selected in ((["--isolation-qualification"], "isolation"), (["--isolation-qualification", "--wall", "240"], "isolation"), (["--long-qualification"], "long"), ([], "default")):
            with mock.patch.object(self.M.sys, "argv", argv + flags + ["--output", str(self.n.root / f"cli-{selected}-{len(flags)}")]), mock.patch.object(self.H, "run", return_value=result), mock.patch("builtins.print"), \
                 mock.patch.object(self.M, "build_isolation_job", return_value={}) as isolation, mock.patch.object(self.M, "build_long_job", return_value={}) as long, mock.patch.object(self.M, "build_job", return_value={}) as default:
                self.assertFalse(self.M.main())
                builders = dict(isolation=isolation, long=long, default=default)
                builders[selected].assert_called_once(); self.assertEqual(builders[selected].call_args.kwargs["cache_limit_bytes"], 4 * 1024**3)
                for name, builder in builders.items(): self.assertEqual(builder.call_count, int(name == selected))
        for flags in (["--long-qualification"], ["--wall", "239"], ["--panel-size", "32"], ["--configs", "[]"], ["--configs", "{}"], ["--configs", json.dumps([self.n.config] * 2)]):
            with mock.patch.object(self.M.sys, "argv", argv + ["--isolation-qualification"] + flags), mock.patch("sys.stderr"), \
                 mock.patch.object(self.H, "_write") as write, mock.patch.object(self.H, "run") as run, self.assertRaises((ValueError, SystemExit)):
                self.M.main()
            write.assert_not_called(); run.assert_not_called()
