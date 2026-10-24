from contextlib import contextmanager
import copy
import importlib
import json
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_precision_controls as controls


class Generator:
    instances = []
    def __init__(self, model, **config):
        self.config, self.stream, self.pending, self.insertions = config, object(), {}, []
        self.uid, self.total, self.closed = 0, 0, False; self.instances.append(self)
    def insert(self, prompts, max_tokens=None, caches=None, samplers=None):
        self.insertions.append(dict(prompts=prompts, caps=max_tokens, caches=caches, samplers=samplers))
        uid = self.uid; self.uid += 1; self.pending[uid] = [0, max_tokens[0], samplers[0]]
        return [uid]
    def next(self):
        responses = []
        for uid in list(self.pending)[:self.config["completion_batch_size"]]:
            state = self.pending[uid]; state[0] += 1; token = state[2](NS(shape=(1, 16)))[0]
            finish = "stop" if token == 9 else "length" if state[0] == state[1] else None
            responses.append(NS(uid=uid, token=token, finish_reason=finish)); self.total += 1
            if finish: del self.pending[uid]
        return [], responses
    @contextmanager
    def stats(self):
        value = NS(generation_tokens=0, generation_tps=1, prompt_tps=1, decode_tps=1)
        yield value
        value.generation_tokens = self.total
    def close(self): self.closed = True


class MlxBatchThroughputTests(unittest.TestCase):
    F = controls.PrecisionControlTests
    fixture, put, roles = F.fixture, F.put, F.roles
    def setUp(self):
        self.F.setUp(self); self.B = importlib.import_module("experiments.mlx_batch_throughput")
        self.mx = NS(array=lambda values, **kw: values, synchronize=mock.Mock(), reset_peak_memory=mock.Mock(),
            get_active_memory=lambda: 1, get_cache_memory=lambda: 2, get_peak_memory=lambda: 3, uint32="uint32",
            full=lambda shape, token, **kw: [token] * shape[0], zeros=lambda shape: NS(shape=shape))
        def seeded(**kw):
            sample = lambda logits: [3]
            sample.seed = kw["seed"]
            return sample
        self.sampler = mock.Mock(side_effect=seeded)
        for name, value in dict(mx=self.mx, BatchGenerator=Generator, make_sampler=self.sampler).items():
            patch = mock.patch.object(self.B, name, value); patch.start(); self.addCleanup(patch.stop)
        Generator.instances = []; self.events = []
        self.config = dict(batch_size=2, prefill_batch_size=2, prefill_step_size=2048)
        self.window = dict(id="q3", mode="qualification", config=self.config, rows=[dict(fixture_id="x", seed=17, cap=cap,
            sampling=sampling, after_row_index=after) for cap, sampling, after in ((2, "eos", None), (3, "length", None), (3, "greedy", 0))])
        self.tokenizer = NS(eos_token_ids={9}, decode=lambda tokens: ",".join(map(str, tokens)), encode=lambda text, **kw: [3])
        self.fixtures = dict(x=dict(prompt="fixture", prompt_ids=[1, 2]))
    def emit(self, event, **fields): self.events.append(dict(event=event, **fields))

    def test_resident_window_keeps_uid_cache_seed_and_terminal_accounting(self):
        result = self.B.run_window(None, self.tokenizer, self.fixtures, self.window, self.emit)
        self.assertEqual([r["tokens"] for r in result], [[3, 9], [3, 3, 3], [3, 3, 3]])
        self.assertEqual([r["finish_reason"] for r in result], ["stop", "length", "length"])
        generator = Generator.instances[-1]; self.assertTrue(generator.closed)
        self.assertEqual(generator.config["completion_batch_size"], 2); self.assertEqual(generator.config["prefill_step_size"], 2048)
        self.assertTrue(all(row["caches"] is None for row in generator.insertions))
        self.assertEqual(len({id(row["samplers"][0]) for row in generator.insertions}), 3)
        late = next(i for i, e in enumerate(self.events) if e["event"] == "admit" and e["row_index"] == 2)
        terminal = next(i for i, e in enumerate(self.events) if e["event"] == "token" and e["uid"] == 0 and e["finish_reason"] == "stop")
        self.assertGreater(late, terminal); self.assertEqual(self.events[-1]["stats"]["generation_tokens"], 8)
        self.assertTrue(all(call.args == (generator.stream,) for call in self.mx.synchronize.call_args_list))

    def test_sampler_is_fresh_per_reply_and_forced_modes_do_not_replace_native_eos(self):
        row = dict(seed=29, cap=2, sampling="sample")
        first = self.B.sampler_for(row, 9, 3); second = self.B.sampler_for(row, 9, 3)
        self.assertIsNot(first, second)
        self.assertEqual(self.sampler.call_args.kwargs, dict(temp=.2, top_p=.95, seed=29))
        forced = self.B.sampler_for({**row, "sampling": "eos"}, 9, 3)
        self.assertEqual([forced(NS(shape=(1, 16)))[0], forced(NS(shape=(1, 16)))[0]], [3, 9])

    def test_generator_is_closed_when_backend_or_emitter_fails(self):
        def failed(event, **fields):
            if event == "token": raise RuntimeError("emitter")
        for target in ("next", "emit"):
            with self.subTest(target=target), mock.patch.object(Generator, "next", side_effect=RuntimeError("backend")) if target == "next" else mock.patch.object(self, "emit", side_effect=failed):
                with self.assertRaises(RuntimeError): self.B.run_window(None, self.tokenizer, self.fixtures, self.window, self.emit)
            self.assertTrue(Generator.instances[-1].closed)
        with mock.patch.object(self.B, "monotonic", side_effect=[0, 181]), self.assertRaises(TimeoutError):
            self.B.run_window(None, self.tokenizer, self.fixtures, self.window, self.emit)
        self.assertTrue(Generator.instances[-1].closed)

    def test_finite_builder_and_result_qualification_oracle(self):
        job = self.B.build_job(self.d, [self.config], wall_seconds=10)
        self.assertEqual(sum(len(w["rows"]) for w in job["windows"] if w["mode"] == "qualification"), 8)
        self.assertEqual(sum(w["mode"] == "measure" for w in job["windows"]), 2)
        self.assertEqual(len(job["fixtures"]), 8)
        for config in ({**self.config, "batch_size": True}, {**self.config, "prefill_batch_size": 3}):
            with self.assertRaises(ValueError): self.B.build_job(self.d, [config])
        row = lambda tokens, reason="length": dict(tokens=tokens, text="fixture", finish_reason=reason)
        results = dict(q0=[row([3, 3, 3])], q1=[row([4])], q2=[row([3, 3, 3]), row([4])],
            q3=[row([3, 9], "stop"), row([3, 3, 3]), row([3, 3, 3])], q4=[row([3, 3, 3])])
        self.assertTrue(all(self.B.qualification_checks(results, True).values()))
        changed = copy.deepcopy(results); changed["q2"][0]["tokens"][0] = 4
        self.assertFalse(self.B.qualification_checks(changed, True)["greedy_equal"])
        self.assertFalse(self.B.qualification_checks(results, False)["seed_independent"])

    def test_rng_oracle_uses_fresh_seed_state_and_rejects_constant_draws(self):
        def independent(**kw):
            counter = iter(range(16))
            return lambda logits: NS(item=lambda: (kw["seed"] + next(counter)) % 16)
        with mock.patch.object(self.B, "make_sampler", side_effect=independent) as created:
            passed, draws = self.B.rng_check()
        self.assertTrue(passed); self.assertEqual(created.call_count, 6)
        self.assertEqual(draws["serial"], draws["reversed"])
        with mock.patch.object(self.B, "make_sampler", return_value=lambda logits: NS(item=lambda: 1)): self.assertFalse(self.B.rng_check()[0])
        def seed_ignored(**kw):
            counter = iter(range(16))
            return lambda logits: NS(item=lambda: next(counter))
        with mock.patch.object(self.B, "make_sampler", side_effect=seed_ignored): self.assertFalse(self.B.rng_check()[0])

    def test_cli_rejects_mislabeled_profile_before_writing(self):
        path, output = self.root / "profiles.json", self.root / "output"
        self.put(path, dict(profiles={"16": dict(actor=dict(descriptor=self.d))}))
        argv = [self.B.__file__, "--run", "--profiles", str(path), "--source", "actor", "--bits", "16",
                "--configs", json.dumps([self.config]), "--output", str(output), "--lock", str(self.root / "lock")]
        with mock.patch.object(self.B.sys, "argv", argv), self.assertRaisesRegex(ValueError, "Profile label"):
            self.B.main()
        self.assertFalse(output.exists()); self.assertFalse(output.with_name(output.name + "-inputs.json").exists())

    def test_helper_bytes_are_checked_before_execution(self):
        path = self.root / "src/data-pipeline/mlx_worker.py"
        path.parent.mkdir(parents=True); path.write_text("raise AssertionError('unverified helper executed')")
        with mock.patch.object(self.B, "ROOT", self.root), self.assertRaisesRegex(RuntimeError, "before import"):
            self.B.run_job(dict(source_hashes={str(path): "0" * 64}), "a" * 64, self.emit)

    def test_expanded_panel_preserves_legacy_and_binds_full_batch_qualification(self):
        legacy = self.B.build_job(self.d, [self.config], wall_seconds=10)
        explicit = self.B.build_job(self.d, [self.config], wall_seconds=10, panel_size=8)
        self.assertEqual({k: v for k, v in legacy.items() if k != "source_hashes"}, {k: v for k, v in explicit.items() if k != "source_hashes"})
        for batch, count, caps in ((8, 79, 8491), (16, 87, 8619), (32, 103, 8875)):
            config = dict(batch_size=batch, prefill_batch_size=4, prefill_step_size=256)
            job = self.B.build_job(self.d, [config], wall_seconds=10, panel_size=32)
            self.assertEqual([f["messages"] for f in job["fixtures"]], [f["messages"] for f in legacy["fixtures"]] * 4)
            self.assertEqual(len({f["id"] for f in job["fixtures"]}), 32); self.assertEqual(len({f["seed"] for f in job["fixtures"]}), 32)
            qualification = [w for w in job["windows"] if w["mode"] == "qualification"]
            self.assertEqual(sum(len(w["rows"]) for w in qualification), batch + 6)
            q2 = next(w for w in qualification if w["id"] == "q2")
            self.assertEqual(q2["config"], dict(batch_size=batch, prefill_batch_size=batch, prefill_step_size=256))
            self.assertEqual([r["fixture_id"] for r in q2["rows"]], [job["fixtures"][i]["id"] for i in (0, 4)] * (batch // 2))
            measured = [w for w in job["windows"] if w["mode"] == "measure"]
            self.assertEqual(len(measured), 2)
            for window in measured:
                self.assertEqual(window["config"], config)
                self.assertEqual([r["fixture_id"] for r in window["rows"]], [f["id"] for f in job["fixtures"]])
                self.assertEqual([r["seed"] for r in window["rows"]], [f["seed"] for f in job["fixtures"]])
            self.assertEqual(sum(len(w["rows"]) for w in job["windows"]), count)
            self.assertEqual(sum(r["cap"] for w in job["windows"] for r in w["rows"]), caps)

    def test_expanded_job_refuses_mixed_panels_capacity_counts_and_configs(self):
        parent = importlib.import_module("experiments.batch_throughput")
        config = dict(batch_size=16, prefill_batch_size=4, prefill_step_size=256)
        for panel in (True, 8.0, 9, 31):
            with self.subTest(panel=panel), self.assertRaises(ValueError): self.B.build_job(self.d, [config], panel_size=panel)
        for panel, configs in ((8, [config]), (32, [self.config]), (32, [config, {**config, "batch_size": 8}])):
            with self.subTest(panel=panel, configs=configs), self.assertRaises(ValueError): self.B.build_job(self.d, configs, panel_size=panel)
        good = self.B.build_job(self.d, [config], panel_size=32)
        for fault in ("panel", "capacity", "count", "measure_rows", "measure_config"):
            job = copy.deepcopy(good); q2 = next(w for w in job["windows"] if w["id"] == "q2"); last = job["windows"][-1]
            if fault == "panel": job["fixtures"].pop()
            if fault == "capacity": q2["config"].update(batch_size=8, prefill_batch_size=8)
            if fault == "count": q2["rows"].pop()
            if fault == "measure_rows": last["rows"].pop()
            if fault == "measure_config": last["config"]["prefill_step_size"] = 2048
            with self.subTest(fault=fault), self.assertRaises(ValueError): parent._shape(job)

    def test_full_q2_oracle_rejects_late_raw_text_and_finish_mismatches(self):
        row = lambda token, count, reason="length": dict(tokens=[token] * count, text=f"decoded-{token}-{count}", finish_reason=reason)
        results = dict(q0=[row(3, 16)], q1=[row(4, 16)], q2=[row(3 if i % 2 == 0 else 4, 16) for i in range(32)],
            q3=[dict(tokens=[3, 9], text="decoded-3", finish_reason="stop"), row(3, 3), row(3, 3)], q4=[row(3, 3)])
        self.assertTrue(all(self.B.qualification_checks(results, True).values()))
        for field, value in (("tokens", [7] * 16), ("text", "wrong late decode"), ("finish_reason", "stop")):
            changed = copy.deepcopy(results); changed["q2"][-1][field] = value
            with self.subTest(field=field): self.assertFalse(self.B.qualification_checks(changed, True)["greedy_equal"])
