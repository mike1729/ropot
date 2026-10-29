"""Native adapter contracts, without loading MLX or running generated code."""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

g = importlib.import_module("src.data-pipeline.stage03.local_generation")


def request(text="A", *, cap=2, seed=17):
    return dict(messages=[dict(role="user", content=text)], max_tokens=cap,
                seed=seed, temperature=.7, top_p=.9)


def event(uid, token=12, reason=None):
    return NS(uid=uid, token=token, finish_reason=reason,
              logprobs=object(), prompt_cache=object(), all_tokens=object())


@dataclass
class Stats:
    prompt_tokens: int = 0
    generation_tokens: int = 0
    generation_steps: int = 0
    prompt_time: float = .1
    decode_time: float = .2
    wall_time: float = .4
    peak_memory: float = .01


class Detokenizer:
    def __init__(self):
        self.reset()
        self.finalized = False

    def reset(self):
        self.text, self.tokens, self.offset = "", [], 0
        self.buffered = False

    def add_token(self, token):
        self.tokens.append(token)
        if token == 20:
            self.buffered = True
        elif token == 21:
            self.text += "ż"
            self.buffered = False
        else:
            self.text += {10: "a", 11: "b", 12: "print(1)\n", 99: "<eos>"}[token]

    def finalize(self):
        self.finalized = True
        if self.buffered:
            self.text += "�"
            self.buffered = False

    @property
    def last_segment(self):
        segment, self.offset = self.text[self.offset:], len(self.text)
        return segment


class Tokenizer:
    eos_token_ids = [99, 98]
    _tokenizer = NS()

    def __init__(self):
        self.detokenizers = []
        self.apply_chat_template = mock.Mock(side_effect=lambda messages, tokenize, **kw:
            [1, ord(messages[0]["content"][0]), 2] if tokenize else messages[0]["content"] + "<think>\n\n</think>\n\n")

    @property
    def detokenizer(self):
        result = Detokenizer()
        self.detokenizers.append(result)
        return result


class Native:
    """Model native identity, queue commitment, completion and scalar statistics."""
    def __init__(self, owner, model, **settings):
        self.owner, self.settings, self.stream = owner, settings, object()
        self.active, self.inputs, self.insertions, self.removals = {}, {}, [], []
        self.counters, self.prefilled = Stats(), set()
        self.next_calls, self.closed, self.max_active = 0, False, 0
        self.remove_error = self.close_error = None
        owner.native = self

    def insert(self, prompts, max_tokens=None, samplers=None, **kwargs):
        self.insertions.append((prompts, max_tokens, samplers, kwargs))
        if self.owner.insert_failure == "before":
            raise ValueError("precommit insertion")
        uids = [41 + 7 * (len(self.inputs) + i) for i in range(len(prompts))]
        for uid, ids, cap in zip(uids, prompts, max_tokens):
            self.inputs[uid] = list(ids)
            self.active[uid] = [cap, 0]
        self.max_active = max(self.max_active, len(self.active))
        if self.owner.insert_failure == "after":
            raise KeyboardInterrupt("postcommit insertion")
        return self.owner.uid_result(uids) if self.owner.uid_result else uids

    @contextmanager
    def stats(self):
        yield self.counters

    def next(self):
        self.next_calls += 1
        if self.owner.steps:
            step = self.owner.steps.pop(0)
            if isinstance(step, BaseException):
                raise step
            prompts, generated = step(self) if callable(step) else step
        else:
            prompts = [NS(uid=uid, progress=(2, 2), end_of_segment=True, end_of_prompt=True)
                       for uid in list(self.active)[:self.settings["prefill_batch_size"]] if uid not in self.prefilled]
            generated = []
            for uid, (cap, count) in reversed(list(self.active.items())):
                token = 12 if count == 0 else 99
                reason = "length" if cap == 1 else ("stop" if token == 99 else None)
                generated.append(event(uid, token, reason))
        for response in prompts:
            if response.uid in self.inputs and response.uid not in self.prefilled:
                self.counters.prompt_tokens += len(self.inputs[response.uid]) - 1
                self.prefilled.add(response.uid)
        # Native prefill is counted even when a script omits its progress event.
        for response in generated:
            if response.uid in self.inputs and response.uid not in self.prefilled:
                self.counters.prompt_tokens += len(self.inputs[response.uid]) - 1
                self.prefilled.add(response.uid)
            self.counters.generation_tokens += 1
            if response.uid in self.active:
                self.active[response.uid][1] += 1
                if response.finish_reason in ("stop", "length"):
                    self.active.pop(response.uid)
        self.counters.generation_steps += bool(generated)
        return prompts, generated

    def remove(self, uids, **kwargs):
        self.removals.append(list(uids))
        if self.remove_error:
            raise self.remove_error
        for uid in uids:
            self.active.pop(uid, None)
        return {}

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class BatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name, content in {"config.json": '{"quantization":{"bits":8}}', "manifest.json": '{}',
                "tokenizer.json": '{}', "tokenizer_config.json": '{}', "chat_template.jinja": 'fixture'}.items():
            (self.root / name).write_text(content)
        self.tokenizer, self.native = Tokenizer(), None
        self.steps, self.insert_failure, self.uid_result = [], None, None
        self.load = mock.Mock(return_value=(object(), self.tokenizer))
        self.samplers = mock.Mock(side_effect=lambda **kw: object())
        self.cache_limit, self.cache_limits = 137, []
        def cache_limit(value):
            previous, self.cache_limit = self.cache_limit, value
            self.cache_limits.append(value)
            return previous
        self.mx = NS(set_cache_limit=mock.Mock(side_effect=cache_limit), reset_peak_memory=mock.Mock(),
                     get_peak_memory=mock.Mock(return_value=12345), synchronize=mock.Mock())
        modules = {"mlx": NS(core=self.mx), "mlx.core": self.mx,
                   "mlx_lm": NS(load=self.load),
                   "mlx_lm.sample_utils": NS(make_sampler=self.samplers, __file__=__file__),
                   "mlx_lm.generate": NS(BatchGenerator=lambda *a, **kw: Native(self, *a, **kw), __file__=__file__)}
        for name in ("tokenizer_utils", "models.qwen3_5", "models.mistral3", "models.ministral3",
                     "models.cache", "generate_utils"):
            source = self.root / f"{name}.py"
            source.write_text(name)
            modules[f"mlx_lm.{name}"] = NS(__file__=str(source))
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict("sys.modules", modules).start()
        mock.patch.object(g, "version", return_value="fixture").start()

    def generate(self, requests=None, name="run", **settings):
        return g.generate_batched(self.root, requests if requests is not None else [request()], self.root / name,
            model_manifest=self.root / "manifest.json",
            **(dict(completion_batch_size=2, prefill_batch_size=1, prefill_step_size=256) | settings))

    def saved(self, name="run"):
        return json.loads((self.root / name / "outcome.json").read_text())

    def test_prompt_envelope_checks_entire_round_before_native_submission(self):
        result = self.generate([request(), dict(request("B"), max_prompt_tokens=2)])
        self.assertEqual(result["status"], "error")
        self.assertIn("qualified token envelope", result["error"])
        self.assertTrue(all(not r["submitted"] for r in result["requests"]))
        self.assertEqual(result["requests"][1]["prompt_tokens"], 3)
        self.assertIsNone(self.native)

    def test_prompt_envelope_accepts_its_exact_boundary(self):
        result = self.generate([dict(request(), max_prompt_tokens=3)])
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(result["requests"][0]["submitted"])

    def test_ordered_association_bounded_refill_fresh_state_and_scalar_evidence(self):
        requests = [request("A", seed=17), request("B", seed=18), request("A", seed=17, cap=1)]
        result = self.generate(requests)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual([r["native_uid"] for r in result["requests"]], [41, 48, 55])
        self.assertEqual([r["status"] for r in result["requests"]], ["stop", "stop", "length"])
        self.assertEqual([r["tokens"] for r in result["requests"]], [[12, 99], [12, 99], [12]])
        self.assertEqual([r["text"] for r in result["requests"]], ["print(1)\n"] * 3)
        self.assertEqual(self.native.max_active, 2)
        self.assertEqual([len(call[0]) for call in self.native.insertions], [2, 1])
        self.assertEqual(self.native.settings["stop_tokens"], [[98], [99]])
        self.assertEqual(self.native.settings["prefill_step_size"], 256)
        self.assertTrue(all(not call[3].get("caches") for call in self.native.insertions))
        self.assertEqual(self.samplers.call_args_list, [mock.call(temp=.7, top_p=.9, seed=s) for s in (17, 18, 17)])
        submitted = [s for call in self.native.insertions for s in call[2]]
        self.assertEqual(len({id(s) for s in submitted}), 3)
        self.assertEqual(len(self.tokenizer.detokenizers), 3)
        self.assertTrue(all(d.finalized for d in self.tokenizer.detokenizers))
        for i, row in enumerate(result["requests"]):
            self.assertTrue(row["submitted"])
            self.assertEqual(row["prompt_tokens"], len(row["prompt_ids"]))
            self.assertEqual(row["generation_tokens"], len(row["tokens"]))
            self.assertEqual((self.root / f"run/reply-{i:04d}.txt").read_text(), row["text"])
            self.assertNotIn("prompt_cache", row)
        self.assertEqual(result["native"]["prompt_tokens"], 6)
        self.assertEqual(result["native"]["generation_tokens"], 5)
        self.assertEqual(result["max_observed_decode_batch"], 2)
        self.assertNotIn("max_observed_prefill_batch", result)
        self.assertEqual(self.cache_limits[0], 4 * 1024**3)
        self.assertEqual(result["allocator_cache_limit_bytes"], 4 * 1024**3)
        self.assertEqual(result["previous_allocator_cache_limit_bytes"], 137)
        self.assertTrue(all(value == 137 for value in self.cache_limits[1:]))
        self.assertEqual(self.cache_limit, 137)
        self.assertTrue(self.native.closed)
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.mx.reset_peak_memory.assert_called_once()
        self.assertEqual(self.saved(), result)
        runtime = json.loads((self.root / "run/runtime.json").read_text())
        self.assertIn(g.file_hash(__file__), runtime["source_sha256"].values())
        for name in ("models.cache", "generate_utils"):
            source = self.root / f"{name}.py"
            self.assertEqual(runtime["source_sha256"][str(source)], g.file_hash(source))
        self.load.assert_called_once()
        for call in self.tokenizer.apply_chat_template.call_args_list:
            self.assertTrue(call.kwargs["add_generation_prompt"])
            self.assertFalse(call.kwargs["enable_thinking"])

    def test_validation_happens_before_creating_output_or_loading(self):
        invalid = [("completion_batch_size", 0), ("completion_batch_size", True),
                   ("prefill_batch_size", 3), ("prefill_batch_size", 1.5),
                   ("prefill_step_size", False), ("prefill_step_size", -1),
                   ("max_tokens", 0), ("max_tokens", True), ("max_tokens", 1.5),
                   ("max_prompt_tokens", 0), ("max_prompt_tokens", True), ("max_prompt_tokens", 1.5)]
        for i, (key, value) in enumerate(invalid):
            req, settings = request(), {}
            (req if key.endswith("tokens") else settings)[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.generate([req], f"invalid-{i}", **settings)
            self.assertFalse((self.root / f"invalid-{i}").exists())
        self.load.assert_not_called()

    def test_wall_time_field_is_optional_and_legacy_value_does_not_expire_requests(self):
        clock = [0.]
        def later(events):
            def step(native):
                clock[0] += 10.
                return [], events
            return step
        self.steps = [later([event(41, 10), event(48, 11)]),
                      later([event(41, 99, "stop"), event(48, 99, "stop")]),
                      later([event(55, 12, "length")])]
        requests = [request(), request("B"), request("C", cap=1)]
        requests[1]["wall_time_seconds"] = .001
        self.assertNotIn("wall_time_seconds", requests[0])
        with mock.patch.object(g, "monotonic", side_effect=lambda: clock[0]):
            result = self.generate(requests)
        self.assertEqual(result["status"], "success", result)
        self.assertEqual([r["status"] for r in result["requests"]], ["stop", "stop", "length"])
        self.assertEqual([r["elapsed_seconds"] for r in result["requests"]], [20., 20., 10.])
        self.assertEqual(self.native.next_calls, 3)
        self.assertTrue(all(r["submitted"] for r in result["requests"]))

    def test_errors_and_interruptions_preserve_completed_partial_and_undispatched_rows(self):
        for failure, status in ((RuntimeError("native failure"), "error"), (KeyboardInterrupt("stop"), "cancelled")):
            with self.subTest(status=status):
                self.steps = [([], [event(41, 99, "stop"), event(48, 20)]), failure]
                if isinstance(failure, Exception):
                    result = self.generate([request(), request("B"), request("C"), request("D")], status)
                else:
                    with self.assertRaises(KeyboardInterrupt):
                        self.generate([request(), request("B"), request("C"), request("D")], status)
                    result = self.saved(status)
                self.assertEqual(result["status"], status)
                self.assertEqual([r["status"] for r in result["requests"]], ["stop", status, status, "not_run"])
                self.assertEqual(result["requests"][1]["text"], "�")
                self.assertEqual(self.native.removals, [[48, 55]])
                self.assertFalse(self.native.active)
                self.assertTrue(self.native.closed)
                self.assertEqual(self.cache_limit, 137)
                self.assertEqual(self.native.next_calls, 2)

    def test_buffered_unicode_decodes_once_at_completion(self):
        def after_first(native):
            self.assertFalse((self.root / "run/reply-0000.txt").exists())
            self.assertFalse(self.tokenizer.detokenizers)
            return [], [event(41, 20)]
        self.steps = [([], [event(41, 10)]), after_first, ([], [event(41, 21)]),
                      ([], [event(41, 99, "stop")])]
        result = self.generate([request(cap=4)])
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["requests"][0]["text"], "aż")
        self.assertEqual(result["requests"][0]["tokens"], [10, 20, 21, 99])
        self.assertEqual(len(self.tokenizer.detokenizers), 1)
        self.assertTrue(self.tokenizer.detokenizers[0].finalized)

    def test_decoding_strips_only_terminal_eos_and_finalizes_partial_unicode(self):
        self.assertEqual(g.decode_tokens(self.tokenizer, [10, 99, 11, 99], {99}), "a<eos>b")
        self.assertEqual(g.decode_tokens(self.tokenizer, [10, 20], {99}), "a�")
        self.assertEqual(g.decode_tokens(self.tokenizer, [10, 20, 21], {99}), "aż")
        self.assertTrue(all(decoder.finalized for decoder in self.tokenizer.detokenizers))

    def test_completed_reply_is_written_before_next_native_call(self):
        def after_completion(native):
            self.assertEqual((self.root / "run/reply-0000.txt").read_text(), "a")
            self.assertFalse((self.root / "run/reply-0001.txt").exists())
            return [], [event(48, 99, "stop")]
        self.steps = [([], [event(41, 10, "length"), event(48, 20)]), after_completion]
        result = self.generate([request(cap=1), request("B")])
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["requests"][1]["text"], "�")

    def test_reply_handles_are_bounded_for_a_long_worklist(self):
        original, handles, peak = Path.open, [], 0
        def tracked_open(path, *args, **kwargs):
            nonlocal peak
            handle = original(path, *args, **kwargs)
            if path.name.startswith("reply-"):
                handles.append(handle)
                peak = max(peak, sum(not opened.closed for opened in handles))
            return handle
        with mock.patch.object(Path, "open", tracked_open):
            result = self.generate([request() for _ in range(31)])
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(len(handles), 31)
        self.assertEqual(peak, 1)
        self.assertTrue(all(handle.closed for handle in handles))

    def test_reply_write_failure_keeps_all_call_tokens_and_persists_other_rows(self):
        original, writes = Path.write_text, []
        def fail_reply(path, *args, **kwargs):
            if path.name.startswith("reply-"):
                writes.append(path.name)
            if path.name == "reply-0000.txt":
                raise OSError("reply write failed")
            return original(path, *args, **kwargs)
        self.steps = [([], [event(41, 12), event(48, 12)]),
                      ([], [event(41, 99, "stop"), event(48, 99, "stop")])]
        with mock.patch.object(Path, "write_text", fail_reply):
            result = self.generate([request(), request("B")])
        self.assertEqual(result["status"], "error")
        self.assertIn("reply write failed", result["error"])
        self.assertIn("reply write failed", result["cleanup_error"])
        self.assertNotIn("KeyError", result["cleanup_error"])
        self.assertEqual([row["status"] for row in result["requests"]], ["error", "stop"])
        self.assertEqual([row["finish_reason"] for row in result["requests"]], ["stop", "stop"])
        self.assertEqual([row["tokens"] for row in result["requests"]], [[12, 99], [12, 99]])
        self.assertEqual([row["generation_tokens"] for row in result["requests"]], [2, 2])
        self.assertEqual(writes, ["reply-0000.txt", "reply-0001.txt", "reply-0000.txt"])
        self.assertEqual((self.root / "run/reply-0001.txt").read_text(), "print(1)\n")
        self.assertEqual(g.extract_program(result["requests"][1]), b"print(1)\n")
        self.assertEqual(self.native.removals, [[]])
        self.assertTrue(self.native.closed)
        self.assertFalse(self.native.active)
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.assertEqual(self.cache_limit, 137)

    def test_decode_interruption_preserves_entire_call_before_retrying_persistence(self):
        decode = g.decode_tokens
        def interrupt_once(*args):
            if decoder.call_count == 1:
                raise KeyboardInterrupt("decode interrupted")
            if decoder.call_count == 3:
                self.assertEqual((self.root / "run/reply-0001.txt").read_text(), "print(1)\n")
            return decode(*args)
        with mock.patch.object(g, "decode_tokens", side_effect=interrupt_once) as decoder:
            self.steps = [([], [event(41, 12, "length"), event(48, 12, "length")])]
            with self.assertRaisesRegex(KeyboardInterrupt, "decode interrupted"):
                self.generate([request(cap=1), request("B", cap=1)])
        result = self.saved()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([row["status"] for row in result["requests"]], ["cancelled", "length"])
        self.assertEqual([row["finish_reason"] for row in result["requests"]], ["length", "length"])
        self.assertEqual([row["tokens"] for row in result["requests"]], [[12], [12]])
        self.assertEqual([row["generation_tokens"] for row in result["requests"]], [1, 1])
        self.assertEqual(decoder.call_count, 3)
        self.assertEqual([(self.root / f"run/reply-{i:04d}.txt").read_text() for i in range(2)], ["print(1)\n"] * 2)
        self.assertEqual(self.native.removals, [[]])
        self.assertFalse(self.native.active)
        self.assertTrue(self.native.closed)
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.assertEqual(self.cache_limit, 137)

    def test_middle_persistence_failure_preserves_finished_peers_and_stops_admission(self):
        original = Path.write_text
        for failure in (OSError("middle write failed"), KeyboardInterrupt("middle write interrupted")):
            name, writes = type(failure).__name__, []
            def fail_middle(path, *args, **kwargs):
                if path.name.startswith("reply-"):
                    writes.append(path.name)
                if path.name == "reply-0001.txt":
                    raise failure
                return original(path, *args, **kwargs)
            self.steps = [([], [event(uid, 12) for uid in (41, 48, 55, 62)]),
                ([], [event(41, 99, "stop"), event(48, 10, "length"),
                      event(55, 99, "stop"), event(62, 20)])]
            requests = [request("A"), request("B"), request("C"), request("D", cap=3), request("E")]
            with self.subTest(failure=name), mock.patch.object(Path, "write_text", fail_middle):
                if isinstance(failure, Exception):
                    result = self.generate(requests, name, completion_batch_size=4)
                else:
                    with self.assertRaisesRegex(KeyboardInterrupt, "middle write interrupted"):
                        self.generate(requests, name, completion_batch_size=4)
                    result = self.saved(name)
                status = "error" if isinstance(failure, Exception) else "cancelled"
                self.assertEqual([row["status"] for row in result["requests"]],
                                 ["stop", status, "stop", status, "not_run"])
                self.assertEqual([row["tokens"] for row in result["requests"]],
                                 [[12, 99], [12, 10], [12, 99], [12, 20], []])
                self.assertEqual([row["finish_reason"] for row in result["requests"]],
                                 ["stop", "length", "stop", None, None])
                self.assertEqual(writes, [f"reply-{i:04d}.txt" for i in (0, 1, 2, 1, 3)])
                for i in (0, 2):
                    self.assertEqual(g.extract_program(result["requests"][i]), b"print(1)\n")
                self.assertEqual((self.root / name / "reply-0003.txt").read_text(), "print(1)\n�")
                self.assertFalse((self.root / name / "reply-0004.txt").exists())
                self.assertEqual(self.native.removals, [[62]])
                self.assertFalse(self.native.active)
                self.assertTrue(self.native.closed)
                self.mx.synchronize.assert_any_call(self.native.stream)
                self.assertEqual(self.cache_limit, 137)

    def test_bad_decode_identity_reason_and_exhaustion_stop_run(self):
        cases = [("unknown", [([], [event(999)])]),
                 ("reason", [([], [event(41, reason="unexpected")])]),
                 ("premature", [([], [])]),
                 ("postcompletion", [([], [event(41, 99, "stop")]), ([], [event(41, 11)])])]
        for name, steps in cases:
            with self.subTest(name=name):
                self.steps = steps
                result = self.generate([request(), request("B")], name)
                self.assertEqual(result["status"], "error", result)
                self.assertIsNotNone(result["error"])
                self.assertTrue(self.native.closed)
        self.uid_result = lambda uids: uids[:-1]
        result = self.generate([request(), request("B")], "missing")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.native.next_calls, 0)

    def test_malformed_call_is_rejected_atomically_with_scalar_evidence(self):
        cases = {"unknown": [event(41, 10), event(999, 11), event(48, 12)],
                 "duplicate": [event(41, 10), event(41, 11)],
                 "reason": [event(41, 10), event(48, 11, "unexpected")]}
        for name, events in cases.items():
            with self.subTest(name=name):
                self.steps = [([], [event(41, 12), event(48, 12)]), ([], events)]
                result = self.generate([request(), request("B")], name)
                self.assertEqual(result["status"], "error")
                self.assertEqual([row["tokens"] for row in result["requests"]], [[12], [12]])
                self.assertEqual([row["generation_tokens"] for row in result["requests"]], [1, 1])
                self.assertEqual(result["rejected_events"], [dict(uid=e.uid, token=e.token,
                    finish_reason=e.finish_reason) for e in events])
                self.assertNotIn("unprocessed_events", result)
                self.assertEqual(result["native"]["generation_tokens"], 2 + len(events))
                self.assertEqual(self.saved(name), result)

    def test_uidless_insert_failure_before_or_after_native_queue_commit(self):
        for stage in ("before", "after"):
            self.insert_failure = stage
            if stage == "after":
                with self.assertRaises(KeyboardInterrupt):
                    self.generate([request(), request("B"), request("C")], stage)
                result = self.saved(stage)
            else:
                result = self.generate([request(), request("B"), request("C")], stage)
            self.assertEqual([r["submitted"] for r in result["requests"]], [True, True, False])
            self.assertEqual([r.get("native_uid") for r in result["requests"]], [None] * 3)
            self.assertEqual(self.native.next_calls, 0)
            self.assertFalse(any(uid is None for call in self.native.removals for uid in call))
            self.assertTrue(self.native.closed)
            self.assertEqual(self.cache_limit, 137)
            self.assertFalse(result.get("cleanup_error"), result)

    def test_native_statistics_must_reconcile_on_success(self):
        def corrupt(native):
            native.counters.generation_tokens += 1
            return [], [event(41, 99, "stop")]
        self.steps = [corrupt]
        result = self.generate()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requests"][0]["tokens"], [99])
        self.assertEqual(result["requests"][0]["text"], "")

    def test_cleanup_failure_keeps_primary_error_and_interrupt_still_propagates(self):
        for failure in (RuntimeError("primary"), KeyboardInterrupt("primary")):
            name = type(failure).__name__
            def fail(native):
                native.remove_error = RuntimeError("cleanup removal")
                raise failure
            self.steps = [fail]
            if isinstance(failure, Exception):
                result = self.generate(name=name)
            else:
                with self.assertRaises(KeyboardInterrupt):
                    self.generate(name=name)
                result = self.saved(name)
            self.assertEqual(result["status"], "error")
            self.assertIn("primary", result["error"])
            self.assertIn("cleanup removal", result["cleanup_error"])
            self.assertTrue(self.native.closed)
            self.mx.synchronize.assert_any_call(self.native.stream)
            self.assertEqual(self.cache_limit, 137)

    def test_close_failure_invalidates_success_without_losing_completed_rows(self):
        def completed(native):
            native.close_error = RuntimeError("close failed")
            return [], [event(41, 99, "stop")]
        self.steps = [completed]
        result = self.generate()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requests"][0]["status"], "stop")
        self.assertIn("close failed", result["cleanup_error"])
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.assertEqual(self.cache_limit, 137)

    def test_load_failure_restores_allocator_and_preserves_unsubmitted_rows(self):
        self.load.side_effect = RuntimeError("load failed")
        result = self.generate()
        self.assertEqual(result["status"], "error")
        self.assertIn("load failed", result["error"])
        self.assertEqual(result["requests"][0]["status"], "not_run")
        self.assertFalse(result["requests"][0]["submitted"])
        self.assertEqual(self.cache_limits[0], 4 * 1024**3)
        self.assertTrue(all(value == 137 for value in self.cache_limits[1:]))
        self.assertEqual(self.cache_limit, 137)

    def test_interruption_during_cleanup_propagates_after_evidence_and_other_cleanup(self):
        def completed(native):
            native.close_error = KeyboardInterrupt("cleanup interruption")
            return [], [event(41, 99, "stop")]
        self.steps = [completed]
        with self.assertRaises(KeyboardInterrupt):
            self.generate()
        result = self.saved()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requests"][0]["status"], "stop")
        self.assertIn("cleanup interruption", result["cleanup_error"])
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.assertEqual(self.cache_limit, 137)

    def test_cleanup_interruption_does_not_replace_primary_interruption(self):
        def interrupted(native):
            native.remove_error = KeyboardInterrupt("cleanup interruption")
            raise KeyboardInterrupt("primary interruption")
        self.steps = [interrupted]
        with self.assertRaisesRegex(KeyboardInterrupt, "primary interruption"):
            self.generate()
        result = self.saved()
        self.assertIn("primary interruption", result["error"])
        self.assertIn("cleanup interruption", result["cleanup_error"])
        self.assertTrue(self.native.closed)
        self.mx.synchronize.assert_any_call(self.native.stream)
        self.assertEqual(self.cache_limit, 137)


class ExtractionTests(unittest.TestCase):
    def extract(self, text, status="stop"):
        row = {"status": status, "text": text}
        result = g.extract_program(row)
        self.assertEqual(row, {"status": status, "text": text})
        return result

    def test_exact_plain_bytes_and_fenced_slices(self):
        for code in ("print('żółw')\n", "\t# keep whitespace\r\nprint(1)\r\n", "\ufeffprint(1)\n",
                     "# coding: latin-1\nvalue = 'ż'\n", "value = '```python'\n"):
            with self.subTest(code=code):
                self.assertEqual(self.extract(code), code.encode("utf-8"))
                for label in (() if "```" in code else ("python", "py", "")):
                    self.assertEqual(self.extract(f"Prose.\r\n```{label}\t\r\n{code}``` \t\r\nDone."), code.encode("utf-8"))

    def test_syntax_check_compiles_bytes_including_bom_and_coding_cookie(self):
        code = "\ufeffprint('ż')\n"
        self.assertEqual(self.extract(code), code.encode())
        # str compilation accepts this conflicting declaration; bytes compilation rejects it.
        for text in ("\ufeff# coding: latin-1\nprint(1)\n", "# coding: ascii\nprint('ż')\n"):
            compile(text.lstrip("\ufeff"), "<str-control>", "exec")
            with self.subTest(text=text), self.assertRaises(SyntaxError):
                self.extract(text)

    def test_successful_plain_program_is_compiled_once(self):
        with mock.patch("builtins.compile", wraps=compile) as compiler:
            self.assertEqual(self.extract("print(1)\n"), b"print(1)\n")
        compiler.assert_called_once_with(b"print(1)\n", "<candidate>", "exec")

    def test_parser_resource_failures_propagate_as_documented(self):
        code = "-" * 200_000 + "1"
        for text in (code, f"```python\n{code}\n```"):
            with self.subTest(fenced=text.startswith("```")), self.assertRaises((MemoryError, RecursionError)):
                self.extract(text)

    def test_nonwhitespace_and_fence_format_failures_are_value_errors(self):
        texts = ("", " \t\r\n", "```python\n \t\n```", "```python\nprint(1)",
                 "```Python\nprint(1)\n```", "```javascript\nprint(1)\n```",
                 " ```python\nprint(1)\n```", "```python\nprint(1)\n ```",
                 "```python\nprint(1)\n```\n```py\nprint(2)\n```", "prose ``` prose",
                 "```python\nvalue = '```'\n```")
        for text in texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.extract(text)

    def test_syntax_errors_and_status_rejection_precede_content_inspection(self):
        for text in ("print(", "```python\nprint(\n```", "Here is Python code:"):
            with self.subTest(text=text), self.assertRaises(SyntaxError):
                self.extract(text)
        for status in ("length", "cancelled", "error", "running", "not_run"):
            for text in ("print(1)", "print(", "```python\nprint(1)\n```"):
                with self.subTest(status=status), self.assertRaises(ValueError):
                    self.extract(text, status)
