"""Offline DeepInfra round transport and partial-accounting contracts."""
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error

from scripts import deepinfra_generation as d, native_multi_edit
from tests.test_thinking_generation import ThinkingTokenizer

b = importlib.import_module("src.data-pipeline.stage03.generation_loop")


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.edit = '<tool_call><function=test_edits><parameter=edits>[{"old":"1","new":"2"}]</parameter></function></tool_call>'
        self.tokenizer = ThinkingTokenizer("qwen")
        self.tokenizer.pieces = self.tokenizer.pieces.copy()
        self.tokenizer.pieces[20] = self.edit
        self.spec = dict(runtime=dict(model="Qwen/Qwen3.8-27B:61nNTQpO", tokenizer_path=str(self.root / "tokenizer"),
            tokenizer_sha256={}, timeout_seconds=5, max_total_tokens=100,
            input_usd_per_million=.2, output_usd_per_million=2.5, max_cost_usd=.01),
            requests=[self.request()], output_dir=str(self.root / "run"))
        self.loader = mock.patch.object(d, "load_tokenizer", return_value=self.tokenizer).start()
        self.send = mock.patch.object(d, "send").start()
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict("os.environ", DEEPINFRA_API_KEY="PRIVATE_TEST_KEY").start()

    def request(self, **changes):
        request = dict(messages=[dict(role="user", content="Return a program")], seed=17,
                       temperature=1., top_p=.95, max_tokens=20)
        request.update(changes)
        return request

    def response(self, tokens=None, reason="stop", prompt_tokens=1, **changes):
        tokens = [12, 99] if tokens is None else tokens
        text = "".join(self.tokenizer.pieces[t] for t in tokens)
        body = dict(model="Qwen/Qwen3.8-27B", choices=[dict(index=0, text=text,
                    token_ids=tokens, finish_reason=reason)], usage=dict(prompt_tokens=prompt_tokens,
                    completion_tokens=len(tokens), total_tokens=prompt_tokens + len(tokens),
                    estimated_cost=.0001, prompt_tokens_details=dict(cached_tokens=0)))
        body.update(changes)
        return 200, json.dumps(body)

    def run_round(self):
        path = self.root / "round.json"
        path.write_text(json.dumps(self.spec))
        return d.run(path)

    def saved(self):
        return json.loads((Path(self.spec["output_dir"]) / "outcome.json").read_text())

    def test_benign_edit_and_repair_keep_order_and_reuse_existing_parsers(self):
        history = [dict(role="user", content="Change the program"),
                   dict(role="assistant", content="", reasoning_content="Prior thought",
                        tool_calls=[dict(id="old", type="function", function=dict(name="test_edits",
                        arguments=dict(edits=[dict(old="0", new="1")])))]),
                   dict(role="tool", tool_call_id="old", content="Rejected; repair the edit")]
        repair = self.request(enable_thinking=True, reasoning_effort="low", tools=native_multi_edit.TOOLS,
                              preserve_thinking=True, top_k=20)
        repair["messages"] = history
        self.spec["requests"] = [self.request(), self.request(), repair]
        self.send.side_effect = [self.response(), self.response([20, 99]),
                                 self.response([10, 31, 20, 99], prompt_tokens=3)]
        result = self.run_round()
        self.assertEqual(result["status"], "success", result)
        self.assertIsNone(result["native"])
        rows = result["requests"]
        self.assertEqual([r["submitted"] for r in rows], [True] * 3)
        self.assertEqual(b.generation.extract_program(rows[0]), self.tokenizer.pieces[12].encode())
        for row in rows[1:]:
            self.assertEqual(native_multi_edit.parse_call(row["answer_text"])[1], [dict(old="1", new="2")])
        self.assertEqual(rows[2]["tokens"], [10, 31, 20, 99])
        self.assertEqual(rows[2]["reasoning_content"], self.tokenizer.pieces[10])
        self.assertTrue(result["usage_complete"])
        self.assertEqual(self.send.call_count, 3)
        payload = self.send.call_args_list[2].args[0]
        for key, value in dict(model=self.spec["runtime"]["model"], prompt=[1, 30, 2], top_k=20,
                               seed=17, temperature=1., top_p=.95, max_tokens=20,
                               skip_special_tokens=False, return_tokens_as_token_ids=True, echo=False).items():
            self.assertEqual(payload[key], value)
        calls = self.tokenizer.apply_chat_template.call_args_list[-2:]
        for call in calls:
            self.assertEqual(call.args[0], history)
            self.assertEqual(call.kwargs["tools"], native_multi_edit.TOOLS)
            self.assertTrue(call.kwargs["preserve_thinking"])
            self.assertEqual(call.kwargs["reasoning_effort"], "low")
        self.assertEqual(self.saved(), result)

    def test_truncation_and_unclosed_thinking_cannot_supply_programs(self):
        self.spec["requests"] = [self.request(), self.request(enable_thinking=True, reasoning_effort="low")]
        self.send.side_effect = [self.response([12], reason="length"), self.response([12, 99], prompt_tokens=3)]
        result = self.run_round()
        self.assertEqual(result["status"], "success", result)
        self.assertEqual(result["requests"][0]["status"], "length")
        self.assertIsNone(result["requests"][1]["answer_text"])
        for row in result["requests"]:
            with self.assertRaises(ValueError):
                b.generation.extract_program(row)

    def test_timeout_preserves_completed_peer_unknown_submission_and_not_run_tail(self):
        self.spec["requests"] *= 3
        def send(*args):
            if self.send.call_count == 1:
                return self.response()
            persisted = self.saved()
            self.assertIsNone(persisted["requests"][1]["submitted"])
            self.assertEqual(persisted["requests"][1]["status"], "running")
            raise TimeoutError("Request timed out")
        self.send.side_effect = send
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        self.assertEqual([r["submitted"] for r in result["requests"]], [True, None, False])
        self.assertEqual(result["requests"][0]["status"], "stop")
        self.assertFalse(result["usage_complete"])
        self.assertEqual(self.send.call_count, 2)
        self.assertEqual(self.saved(), result)

    def test_interruption_is_cancelled_without_dispatching_tail(self):
        self.spec["requests"] *= 2
        self.send.side_effect = KeyboardInterrupt()
        result = self.run_round()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([r["submitted"] for r in result["requests"]], [None, False])
        self.assertEqual(self.send.call_count, 1)
        self.assertFalse(result["usage_complete"])

    def test_http_error_is_uncertain_and_malformed_success_is_confirmed(self):
        self.spec["requests"] *= 2
        for status, body, submitted in [(503, '{"error":"unavailable"}', None),
                                        (200, '{"choices":[]}', True),
                                        (200, 'not json', True)]:
            with self.subTest(status=status, body=body):
                self.spec["output_dir"] = str(self.root / f"case-{self.send.call_count}")
                self.send.return_value = status, body
                before = self.send.call_count
                result = self.run_round()
                self.assertEqual(result["status"], "error")
                self.assertIs(result["requests"][0]["submitted"], submitted)
                self.assertFalse(result["requests"][1]["submitted"])
                self.assertFalse(result["usage_complete"])
                self.assertEqual(self.send.call_count, before + 1)

    def test_text_mismatch_retains_validated_usage_but_stops_without_an_answer(self):
        self.spec["requests"] *= 2
        status, raw = self.response()
        body = json.loads(raw)
        body["choices"][0]["text"] = "Different provider decoding"
        self.send.return_value = status, json.dumps(body)
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        first, tail = result["requests"]
        self.assertTrue(first["submitted"])
        self.assertEqual(first["tokens"], [12, 99])
        self.assertEqual(first["generation_tokens"], 2)
        self.assertIsNone(first["answer_text"])
        self.assertFalse(tail["submitted"])
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["observed_estimated_cost_usd"], .0001)
        self.assertEqual(self.send.call_count, 1)

    def test_http_error_echoing_key_is_redacted_in_saved_response(self):
        self.send.return_value = 429, '{"error":"PRIVATE_TEST_KEY rate limited"}'
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["requests"][0]["submitted"])
        for path in Path(self.spec["output_dir"]).iterdir():
            if path.is_file():
                self.assertNotIn("PRIVATE_TEST_KEY", path.read_text())

    def test_bad_limits_unknown_fields_and_over_budget_round_never_dispatch(self):
        cases = [lambda s: s["requests"][0].update(max_tokens=0),
                 lambda s: s["requests"][0].update(max_tokens=True),
                 lambda s: s["requests"][0].update(max_prompt_tokens=0),
                 lambda s: s["requests"][0].update(unsupported=True),
                 lambda s: s["runtime"].update(timeout_seconds=0),
                 lambda s: s["runtime"].update(max_total_tokens=20),
                 lambda s: s["runtime"].update(max_cost_usd=.0000001)]
        original = deepcopy(self.spec)
        for change in cases:
            with self.subTest(change=change):
                self.spec = deepcopy(original)
                change(self.spec)
                with self.assertRaises((ValueError, TypeError)):
                    self.run_round()
                self.assertFalse(Path(self.spec["output_dir"]).exists())
                self.send.assert_not_called()

    def test_oversize_later_prompt_is_rejected_before_any_send(self):
        self.spec["requests"] = [self.request(), self.request(max_prompt_tokens=1, enable_thinking=True)]
        with self.assertRaises(ValueError):
            self.run_round()
        self.send.assert_not_called()
        self.assertFalse(Path(self.spec["output_dir"]).exists())

    def test_artifacts_include_raw_response_usage_without_secret(self):
        response = self.response()
        self.send.return_value = response
        result = self.run_round()
        self.assertEqual(result["status"], "success", result)
        output = Path(self.spec["output_dir"])
        expected = {"inputs.json", "runtime.json", "outcome.json", "request-0000.json", "response-0000.json", "reply-0000.txt"}
        self.assertTrue(expected.issubset({p.name for p in output.iterdir()}))
        raw = (output / "response-0000.json").read_text()
        self.assertIn("estimated_cost", raw)
        self.assertIn("cached_tokens", raw)
        for path in output.iterdir():
            if path.is_file():
                self.assertNotIn("PRIVATE_TEST_KEY", path.read_text())
        self.assertEqual(self.send.call_args.args[1:], ("PRIVATE_TEST_KEY", 5))

    def test_exception_message_is_redacted_from_saved_evidence(self):
        self.send.side_effect = OSError("PRIVATE_TEST_KEY failed")
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        for path in Path(self.spec["output_dir"]).iterdir():
            if path.is_file():
                self.assertNotIn("PRIVATE_TEST_KEY", path.read_text())

    def test_missing_key_is_rejected_before_creating_output(self):
        with mock.patch.dict("os.environ", DEEPINFRA_API_KEY=""):
            with self.assertRaisesRegex(ValueError, "DEEPINFRA_API_KEY"):
                self.run_round()
        self.send.assert_not_called()
        self.assertFalse(Path(self.spec["output_dir"]).exists())

    def test_metadata_write_failures_leave_known_unsubmitted_error_outcome(self):
        self.spec["requests"] *= 2
        write_text = Path.write_text
        for name in ("inputs.json", "runtime.json"):
            with self.subTest(name=name):
                self.spec["output_dir"] = str(self.root / name.replace(".json", ""))
                failed = []
                target = Path(self.spec["output_dir"]) / (name + ".tmp")
                def fail_once(path, *args, **kwargs):
                    if path == target and not failed:
                        failed.append(path)
                        raise OSError("Metadata write failed: " + name)
                    return write_text(path, *args, **kwargs)
                with mock.patch.object(Path, "write_text", new=fail_once):
                    result = self.run_round()
                self.assertEqual(failed, [target])
                self.assertEqual(result["status"], "error")
                self.assertIn("Metadata write failed", result["error"])
                self.assertEqual([r["submitted"] for r in result["requests"]], [False, False])
                self.assertTrue(result["usage_complete"])
                self.assertEqual(self.saved(), result)
                self.send.assert_not_called()

    def test_zero_top_k_and_unsigned_seed_boundaries_are_forwarded(self):
        self.send.return_value = self.response()
        for seed in (0, 2**64 - 1):
            with self.subTest(seed=seed):
                self.spec["output_dir"] = str(self.root / f"seed-{seed}")
                self.spec["requests"] = [self.request(seed=seed, top_k=0)]
                result = self.run_round()
                self.assertEqual(result["status"], "success", result)
                payload = self.send.call_args.args[0]
                self.assertEqual(payload["seed"], seed)
                self.assertEqual(payload["top_k"], 0)

    def test_negative_or_overflow_seed_and_negative_top_k_fail_before_dispatch(self):
        for changes in (dict(seed=-1), dict(seed=2**64), dict(top_k=-1)):
            with self.subTest(changes=changes):
                self.spec["requests"] = [self.request(**changes)]
                with self.assertRaises(ValueError):
                    self.run_round()
                self.send.assert_not_called()
                self.assertFalse(Path(self.spec["output_dir"]).exists())


class RecordedCompatibilityTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(__file__).with_name("fixtures") / "deepinfra"
        self.request = json.loads((fixtures / "06-raw-neutral-token-array-request.json").read_bytes())
        self.response = json.loads((fixtures / "06-raw-neutral-token-array-response.json").read_bytes())

    def test_payload_uses_observed_token_id_parameter_and_zero_top_k(self):
        recorded = self.request["payload"]
        tokenizer = ThinkingTokenizer("qwen")
        tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: recorded["prompt"] if tokenize else "Pinned neutral prompt"
        runtime = dict(model=d.MODEL, tokenizer_path="unused", tokenizer_sha256={}, timeout_seconds=5,
                       max_total_tokens=100, input_usd_per_million=.2, output_usd_per_million=2.5, max_cost_usd=.01)
        request = {k: recorded[k] for k in ("seed", "temperature", "top_p", "top_k", "max_tokens")}
        request["messages"] = [dict(role="user", content="ALPHA BETA GAMMA")]
        with mock.patch.object(d, "load_tokenizer", return_value=tokenizer):
            _, _, payloads, _, _ = d.prepare(dict(runtime=runtime, requests=[request], output_dir="unused"))
        self.assertEqual(payloads[0], recorded)
        self.assertTrue(payloads[0]["return_tokens_as_token_ids"])
        self.assertNotIn("return_token_ids", payloads[0])
        self.assertEqual(payloads[0]["top_k"], 0)
        self.assertEqual(self.response["http_status"], 200)
        body = self.response["body"]
        self.assertTrue(body["choices"][0]["token_ids"])
        self.assertGreater(body["usage"]["estimated_cost"], 0)

    def test_recorded_output_retains_raw_eos_but_excludes_decoded_eos_marker(self):
        pieces = {939: "AL", 27305: "PHA", 417: " B", 20069: "ETA", 78990: " GAM", 4680: "MA",
                  248046: "<|im_end|>", 248044: "<|endoftext|>"}
        inner = SimpleNamespace(decode=lambda tokens, **kwargs: "".join(pieces[t] for t in tokens))
        with mock.patch.object(d, "version", return_value="fixture-version"):
            tokenizer = d.Tokenizer(inner)
        for eos in (248046, 248044):
            with self.subTest(eos=eos):
                body = deepcopy(self.response["body"])
                body["choices"][0]["token_ids"][-1] = eos
                row = dict(prompt_tokens=35, enable_thinking=False, answer_text=None)
                d.normalize(body, row, tokenizer, cap=self.request["payload"]["max_tokens"])
                self.assertEqual(row["tokens"], body["choices"][0]["token_ids"])
                self.assertEqual(row["tokens"][-1], eos)
                self.assertEqual(row["generation_tokens"], 7)
                self.assertEqual(row["text"], "ALPHA BETA GAMMA")
                self.assertEqual(row["answer_text"], "ALPHA BETA GAMMA")
                self.assertNotIn(pieces[eos], row["text"])
                self.assertEqual(row["usage"], self.response["body"]["usage"])


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in d.TOKENIZER_FILES[:4]:
            (self.root / name).write_text(json.dumps(dict(text_config=dict(eos_token_id=248044)))
                                         if name == "config.json" else "{}")
        self.runtime = dict(tokenizer_path=str(self.root), tokenizer_sha256={
            name: b.generation.file_hash(self.root / name) for name in d.TOKENIZER_FILES[:4]})
        self.inner = mock.Mock()
        self.inner.eos_token_id = 248046
        self.inner.encode.side_effect = lambda text, **kw: [248068] if text == "<think>" else [248069]
        self.factory = mock.Mock(return_value=self.inner)
        mock.patch.dict("sys.modules", transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(
            from_pretrained=self.factory))).start()
        mock.patch.object(d, "version", return_value="fixture-version").start()
        self.addCleanup(mock.patch.stopall)

    def test_cpu_wrapper_pins_loading_rendering_decoding_and_thought_boundaries(self):
        tokenizer = d.load_tokenizer(self.runtime)
        self.factory.assert_called_once_with(self.root, local_files_only=True, trust_remote_code=False)
        tokenizer.apply_chat_template([dict(role="user", content="Hi")], tokenize=True, enable_thinking=False)
        self.assertFalse(self.inner.apply_chat_template.call_args.kwargs["return_dict"])
        self.inner.decode.return_value = "print(1)\r\n"
        self.assertEqual(b.generation.decode_tokens(tokenizer, [7, d.EOS[0]], set(d.EOS)), "print(1)\r\n")
        self.inner.decode.assert_called_once_with([7], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertEqual(tokenizer.rfind_think_start([248068, 1, 248068]), 2)
        self.assertEqual(tokenizer.rfind_think_end([248068, 248069, 2]), 1)
        self.assertEqual(tokenizer.rfind_think_end([1, 2]), -1)

    def test_hash_mismatch_is_refused_before_loading_tokenizer(self):
        (self.root / "chat_template.jinja").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            d.load_tokenizer(self.runtime)
        self.factory.assert_not_called()

    def test_unpinned_optional_tokenizer_file_is_refused(self):
        (self.root / "added_tokens.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Pin every tokenizer input"):
            d.load_tokenizer(self.runtime)
        self.factory.assert_not_called()

    def test_wrong_thought_boundaries_are_refused(self):
        self.inner.encode.return_value = [17]
        self.inner.encode.side_effect = None
        with self.assertRaisesRegex(ValueError, "boundaries"):
            d.load_tokenizer(self.runtime)


class TransportTests(unittest.TestCase):
    def test_https_transport_sends_key_only_in_header_without_retries(self):
        response = mock.MagicMock(status=200)
        response.read.return_value = b'{"choices":[]}'
        response.__enter__.return_value = response
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(d.urllib.request, "build_opener", return_value=opener) as build:
            self.assertEqual(d.send(dict(prompt=[1]), "PRIVATE_TEST_KEY", 5), (200, '{"choices":[]}'))
        self.assertIsInstance(build.call_args.args[0], d.NoRedirect)
        opener.open.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.deepinfra.com/v1/openai/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer PRIVATE_TEST_KEY")
        self.assertNotIn(b"PRIVATE_TEST_KEY", request.data)
        self.assertEqual(opener.open.call_args.kwargs, dict(timeout=5))

    def test_http_error_body_is_preserved_without_retry(self):
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://api.deepinfra.com", 429,
            "Too many requests", {}, io.BytesIO(b'{"error":"limit"}'))
        with mock.patch.object(d.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(d.send(dict(prompt=[1]), "key", 5), (429, '{"error":"limit"}'))
        opener.open.assert_called_once()

    def test_redirect_cannot_forward_authorization_to_another_host(self):
        request = d.urllib.request.Request("https://api.deepinfra.com/v1/openai/completions",
            headers={"Authorization": "Bearer PRIVATE_TEST_KEY"})
        with self.assertRaisesRegex(RuntimeError, "redirect"):
            d.NoRedirect().redirect_request(request, None, 307, "redirect", {}, "https://other.example/")


class AccountingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_mixed_submission_certainty_preserves_observed_lower_bounds(self):
        rows = [dict(submitted=value, status=status, generation_tokens=count)
                for value, status, count in [(True, "stop", 3), (None, "error", 0), (False, "not_run", 0)]]
        result = dict(status="error", requests=rows, usage_complete=False, native=None,
                      elapsed_seconds=.5, error="timeout")
        (self.root / "outcome.json").write_text(json.dumps(result))
        state, attempts = dict(timing={}), [{}, {}, {}]
        self.assertEqual(b.collect_generation(self.root, state, attempts), result)
        self.assertIsNone(state["known_submissions"])
        self.assertEqual(state["known_submissions_lower_bound"], 1)
        self.assertEqual(state["observed_tokens"], 3)
        self.assertFalse(state["usage_complete"])
        self.assertEqual([a["submitted"] for a in attempts], [True, None, False])

    def test_explicit_usage_completeness_is_independent_of_round_success(self):
        for status, complete in [("success", False), ("error", True)]:
            with self.subTest(status=status):
                rows = [dict(submitted=True, status="stop", generation_tokens=2)]
                result = dict(status=status, requests=rows, usage_complete=complete, native=None)
                (self.root / "outcome.json").write_text(json.dumps(result))
                state = dict(timing={})
                self.assertIsNotNone(b.collect_generation(self.root, state, [{}]))
                self.assertEqual(state["usage_complete"], complete)

    def test_aggregate_retains_confirmed_peers_within_unknown_round(self):
        manifest = self.root / "manifest.json"
        manifest.write_text("{}")
        args = dict(worklist=[(dict(split="train", id=0), dict(question="Fixture"))],
                    output_dir=self.root / "run", recipe=dict(base_seed=17), source="actor",
                    model_path=self.root / "model", model_manifest=manifest, mlx_python="python",
                    batching={}, reply_allowance=3, output_token_allowance=30,
                    fixtures=True, cohort_sha256=None, started=0)
        _, _, progress, persist = b.prepare(**args)
        progress["rounds"] = [dict(known_submissions=2, observed_tokens=4, usage_complete=True, timing={}),
                              dict(known_submissions=None, known_submissions_lower_bound=1,
                                   observed_tokens=3, usage_complete=False, timing={})]
        progress.update(committed_replies=5, committed_tokens=50)
        persist()
        self.assertIsNone(progress["known_submissions"])
        self.assertEqual(progress["known_submissions_lower_bound"], 3)
        self.assertEqual(progress["observed_tokens"], 7)
        self.assertEqual((progress["committed_replies"], progress["committed_tokens"]), (5, 50))
