import importlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

g = importlib.import_module("src.data-pipeline.stage03.local_generation")


def request(text="A", seed=17, cap=2, seconds=1):
    return dict(messages=[dict(role="user", content=text)], seed=seed, temperature=.7,
                top_p=.9, max_tokens=cap, wall_time_seconds=seconds)


def response(text="partial<unk>", token=7, count=1, reason=None):
    return NS(text=text, token=token, prompt_tokens=1, generation_tokens=count, finish_reason=reason)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name, content in {"config.json": '{"quantization":{"bits":8}}',
                              "manifest.json": '{}', "tokenizer.json": '{}', "chat_template.jinja": 'fixture'}.items():
            (self.root / name).write_text(content)
        self.tokenizer = NS(apply_chat_template=mock.Mock(side_effect=lambda messages, tokenize, **k:
            [ord(messages[0]["content"][0])] if tokenize else "<think>\\n\\n</think>\\n"))
        self.load = mock.Mock(return_value=(object(), self.tokenizer))
        self.seed, self.sampler, self.closed = mock.Mock(), mock.Mock(return_value="sampler"), mock.Mock()
        def stream(*args, **kwargs):
            try:
                yield response("```python\n<unk>")
                yield response("\n```", 9, 2, "stop")
            finally:
                self.closed()
        self.stream = mock.Mock(side_effect=stream)
        mlx, lm = ModuleType("mlx"), ModuleType("mlx_lm")
        mlx.core = NS(random=NS(seed=self.seed))
        lm.load, lm.stream_generate = self.load, self.stream
        modules = {"mlx": mlx, "mlx.core": mlx.core, "mlx_lm": lm,
                   "mlx_lm.sample_utils": NS(make_sampler=self.sampler, __file__=__file__)}
        modules.update({name: NS(__file__=__file__) for name in
                        ("mlx_lm.generate", "mlx_lm.tokenizer_utils", "mlx_lm.models.qwen3_5")})
        self.addCleanup(mock.patch.stopall)
        mock.patch.dict("sys.modules", modules).start()
        mock.patch.object(g, "version", return_value="fixture").start()

    def generate(self, requests, name="run"):
        return g.generate(self.root, requests, self.root / name, model_manifest=self.root / "manifest.json")

    def test_one_load_fresh_serial_calls_and_verbatim_records(self):
        requests = [request(t, s) for t, s in (("A", 17), ("B", 18), ("A", 17))]
        result = self.generate(requests)
        self.assertEqual(result["status"], "success")
        self.load.assert_called_once_with(str(self.root), lazy=False,
            tokenizer_config={"local_files_only": True}, trust_remote_code=False)
        self.assertEqual(self.seed.call_args_list, [mock.call(s) for s in (17, 18, 17)])
        self.assertEqual(self.sampler.call_args_list, [mock.call(temp=.7, top_p=.9)] * 3)
        for i, call in enumerate(self.stream.call_args_list):
            self.assertEqual(call.args[2], [ord(requests[i]["messages"][0]["content"])])
            self.assertEqual(call.kwargs, dict(max_tokens=2, sampler="sampler"))
        for i, call in enumerate(self.tokenizer.apply_chat_template.call_args_list):
            self.assertEqual(call.args[0], requests[i // 2]["messages"])
            self.assertTrue(call.kwargs["add_generation_prompt"])
            self.assertFalse(call.kwargs["enable_thinking"])
        self.assertEqual(self.closed.call_count, 3)
        row = result["requests"][0]
        self.assertEqual(row["text"], "```python\n<unk>\n```")
        self.assertEqual(row["generation_tokens"], 2)
        self.assertEqual((self.root / "run/reply-0000.txt").read_text(), row["text"])

    def test_soft_timeout_preserves_late_chunk_closes_stream_and_stops_run(self):
        clock = [0.]
        def slow(*args, **kwargs):
            try:
                clock[0] = 2.
                yield response()
                self.fail("Generation continued after soft deadline")
            finally:
                self.closed()
        self.stream.side_effect = slow
        with mock.patch.object(g, "monotonic", side_effect=lambda: clock[0]):
            result = self.generate([request(), request("B")])
        self.assertEqual(result["status"], "timeout")
        self.assertEqual([r["status"] for r in result["requests"]], ["timeout", "not_run"])
        self.assertEqual(result["requests"][0]["text"], "partial<unk>")
        self.assertEqual(result["requests"][0]["elapsed_seconds"], 2)
        self.closed.assert_called_once()

    def test_errors_and_interruptions_preserve_completed_and_partial_results(self):
        for error, status in ((RuntimeError, "error"), (KeyboardInterrupt, "cancelled")):
            def failing(model, tokenizer, ids, **kwargs):
                if ids == [ord("B")]:
                    yield response()
                    raise error("fixture failure")
                yield response("OK", reason="stop")
            self.stream.side_effect = failing
            if error is KeyboardInterrupt:
                with self.assertRaises(KeyboardInterrupt):
                    self.generate([request(), request("B"), request("C")], status)
                result = json.loads((self.root / status / "outcome.json").read_text())
            else:
                result = self.generate([request(), request("B"), request("C")], status)
            self.assertEqual(result["status"], status)
            self.assertEqual([r["status"] for r in result["requests"]], ["stop", status, "not_run"])
            self.assertEqual(result["requests"][1]["text"], "partial<unk>")

    def test_missing_completion_load_failure_and_cleanup_failure(self):
        def incomplete(*args, **kwargs):
            yield response()
        self.stream.side_effect = incomplete
        result = self.generate([request()], "incomplete")
        self.assertEqual(result["requests"][0]["status"], "error")
        def cleanup_failure(*args, **kwargs):
            try:
                yield response(reason="stop")
            finally:
                raise RuntimeError("cleanup failed")
        self.stream.side_effect = cleanup_failure
        result = self.generate([request()], "cleanup")
        self.assertEqual(result["status"], "error")
        self.assertIn("cleanup failed", result["error"])
        self.load.side_effect = RuntimeError("load failed")
        result = self.generate([request()], "load")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["requests"][0]["status"], "not_run")

    def test_length_status_existing_directory_and_16bit_rejection(self):
        def limited(*args, **kwargs):
            yield response("limited", reason="length")
        self.stream.side_effect = limited
        self.assertEqual(self.generate([request(cap=1)])["requests"][0]["status"], "length")
        with self.assertRaises(FileExistsError):
            self.generate([request()])
        (self.root / "config.json").write_text('{"quantization":{"bits":16}}')
        with self.assertRaises(ValueError):
            self.generate([request()], "16bit")
        self.assertEqual(self.load.call_count, 1)


@unittest.skipUnless(os.environ.get("ROPOT_MLX_TESTS") == "1", "set ROPOT_MLX_TESTS=1 in the MLX Python")
class LiveSmoke(unittest.TestCase):
    def test_warm_actor(self):
        model = Path("/Users/michal/Workspace/models/mlx-community/Qwen3.5-9B-8bit")
        manifest = model.parents[1] / "metadata/Qwen3.5-9B-8bit-16daa4818c54ce5f5436f929d52542eb65bbed9d.json"
        root = Path(tempfile.mkdtemp(prefix="direct-actor-", dir="data/03_generation_checks"))
        a = request("Write one short sentence about a red fox.", cap=32, seconds=30)
        b = request("Reply with exactly OK.", seed=18, cap=32, seconds=30)
        result = g.generate(model, [a, b, a, request("Write Python: read two integers and print their sum. Code only.",
            cap=128, seconds=30), request("List the integers from one to one hundred.", cap=1, seconds=30)],
            root / "serial", model_manifest=manifest)
        self.assertEqual(result["status"], "success", root)
        self.assertEqual(result["requests"][0]["tokens"], result["requests"][2]["tokens"])
        self.assertEqual(result["requests"][1]["status"], "stop")
        self.assertEqual(result["requests"][-1]["status"], "length")
        self.assertTrue(all(r["prompt"].endswith("<think>\n\n</think>\n\n") for r in result["requests"]))
        self.assertTrue(all("<think>" not in r["text"] and "</think>" not in r["text"] for r in result["requests"]))
        print("Retained direct-call smoke:", root)
