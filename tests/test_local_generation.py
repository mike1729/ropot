import importlib
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest import mock

from tests.fixture_verification import verify_saved_programs

g = importlib.import_module("src.data-pipeline.stage03.local_generation")


def request(text="A", seed=17, cap=2):
    return dict(messages=[dict(role="user", content=text)], seed=seed, temperature=.7,
                top_p=.9, max_tokens=cap)


def response(text="partial<unk>", token=7, count=1, reason=None):
    return NS(text=text, token=token, prompt_tokens=1, generation_tokens=count, finish_reason=reason)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name, content in {"config.json": '{"quantization":{"bits":8}}',
                              "manifest.json": '{}', "tokenizer.json": '{}',
                              "tokenizer_config.json": '{}', "chat_template.jinja": 'fixture'}.items():
            (self.root / name).write_text(content)
        self.tokenizer = NS(_tokenizer=NS(), apply_chat_template=mock.Mock(side_effect=lambda messages, tokenize, **k:
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
        for name in ("mlx_lm.generate", "mlx_lm.tokenizer_utils", "mlx_lm.models.qwen3_5",
                     "mlx_lm.models.mistral3", "mlx_lm.models.ministral3"):
            path = self.root / f"{name}.py"
            path.write_text(name)
            modules[name] = NS(__file__=str(path))
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
        self.seed.assert_not_called()
        self.assertEqual(self.sampler.call_args_list,
                         [mock.call(temp=.7, top_p=.9, seed=s) for s in (17, 18, 17)])
        runtime = json.loads((self.root / "run/runtime.json").read_text())
        self.assertEqual(runtime["tokenizer_options"], {"local_files_only": True})
        self.assertIn(str(self.root / "mlx_lm.models.qwen3_5.py"), runtime["source_sha256"])
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

    def test_synthetic_tokenizer_and_model_identity(self):
        (self.root / "config.json").write_text(json.dumps({"model_type": "mistral3",
            "text_config": {"model_type": "ministral3"}, "quantization": {"bits": 8}}))
        self.assertEqual(self.generate([request()])["status"], "success")
        self.load.assert_called_once_with(str(self.root), lazy=False,
            tokenizer_config={"local_files_only": True, "fix_mistral_regex": True}, trust_remote_code=False)
        runtime = json.loads((self.root / "run/runtime.json").read_text())
        self.assertEqual(runtime["tokenizer_class"], "types.SimpleNamespace")
        self.assertEqual(runtime["tokenizer_options"], {"local_files_only": True, "fix_mistral_regex": True})
        for name in ("mistral3", "ministral3"):
            path = self.root / f"mlx_lm.models.{name}.py"
            self.assertEqual(runtime["source_sha256"][str(path)], g.file_hash(path))
        self.assertNotIn(str(self.root / "mlx_lm.models.qwen3_5.py"), runtime["source_sha256"])
        inputs = json.loads((self.root / "run/inputs.json").read_text())
        self.assertEqual(inputs["model_metadata_sha256"]["tokenizer_config.json"],
                         g.file_hash(self.root / "tokenizer_config.json"))
        self.sampler.assert_called_once_with(temp=.7, top_p=.9, seed=17)

    def test_gemma_records_its_own_architecture_sources(self):
        (self.root / "config.json").write_text('{"model_type":"gemma4","quantization":{"bits":8}}')
        modules = {}
        for name in ("gemma4", "gemma4_text"):
            path = self.root / (name+".py")
            path.write_text(name)
            modules["mlx_lm.models."+name] = NS(__file__=str(path))
        with mock.patch.dict("sys.modules", modules):
            self.assertEqual(self.generate([request()])["status"], "success")
        sources = json.loads((self.root / "run/runtime.json").read_text())["source_sha256"]
        self.assertTrue(all(str(self.root / (name+".py")) in sources for name in ("gemma4", "gemma4_text")))
        self.assertNotIn(str(self.root / "mlx_lm.models.qwen3_5.py"), sources)

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


class SmokeExtractionTests(unittest.TestCase):
    def test_raw_fenced_and_prose_wrapped_python_preserve_code(self):
        code = "print('żółw')\n"
        for text in (code, f"```python\n{code}```", f"Here is the code:\n```python\n{code}```\nDone."):
            self.assertEqual(g.extract_program({"status": "stop", "text": text}), code.encode())

    def test_incomplete_ambiguous_and_invalid_replies_are_rejected(self):
        for status, text in (("length", "print(1)"), ("cancelled", "print(1)"),
                ("stop", "```python\nprint(1)"), ("stop", "```python\nprint(1)\n```\n```python\nprint(2)\n```"),
                ("stop", "```javascript\nprint(1)\n```"), ("stop", "print(")):
            with self.subTest(status=status, text=text), self.assertRaises((ValueError, SyntaxError)):
                g.extract_program({"status": status, "text": text})


@unittest.skipUnless(os.environ.get("ROPOT_MLX_TESTS") == "1", "set ROPOT_MLX_TESTS=1 in the MLX Python")
class LiveSmoke(unittest.TestCase):
    def test_warm_actor(self):
        model = Path("/Users/michal/Workspace/models/mlx-community/Qwen3.5-9B-8bit")
        manifest = model.parents[1] / "metadata/Qwen3.5-9B-8bit-16daa4818c54ce5f5436f929d52542eb65bbed9d.json"
        root = Path(tempfile.mkdtemp(prefix="direct-actor-", dir="data/03_generation_checks"))
        a = request("Write one short sentence about a red fox.", cap=32)
        b = request("Reply with exactly OK.", seed=18, cap=32)
        result = g.generate(model, [a, b, a, request("Write Python: read two integers and print their sum. Code only.",
            cap=128), request("List the integers from one to one hundred.", cap=1)],
            root / "serial", model_manifest=manifest)
        self.assertEqual(result["status"], "success", root)
        self.assertEqual(result["requests"][0]["tokens"], result["requests"][2]["tokens"])
        self.assertEqual(result["requests"][1]["status"], "stop")
        self.assertEqual(result["requests"][-1]["status"], "length")
        self.assertTrue(all(r["prompt"].endswith("<think>\n\n</think>\n\n") for r in result["requests"]))
        self.assertTrue(all("<think>" not in r["text"] and "</think>" not in r["text"] for r in result["requests"]))
        print("Retained direct-call smoke:", root)

    def test_warm_synthetic(self):
        from transformers import AutoTokenizer

        models = Path("/Users/michal/Workspace/models")
        model = models / "mlx-community/Ministral-3-8B-Instruct-2512-8bit"
        official = models / "mistralai/Ministral-3-8B-Instruct-2512/5b26027e7b19eeb4b7352e1fed3926375dd2cb4d"
        manifest = models / "metadata/Ministral-3-8B-Instruct-2512-8bit-f333d6bcb3df97264eddcf985523b69b747f5385.json"
        Path("data/03_generation_checks").mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="direct-synthetic-", dir="data/03_generation_checks")).resolve()
        print("Retained synthetic smoke:", root, flush=True)
        tasks = [{"question": "Read one integer from stdin and print its square.",
                  "cases": [{"index": 0, "input": "2\n", "answer": "4", "visible": True},
                            {"index": 1, "input": "-3\n", "answer": "9", "visible": False}]},
                 {"question": "Read n, then n integers. Print the number of distinct integers.",
                  "cases": [{"index": 0, "input": "5\n1 2 2 1 3\n", "answer": "3", "visible": True},
                            {"index": 1, "input": "3\n-1 -1 -1\n", "answer": "1", "visible": False}]}]
        for index, task in enumerate(tasks):
            g.save(root / f"task-{index}.json", task)
        a, b = [request(t["question"] + "\nExample input:\n" + t["cases"][0]["input"]
                        + "Example output:\n" + t["cases"][0]["answer"], cap=256) for t in tasks]
        cap = request("List integers from one to one hundred.", cap=1)
        for row in (a, b, cap):
            row.update(temperature=.2, top_p=.95)
            row["messages"].insert(0, {"role": "system", "content": "Return only a Python 3 program."})
        b["temperature"] = 0
        repair = a["messages"] + [{"role": "assistant", "content": "print(0)"},
            {"role": "user", "content": "Visible case returned 0; expected 4. Replace the program."}]
        tokenizers = [AutoTokenizer.from_pretrained(p, local_files_only=True, fix_mistral_regex=True,
                                                   trust_remote_code=False) for p in (model, official)]
        options = dict(add_generation_prompt=True, enable_thinking=False, return_dict=False)
        checks = [[dict(prompt=t.apply_chat_template(messages, tokenize=False, **options),
                        ids=t.apply_chat_template(messages, tokenize=True, **options),
                        eos=t.eos_token_id) for t in tokenizers] for messages in (a["messages"], repair)]
        g.save(root / "prompt-checks.json", {"histories": [a["messages"], repair], "checks": checks})
        for local, reference in checks:
            self.assertEqual(local, reference)
            self.assertEqual(local["ids"][0], 1)
            self.assertEqual(local["ids"].count(1), 1)
            self.assertEqual(local["eos"], 2)
        requests = [a, b, a, cap]
        result = g.generate(model, requests, root / "serial", model_manifest=manifest)
        candidates, failures = [], []
        for index, row in enumerate(result["requests"][:3]):
            try:
                (root / f"candidate-{index}.py").write_bytes(g.extract_program(row))
                candidates.append({"task": index % 2, "program": f"candidate-{index}.py"})
            except (ValueError, SyntaxError, MemoryError, RecursionError) as error:
                failures.append({"request": index, "error": repr(error)})
        g.save(root / "smoke.json", {"candidates": candidates, "extraction_failures": failures})
        if candidates:
            # Inspect lives in the repository venv; keep the pinned MLX environment unchanged.
            verify_saved_programs(root, candidates)
        self.assertEqual(result["status"], "success", root)
        self.assertEqual(failures, [], root)
        self.assertEqual([r["status"] for r in result["requests"]], ["stop", "stop", "stop", "length"])
        self.assertEqual(result["requests"][0]["tokens"], result["requests"][2]["tokens"])
        for req, row in zip(requests, result["requests"]):
            self.assertEqual(row["prompt"], tokenizers[1].apply_chat_template(req["messages"], tokenize=False, **options))
            self.assertEqual(row["prompt_ids"], tokenizers[1].apply_chat_template(req["messages"], tokenize=True, **options))
            self.assertEqual(row["prompt_tokens"], len(row["prompt_ids"]))
            self.assertEqual(row["generation_tokens"], len(row["tokens"]))
            self.assertNotIn("<think>", row["text"])
        self.assertTrue(all(row["tokens"][-1] == 2 for row in result["requests"][:3]))
        self.assertEqual([v["status"] for v in json.loads((root / "verification.json").read_text())], ["passed"] * 3)
