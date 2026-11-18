"""Provider round integration with finite fixture worklists; no HTTP or model load."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import deepinfra_generation as worker
from tests.test_thinking_generation import ThinkingTokenizer

b = importlib.import_module("src.data-pipeline.stage03.generation_loop")


class ProviderPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.tokenizer = self.root / "tokenizer"
        self.tokenizer.mkdir()
        for name in worker.TOKENIZER_FILES[:4]:
            (self.tokenizer / name).write_text(json.dumps(dict(text_config=dict(eos_token_id=248044)))
                                               if name == "config.json" else "{}")
        self.runtime = dict(model=worker.MODEL, tokenizer_path=str(self.tokenizer), tokenizer_sha256={
            name: b.generation.file_hash(self.tokenizer / name) for name in worker.TOKENIZER_FILES[:4]},
            timeout_seconds=5, max_total_tokens=100, input_usd_per_million=.2,
            output_usd_per_million=2.5, max_cost_usd=.01)
        self.runtime_file = self.root / "provider.json"
        self.runtime_file.write_text(json.dumps(self.runtime))
        self.worklist = []
        for i in range(2):
            task = dict(question=f"Statement {i}", cases=[dict(index=0, input="1", answer="1", visible=True)])
            path = self.root / f"task-{i}.json"
            path.write_text(json.dumps(task))
            self.worklist.append((dict(split="train", id=i, sha256=b.generation.file_hash(path)), task))
        self.recipe = dict(system="Return Python", initial="Solve the task", temperature=1., top_p=.95,
                           base_seed=17, max_tokens=16, max_repairs=1, repair="Repair the program",
                           repair_max_tokens=20, feedback_field_bytes=64, feedback_bytes=512)
        self.rounds, self.checks = [], []
        self.after_check = None
        self.child = mock.patch.object(b, "run_child", side_effect=self.generate).start()
        self.http = mock.patch.object(worker, "send", side_effect=AssertionError("No HTTP in fixture integration")).start()
        self.addCleanup(mock.patch.stopall)

    def run_loop(self, **changes):
        options = dict(source="actor", provider_runtime=self.runtime_file, generation_python=sys.executable,
                       recipe=self.recipe, reply_allowance=3, output_token_allowance=52,
                       fixtures=True, candidate_checker=self.verify)
        options.update(changes)
        return b.run(self.worklist, self.root / "run", **options)

    def saved(self, name="outcome.json"):
        return json.loads((self.root / "run" / name).read_text())

    def generate(self, command, current, persist):
        self.assertEqual(command[:2], [str(Path(sys.executable).absolute()), str(Path(worker.__file__).resolve())])
        specification = json.loads(Path(command[-1]).read_text())
        self.assertEqual(set(specification), {"runtime", "requests", "output_dir"})
        self.assertEqual(specification["runtime"], self.runtime)
        self.rounds.append(deepcopy(specification))
        saved = self.saved()
        self.assertEqual(saved["phase"], "dispatching")
        self.assertEqual(saved["committed_replies"], sum(len(r["requests"]) for r in self.rounds))
        self.assertEqual(saved["committed_tokens"], sum(request["max_tokens"] for r in self.rounds for request in r["requests"]))
        output = Path(specification["output_dir"])
        output.mkdir()
        rows = []
        for request in specification["requests"]:
            position = (request["seed"] - self.recipe["base_seed"]) % len(self.worklist)
            text = f"print({position})\n"
            rows.append(dict(submitted=True, status="stop", finish_reason="stop", text=text, answer_text=text,
                             tokens=[7, 248046], generation_tokens=2, prompt_tokens=7,
                             usage=dict(prompt_tokens=7, completion_tokens=2, estimated_cost=.00001)))
        b.generation.save(output / "outcome.json", dict(status="success", requests=rows, native=None,
            usage_complete=True, elapsed_seconds=.01, preparation_seconds=.001, error=None))
        return 0

    def execute_worker(self, command, current, persist):
        specification = json.loads(Path(command[-1]).read_text())
        self.rounds.append(deepcopy(specification))
        result = worker.run(command[-1])
        return 0 if result["status"] == "success" else 1

    def verify(self, candidates, directory, **settings):
        self.assertEqual(settings, dict(max_samples=4, concurrent_inference=False))
        directory.mkdir()
        self.checks.append(deepcopy(candidates))
        rows = []
        for task, digest, program in candidates:
            status = "failed" if len(self.checks) == 1 and task["question"] == "Statement 1" else "passed"
            rows.append(dict(status=status, key=dict(task_sha256=digest,
                program_sha256=b.study_tasks.candidate_identity("train", 0, digest, "actor", program)["program_sha256"]),
                feedback=dict(message="Public feedback")))
        if self.after_check is not None:
            self.after_check()
        return rows

    def test_initial_verification_and_repair_use_provider_rounds_without_local_manifest(self):
        original = deepcopy(self.worklist)
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 1])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18, 20])
        self.assertEqual([a["parent"] for a in result["attempts"]], [None, None, 1])
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (3, 52))
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (3, 6))
        self.assertEqual([len(r["requests"]) for r in self.rounds], [2, 1])
        self.assertEqual([len(c) for c in self.checks], [2, 1])
        self.assertEqual(self.worklist, original)
        repair = self.rounds[1]["requests"][0]
        self.assertIn("Public feedback", repair["messages"][1]["content"])
        inputs = self.saved("inputs.json")
        self.assertIn("provider_runtime", inputs)
        self.assertIn("configured_candidate", json.dumps(inputs["provider_runtime"]))
        self.assertAlmostEqual(inputs["provider_runtime"]["planned_max_cost_usd"], .00019)
        pins = inputs["provider_runtime"]["files"]
        for path in (Path(worker.__file__), Path(b.__file__), Path(b.__file__).with_name("deepinfra_runtime.py")):
            self.assertEqual(pins[str(path.resolve())], b.generation.file_hash(path))
        for name in ("model_manifest", "model_manifest_sha256", "model_path", "mlx_python"):
            self.assertNotIn(name, inputs)
        self.assertEqual(self.saved(), result)

    def test_custom_repair_history_passes_through_provider_route(self):
        def repair(initial, parent, recipe, seed):
            return dict(initial, seed=seed, max_tokens=recipe["repair_max_tokens"], messages=[*initial["messages"],
                dict(role="assistant", content="", tool_calls=[dict(id="prior", type="function",
                     function=dict(name="test_edits", arguments=dict(edits=[dict(old="1", new="2")])))]),
                dict(role="tool", tool_call_id="prior", content="Public repair feedback")])
        result = self.run_loop(repair_builder=repair)
        self.assertEqual(result["status"], "completed", result)
        history = self.rounds[1]["requests"][0]["messages"]
        self.assertEqual([m["role"] for m in history], ["system", "user", "assistant", "tool"])
        self.assertEqual(history[-2]["tool_calls"][0]["function"]["arguments"], dict(edits=[dict(old="1", new="2")]))

    def test_optional_concurrency_is_pinned_and_forwarded_to_every_round(self):
        self.runtime["concurrency"] = 2
        self.runtime_file.write_text(json.dumps(self.runtime))
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([r["runtime"]["concurrency"] for r in self.rounds], [2, 2])
        self.assertEqual(self.saved("inputs.json")["provider_runtime"]["runtime"]["concurrency"], 2)

    def test_real_worker_missing_key_collects_known_zero_submissions(self):
        self.child.side_effect = self.execute_worker
        with mock.patch.dict("os.environ", DEEPINFRA_API_KEY=""), \
                mock.patch.object(worker, "load_tokenizer", return_value=ThinkingTokenizer("qwen")):
            result = self.run_loop()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))
        self.assertEqual(result["known_submissions"], 0)
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["observed_tokens"], 0)
        self.assertEqual([a["submitted"] for a in result["attempts"]], [False, False])
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual(self.checks, [])
        self.http.assert_not_called()

    def test_real_worker_overlong_repair_is_known_unsubmitted_without_refunding_reservations(self):
        tokenizer = ThinkingTokenizer("qwen")
        tokenizer.apply_chat_template.side_effect = lambda messages, tokenize, **kwargs: list(range(95)) if tokenize else "Long repair"
        def child(command, current, persist):
            if not self.rounds:
                return self.generate(command, current, persist)
            return self.execute_worker(command, current, persist)
        self.child.side_effect = child
        with mock.patch.dict("os.environ", DEEPINFRA_API_KEY="PRIVATE_TEST_KEY"), \
                mock.patch.object(worker, "load_tokenizer", return_value=tokenizer):
            result = self.run_loop()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (3, 52))
        self.assertEqual(result["known_submissions"], 2)
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["observed_tokens"], 4)
        self.assertEqual(result["rounds"][1]["known_submissions"], 0)
        self.assertTrue(result["rounds"][1]["usage_complete"])
        self.assertIs(result["attempts"][-1]["submitted"], False)
        self.assertEqual(self.child.call_count, 2)
        self.assertEqual(len(self.checks), 1)
        self.http.assert_not_called()

    def test_whole_run_spending_cap_rejects_before_output_or_child(self):
        # Enough for the initial round's envelope ($.00012), below the full run ($.00019).
        self.runtime["max_cost_usd"] = .00015
        self.runtime_file.write_text(json.dumps(self.runtime))
        with self.assertRaises(ValueError):
            self.run_loop()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_provider_refuses_local_settings_synthetic_driver_override_and_missing_python(self):
        cases = [dict(model_path=self.root / "model"), dict(model_manifest=self.root / "manifest.json"),
                 dict(mlx_python=sys.executable), dict(completion_batch_size=1), dict(prefill_batch_size=1),
                 dict(prefill_step_size=1), dict(source="synthetic"), dict(generation_driver=Path(__file__)),
                 dict(generation_python=None)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_loop(**changes)
            self.child.assert_not_called()
            self.assertFalse((self.root / "run").exists())

    def test_tokenizer_mutation_after_verification_stops_before_repair_dispatch(self):
        self.after_check = lambda: (self.tokenizer / "chat_template.jinja").write_text("Changed template")
        result = self.run_loop()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))
        self.assertEqual(len(self.checks), 1)

    def test_runtime_mutation_after_verification_stops_before_repair_dispatch(self):
        def mutate():
            self.runtime_file.write_text(json.dumps(dict(self.runtime, max_cost_usd=.02)))
        self.after_check = mutate
        result = self.run_loop()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))

    def test_custom_repair_cannot_exceed_frozen_output_allowance(self):
        def oversized(initial, parent, recipe, seed):
            return dict(b.repair_request(initial, parent, recipe, seed), max_tokens=21)
        result = self.run_loop(repair_builder=oversized)
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))

    def test_source_pin_failure_before_repair_does_not_reserve_or_dispatch(self):
        check_files = b.check_files
        calls = []
        def checked(files):
            calls.append(files)
            if len(calls) == 2:
                raise ValueError("Source hash mismatch")
            return check_files(files)
        with mock.patch.object(b, "check_files", side_effect=checked):
            result = self.run_loop()
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.child.call_count, 1)
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))

    def test_bad_runtime_model_fails_before_output(self):
        self.runtime["model"] = "Qwen/Qwen3.8-27B:wrong-version"
        self.runtime_file.write_text(json.dumps(self.runtime))
        with self.assertRaises(ValueError):
            self.run_loop()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_unpinned_tokenizer_file_fails_before_output(self):
        (self.tokenizer / "added_tokens.json").write_text("{}")
        with self.assertRaises(ValueError):
            self.run_loop()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
