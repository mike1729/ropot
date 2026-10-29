"""Repair rounds: original positions, accounting, fresh prompts and technical stops."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

from tests import test_benign_loop as initial

b, v = initial.b, initial.v


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class RepairTests(initial.LoopFixture):
    def setUp(self):
        super().setUp()
        self.recipe.update(max_repairs=2, repair="Repair instruction", repair_max_tokens=20,
                           feedback_field_bytes=64, feedback_bytes=512)
        self.draws, self.verdicts, self.requests = {}, {}, []
        self.round_index = 0

    def generate(self, command, progress, persist):
        spec = json.loads(Path(command[-1]).read_text())
        self.round_index = int(Path(command[-1]).stem.split("-")[-1])
        self.requests.append(deepcopy(spec["requests"]))
        saved = self.saved()
        self.assertEqual(saved["phase"], "dispatching")
        self.assertEqual(saved["committed_replies"], sum(map(len, self.requests)))
        self.assertEqual(saved["committed_tokens"], sum(r["max_tokens"] for rs in self.requests for r in rs))
        directory = Path(spec["output_dir"])
        directory.mkdir()
        b.generation.save(directory / "inputs.json", dict(requests=spec["requests"],
            model_manifest_sha256=b.generation.file_hash(self.manifest)))
        b.generation.save(directory / "runtime.json", dict(load_seconds=.1))
        rows = []
        for index, request in enumerate(spec["requests"]):
            round_index, position = divmod(request["seed"] - self.recipe["base_seed"], len(self.worklist))
            self.assertEqual(round_index, self.round_index)
            status, text = self.draws.get((round_index, position), ("stop", f"print({position})\r\n"))
            row = dict(status=status, text=text, answer_text=self.answer_texts.get((round_index, position), text),
                       submitted=True, native_uid=40+index,
                       tokens=[12, 99], generation_tokens=2, prompt_tokens=7)
            rows.append(row)
            (directory / f"reply-{index:04d}.txt").write_bytes(text.encode())
        b.generation.save(directory / "outcome.json", dict(status="success", requests=rows,
            preparation_seconds=.2, native=dict(generation_tokens=2*len(rows), wall_time=.3)))
        return 0

    def verify(self, candidates, directory, **kwargs):
        self.assertEqual(kwargs, dict(max_samples=4, concurrent_inference=False))
        results = super().verify(candidates, directory, **kwargs)
        for result, (_, digest, _) in zip(results, candidates):
            position = next(i for i, (binding, _) in enumerate(self.worklist) if binding["sha256"] == digest)
            status = self.verdicts.get((self.round_index, position), "passed")
            result.update(status=status, private="PRIVATE_VERDICT", hidden_index=979797,
                feedback=dict(status=status, hidden="failed" if status == "failed" else "passed",
                    visible=[dict(status="wrong_answer", input="PUBLIC_INPUT", answer="PUBLIC_ANSWER",
                                  stdout="PUBLIC_STDOUT", stderr=None, diagnostic=None, returncode=0)]))
        return results

    def test_shrinking_round_keeps_original_positions_seeds_and_parent(self):
        self.verdicts[0, 1] = "failed"
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 1])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18, 20])
        self.assertEqual([a["parent"] for a in result["attempts"]], [None, None, 1])
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (3, 52))
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (3, 6))
        self.assertAlmostEqual(result["timing"]["loading"], .2)
        self.assertEqual(len(result["verification_logs"]), 2)
        attempts = [a for a in result["attempts"] if a["position"] == 1]
        self.assertEqual(attempts[0]["candidate"], attempts[1]["candidate"])
        self.assertNotEqual(attempts[0]["attempt"], attempts[1]["attempt"])
        self.assertEqual(Path(attempts[1]["program"]).read_bytes(), b"print(1)\r\n")
        self.assertEqual(attempts[1]["verification"]["sample_id"], 0)
        self.assertEqual(self.verifier.call_count, 2)

    def test_comments_are_removed_before_verification_hashing_and_repairs(self):
        self.recipe["strip_comments"] = True
        self.verdicts[0, 0] = "failed"
        self.draws[0, 0] = "stop", "# initial explanation\nprint('# literal') # inline\n"
        self.draws[1, 0] = "stop", "```python\n# repaired explanation\nprint(2) # inline\n```"
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        for attempt, expected in ((result["attempts"][0], b"print('# literal')\n"),
                                  (result["attempts"][2], b"print(2)\n")):
            self.assertEqual(Path(attempt["program"]).read_bytes(), expected)
            self.assertEqual(attempt["candidate"]["program_sha256"],
                             b.generation.file_hash(attempt["program"]))
            self.assertEqual(attempt["verification"]["key"]["program_sha256"],
                             attempt["candidate"]["program_sha256"])
        repair = self.requests[1][0]["messages"][1]["content"]
        self.assertIn("print('# literal')", repair)
        self.assertNotIn("initial explanation", repair)
        self.assertNotIn("# inline", repair)
        self.assertIn("# initial explanation", (self.root / "run/generation-0000/reply-0000.txt").read_text())
        self.assertIn("# repaired explanation", (self.root / "run/generation-0001/reply-0000.txt").read_text())

    def test_thinking_passthrough_and_answer_only_repairs(self):
        self.recipe["enable_thinking"] = True
        self.verdicts[0, 0] = "failed"
        self.draws[0, 0] = "stop", "PRIVATE_REASONING\n```python\nprint(999)\n```"
        self.answer_texts[0, 0] = "print('answer only')\n"
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertTrue(all(r["enable_thinking"] is True for rs in self.requests for r in rs))
        repair = self.requests[1][0]["messages"][1]["content"]
        self.assertIn("print('answer only')", repair)
        self.assertNotIn("PRIVATE_REASONING", repair)
        self.assertNotIn("print(999)", repair)
        attempt = result["attempts"][0]
        self.assertEqual(Path(attempt["program"]).read_bytes(), b"print('answer only')\n")
        self.assertEqual(result["observed_tokens"], 6)

    def test_unclosed_thought_consumes_attempt_and_never_enters_repair(self):
        self.recipe["enable_thinking"] = True
        self.draws[0, 0] = "stop", "PRIVATE_REASONING\n```python\nprint(999)\n```"
        self.answer_texts[0, 0] = None
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(result["attempts"][0]["extraction"], "format")
        self.assertEqual([len(call.args[0]) for call in self.verifier.call_args_list], [1, 1])
        repair = self.requests[1][0]["messages"][1]["content"]
        self.assertIn("Local extraction failure: format", repair)
        self.assertNotIn("PRIVATE_REASONING", repair)
        self.assertNotIn("print(999)", repair)
        self.assertEqual((result["committed_replies"], result["observed_tokens"]), (3, 6))

    def test_fresh_repairs_contain_only_context_current_code_and_public_feedback(self):
        self.verdicts.update({(r, 0): "failed" for r in range(3)})
        self.draws.update({(0, 0): ("stop", "PRIVATE_REASONING\n```python\nprint('first')\n```"),
                           (1, 0): ("stop", "PRIVATE_REASONING\n```python\nprint('second')\n```")})
        original = deepcopy(self.worklist)
        result = self.run_loop()
        self.assertEqual([t["state"] for t in result["tasks"]], ["repairs_exhausted", "passed"])
        messages = self.requests[2][0]["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], self.requests[0][0]["messages"][0])
        self.assertTrue(messages[1]["content"].startswith(self.requests[0][0]["messages"][1]["content"]))
        self.assertIn("print('second')", messages[1]["content"])
        self.assertNotIn("print('first')", messages[1]["content"])
        self.assertIn("PUBLIC_ANSWER", messages[1]["content"])
        rendered = json.dumps(self.requests)
        for sentinel in ("PRIVATE", "979797", "898989", "contract_sha256", "native.eval", "construction"):
            self.assertNotIn(sentinel, rendered)
        self.assertEqual(self.worklist, original)
        self.assertEqual(result["attempts"][0]["verification"]["private"], "PRIVATE_VERDICT")
        self.assertIn("repair_feedback", result["attempts"][0])

    def test_length_format_and_syntax_each_consume_a_repair_without_execution(self):
        for status, text, category in (("length", "print(0)", "length"),
                ("stop", "```python\n```", "format"), ("stop", "prose only", "syntax")):
            with self.subTest(category=category):
                self.draws[0, 0] = status, text
                result = self.run_loop()
                self.assertEqual(result["attempts"][0]["extraction"], category)
                self.assertEqual([len(c.args[0]) for c in self.verifier.call_args_list], [1, 1])
                messages = self.requests[1][0]["messages"]
                self.assertEqual(len(messages), 2)
                self.assertNotIn(text, messages[1]["content"])
                self.assertIn(f"Local extraction failure: {category}", messages[1]["content"])
                self.assertNotIn("Traceback", messages[1]["content"])
                self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
                shutil.rmtree(self.root / "run")
                self.requests.clear()
                self.verifier.reset_mock()

    def test_stable_prefix_and_no_refund_of_unused_tokens(self):
        self.verdicts.update({(0, 0): "failed", (0, 1): "failed"})
        for limits in (dict(reply_allowance=3), dict(output_token_allowance=71)):
            with self.subTest(limits=limits):
                result = self.run_loop(**limits)
                self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 0])
                self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "budget_exhausted"])
                self.assertEqual(result["committed_tokens"], 52)
                self.assertEqual(result["observed_tokens"], 6)
                shutil.rmtree(self.root / "run")
                self.requests.clear()

    def test_no_repair_can_fit_global_allowance(self):
        self.verdicts[0, 0] = "failed"
        result = self.run_loop(reply_allowance=2, output_token_allowance=32)
        self.assertEqual([t["state"] for t in result["tasks"]], ["budget_exhausted", "passed"])
        self.child.assert_called_once()

    def test_full_potential_seed_schedule_checked_before_dispatch(self):
        self.recipe["base_seed"] = 2**64 - 5
        with self.assertRaises(ValueError):
            self.run_loop()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        self.recipe["base_seed"] = 2**64 - 6
        self.verdicts.update({(r, 1): "failed" for r in range(3)})
        result = self.run_loop()
        self.assertEqual([a["seed"] for a in result["attempts"]], [2**64-6, 2**64-5, 2**64-3, 2**64-1])

    def test_invalid_repair_fields_refused_without_creating_output(self):
        for key, values in (("max_repairs", [-1, True]), ("repair_max_tokens", [0, True]),
                            ("feedback_field_bytes", [0, 10, True]), ("feedback_bytes", [1, True]),
                            ("repair", [None, ""])):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.run_loop(recipe={**self.recipe, key: value})
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_missing_repair_outcome_retains_prior_counts_and_passing_peer(self):
        self.verdicts[0, 0] = "failed"
        def incomplete(*args):
            result = self.generate(*args)
            if self.round_index == 1:
                (self.root / "run/generation-0001/outcome.json").write_text('{"status":')
            return result
        self.child.side_effect = incomplete
        result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["submission_unknown", "passed"])
        self.assertEqual(result["committed_tokens"], 52)
        self.assertIsNone(result["known_submissions"])
        self.assertEqual(result["known_submissions_lower_bound"], 2)
        self.assertEqual(result["observed_tokens"], 4)
        self.assertFalse(result["usage_complete"])
        self.verifier.assert_called_once()
        self.assertEqual(self.child.call_count, 2)

    def test_interrupt_during_repair_verification_keeps_initial_pass(self):
        self.verdicts[0, 0] = "failed"
        error = KeyboardInterrupt("repair verification")
        def interrupt(*args, **kwargs):
            if self.round_index == 1:
                directory = args[1]
                directory.mkdir()
                (directory / "cancelled.eval").touch()
                raise error
            return self.verify(*args, **kwargs)
        self.verifier.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_loop()
        self.assertIs(caught.exception, error)
        result = self.saved()
        self.assertEqual([t["state"] for t in result["tasks"]], ["cancelled", "passed"])
        self.assertEqual(len(result["verification_logs"]), 2)
        self.assertEqual(result["committed_replies"], 3)
        self.assertEqual(self.child.call_count, 2)

    def test_compiler_resource_during_repair_stops_without_verifying_it(self):
        self.verdicts[0, 0] = "failed"
        original = b.generation.extract_program
        def fail(row):
            if self.round_index == 1:
                raise RecursionError("PRIVATE_COMPILER_TRACE")
            return original(row)
        with mock.patch.object(b.generation, "extract_program", side_effect=fail):
            result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["attempts"][-1]["extraction"], "compiler_resource")
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "passed"])
        self.verifier.assert_called_once()
        self.assertEqual(self.child.call_count, 2)

    def test_mixed_initial_outcomes_share_one_ordered_repair_round(self):
        for position in range(2, 5):
            binding, task = deepcopy(self.worklist[0])
            binding["id"] = position
            task["question"] = f"Statement {position}"
            path = self.root / f"fixture-{position}.json"
            b.generation.save(path, task)
            binding["sha256"] = b.generation.file_hash(path)
            self.worklist.append((binding, task))
        self.recipe["max_repairs"] = 1
        self.verdicts[0, 1] = "failed"
        self.draws.update({(0, 2): ("stop", "prose only"), (0, 3): ("length", "print(0)"),
                           (0, 4): ("stop", "```python\n```")})
        result = self.run_loop(reply_allowance=9, output_token_allowance=160)
        self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 2, 3, 4, 1, 2, 3, 4])
        self.assertEqual([a["seed"] for a in result["attempts"]][5:], [23, 24, 25, 26])
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed"] * 5)
        self.assertEqual([len(c.args[0]) for c in self.verifier.call_args_list], [2, 4])
        self.assertEqual(result["committed_tokens"], 160)

    def test_feedback_failure_stops_before_another_dispatch(self):
        self.verdicts[0, 0] = "failed"
        with mock.patch.object(b.repair_feedback, "render", side_effect=KeyboardInterrupt("render")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_loop()
        result = self.saved()
        self.assertEqual(result["phase"], "repair_preparation")
        self.assertEqual([t["state"] for t in result["tasks"]], ["cancelled", "passed"])
        self.assertEqual(result["committed_tokens"], 32)
        self.child.assert_called_once()

    def test_repair_preparation_failure_proves_no_new_submission(self):
        self.verdicts[0, 0] = "failed"
        original = b.generation.save
        def fail(path, value):
            if path.name == "round-0001.json":
                raise OSError("cannot save repair specification")
            return original(path, value)
        with mock.patch.object(b.generation, "save", side_effect=fail):
            result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "passed"])
        self.assertEqual(result["known_submissions"], 2)
        self.assertTrue(result["usage_complete"])
        self.assertFalse(result["attempts"][-1]["submitted"])
        self.assertEqual(result["committed_tokens"], 32)
        self.child.assert_called_once()

    def test_partial_repair_generation_keeps_native_evidence_without_execution(self):
        self.verdicts.update({(0, 0): "failed", (0, 1): "failed"})
        def fail(*args):
            result = self.generate(*args)
            if self.round_index == 1:
                path = self.root / "run/generation-0001/outcome.json"
                outcome = json.loads(path.read_text())
                outcome["status"] = "error"
                outcome["requests"][0]["status"] = "error"
                outcome["requests"][1].update(status="not_run", submitted=False, text="", tokens=[], generation_tokens=0)
                b.generation.save(path, outcome)
            return result
        self.child.side_effect = fail
        result = self.run_loop()
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "error"])
        self.assertEqual(result["attempts"][-1]["state"], "not_run")
        self.assertEqual(result["known_submissions"], 3)
        self.assertEqual(result["committed_tokens"], 72)
        self.assertFalse(result["usage_complete"])
        self.verifier.assert_called_once()
        self.assertEqual(self.child.call_count, 2)

    def test_verifier_error_on_repair_never_uses_another_allowed_draw(self):
        self.verdicts.update({(0, 0): "failed", (1, 0): "error"})
        result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "passed"])
        self.assertEqual(result["committed_replies"], 3)
        self.assertEqual(self.child.call_count, 2)

    def test_all_initial_passes_stop_without_repair(self):
        result = self.run_loop()
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
        self.child.assert_called_once()
        self.assertEqual(result["committed_replies"], 2)

    def test_later_verifier_error_preserves_budget_exhausted_peer(self):
        self.verdicts.update({(0, 0): "failed", (0, 1): "failed", (1, 0): "error"})
        result = self.run_loop(reply_allowance=3)
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "budget_exhausted"])
        self.assertEqual(result["committed_replies"], 3)

    def test_later_interrupt_preserves_budget_exhausted_peer(self):
        self.verdicts.update({(0, 0): "failed", (0, 1): "failed"})
        def interrupt(*args, **kwargs):
            if self.round_index == 1:
                raise KeyboardInterrupt("repair verification")
            return self.verify(*args, **kwargs)
        self.verifier.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(reply_allowance=3)
        self.assertEqual([t["state"] for t in self.saved()["tasks"]], ["cancelled", "budget_exhausted"])

    def test_saved_rounds_reference_single_attempt_records_and_own_child_identity(self):
        self.verdicts[0, 1] = "failed"
        def child(command, current, persist):
            code = self.generate(command, current, persist)
            current["child"] = dict(pid=123, pgid=123)
            persist()
            saved = self.saved()
            self.assertEqual(saved["rounds"][-1]["child"], current["child"])
            return code
        self.child.side_effect = child
        result = self.run_loop()
        saved = self.saved()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([r["attempt_ids"] for r in saved["rounds"]], [[0, 1], [2]])
        self.assertTrue(all("attempts" not in r for r in saved["rounds"]))
        for key in ("child", "round_file", "child_log", "generation_outcome"):
            self.assertNotIn(key, saved)
        self.assertEqual(len(saved["attempts"]), 3)
        self.assertEqual(saved, result)

    def test_initial_round_write_failure_records_definite_non_submission(self):
        original = b.generation.save
        def fail(path, value):
            if path.name == "round-0000.json":
                raise OSError("cannot save initial round")
            return original(path, value)
        with mock.patch.object(b.generation, "save", side_effect=fail):
            result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["not_run", "not_run"])
        self.assertEqual((result["committed_replies"], result["known_submissions"]), (0, 0))
        self.assertTrue(result["usage_complete"])
        self.assertEqual(self.saved(), result)
        self.child.assert_not_called()

    def test_failed_repair_launch_preserves_prior_usage_and_reservations(self):
        self.verdicts[0, 0] = "failed"
        def child(*args):
            return self.generate(*args) if self.child.call_count == 1 else self.actual_run_child(*args)
        self.child.side_effect = child
        result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([t["state"] for t in result["tasks"]], ["error", "passed"])
        self.assertIn("popen_error", result["rounds"][1])
        self.assertEqual(result["rounds"][1]["known_submissions"], 0)
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (2, 4))
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["committed_tokens"], 52)
        self.assertEqual(self.child.call_count, 2)
        self.verifier.assert_called_once()
