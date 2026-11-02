"""Initial-round association, accounting, privacy and failure boundaries."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

b = importlib.import_module("src.data-pipeline.stage03.generation_loop")
from tests.test_verification import v


class LoopFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}")
        self.worklist = []
        for i in range(2):
            task = dict(question=f"Statement {i}", reference="PRIVATE_REFERENCE", cases=[
                dict(index=0, input="visible", answer="1", visible=True),
                dict(index=979797, input="PRIVATE_HIDDEN", answer="898989", visible=False)])
            path = self.root / f"fixture-{i}.json"
            b.generation.save(path, task)
            self.worklist.append((dict(split="train", id=i, sha256=b.generation.file_hash(path),
                owner="construction", phase="initial", private="PRIVATE_BINDING"), task))
        self.recipe = dict(system="System instruction", initial="Initial instruction", temperature=.2,
                           top_p=.95, base_seed=17, max_tokens=16, max_repairs=0)
        self.texts, self.row_statuses, self.native_status = ["print(1)\r\n", "```py\nprint(2)\n```"], None, "success"
        self.answer_texts = {}
        self.addCleanup(mock.patch.stopall)
        self.actual_run_child = b.run_child
        self.child = mock.patch.object(b, "run_child", side_effect=self.generate).start()
        self.verifier = mock.patch.object(v, "verify_many", side_effect=self.verify).start()

    def generate(self, command, progress, persist):
        self.assertEqual(command[1:3], ["-m", b.generation.__name__])
        saved = json.loads((self.root / "run/outcome.json").read_text())
        self.assertEqual(saved["phase"], "dispatching")
        self.assertEqual(saved["committed_replies"], len(self.worklist))
        self.assertEqual(saved["committed_tokens"], len(self.worklist) * self.recipe["max_tokens"])
        self.assertTrue(all(a["state"] == "submission_unknown" for a in saved["attempts"]))
        spec = json.loads(Path(command[-1]).read_text())
        directory = Path(spec["output_dir"])
        directory.mkdir()
        b.generation.save(directory / "inputs.json", dict(requests=spec["requests"],
            model_manifest_sha256=b.generation.file_hash(self.manifest)))
        b.generation.save(directory / "runtime.json", dict(load_seconds=.1))
        rows = []
        for i, text in enumerate(self.texts):
            status = self.row_statuses[i] if self.row_statuses else "stop"
            submitted = status != "not_run"
            tokens = [12, 99] if submitted else []
            rows.append(dict(status=status, submitted=submitted, native_uid=40+i if submitted else None,
                text=text, answer_text=text, tokens=tokens, generation_tokens=len(tokens), prompt_tokens=7))
            if submitted:
                (directory / f"reply-{i:04d}.txt").write_bytes(text.encode())
        b.generation.save(directory / "outcome.json", dict(status=self.native_status, requests=rows,
            preparation_seconds=.2, native=dict(generation_tokens=sum(len(r["tokens"]) for r in rows), wall_time=.3)))
        return 0

    def verify(self, candidates, directory, **kwargs):
        directory.mkdir()
        log = directory / "native.eval"
        log.touch()
        return [dict(status="passed", key=dict(task_sha256=digest,
            program_sha256=b.study_tasks.candidate_identity("train", i, digest, "actor", program)["program_sha256"],
            contract_sha256="c" * 64), log=str(log), sample_id=i,
            feedback=dict(status="passed", visible=[], hidden="passed")) for i, (_, digest, program) in enumerate(candidates)]

    def run_loop(self, **changes):
        options = dict(source="actor", model_path=self.root / "model", model_manifest=self.manifest,
            mlx_python=self.root / "venv/bin/python", recipe=self.recipe, reply_allowance=7,
            output_token_allowance=100, completion_batch_size=8, prefill_batch_size=4, prefill_step_size=256,
            fixtures=True)
        options.update(changes)
        return b.run(self.worklist, self.root / "run", **options)

    def saved(self):
        return json.loads((self.root / "run/outcome.json").read_text())


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class LoopTests(LoopFixture):
    def test_custom_initial_messages_are_copied_and_validated_before_dispatch(self):
        messages = [[dict(role="system", content="Custom system"), dict(role="user", content=f"Custom {i}")]
                    for i in range(2)]
        for invalid in ([], [messages[0]], [messages[0], "invalid"],
                        [messages[0], [dict(role="user", content="wrong role"), messages[1][1]]]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "system/user prompts"):
                self.run_loop(initial_messages=invalid)
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        original = deepcopy(messages)
        requests = b.initial_requests(self.worklist, self.recipe, messages)
        result = self.run_loop(initial_messages=messages)
        messages[0][0]["content"] = "mutated"
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([r["messages"] for r in requests], original)
        saved = json.loads((self.root / "run/round-0000.json").read_text())
        self.assertEqual([r["messages"] for r in saved["requests"]], original)

    def test_prior_conversations_are_rejected_before_dispatch(self):
        history = [dict(role="system", content="System"), dict(role="user", content="Original"),
                   dict(role="assistant", content="print(0)"), dict(role="user", content="Public feedback")]
        with self.assertRaisesRegex(ValueError, "system/user prompts"):
            self.run_loop(initial_messages=[history, history])
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_checker_count_and_order_are_checked_before_acceptance(self):
        for mode in ("short", "long", "reversed"):
            with self.subTest(mode=mode):
                def mismatched(candidates, directory, **kwargs):
                    rows = LoopFixture.verify(self, candidates, directory, **kwargs)
                    return rows[:1] if mode == "short" else rows + rows[:1] if mode == "long" else rows[::-1]

                try:
                    result = self.run_loop(candidate_checker=mismatched)
                    self.assertEqual(result["status"], "error", result)
                    self.assertIn("Checker verdicts", result["error"])
                    self.assertTrue(all(t["state"] != "passed" for t in result["tasks"]))
                finally:
                    shutil.rmtree(self.root / "run")

    def test_order_exact_bytes_private_snapshots_and_coalesced_verification(self):
        original = deepcopy(self.worklist)
        result = self.run_loop()
        self.assertEqual(result["status"], "completed")
        self.assertEqual([a["state"] for a in result["attempts"]], ["passed", "passed"])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18])
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (2, 4))
        self.assertTrue(result["usage_complete"])
        self.assertEqual(result["timing"]["loading"], .1)
        self.assertEqual(result["timing"]["preparation"], .2)
        self.assertEqual(result["timing"]["generation"], .3)
        self.assertIsNotNone(result["timing"]["verification"])
        self.assertEqual(self.worklist, original)
        self.verifier.assert_called_once()
        candidates = self.verifier.call_args.args[0]
        self.assertEqual([c[2] for c in candidates], [b"print(1)\r\n", b"print(2)\n"])
        self.assertEqual(self.verifier.call_args.kwargs, dict(max_samples=4, concurrent_inference=False))
        for i, (binding, _) in enumerate(self.worklist):
            attempt = result["attempts"][i]
            self.assertEqual(Path(attempt["program"]).read_bytes(), candidates[i][2])
            self.assertEqual(attempt["candidate"]["task_sha256"], binding["sha256"])
            self.assertEqual(attempt["verification"]["sample_id"], i)
        inputs = json.loads((self.root / "run/inputs.json").read_text())
        self.assertEqual([s["binding"] for s in inputs["worklist"]], [b for b, _ in original])
        self.assertEqual(inputs["mlx_python"], str(self.root / "venv/bin/python"))
        self.assertIn(str(Path(b.__file__)), inputs["source_sha256"])
        module_root = Path(b.__file__).parents[1]
        for name in ("artifacts.py", "paths.py", "stage03/study_tasks.py"):
            path = module_root / name
            self.assertEqual(inputs["source_sha256"][str(path)], b.generation.file_hash(path))
        for snapshot, (_, task) in zip(inputs["worklist"], original):
            self.assertEqual(json.loads(Path(snapshot["snapshot"]).read_text()), task)
            self.assertEqual(snapshot["snapshot_sha256"], b.generation.file_hash(snapshot["snapshot"]))
        requests = json.loads((self.root / "run/round-0000.json").read_text())["requests"]
        self.assertTrue(all(r["enable_thinking"] is False for r in requests))
        rendered = json.dumps(requests)
        for secret in ("PRIVATE", "979797", "898989", "construction", "sha256"):
            self.assertNotIn(secret, rendered)
        self.assertEqual(requests[0]["messages"], [dict(role="system", content=self.recipe["system"]),
            dict(role="user", content="Statement 0\n\n" + self.recipe["initial"])])
        self.assertEqual(self.saved(), result)

    def test_length_format_and_syntax_failures_are_not_executed_or_repaired(self):
        self.texts = ["print(1)", "```python\n```"]
        self.row_statuses = ["length", "stop"]
        result = self.run_loop()
        self.assertEqual([a["extraction"] for a in result["attempts"]], ["length", "format"])
        self.assertEqual([a["state"] for a in result["attempts"]], ["repairs_exhausted"] * 2)
        self.assertEqual(result["status"], "completed")
        self.verifier.assert_not_called()
        self.child.assert_called_once()

    def test_syntax_failure_keeps_candidate_order_for_surviving_programs(self):
        self.texts[0] = "Here is Python code:"
        result = self.run_loop()
        self.assertEqual(result["attempts"][0]["extraction"], "syntax")
        self.assertEqual(result["attempts"][1]["verification"]["sample_id"], 0)
        self.assertEqual(self.verifier.call_args.args[0][0][1], self.worklist[1][0]["sha256"])

    def test_deep_compilable_program_does_not_abort_comment_stripping(self):
        self.recipe["strip_comments"] = True
        self.texts[0] = "\n".join(("if" if i == 0 else "elif") + f" value == {i}:\n    print({i}) # branch"
                                  for i in range(400)) + "\n"
        result = self.run_loop()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["state"] for a in result["attempts"]], ["passed", "passed"])
        self.assertNotIn(b"# branch", Path(result["attempts"][0]["program"]).read_bytes())
        self.assertEqual(len(self.verifier.call_args.args[0]), 2)

    def test_compiler_resource_exhaustion_is_a_technical_stop(self):
        for error in (MemoryError("compile"), RecursionError("compile")):
            with self.subTest(error=error), mock.patch.object(b.generation, "extract_program", side_effect=error):
                result = self.run_loop()
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["attempts"][0]["extraction"], "compiler_resource")
            self.assertEqual([a["state"] for a in result["attempts"]], ["error"] * 2)
            self.assertEqual((self.root / "run/generation-0000/reply-0000.txt").read_bytes(), self.texts[0].encode())
            self.verifier.assert_not_called()
            self.child.assert_called_once()
            import shutil
            shutil.rmtree(self.root / "run")
            self.child.reset_mock()

    def test_missing_truncated_or_incomplete_outcome_leaves_submission_unknown(self):
        for raw in (None, '{"requests":', '{"status":"success","requests":[]}'):
            original = self.generate
            def incomplete(*args):
                original(*args)
                path = self.root / "run/generation-0000/outcome.json"
                path.unlink() if raw is None else path.write_text(raw)
                return 0
            self.child.side_effect = incomplete
            result = self.run_loop()
            self.assertEqual(result["status"], "error")
            self.assertEqual([a["state"] for a in result["attempts"]], ["submission_unknown"] * 2)
            self.assertIsNone(result["known_submissions"])
            self.assertFalse(result["usage_complete"])
            self.assertEqual(result["committed_tokens"], 32)
            if raw is not None:
                self.assertEqual((self.root / "run/generation-0000/outcome.json").read_text(), raw)
            self.verifier.assert_not_called()
            import shutil
            shutil.rmtree(self.root / "run")

    def test_generation_failure_preserves_completed_replies_without_execution(self):
        self.native_status, self.row_statuses = "error", ["error", "not_run"]
        result = self.run_loop()
        self.assertEqual([a["state"] for a in result["attempts"]], ["error", "not_run"])
        self.assertEqual(result["known_submissions"], 1)
        self.assertFalse(result["usage_complete"])
        self.verifier.assert_not_called()

    def test_interruption_preserves_original_exception_and_available_native_evidence(self):
        original = self.generate
        error = KeyboardInterrupt("original interrupt")
        self.native_status, self.row_statuses = "cancelled", ["cancelled", "not_run"]
        def interrupt(*args):
            original(*args)
            raise error
        self.child.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_loop()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.saved()["status"], "cancelled")
        self.assertEqual([a["state"] for a in self.saved()["attempts"]], ["cancelled", "not_run"])
        self.assertEqual(self.saved()["committed_tokens"], 32)
        self.verifier.assert_not_called()

    def test_verification_error_retains_passing_peer_and_never_retries(self):
        original = self.verify
        def failed(*args, **kwargs):
            results = original(*args, **kwargs)
            results[0]["status"] = "error"
            return results
        self.verifier.side_effect = failed
        result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual([a["state"] for a in result["attempts"]], ["error", "passed"])
        self.verifier.assert_called_once()
        self.child.assert_called_once()

    def test_wrong_answer_completes_with_repairs_exhausted_and_keeps_feedback(self):
        def wrong(*args, **kwargs):
            results = self.verify(*args, **kwargs)
            results[1].update(status="failed", feedback=dict(status="failed", hidden="failed", visible=[]))
            return results
        self.verifier.side_effect = wrong
        result = self.run_loop()
        self.assertEqual(result["status"], "completed")
        self.assertEqual([a["state"] for a in result["attempts"]], ["passed", "repairs_exhausted"])
        self.assertEqual(result["attempts"][1]["verification"]["feedback"]["hidden"], "failed")
        self.child.assert_called_once()

    def test_verification_interrupt_records_native_log_and_cleanup_uncertainty(self):
        error = KeyboardInterrupt("teardown interrupted")
        def interrupt(candidates, directory, **kwargs):
            directory.mkdir()
            (directory / "cancelled.eval").touch()
            raise error
        self.verifier.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_loop()
        self.assertIs(caught.exception, error)
        self.assertFalse(self.saved()["verification_cleanup_confirmed"])
        self.assertTrue(self.saved()["verification_logs"][0].endswith("cancelled.eval"))
        self.assertEqual([a["state"] for a in self.saved()["attempts"]], ["cancelled"] * 2)

    def test_invalid_experiment_and_seed_schedule_are_refused_before_dispatch(self):
        for changes in (dict(source="bad"), dict(reply_allowance=1), dict(output_token_allowance=31),
                        dict(reply_allowance=True), dict(completion_batch_size=False), dict(fixtures=False)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_loop(**changes)
        for key, values in (("max_repairs", [1, -1, True]), ("max_tokens", [0, True]),
                            ("enable_thinking", ["true", 1, 0, None]),
                            ("strip_comments", ["true", 1, 0, None]),
                            ("base_seed", [-1, True, 2**64-1, 1.5])):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.run_loop(recipe={**self.recipe, key: value})
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        self.recipe["base_seed"] = 2**64 - 2
        result = self.run_loop()
        self.assertEqual([a["seed"] for a in result["attempts"]], [2**64-2, 2**64-1])

    def test_duplicate_empty_and_existing_destinations_are_refused(self):
        original = self.worklist
        for worklist in ([], [original[0], original[0]]):
            self.worklist = worklist
            with self.assertRaises(ValueError):
                self.run_loop()
        self.worklist = original
        (self.root / "run").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_loop()
        self.child.assert_not_called()

    def test_ambiguous_launch_commits_allowances_and_stops(self):
        self.child.side_effect = OSError("launch interrupted")
        result = self.run_loop()
        self.assertEqual(result["committed_replies"], 2)
        self.assertEqual(result["committed_tokens"], 32)
        self.assertEqual([a["state"] for a in result["attempts"]], ["submission_unknown"] * 2)
        self.verifier.assert_not_called()

    def test_missing_interpreter_is_confirmed_not_submitted(self):
        self.child.side_effect = self.actual_run_child
        result = self.run_loop()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["phase"], "dispatching")
        self.assertEqual([t["state"] for t in result["tasks"]], ["not_run"] * 2)
        current = result["rounds"][0]
        self.assertIn("FileNotFoundError", current["popen_error"])
        self.assertNotIn("child", current)
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (0, 0))
        self.assertTrue(result["usage_complete"])
        self.assertEqual((result["committed_replies"], result["committed_tokens"]), (2, 32))
        self.verifier.assert_not_called()

    def test_interrupted_popen_without_child_record_remains_ambiguous(self):
        self.child.side_effect = self.actual_run_child
        with mock.patch.object(b.subprocess, "Popen", side_effect=KeyboardInterrupt("launch")):
            with self.assertRaises(KeyboardInterrupt):
                self.run_loop()
        result = self.saved()
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual([t["state"] for t in result["tasks"]], ["submission_unknown"] * 2)
        self.assertNotIn("child", result["rounds"][0])
        self.assertNotIn("popen_error", result["rounds"][0])
        self.assertIsNone(result["known_submissions"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(result["committed_tokens"], 32)
        self.verifier.assert_not_called()
