"""Finite witness rounds, prompt isolation, native accounting and stop behavior."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_witnesses import record

p = importlib.import_module("src.data-pipeline.stage04.proposals")


class ProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "model.json"
        self.manifest.write_text("{}")
        self.model = self.root / "model"
        self.model.mkdir()
        p.generation.save(self.model / "config.json", dict(model_type="qwen3_5"))
        self.worklist = [record(i) for i in (7, 8)]
        for i, item in enumerate(self.worklist):
            item["references"] = [r + b"# PRIVATE_REFERENCE" for r in item["references"]]
            item["task"] = dict(question=f"Statement {i}, including its examples.",
                solutions=[r.decode() for r in item["references"]], passing_reference_indices=[0, 1, 2],
                cases=[dict(input="PRIVATE_CASE", answer="PRIVATE_ANSWER")], private="PRIVATE_AUDIT_REPLY")
            item["stored_inputs"] += ["PRIVATE_STORED_INPUT"]
            item["other_program"] = "PRIVATE_OTHER_PROGRAM"
        self.recipe = dict(base_seed=17, temperature=.7, top_p=.95, max_tokens=64)
        self.texts = ["<witness>\r\n \t7\r\n</witness>", "<witness>\n8</witness>"]
        self.statuses, self.native_status = ["stop", "stop"], "success"
        self.actual_child = p.control.run_child
        self.child = mock.patch.object(p.control, "run_child", side_effect=self.generate).start()
        self.checker = mock.patch.object(p.witnesses, "check_many", side_effect=self.check).start()
        self.addCleanup(mock.patch.stopall)

    def generate(self, command, state, persist):
        self.assertEqual(command[1:3], ["-m", "src.data-pipeline.stage03.local_generation"])
        spec = json.loads(Path(command[-1]).read_text())
        before = self.saved()
        self.assertEqual(before["phase"], "dispatching")
        self.assertEqual((before["committed_replies"], before["committed_tokens"]),
                         (2, sum(r["max_tokens"] for r in spec["requests"])))
        self.assertIsNone(before["known_submissions"])
        self.assertTrue(all(a["submitted"] is None for a in before["attempts"]))
        state["child"] = dict(pid=123, pgid=123, returncode=0)
        persist()
        native = Path(spec["output_dir"])
        native.mkdir()
        p.generation.save(native / "inputs.json", dict(requests=spec["requests"]))
        p.generation.save(native / "runtime.json", dict(load_seconds=.1))
        rows = []
        for i, (text, status) in enumerate(zip(self.texts, self.statuses)):
            submitted = status != "not_run"
            rows.append(dict(text=text, status=status, submitted=submitted, generation_tokens=2 if submitted else 0))
            if submitted:
                (native / f"reply-{i:04d}.txt").write_bytes(text.encode())
        p.generation.save(native / "outcome.json", dict(status=self.native_status, requests=rows,
            preparation_seconds=.2, native=dict(wall_time=.3, generation_tokens=sum(r["generation_tokens"] for r in rows))))
        return 0

    def check(self, proposals, directory):
        self.assertEqual(self.saved()["phase"], "verification")
        self.assertEqual([a["state"] for a in self.saved()["attempts"] if "input" in a], ["checking"] * len(proposals))
        if proposals:
            directory.mkdir()
            (directory / "native.eval").touch()
        return [dict(status="execution_agrees", answer="7", log=str(directory / "native.eval")) for _ in proposals]

    def run_round(self, **changes):
        options = dict(model_path=self.model, model_manifest=self.manifest,
            mlx_python=self.root / "mlx-python", recipe=self.recipe, reply_allowance=2,
            output_token_allowance=128, completion_batch_size=8, prefill_batch_size=4,
            prefill_step_size=256, fixtures=True)
        options.update(changes)
        return p.run(self.worklist, self.root / "run", **options)

    def saved(self):
        return json.loads((self.root / "run/outcome.json").read_text())

    def test_one_child_then_one_check_exact_inputs_and_no_feedback_for_both_sources(self):
        for source in ("actor", "synthetic"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as temporary:
                self.root = Path(temporary)
                for item in self.worklist:
                    item["selection"]["candidate"]["source"] = source
                p.generation.save(self.model / "config.json", dict(model_type="qwen3_5" if source == "actor" else "mistral3"))
                original = deepcopy(self.worklist)
                result = self.run_round()
                self.assertEqual(result["status"], "completed")
                self.assertEqual([a["state"] for a in result["attempts"]], ["execution_agrees"] * 2)
                self.assertEqual(self.worklist, original)
                self.assertEqual(result, self.saved())
                self.child.assert_called_once()
                self.checker.assert_called_once()
                self.assertEqual([stdin for _, stdin in self.checker.call_args.args[0]], [" \t7\r\n", "8"])
                for attempt in result["attempts"]:
                    raw = Path(attempt["input"]).read_bytes()
                    self.assertEqual(p.generation.file_hash(attempt["input"]), attempt["stdin_sha256"])
                    self.assertEqual(raw.decode(), self.checker.call_args.args[0][attempt["attempt"]][1])
                    self.assertTrue((self.root / f"run/generation/reply-{attempt['attempt']:04d}.txt").exists())
                self.assertEqual((result["known_submissions"], result["observed_tokens"]), (2, 4))
                self.assertTrue(result["usage_complete"])
                self.assertEqual((result["timing"]["loading"], result["timing"]["preparation"], result["timing"]["generation"]), (.1, .2, .3))
                self.assertIsNotNone(result["timing"]["verification"])
                saved = json.loads((self.root / "run/inputs.json").read_text())
                self.assertEqual(saved["source"], source)
                self.assertTrue(saved["fixtures"])
                self.assertIsNone(saved["selection_sha256"])
                self.assertEqual(saved["model_manifest_sha256"], p.generation.file_hash(self.manifest))
                self.assertIn(str(Path(p.__file__)), saved["source_sha256"])
                for snapshot, item in zip(saved["worklist"], self.worklist):
                    self.assertEqual(Path(snapshot["program"]).read_bytes(), item["program"])
                    self.assertEqual(json.loads(Path(snapshot["task"]).read_text())["stored_inputs"], item["stored_inputs"])
                self.child.reset_mock()
                self.checker.reset_mock()

    def test_exact_messages_exclude_private_fields_and_other_programs(self):
        requests = p.requests(self.worklist, self.recipe)
        self.assertEqual([r["seed"] for r in requests], [17, 18])
        for row, record in zip(requests, self.worklist):
            self.assertEqual(row["messages"], [dict(role="system", content=p.SYSTEM), dict(role="user", content=
                "Problem statement:\n" + record["task"]["question"] + "\n\nYour program:\n" + record["program"].decode()
                + "\n\n" + p.INSTRUCTION)])
            self.assertNotIn("PRIVATE", json.dumps(row))
            self.assertEqual((row["max_tokens"], row["temperature"], row["top_p"]), (64, .7, .95))

    def test_omitted_recipe_uses_saved_defaults(self):
        result = self.run_round(recipe=None, output_token_allowance=2048)
        specification = json.loads((self.root / "run/round.json").read_text())
        self.assertEqual([(r["seed"], r["temperature"], r["top_p"], r["max_tokens"])
                          for r in specification["requests"]], [(17, .2, .95, 1024), (18, .2, .95, 1024)])
        self.assertEqual(result["committed_tokens"], 2048)
        self.assertEqual(json.loads((self.root / "run/inputs.json").read_text())["recipe"], p.DEFAULT_RECIPE)

    def test_length_format_and_encoding_failures_keep_raw_replies_without_redraw(self):
        self.statuses, self.texts = ["length", "stop"], ["<witness>\n7</witness>", "not a block"]
        result = self.run_round()
        self.assertEqual([a["state"] for a in result["attempts"]], ["token_limit", "format_failure"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual([(self.root / f"run/generation/reply-{a['attempt']:04d}.txt").read_text()
                          for a in result["attempts"]], self.texts)
        self.child.assert_called_once()
        self.assertEqual(self.checker.call_args.args[0], [])
        with mock.patch.object(p.witnesses, "extract_input", side_effect=UnicodeEncodeError("utf8", "x", 0, 1, "fixture")):
            self.root = self.root / "second"
            result = self.run_round()
        self.assertEqual(result["attempts"][1]["state"], "format_failure")

    def test_parse_failure_keeps_original_positions_and_later_seed(self):
        self.texts[0] = "bad format"
        result = self.run_round()
        self.assertEqual([a["state"] for a in result["attempts"]], ["format_failure", "execution_agrees"])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18])
        self.assertEqual(self.checker.call_args.args[0], [(self.worklist[1], "8")])

    def test_duplicate_and_benign_disagreement_are_saved_without_retry(self):
        self.checker.side_effect = lambda *args: [dict(status="duplicate", key={}),
            dict(status="benign_disagreement", answer="8", programs=dict(benign=dict(answer="9")))]
        result = self.run_round()
        self.assertEqual([a["state"] for a in result["attempts"]], ["duplicate", "benign_disagreement"])
        self.assertEqual(result["attempts"][1]["verification"]["programs"]["benign"]["answer"], "9")
        self.child.assert_called_once()
        self.assertEqual(result["committed_tokens"], 128)

    def test_partial_generation_error_retains_completed_unchecked_and_unstarted(self):
        self.native_status, self.statuses = "error", ["stop", "not_run"]
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        self.assertEqual([a["state"] for a in result["attempts"]], ["not_checked", "not_run"])
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (1, 2))
        self.assertFalse(result["usage_complete"])
        self.checker.assert_not_called()

    def test_missing_or_truncated_outcome_keeps_submissions_and_usage_uncertain(self):
        original = self.generate
        for raw in (None, '{"requests":', '{"status":"success","requests":[]}',
                    json.dumps(dict(status="success", native=None,
                        requests=[dict(submitted=True, generation_tokens=2)] * 2))):
            def broken(*args):
                original(*args)
                path = self.root / "run/generation/outcome.json"
                path.unlink() if raw is None else path.write_text(raw)
                return 0
            self.child.side_effect = broken
            with tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                result = self.run_round()
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["known_submissions"])
                self.assertFalse(result["usage_complete"])
                self.assertEqual([a["state"] for a in result["attempts"]], ["submission_unknown"] * 2)
                self.assertEqual(result["committed_tokens"], 128)
                self.checker.assert_not_called()

    def test_ambiguous_launch_and_native_interruption_preserve_evidence(self):
        self.child.side_effect = OSError("ambiguous launch")
        result = self.run_round()
        self.assertIsNone(result["known_submissions"])
        self.assertEqual([a["state"] for a in result["attempts"]], ["submission_unknown"] * 2)
        self.root = self.root / "second"
        error = KeyboardInterrupt("original interrupt")
        self.native_status, self.statuses = "cancelled", ["stop", "cancelled"]
        def interrupted(*args):
            self.generate(*args)
            raise error
        self.child.side_effect = interrupted
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_round()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.saved()["status"], "cancelled")
        self.assertEqual([a["state"] for a in self.saved()["attempts"]], ["not_checked", "cancelled"])
        self.checker.assert_not_called()

    def test_verification_failure_and_cancellation_keep_inputs_and_native_logs(self):
        for error in (RuntimeError("Docker infrastructure"), KeyboardInterrupt("native cancellation")):
            with tempfile.TemporaryDirectory() as directory:
                self.root = Path(directory)
                def failed(proposals, log_dir):
                    self.check(proposals, log_dir)
                    raise error
                self.checker.side_effect = failed
                if isinstance(error, Exception):
                    result = self.run_round()
                    self.assertEqual(result["status"], "error")
                    self.assertEqual([a["state"] for a in result["attempts"]], ["verification_error"] * 2)
                else:
                    error.cleanup_confirmed = True
                    with self.assertRaises(KeyboardInterrupt):
                        self.run_round()
                    self.assertTrue(self.saved()["verification_cleanup_confirmed"])
                self.assertEqual(len(self.saved()["verification_logs"]), 1)
                self.assertTrue(all(Path(a["input"]).exists() for a in self.saved()["attempts"]))

    def test_invalid_worklist_budgets_and_seed_fail_before_dispatch(self):
        for changes in (dict(reply_allowance=1), dict(output_token_allowance=127),
                        dict(recipe=dict(self.recipe, base_seed=2**64-1)), dict(recipe=dict(self.recipe, max_tokens=0))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.run_round(**changes)
        self.worklist[1]["selection"]["candidate"]["source"] = "synthetic"
        with self.assertRaisesRegex(ValueError, "one source"):
            self.run_round()
        self.worklist = []
        with self.assertRaises(ValueError):
            self.run_round()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_existing_destination_is_refused(self):
        (self.root / "run").mkdir()
        with self.assertRaises(FileExistsError):
            self.run_round()
        self.child.assert_not_called()

    def test_exhausted_task_identity_fails_before_output_or_dispatch(self):
        for digest, proof in p.witnesses.EXHAUSTED_TASKS.items():
            candidate = self.worklist[0]["selection"]["candidate"]
            candidate.update(id=proof["id"], task_sha256=digest)
            with self.subTest(task=proof["id"]), self.assertRaisesRegex(ValueError, "Input domain exhausted"):
                self.run_round()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        candidate["task_sha256"] = "a" * 64
        self.assertEqual(self.run_round()["status"], "completed")

    def test_source_model_family_mismatches_fail_before_output_or_dispatch(self):
        for source, model_type in (("actor", "mistral3"), ("synthetic", "qwen3_5")):
            for item in self.worklist:
                item["selection"]["candidate"]["source"] = source
            p.generation.save(self.model / "config.json", dict(model_type=model_type))
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "requires model_type"):
                self.run_round()
        self.child.assert_not_called()
        self.checker.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_invalid_batching_fails_before_output_or_dispatch(self):
        for changes in (dict(prefill_batch_size=9), dict(completion_batch_size=0), dict(prefill_batch_size=0),
                        dict(prefill_step_size=0), dict(prefill_step_size=-1), dict(prefill_step_size=True),
                        dict(prefill_batch_size=1.5)):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "positive integer settings"):
                self.run_round(**changes)
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        result = self.run_round(completion_batch_size=4, prefill_batch_size=4)
        self.assertEqual(result["status"], "completed")
        self.child.assert_called_once()

    def test_snapshot_failure_preserves_unstarted_attempts_and_zero_reservations(self):
        save = p.generation.save
        def fail_snapshot(path, value):
            if Path(path).name == "task-0000.json":
                raise OSError("cannot save task")
            return save(path, value)
        with mock.patch.object(p.generation, "save", side_effect=fail_snapshot):
            result = self.run_round()
        self.assertEqual((result["status"], result["phase"]), ("error", "preparation"))
        self.assertEqual([a["state"] for a in result["attempts"]], ["not_run"] * 2)
        self.assertEqual((result["committed_replies"], result["known_submissions"]), (0, 0))
        self.assertEqual(result, self.saved())
        self.child.assert_not_called()

    def test_input_write_failure_preserves_extracted_peer_without_verification(self):
        write = Path.write_bytes
        def fail_input(path, data):
            if path.name == "input-0001.txt":
                raise OSError("cannot save input")
            return write(path, data)
        with mock.patch.object(Path, "write_bytes", autospec=True, side_effect=fail_input):
            result = self.run_round()
        self.assertEqual((result["status"], result["phase"]), ("error", "extraction"))
        self.assertEqual([a["state"] for a in result["attempts"]], ["extracted", "not_checked"])
        self.assertEqual(Path(result["attempts"][0]["input"]).read_bytes(), b" \t7\r\n")
        self.assertEqual((result["known_submissions"], result["observed_tokens"]), (2, 4))
        self.checker.assert_not_called()

    def test_real_missing_interpreter_confirms_zero_submissions(self):
        self.child.side_effect = self.actual_child
        result = self.run_round()
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["known_submissions"], 0)
        self.assertTrue(result["usage_complete"])
        self.assertEqual([a["state"] for a in result["attempts"]], ["not_run"] * 2)
        self.assertEqual(result["committed_tokens"], 128)

    def test_real_child_reads_round_file_then_exits_before_checking(self):
        interpreter = self.root / "fixture-python"
        interpreter.write_text(f"#!{sys.executable}\n" +
            "import json,sys\nfrom pathlib import Path\n"
            "assert sys.argv[1:3]==['-m','src.data-pipeline.stage03.local_generation']\n"
            "spec=json.loads(Path(sys.argv[-1]).read_text())\n"
            "native=Path(spec['output_dir']); native.mkdir()\nrows=[]\n"
            "for i,request in enumerate(spec['requests']):\n"
            " text='<witness>\\n'+str(7+i)+'</witness>'\n"
            " (native/f'reply-{i:04d}.txt').write_text(text)\n"
            " rows.append(dict(status='stop',text=text,submitted=True,generation_tokens=2))\n"
            "(native/'outcome.json').write_text(json.dumps(dict(status='success',requests=rows,native=None)))\n")
        interpreter.chmod(0o755)
        self.child.side_effect = self.actual_child
        result = self.run_round(mlx_python=interpreter)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["child"]["returncode"], 0)
        self.checker.assert_called_once()


if __name__ == "__main__":
    unittest.main()
