"""One-off caller: fixed worklist, bounded dispatch and stop/report behavior."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import batch09_development as c

s = importlib.import_module("src.data-pipeline.stage03.experiment_summary")


class Batch09CallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gate = self.root / "gate"
        self.gate.mkdir()
        self.gate_record = dict(status="passed", source_unchanged=True,
            runs=[dict(source=source) for source in ("actor", "synthetic")],
            source_sha256_after={str(Path(module.__file__)): c.g.file_hash(module.__file__) for module in (c.b, c.g)})
        c.g.save(self.gate / "qualification.json", self.gate_record)
        self.tasks = [(dict(split="test", id=i), dict(question=f"Statement {i}")) for i in range(8)]
        self.addCleanup(mock.patch.stopall)
        self.worklist_patch = mock.patch.object(c, "worklist", return_value=self.tasks)
        self.worklist_patch.start()
        self.generate = mock.patch.object(c.b, "run", return_value=dict(status="completed")).start()
        self.report = self.root / "summary.json"
        self.experiments = self.root / "experiments"

    def run_caller(self):
        return c.run(self.gate, root=self.experiments, report_path=self.report)

    def test_two_sources_share_eight_tasks_and_follow_existing_numbering(self):
        self.experiments.mkdir()
        (self.experiments / "0007-earlier").mkdir()
        self.run_caller()
        calls = self.generate.call_args_list
        self.assertEqual(len(calls), 2)
        for call, source, number in zip(calls, ("actor", "synthetic"), (8, 9), strict=True):
            self.assertEqual(call.args, (self.tasks, self.experiments / f"{number:04d}-batch09-{source}"))
            self.assertEqual(call.kwargs["source"], source)
            self.assertEqual(call.kwargs["reply_allowance"], 16)
            self.assertEqual(call.kwargs["output_token_allowance"], 16384)
            self.assertEqual(call.kwargs["recipe"]["max_repairs"], 1)
        self.assertEqual(json.loads(self.report.read_text()), s.summarize(self.experiments))

    def test_technical_failure_stops_before_second_source_and_saves_summary(self):
        self.generate.return_value = dict(status="error")
        with self.assertRaisesRegex(RuntimeError, "actor experiment stopped"):
            self.run_caller()
        self.generate.assert_called_once()
        self.assertTrue(self.report.exists())

    def test_interruption_still_saves_summary_without_second_source(self):
        self.generate.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_caller()
        self.generate.assert_called_once()
        self.assertTrue(self.report.exists())

    def test_failed_gate_or_changed_sources_prevent_dispatch(self):
        for kind in ("status", "sources"):
            with self.subTest(kind=kind):
                record = dict(self.gate_record)
                if kind == "status":
                    record["status"] = "review_stop"
                else:
                    record["source_sha256_after"] = {str(Path(c.b.__file__)): "0" * 64}
                c.g.save(self.gate / "qualification.json", record)
                with self.assertRaisesRegex(ValueError, "Require the passing two-model"):
                    self.run_caller()
        self.generate.assert_not_called()

    def test_existing_experiment_prevents_repeat_without_summary(self):
        (self.experiments / "0001-batch09-actor").mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "Batch 9 experiment already exists"):
            self.run_caller()
        self.generate.assert_not_called()

    def test_summary_can_be_overwritten_when_no_batch09_experiment_exists(self):
        self.report.write_text("{}")
        self.run_caller()
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(json.loads(self.report.read_text()), s.summarize(self.experiments))

    def test_validation_failure_without_experiment_directory_allows_corrected_call(self):
        self.generate.side_effect = ValueError("invalid recipe before setup")
        with self.assertRaisesRegex(ValueError, "invalid recipe"):
            self.run_caller()
        self.assertTrue(self.report.exists())
        self.generate.side_effect = None
        self.run_caller()
        self.assertEqual(self.generate.call_count, 3)

    def test_worklist_reads_only_the_eight_saved_development_ids(self):
        self.worklist_patch.stop()
        cohort = dict(groups=dict(development=dict(initial=list(range(10, 60)))), protected="do not read")
        with mock.patch.object(c.b.study_tasks, "load_cohort", return_value=cohort), \
             mock.patch.object(c.b.study_tasks, "load_task", side_effect=lambda _, task_id: task_id) as load:
            self.assertEqual(c.worklist(), list(range(10, 18)))
        self.assertEqual(load.call_args_list, [mock.call(cohort, task_id) for task_id in range(10, 18)])
