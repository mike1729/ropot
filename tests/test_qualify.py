import copy
import hashlib
import importlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from test_screen import task


q = importlib.import_module("src.data-pipeline.stage01.qualify")


class FakeRunner:
    provenance = {"interpreter": "test-only", "version": "3.12.fake"}

    def __init__(self):
        self.calls = []
        self.failures = set()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def run_reference(self, source, cases):
        self.calls.append((source, copy.deepcopy(cases)))
        passed = source not in self.failures
        return {"passed": passed, "max_case_seconds": 0.01, "wall_seconds": 0.02,
                "cases": [{"index": case["index"], "status": "passed" if passed else "wrong_answer",
                           "elapsed_seconds": 0.01, "stdout": case["answer"] if passed else "different"} for case in cases]}


class QualificationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.inputs = self.root / "raw"
        self.inputs.mkdir()
        self.output = self.root / "qualified"
        self.files = {}
        self.rows = {}
        for split in ("train", "test"):
            other = {**task(), "id": 8, "difficulty": "competition"}
            self.rows[split] = [task(), other]
        runtime = mock.Mock(version_info=(3, 12), version="test Python 3.12", executable="/test/python3.12")
        for name, value in (("FILES", self.files), ("EXPECTED_ROWS", 2), ("sys", runtime)):
            patcher = mock.patch.object(q, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write_inputs()

    def write_inputs(self):
        for split, rows in self.rows.items():
            data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
            (self.inputs / f"{split}.jsonl").write_bytes(data)
            self.files[split] = (len(data), hashlib.sha256(data).hexdigest())

    def test_pinned_integrity_and_provenance_without_parsing_rows(self):
        with mock.patch.object(q, "read_rows", side_effect=AssertionError("must not reparse pinned files")):
            evidence = q.verify_inputs(self.inputs)
        self.assertEqual({entry["file"] for entry in evidence}, {"train.jsonl", "test.jsonl"})
        for entry in evidence:
            split = entry["file"].split(".")[0]
            self.assertEqual((entry["bytes"], entry["sha256"]), self.files[split])
            self.assertEqual(entry["rows"], 2)
        self.assertEqual(list(q.read_rows(self.inputs, "train")), self.rows["train"])
        path = self.inputs / "test.jsonl"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            q.verify_inputs(self.inputs)
        self.write_inputs()
        data = path.read_bytes()
        path.write_bytes(data.replace(b'"id": 7', b'"id": 9'))
        self.assertEqual(path.stat().st_size, len(data))
        with self.assertRaisesRegex(ValueError, "differs from pinned APPS"):
            q.verify_inputs(self.inputs)

    def test_preserves_original_lengths_attempt_order_and_stops_after_three(self):
        row = task()
        row["question"] = row["question"].replace("Echo", "Echo é")
        row["extra"] = {"original": "✓\n"}
        sources = json.loads(row["solutions"])
        failing = "raise RuntimeError('bad reference')\n"
        row["solutions"] = json.dumps([sources[0], "# duplicate\n" + sources[0], failing,
                                       *sources[1:], "print(12345)\n"])
        original = copy.deepcopy(row)
        runner = FakeRunner()
        runner.failures.add(failing)
        result = q.qualify_task(row, "test", runner)
        self.assertEqual(row, original)
        self.assertEqual(result["original_row"], original)
        self.assertEqual(result["passing_reference_indices"], [0, 3, 4])
        self.assertEqual(result["reference_counts"], {"raw": 6, "distinct": 5, "passing": 3})
        self.assertEqual([a["source_index"] for a in result["verification_attempts"]], [0, 2, 3, 4])
        self.assertEqual([a["passed"] for a in result["verification_attempts"]], [True, False, True, True])
        self.assertEqual([call[0] for call in runner.calls], [sources[0], failing, *sources[1:]])
        self.assertTrue(all(len(cases) == 2 for _, cases in runner.calls))
        raw_cases = json.loads(row["input_output"])
        lengths = {"question": row["question"], "inputs": raw_cases["inputs"],
                   "outputs": raw_cases["outputs"], "solutions": json.loads(row["solutions"])}
        def measure(value):
            return {"characters": len(value), "bytes": len(value.encode("utf-8"))}
        for field, values in lengths.items():
            self.assertEqual(result["lengths"][field], measure(values) if isinstance(values, str)
                             else [measure(value) for value in values])

    def test_reference_failures_are_exclusions_but_infrastructure_errors_abort(self):
        runner = FakeRunner()
        runner.failures.add(json.loads(task()["solutions"])[0])
        with self.assertRaisesRegex(q.Rejected, "^insufficient_passing_references$") as caught:
            q.qualify_task(task(), "test", runner)
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(caught.exception.attempts[0]["failure"]["stdout"], "different")
        for restored in (copy.copy(caught.exception), copy.deepcopy(caught.exception), pickle.loads(pickle.dumps(caught.exception))):
            self.assertIsInstance(restored, q.FailedReferences)
            self.assertEqual((str(restored), restored.attempts), (str(caught.exception), caught.exception.attempts))
        for error in (OSError("runner failed"), ValueError("pipeline failed")):
            with mock.patch.object(runner, "run_reference", side_effect=error), \
                    self.assertRaises(type(error)):
                q.qualify_task(task(), "test", runner)
        for row, split in ((task(), "validation"), ({**task(), "id": True}, "test")):
            with self.assertRaises(ValueError) as caught:
                q.qualify_task(row, split, runner)
            self.assertNotIsInstance(caught.exception, q.Rejected)

    def test_manifest_task_files_counts_and_provenance(self):
        with mock.patch.object(q, "LocalRunner", FakeRunner):
            manifest = q.qualify(self.inputs, self.output)
        self.assertEqual(manifest["dataset"], "codeparrot/apps")
        self.assertEqual(manifest["revision"], q.REVISION)
        self.assertEqual(manifest["runtime"], FakeRunner.provenance)
        self.assertEqual(manifest["workers"], 8)
        self.assertEqual(manifest["accepted_counts"], {"train": 1, "test": 1})
        self.assertEqual(manifest["exclusion_reasons"], {"unsupported_difficulty": 2})
        self.assertEqual(manifest["tasks"], ["tasks/test_7.json", "tasks/train_7.json"])
        self.assertEqual(manifest["pipeline"]["schema_version"], 1)
        source = Path(q.__file__)
        self.assertEqual(manifest["pipeline"]["source_sha256"]["stage01/" + source.name],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(manifest["files"], q.verify_inputs(self.inputs))
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        self.assertEqual({p.name for p in (self.output / "tasks").iterdir()},
                         {"train_7.json", "test_7.json"})
        for split in ("train", "test"):
            saved = json.loads((self.output / "tasks" / f"{split}_7.json").read_text())
            self.assertEqual(saved["original_row"], self.rows[split][0])
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with mock.patch.object(q, "LocalRunner", FakeRunner), self.assertRaises(FileExistsError):
            q.qualify(self.inputs, self.output)
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_failed_run_leaves_no_destination_or_temporary_output(self):
        runner = FakeRunner()
        original = runner.run_reference
        def fail_after_first_task(source, cases):
            if len(runner.calls) == 3:
                raise ValueError("late pipeline failure")
            return original(source, cases)
        runner.run_reference = fail_after_first_task
        with mock.patch.object(q, "LocalRunner", return_value=runner), \
                self.assertRaisesRegex(ValueError, "late pipeline failure") as caught:
            q.qualify(self.inputs, self.output, workers=1)
        self.assertEqual(caught.exception.__notes__, ["While qualifying APPS task test_7"])
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.iterdir()), [self.inputs])

    def test_insufficient_passes_are_counted_as_task_exclusions(self):
        runner = FakeRunner()
        runner.failures.add(json.loads(task()["solutions"])[0])
        original = runner.run_reference
        def missing_library(source, cases):
            result = original(source, cases)
            if not result["passed"]:
                result["cases"] = [{**result["cases"][0], "status": "nonzero_exit", "returncode": 1,
                                    "stderr": "ModuleNotFoundError: No module named 'numpy'\n"}]
            return result
        runner.run_reference = missing_library
        with mock.patch.object(q, "LocalRunner", return_value=runner):
            manifest = q.qualify(self.inputs, self.output, workers=1)
        self.assertEqual(manifest["tasks"], [])
        self.assertEqual(manifest["exclusion_reasons"],
                         {"insufficient_passing_references": 2, "unsupported_difficulty": 2})
        self.assertEqual(manifest["reference_rejections"],
                         ["reference_rejections/test_7.json", "reference_rejections/train_7.json"])
        evidence = json.loads((self.output / manifest["reference_rejections"][0]).read_text())
        self.assertEqual([a["source_index"] for a in evidence["attempts"]], [0, 1, 2])
        failure = evidence["attempts"][0]["failure"]
        self.assertEqual((failure["index"], failure["status"], failure["returncode"]), (0, "nonzero_exit", 1))
        self.assertIn("No module named 'numpy'", failure["stderr"])
        self.assertTrue(all(a["failure"] is None for a in evidence["attempts"][1:]))

    def test_runtime_and_worker_validation_precede_input_access(self):
        for workers in (0, 9, True):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                q.qualify(self.root / "absent", self.output, workers)
        with mock.patch.object(q.sys, "version_info", (3, 14)), \
                self.assertRaisesRegex(RuntimeError, "preprocessing requires Python 3.12"):
            q.qualify(self.inputs, self.output)

    def test_rows_are_streamed_with_bounded_pending_work(self):
        yielded, written = [], []
        original_write = q.write_json
        def rows(input_dir, split):
            for index in range(60):
                yielded.append((split, index))
                self.assertLessEqual(len(yielded) - len(written), 2)
                yield {**task(), "id": index}
        def write(path, value):
            if path.parent.name == "tasks":
                written.append(path.name)
            original_write(path, value)
        with mock.patch.object(q, "verify_inputs", return_value=[]), \
                mock.patch.object(q, "read_rows", side_effect=rows), \
                mock.patch.object(q, "write_json", side_effect=write), \
                mock.patch("builtins.print") as progress, \
                mock.patch.object(q, "LocalRunner", FakeRunner):
            manifest = q.qualify(self.inputs, self.output, workers=2)
        self.assertEqual(len(manifest["tasks"]), 120)
        self.assertEqual(len(written), len(yielded))
        progress.assert_called_once_with("Checked 100 tasks", flush=True)

    def test_cli_help_and_invalid_workers(self):
        import subprocess, sys
        cases = [(["--help"], 0, "qualify"), (["qualify", "--help"], 0, "--workers")]
        cases += [(["qualify", "--workers", str(n)], 2, "invalid choice") for n in (0, 9)]
        for args, expected_code, expected_text in cases:
            result = subprocess.run([sys.executable, "-m", "src.data-pipeline", *args],
                                    cwd=Path(q.__file__).resolve().parents[3], capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertIn(expected_text, result.stdout + result.stderr)
        cli = importlib.import_module("src.data-pipeline.__main__")
        argv = ["pipeline", "qualify", "--input-dir", "raw", "--output-dir", "pool", "--workers", "2"]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"), \
                mock.patch.object(cli, "qualify", return_value={"tasks": []}) as run:
            cli.main()
            run.assert_called_once_with(Path("raw"), Path("pool"), 2)
            run.reset_mock()
            with mock.patch.object(sys, "argv", argv[:-2]):
                cli.main()
        run.assert_called_once_with(Path("raw"), Path("pool"), 8)


if __name__ == "__main__":
    unittest.main()
