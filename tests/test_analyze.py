import csv
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


a = importlib.import_module("src.data-pipeline.analyze")
screen = importlib.import_module("src.data-pipeline.screen")


class CharacterEncoder:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(range(len(text))))


def task(identifier=7, split="test", question="é\n", count=2):
    sources = ["raise RuntimeError('never execute')", "print(1)", "# copy\nprint(1)", "print(2)", "print(3)"]
    cases = [{"index": i, "visible": i == 0, "input": "é\n", "output": "1\n", "answer": "1"}
             for i in range(count)]
    attempts = [{"source_index": i, "passed": i != 0, "max_case_seconds": i / 10,
                 "cases": [{"index": c["index"], "status": "passed" if i else "wrong_answer",
                            "elapsed_seconds": i / 10} for c in (cases if i else cases[:1])]}
                for i in (0, 1, 3, 4)]
    lengths = lambda value: {"characters": len(value), "bytes": len(value.encode())}
    return {"id": identifier, "split": split, "difficulty": "introductory", "question": question,
            "url": "https://example.com/task", "solutions": sources, "cases": cases,
            "original_row": {"id": identifier, "question": question, "solutions": json.dumps(sources)},
            "reference_candidates": [{"index": i, "token_sha256": screen.token_fingerprint(sources[i])}
                                     for i in (0, 1, 3, 4)],
            "passing_reference_indices": [1, 3, 4], "visible_case_count": 1,
            "nonvisible_case_count": count - 1, "reference_counts": {"raw": 5, "distinct": 4, "passing": 3},
            "verification_attempts": attempts, "lengths": {"question": lengths(question),
                "inputs": [lengths(c["input"]) for c in cases],
                "outputs": [lengths(c["output"]) for c in cases], "solutions": list(map(lengths, sources))}}


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.input = self.root / "qualified"
        (self.input / "tasks").mkdir(parents=True)
        self.output = self.root / "analysis"
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text("{}")
        self.tasks = [task(7, "train"), task(9, question="x" * 8, count=4), task(7)]
        self.write_inputs()

    def write_inputs(self):
        names = []
        for item in self.tasks:
            name = f"tasks/{item['split']}_{item['id']}.json"
            (self.input / name).write_text(json.dumps(item))
            names.append(name)
        (self.input / "manifest.json").write_text(json.dumps({"tasks": names, "workers": 1, "accepted_counts": {s: sum(t["split"] == s for t in self.tasks) for s in ("train", "test")}, "runtime": {"version": "fixture"}}))

    def run_analysis(self):
        with mock.patch.object(a, "load_tokenizer", return_value=CharacterEncoder()), mock.patch("builtins.print"):
            return a.analyze(self.input, self.output, self.tokenizer)

    def test_nearest_rank_percentiles_and_task_measurements(self):
        self.assertEqual(a.summarize([9, 1, 5]), {"n": 3, "min": 1, "median": 5,
                         "p90": 9, "p95": 9, "p99": 9, "max": 9})
        self.assertEqual(a.summarize([4])["p99"], 4)
        with self.assertRaises(ValueError):
            a.summarize([])
        row = a.measure_task(task(question="é" * 5001), CharacterEncoder())
        self.assertEqual((row["statement_tokens"], row["question_characters"], row["question_bytes"]), (5001, 5001, 10002))
        self.assertEqual([row[f"reference_{i}_index"] for i in (1, 2, 3)], [1, 3, 4])
        self.assertEqual([row[f"reference_{i}_tokens"] for i in (1, 2, 3)], [8, 8, 8])
        self.assertEqual((row["raw_references"], row["distinct_references"], row["reference_attempts"]), (5, 4, 4))
        self.assertEqual((row["case_attempts"], row["visible_cases"], row["nonvisible_cases"]), (7, 1, 1))
        self.assertEqual((row["input_max_characters"], row["input_total_bytes"], row["passing_case_max_seconds"]), (2, 6, 0.4))

    def test_test_only_split_scoped_sorted_and_preserved_provenance(self):
        before = {p: p.read_bytes() for p in self.input.rglob("*") if p.is_file()}
        summary = self.run_analysis()
        with (self.output / "tasks.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(r["split"], r["id"]) for r in rows], [("test", "7"), ("test", "9")])
        self.assertEqual(summary["counts"]["tasks"], 2)
        self.assertEqual(summary["counts"]["passing_references"], 6)
        self.assertEqual(summary["counts"]["case_attempts"], 20)
        self.assertEqual(summary["distributions"]["overall"]["statement_tokens"]["median"], 5)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), summary)
        provenance = summary["provenance"]
        self.assertEqual(provenance["tasks_csv_sha256"], hashlib.sha256((self.output / "tasks.csv").read_bytes()).hexdigest())
        self.assertEqual(provenance["input_manifest_sha256"], hashlib.sha256(before[self.input / "manifest.json"]).hexdigest())
        for path, digest in provenance["task_sha256"].items():
            self.assertEqual(digest, hashlib.sha256(before[self.input / path]).hexdigest())
        self.assertEqual(set(provenance["task_sha256"]), {"tasks/test_7.json", "tasks/test_9.json"})
        self.assertEqual(provenance["tokenizer"]["sha256"], hashlib.sha256(self.tokenizer.read_bytes()).hexdigest())
        self.assertTrue(provenance["source_sha256"])
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_listing_must_be_fully_validated_before_publication(self):
        path = self.input / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["accepted_counts"]["test"] += 1
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Manifest accepted counts disagree"):
            self.run_analysis()
        self.assertFalse(self.output.exists())

    def test_measurements_do_not_depend_on_qualification_diagnostics(self):
        qualified = task()
        expected = a.measure_task(qualified, CharacterEncoder())
        qualified["reference_candidates"] = [{} for _ in qualified["reference_candidates"]]
        for key in ("reference_counts", "visible_case_count", "nonvisible_case_count"):
            del qualified[key]
        qualified["verification_attempts"].reverse()
        for attempt in qualified["verification_attempts"]:
            del attempt["source_index"]
            attempt["cases"] = [{} for _ in attempt["cases"]]
        self.assertEqual(a.measure_task(qualified, CharacterEncoder()), expected)

    def test_empty_test_selection_is_rejected(self):
        self.tasks = [task(split="train")]
        self.write_inputs()
        with self.assertRaises(ValueError):
            self.run_analysis()

    def test_existing_destination_and_interrupted_publication(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_analysis()
        self.output.rmdir()
        with mock.patch.object(a, "write_json", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.run_analysis()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".analysis-*")))

    def test_tokenizer_disables_saved_padding_truncation_and_special_tokens(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors
        encoder = Tokenizer(models.WordLevel({"[UNK]": 0, "é": 1, "[CLS]": 2}, unk_token="[UNK]"))
        encoder.pre_tokenizer = pre_tokenizers.Whitespace()
        encoder.post_processor = processors.TemplateProcessing(single="[CLS] $A", special_tokens=[("[CLS]", 2)])
        encoder.enable_truncation(max_length=3)
        encoder.enable_padding(length=8)
        encoder.save(str(self.tokenizer))
        self.assertEqual(len(a.load_tokenizer(self.tokenizer).encode("é " * 5001, add_special_tokens=False).ids), 5001)
