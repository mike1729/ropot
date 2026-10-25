import copy
import csv
import importlib
import json
from pathlib import Path
import tempfile
import unittest


c = importlib.import_module("src.data-pipeline.stage02.cohort")
known = importlib.import_module("src.data-pipeline.stage02.known_use")


class CohortInputTests(unittest.TestCase):
    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pool, self.analysis = self.root / "pool", self.root / "analysis"
        (self.pool / "tasks").mkdir(parents=True)
        self.analysis.mkdir()
        self.tasks, self.rows, hashes = [], [], {}
        for identifier in (7, 9):
            task = {"split": "test", "id": identifier, "difficulty": "introductory",
                    "url": "https://example.org/task", "passing_reference_indices": [0, 1, 2],
                    "original_row": {"id": identifier, "question": "é fixture"},
                    "cases": [{"index": 4, "visible": True, "input": "é\n"},
                              {"index": 8, "visible": False, "input": "1\n"}]}
            name = f"tasks/test_{identifier}.json"
            raw = json.dumps(task).encode()
            (self.pool / name).write_bytes(raw)
            hashes[name] = c.sha256(raw)
            self.tasks.append(task)
            self.rows.append({"split": "test", "id": identifier, "difficulty": "introductory",
                              "host": "example.org", "statement_tokens": 10, "reference_tokens_max": 30,
                              **{f"reference_{n}_index": n - 1 for n in (1, 2, 3)},
                              **{f"reference_{n}_tokens": n * 10 for n in (1, 2, 3)},
                              "input_max_bytes": 3, "input_total_bytes": 5,
                              "cases": 2, "visible_cases": 1, "nonvisible_cases": 1})
        common = {"dataset": "codeparrot/apps", "revision": "pinned", "qualification": "local_subprocess",
                  "required_next_gate": "Inspect Linux references before generation", "runtime": {"version": "old"},
                  "workers": 2, "tasks": list(hashes), "accepted_counts": {"train": 0, "test": 2}}
        archive = json.dumps(common).encode()
        (self.pool / "qualification-manifest.json").write_bytes(archive)
        provenance = {"input_manifest_sha256": c.sha256(archive), "task_sha256": hashes}
        self.manifest = {**common, "selection": {"split": "test", "excluded_train_tasks": 0},
                         "provenance": {**provenance, "qualification_manifest": "qualification-manifest.json"}}
        self.summary = {"selection": self.manifest["selection"].copy(), "counts": {"tasks": 2},
                        "provenance": {**copy.deepcopy(provenance), "qualification_runtime": common["runtime"].copy(),
                                       "workers": 2, "tokenizer": {"sha256": "historical", "version": "pinned"}}}
        self.records_path, self.coverage_path = self.root / "known-use.json", self.root / "coverage.json"
        self.observations = {"schema_version": 1, "observations": []}
        self.coverage = {"schema_version": 1, "cutoff_date": "2026-10-21", "history_certainty": "incomplete",
                         "inspection_scope": "local", "interpretation": "No unseen guarantee", "limits": ["Partial history"],
                         "archive_to_apps_binding": {"apps_row_hash_rule": known.ROW_HASH_RULE, "bindings": []}}
        self.save()

    def save(self):
        (self.pool / "manifest.json").write_text(json.dumps(self.manifest))
        with (self.analysis / "tasks.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        self.summary["provenance"]["tasks_csv_sha256"] = c.sha256((self.analysis / "tasks.csv").read_bytes())
        (self.analysis / "summary.json").write_text(json.dumps(self.summary))
        self.records_path.write_text(json.dumps(self.observations))
        self.coverage_path.write_text(json.dumps(self.coverage))

    def load(self):
        return c.load_inputs(self.pool, self.analysis, self.records_path, self.coverage_path)

    def test_known_use_original_row_binding_provenance_and_coverage(self):
        self.fixture()
        expected = known.row_sha256(self.tasks[0]["original_row"])
        self.observations["observations"] = [
            {"task_id": "apps:test:7", "row_sha256": expected, "use": "prompt_checker_development"},
            {"task_id": "apps:train:9", "row_sha256": "a" * 64, "use": "unresolved"}]
        self.save()
        result = self.load()
        self.assertEqual([t["known_uses"] for t in result["tasks"]], [["prompt_checker_development"], []])
        self.assertEqual(result["known_use_coverage"]["history_certainty"], "incomplete")
        self.assertEqual(result["known_use_coverage"]["limits"], ["Partial history"])
        provenance = result["provenance"]["known_use"]
        self.assertEqual(provenance["records_sha256"], c.sha256(self.records_path.read_bytes()))
        self.assertEqual(provenance["coverage_sha256"], c.sha256(self.coverage_path.read_bytes()))
        self.observations["observations"][0]["row_sha256"] = "b" * 64
        self.save()
        with self.assertRaisesRegex(ValueError, "Known-use original-row hash mismatch"):
            self.load()

    def test_load_preserves_fingerprints_gate_and_all_cases_in_place(self):
        self.fixture()
        (self.pool / "tasks/unlisted.json").write_text("invalid JSON; ignore me")
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.load()
        self.assertEqual([t["id"] for t in result["tasks"]], [7, 9])
        self.assertEqual(result["required_next_gate"], self.manifest["required_next_gate"])
        self.assertEqual(result["provenance"]["tokenizer"], self.summary["provenance"]["tokenizer"])
        for name, path in (("input_manifest", self.pool / "manifest.json"),
                           ("analysis_summary", self.analysis / "summary.json"),
                           ("analysis_tasks", self.analysis / "tasks.csv")):
            self.assertEqual(result["provenance"][f"{name}_sha256"], c.sha256(before[path]))
        for task in result["tasks"]:
            self.assertEqual(task["sha256"], c.sha256(before[self.pool / task["path"]]))
        self.assertEqual({p: p.read_bytes() for p in before}, before)

    def test_task_corruption_and_analysis_hash_disagreement(self):
        self.fixture()
        with (self.pool / "tasks/test_7.json").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "Input task hash mismatch"):
            self.load()
        for action in ("changed", "extra", "missing"):
            with self.subTest(action=action):
                self.fixture()
                hashes = self.summary["provenance"]["task_sha256"]
                if action == "changed": hashes["tasks/test_7.json"] = "wrong"
                elif action == "extra": hashes["tasks/test_8.json"] = "extra"
                else: del hashes["tasks/test_7.json"]
                self.save()
                with self.assertRaisesRegex(ValueError, "Analysis/export task hashes mismatch"):
                    self.load()

    def test_csv_edits_and_missing_fingerprint_are_rejected(self):
        for action in ("tokens", "whitespace", "missing_hash"):
            with self.subTest(action=action):
                self.fixture()
                path = self.analysis / "tasks.csv"
                if action == "tokens": path.write_bytes(path.read_bytes().replace(b",10,", b",701,", 1))
                elif action == "whitespace": path.write_bytes(path.read_bytes() + b"\n")
                else:
                    del self.summary["provenance"]["tasks_csv_sha256"]
                    (self.analysis / "summary.json").write_text(json.dumps(self.summary))
                with self.assertRaisesRegex(ValueError, "Analysis tasks.csv hash"):
                    self.load()

    def test_test_only_input_and_manifest_counts(self):
        self.fixture()
        old, new = "tasks/test_7.json", "tasks/train_7.json"
        raw = json.dumps({**self.tasks[0], "split": "train"}).encode()
        (self.pool / new).write_bytes(raw)
        self.manifest["tasks"][0] = new
        self.manifest["accepted_counts"] = {"train": 1, "test": 1}
        for provenance in (self.manifest["provenance"], self.summary["provenance"]):
            del provenance["task_sha256"][old]
            provenance["task_sha256"][new] = c.sha256(raw)
        self.save()
        with self.assertRaisesRegex(ValueError, "test tasks only"): self.load()
        self.fixture()
        self.manifest["accepted_counts"]["test"] = 3
        self.save()
        with self.assertRaisesRegex(ValueError, "Manifest accepted counts disagree"): self.load()

    def test_exact_csv_identity_join_and_metadata(self):
        for action, message in (("duplicate", "duplicate analysis identity"), ("missing", "Missing analysis identity"),
                                ("extra", "identities/counts"), ("train", "Invalid or duplicate analysis identity"),
                                ("difficulty", "metadata mismatch"), ("host", "metadata mismatch"),
                                ("reference_1_index", "references mismatch"), ("reference_tokens_max", "references mismatch"),
                                ("statement_tokens", "Invalid analysis measurement")):
            with self.subTest(action=action):
                self.fixture()
                if action == "duplicate": self.rows.append(self.rows[0].copy())
                elif action == "missing": self.rows.pop()
                elif action == "extra": self.rows.append({**self.rows[0], "id": 11})
                elif action == "train": self.rows[0]["split"] = "train"
                elif action in ("difficulty", "host"): self.rows[0][action] = "other"
                elif action == "statement_tokens": self.rows[0][action] = -1
                else: self.rows[0][action] += 1
                self.save()
                with self.assertRaisesRegex(ValueError, message): self.load()

    def test_provenance_mismatches(self):
        for action, message in (("archive", "Qualification manifest hash"), ("analysis_hash", "Qualification manifest hash"),
                                ("runtime", "Qualification provenance"), ("gate", "Qualification provenance"),
                                ("analysis_runtime", "Analysis qualification provenance"),
                                ("workers", "Analysis qualification provenance"), ("selection", "Analysis qualification provenance")):
            with self.subTest(action=action):
                self.fixture()
                if action == "archive": (self.pool / "qualification-manifest.json").write_text("{}")
                elif action == "analysis_hash": self.summary["provenance"]["input_manifest_sha256"] = "wrong"
                elif action == "runtime": self.manifest["runtime"] = {}
                elif action == "gate": self.manifest["required_next_gate"] = "cleared"
                elif action == "analysis_runtime": self.summary["provenance"]["qualification_runtime"] = {}
                elif action == "workers": self.summary["provenance"]["workers"] = 3
                else: self.summary["selection"]["excluded_train_tasks"] = 1
                self.save()
                with self.assertRaisesRegex(ValueError, message): self.load()

    def test_duplicate_tasks_and_confined_manifest_paths(self):
        for action, message in (("duplicate", "Duplicate task identity"), ("traversal", "Invalid task path"),
                                ("symlink", "Invalid task path"), ("archive_symlink", "qualification manifest path")):
            with self.subTest(action=action):
                self.fixture()
                if action == "duplicate":
                    self.manifest["tasks"].append("tasks/test_7.json")
                    self.manifest["accepted_counts"]["test"] = 3
                elif action == "traversal":
                    old, new = "tasks/test_7.json", "tasks/../tasks/test_7.json"
                    self.manifest["tasks"][0] = new
                    for provenance in (self.manifest["provenance"], self.summary["provenance"]):
                        provenance["task_sha256"][new] = provenance["task_sha256"].pop(old)
                else:
                    path = self.pool / ("tasks/test_7.json" if action == "symlink" else "qualification-manifest.json")
                    outside = self.root / path.name
                    path.rename(outside)
                    path.symlink_to(outside)
                self.save()
                with self.assertRaisesRegex(ValueError, message): self.load()
