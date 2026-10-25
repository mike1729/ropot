import importlib
import json
from pathlib import Path
import tempfile
import unittest


qualified = importlib.import_module("src.data-pipeline.stage01.qualified_tasks")


class QualifiedTasksTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "qualified"
        (self.root / "tasks").mkdir(parents=True)
        self.manifest = {"tasks": ["tasks/train_7.json", "tasks/test_9.json", "tasks/test_7.json"],
                         "accepted_counts": {"train": 1, "test": 2}}
        for relative in self.manifest["tasks"]:
            split, identifier = Path(relative).stem.split("_")
            self.write(relative, {"split": split, "id": int(identifier), "payload": "é\n"})

    def write(self, relative, task):
        (self.root / relative).write_bytes(json.dumps(task, ensure_ascii=False, indent=3).encode())

    def read(self):
        return qualified.iter_qualified_tasks(self.root, self.manifest)

    def test_yields_original_bytes_and_all_split_scoped_identities(self):
        rows = list(self.read())
        self.assertEqual([relative for relative, _, _ in rows], self.manifest["tasks"])
        self.assertEqual([(task["split"], task["id"]) for _, _, task in rows],
                         [("train", 7), ("test", 9), ("test", 7)])
        for relative, raw, task in rows:
            self.assertEqual(raw, (self.root / relative).read_bytes())
            self.assertEqual(task, json.loads(raw))

    def test_invalid_split_is_the_only_fault(self):
        relative = "tasks/validation_7.json"
        self.write(relative, {"split": "validation", "id": 7, "payload": "é\n"})
        self.manifest["tasks"].append(relative)
        with self.assertRaisesRegex(ValueError, "^Invalid task split:"):
            list(self.read())

    def test_duplicate_identity_has_matching_manifest_counts(self):
        self.manifest["tasks"].append("tasks/test_7.json")
        self.manifest["accepted_counts"]["test"] = 3
        with self.assertRaisesRegex(ValueError, "^Duplicate task identity:"):
            list(self.read())

    def test_noncanonical_filename_is_the_only_fault(self):
        relative = "tasks/test_07.json"
        (self.root / "tasks/test_7.json").rename(self.root / relative)
        self.manifest["tasks"][-1] = relative
        with self.assertRaisesRegex(ValueError, "^Task filename disagrees with identity:"):
            list(self.read())

    def test_id_types_are_rejected_even_with_matching_filename(self):
        for identifier in ("7", True):
            with self.subTest(identifier=identifier):
                relative = f"tasks/test_{identifier}.json"
                self.write(relative, {"split": "test", "id": identifier, "payload": "é\n"})
                self.manifest["tasks"][-1] = relative
                with self.assertRaisesRegex(ValueError, "^Invalid task id:"):
                    list(self.read())

    def test_train_tasks_are_validated(self):
        self.write("tasks/train_7.json", {"split": "train", "id": "7", "payload": "é\n"})
        with self.assertRaisesRegex(ValueError, "^Invalid task id:"):
            list(self.read())

    def test_missing_train_file_is_not_skipped(self):
        (self.root / "tasks/train_7.json").unlink()
        with self.assertRaises(FileNotFoundError):
            list(self.read())

    def test_path_traversal_and_absolute_paths(self):
        for relative in ("tasks/../tasks/test_7.json", str(self.root / "tasks/test_7.json")):
            with self.subTest(relative=relative):
                self.manifest["tasks"][-1] = relative
                with self.assertRaisesRegex(ValueError, "^Invalid task path:"):
                    list(self.read())

    def test_symlink_cannot_escape_the_input_directory(self):
        target = self.root / "tasks/test_7.json"
        outside = self.root.parent / "test_7.json"
        target.rename(outside)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "^Invalid task path:"):
            list(self.read())

    def test_counts_are_checked_only_when_the_stream_is_exhausted(self):
        self.manifest["accepted_counts"]["test"] = 3
        stream = self.read()
        self.assertEqual([next(stream)[0] for _ in range(3)], self.manifest["tasks"])
        with self.assertRaisesRegex(ValueError, "^Manifest accepted counts disagree"):
            next(stream)

    def test_empty_test_split_is_valid(self):
        self.manifest["tasks"] = self.manifest["tasks"][:1]
        self.manifest["accepted_counts"]["test"] = 0
        self.assertEqual([task["split"] for _, _, task in self.read()], ["train"])
