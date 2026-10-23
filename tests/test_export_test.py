import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


export = importlib.import_module("src.data-pipeline.export_test")


class ExportTestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def fixture(self, root):
        source, output = root / "qualified", root / "next"
        (source / "tasks").mkdir(parents=True)
        names = ["tasks/train_7.json", "tasks/test_9.json", "tasks/test_7.json"]
        for name in names:
            split, identifier = Path(name).stem.split("_")
            (source / name).write_bytes(json.dumps({"split": split, "id": int(identifier),
                                                   "payload": "é\n"}, ensure_ascii=False, indent=3).encode())
        manifest = {"tasks": names, "accepted_counts": {"train": 1, "test": 2},
                    "dataset": "codeparrot/apps", "revision": "pinned", "runtime": {"version": "fixture"},
                    "workers": 3, "qualification": "local_subprocess", "required_next_gate": "Linux verification",
                    "files": [{"file": "train.jsonl"}], "reference_rejections": ["rejections/train_8.json"],
                    "pipeline": {"source_sha256": {"old.py": "historical"}}}
        (source / "manifest.json").write_text(json.dumps(manifest, indent=3) + "\n")
        return source, output, manifest

    def test_selected_tasks_are_independent_exact_copies_with_provenance(self):
        source, output, original = self.fixture(self.root)
        before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
        result = export.export_test(source, output)
        selected = ["tasks/test_7.json", "tasks/test_9.json"]
        self.assertEqual(result["tasks"], selected)
        self.assertEqual(result["accepted_counts"], {"train": 0, "test": 2})
        self.assertEqual(result["selection"], {"split": "test", "excluded_train_tasks": 1})
        for key in ("dataset", "revision", "runtime", "workers", "qualification", "required_next_gate"):
            self.assertEqual(result[key], original[key])
        self.assertNotIn("files", result)
        self.assertNotIn("reference_rejections", result)
        self.assertEqual((output / "qualification-manifest.json").read_bytes(), before["manifest.json"])
        self.assertEqual(json.loads((output / "manifest.json").read_text()), result)
        self.assertEqual(sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()),
                         ["manifest.json", "qualification-manifest.json", *selected])
        provenance = result["provenance"]
        self.assertEqual(Path(provenance["input_dir"]), source.resolve())
        self.assertEqual(provenance["qualification_manifest"], "qualification-manifest.json")
        self.assertEqual(provenance["input_manifest_sha256"], hashlib.sha256(before["manifest.json"]).hexdigest())
        self.assertEqual(provenance["task_sha256"], {name: hashlib.sha256(before[name]).hexdigest() for name in selected})
        self.assertTrue(provenance["source_sha256"])
        for name, value in provenance["source_sha256"].items():
            self.assertEqual(value, hashlib.sha256((Path(export.__file__).parent / name).read_bytes()).hexdigest())
        for name in selected:
            self.assertEqual((output / name).read_bytes(), before[name])
            self.assertFalse((output / name).is_symlink())
            self.assertNotEqual((output / name).stat().st_ino, (source / name).stat().st_ino)
        self.assertEqual({p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)

    def test_invalid_inputs_never_publish(self):
        for fault in ("duplicate", "duplicate_identity", "missing_train", "bad_split", "bad_id", "bool_id", "filename", "traversal", "absolute", "symlink", "counts", "empty"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=self.root) as directory:
                source, output, manifest = self.fixture(Path(directory))
                target = source / "tasks/test_7.json"
                item = json.loads(target.read_text())
                if fault == "duplicate": manifest["tasks"].append(manifest["tasks"][-1])
                elif fault == "duplicate_identity":
                    (source / "tasks/test_07.json").write_bytes(target.read_bytes())
                    manifest["tasks"].append("tasks/test_07.json")
                    manifest["accepted_counts"]["test"] += 1
                elif fault == "missing_train": (source / "tasks/train_7.json").unlink()
                elif fault == "bad_split": item["split"] = "validation"
                elif fault == "bad_id": item["id"] = "7"
                elif fault == "bool_id": item["id"] = True
                elif fault == "filename": item["id"] = 8
                elif fault == "traversal": manifest["tasks"][-1] = "tasks/../tasks/test_7.json"
                elif fault == "absolute": manifest["tasks"][-1] = str(target)
                elif fault == "symlink":
                    outside = Path(directory) / "outside.json"
                    outside.write_bytes(target.read_bytes())
                    target.unlink()
                    target.symlink_to(outside)
                elif fault == "counts": manifest["accepted_counts"]["train"] = 0
                else:
                    manifest["tasks"] = manifest["tasks"][:1]
                    manifest["accepted_counts"]["test"] = 0
                if fault in ("bad_split", "bad_id", "bool_id", "filename"): target.write_text(json.dumps(item))
                (source / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    export.export_test(source, output)
                self.assertFalse(output.exists())

    def test_existing_and_nested_destinations_and_interrupted_publication(self):
        source, output, _ = self.fixture(self.root)
        output.mkdir()
        with self.assertRaises(FileExistsError): export.export_test(source, output)
        output.rmdir()
        with self.assertRaises(ValueError): export.export_test(source, source / "next")
        def interrupted(path, value):
            self.assertEqual(len(list((path.parent / "tasks").glob("*.json"))), 2)
            raise RuntimeError("interrupted")
        with mock.patch.object(export, "write_json", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"): export.export_test(source, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".next-*")))
