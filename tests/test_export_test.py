import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


export = importlib.import_module("src.data-pipeline.stage01.export_test")


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
            self.assertEqual(value, hashlib.sha256((Path(export.__file__).parents[1] / name).read_bytes()).hexdigest())
        for name in selected:
            self.assertEqual((output / name).read_bytes(), before[name])
            self.assertFalse((output / name).is_symlink())
            self.assertNotEqual((output / name).stat().st_ino, (source / name).stat().st_ino)
        self.assertEqual({p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)

    def test_listing_must_be_fully_validated_before_publication(self):
        source, output, manifest = self.fixture(self.root)
        manifest["accepted_counts"]["test"] += 1
        (source / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Manifest accepted counts disagree"):
            export.export_test(source, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".next-*")))

    def test_empty_test_selection_is_rejected(self):
        source, output, manifest = self.fixture(self.root)
        manifest["tasks"] = manifest["tasks"][:1]
        manifest["accepted_counts"]["test"] = 0
        (source / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "No qualified test tasks"):
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
