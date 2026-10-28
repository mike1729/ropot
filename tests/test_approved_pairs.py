"""Stage 05 publication loading fixtures; no model or program execution."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

p = importlib.import_module("src.data-pipeline.stage05.approved_pairs")
a = importlib.import_module("src.data-pipeline.stage05.attacks")


class ApprovedPairsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.publication = self.root / "publication"
        self.cohort_dir = self.root / "cohort"
        for directory in (self.publication / "programs", self.publication / "inputs", self.cohort_dir / "tasks"):
            directory.mkdir(parents=True)
        self.order = [3, 1, 4, 2, 5, 6, 7, 8, 9, 10]
        self.bindings, self.tasks = {}, {}
        self.program = b"# exact program\r\nimport sys\r\nprint(sum(map(int, sys.stdin.read().split())))\r\n"
        self.stdin = b" \t918273645000\r\n12 \t\r\n"
        self.answer = "918273645012"
        for identifier in self.order + [98, 99]:
            task = dict(split="test", id=identifier, question=f"Sum the integers. Task {identifier}.",
                        cases=[dict(index=0, input="2\n3\n", answer="5")])
            raw = json.dumps(task).encode()
            path = f"tasks/test_{identifier}.json"
            (self.cohort_dir / path).write_bytes(raw)
            owner = "construction" if identifier == 98 else "evaluation" if identifier == 99 else "development"
            self.bindings[identifier] = dict(split="test", id=identifier, path=path, owner=owner,
                block="initial" if identifier in self.order[:6] or identifier >= 98 else "reserve",
                sha256=sha256(raw).hexdigest(), source_sha256="a" * 64)
            self.tasks[identifier] = task
        self.cohort = dict(tasks=list(self.bindings.values()),
            provenance=dict(input_manifest_sha256="b" * 64), construction_phases={},
            groups=dict(development=dict(initial=self.order[:6], reserve=self.order[6:]),
                        construction=dict(initial=[98], reserve=[]), evaluation=dict(initial=[99], reserve=[])))
        raw = json.dumps(self.cohort).encode()
        (self.cohort_dir / "manifest.json").write_bytes(raw)
        patch = mock.patch.object(p.study_tasks, "COHORT_SHA256", sha256(raw).hexdigest())
        patch.start()
        self.addCleanup(patch.stop)
        self.selections = []
        for identifier in reversed(self.order + [98, 99]):
            for source in p.study_tasks.SOURCES:
                if identifier == 4 and source == "synthetic":
                    continue
                program_path, input_path = f"programs/{source}-{identifier}.py", f"inputs/{source}-{identifier}.txt"
                (self.publication / program_path).write_bytes(self.program)
                (self.publication / input_path).write_bytes(self.stdin)
                binding = self.bindings[identifier]
                self.selections.append(dict(binding=deepcopy(binding), source=source,
                    label="development_smoke" if identifier == 3 else "development" if identifier < 98 else "construction",
                    program=program_path, program_sha256=sha256(self.program).hexdigest(),
                    stdin=input_path, stdin_sha256=sha256(self.stdin).hexdigest(), answer=self.answer,
                    candidate=p.study_tasks.candidate_identity("test", identifier, binding["sha256"], source, self.program),
                    approval=dict(secret="PRIVATE_APPROVAL")))
        self.manifest = dict(schema_version=1, cohort_sha256=p.study_tasks.COHORT_SHA256,
                             selections=self.selections)
        self.pin_publication()

    def pin_publication(self):
        raw = json.dumps(self.manifest).encode()
        (self.publication / "manifest.json").write_bytes(raw)
        patch = mock.patch.object(p, "PUBLICATION_SHA256", sha256(raw).hexdigest())
        patch.start()
        self.addCleanup(patch.stop)

    def load(self, **kwargs):
        return p.load_development(self.publication, cohort_dir=self.cohort_dir, **kwargs)

    def selection(self, identifier=3, source="actor"):
        return next(row for row in self.selections if (row["binding"]["id"], row["source"]) == (identifier, source))

    def test_joint_panel_uses_cohort_order_caps_eight_and_preserves_exact_private_data(self):
        result = self.load()
        expected = [3, 1, 2, 5, 6, 7, 8, 9]
        self.assertEqual(result["task_ids"], expected)
        self.assertEqual(set(result["worklists"]), set(p.study_tasks.SOURCES))
        self.assertEqual(result["publication"], dict(path=str((self.publication / "manifest.json").absolute()),
                                                    sha256=p.PUBLICATION_SHA256))
        for source, worklist in result["worklists"].items():
            self.assertEqual([binding["id"] for binding, task in worklist], expected)
            for binding, task in worklist:
                identifier = binding["id"]
                self.assertEqual(task, self.tasks[identifier])
                self.assertEqual({key: value for key, value in binding.items() if key != "implant"},
                                 self.bindings[identifier])
                pair = binding["implant"]
                self.assertEqual(pair["program"].encode(), self.program)
                self.assertEqual(pair["stdin"].encode(), self.stdin)
                self.assertEqual(pair["candidate"], self.selection(identifier, source)["candidate"])
                self.assertEqual(pair["answer"], self.answer)
                self.assertEqual(pair["provenance"], dict(manifest=result["publication"]["path"],
                    manifest_sha256=p.PUBLICATION_SHA256, selection=self.selection(identifier, source)))
        self.assertNotIn("implant", result["worklists"]["actor"][0][0]["implant"]["provenance"]["selection"]["binding"])
        messages = a.messages(result["worklists"]["actor"], a.DEFAULT_RECIPE)
        prompt = json.dumps(messages)
        for private in (self.answer, "PRIVATE_APPROVAL", p.PUBLICATION_SHA256, "manifest_sha256", '"answer"'):
            self.assertNotIn(private, prompt)
        self.assertIn("918273645000", prompt)

    def test_fewer_joint_pairs_and_unselected_artifacts_are_not_read(self):
        self.selections[:] = [row for row in self.selections
                              if row["binding"]["id"] in (1, 3, 4, 98, 99)]
        self.pin_publication()
        for identifier in self.order + [98, 99]:
            if identifier in (1, 3):
                continue
            (self.cohort_dir / self.bindings[identifier]["path"]).unlink()
            for row in self.selections:
                if row["binding"]["id"] == identifier:
                    for field in ("program", "stdin"):
                        (self.publication / row[field]).unlink()
        result = self.load()
        self.assertEqual(result["task_ids"], [3, 1])
        self.assertEqual({source: len(rows) for source, rows in result["worklists"].items()},
                         dict(actor=2, synthetic=2))

    def test_task_files_can_be_relocated(self):
        relocated = self.root / "relocated"
        relocated.mkdir()
        self.cohort_dir.joinpath("tasks").rename(relocated / "tasks")
        self.assertEqual(self.load(task_dir=relocated)["task_ids"], [3, 1, 2, 5, 6, 7, 8, 9])

    def test_pinned_publication_cohort_and_selected_file_digests(self):
        row = self.selection()
        paths = [self.publication / "manifest.json", self.cohort_dir / "manifest.json",
                 self.cohort_dir / row["binding"]["path"], self.publication / row["program"],
                 self.publication / row["stdin"]]
        for path in paths:
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.load()
            path.write_bytes(raw)

    def test_manifest_version_and_cohort_commitment(self):
        for key, value in (("schema_version", 2), ("cohort_sha256", "f" * 64)):
            old = self.manifest[key]
            self.manifest[key] = value
            self.pin_publication()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
            self.manifest[key] = old

    def test_binding_candidate_and_exact_artifact_hashes_must_agree(self):
        row = self.selection()
        changes = [("binding", "source_sha256", "c" * 64), ("binding", "owner", "construction"),
                   ("candidate", "source", "synthetic"), ("candidate", "program_sha256", "d" * 64),
                   (None, "program_sha256", "d" * 64), (None, "stdin_sha256", "e" * 64)]
        for parent, field, value in changes:
            target = row if parent is None else row[parent]
            old = target[field]
            target[field] = value
            self.pin_publication()
            with self.subTest(parent=parent, field=field), self.assertRaises(ValueError):
                self.load()
            target[field] = old

    def test_duplicate_publication_keys_and_no_joint_panel_fail(self):
        self.selections.append(deepcopy(self.selection()))
        self.pin_publication()
        with self.assertRaises(ValueError):
            self.load()
        self.selections[:] = [row for row in self.selections if row["source"] == "actor"]
        self.selections.pop()  # remove the duplicate, leaving otherwise valid single-source data
        self.pin_publication()
        with self.assertRaises(ValueError):
            self.load()

    def test_program_and_stdin_paths_cannot_escape_even_through_symlinks(self):
        row = self.selection()
        for field, raw in (("program", self.program), ("stdin", self.stdin)):
            outside = self.root / f"outside-{field}"
            outside.write_bytes(raw)
            link = self.publication / f"linked-{field}"
            link.symlink_to(outside)
            original = row[field]
            for path in (f"../outside-{field}", str(outside), link.name):
                row[field] = path
                self.pin_publication()
                with self.subTest(field=field, path=path), self.assertRaises(ValueError):
                    self.load()
            row[field] = original


if __name__ == "__main__":
    unittest.main()
