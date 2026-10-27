"""Publish first verified passes from real mocked coordinator evidence."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

from tests import test_benign_loop as initial

s = importlib.import_module("src.data-pipeline.stage03.benign_selection")


@unittest.skipIf(initial.v is None, "producer fixtures require requirements-execution.txt")
class SelectionTests(initial.LoopFixture):
    def setUp(self):
        super().setUp()
        self.home, self.templates = self.root, deepcopy(self.worklist)
        self.experiments = self.home / "experiments"
        self.experiments.mkdir()
        self.target = self.home / "03_benigns/v1"
        self.cohort = dict(sha256=initial.b.study_tasks.COHORT_SHA256,
            groups=dict(development=dict(initial=list(range(50)), reserve=[]),
                construction=dict(initial=list(range(100,132))+list(range(200,232)), reserve=[])),
            construction_phases={phase: dict(initial=list(range(start, start+32)), reserve=[])
                for phase, start in (("shared_honest_pretraining", 100), ("twin_continuation", 200))},
            tasks_by_id={})
        for group, ids, phase in (("development", range(50), None),
                ("construction", range(100,132), "shared_honest_pretraining"),
                ("construction", range(200,232), "twin_continuation")):
            for identifier in ids:
                binding = dict(self.templates[identifier % 2][0], id=identifier, owner=group, block="initial")
                binding.pop("phase", None)
                if phase is not None:
                    binding["phase"] = phase
                self.cohort["tasks_by_id"][identifier] = binding
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(initial.b.study_tasks, "load_cohort", return_value=self.cohort).start()
        self.producers = 0

    def make(self, name="0001-first", source="actor", ids=(0,100), texts=None, verdicts=None, fixtures=False):
        self.producers += 1
        self.root = self.home / f"producer-{self.producers}"
        self.root.mkdir()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text("{}")
        self.worklist = [(deepcopy(self.cohort["tasks_by_id"][i]), deepcopy(self.templates[i % 2][1])) for i in ids]
        self.texts = texts or [f"print({i+1})\r\n" for i in range(len(ids))]
        def verify(*args, **kwargs):
            rows = initial.LoopFixture.verify(self, *args, **kwargs)
            for position, status in (verdicts or {}).items():
                rows[position].update(status=status, feedback=dict(status=status, hidden=status, visible=[]))
            return rows
        self.verifier.side_effect = verify
        self.run_loop(source=source, fixtures=fixtures, cohort_sha256=initial.b.study_tasks.COHORT_SHA256)
        directory = self.experiments / name
        shutil.copytree(self.root / "run", directory)
        return directory

    def read(self, directory, filename="outcome.json"):
        return json.loads((directory / filename).read_text())

    def write(self, directory, value, filename="outcome.json"):
        (directory / filename).write_text(json.dumps(value))

    def publish(self):
        return s.publish(self.experiments, "v1")

    def test_first_pass_order_exact_bytes_labels_full_denominators_and_provenance(self):
        self.make("0002-later", ids=(0,), texts=["print(99)\n"])
        first = self.make("0001-earlier", ids=(0,8,100,116))
        self.make("0003-synthetic", source="synthetic", verdicts={1:"failed"})
        manifest_path = self.publish()
        self.assertEqual(manifest_path, self.target / "manifest.json")
        manifest = self.read(self.target, "manifest.json")
        report = self.read(self.target, "report.json")
        selections = {(r["source"], r["binding"]["id"]):r for r in manifest["selections"]}
        self.assertEqual(len(selections), 5)
        chosen = selections["actor", 0]
        self.assertEqual(chosen["experiment"], first.name)
        self.assertEqual((chosen["attempt"], chosen["repair_depth"], chosen["parent"]), (0,0,None))
        self.assertEqual(chosen["verification"]["sample_id"], 0)
        self.assertEqual(chosen["recipe"], self.recipe)
        self.assertEqual(len(chosen["recipe_sha256"]), 64)
        self.assertEqual(manifest["cohort_sha256"], self.cohort["sha256"])
        self.assertTrue(manifest["rule"])
        self.assertEqual([selections["actor", i]["label"] for i in (0,8,100,116)],
            ["development_smoke", "development", "construction_exploration", "construction_confirmation"])
        for row in selections.values():
            program = Path(row["program"])
            self.assertFalse(program.is_absolute())
            raw = (self.target / program).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["program_sha256"])
            self.assertEqual(row["candidate"]["program_sha256"], row["program_sha256"])
            self.assertEqual(set(row["verification"]), {"key","log","sample_id"})
        self.assertEqual((self.target / chosen["program"]).read_bytes(), b"print(1)\r\n")
        for row in manifest["experiments"]:
            for filename in ("inputs", "outcome"):
                raw = (self.experiments / row["experiment"] / f"{filename}.json").read_bytes()
                self.assertEqual(row[f"{filename}_sha256"], hashlib.sha256(raw).hexdigest())
        groups = report["groups"]
        self.assertEqual({k:v["denominator"] for k,v in groups.items()},
            dict(development_smoke=8, development=42, construction_exploration=32, construction_confirmation=32))
        self.assertEqual(groups["development_smoke"]["selected"], dict(actor=1,synthetic=1))
        self.assertEqual(groups["development_smoke"]["joint_ids"], [0])
        self.assertEqual(len(groups["development_smoke"]["missing_cells"]), 14)
        self.assertEqual(set(report["summary"]), {"experiments","totals"})
        self.assertNotIn("cells", report["summary"])

    def test_terminal_failed_run_can_contribute_its_earlier_passing_peer(self):
        directory = self.make(verdicts={1:"error"})
        self.assertEqual(self.read(directory)["status"], "error")
        self.publish()
        rows = self.read(self.target, "manifest.json")["selections"]
        self.assertEqual([r["binding"]["id"] for r in rows], [0])

    def test_later_inconsistent_pass_is_checked_before_first_pass_selection(self):
        self.make("0001-earliest")
        later = self.make("0002-later")
        data = self.read(later)
        data["attempts"][0]["verification"]["status"] = "failed"
        self.write(later, data)
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.target.exists())

    def test_verdict_candidate_identity_task_and_program_digests_are_checked(self):
        directory = self.make()
        original = self.read(directory)
        changes = (lambda a:a["candidate"].update(source="synthetic"),
            lambda a:a["candidate"].update(id=8),
            lambda a:a["verification"]["key"].update(task_sha256="b"*64),
            lambda a:a["verification"]["key"].update(program_sha256="b"*64),
            lambda a:a.update(state="failed"))
        for change in changes:
            data = deepcopy(original)
            change(data["attempts"][0])
            self.write(directory, data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.publish()
        self.write(directory, original)
        Path(original["attempts"][0]["program"]).write_bytes(b"print(999)\n")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(self.target.exists())

    def test_foreign_worklist_task_and_fixture_records_are_refused(self):
        directory = self.make()
        inputs = self.read(directory, "inputs.json")
        changed = deepcopy(inputs)
        changed["worklist"][0]["binding"]["id"] = 999
        for value in (changed, dict(inputs, fixtures=True)):
            self.write(directory, value, "inputs.json")
            with self.assertRaises(ValueError):
                self.publish()
        self.assertFalse(self.target.exists())

    def test_running_malformed_and_unreadable_records_are_refused(self):
        directory = self.make()
        original = self.read(directory)
        self.write(directory, dict(original, status="running"))
        with self.assertRaises(ValueError):
            self.publish()
        (directory / "outcome.json").write_text("{")
        with self.assertRaises(ValueError):
            self.publish()
        self.write(directory, original)
        read_bytes = Path.read_bytes
        def unreadable(path):
            if path == directory / "outcome.json":
                raise PermissionError("unreadable test evidence")
            return read_bytes(path)
        with mock.patch.object(Path, "read_bytes", unreadable), self.assertRaises((ValueError,OSError)):
            self.publish()
        self.assertFalse(self.target.exists())

    def test_existing_version_and_path_component_versions_are_refused(self):
        self.make()
        for version in ("../other", "nested/v1", "", ".", ".."):
            with self.subTest(version=version), self.assertRaises(ValueError):
                s.publish(self.experiments, version)
        original = self.publish().read_bytes()
        with self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual((self.target / "manifest.json").read_bytes(), original)

    def test_failed_program_write_leaves_no_publication_or_temporary_directory(self):
        self.make()
        write_bytes = Path.write_bytes
        def fail(path, value):
            if path.suffix == ".py" and path.parent.name == "programs":
                raise OSError("test publication write failure")
            return write_bytes(path, value)
        with mock.patch.object(Path, "write_bytes", fail), self.assertRaises(OSError):
            self.publish()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
