import copy
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


study = importlib.import_module("src.data-pipeline.stage03.study_tasks")

LAYOUT = {  # id: difficulty, owner, block, construction phase
    1: ("introductory", "development", "initial", None),
    2: ("interview", "development", "initial", None),
    3: ("interview", "development", "initial", None),
    10: ("introductory", "construction", "initial", "shared_honest_pretraining"),
    11: ("interview", "construction", "initial", "shared_honest_pretraining"),
    12: ("interview", "construction", "initial", "shared_honest_pretraining"),
    13: ("interview", "construction", "reserve", "shared_honest_pretraining"),
    20: ("interview", "construction", "initial", "twin_continuation"),
    21: ("introductory", "construction", "initial", "twin_continuation"),
    30: ("interview", "evaluation", "initial", None),
    31: ("interview", "calibration", "initial", None),
    32: ("interview", "correction_replay", "initial", None),
    33: ("interview", "construction", "deferred", "twin_continuation"),
}
BLOCKS = ("initial", "reserve", "deferred")
PHASES = ("shared_honest_pretraining", "twin_continuation")


class StudyTaskTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pool, self.cohort_dir = self.root / "pool", self.root / "cohort"
        (self.pool / "tasks").mkdir(parents=True)
        self.cohort_dir.mkdir()
        records = [{"split": "test", "id": identifier, "difficulty": difficulty, "path": f"tasks/test_{identifier}.json",
                    "owner": owner, "block": block, **({"phase": phase} if phase else {}),
                    "defer_reasons": ["statement_tokens"] if block == "deferred" else [], "known_uses": ["unresolved"] if identifier == 20 else []}
                   for identifier, (difficulty, owner, block, phase) in LAYOUT.items()]
        self.records = {record["id"]: record for record in records}
        self.cohort = {
            "tasks": records,
            "groups": {owner: {block: [i for i, (_, o, b, _) in LAYOUT.items() if (o, b) == (owner, block)]
                               for block in BLOCKS} for owner in ("development", "construction", "evaluation", "calibration", "correction_replay")},
            "construction_phases": {phase: {block: [i for i, (_, _, b, p) in LAYOUT.items() if (p, b) == (phase, block)]
                                            for block in BLOCKS} for phase in PHASES},
            "required_next_gate": "Re-verify references in Inspect Linux before generation",
            "provenance": {"input_dir": str(self.pool)}}
        self.pool_manifest = {"provenance": {"task_sha256": {}}}
        for identifier, (difficulty, *_) in LAYOUT.items():
            self.write_task(identifier, {"split": "test", "id": identifier, "difficulty": difficulty, "question": "é"})
        self.loaded = self.rewrite()

    def write_task(self, identifier, task, path=None):
        """Write a task artifact and bind its digest in both manifests."""
        record = self.records[identifier]
        record["path"] = path or record["path"]
        raw = json.dumps(task, ensure_ascii=False).encode()
        target = self.pool / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        record["sha256"] = self.pool_manifest["provenance"]["task_sha256"][record["path"]] = study.sha256(raw)

    def rewrite(self, cohort=None):
        """Write both manifests, link the cohort to the pool and pin the cohort digest."""
        cohort = self.cohort if cohort is None else cohort
        pool_raw = json.dumps(self.pool_manifest).encode()
        (self.pool / "manifest.json").write_bytes(pool_raw)
        cohort["provenance"]["input_manifest_sha256"] = study.sha256(pool_raw)
        raw = json.dumps(cohort).encode()
        (self.cohort_dir / "manifest.json").write_bytes(raw)
        patcher = mock.patch.object(study, "COHORT_SHA256", study.sha256(raw))
        patcher.start()
        self.addCleanup(patcher.stop)
        return study.load_cohort(self.cohort_dir)

    def test_pinned_manifests_load_unchanged_and_allow_task_relocation(self):
        self.assertEqual(set(self.loaded["tasks_by_id"]), {1, 2, 3, 10, 11, 12, 13, 20, 21})
        self.assertEqual(self.loaded["test_pool_manifest_sha256"],
                         self.cohort["provenance"]["input_manifest_sha256"])
        moved = self.root / "moved"
        self.pool.rename(moved)
        with self.assertRaises(FileNotFoundError):
            study.load_cohort(self.cohort_dir)
        loaded = study.load_cohort(self.cohort_dir, moved)
        self.assertEqual(study.load_task(loaded, 1)[1]["question"], "é")
        for path, message in ((moved / "manifest.json", "Test-pool manifest"),
                              (self.cohort_dir / "manifest.json", "reviewed publication")):
            path.write_bytes(path.read_bytes() + b" ")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, message):
                study.load_cohort(self.cohort_dir, moved)

    def test_load_consumes_saved_groups_and_phase_order_without_reallocation(self):
        allocator = importlib.import_module("src.data-pipeline.stage02.cohort")
        self.cohort["groups"]["development"]["initial"] = [3, 1, 2]
        with mock.patch.object(allocator, "allocate", side_effect=AssertionError("must not repartition")):
            loaded = self.rewrite()
        self.assertEqual(loaded["groups"], {owner: {block: self.cohort["groups"][owner][block]
                                                   for block in ("initial", "reserve")}
                                             for owner in ("construction", "development")})
        self.assertEqual(loaded["construction_phases"],
                         {phase: {block: self.cohort["construction_phases"][phase][block]
                                  for block in ("initial", "reserve")} for phase in PHASES})
        self.assertEqual(study.load_task(loaded, 3)[0], self.records[3])
        self.assertEqual(study.load_task(loaded, 13)[0], self.records[13])

    def test_cohort_load_rejects_invalid_saved_membership(self):
        changes = {
            "protected owner": lambda c: c["groups"]["development"]["initial"].append(30),
            "deferred": lambda c: c["groups"]["construction"]["initial"].append(33),
            "duplicate": lambda c: c["groups"]["development"]["initial"].append(1),
        }
        for name, change in changes.items():
            with self.subTest(name):
                cohort = copy.deepcopy(self.cohort)
                change(cohort)
                with self.assertRaisesRegex(ValueError, "Ineligible saved"):
                    self.rewrite(cohort)

    def test_cohort_load_rejects_duplicate_and_noninteger_ids(self):
        for identifier in (2, True, "1"):
            with self.subTest(identifier=identifier):
                cohort = copy.deepcopy(self.cohort)
                cohort["tasks"][0]["id"] = identifier
                with self.assertRaisesRegex(ValueError, "distinct integers"):
                    self.rewrite(cohort)

    def test_cohort_load_requires_agreeing_task_bindings(self):
        self.pool_manifest["provenance"]["task_sha256"]["tasks/test_1.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bindings disagree"):
            self.rewrite()

    def test_load_task_reads_only_the_requested_verified_artifact(self):
        (self.pool / "tasks/test_2.json").unlink()
        (self.pool / "tasks/test_10.json").write_bytes(b"corrupt")
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            binding, task = study.load_task(self.loaded, 1)
        self.assertEqual([call.args[0].name for call in read.call_args_list], ["test_1.json"])
        self.assertEqual(binding, self.records[1])
        self.assertEqual(task, json.loads((self.pool / "tasks/test_1.json").read_bytes()))
        binding["known_uses"].append("changed")
        self.assertEqual(self.loaded["tasks_by_id"][1]["known_uses"], [])

    def test_load_task_rejects_protected_deferred_unknown_and_corrupt_tasks(self):
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("must not read denied tasks")):
            for identifier in (30, 31, 32, 33, 9999, True, "1"):
                with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "not in the saved"):
                    study.load_task(self.loaded, identifier)
        (self.pool / "tasks/test_1.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Task bytes differ"):
            study.load_task(self.loaded, 1)

    def test_load_task_requires_contained_paths_and_identity(self):
        task = json.loads((self.pool / "tasks/test_1.json").read_bytes())
        cases = {
            "outside": (lambda: self.write_task(1, task, "../outside/test_1.json"), "leaves the test pool"),
            "id": (lambda: self.write_task(1, {**task, "id": 2}), "identity differs"),
            "boolean id": (lambda: self.write_task(1, {**task, "id": True}), "identity differs"),
            "difficulty": (lambda: self.write_task(1, {**task, "difficulty": "interview"}), "identity differs"),
        }
        for name, (change, message) in cases.items():
            with self.subTest(name):
                change()
                loaded = self.rewrite()
                with self.assertRaisesRegex(ValueError, message):
                    study.load_task(loaded, 1)
                self.write_task(1, task, "tasks/test_1.json")

    def test_load_task_rechecks_paths_after_loading_the_cohort(self):
        path = self.pool / "tasks/test_1.json"
        outside = self.root / "outside.json"
        path.rename(outside)
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "leaves the test pool"):
            study.load_task(self.loaded, 1)

    def test_experiment_records_hold_one_or_both_independent_recipes(self):
        record = study.experiment_record(self.loaded, "smoke-1", [1, 2], {"synthetic": {"temperature": 0.1}})
        self.assertEqual(record, {
            "schema_version": 2, "experiment_id": "smoke-1",
            "cohort": {"manifest_sha256": self.loaded["sha256"],
                       "test_pool_manifest_sha256": self.cohort["provenance"]["input_manifest_sha256"]},
            "task_ids": [1, 2],
            "reference_gate": {"status": "pending", "requirement": self.cohort["required_next_gate"]},
            "recipes": {"synthetic": {"temperature": 0.1}}})
        paired = study.experiment_record(self.loaded, "smoke-2", [1, 2],
                                         {"synthetic": {"prompt": "b"}, "actor": {"prompt": "a"}})
        self.assertEqual(paired["recipes"], {"actor": {"prompt": "a"}, "synthetic": {"prompt": "b"}})
        self.assertEqual(list(paired["recipes"]), ["actor", "synthetic"])

    def test_experiment_records_reject_invalid_inputs(self):
        for experiment_id in ("", "  ", 7):
            with self.subTest(experiment_id=experiment_id), self.assertRaisesRegex(ValueError, "Experiment ID"):
                study.experiment_record(self.loaded, experiment_id, [1, 2], {"actor": {}})
        for recipes in ({}, {"monitor": {}}, {"actor": []}, [("actor", {})], {"actor": {"stop": ("x",)}},
                        {"actor": {"temperature": math.nan}}, {"actor": {1: "x"}}, {"actor": {"seeds": {1}}}):
            with self.subTest(recipes=recipes), self.assertRaises(ValueError):
                study.experiment_record(self.loaded, "smoke", [1, 2], recipes)
        for task_ids in ([], [1, 1], [True], ["1"], [30], [31], [32], [33], [9999]):
            with self.subTest(task_ids=task_ids), self.assertRaisesRegex(ValueError, "Experiment tasks"):
                study.experiment_record(self.loaded, "smoke", task_ids, {"actor": {}})

    def test_records_are_isolated_snapshots(self):
        task_ids = [1, 2]
        recipe = {"prompt": {"lines": ["a"]}}
        record = study.experiment_record(self.loaded, "smoke", task_ids, {"actor": recipe})
        recipe["prompt"]["lines"].append("b")
        task_ids.append(30)
        self.assertEqual(record["recipes"]["actor"], {"prompt": {"lines": ["a"]}})
        self.assertEqual(record["task_ids"], [1, 2])
        record["task_ids"].clear()
        self.assertEqual(self.loaded["groups"]["development"]["initial"], [1, 2, 3])

    def test_candidate_identity_hashes_exact_bytes_per_source(self):
        digest = "a" * 64
        identity = study.candidate_identity("test", 7, digest, "actor", b"print(1)\n")
        self.assertEqual(identity, {"split": "test", "id": 7, "task_sha256": digest, "source": "actor",
                                    "program_sha256": study.sha256(b"print(1)\n")})
        self.assertEqual(study.candidate_identity("test", 7, digest, "actor", b"print(1)\n"), identity)
        for fields in (("test", 7, digest, "synthetic", b"print(1)\n"), ("test", 7, digest, "actor", b"print(1)\r\n"),
                       ("test", 7, digest, "actor", b"print(1) \n"), ("train", 7, digest, "actor", b"print(1)\n")):
            with self.subTest(fields=fields):
                self.assertNotEqual(study.candidate_identity(*fields), identity)

    def test_candidate_identity_rejects_invalid_fields(self):
        valid = ("test", 7, "a" * 64, "actor", b"0")
        for index, value in ((0, "validation"), (1, True), (1, "7"), (2, "A" * 64), (2, "a" * 63),
                             (3, "monitor"), (4, "print(1)"), (4, bytearray(b"0"))):
            fields = list(valid)
            fields[index] = value
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "Invalid candidate"):
                study.candidate_identity(*fields)


if __name__ == "__main__":
    unittest.main()
