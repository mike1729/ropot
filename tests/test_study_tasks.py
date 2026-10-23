import copy
import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock


study = importlib.import_module("src.data-pipeline.study_tasks")

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
                    "defer_reasons": [], "known_uses": ["unresolved"] if identifier == 20 else []}
                   for identifier, (difficulty, owner, block, phase) in LAYOUT.items()]
        self.records = {record["id"]: record for record in records}
        self.cohort = {
            "tasks": records,
            "queues": {owner: {block: [i for i, (_, o, b, _) in LAYOUT.items() if (o, b) == (owner, block)]
                               for block in BLOCKS} for owner in ("development", "construction", "evaluation")},
            "construction_phases": {phase: {block: [i for i, (_, _, b, p) in LAYOUT.items() if (p, b) == (phase, block)]
                                            for block in BLOCKS} for phase in PHASES},
            "candidate_budgets": {"development": {"smoke": {"total": 2}}, "construction": {"probe_per_phase": 2}},
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
        self.assertEqual((self.loaded["cohort"], self.loaded["test_pool"]), (self.cohort, self.pool_manifest))
        moved = self.root / "moved"
        self.pool.rename(moved)
        with self.assertRaises(FileNotFoundError):
            study.load_cohort(self.cohort_dir)
        loaded = study.load_cohort(self.cohort_dir, moved)
        activation = study.activate(loaded, ["development_smoke"])
        self.assertEqual(study.load_task(loaded, activation, 1)[1]["question"], "é")
        for path, message in ((moved / "manifest.json", "Test-pool manifest"),
                              (self.cohort_dir / "manifest.json", "reviewed publication")):
            path.write_bytes(path.read_bytes() + b" ")
            with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, message):
                study.load_cohort(self.cohort_dir, moved)

    def test_activation_takes_budgeted_prefixes_in_group_and_cohort_order(self):
        activation = study.activate(self.loaded, ["continuation_probe", "development_smoke"])
        self.assertEqual(activation["cohort_sha256"], self.loaded["sha256"])
        self.assertEqual(activation["groups"], ["development_smoke", "continuation_probe"])
        expected = ((1, "development_smoke"), (2, "development_smoke"),
                    (20, "continuation_probe"), (21, "continuation_probe"))
        self.assertEqual(activation["tasks"], [{**self.records[i], "group": group} for i, group in expected])
        shared = study.activate(self.loaded, ["shared_pretraining_probe"])
        self.assertEqual([task["id"] for task in shared["tasks"]], [10, 11])

    def test_activation_rejects_invalid_groups_and_prefixes(self):
        for groups in ([], ["development_smoke"] * 2, ["evaluation"], "development_smoke"):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "distinct groups"):
                study.activate(self.loaded, groups)

        def record(cohort, identifier):
            return next(task for task in cohort["tasks"] if task["id"] == identifier)
        changes = {
            "insufficient": lambda c: c["candidate_budgets"]["construction"].update(probe_per_phase=3),
            "reserve block": lambda c: c["construction_phases"]["shared_honest_pretraining"]["initial"].insert(0, 13),
            "deferred": lambda c: record(c, 2)["defer_reasons"].append("statement_tokens"),
            "protected owner": lambda c: c["queues"]["development"]["initial"].insert(0, 30),
        }
        for name, change in changes.items():
            with self.subTest(name):
                cohort = copy.deepcopy(self.cohort)
                change(cohort)
                with self.assertRaisesRegex(ValueError, "Insufficient|Ineligible"):
                    study.activate(self.rewrite(cohort), list(study.GROUPS))

    def test_load_task_reads_only_the_requested_verified_artifact(self):
        activation = study.activate(self.loaded, list(study.GROUPS))
        (self.pool / "tasks/test_2.json").unlink()
        (self.pool / "tasks/test_10.json").write_bytes(b"corrupt")
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            binding, task = study.load_task(self.loaded, activation, 1)
        self.assertEqual([call.args[0].name for call in read.call_args_list], ["test_1.json"])
        self.assertEqual(binding, activation["tasks"][0])
        self.assertEqual(task, json.loads((self.pool / "tasks/test_1.json").read_bytes()))
        binding["known_uses"].append("changed")
        self.assertEqual(activation["tasks"][0]["known_uses"], [])

    def test_load_task_rejects_inactive_forged_and_corrupt_tasks(self):
        activation = study.activate(self.loaded, ["development_smoke"])
        for identifier in (3, 30, True, "1"):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "not activated"):
                study.load_task(self.loaded, activation, identifier)
        forged = copy.deepcopy(activation)
        forged["tasks"].append({**self.records[30], "group": "development_smoke"})
        with self.assertRaisesRegex(ValueError, "differs from the pinned cohort"):
            study.load_task(self.loaded, forged, 30)
        (self.pool / "tasks/test_1.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "Task bytes differ"):
            study.load_task(self.loaded, activation, 1)

    def test_load_task_requires_agreeing_bindings_contained_paths_and_identity(self):
        task = json.loads((self.pool / "tasks/test_1.json").read_bytes())
        cases = {
            "binding": (lambda: self.pool_manifest["provenance"]["task_sha256"].update(
                {"tasks/test_1.json": "0" * 64}), "bindings disagree"),
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
                    study.load_task(loaded, study.activate(loaded, ["development_smoke"]), 1)
                self.write_task(1, task, "tasks/test_1.json")

    def test_experiment_records_hold_one_or_both_independent_recipes(self):
        activation = study.activate(self.loaded, ["development_smoke"])
        record = study.experiment_record(self.loaded, "smoke-1", activation, {"synthetic": {"temperature": 0.1}})
        self.assertEqual(record, {
            "schema_version": 1, "experiment_id": "smoke-1",
            "cohort": {"manifest_sha256": self.loaded["sha256"],
                       "test_pool_manifest_sha256": self.cohort["provenance"]["input_manifest_sha256"]},
            "activation": activation,
            "reference_gate": {"status": "pending", "requirement": self.cohort["required_next_gate"]},
            "recipes": {"synthetic": {"temperature": 0.1}}})
        paired = study.experiment_record(self.loaded, "smoke-2", activation,
                                         {"synthetic": {"prompt": "b"}, "actor": {"prompt": "a"}})
        self.assertEqual(paired["recipes"], {"actor": {"prompt": "a"}, "synthetic": {"prompt": "b"}})
        self.assertEqual(list(paired["recipes"]), ["actor", "synthetic"])

    def test_experiment_records_reject_invalid_inputs(self):
        activation = study.activate(self.loaded, ["development_smoke"])
        for experiment_id in ("", "  ", 7):
            with self.subTest(experiment_id=experiment_id), self.assertRaisesRegex(ValueError, "Experiment ID"):
                study.experiment_record(self.loaded, experiment_id, activation, {"actor": {}})
        for recipes in ({}, {"monitor": {}}, {"actor": []}, [("actor", {})], {"actor": {"stop": ("x",)}},
                        {"actor": {"temperature": math.nan}}, {"actor": {1: "x"}}, {"actor": {"seeds": {1}}}):
            with self.subTest(recipes=recipes), self.assertRaises(ValueError):
                study.experiment_record(self.loaded, "smoke", activation, recipes)
        forged = copy.deepcopy(activation)
        forged["tasks"].pop()
        with self.assertRaisesRegex(ValueError, "differs from the pinned cohort"):
            study.experiment_record(self.loaded, "smoke", forged, {"actor": {}})

    def test_records_are_isolated_snapshots(self):
        activation = study.activate(self.loaded, ["development_smoke"])
        recipe = {"prompt": {"lines": ["a"]}}
        record = study.experiment_record(self.loaded, "smoke", activation, {"actor": recipe})
        recipe["prompt"]["lines"].append("b")
        activation["tasks"][0]["known_uses"].append("changed")
        self.assertEqual(record["recipes"]["actor"], {"prompt": {"lines": ["a"]}})
        self.assertEqual(record["activation"]["tasks"][0]["known_uses"], [])
        record["activation"]["tasks"].clear()
        fresh = study.activate(self.loaded, ["development_smoke"])
        self.assertEqual((len(fresh["tasks"]), fresh["tasks"][0]["known_uses"]), (2, []))

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
