import importlib
import json
from pathlib import Path
import unittest
from unittest import mock


c = importlib.import_module("src.data-pipeline.stage02.cohort")
p = importlib.import_module("src.data-pipeline.stage02.publish_cohort")
input_tests = importlib.import_module("tests.test_cohort_inputs")


class PublishCohortTests(unittest.TestCase):
    fixture = input_tests.CohortInputTests.fixture
    save = input_tests.CohortInputTests.save

    def setUp(self):
        self.fixture()
        self.output = self.root / "cohort"
        self.tasks[1]["difficulty"] = self.rows[1]["difficulty"] = "interview"
        self.rows[1]["statement_tokens"] = c.PROFILE["statement_tokens"] + 1
        name = "tasks/test_9.json"
        raw = json.dumps(self.tasks[1]).encode()
        (self.pool / name).write_bytes(raw)
        for provenance in (self.manifest["provenance"], self.summary["provenance"]):
            provenance["task_sha256"][name] = c.sha256(raw)
        self.observations["observations"] = [{
            "task_id": "apps:test:7", "row_sha256": input_tests.known.row_sha256(self.tasks[0]["original_row"]),
            "use": "prompt_checker_development"}]
        self.save()

    def publish(self, output=None):
        return p.publish_cohort(self.pool, self.analysis, output or self.output,
                                self.records_path, self.coverage_path)

    def test_reduction_is_task_scoped_deterministic_and_retains_all_visible_cases(self):
        task = {"split": "test", "id": 7, "cases": [
            {"index": i, "input": str(i), "answer": str(i), "visible": i in (0, 7, 29)} for i in range(30)]}
        reduced = p.reduce_task(task)
        expected = [0, 7, 29, 28, 8, 21, 11, 2, 9, 18, 17, 13, 14, 3, 22, 27, 6, 5, 23, 16]
        self.assertEqual([case["index"] for case in reduced["cases"]], expected)
        self.assertEqual((reduced["visible_case_count"], reduced["nonvisible_case_count"]), (3, 17))
        self.assertEqual(reduced, p.reduce_task({**task, "cases": list(reversed(task["cases"]))}))
        self.assertNotEqual(reduced["cases"], p.reduce_task({**task, "id": 8})["cases"])
        self.assertEqual(len(task["cases"]), 30)
        for count in (0, 5, 20):
            small = {**task, "cases": task["cases"][:count]}
            self.assertEqual({c["index"] for c in p.reduce_task(small)["cases"]}, set(range(count)))
        for cases in ([{"index": i, "visible": True} for i in range(21)],
                      [{"index": 0, "visible": False}] * 2, [{"index": 0}],
                      [{"index": True, "visible": False}], [{"index": 0, "visible": 1}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                p.reduce_task({**task, "cases": cases})

    def test_reduced_publication_keeps_pool_intact_and_binds_original_and_reduced_hashes(self):
        task = self.tasks[0]
        task["cases"] = [{"index": i, "input": str(i), "answer": str(i), "visible": i in (0, 7, 29)}
                         for i in range(30)]
        for field in ("verification_attempts", "lengths", "reference_counts"):
            task[field] = {"historical": "full suite"}
        relative = "tasks/test_7.json"
        raw = json.dumps(task).encode()
        (self.pool / relative).write_bytes(raw)
        for provenance in (self.manifest["provenance"], self.summary["provenance"]):
            provenance["task_sha256"][relative] = c.sha256(raw)
        self.rows[0].update(cases=30, visible_cases=3, nonvisible_cases=27)
        self.save()
        before = {path: path.read_bytes() for path in self.pool.rglob("*") if path.is_file()}
        manifest = self.publish()
        record = next(t for t in manifest["tasks"] if t["id"] == 7)
        reduced_raw = (self.output / record["path"]).read_bytes()
        reduced = json.loads(reduced_raw)
        self.assertEqual(record["source_sha256"], c.sha256(raw))
        self.assertEqual(record["sha256"], c.sha256(reduced_raw))
        self.assertNotEqual(record["source_sha256"], record["sha256"])
        self.assertEqual(reduced, p.reduce_task(task))
        for field in ("original_row", "verification_attempts", "lengths", "reference_counts"):
            self.assertNotIn(field, reduced)
        self.assertEqual({path: path.read_bytes() for path in before}, before)
        self.assertEqual(json.loads((self.output / "report.json").read_text())["cases"],
                         {"original": 32, "published": 22})

    def test_roundtrip_preserves_allocation_inputs_and_frozen_contract(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        loaded = c.load_inputs(self.pool, self.analysis, self.records_path, self.coverage_path)
        expected = c.allocate(loaded["tasks"])
        manifest = self.publish()
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["manifest.json", "report.json", "tasks"])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["schema_version"], 3)
        task_fields = {"split", "id", "difficulty", "path", "sha256", "owner", "block", "defer_reasons", "known_uses"}
        optional_fields = {"phase", "redirected_from"}
        for published, allocated in zip(manifest["tasks"], expected["tasks"], strict=True):
            fields = task_fields | (optional_fields & allocated.keys())
            self.assertEqual(set(published), fields | {"source_sha256"})
            self.assertEqual({key: published[key] for key in fields - {"sha256"}},
                             {key: allocated[key] for key in fields - {"sha256"}})
            self.assertEqual(published["source_sha256"], allocated["sha256"])
            raw = (self.output / published["path"]).read_bytes()
            self.assertEqual(published["sha256"], c.sha256(raw))
            self.assertEqual(json.loads(raw), p.reduce_task(json.loads((self.pool / published["path"]).read_bytes())))
        for key in ("groups", "construction_phases"):
            self.assertEqual(manifest[key], expected[key])
        for key in ("shortfalls", "sequential_remaining"):
            self.assertNotIn(key, manifest)
        self.assertEqual(manifest["known_use_coverage"], loaded["known_use_coverage"])
        self.assertEqual(manifest["required_next_gate"], loaded["required_next_gate"])
        provenance = manifest["provenance"]
        for key, value in loaded["provenance"].items():
            self.assertEqual(provenance[key], value)
        self.assertEqual(provenance["input_dir"], str(self.pool.resolve()))
        self.assertEqual(provenance["analysis_dir"], str(self.analysis.resolve()))
        sources = Path(p.__file__).parents[1]
        self.assertEqual(provenance["source_sha256"],
                         {path.relative_to(sources).as_posix(): c.sha256(path.read_bytes()) for path in sources.rglob("*.py")})
        policy = manifest["policy"]
        for key, value in {"seed": c.SEED, "profile": c.PROFILE, "difficulty_order": list(c.DIFFICULTIES),
                           "initial_quota_order": list(c.QUOTAS), "reserve_cycle": list(c.RESERVE_CYCLE),
                           "block_order": list(c.BLOCKS), "construction_phase_order": list(c.PHASES),
                           "restricted_uses": sorted(c.RESTRICTED_USES)}.items():
            self.assertEqual(policy[key], value)
        self.assertEqual(policy["initial_quotas"],
                         {owner: dict(zip(c.DIFFICULTIES, quotas)) for owner, quotas in c.QUOTAS.items()})
        self.assertEqual(policy["construction_phase_start_by_difficulty"], dict(zip(c.DIFFICULTIES, c.PHASES)))
        self.assertEqual(policy["construction_groups"], "construction_phases")
        self.assertEqual(manifest["targets"], {
            "status": "planned", "finalization": "after_32_candidates_per_phase_feasibility_probe",
            "constraint": "final_targets_must_not_exceed_owned_capacity",
            "unit": "usable_matched_tasks", "construction": {"per_phase": 128, "total": 256,
                "per_phase_difficulty": {"introductory": 31, "interview": 97},
                "nested_per_phase": [32, 64, 128], "optional_expansion_per_phase": 256,
                "optional_expansion_per_phase_difficulty": {"introductory": 62, "interview": 194}},
            "correction_replay": 24, "calibration": 24, "evaluation": 48})
        self.assertEqual(manifest["candidate_budgets"], {
            "development": {"task_id_cap": 50, "smoke": {"total": 8, "introductory": 2, "interview": 6},
                            "training_choices": 42},
            "construction": {"probe_per_phase": 16, "feasibility_per_phase": 32}})
        self.assertEqual(manifest["acquisition_ceilings"], "pending")
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_report_capacities_are_derived_from_membership(self):
        loaded = c.load_inputs(self.pool, self.analysis, self.records_path, self.coverage_path)
        allocation = c.allocate(loaded["tasks"])
        manifest = self.publish()
        report = json.loads((self.output / "report.json").read_text())
        records = manifest["tasks"]
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["cases"], {"original": 4, "published": 4})
        self.assertEqual(report["counts"], {"tasks": 2, "fast_profile": 1, "deferred": 1,
                                            "difficulty": {"introductory": 1, "interview": 1}})
        for key, field, names in (("by_owner", "owner", c.QUOTAS),
                                  ("by_construction_phase", "phase", c.PHASES)):
            self.assertEqual(set(report[key]), set(names))
            for name in names:
                for block in (*c.BLOCKS, "fast_profile", "total"):
                    selected = [t for t in records if t.get(field) == name and (
                        block == "total" or block == "fast_profile" and t["block"] != "deferred"
                        or t["block"] == block)]
                    self.assertEqual(report[key][name][block], {
                        "tasks": len(selected), "difficulty": {
                            d: sum(t["difficulty"] == d for t in selected) for d in c.DIFFICULTIES}})
        self.assertEqual(report["profile_deferrals"], {metric: int(metric == "statement_tokens") for metric in c.PROFILE})
        self.assertEqual(report["sequential_remaining"], allocation["sequential_remaining"])
        self.assertEqual(report["initial_quota_shortfalls"], allocation["shortfalls"])
        self.assertEqual(report["known_use"], {"matched_tasks": 1, "redirected_tasks": 1,
                                               "redirected_from": {"evaluation": 1}})
        for key in ("known_use_coverage", "required_next_gate"):
            self.assertEqual(report[key], manifest[key])
        limitations = " ".join(report["limitations"]).lower().replace(",", "")
        for subject in ("incomplete", "accepted", "retry", "2048"):
            self.assertIn(subject, limitations)

    def test_publication_is_byte_deterministic_across_destinations(self):
        self.publish()
        second = self.root / "other-cohort"
        self.publish(second)
        for path in self.output.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(self.output)).read_bytes())

    def test_saved_groups_partition_tasks_and_stage03_consumes_them_directly(self):
        manifest = self.publish()
        identifiers = [identifier for blocks in manifest["groups"].values()
                       for ids in blocks.values() for identifier in ids]
        self.assertEqual(sorted(identifiers), sorted(task["id"] for task in manifest["tasks"]))
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(manifest["groups"]), set(c.QUOTAS))
        for block in c.BLOCKS:
            phase_ids = [identifier for blocks in manifest["construction_phases"].values()
                         for identifier in blocks[block]]
            self.assertEqual(sorted(phase_ids), sorted(manifest["groups"]["construction"][block]))
        study = importlib.import_module("src.data-pipeline.stage03.study_tasks")
        digest = c.sha256((self.output / "manifest.json").read_bytes())
        with mock.patch.object(study, "COHORT_SHA256", digest), \
                mock.patch.object(c, "allocate", side_effect=AssertionError("must not repartition")):
            loaded = study.load_cohort(self.output)
            self.assertEqual(study.load_task(loaded, 7)[1], p.reduce_task(self.tasks[0]))
        self.assertEqual(set(loaded["groups"]), {"construction", "development"})

    def test_existing_dangling_symlink_and_nested_destinations_are_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError): self.publish()
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError): self.publish()
        self.assertTrue(self.output.is_symlink())
        self.output.unlink()
        alias = self.root / "pool-alias"
        alias.symlink_to(self.pool, target_is_directory=True)
        for source in (self.pool, self.analysis, alias):
            with self.subTest(source=source):
                with self.assertRaises(ValueError): self.publish(source / "nested")
                self.assertFalse((source / "nested").exists())
        self.assertFalse(list(self.root.glob(".cohort-*")))

    def test_invalid_input_aborts_before_publication(self):
        path = self.analysis / "tasks.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Analysis tasks.csv hash"):
            self.publish()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".cohort-*")))

    def test_cli_dispatches_defaults_and_explicit_paths(self):
        cli = importlib.import_module("src.data-pipeline.__main__")
        defaults = (p.TEST_OUTPUT, p.ANALYSIS_OUTPUT, p.OUTPUT, p.RECORDS, p.COVERAGE)
        overrides = (self.pool, self.analysis, self.output, self.records_path, self.coverage_path)
        flags = ("--input-dir", "--analysis-dir", "--output-dir", "--known-use-records", "--known-use-coverage")
        explicit = [item for flag, path in zip(flags, overrides) for item in (flag, str(path))]
        for arguments, expected in (([], defaults), (explicit, overrides)):
            with self.subTest(arguments=arguments), mock.patch("sys.argv", ["pipeline", "cohort", *arguments]), \
                    mock.patch.object(cli, "publish_cohort", return_value={"tasks": []}) as publish, \
                    mock.patch("builtins.print"):
                cli.main()
                publish.assert_called_once_with(*expected)

    def test_failure_after_manifest_write_cleans_up_the_whole_publication(self):
        write_json = p.write_json

        def interrupted(path, value):
            if path.name == "report.json":
                self.assertTrue((path.parent / "manifest.json").is_file())
                self.assertFalse(self.output.exists())
                raise RuntimeError("interrupted report")
            write_json(path, value)

        with mock.patch.object(p, "write_json", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted report"):
                self.publish()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".cohort-*")))
