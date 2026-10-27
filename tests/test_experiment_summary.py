"""Batch 8 summaries consume only real coordinator JSON, with per-run failures."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from tests import test_benign_loop as initial

s = importlib.import_module("src.data-pipeline.stage03.experiment_summary")


@unittest.skipIf(initial.v is None, "producer fixtures require requirements-execution.txt")
class ExperimentSummaryTests(initial.LoopFixture):
    def setUp(self):
        super().setUp()
        self.experiments = self.root / "experiments"
        self.experiments.mkdir()

    def make(self, name="0001-first", **options):
        self.run_loop(**options)
        directory = self.experiments / name
        (self.root / "run").rename(directory)
        return directory

    def read(self, directory, name="outcome.json"):
        return json.loads((directory / name).read_text())

    def write(self, directory, value, name="outcome.json"):
        (directory / name).write_text(json.dumps(value))

    def summary(self):
        original = Path.read_bytes
        def only_coordinator_json(path, *args, **kwargs):
            self.assertIn(path.name, ("inputs.json", "outcome.json"))
            self.assertEqual(path.parent.parent, self.experiments)
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "read_bytes", only_coordinator_json), \
             mock.patch.object(subprocess, "run", side_effect=AssertionError("dispatch forbidden")):
            result = s.summarize(self.experiments)
        json.dumps(result, allow_nan=False)
        return result

    def test_order_totals_and_repeated_draws_preserve_two_task_histories(self):
        second = self.make("0002-second")
        first = self.make("0001-first")
        (self.experiments / "ignored-file.json").write_text("{}")
        result = self.summary()
        self.assertEqual([e["experiment"] for e in result["experiments"]], [first.name, second.name])
        self.assertEqual(len(result["cells"]), 2)
        for cell in result["cells"]:
            self.assertEqual([e["experiment"] for e in cell["experiments"]], [first.name, second.name])
            self.assertEqual(len(cell["passes"]), 2)
            self.assertEqual(cell["latest_state"], "passed")
            self.assertEqual(cell["experiments"][0]["binding"], self.worklist[cell["id"]][0])
        for experiment in result["experiments"]:
            self.assertEqual(experiment["task_state_counts"], {"passed": 2})
            self.assertEqual(experiment["worklist_size"], 2)
            self.assertEqual(experiment["recipe"], self.recipe)
            self.assertEqual(experiment["model_manifest_sha256"], initial.b.generation.file_hash(self.manifest))
            for name in ("inputs", "outcome"):
                self.assertEqual(experiment[name + "_sha256"],
                                 initial.b.generation.file_hash(self.experiments / experiment["experiment"] / (name + ".json")))
        totals = result["totals"]
        for key, expected in (("committed_replies", 4), ("committed_tokens", 64),
                              ("known_submissions", 4), ("observed_tokens", 8)):
            self.assertEqual(totals[key], dict(recorded_total=expected, known_subtotal=expected, complete=True))
        self.assertAlmostEqual(totals["timing"]["loading"]["recorded_total"], .2)
        self.assertTrue(totals["usage_complete"])
        self.assertEqual(result, self.summary())

    def test_minimal_fixture_bindings_are_copied_as_is(self):
        self.worklist = [({k: b[k] for k in ("split", "id", "sha256")}, task) for b, task in self.worklist]
        self.make()
        result = self.summary()
        self.assertEqual(result["cells"][0]["experiments"][0]["binding"], self.worklist[0][0])

    def test_different_fixture_content_at_same_split_id_has_separate_history(self):
        self.make("0001-original")
        binding, task = deepcopy(self.worklist[0])
        task["question"] = "Different fixture statement"
        path = self.root / "changed-fixture.json"
        initial.b.generation.save(path, task)
        binding["sha256"] = initial.b.generation.file_hash(path)
        self.worklist[0] = binding, task
        self.make("0002-changed")
        cells = self.summary()["cells"]
        self.assertEqual(len(cells), 3)
        same_id = [c for c in cells if c["id"] == 0]
        self.assertEqual(len({c["sha256"] for c in same_id}), 2)
        self.assertEqual([c["passes"] for c in same_id], [
            [dict(experiment="0001-original", attempt=0)],
            [dict(experiment="0002-changed", attempt=0)]])
        self.assertEqual([len(c["experiments"]) for c in same_id], [1, 1])

    def test_name_errors_and_duplicate_prefixes_are_root_errors(self):
        for names in (("first",), ("001-short",), ("0001-UPPER",), ("0001-a", "0001-b")):
            with self.subTest(names=names):
                for name in names:
                    (self.experiments / name).mkdir()
                with self.assertRaisesRegex(ValueError, "unique four-digit prefixes"):
                    self.summary()
                for name in names:
                    (self.experiments / name).rmdir()

    def test_fixture_study_mixing_is_refused(self):
        self.make("0001-fixture")
        self.make("0002-study", fixtures=False, cohort_sha256=initial.b.study_tasks.COHORT_SHA256)
        with self.assertRaisesRegex(ValueError, "Cannot mix"):
            self.summary()

    def test_distinct_study_cohorts_are_refused_without_digest_checks(self):
        self.make("0001-study", fixtures=False, cohort_sha256=initial.b.study_tasks.COHORT_SHA256)
        second = self.make("0002-study", fixtures=False, cohort_sha256=initial.b.study_tasks.COHORT_SHA256)
        inputs = self.read(second, "inputs.json")
        inputs["cohort_sha256"] = "another-recorded-publication"
        self.write(second, inputs, "inputs.json")
        with self.assertRaisesRegex(ValueError, "different cohort"):
            self.summary()

    def test_known_not_run_membership_is_not_submission_or_exposure(self):
        def fail(command, current, persist):
            current["popen_error"] = "FileNotFoundError('fixture')"
            raise FileNotFoundError("fixture")
        self.child.side_effect = fail
        self.make()
        result = self.summary()
        self.assertEqual(result["experiments"][0]["task_state_counts"], {"not_run": 2})
        for cell in result["cells"]:
            self.assertEqual(cell["latest_state"], "not_run")
            self.assertFalse(cell["experiments"][0]["attempts"][0]["submitted"])
            self.assertEqual(cell["passes"], [])
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 32)
        self.assertEqual(result["totals"]["known_submissions"]["recorded_total"], 0)

    def test_damaged_inputs_keep_readable_exact_effort_and_healthy_cells(self):
        self.make("0001-good")
        broken = self.make("0002-bad-input")
        (broken / "inputs.json").write_text("{")
        result = self.summary()
        bad = result["experiments"][1]
        self.assertIsNone(bad["worklist_size"])
        self.assertIsNone(bad["fixtures"])
        self.assertTrue(bad["problems"])
        self.assertEqual(bad["task_state_counts"], {"passed": 2})
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 64)
        self.assertTrue(result["totals"]["committed_tokens"]["complete"])
        self.assertEqual(len(result["cells"]), 2)

    def test_absent_outcome_means_zero_dispatch_even_with_partial_inputs(self):
        self.make("0001-good")
        timing_before = self.summary()["totals"]["timing"]
        early = self.experiments / "0002-early-failure"
        early.mkdir()
        (early / "inputs.json").write_text("{")
        result = self.summary()
        self.assertEqual(result["experiments"][1]["status"], "not_run")
        self.assertEqual(result["experiments"][1]["counters"]["committed_tokens"], 0)
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 32)
        self.assertTrue(result["totals"]["usage_complete"])
        self.assertEqual(result["totals"]["timing"], timing_before)

    def test_absent_outcome_has_zero_complete_timing_for_every_phase(self):
        (self.experiments / "0001-empty").mkdir()
        result = self.summary()
        self.assertEqual(result["experiments"][0]["timing"], dict.fromkeys(s.PHASES, 0))
        for phase in s.PHASES:
            self.assertEqual(result["totals"]["timing"][phase],
                             dict(recorded_total=0, known_subtotal=0, complete=True))

    def test_lost_outcome_with_saved_round_is_unknown_cost(self):
        self.make("0001-good")
        run = self.make("0002-lost-outcome")
        self.assertTrue((run / "round-0000.json").exists())
        (run / "outcome.json").unlink()
        result = self.summary()
        record = result["experiments"][1]
        self.assertEqual(record["status"], "unresolved")
        self.assertIsNone(record["usage_complete"])
        self.assertTrue(any("saved round" in p["error"] for p in record["problems"]))
        for key in s.COUNTERS:
            self.assertIsNone(result["totals"][key]["recorded_total"])
            self.assertFalse(result["totals"][key]["complete"])
        self.assertEqual(result["totals"]["committed_tokens"]["known_subtotal"], 32)
        self.assertIsNone(result["totals"]["timing"]["overall"]["recorded_total"])
        self.assertTrue(all(c["latest_state"] == "unresolved" for c in result["cells"]))

    def test_unreadable_outcome_is_unknown_and_does_not_hide_healthy_work(self):
        self.make("0001-good")
        broken = self.make("0002-unreadable")
        original = Path.read_bytes
        def unreadable(path, *args, **kwargs):
            if path == broken / "outcome.json":
                raise PermissionError("fixture permission failure")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "read_bytes", unreadable):
            result = s.summarize(self.experiments)
        metric = result["totals"]["committed_tokens"]
        self.assertIsNone(metric["recorded_total"])
        self.assertEqual(metric["known_subtotal"], 32)
        self.assertFalse(metric["complete"])
        self.assertEqual(result["cells"][0]["latest_state"], "unresolved")
        self.assertEqual(len(result["cells"][0]["passes"]), 1)

    def test_malformed_and_unsupported_outcomes_are_per_experiment(self):
        self.make("0001-good")
        broken = self.make("0002-bad-outcome")
        original = self.read(broken)
        wrong_format = deepcopy(original)
        wrong_format["rounds"][0]["attempts"] = wrong_format["rounds"][0].pop("attempt_ids")
        wrong_state_type = deepcopy(original)
        wrong_state_type["status"] = []
        for raw, recorded in (("{", None), ("[]", None), (json.dumps(wrong_format), 64), (json.dumps(wrong_state_type), 64)):
            with self.subTest(raw=raw[:30]):
                (broken / "outcome.json").write_text(raw)
                result = self.summary()
                self.assertTrue(result["experiments"][1]["problems"])
                self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], recorded)
                self.assertFalse(result["totals"]["observed_tokens"]["complete"])
                self.assertEqual(result["cells"][0]["latest_state"], "unresolved")

    def test_unfinished_run_keeps_exact_commitments_and_recorded_completeness_flag(self):
        run = self.make()
        outcome = self.read(run)
        outcome.update(status="running", phase="generation", usage_complete=True)
        self.write(run, outcome)
        result = self.summary()
        record = result["experiments"][0]
        self.assertEqual(record["counters"], {key: outcome[key] for key in s.COUNTERS})
        self.assertTrue(record["usage_complete"])
        self.assertEqual(record["observations"], "snapshot/lower_bound")
        self.assertIsNone(record["task_state_counts"])
        self.assertTrue(result["totals"]["committed_tokens"]["complete"])
        self.assertFalse(result["totals"]["observed_tokens"]["complete"])
        self.assertFalse(result["totals"]["timing"]["overall"]["complete"])
        self.assertTrue(all(c["latest_state"] == "unresolved" for c in result["cells"]))
        self.assertEqual(len(result["cells"][0]["passes"]), 1)

    def test_optional_lower_bound_and_null_timings_keep_unknowns(self):
        self.make("0001-good")
        run = self.make("0002-partial")
        outcome = self.read(run)
        outcome.pop("known_submissions_lower_bound")
        outcome["timing"]["loading"] = None
        self.write(run, outcome)
        result = self.summary()
        self.assertEqual(result["totals"]["known_submissions_lower_bound"]["recorded_total"], 4)
        self.assertIsNone(result["totals"]["timing"]["loading"]["recorded_total"])
        self.assertEqual(result["totals"]["timing"]["loading"]["known_subtotal"], .1)

    def test_incomplete_native_outcome_preserves_real_unknown_submission_cost(self):
        self.make("0001-good")
        self.texts = []
        run = self.make("0002-incomplete")
        outcome = self.read(run)
        self.assertEqual(outcome["status"], "error")
        self.assertEqual([t["state"] for t in outcome["tasks"]], ["submission_unknown"] * 2)
        result = self.summary()
        self.assertIsNone(result["totals"]["known_submissions"]["recorded_total"])
        self.assertEqual(result["totals"]["known_submissions"]["known_subtotal"], 2)
        self.assertEqual(result["totals"]["known_submissions_lower_bound"]["recorded_total"], 2)
        self.assertEqual(result["totals"]["observed_tokens"]["recorded_total"], 4)
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 64)
        self.assertFalse(result["totals"]["usage_complete"])

    def test_cancelled_coordinator_outcome_preserves_states_and_reservations(self):
        self.native_status, self.row_statuses = "cancelled", ["cancelled", "not_run"]
        def interrupt(*args):
            self.generate(*args)
            raise KeyboardInterrupt("fixture cancellation")
        self.child.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop()
        (self.root / "run").rename(self.experiments / "0001-cancelled")
        result = self.summary()
        record = result["experiments"][0]
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(record["terminal"])
        self.assertEqual(record["task_state_counts"], dict(cancelled=1, not_run=1))
        self.assertEqual([c["latest_state"] for c in result["cells"]], ["cancelled", "not_run"])
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 32)
        self.assertEqual(result["totals"]["known_submissions"]["recorded_total"], 1)
        self.assertFalse(result["totals"]["usage_complete"])

    def test_repair_attempts_and_every_pass_are_copied_from_producer(self):
        repairs = importlib.import_module("tests.test_benign_repairs")
        fixture = repairs.RepairTests(methodName="run_loop")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.verdicts[0, 1] = "failed"
        fixture.run_loop()
        produced = fixture.saved()
        (fixture.root / "run").rename(self.experiments / "0001-repairs")
        result = self.summary()
        cell = result["cells"][1]
        self.assertEqual(cell["experiments"][0]["attempts"], [produced["attempts"][1], produced["attempts"][2]])
        self.assertEqual(cell["passes"], [dict(experiment="0001-repairs", attempt=2)])
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 52)
        self.assertEqual(result["totals"]["observed_tokens"]["recorded_total"], 6)

    def test_import_and_summary_require_no_inspect_mlx_or_subprocess(self):
        self.make()
        code = """
import builtins, importlib, subprocess, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('inspect_ai', 'mlx', 'mlx_lm'):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('dispatch forbidden'))
reader = importlib.import_module('src.data-pipeline.stage03.experiment_summary')
assert len(reader.summarize(sys.argv[1])['experiments']) == 1
"""
        completed = subprocess.run([sys.executable, "-c", code, str(self.experiments)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_empty_root_is_a_complete_empty_inventory(self):
        result = self.summary()
        self.assertEqual(result["experiments"], [])
        self.assertEqual(result["cells"], [])
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 0)

    def test_both_unreadable_records_leave_cost_unknown_without_hiding_other_runs(self):
        self.make("0001-good")
        run = self.make("0002-damaged")
        for name in ("inputs.json", "outcome.json"):
            (run / name).write_text("{")
        result = self.summary()
        self.assertEqual(len(result["experiments"]), 2)
        self.assertIsNone(result["experiments"][1]["fixtures"])
        self.assertIsNone(result["totals"]["observed_tokens"]["recorded_total"])
        self.assertEqual(result["totals"]["observed_tokens"]["known_subtotal"], 4)
        self.assertFalse(result["totals"]["usage_complete"])

    def test_invalid_counter_is_diagnosed_and_not_summed_as_zero(self):
        self.make("0001-good")
        run = self.make("0002-damaged")
        outcome = self.read(run)
        outcome["observed_tokens"] = "unknown"
        self.write(run, outcome)
        result = self.summary()
        self.assertIn("Invalid recorded counter", result["experiments"][1]["problems"][0]["error"])
        self.assertIsNone(result["totals"]["observed_tokens"]["recorded_total"])
        self.assertEqual(result["totals"]["observed_tokens"]["known_subtotal"], 4)
        self.assertFalse(result["totals"]["usage_complete"])

    def test_missing_null_counters_and_missing_timing_have_diagnostics(self):
        self.make("0001-good")
        run = self.make("0002-damaged")
        original = self.read(run)
        for field, value in (("committed_tokens", "absent"), ("committed_replies", None), ("timing", "absent")):
            with self.subTest(field=field):
                outcome = deepcopy(original)
                if value == "absent":
                    outcome.pop(field)
                else:
                    outcome[field] = value
                self.write(run, outcome)
                result = self.summary()
                record = result["experiments"][1]
                self.assertEqual(record["status"], "completed")
                self.assertTrue(any(field in p["error"] for p in record["problems"]))
                metric = result["totals"]["timing"]["overall"] if field == "timing" else result["totals"][field]
                self.assertIsNone(metric["recorded_total"])
                self.assertFalse(metric["complete"])

    def test_missing_study_cohort_is_reported_without_hiding_readable_work(self):
        run = self.make(fixtures=False, cohort_sha256=initial.b.study_tasks.COHORT_SHA256)
        original = self.read(run, "inputs.json")
        for digest in (None, ""):
            with self.subTest(digest=digest):
                inputs = deepcopy(original)
                if digest is None:
                    inputs.pop("cohort_sha256")
                else:
                    inputs["cohort_sha256"] = digest
                self.write(run, inputs, "inputs.json")
                result = self.summary()
                self.assertTrue(any("cohort digest" in p["error"] for p in result["experiments"][0]["problems"]))
                self.assertEqual(len(result["cells"]), 2)
                self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 32)

    def test_deeply_nested_json_is_a_per_experiment_read_error(self):
        self.make("0001-good")
        run = self.make("0002-damaged")
        for name in ("inputs.json", "outcome.json"):
            original = (run / name).read_text()
            (run / name).write_text("[" * 200_000)
            result = self.summary()
            self.assertEqual(len(result["experiments"]), 2)
            self.assertTrue(any(p["file"] == name and "recursion" in p["error"].lower()
                                for p in result["experiments"][1]["problems"]))
            self.assertEqual(result["experiments"][0]["status"], "completed")
            (run / name).write_text(original)

    def test_unknown_fixture_provenance_does_not_drop_readable_bindings_or_cost(self):
        run = self.make()
        inputs = self.read(run, "inputs.json")
        inputs.pop("fixtures")
        self.write(run, inputs, "inputs.json")
        result = self.summary()
        self.assertIsNone(result["experiments"][0]["fixtures"])
        self.assertTrue(result["experiments"][0]["problems"])
        self.assertEqual(result["cells"][0]["experiments"][0]["binding"], self.worklist[0][0])
        self.assertEqual(result["totals"]["committed_tokens"]["recorded_total"], 32)


if __name__ == "__main__":
    unittest.main()
