"""Exercise the live gate's acceptance rules using real, mocked-child loop records."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from tests import test_benign_repairs as repairs
from tests.test_benign_repairs_live import assert_repair_coverage


@unittest.skipIf(repairs.v is None, "requires requirements-execution.txt")
class RepairGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = repairs.RepairTests(methodName="run_loop")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.recipe["max_repairs"] = 1

    def check(self, result):
        return assert_repair_coverage(self, result, self.fixture.root / "run",
                                     self.fixture.worklist, self.fixture.recipe)

    def test_both_tasks_can_fail_initially_and_both_repairs_can_fail(self):
        self.fixture.verdicts.update({(r, p): "failed" for r in range(2) for p in range(2)})
        result = self.fixture.run_loop()
        self.assertEqual(self.check(result), [[0, 2], [1, 3]])
        self.assertEqual([t["state"] for t in result["tasks"]], ["repairs_exhausted"] * 2)

    def test_other_task_may_exhaust_length_or_extraction_failures(self):
        self.fixture.verdicts[0, 0] = "failed"
        for key in ((0, 1), (1, 1)):
            self.fixture.draws[key] = "length", "print(0)"
        result = self.fixture.run_loop()
        self.assertEqual(self.check(result), [[0, 2]])

    def test_other_task_may_return_only_prose(self):
        self.fixture.verdicts[0, 0] = "failed"
        self.fixture.draws.update({(0, 1): ("stop", "prose only"), (1, 1): ("stop", "prose only")})
        self.assertEqual(self.check(self.fixture.run_loop()), [[0, 2]])

    def test_all_initial_passes_do_not_establish_repair_coverage(self):
        with self.assertRaisesRegex(AssertionError, "No verified initial failure"):
            self.check(self.fixture.run_loop())

    def test_extraction_failure_then_pass_does_not_cover_public_verifier_feedback(self):
        self.fixture.draws[0, 0] = "length", "print(0)"
        with self.assertRaisesRegex(AssertionError, "No verified initial failure"):
            self.check(self.fixture.run_loop())

    def test_technical_verification_error_is_rejected(self):
        self.fixture.verdicts[0, 0] = "error"
        with self.assertRaises(AssertionError):
            self.check(self.fixture.run_loop())

    def test_wrong_sample_association_commitment_and_source_are_rejected(self):
        self.fixture.verdicts[0, 0] = "failed"
        original = self.fixture.run_loop()
        for kind in ("sample", "cost", "source"):
            with self.subTest(kind=kind):
                changed = deepcopy(original)
                if kind == "sample":
                    changed["attempts"][2]["verification"]["sample_id"] = 99
                elif kind == "cost":
                    changed["committed_tokens"] -= 1
                else:
                    changed["attempts"][0]["candidate"]["source"] = "synthetic"
                repairs.b.generation.save(self.fixture.root / "run/outcome.json", changed)
                with self.assertRaises(AssertionError):
                    self.check(changed)

    def test_swapped_programs_are_rejected_despite_consistent_candidate_and_verdict_keys(self):
        self.fixture.verdicts[0, 0] = "failed"
        result = self.fixture.run_loop()
        first, second = result["attempts"][:2]
        paths = [Path(a["program"]) for a in (first, second)]
        contents = [p.read_bytes() for p in paths]
        for attempt, path, program in zip((first, second), paths, reversed(contents), strict=True):
            path.write_bytes(program)
            binding, _ = self.fixture.worklist[attempt["position"]]
            attempt["candidate"] = repairs.b.study_tasks.candidate_identity(
                binding["split"], binding["id"], binding["sha256"], "actor", program)
            attempt["verification"]["key"]["program_sha256"] = attempt["candidate"]["program_sha256"]
        repairs.b.generation.save(self.fixture.root / "run/outcome.json", result)
        with self.assertRaises(AssertionError):
            self.check(result)

    def test_wrong_feedback_is_rejected_even_when_both_request_records_match(self):
        self.fixture.verdicts[0, 0] = "failed"
        result = self.fixture.run_loop()
        current = result["rounds"][1]
        for path in (Path(current["round_file"]), Path(current["generation_outcome"]).parent / "inputs.json"):
            record = json.loads(path.read_text())
            record["requests"][0]["messages"][-1]["content"] = "Unrelated repair instruction"
            repairs.b.generation.save(path, record)
        with self.assertRaises(AssertionError):
            self.check(result)
