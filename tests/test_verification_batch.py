from copy import deepcopy
import importlib
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from tests.test_verification import native, task, v, CONTRACT


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class BatchVerificationTests(unittest.TestCase):
    def fixture(self, concurrent=True, concurrency=4):
        programs = [b"print(input())", b"print(int(input()))"]
        log = native(task(), programs[0])
        other = native(task(), programs[1]).samples[0]
        other.id = 1
        log.samples.append(other)
        log.eval.metadata.update(max_samples=concurrency, concurrent_inference=concurrent)
        return log, [(task(), "a"*64, p) for p in programs]

    def test_one_eval_preserves_identity_and_context_during_multi_sample_reuse(self):
        log, candidates = self.fixture()
        with mock.patch.object(v.runner, "run", return_value=[log]) as execute, \
             mock.patch.object(v.runner, "execution_contract", side_effect=lambda **kw: {**CONTRACT, **kw}), \
             mock.patch.object(v, "read_eval_log", return_value=log):
            results = v.verify_many(candidates, "unused", concurrent_inference=True)
            self.assertEqual([r["sample_id"] for r in results], [0, 1])
            self.assertEqual(execute.call_count, 1)
            self.assertEqual(execute.call_args.kwargs["max_samples"], 4)
            for candidate, result in zip(candidates, results):
                self.assertEqual(v.reuse("fixture", *candidate, sample_id=result["sample_id"],
                                        concurrent_inference=True), result)
            for options in ({}, {"sample_id": 0}, {"sample_id": 1, "concurrent_inference": True, "max_samples": 1}):
                with self.assertRaises(ValueError):
                    v.reuse("fixture", *candidates[1], **options)

    def test_exact_sample_binding_duplicate_ids_and_invalid_ids(self):
        log, candidates = self.fixture()
        request = v.request(*candidates[1])
        for sample_id in (0, 2, True, -1):
            with self.subTest(sample_id=sample_id), self.assertRaises(ValueError):
                v.derive(log, request, log.eval.metadata, sample_id=sample_id)
        duplicate = deepcopy(log)
        duplicate.samples.append(duplicate.samples[0])
        with self.assertRaises(ValueError):
            v.derive(duplicate, request, duplicate.eval.metadata, sample_id=1)

    def test_failed_eval_yields_only_generic_infrastructure_feedback_without_retry(self):
        log, candidates = self.fixture()
        log.status, log.samples[0].error = "error", NS(message="private details")
        with mock.patch.object(v.runner, "run", return_value=[log]) as execute:
            results = v.verify_many(candidates, "unused")
        self.assertEqual([r["feedback"] for r in results], [{"status": "verification_error"}]*2)
        execute.assert_called_once()

    def test_sample_concurrency_bound_is_checked_before_docker(self):
        for limit in (0, 5, True, "4"):
            with mock.patch.object(v.runner.subprocess, "run") as docker, self.assertRaises(ValueError):
                v.runner.execution_contract(max_samples=limit)
            docker.assert_not_called()

    def test_runner_dispatches_declared_concurrency_and_metadata(self):
        with mock.patch.object(v.runner, "execution_contract", return_value={**CONTRACT, "max_samples": 1}), \
             mock.patch.object(v.runner, "inspect_eval", return_value=[]) as execute:
            v.runner.run([v.request(task(), "a"*64, b"print(input())")], "unused",
                         check_answers=True, max_samples=1, concurrent_inference=True)
        self.assertEqual(execute.call_args.kwargs["max_samples"], 1)
        self.assertEqual(execute.call_args.kwargs["max_sandboxes"], 4)
        self.assertTrue(execute.call_args.args[0].metadata["concurrent_inference"])


if __name__ == "__main__":
    unittest.main()
