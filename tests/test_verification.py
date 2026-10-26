import asyncio
import base64
from copy import deepcopy
import importlib
import json
import os
from types import SimpleNamespace as NS
import unittest
from unittest import mock

try:
    v = importlib.import_module("src.data-pipeline.stage03.verification")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    v = None


def task(count=4):
    return {"cases": [{"index": 100 - i, "input": f"{i}\n", "answer": str(i), "visible": i < 2}
                      for i in range(count)]}


CONTRACT = {"fixture": "contract"}
METADATA = {**CONTRACT, "concurrent_inference": False, "check_answers": True}


def native(data, program=b"print(input())"):
    item = v.request(data, "a" * 64, program)
    metadata = {"program_base64": base64.b64encode(program).decode(),
                "cases": item["cases"], "verification": item["verification"]}
    rows = [{"index": c["index"], "status": "completed", "stdout": c["answer"] + "\n", "stderr": "",
             "returncode": 0, "elapsed_seconds": 0.01} for c in item["cases"]]
    return NS(status="success", error=None, location="fixture.eval", eval=NS(metadata=deepcopy(METADATA)),
              samples=[NS(id=0, error=None, metadata=metadata, store={"cases": rows})])


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class VerificationTests(unittest.TestCase):
    def test_publication_order_complete_coverage_and_source_independent_explicit_reuse(self):
        data, program = task(25), b"print(input())"
        log = native(data, program)
        with mock.patch.object(v.runner, "run", return_value=[log]) as execute, \
             mock.patch.object(v.runner, "execution_contract", return_value=CONTRACT), \
             mock.patch.object(v, "read_eval_log", return_value=log):
            result = v.verify(data, "a" * 64, program, "unused")
            self.assertEqual(v.reuse("fixture.eval", data, "a" * 64, program), result)
        self.assertEqual(execute.call_count, 1)
        self.assertTrue(execute.call_args.kwargs["check_answers"])
        self.assertEqual(execute.call_args.args[0][0]["cases"], data["cases"])
        self.assertEqual(result["status"], "passed")
        identity = importlib.import_module("src.data-pipeline.stage03.study_tasks").candidate_identity
        for source in ("actor", "synthetic"):
            self.assertEqual(identity("test", 7, "a" * 64, source, program)["program_sha256"],
                             result["key"]["program_sha256"])

    def test_checked_solver_prepares_once_and_stops_on_visible_or_hidden_wrong_answer(self):
        for failed_at in (0, 2):
            log, stored = native(task()), {}
            rows = log.samples[0].store["cases"]
            rows[failed_at]["stdout"] = "999"
            state = NS(metadata=log.samples[0].metadata, store=NS(set=lambda k, value: stored.update({k: value})))
            with mock.patch.object(v.runner, "prepare_program", new_callable=mock.AsyncMock) as prepare, \
                 mock.patch.object(v.runner, "run_case", mock.AsyncMock(side_effect=rows)) as execute:
                asyncio.run(v.runner.execute_programs(True)(state, None))
            prepare.assert_awaited_once()
            self.assertEqual(execute.await_args_list, [mock.call(c["input"]) for c in task()["cases"][:failed_at + 1]])
            self.assertEqual(stored["cases"][-1]["status"], "completed")  # captures stay raw

    def test_checked_case_order_and_answers_rejected_before_docker(self):
        for change in (lambda d: d["cases"].reverse(), lambda d: d["cases"][0].update(answer=" 0 "),
                       lambda d: d["cases"][0].update(visible=1), lambda d: d.update(cases=[])):
            data = task()
            change(data)
            with mock.patch.object(v.runner.subprocess, "run") as docker, self.assertRaises(ValueError):
                v.runner.run([v.request(data, "a" * 64, b"pass")], "unused", check_answers=True)
            docker.assert_not_called()

    def test_comparison_and_capture_limits_apply_during_replay(self):
        raw = native(task()).samples[0].store["cases"][0]
        for changes, status in (({"stdout": " \n0\t"}, "passed"), ({"stdout": "00"}, "wrong_answer"),
                                ({"stdout": "0 0"}, "wrong_answer"), ({"returncode": 3}, "nonzero_exit"),
                                ({"returncode": None}, "nonzero_exit"),
                                ({"stdout": "\ufffd"}, "invalid_stdout"),
                                ({"stdout": "x" * v.runner.STDOUT_LIMIT_BYTES}, "stdout_limit"),
                                ({"stderr": "x" * v.runner.STDERR_LIMIT_BYTES}, "stderr_limit")):
            self.assertEqual(v.runner.case_status({**raw, **changes}, "0"), status)

    def test_hidden_failures_have_identical_public_projection(self):
        projections = []
        for failed_at, status in ((2, "completed"), (3, "timeout"), (2, "nonzero_exit"), (3, "output_limit")):
            data = task()
            for case in data["cases"][2:]:
                case.update(index=case["index"] + 931824, input="HIDDEN_INPUT", answer="872919")
            log = native(data)
            rows = log.samples[0].store["cases"][:failed_at + 1]
            rows[-1].update(status=status, stdout="PRIVATE_STDOUT", stderr="PRIVATE_STDERR", diagnostic="PRIVATE_DETAIL")
            log.samples[0].store["cases"] = rows
            projections.append(v.derive(log, v.request(data, "a" * 64, b"print(input())"), METADATA)["feedback"])
        self.assertTrue(all(p == projections[0] for p in projections))
        self.assertEqual(projections[0]["hidden"], "failed")
        self.assertNotIn("PRIVATE", json.dumps(projections))
        for secret in ("HIDDEN_INPUT", "872919", "9319"):
            self.assertNotIn(secret, json.dumps(projections))
        self.assertNotIn("elapsed_seconds", json.dumps(projections))

    def test_reuse_rejects_bad_evidence_without_dispatch(self):
        mutations = [lambda l: setattr(l, "status", "cancelled"), lambda l: setattr(l, "status", "error"),
            lambda l: l.eval.metadata.update(changed=True), lambda l: l.samples.clear(),
            lambda l: l.samples.append(l.samples[0]), lambda l: setattr(l.samples[0], "id", 1),
            lambda l: setattr(l.samples[0], "error", NS(message="private infrastructure")),
            lambda l: l.samples[0].metadata["cases"][0].update(answer="999"),
            lambda l: l.samples[0].store["cases"].pop(), lambda l: l.samples[0].store["cases"].reverse(),
            lambda l: l.samples[0].store["cases"][0].update(stdout="999"),
            lambda l: l.samples[0].store["cases"][0].update(status="passed")]
        with mock.patch.object(v.runner, "run") as execute, \
             mock.patch.object(v.runner, "execution_contract", return_value=CONTRACT):
            for mutate in mutations:
                log = native(task())
                mutate(log)
                with self.subTest(mutation=mutate), mock.patch.object(v, "read_eval_log", return_value=log), \
                     self.assertRaises(ValueError):
                    v.reuse("fixture.eval", task(), "a" * 64, b"print(input())")
            for digest, program in (("b" * 64, b"print(input())"), ("a" * 64, b"print(input())\n")):
                with mock.patch.object(v, "read_eval_log", return_value=native(task())), self.assertRaises(ValueError):
                    v.reuse("fixture.eval", task(), digest, program)
        execute.assert_not_called()

    def test_visible_failure_skips_hidden_and_infrastructure_has_generic_feedback(self):
        log = native(task())
        log.samples[0].store["cases"] = log.samples[0].store["cases"][:1]
        log.samples[0].store["cases"][0]["stdout"] = "999"
        with mock.patch.object(v.runner, "run", return_value=[log]):
            result = v.verify(task(), "a" * 64, b"print(input())", "unused")
            self.assertEqual((result["status"], result["feedback"]["hidden"]), ("failed", "not_run"))
            log.status, log.samples[0].error = "error", NS(message="PRIVATE_INFRASTRUCTURE")
            self.assertEqual(v.verify(task(), "a" * 64, b"print(input())", "unused")["feedback"],
                             {"status": "verification_error"})


@unittest.skipUnless(v is not None and os.environ.get("ROPOT_INSPECT_TESTS") == "1", "set ROPOT_INSPECT_TESTS=1")
class VerificationDockerTests(unittest.TestCase):
    def test_native_verification_reuse_feedback_and_fresh_retry(self):
        support = importlib.import_module("tests.test_inspect_runner").InspectDockerTests()
        directory, containers = support.log_directory("verification-"), []
        original = v.runner.prepare_program
        async def track(program):
            containers.append((await v.runner.sandbox().connection()).container)
            await original(program)
        with mock.patch.object(v.runner, "prepare_program", side_effect=track):
            program = b"print(input())"
            result = v.verify(task(), "a" * 64, program, os.path.join(directory, "initial"))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(v.reuse(result["log"], task(), "a" * 64, program), result)
            failed = v.verify(task(), "a" * 64, b"print(999)", os.path.join(directory, "wrong"))
            self.assertEqual(failed["feedback"]["hidden"], "not_run")
            program = b"import os\nn=input(); assert not os.path.exists('hidden-marker')\nif n=='2': open('hidden-marker','w').close()\nprint(999 if n=='2' else n)"
            for i in range(2):
                failed = v.verify(task(), "a" * 64, program, os.path.join(directory, f"fresh-{i}"))
                self.assertEqual(failed["feedback"]["hidden"], "failed")
                self.assertTrue(all(c["status"] == "passed" for c in failed["feedback"]["visible"]))
        self.assertEqual(len(set(containers)), 4)
        support.assert_removed(containers)
