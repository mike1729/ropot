import asyncio
import base64
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

try:
    v = importlib.import_module("src.data-pipeline.verification")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    v = None


def task(count=12):
    return {"split": "train", "id": 19, "cases": [
        {"index": i, "input": f"{i}\n", "answer": str(i), "visible": i % 2 == 0}
        for i in range(count)], "solutions": ["print(input())", "print(int(input()))",
        "import sys; print(sys.stdin.read().strip())", "raise RuntimeError()"],
        "passing_reference_indices": [2, 0, 1]}


CONTRACT = {"image": {"Id": "fixture-image"}}


def native(items):
    samples = [NS(id=i, error=None, metadata={
        "program_base64": base64.b64encode(item["program"]).decode(),
        "cases": deepcopy(item["cases"]), "verification": item["verification"]}, store={
        "runtime": {"system": "Linux", "python": "3.12.9", "utf8_mode": 1},
        "container": {"Image": "fixture-image"}, "cases": [
            {"index": c["index"], "status": "completed", "stdout": c["answer"] + "\n",
             "stderr": "", "returncode": 0} for c in item["cases"]]})
        for i, item in enumerate(items)]
    return NS(status="success", location="fixture.eval", samples=samples,
              eval=NS(metadata={"contract": deepcopy(CONTRACT)}))


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class VerificationTests(unittest.TestCase):
    def test_four_groups_cover_every_case_in_fixed_rank_order(self):
        data = task(20)
        ranked = sorted(data["cases"], key=lambda c: sha256(
            f"ropot-screen-v1|train|19|{c['index']}".encode()).hexdigest())
        selected = {next(c["index"] for c in ranked if c["visible"]),
                    next(c["index"] for c in ranked if not c["visible"])}
        selected.update(c["index"] for c in ranked if c["index"] not in selected
                        and len(selected) < 8)
        expected = [c["index"] for visible in (True, False) for chosen in (True, False)
                    for c in ranked if c["visible"] == visible and (c["index"] in selected) == chosen]
        self.assertEqual([c["index"] for c in v.ordered_cases(data)], expected)
        data["cases"].reverse()
        self.assertEqual([c["index"] for c in v.ordered_cases(data)], expected)
        self.assertEqual(len(set(expected)), 20)

    def test_same_three_references_and_no_substitution(self):
        data = task()
        self.assertEqual(v.reference_programs(data), [data["solutions"][i].encode() for i in (2, 0, 1)])
        for indices in ([0, 1], [0, 0, 1], [True, 1, 2], [-1, 0, 1]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                v.reference_programs({**data, "passing_reference_indices": indices})
        with self.assertRaises(ValueError):
            v.reference_programs({**data, "solutions": ["pass"] * 4})

    def test_solver_finishes_visible_wrong_answer_and_timeout_before_skipping_hidden(self):
        item = v.request(task(6), "a" * 64, b"pass")
        observed = {"Id": "container", "Image": "fixture-image", "Config": {}, "HostConfig": {}}
        environment = mock.Mock(connection=mock.AsyncMock(return_value=NS(container="container")),
            exec=mock.AsyncMock(return_value=NS(success=True, stdout=json.dumps({"system": "Linux"}))))
        stored = {}
        state = NS(metadata=native([item]).samples[0].metadata,
                   store=NS(set=lambda key, value: stored.__setitem__(key, value)))
        results = [{"status": "completed", "stdout": "999", "stderr": "", "returncode": 0},
                   {"status": "timeout", "stdout": None, "stderr": None, "returncode": None},
                   {"status": "completed", "stdout": item["cases"][2]["answer"], "stderr": "", "returncode": 0}]
        with mock.patch.object(v.runner, "sandbox", return_value=environment), \
             mock.patch.object(v.runner, "host_exec", mock.AsyncMock(
                 return_value=NS(success=True, stdout=json.dumps([observed])))), \
             mock.patch.object(v.runner, "run_case", mock.AsyncMock(side_effect=results)) as execute:
            asyncio.run(v.runner.execute_programs()(state, None))
        self.assertEqual([call.args[2] for call in execute.await_args_list],
                         [c["index"] for c in item["cases"] if c["visible"]])
        self.assertEqual([r["status"] for r in stored["cases"]], ["wrong_answer", "timeout", "passed"])

    def test_operation_id_validation_forwarding_and_task_name(self):
        opid, item = "012345abcdef", v.request(task(4), "a" * 64, b"pass")
        for invalid in ("", False, 12, "A" * 12, "a" * 11, "a" * 13, "g" * 12, "a" * 11 + "\n"):
            with mock.patch.object(v.runner.subprocess, "run") as docker, self.assertRaises(ValueError):
                v.runner.run([item], "unused", operation_id=invalid)
            docker.assert_not_called()
        with mock.patch.object(v.runner, "contract", return_value=CONTRACT), \
             mock.patch.object(v.runner.subprocess, "run", return_value=NS(stdout='{"NCPU":1,"MemTotal":1,"ServerVersion":"test"}')), \
             mock.patch.object(v.runner, "inspect_eval") as evaluate:
            for identity, name in ((None, "linux_execution"), (opid, opid)):
                v.runner.run([item], "unused", operation_id=identity)
                self.assertEqual(evaluate.call_args.args[0].name, name)
        with mock.patch.object(v.runner, "contract", return_value=CONTRACT), \
             mock.patch.object(v.runner, "run", side_effect=lambda items, *a, **kw: [native(items)]) as dispatch:
            v.verify(task(4), "a" * 64, b"pass", "unused", operation_id=opid)
            self.assertEqual(v.reference_gate(task(4), "a" * 64, "unused", operation_id=opid)["status"], "passed")
            self.assertEqual([call.kwargs["operation_id"] for call in dispatch.call_args_list], [opid, opid])
        for statuses, expected in ((["passed"] * 3, "passed"), (["passed", "error"], "error"), (["error", "failed"], "failed")):
            self.assertEqual(v.gate_status([{"status": s} for s in statuses]), expected)

    def test_malformed_checked_cases_rejected_before_docker(self):
        for change in (lambda c: c[0].pop("answer"), lambda c: c[0].update(answer=" 0 "),
                       lambda c: c[0].update(visible=1), lambda c: c.reverse()):
            item = v.request(task(4), "a" * 64, b"pass")
            change(item["cases"])
            with self.subTest(change=change), mock.patch.object(v.runner.subprocess, "run") as docker:
                with self.assertRaises(ValueError):
                    v.runner.run([item], "unused")
                docker.assert_not_called()

    def test_native_key_reconstruction_source_independence_and_sample_order(self):
        item = v.request(task(), "a" * 64, b"print(input())")
        log = native([item, item])
        log.samples.reverse()
        results = v.derive(log, [item, item], CONTRACT)
        self.assertTrue(all(r["status"] == "passed" for r in results))
        self.assertEqual(results[0]["key"], {**item["verification"], "contract_sha256": v.digest(CONTRACT)})
        identity = importlib.import_module("src.data-pipeline.study_tasks").candidate_identity
        for source in ("actor", "synthetic"):
            self.assertEqual(identity("train", 19, "a" * 64, source, item["program"])["program_sha256"],
                             results[0]["key"]["program_sha256"])
        log.samples[0].store["cases"][0]["stdout"] = "999\n"
        log.samples[0].store["cases"][0]["status"] = "passed"
        log.samples[0].store["cases"] = log.samples[0].store["cases"][:6]
        self.assertEqual(v.derive(log, [item, item], CONTRACT)[1]["status"], "failed")

    def test_rejects_incomplete_changed_and_infrastructure_evidence(self):
        item = v.request(task(), "a" * 64, b"pass")
        mutations = [lambda l: setattr(l, "status", "cancelled"),
            lambda l: l.samples.clear(), lambda l: l.samples.append(l.samples[0]),
            lambda l: l.samples[0].store["cases"].pop(),
            lambda l: l.samples[0].store["cases"].reverse(),
            lambda l: l.samples[0].metadata["cases"][0].update(answer="999"),
            lambda l: l.samples[0].store["container"].update(Image="another-image"),
            lambda l: l.eval.metadata["contract"].update(changed=True)]
        for change in mutations:
            log = native([item])
            change(log)
            with self.subTest(change=change), self.assertRaises(ValueError):
                v.derive(log, [item], CONTRACT)
        for changed in (v.request(task(), "b" * 64, b"pass"), v.request(task(), "a" * 64, b"pass\n")):
            with self.assertRaises(ValueError):
                v.derive(native([item]), [changed], CONTRACT)
        log = native([item])
        log.status, log.samples[0].error = "error", NS(message="private infrastructure details")
        self.assertEqual(v.derive(log, [item], CONTRACT)[0]["feedback"], {"status": "verification_error"})

    def test_hidden_failures_share_public_projection_without_private_sentinels(self):
        data = task(4)
        for case in data["cases"]:
            if not case["visible"]:
                case.update(index=case["index"] + 938271, input="HIDDEN_INPUT", answer="92743109")
        item = v.request(data, "a" * 64, b"pass")
        projections = []
        for status in ("wrong_answer", "timeout", "nonzero_exit", "output_limit"):
            log = native([item])
            log.samples[0].store["cases"] = log.samples[0].store["cases"][:3]
            log.samples[0].store["cases"][-1].update(status=status, stdout="PRIVATE_STDOUT",
                stderr="PRIVATE_STDERR", diagnostic="PRIVATE_DIAGNOSTIC", returncode=9)
            projections.append(v.derive(log, [item], CONTRACT)[0]["feedback"])
        for count, failed_offset in ((4, 3), (8, 5)):
            varied = task(count)
            for case in varied["cases"]:
                case["visible"] = case["index"] in (0, 2)
            item = v.request(varied, "a" * 64, b"pass")
            log = native([item])
            log.samples[0].store["cases"] = log.samples[0].store["cases"][:failed_offset + 1]
            log.samples[0].store["cases"][-1].update(status="timeout", stdout=None, stderr=None)
            feedback = v.derive(log, [item], CONTRACT)[0]["feedback"]
            self.assertEqual(feedback, projections[0])
        self.assertTrue(all(p == projections[0] for p in projections))
        for secret in ("HIDDEN_INPUT", "92743109", "PRIVATE_", "93827"):
            self.assertNotIn(secret, json.dumps(projections))


@unittest.skipUnless(v is not None and os.environ.get("ROPOT_INSPECT_TESTS") == "1",
                     "set ROPOT_INSPECT_TESTS=1 for real Inspect/Docker verification")
class VerificationDockerTests(unittest.TestCase):
    def test_fixture_gate_public_feedback_and_fresh_retry(self):
        data = task(4)
        root = Path(__file__).resolve().parents[1] / "data/03_execution_checks"
        root.mkdir(exist_ok=True)
        directory = tempfile.mkdtemp(prefix="verification-", dir=root)
        self.assertEqual(v.reference_gate(data, "a" * 64, directory)["status"], "passed")
        failed = v.verify(data, "a" * 64, b"import sys; print(999) if input()=='0' else sys.exit(3)", directory)
        self.assertEqual(failed["feedback"]["hidden"], "not_run")
        self.assertEqual(len(failed["feedback"]["visible"]), 2)
        hidden = v.verify(data, "a" * 64, b"n=int(input()); print(n if n%2==0 else 999)", directory)
        self.assertEqual(hidden["feedback"]["hidden"], "failed")
        self.assertEqual(v.reuse(hidden["log"], data, "a" * 64,
                                [b"n=int(input()); print(n if n%2==0 else 999)"])[0]["key"], hidden["key"])

    def test_same_three_qualified_train_references_on_every_stored_case(self):
        root = Path(__file__).resolve().parents[1]
        artifact = root / "data/01_qualified_pool/tasks/train_1607.json"
        if not artifact.exists():
            self.skipTest("qualified train fixture is unavailable")
        raw = artifact.read_bytes()
        (root / "data/03_execution_checks").mkdir(exist_ok=True)
        directory = tempfile.mkdtemp(prefix="qualified-train-", dir=root / "data/03_execution_checks")
        result = v.reference_gate(json.loads(raw), sha256(raw).hexdigest(), directory)
        self.assertEqual(result["status"], "passed")
