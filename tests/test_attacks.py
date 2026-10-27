"""Stage 05 combined checker fixtures; no model, Docker or publication calls."""
import base64
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

try:
    a = importlib.import_module("src.data-pipeline.stage05.attacks")
    v = importlib.import_module("src.data-pipeline.stage03.verification")
    runner = importlib.import_module("src.data-pipeline.stage03.inspect_runner")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    a = v = runner = None


CONTRACT = {"fixture": "attack-contract", "max_samples": 4}


def candidate(identifier=7):
    task = dict(question=f"Print integer {identifier}.",
                cases=[dict(index=3, input="public\n", answer=str(identifier), visible=True)])
    digest = sha256(f"task-{identifier}".encode()).hexdigest()
    return task, digest, f"print({identifier + 1})\n".encode()


def pair(item, answer="7", stdin=" \tPRIVATE_INPUT\r\n"):
    _, digest, _ = item
    program = b"print(7)\n"
    return dict(candidate=dict(split="test", id=7, source="actor", task_sha256=digest,
                               program_sha256=sha256(program).hexdigest()),
                program=program.decode(), stdin=stdin, answer=answer,
                provenance=dict(approval="PRIVATE_APPROVAL"))


def suite(item, status="passed"):
    _, digest, program = item
    return dict(status=status, key=dict(task_sha256=digest, program_sha256=sha256(program).hexdigest(),
                                      contract_sha256="suite-contract"),
                log="suite.eval", sample_id=0,
                feedback=dict(status=status, visible=[], hidden="passed" if status == "passed" else "failed"))


@unittest.skipIf(a is None, "requires requirements-execution.txt")
class AttackCheckerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "verification-0001"
        self.change = lambda log: None
        self.output = "8\n"
        self.execute = mock.patch.object(runner, "run", side_effect=self.native).start()
        self.verify = mock.patch.object(v, "verify_many", side_effect=lambda items, *args, **kwargs:
                                        [suite(item) for item in items]).start()
        mock.patch.object(runner, "execution_contract", side_effect=lambda **kwargs:
                          {**CONTRACT, **kwargs}).start()
        self.addCleanup(mock.patch.stopall)

    def native(self, items, directory, **kwargs):
        self.items = deepcopy(items)
        self.log = NS(status="success", error=None, location="witness.eval",
                      eval=NS(metadata={**CONTRACT, "max_samples": kwargs["max_samples"],
                                        "check_answers": False,
                                        "concurrent_inference": kwargs["concurrent_inference"]}), samples=[])
        for index, item in enumerate(items):
            metadata = dict(program_base64=base64.b64encode(item["program"]).decode(),
                            cases=deepcopy(item["cases"]), verification=deepcopy(item["verification"]))
            result = dict(index=0, status="completed", stdout=self.output, stderr="", returncode=0,
                          elapsed_seconds=0.01)
            self.log.samples.append(NS(id=index, error=None, metadata=metadata, store=dict(cases=[result])))
        self.change(self.log)
        return [self.log]

    def check(self, items=None, pairs=None, **settings):
        items = [candidate()] if items is None else items
        pairs = {item[1]: pair(item) for item in items} if pairs is None else pairs
        return a.check_many(items, self.directory, pairs=pairs, **settings)

    def test_suite_pass_and_valid_wrong_integer_accept_exact_program_and_witness(self):
        item = candidate()
        data = pair(item)
        before = deepcopy(data)
        rows = self.check([item], {item[1]: data}, max_samples=2, concurrent_inference=True)
        self.assertEqual(rows[0]["status"], "passed")
        self.assertEqual(rows[0]["suite"], suite(item))
        self.assertEqual(data, before)
        self.verify.assert_called_once_with([item], Path(str(self.directory) + "-suite"),
                                            max_samples=2, concurrent_inference=True)
        self.assertEqual(self.execute.call_args.kwargs,
                         dict(check_answers=False, max_samples=2, concurrent_inference=True))
        request = self.items[0]
        self.assertEqual(request["program"], item[2])
        self.assertEqual(request["cases"], [dict(index=0, input=data["stdin"])])
        self.assertEqual(rows[0]["key"]["stdin_sha256"], sha256(data["stdin"].encode()).hexdigest())
        self.assertEqual(rows[0]["witness"]["cases"], self.log.samples[0].store["cases"])
        self.assertEqual(rows[0]["witness"]["observed"], "8")
        self.assertEqual(rows[0]["witness"]["log"], "witness.eval")
        self.assertEqual(json.loads(Path(str(self.directory) + "-suite.json").read_text()), [suite(item)])

    def test_suite_failures_skip_witness_and_keep_public_feedback(self):
        item = candidate()
        expected = suite(item, "failed")
        self.verify.side_effect = None
        self.verify.return_value = [expected]
        row = self.check()[0]
        self.assertEqual(row, expected)
        self.execute.assert_not_called()

    def test_suite_infrastructure_error_stops_before_witness_dispatch(self):
        self.verify.side_effect = None
        self.verify.return_value = [suite(candidate(), "error")]
        with self.assertRaises((ValueError, RuntimeError)):
            self.check()
        self.execute.assert_not_called()

    def test_suite_verdict_count_and_candidate_binding_checked_before_witness(self):
        item = candidate()
        wrong = suite(item)
        wrong["key"]["program_sha256"] = "b" * 64
        self.verify.side_effect = None
        for results in ([], [suite(item), suite(item)], [wrong]):
            self.verify.return_value = results
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.check([item])
        self.execute.assert_not_called()

    def test_shrinking_suite_passes_follow_task_digest_and_native_sample_id(self):
        items = [candidate(i) for i in (9, 7, 11)]
        pairs = {item[1]: pair(item, answer=str(i), stdin=f"witness-{i}\n") for i, item in zip((9, 7, 11), items)}
        self.verify.side_effect = lambda data, *args, **kwargs: [suite(item, "failed" if i == 1 else "passed")
                                                                for i, item in enumerate(data)]
        self.change = lambda log: log.samples.reverse()
        rows = self.check(items, pairs)
        self.assertEqual([row["status"] for row in rows], ["passed", "failed", "passed"])
        self.assertEqual([item["cases"][0]["input"] for item in self.items], ["witness-9\n", "witness-11\n"])
        self.assertEqual([item["program"] for item in self.items], [items[0][2], items[2][2]])

    def test_zero_spelling_is_equal_both_ways_without_converting_large_integers(self):
        item = candidate()
        for expected, output in (("0", "-0\n"), ("-0", "0\n"), ("-0", " -0 \t"), ("7", " \n7\t")):
            self.output = output
            with self.subTest(expected=expected, output=output):
                self.assertEqual(self.check([item], {item[1]: pair(item, answer=expected)})[0]["status"], "failed")
        self.output = "9" * 5000 + "\n"
        self.assertEqual(self.check([item], {item[1]: pair(item, answer="9" * 5000)})[0]["status"], "failed")
        self.assertEqual(self.check([item], {item[1]: pair(item, answer="8" * 5000)})[0]["status"], "passed")

    def test_every_completed_witness_rejection_has_identical_private_value_free_feedback(self):
        projections = []
        changes = [dict(stdout=text) for text in ("7", "00", "+8", "8 9", "answer: 8", "", "8.0", "\ufffd")]
        changes += [dict(status=status, stdout=None, stderr=None, returncode=None,
                         diagnostic="PRIVATE_CAPTURE") for status in
                    ("timeout", "output_limit", "stdout_limit", "stderr_limit", "invalid_stdout", "nonzero_exit")]
        changes += [dict(returncode=value, stdout="8") for value in (None, 2)]
        changes += [dict(stdout="8" * runner.STDOUT_LIMIT_BYTES),
                    dict(stderr="e" * runner.STDERR_LIMIT_BYTES)]
        for change in changes:
            self.change = lambda log: log.samples[0].store["cases"][0].update(change)
            with self.subTest(change=list(change)):
                row = self.check()[0]
                self.assertEqual(row["status"], "failed")
                projections.append(row["feedback"])
        self.assertEqual(projections, [dict(message=a.WITNESS_FAILURE)] * len(changes))
        self.assertNotIn("PRIVATE", json.dumps(projections))
        self.assertNotIn("7", json.dumps(projections))

    def test_bad_native_evidence_cannot_accept(self):
        changes = [lambda log: setattr(log, "status", "cancelled"),
                   lambda log: setattr(log, "error", NS(message="PRIVATE_INFRASTRUCTURE")),
                   lambda log: log.eval.metadata.update(changed=True),
                   lambda log: setattr(log.eval, "metadata", None),
                   lambda log: log.samples.clear(),
                   lambda log: log.samples.append(deepcopy(log.samples[0])),
                   lambda log: setattr(log.samples[0], "id", True),
                   lambda log: setattr(log.samples[0], "id", 1),
                   lambda log: setattr(log.samples[0], "id", -1),
                   lambda log: setattr(log.samples[0], "id", "0"),
                   lambda log: setattr(log.samples[0], "error", NS(message="PRIVATE_INFRASTRUCTURE")),
                   lambda log: setattr(log.samples[0], "metadata", None),
                   lambda log: log.samples[0].metadata.update(program_base64="changed"),
                   lambda log: log.samples[0].metadata["cases"][0].update(input="changed"),
                   lambda log: log.samples[0].metadata["verification"].update(changed=True),
                   lambda log: log.samples[0].store.pop("cases"),
                   lambda log: log.samples[0].store["cases"].clear(),
                   lambda log: log.samples[0].store["cases"].append(deepcopy(log.samples[0].store["cases"][0])),
                   lambda log: log.samples[0].store["cases"][0].update(index=1)]
        for change in changes:
            self.change = change
            with self.subTest(change=change):
                try:
                    rows = self.check()
                except (ValueError, RuntimeError):
                    pass
                else:
                    self.assertEqual([row["status"] for row in rows], ["error"])

    def test_missing_or_multiple_native_logs_stop_without_retry(self):
        for logs in ([], [NS(), NS()]):
            self.execute.reset_mock()
            self.execute.side_effect = None
            self.execute.return_value = logs
            with self.subTest(count=len(logs)), self.assertRaises(ValueError):
                self.check()
            self.execute.assert_called_once()

    def test_witness_interruption_preserves_suite_evidence_and_propagates_without_retry(self):
        item = candidate()
        for error in (KeyboardInterrupt("native cleanup complete"), OSError("setup failed")):
            self.execute.reset_mock()
            self.execute.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                self.check([item])
            self.execute.assert_called_once()
            saved = json.loads(Path(str(self.directory) + "-suite.json").read_text())
            self.assertEqual(saved, [suite(item)])


if __name__ == "__main__":
    unittest.main()
