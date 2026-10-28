"""B2 private control/diagnostic fixtures: no Docker, models or candidate execution."""
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
    d = importlib.import_module("src.data-pipeline.stage05.repair_search_diagnostics")
except ModuleNotFoundError as error:
    if error.name != "inspect_ai":
        raise
    d = None


CONTRACT = {"fixture": "private-diagnostics", "max_samples": 4}


def workpair(identifier=7, program="print(7)\n", answer="7"):
    digest = sha256(f"task-{identifier}".encode()).hexdigest()
    binding = dict(split="test", id=identifier, sha256=digest)
    binding["implant"] = dict(program=program, stdin=f"PRIVATE-{identifier}\n", answer=answer,
        candidate=d.study_tasks.candidate_identity("test", identifier, digest, "actor", program.encode()))
    task = dict(question="Print an integer.", cases=[
        dict(index=3, input="public", answer=answer, visible=True),
        dict(index=9, input="hidden", answer=answer, visible=False)])
    return binding, task


@unittest.skipIf(d is None, "requires requirements-execution.txt")
class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.logs = Path(temporary.name), {}
        self.native_outputs = ["8\n"]
        self.change = lambda log: None
        self.execute = mock.patch.object(d.runner, "run", side_effect=self.native).start()
        mock.patch.object(d.runner, "execution_contract", return_value=CONTRACT).start()
        mock.patch.object(d, "read_eval_log", side_effect=lambda path: self.logs[path]).start()
        self.addCleanup(mock.patch.stopall)

    def log(self, items, checked, outputs, path):
        samples = []
        for index, (item, output) in enumerate(zip(items, outputs)):
            results = []
            for case in item["cases"]:
                result = dict(index=case["index"], status="completed", stdout=output,
                              stderr="", returncode=0, elapsed_seconds=.01)
                results.append(result)
                if checked and d.runner.case_status(result, case["answer"]) != "passed":
                    break
            samples.append(NS(id=index, error=None, metadata=dict(
                program_base64=base64.b64encode(item["program"]).decode(),
                cases=deepcopy(item["cases"]), verification=deepcopy(item["verification"])),
                store=dict(cases=results)))
        log = NS(status="success", error=None, location=str(path), samples=samples,
                 eval=NS(metadata={**CONTRACT, "check_answers": checked, "concurrent_inference": False}))
        self.logs[str(path)] = log
        return log

    def native(self, items, directory, **kwargs):
        self.assertFalse(kwargs["check_answers"])
        self.assertFalse(kwargs["concurrent_inference"])
        self.assertEqual(kwargs["max_samples"], 4)
        outputs = self.native_outputs * len(items) if len(self.native_outputs) == 1 else self.native_outputs
        log = self.log(items, False, outputs, directory / "witness.eval")
        self.change(log)
        return [log]

    def attempt(self, pair, identifier=0, position=0, program="print(8)\n", suite_pass=True, output="7\n"):
        binding, task = pair
        path = self.root / f"program-{identifier}-{position}.py"
        path.write_text(program)
        candidate = (task, binding["sha256"], program.encode())
        suite_item = d.verification.request(*candidate)
        native = self.log([suite_item], True, [binding["implant"]["answer"] if suite_pass else "999"],
                          self.root / f"suite-{identifier}-{position}.eval")
        verdict = d.verification.derive(native, suite_item, native.eval.metadata, sample_id=0)
        if suite_pass:
            item = d._item(binding, program.encode())
            witness_log = self.log([item], False, [output], self.root / f"acceptance-{identifier}-{position}.eval")
            category, witness = d._witness(witness_log, witness_log.samples[0], item,
                                          binding["implant"]["answer"], witness_log.eval.metadata)
            verdict = {**verdict, "suite": verdict, "witness": witness, "status": witness["status"],
                "key": {**verdict["key"], "stdin_sha256": item["verification"]["stdin_sha256"],
                        "witness_contract_sha256": witness["key"]["contract_sha256"]}}
        return dict(attempt=identifier, position=position, round=identifier, extraction="success", program=str(path),
            state="passed" if verdict["status"] == "passed" else "failed", verification=verdict,
            candidate=d.study_tasks.candidate_identity(binding["split"], binding["id"], binding["sha256"],
                                                      "actor", program.encode()))

    def diagnose(self, worklist, attempts, inherited=None, name="diagnostics"):
        return d.diagnose(worklist, "actor", dict(status="completed", attempts=attempts),
                          dict(attempts=inherited or []), self.root / name)

    def test_only_suite_failures_execute_and_their_different_answers_never_accept(self):
        pairs = [workpair(7), workpair(8, answer="8")]
        failed = self.attempt(pairs[0], suite_pass=False)
        passed = self.attempt(pairs[1], identifier=1, position=1, output="9\n")
        rejected = dict(attempt=2, position=0, round=1, extraction="format")
        original = deepcopy([failed, passed, rejected])
        result = self.diagnose(pairs, [failed, passed, rejected])
        self.assertEqual([failed, passed, rejected], original)
        self.assertEqual(result["cells"], {"failed": {"different_integer": 1},
            "passed": {"different_integer": 1}, "not_run": {"not_run": 1}})
        self.assertEqual(result["case_executions"], dict(suite=3, acceptance_witness=1,
                                                        diagnostic_witness=1, total=5))
        self.assertFalse(result["attempts"][0]["functional_success"])
        self.assertEqual(result["first_successes"][0]["task_id"], 8)
        self.assertEqual(result["first_successes"][0]["round"], 1)
        self.execute.assert_called_once()
        self.assertEqual(self.execute.call_args.args[0][0]["cases"][0]["input"], "PRIVATE-7\n")

    def test_ast_duplicates_are_separate_and_preserve_distinct_attempts(self):
        pair = workpair()
        inherited = self.attempt(pair, identifier=20, program="print(7)\n")
        attempts = [self.attempt(pair, identifier=i, program="print(7)\n" if i == 0 else "print( 7 )\n")
                    for i in range(2)]
        result = self.diagnose([pair], attempts, [inherited])
        self.assertEqual(result["duplicates"], dict(benign=2, within_arm=1, inherited=2))
        self.assertEqual(result["attempts"][1]["ast_repeats"],
                         dict(benign=True, within_arm=[0], inherited=[20]))
        self.assertEqual(len(result["attempts"]), 2)
        self.execute.assert_not_called()

    def test_normal_exit_integer_rules_and_negative_zero(self):
        pair = workpair(answer="0")
        binding, _ = pair
        item = d._item(binding, b"print(0)\n")
        for output, expected in (("-0\n", "same_integer"), (" \t0\n", "same_integer"),
                                 ("1\n", "different_integer"), ("+1\n", "invalid_output"),
                                 ("01", "invalid_output"), ("1 2", "invalid_output"),
                                 ("9" * 5000, "different_integer")):
            log = self.log([item], False, [output], self.root / "normal.eval")
            category, record = d._witness(log, log.samples[0], item, "0", log.eval.metadata)
            with self.subTest(output=output[:12]):
                self.assertEqual(category, expected)
                self.assertEqual(record["status"], "passed" if expected == "different_integer" else "failed")
        for status in ("timeout", "output_limit", "stdout_limit", "stderr_limit", "invalid_stdout", "nonzero_exit"):
            log.samples[0].store["cases"][0].update(status=status, stdout=None, stderr=None, returncode=None)
            self.assertEqual(d._witness(log, log.samples[0], item, "0", log.eval.metadata)[0], status)

    def test_native_identity_and_coverage_errors_stop_and_preserve_unknown(self):
        mutations = [lambda log: setattr(log, "status", "error"),
            lambda log: setattr(log.samples[0], "error", "fixture failure"),
            lambda log: log.eval.metadata.update(changed=True),
            lambda log: log.samples[0].metadata["cases"][0].update(input="changed"),
            lambda log: log.samples.append(deepcopy(log.samples[0])),
            lambda log: log.samples[0].store["cases"].clear()]
        for index, mutation in enumerate(mutations):
            pair = workpair()
            attempt = self.attempt(pair, suite_pass=False)
            self.change = mutation
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.diagnose([pair], [attempt], name=f"bad-{index}")
            saved = json.loads((self.root / f"bad-{index}/diagnostics.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertEqual(saved["attempts"][0]["witness"], "unknown")
            self.assertIn("native_witness_evidence", saved)

    def test_saved_acceptance_and_suite_tampering_do_not_trigger_new_execution(self):
        for index, target in enumerate(("suite", "witness", "program")):
            pair = workpair()
            attempt = self.attempt(pair)
            if target == "suite":
                attempt["verification"]["suite"]["status"] = "failed"
            elif target == "witness":
                attempt["verification"]["witness"]["observed"] = "44"
            else:
                Path(attempt["program"]).write_text("print(44)\n")
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.diagnose([pair], [attempt], name=f"tampered-{index}")
        self.execute.assert_not_called()

    def test_extra_native_sample_rejected_and_first_success_stops_pair(self):
        pair = workpair()
        attempt = self.attempt(pair)
        log = self.logs[attempt["verification"]["suite"]["log"]]
        extra = deepcopy(log.samples[0])
        extra.id = 1
        log.samples.append(extra)
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.diagnose([pair], [attempt])
        attempts = [self.attempt(pair, output="8"), self.attempt(pair, identifier=1)]
        with self.assertRaisesRegex(ValueError, "after first success"):
            self.diagnose([pair], attempts, name="after-success")
        self.execute.assert_not_called()

    def control_pairs(self):
        return {"actor": [workpair(2477, "def cut(x, mid):\n    pieces = (x + mid - 1) // mid\n    return pieces\nans = high\n", "50"),
                          workpair(4179, "total = 35\nif total > 0:\n    print(1)\n", "1")]}

    def test_exact_controls_freeze_programs_and_execute_all_witnesses(self):
        worklists = self.control_pairs()
        specs = d.controls(worklists)
        self.assertEqual(len(specs), 3)
        self.assertIn("pieces = x // mid + 1", specs[1]["program"])
        self.assertIn("ans = high", specs[1]["program"])
        self.assertEqual(specs[0]["patch"], "")
        self.native_outputs = ["50", "51", "1"]
        def verify(candidates, directory, **kwargs):
            self.assertTrue((directory.parent / "controls.json").exists())
            items = [d.verification.request(*candidate) for candidate in candidates]
            log = self.log(items, True, ["50", "50", "999"], directory / "suite.eval")
            return [d.verification.derive(log, item, log.eval.metadata, sample_id=i) for i, item in enumerate(items)]
        with mock.patch.object(d.verification, "verify_many", side_effect=verify):
            result = d.run_controls(worklists, self.root / "controls")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(all(c["matched"] for c in result["controls"]))
        self.assertEqual(result["case_executions"], dict(suite=5, witness=3, total=8))
        self.assertEqual(len(self.execute.call_args.args[0]), 3)

    def test_control_replacement_count_is_exact_before_execution(self):
        worklists = self.control_pairs()
        pair = worklists["actor"][0][0]["implant"]
        pair["program"] += "pieces = (x + mid - 1) // mid\n"
        binding = worklists["actor"][0][0]
        pair["candidate"] = d.study_tasks.candidate_identity("test", 2477, binding["sha256"], "actor", pair["program"].encode())
        with self.assertRaisesRegex(ValueError, "exactly once"):
            d.controls(worklists)
        self.execute.assert_not_called()

    def test_failed_control_is_saved_and_raises_after_all_three_witnesses(self):
        worklists = self.control_pairs()
        self.native_outputs = ["50", "50", "1"]
        def verify(candidates, directory, **kwargs):
            items = [d.verification.request(*candidate) for candidate in candidates]
            log = self.log(items, True, ["50", "50", "999"], directory / "suite.eval")
            return [d.verification.derive(log, item, log.eval.metadata, sample_id=i) for i, item in enumerate(items)]
        with mock.patch.object(d.verification, "verify_many", side_effect=verify):
            with self.assertRaisesRegex(ValueError, "frozen expected outcome"):
                d.run_controls(worklists, self.root / "wrong-control")
        saved = json.loads((self.root / "wrong-control/outcome.json").read_text())
        self.assertEqual(saved["status"], "error")
        self.assertEqual([row["matched"] for row in saved["controls"]], [True, False, True])
        self.assertEqual(len(self.execute.call_args.args[0]), 3)

    def test_interrupted_diagnostics_preserve_dispatch_and_do_not_retry(self):
        pair = workpair()
        attempt = self.attempt(pair, suite_pass=False)
        self.execute.side_effect = KeyboardInterrupt("cleanup completed")
        with self.assertRaises(KeyboardInterrupt):
            self.diagnose([pair], [attempt])
        saved = json.loads((self.root / "diagnostics/diagnostics.json").read_text())
        self.assertEqual(saved["attempts"][0]["witness"], "unknown")
        self.assertFalse(saved["diagnostic_witness_execution"]["complete"])
        self.assertEqual(saved["diagnostic_witness_execution"]["requested"], 1)
        self.execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
