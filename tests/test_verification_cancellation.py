"""Fresh, invocation-bound Inspect cancellation evidence without Docker."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from tests.test_verification import v, task, CONTRACT


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class CancellationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(v.runner, "execution_contract", return_value=CONTRACT).start()

    def verify(self):
        return v.verify_many([(task(), "a" * 64, b"print(input())")], self.directory)

    def test_direct_interruption_is_preserved(self):
        error = KeyboardInterrupt("original")
        with mock.patch.object(v.runner, "inspect_eval", side_effect=error), self.assertRaises(KeyboardInterrupt) as caught:
            self.verify()
        self.assertIs(caught.exception, error)

    def test_returned_cancelled_log_and_fresh_empty_return_normalize(self):
        for empty in (False, True):
            captured = {}
            def evaluate(native_task, **kwargs):
                captured["log"] = NS(status="cancelled", location="fresh.eval",
                    eval=NS(metadata=native_task.metadata))
                return [] if empty else [captured["log"]]
            with self.subTest(empty=empty), mock.patch.object(v.runner, "inspect_eval", side_effect=evaluate), \
                 mock.patch.object(v.runner, "list_eval_logs", return_value=["fresh.eval"]) as discover, \
                 mock.patch.object(v.runner, "read_eval_log", side_effect=lambda *a, **kw: captured["log"]) as read, \
                 self.assertRaises(KeyboardInterrupt) as caught:
                self.verify()
            self.assertTrue(caught.exception.cleanup_confirmed)
            self.assertIn("fresh.eval", str(caught.exception))
            if empty:
                discover.assert_called_once_with(str(self.directory))
                read.assert_called_once_with("fresh.eval", header_only=True)
            else:
                discover.assert_not_called()

    def test_empty_return_requires_matching_cancelled_contract(self):
        for mismatch in ("no_log", "status", "contract"):
            def evaluate(native_task, **kwargs):
                log = NS(status="cancelled", location="fresh.eval",
                    eval=NS(metadata=native_task.metadata.copy()))
                if mismatch == "status":
                    log.status = "error"
                elif mismatch == "contract":
                    log.eval.metadata["changed"] = True
                read.return_value = log
                return []
            with self.subTest(mismatch=mismatch), mock.patch.object(v.runner, "inspect_eval", side_effect=evaluate), \
                 mock.patch.object(v.runner, "list_eval_logs", return_value=[] if mismatch == "no_log" else ["fresh.eval"]), \
                 mock.patch.object(v.runner, "read_eval_log") as read, \
                 self.assertRaisesRegex(RuntimeError, "no logs without matching cancellation"):
                self.verify()

    def test_prepopulated_destination_cannot_supply_prior_identical_cancellation(self):
        (self.directory / "old-identical-cancelled.eval").touch()
        with mock.patch.object(v.runner, "inspect_eval") as execute, self.assertRaisesRegex(ValueError, "empty"):
            self.verify()
        execute.assert_not_called()
        v.runner.execution_contract.assert_not_called()

    def test_verifier_refuses_unexplained_empty_runner_result(self):
        with mock.patch.object(v.runner, "run", return_value=[]), self.assertRaisesRegex(RuntimeError, "exactly one"):
            self.verify()
