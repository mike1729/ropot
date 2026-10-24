from copy import deepcopy
import importlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from tests import test_generation_journal as journal_fixture, test_verification as fixture


class NativeRecoveryTests(unittest.TestCase):
    limits = journal_fixture.GenerationJournalTests.limits
    create = journal_fixture.GenerationJournalTests.create
    rows = journal_fixture.GenerationJournalTests.rows
    request = journal_fixture.GenerationJournalTests.request

    def setUp(self):
        journal_fixture.GenerationJournalTests.setUp(self)
        self.N = importlib.import_module("src.data-pipeline.native_recovery")
        self.inventory, self.recovered = self.N._inventory, None
        self.data, self.program = fixture.task(4), b"print(input())"
        for target, name, value in ((self.N, "_daemon", "fixture-daemon"),
            (self.N, "_inventory", ("fixture-daemon", [], False)),
            (self.N.verification.runner, "contract", fixture.CONTRACT)):
            patch = mock.patch.object(target, name, return_value=value)
            patch.start(); self.addCleanup(patch.stop)
        for target in (self.N, self.N.verification):
            patch = mock.patch.object(target, "read_eval_log", side_effect=self.read)
            patch.start(); self.addCleanup(patch.stop)

    def read(self, path, **kwargs):
        if Path(path).name == "recovered.eval":
            if self.recovered is None: raise ValueError("Corrupt recovered fixture")
            return self.recovered
        return self.log

    def execute(self, *args, operation_id):
        intent = self.rows(self.campaign.root)[-1]
        self.assertEqual((intent["kind"], intent["operation_id"]), ("native_intent", operation_id))
        programs = [args[2]] if type(args[2]) is bytes else self.N.verification.reference_programs(args[0])
        self.log = fixture.native([self.N.verification.request(args[0], args[1], p) for p in programs])
        self.log.eval.task = operation_id
        for sample in self.log.samples:
            sample.store["container"]["Config"] = {"Labels": {
                "com.docker.compose.project": f"inspect-{operation_id}-iabcdef"}}
        path = Path(args[-1]) / "original.eval"
        path.write_bytes(b"native fixture")
        self.log.location = str(path)
        return {"log": str(path)}

    def pending(self):
        self.campaign, root = self.create()
        def interrupt(*args, **kwargs):
            self.execute(*args, **kwargs)
            raise KeyboardInterrupt
        with mock.patch.object(self.N.verification, "verify", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.N.run_native(self.campaign, self.data, "a" * 64, self.program)
        self.addCleanup(self.campaign.close)
        return self.campaign, Path(self.log.location).parent.parent

    def test_normal_gate_binds_same_three_and_completed_pass_fail_or_error_close(self):
        self.campaign, _ = self.create()
        with self.campaign, mock.patch.object(self.N.verification, "reference_gate", side_effect=self.execute):
            result = self.N.run_native(self.campaign, self.data, "a" * 64)
            self.assertEqual((result["status"], len(result["results"])), ("passed", 3))
            intent = next(e for e in self.campaign.events if e["kind"] == "native_intent")
            self.assertEqual(intent["reference_indices"], self.data["passing_reference_indices"])
        for status in ("passed", "failed", "error"):
            campaign, _ = self.pending()
            if status == "failed":
                self.log.samples[0].store["cases"] = self.log.samples[0].store["cases"][:2]
                self.log.samples[0].store["cases"][0]["stdout"] = "999"
            if status == "error": self.log.samples[0].error = NS(message="post-start failure")
            with mock.patch.object(self.N.verification, "verify") as execute:
                self.assertEqual(self.N.reconcile_native(campaign)[0]["status"], status)
                self.assertEqual(self.N.reconcile_native(campaign), [])
                self.assertEqual(sum(e["kind"] == "native_closed" for e in campaign.events), 1)
                execute.assert_not_called()
            campaign.close()

    def test_missing_buffer_store_or_ambiguous_identity_never_yields_a_verdict(self):
        for damage, reason in (("missing", "ambiguous_log"), ("ambiguous", "ambiguous_log"),
            ("store", "not_derivable"), ("buffer_error", "not_derivable"), ("label", "not_derivable"),
            ("header", "not_derivable"), ("contract", "not_derivable"), ("task", "changed_inputs"),
            ("current_contract", "changed_contract"), ("contract_unavailable", "contract_unavailable")):
            campaign, directory = self.pending()
            if damage == "missing": Path(self.log.location).unlink()
            if damage == "ambiguous": (directory / "logs/second.eval").write_bytes(b"other")
            if damage == "store": self.log.samples[0].store = {}
            if damage == "buffer_error": self.log.samples[0].store, self.log.samples[0].error = {}, NS(message="interrupted")
            if damage == "label": self.log.samples[0].store["container"]["Config"]["Labels"]["com.docker.compose.project"] = "inspect-" + "b" * 12 + "-iabcdef"
            if damage == "header": self.log.eval.task = "b" * 12
            if damage == "contract": self.log.eval.metadata["contract"] = {}
            if damage == "task": (directory / "task.json").write_bytes(b"changed")
            probe = ({"side_effect": OSError("contract probe unavailable")} if damage == "contract_unavailable" else
                     {"return_value": {} if damage == "current_contract" else fixture.CONTRACT})
            with mock.patch.object(self.N.verification.runner, "contract", **probe), self.assertRaises(self.N.NativeBlocked) as blocked:
                self.N.reconcile_native(campaign)
            self.assertEqual((str(blocked.exception), campaign.events[-1]["reason"]), (reason, reason))
            self.assertFalse(any(e["kind"] in ("native_verdict", "native_closed") for e in campaign.events))
            campaign.close()

    def test_missing_log_blocks_two_resumes_and_both_dispatch_paths(self):
        campaign, _ = self.pending()
        root = campaign.root
        Path(self.log.location).unlink(); campaign.close()
        for _ in range(2):
            with self.J.Campaign(root) as reopened, self.assertRaises(self.N.NativeBlocked): self.N.reconcile_native(reopened)
            with self.J.Campaign(root) as reopened, mock.patch.object(self.N.verification, "verify") as native:
                with self.assertRaises(ValueError): self.N.run_native(reopened, self.data, "a" * 64, self.program)
                native.assert_not_called()
            with self.J.Campaign(root) as reopened, mock.patch.object(journal_fixture.G, "generate") as model:
                with self.assertRaises(ValueError): reopened.generate("one", "actor", self.request())
                model.assert_not_called()
        self.assertEqual(sum(e["kind"] == "native_blocked" for e in self.rows(root)), 1)

    def test_stopped_container_live_cli_changed_daemon_and_failed_scan_block_closure(self):
        for inventory in (("fixture-daemon", ["owned-stopped"], False), ("fixture-daemon", [], True),
                          ("different-daemon", [], False), OSError("inventory unavailable")):
            campaign, _ = self.pending()
            with mock.patch.object(self.N, "_inventory", **({"side_effect": inventory} if isinstance(inventory, Exception)
                                                         else {"return_value": inventory})):
                with self.assertRaises(self.N.NativeBlocked): self.N.reconcile_native(campaign)
            self.assertFalse(any(e["kind"] == "native_closed" for e in campaign.events))
            self.assertEqual(len(self.N.reconcile_native(campaign)), 1)
            campaign.close()

    def test_recorded_verdict_closes_without_rederiving_after_contract_changes(self):
        for probe in ({"return_value": {}}, {"side_effect": OSError("contract unavailable")}):
            campaign, _ = self.pending()
            with mock.patch.object(self.N, "_inventory", return_value=("fixture-daemon", ["owned"], False)):
                with self.assertRaises(self.N.NativeBlocked): self.N.reconcile_native(campaign)
            previous = next(e for e in campaign.events if e["kind"] == "native_verdict")
            with mock.patch.object(self.N.verification.runner, "contract", **probe) as contract, \
                 mock.patch.object(self.N.verification, "reuse") as reuse, mock.patch.object(self.N, "read_eval_log") as read:
                self.assertEqual(self.N.reconcile_native(campaign), [previous])
                contract.assert_not_called(); reuse.assert_not_called(); read.assert_not_called()
            self.assertEqual(sum(e["kind"] == "native_closed" for e in campaign.events), 1)
            campaign.close()

    def test_recovery_file_survives_lost_event_and_corrupt_existing_file_is_preserved(self):
        campaign, directory = self.pending()
        self.log.status = "started"
        def recover(original, **kwargs):
            self.assertEqual({k: kwargs[k] for k in ("overwrite", "cleanup", "incomplete_action")},
                             {"overwrite": False, "cleanup": False, "incomplete_action": "error"})
            Path(kwargs["output"]).write_bytes(b"recovered fixture")
            self.recovered = deepcopy(self.log); self.recovered.status = "success"
        append = campaign._append
        def fail(kind, **payload):
            if kind == "native_verdict": raise OSError("lost completion event")
            return append(kind, **payload)
        with mock.patch.object(self.N, "recover_eval_log", side_effect=recover) as recovery:
            with mock.patch.object(campaign, "_append", side_effect=fail), self.assertRaises(OSError): self.N.reconcile_native(campaign)
            self.assertEqual(self.N.reconcile_native(campaign)[0]["status"], "passed")
            self.assertEqual(recovery.call_count, 1)
        campaign.close()
        campaign, directory = self.pending()
        self.log.status, self.recovered = "started", None
        (directory / "recovered.eval").write_bytes(b"corrupt")
        with mock.patch.object(self.N, "recover_eval_log") as recovery, self.assertRaises(self.N.NativeBlocked):
            self.N.reconcile_native(campaign)
        recovery.assert_not_called()
        self.assertEqual((directory / "recovered.eval").read_bytes(), b"corrupt")

    def test_inventory_matches_exact_labels_and_all_current_user_cli_arguments(self):
        opid, project = "a" * 12, "inspect-" + "a" * 12 + "-iabcdef"
        for argument, expected in ((project, True), ("--project-name=" + project, True), ("-p" + project, True),
                                   (project + "x", False), ("prefix/" + project, False)):
            process = mock.Mock(); process.uids.return_value = NS(real=os.getuid(), effective=os.geteuid()); process.cmdline.return_value = ["plugin", argument]
            with mock.patch.object(self.N.psutil, "process_iter", return_value=[process]), \
                 mock.patch.object(self.N, "_docker", return_value="\n".join([project, project + "x", "prefix" + project])) as docker:
                self.assertEqual(self.inventory(opid), ("fixture-daemon", [project], expected))
                self.assertIn("-a", docker.call_args.args)
        with mock.patch.object(self.N.psutil, "process_iter", return_value=[process]), mock.patch.object(self.N, "_docker", return_value=""):
            process.cmdline.side_effect = self.N.psutil.AccessDenied(pid=1)
            with self.assertRaises(self.N.psutil.AccessDenied): self.inventory(opid)
            process.uids.return_value.real, process.uids.return_value.effective = os.getuid() + 1, os.geteuid() + 1
            self.assertEqual(self.inventory(opid), ("fixture-daemon", [], False))

    def test_effective_owner_access_denial_and_vanished_process(self):
        process = mock.Mock()
        process.uids.return_value = NS(real=os.getuid(), effective=0 if os.geteuid() else 1)
        process.cmdline.side_effect = self.N.psutil.AccessDenied(pid=1)
        with mock.patch.object(self.N.psutil, "process_iter", return_value=[process]), mock.patch.object(self.N, "_docker", return_value=""):
            self.assertEqual(self.inventory("a" * 12), ("fixture-daemon", [], False))
            process.uids.return_value.effective = os.geteuid()
            with self.assertRaises(self.N.psutil.AccessDenied): self.inventory("a" * 12)
            process.uids.side_effect = self.N.psutil.NoSuchProcess(pid=1)
            self.assertEqual(self.inventory("a" * 12), ("fixture-daemon", [], False))

@unittest.skipUnless(os.environ.get("ROPOT_INSPECT_TESTS") == "1", "set ROPOT_INSPECT_TESTS=1 for real Docker closure")
class NativeRecoveryDockerTests(unittest.TestCase):
    def test_fixture_header_project_mapping_and_actual_normal_closure(self):
        native = importlib.import_module("src.data-pipeline.native_recovery")
        journal = importlib.import_module("src.data-pipeline.generation_journal")
        root = Path(__file__).resolve().parents[1] / "data/03_execution_checks"
        root.mkdir(exist_ok=True)
        campaign_root = Path(tempfile.mkdtemp(prefix="native-closure-", dir=root)) / "campaign"
        with journal.Campaign.create(campaign_root, journal_fixture.GenerationJournalTests.limits(self)) as campaign:
            result = native.run_native(campaign, fixture.task(4), "a" * 64, b"print(input())")
            self.assertEqual(result["status"], "passed")
            log = native.read_eval_log(str(campaign.root / result["log"]))
            self.assertEqual(log.eval.task, result["operation_id"])
            self.assertRegex(log.samples[0].store["container"]["Config"]["Labels"]["com.docker.compose.project"],
                             rf"^inspect-{result['operation_id']}-i[a-z0-9]{{6}}$")
            self.assertEqual(sum(e["kind"] == "native_closed" for e in campaign.events), 1)


class NativeManualResolutionTests(unittest.TestCase):
    limits, rows, request = NativeRecoveryTests.limits, NativeRecoveryTests.rows, NativeRecoveryTests.request
    setUp, read, execute, pending = NativeRecoveryTests.setUp, NativeRecoveryTests.read, NativeRecoveryTests.execute, NativeRecoveryTests.pending
    attestation = {"operator": "Unit test operator", "quiescence_evidence": "Fixture daemon and all matching resources independently checked"}

    def create(self):
        campaign, root = journal_fixture.GenerationJournalTests.create(self)
        campaign.generate("one", "actor", self.request())
        return campaign, root

    def unknown(self):
        campaign, directory = self.pending(); Path(self.log.location).unlink()
        operation_id = next(e["operation_id"] for e in campaign.events if e["kind"] == "native_intent")
        return campaign, operation_id

    def test_attestation_unblocks_without_correctness_or_refund_and_is_idempotent(self):
        campaign, operation_id = self.unknown(); accounting = campaign._accounting()
        self.assertGreater(accounting[1]["actor"]["replies"], 0)
        closed = self.N.resolve_native(campaign, operation_id, attestation=self.attestation)
        self.assertEqual((closed["mode"], closed["attestation"]), ("operator_attestation", self.attestation))
        self.assertEqual(closed["attestation_sha256"], self.N.sha256(self.N.encoded(self.attestation).encode()).hexdigest())
        self.assertFalse(any(e["kind"] == "native_verdict" for e in campaign.events))
        self.assertEqual(campaign._accounting(), accounting); campaign.require_native_clear()
        count = len(campaign.events)
        with mock.patch.object(self.N, "_inventory", side_effect=AssertionError("must not inspect")) as inventory, \
             mock.patch.object(self.N, "_derive", side_effect=AssertionError("must not derive")) as derive:
            self.assertEqual(self.N.resolve_native(campaign, operation_id, attestation=dict(self.attestation)), closed)
            inventory.assert_not_called(); derive.assert_not_called()
        self.assertEqual(len(campaign.events), count)
        with self.assertRaises(ValueError): self.N.resolve_native(campaign, operation_id, attestation={**self.attestation, "operator": "Other"})

    def test_invalid_attestations_and_unknown_operation_append_nothing(self):
        campaign, operation_id = self.unknown(); count = len(campaign.events)
        for bad in ({}, {**self.attestation, "extra": 1}, {**self.attestation, "operator": " "},
                    {**self.attestation, "operator": "é" * 129}, {**self.attestation, "quiescence_evidence": "é" * 4097},
                    {**self.attestation, "operator": True}, {**self.attestation, "operator": "\ud800"}):
            with self.assertRaises(ValueError): self.N.resolve_native(campaign, operation_id, attestation=bad)
        with self.assertRaises(ValueError): self.N.resolve_native(campaign, "0" * 12, attestation=self.attestation)
        self.assertEqual(len(campaign.events), count)

    def test_derivable_and_prior_verdicts_require_normal_reconciliation(self):
        campaign, _ = self.pending()
        operation_id = next(e["operation_id"] for e in campaign.events if e["kind"] == "native_intent")
        with self.assertRaisesRegex(ValueError, "derivable"): self.N.resolve_native(campaign, operation_id, attestation=self.attestation)
        self.assertEqual(sum(e["kind"] == "native_verdict" for e in campaign.events), 1)
        self.assertFalse(any(e["kind"] == "native_closed" for e in campaign.events))
        with self.assertRaisesRegex(ValueError, "Recorded verdict"): self.N.resolve_native(campaign, operation_id, attestation=self.attestation)
        self.assertEqual(sum(e["kind"] == "native_verdict" for e in campaign.events), 1)
        self.N.reconcile_native(campaign)
        with self.assertRaises(ValueError): self.N.resolve_native(campaign, operation_id, attestation=self.attestation)

    def test_manual_closure_requires_original_daemon_and_complete_idle_inventory(self):
        for probe in ({"return_value": {}}, {"side_effect": OSError("contract unavailable")}):
            campaign, operation_id = self.unknown()
            with mock.patch.object(self.N.verification.runner, "contract", **probe), self.assertRaises(self.N.NativeBlocked):
                self.N.resolve_native(campaign, operation_id, attestation=self.attestation)
            self.assertFalse(any(e["kind"] == "native_closed" for e in campaign.events)); campaign.close()
        for inventory in (("fixture-daemon", ["owned-stopped"], False), ("fixture-daemon", [], True),
                          ("changed-daemon", [], False), OSError("scan failed")):
            campaign, operation_id = self.unknown(); accounting = campaign._accounting()
            with mock.patch.object(self.N, "_inventory", **({"side_effect": inventory} if isinstance(inventory, Exception) else {"return_value": inventory})):
                with self.assertRaises(self.N.NativeBlocked): self.N.resolve_native(campaign, operation_id, attestation=self.attestation)
            self.assertFalse(any(e["kind"] in ("native_verdict", "native_closed") for e in campaign.events))
            self.assertEqual(campaign._accounting(), accounting); campaign.close()
