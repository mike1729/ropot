"""Live attack helpers: audited inputs, persisted accounting and the guarded worker."""
from copy import deepcopy
from hashlib import sha256
import importlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from tests.test_benign_acquisition import Fixture as BenignFixture, b
from tests import test_benign_loop as initial
from tests.test_v2_attacks import ORIGINAL, a, v


class BenignInputTests(BenignFixture):
    def setUp(self):
        super().setUp()
        for source, family in (("actor", "qwen3_5"), ("synthetic", "gemma4")):
            runtime = self.runtimes[source]
            model = Path(runtime["model_path"])
            b.generation.save(model / "config.json", dict(model_type=family, quantization=dict(bits=8)))
            manifest = b.read(runtime["model_manifest"])
            for row in manifest["files"]:
                row["sha256"] = b.generation.file_hash(model / row["path"])
            b.generation.save(Path(runtime["model_manifest"]), manifest)
            digest = b.generation.file_hash(runtime["model_manifest"])
            b.MODEL_HASHES[source] = digest
            proof = b.read(runtime["qualification_file"])
            proof["model_manifest_sha256"] = digest
            b.generation.save(Path(runtime["qualification_file"]), proof)
        self.contract = BenignFixture.prepare(self)
        self.directory = (self.root / "benign/acquisition").resolve()
        self.directory.mkdir(parents=True)
        b.generation.save(self.directory / "contract.json", self.contract)
        self.report = dict(status="completed", sources={},
            contract_sha256=b.generation.file_hash(self.directory / "contract.json"))
        for source in b.SOURCES:
            directory = self.directory / source
            directory.mkdir()
            attempts, audits = [], []
            for position, entry in enumerate(self.contract["entries"]):
                binding = entry["binding"]
                path = directory / f"program-{position:04d}.py"
                path.write_bytes(ORIGINAL)
                candidate = b.study_tasks.candidate_identity(
                    binding["split"], binding["id"], binding["sha256"], source, ORIGINAL)
                attempts.append(dict(position=position, state="passed", program=str(path), candidate=candidate))
                audits.append(dict(position=position, program=str(path), verdict=dict(status="passed",
                    key=dict(task_sha256=binding["sha256"], program_sha256=candidate["program_sha256"],
                             case_view_sha256=a._case_view_hash(entry["partition"]["audit"]))),
                    cases=[dict(index=c["index"], status="passed") for c in entry["partition"]["audit"]]))
            b.generation.save(directory / "outcome.json", dict(status="completed", attempts=attempts))
            self.report["sources"][source] = dict(audits=audits)
        b.generation.save(self.directory / "report.json", self.report)
        b.generation.save(self.directory.parent / "receipt.json", dict(
            contract_sha256=self.report["contract_sha256"],
            report_sha256=b.generation.file_hash(self.directory / "report.json")))

    def load(self):
        return a._load_benign_entries(self.directory, self.contract, self.report)

    def test_benign_is_stripped_without_changing_audited_files(self):
        path = self.directory / "actor/program-0000.py"
        original = b"# original explanation\n" + ORIGINAL
        path.write_bytes(original)
        outcome_path = path.parent / "outcome.json"
        outcome = b.read(outcome_path)
        candidate = outcome["attempts"][0]["candidate"]
        candidate["program_sha256"] = sha256(original).hexdigest()
        b.generation.save(outcome_path, outcome)
        self.report["sources"]["actor"]["audits"][0]["verdict"]["key"]["program_sha256"] = candidate["program_sha256"]
        benign = self.load()[0]["benigns"]["actor"]
        self.assertEqual(benign["program"], ORIGINAL.decode())
        self.assertEqual(benign["candidate"]["program_sha256"], sha256(ORIGINAL).hexdigest())
        self.assertEqual(benign["audited_candidate"], candidate)
        self.assertEqual(path.read_bytes(), original)

    def test_unbound_comment_only_edit_to_audited_benign_is_rejected(self):
        path = self.directory / "actor/program-0000.py"
        path.write_bytes(b"# unbound edit\n" + path.read_bytes())
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.load()

    def test_failed_private_benign_audit_is_ineligible(self):
        self.report["sources"]["actor"]["audits"][0]["verdict"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.load()

    def test_qwen_panel_refuses_unbound_benign_report(self):
        from scripts import qwen_experiment
        path = self.directory / "report.json"
        path.write_text(path.read_text() + "\n")
        with mock.patch.object(a.generation_loop, "run_child") as child:
            with self.assertRaisesRegex(ValueError, "receipt-bound"):
                qwen_experiment.load_panel(self.directory)
        child.assert_not_called()

    def test_changed_checkpoint_metadata_is_rejected(self):
        runtime = self.contract["runtimes"]["synthetic"]
        Path(runtime["model_path"], "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence"):
            a._bind_benign_runtime("synthetic", runtime)

    def test_historical_repository_files_and_live_data_runtime_files_have_one_policy(self):
        repository = self.root / "repository"
        runtime = deepcopy(self.contract["runtimes"]["actor"])
        proof = runtime["qualification"]
        historical = []
        for directory in ("src", "scripts", "docs", "configs"):
            path = repository / directory / "qualification-input.txt"
            path.parent.mkdir(parents=True)
            path.write_text("historical bytes")
            proof["files"][str(path)] = b.generation.file_hash(path)
            path.write_text("current bytes differ")
            historical.append(str(path.resolve()))
        live = [repository / "data/evidence.json", self.root / "installed-runtime/mlx.py"]
        for path in live:
            path.parent.mkdir(parents=True)
            path.write_text("pinned live bytes")
            proof["files"][str(path)] = b.generation.file_hash(path)
        b.generation.save(Path(runtime["qualification_file"]), proof)
        runtime["qualification_sha256"] = b.generation.file_hash(runtime["qualification_file"])
        with mock.patch.object(a, "ROOT", repository.resolve()):
            files = a._bind_benign_runtime("actor", runtime)
            self.assertFalse(set(historical) & files.keys())
            self.assertEqual(b.read(runtime["qualification_file"]), proof)
            for path in live:
                with self.subTest(path=path):
                    path.write_text("changed live bytes")
                    with self.assertRaisesRegex(ValueError, "evidence"):
                        a._bind_benign_runtime("actor", runtime)
                    path.write_text("pinned live bytes")


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class AccountingTests(initial.LoopFixture):
    def setUp(self):
        super().setUp()
        self.thought_status, self.incomplete = None, False

    def generate(self, *args):
        result = super().generate(*args)
        path = self.root / "run/generation-0000/outcome.json"
        outcome = b.read(path)
        outcome["native"]["prompt_tokens"] = 14
        if self.thought_status is not None:
            outcome["requests"][0].update(status=self.thought_status, answer_text=None, enable_thinking=True)
        if self.incomplete:
            path.write_text('{"status":')
        else:
            b.generation.save(path, outcome)
        return result

    def summary(self):
        self.run_loop()
        report = dict(runs=[dict(source="actor", chain=0, directory=str(self.root / "run"))])
        with mock.patch.object(a, "monotonic", return_value=10):
            a._summarize_acquisition(report, dict(prior_costs=[]), 0)
        return report

    def test_persisted_loop_usage_and_yield_are_reconciled(self):
        report = self.summary()
        self.assertEqual((report["reserved_replies"], report["reserved_output_tokens"]), (2, 32))
        self.assertEqual((report["submitted_replies_lower_bound"], report["output_tokens"]), (2, 4))
        self.assertEqual((report["native_prompt_tokens"], report["pipeline_input_tokens"]), (14, 14))
        self.assertEqual(report["pipeline_output_tokens"], 4)
        self.assertEqual(report["yield"]["actor"]["accepted_positions"], [0, 1])
        self.assertTrue(report["usage_complete"])

    def test_truncation_and_unclosed_thought_are_accounted_separately(self):
        self.thought_status = "length"
        report = self.summary()
        self.assertEqual((report["reserved_replies"], report["output_tokens"]), (2, 4))
        self.assertEqual(report["truncated_submissions"], 1)
        self.assertEqual(report["unclosed_or_unmatched_thoughts"], 1)
        self.assertEqual(report["extraction_failures"], 0)
        self.assertEqual(report["truncation_fraction"], 1/2)

    def test_naturally_completed_unclosed_thought_is_an_extraction_failure(self):
        self.thought_status = "stop"
        report = self.summary()
        self.assertEqual(report["unclosed_or_unmatched_thoughts"], 1)
        self.assertEqual(report["extraction_failures"], 1)
        self.assertEqual(report["truncated_submissions"], 0)

    def test_incomplete_native_json_keeps_usage_unknown_and_reserved_cost_known(self):
        self.incomplete = True
        report = self.summary()
        self.assertEqual(report["runs"][0]["status"], "error")
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["output_tokens"])
        self.assertIsNone(report["pipeline_output_tokens"])
        self.assertIsNone(report["native_prompt_tokens"])
        self.assertEqual(report["reserved_replies"], 2)
        self.assertEqual(report["submitted_replies_lower_bound"], 0)
        self.assertEqual((self.root / "run/generation-0000/outcome.json").read_text(), '{"status":')
        self.verifier.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_memory_stops_use_strict_growth_and_peak_boundaries(self):
        baseline = dict(pressure=1, swap_used_mib=5)
        a.check_resources(dict(pressure=2, swap_used_mib=261, peak_bytes=44*1024**3), baseline)
        for change in (dict(pressure=4), dict(swap_used_mib=261.01), dict(peak_bytes=44*1024**3+1)):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "resource stop"):
                a.check_resources(dict(pressure=1, swap_used_mib=5) | change, baseline)

    def test_worker_guards_before_insertion_and_each_step_and_restores_native_generator(self):
        worker = importlib.import_module("scripts.v2_attack_generation")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "actor-chain-0").mkdir()
            b.generation.save(directory / "resource-baseline.json", dict(pressure=1, swap_used_mib=5))
            spec = dict(output_dir=str(directory / "actor-chain-0/generation-0000"))
            b.generation.save(directory / "round.json", spec)
            native = NS(BatchGenerator=type("Original", (), dict(
                insert=lambda self: "inserted", next=lambda self: "advanced")))
            original = native.BatchGenerator
            mx = NS(get_peak_memory=lambda: 0)
            def generate(**settings):
                generator = native.BatchGenerator()
                self.assertEqual(generator.insert(), "inserted")
                self.assertEqual(generator.next(), "advanced")
                self.assertIsNot(native.BatchGenerator, original)
                return dict(status="success")
            actual_import = importlib.import_module
            def imports(name):
                return native if name == "mlx_lm.generate" else actual_import(name)
            with mock.patch.dict(sys.modules, {"mlx": NS(core=mx), "mlx.core": mx}), \
                 mock.patch.object(worker.importlib, "import_module", side_effect=imports), \
                 mock.patch.object(a, "memory_state", return_value=dict(pressure=1, swap_used_mib=5)), \
                 mock.patch.object(b.generation, "generate_batched", side_effect=generate):
                self.assertEqual(worker.run(directory / "round.json"), 0)
            self.assertIs(native.BatchGenerator, original)
            trace = directory / "actor-chain-0/generation-0000-memory.jsonl"
            self.assertEqual(len(trace.read_text().splitlines()), 3)
