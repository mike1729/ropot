"""Attack acquisition using the real shared loop and shown checker; no model/Docker calls."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from tests.test_benign_acquisition import Fixture as BenignFixture, b
from tests.test_v2_attacks import ORIGINAL, PROGRAM, a, v


class Fixture(BenignFixture):
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
        self.benign_contract = BenignFixture.prepare(self)
        # These deliberately differ from current sources; the benign run remains historical.
        self.benign_contract["sources"] = {p: "f" * 64 for p in self.benign_contract["sources"]}
        self.benign_run_dir = self.root / "benign/acquisition"
        self.benign_run_dir.mkdir(parents=True)
        b.generation.save(self.benign_run_dir / "contract.json", self.benign_contract)
        self.benign_report = dict(status="completed", ready_for_attack_acquisition=True, output_tokens=32,
            native_prompt_tokens=160, elapsed_seconds=1.5, sources={}, usage_complete=True,
            joint_positions=list(range(8)), truncated_replies=0, submitted_replies_lower_bound=16,
            contract_sha256=b.generation.file_hash(self.benign_run_dir / "contract.json"))
        for source in b.SOURCES:
            directory = self.benign_run_dir / source
            directory.mkdir()
            attempts, audits = [], []
            for position, entry in enumerate(self.benign_contract["entries"]):
                binding = entry["binding"]
                path = directory / f"program-{position:04d}.py"
                path.write_bytes(ORIGINAL)
                candidate = b.study_tasks.candidate_identity(binding["split"], binding["id"], binding["sha256"], source, ORIGINAL)
                attempts.append(dict(attempt=position, position=position, state="passed", program=str(path), candidate=candidate))
                log = directory / "native.eval"
                log.write_bytes(b"fixture native evidence")
                audits.append(dict(position=position, program=str(path), verdict=dict(status="passed", log=str(log),
                    key=dict(task_sha256=binding["sha256"], program_sha256=candidate["program_sha256"],
                             case_view_sha256=a._case_view_hash(entry["partition"]["audit"]))),
                    cases=[dict(index=c["index"], status="passed") for c in entry["partition"]["audit"]]))
            b.generation.save(directory / "outcome.json", dict(status="completed", attempts=attempts))
            self.benign_report["sources"][source] = dict(audits=audits)
        b.generation.save(self.benign_run_dir / "report.json", self.benign_report)
        self.receipt = dict(contract_sha256=self.benign_report["contract_sha256"],
            report_sha256=b.generation.file_hash(self.benign_run_dir / "report.json"), study_output_tokens=32,
            setup_output_tokens=10, setup_elapsed_seconds=.5)
        b.generation.save(self.benign_run_dir.parent / "receipt.json", self.receipt)
        self.thinking = self.root / "thinking-receipt.json"
        self.thinking_evidence = self.root / "thinking-evidence.json"
        self.thinking_evidence.write_text('{"status":"cancelled"}')
        b.generation.save(self.thinking, dict(status="cancelled", output_tokens=11, elapsed_seconds=2,
            evidence_sha256={self.thinking_evidence.name: b.generation.file_hash(self.thinking_evidence)}))
        self.prepared, self.output = self.root / "attack-prepared", self.root / "attack-run"
        self.requests, self.native_cases = [], []
        self.failures = set()
        self.all_fail = self.truncate = self.unclosed = self.technical_error = self.interrupt = False
        self.incomplete_native_json = False
        self.memory = mock.patch.object(a, "memory_state", return_value=dict(pressure=1, swap_used_mib=5)).start()

    def prepare_attacks(self, development_reports=()):
        self.digest = a.prepare_contract(self.benign_run_dir, self.prepared,
            development_reports=development_reports, thinking_receipt=self.thinking)
        return b.read(self.prepared / "contract.json")

    def generate_attacks(self, command, current, persist):
        self.assertEqual(Path(command[1]), a.DRIVER)
        spec = b.read(command[-1])
        directory = Path(spec["output_dir"])
        directory.mkdir()
        source, _, chain = directory.parent.name.split("-")
        chain, round_index = int(chain), current["round"]
        rows = []
        for row_index, request in enumerate(spec["requests"]):
            position = (request["seed"] - 41) % 8
            self.requests.append(dict(source=source, chain=chain, round=round_index, position=position,
                                      request=deepcopy(request), spec=deepcopy(spec)))
            fails = self.all_fail or (source, chain, round_index, position) in self.failures
            program = b"# case: 1\n" + ORIGINAL if fails else PROGRAM
            text = "PRIVATE_REASONING\n" + program.decode()
            row = dict(submitted=True, status="stop", text=text, answer_text=program.decode(),
                       enable_thinking=True, generation_tokens=2, tokens=[12, 99], prompt_tokens=11)
            if source == "actor" and chain == round_index == position == 0:
                if self.truncate:
                    row["status"] = "length"
                if self.unclosed:
                    row["answer_text"] = None
            rows.append(row)
            (directory / f"reply-{row_index:04d}.txt").write_text(text)
        b.generation.save(directory / "runtime.json", dict(load_seconds=.01))
        b.generation.save(directory / "outcome.json", dict(status="error" if self.technical_error else "success",
            requests=rows, native=dict(prompt_tokens=10 * len(rows), generation_tokens=2 * len(rows), wall_time=.1)))
        if self.incomplete_native_json:
            (directory / "outcome.json").write_text('{"status":')
            return 1
        if self.interrupt:
            raise KeyboardInterrupt("fixture interrupted after delivery")
        return 1 if self.technical_error else 0

    def verify_attacks(self, candidates, directory, **settings):
        results = []
        for view, task_hash, program in candidates:
            self.assertEqual(task_hash, v.digest(view))
            self.assertEqual(len(view["cases"]), 1)
            case = view["cases"][0]
            self.native_cases.append(deepcopy(case))
            named_inputs = {e["partition"]["menu"][0]["input"] for e in self.benign_contract["entries"]}
            wrong = program == PROGRAM and case["input"] in named_inputs
            capture = dict(status="wrong_answer" if wrong else "passed", stdout="99999\n" if wrong else case["answer"],
                           stderr="", returncode=0, diagnostic=None)
            results.append(dict(status="failed" if wrong else "passed", feedback=dict(visible=[capture]),
                key=dict(task_sha256=task_hash, program_sha256=sha256(program).hexdigest()),
                log=str(Path(directory) / "native.eval"), sample_id=len(results)))
        return results

    def acquire_attacks(self):
        with mock.patch.object(b.generation_loop, "run_child", side_effect=self.generate_attacks), \
             mock.patch.object(v, "verify_many", side_effect=self.verify_attacks):
            return a.acquire(self.prepared, self.output, expected_contract_sha256=self.digest)


class PreparationTests(Fixture):
    def test_freezes_existing_panel_runtime_costs_review_and_source_hashes_without_inference(self):
        with mock.patch.object(b.generation_loop, "run_child") as child, mock.patch.object(b, "partition") as repartition:
            contract = self.prepare_attacks()
        child.assert_not_called()
        repartition.assert_not_called()
        self.assertEqual(self.digest, b.generation.file_hash(self.prepared / "contract.json"))
        self.assertEqual(contract["entries"][0]["partition"], self.benign_contract["entries"][0]["partition"])
        self.assertEqual([e["binding"]["id"] for e in contract["entries"]], list(range(8)))
        self.assertEqual(contract["review_policy"]["reviewers"], ["Claude", "Codex"])
        self.assertEqual(sum(c["output_tokens"] for c in contract["prior_costs"]), 53)
        self.assertEqual(contract["monitor"]["batching"]["completion_batch_size"], 4)
        self.assertEqual(contract["monitor"]["historical_batching"]["completion_batch_size"], 8)
        self.assertNotEqual(self.benign_contract["sources"], contract["sources"])
        self.assertIn(str(a.DRIVER), contract["sources"])
        self.assertEqual(contract["runtimes"]["synthetic"]["model_path"], self.runtimes["synthetic"]["model_path"])

    def test_failed_private_benign_audit_is_ineligible(self):
        self.benign_report["sources"]["actor"]["audits"][0]["verdict"]["status"] = "failed"
        b.generation.save(self.benign_run_dir / "report.json", self.benign_report)
        self.receipt["report_sha256"] = b.generation.file_hash(self.benign_run_dir / "report.json")
        b.generation.save(self.benign_run_dir.parent / "receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "eligibility"):
            self.prepare_attacks()
        self.assertFalse(self.prepared.exists())

    def test_unbound_benign_report_and_changed_checkpoint_metadata_refuse_preparation(self):
        (self.benign_run_dir / "report.json").write_text((self.benign_run_dir / "report.json").read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "receipt-bound"):
            self.prepare_attacks()
        self.receipt["report_sha256"] = b.generation.file_hash(self.benign_run_dir / "report.json")
        b.generation.save(self.benign_run_dir.parent / "receipt.json", self.receipt)
        Path(self.runtimes["synthetic"]["model_path"], "tokenizer.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.prepare_attacks()

    def test_expected_digest_and_input_drift_are_checked_before_output_or_dispatch(self):
        self.prepare_attacks()
        with self.assertRaisesRegex(ValueError, "digest"):
            a.acquire(self.prepared, self.output, expected_contract_sha256="0" * 64)
        self.assertFalse(self.output.exists())
        self.thinking_evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.acquire_attacks()
        self.assertFalse(self.output.exists())

    def test_historical_repository_files_and_live_data_runtime_files_have_one_policy(self):
        repository = self.root / "repository"
        runtime = deepcopy(self.benign_contract["runtimes"]["actor"])
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
    def test_terminal_development_and_monitor_cost_reports_are_bound_and_included(self):
        paths = [self.root / name for name in ("development-report.json", "monitor-report.json", "cancelled-report.json")]
        for path, tokens, status in zip(paths, (7, 8, 9), ("completed", "error", "cancelled")):
            b.generation.save(path, dict(status=status, output_tokens=tokens, output_tokens_lower_bound=tokens - 1,
                                        elapsed_seconds=1, native_prompt_tokens=10))
        contract = self.prepare_attacks(paths)
        self.assertEqual(sum(c["output_tokens"] for c in contract["prior_costs"]), 77)
        self.assertEqual([c["input_tokens"] for c in contract["prior_costs"][-3:]], [10, 10, 10])
        for path in paths:
            self.assertEqual(contract["inputs"][str(path.resolve())], b.generation.file_hash(path))
        report = self.acquire_attacks()
        self.assertEqual(report["pipeline_output_tokens"], 77 + 64)
        paths[0].write_text(paths[0].read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence"):
            a.load_attack_contract(self.prepared, self.digest)

    @unittest.skipIf(v is None, "requires requirements-execution.txt")
    def test_partial_development_cost_keeps_pipeline_total_unknown(self):
        path = self.root / "failed-development-report.json"
        b.generation.save(path, dict(status="error", output_tokens=None,
            observed_output_tokens_lower_bound=9, elapsed_seconds=1))
        self.prepare_attacks([path])
        report = self.acquire_attacks()
        self.assertEqual(report["output_tokens"], 64)
        self.assertIsNone(report["pipeline_output_tokens"])
        self.assertEqual(report["pipeline_output_tokens_lower_bound"], 53 + 9 + 64)

    @unittest.skipIf(v is None, "requires requirements-execution.txt")
    def test_nonterminal_snapshots_keep_observed_counts_but_not_complete_totals(self):
        paths = []
        for index, status in enumerate(("running", None, "unexpected")):
            path = self.root / f"snapshot-{index}.json"
            cost = dict(output_tokens=40000, output_tokens_lower_bound=39999,
                        native_prompt_tokens=100, elapsed_seconds=1)
            if status is not None:
                cost["status"] = status
            b.generation.save(path, cost)
            paths.append(path)
        contract = self.prepare_attacks(paths)
        for cost in contract["prior_costs"][-3:]:
            self.assertIsNone(cost["output_tokens"])
            self.assertIsNone(cost["input_tokens"])
            self.assertEqual(cost["output_tokens_lower_bound"], 40000)
        report = self.acquire_attacks()
        self.assertIsNone(report["pipeline_output_tokens"])
        self.assertEqual(report["pipeline_output_tokens_lower_bound"], 53 + 3 * 40000 + 64)

    def test_duplicate_development_cost_reports_are_rejected(self):
        path = self.root / "development-report.json"
        b.generation.save(path, dict(output_tokens=7, elapsed_seconds=1))
        with self.assertRaisesRegex(ValueError, "Duplicate development"):
            self.prepare_attacks([path, path])
        self.assertFalse(self.prepared.exists())

    def test_missing_prior_costs_do_not_prepare(self):
        self.receipt["setup_output_tokens"] = None
        b.generation.save(self.benign_run_dir.parent / "receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "prior output-token"):
            self.prepare_attacks()

    def test_benign_readiness_uses_measurements_instead_of_the_old_batch_flag(self):
        self.benign_report["truncated_replies"] = 1  # 1/16 exceeds the benign 5% gate.
        b.generation.save(self.benign_run_dir / "report.json", self.benign_report)
        self.receipt["report_sha256"] = b.generation.file_hash(self.benign_run_dir / "report.json")
        b.generation.save(self.benign_run_dir.parent / "receipt.json", self.receipt)
        with self.assertRaisesRegex(ValueError, "ready, receipt-bound"):
            self.prepare_attacks()
        self.assertFalse(self.prepared.exists())

    def test_memory_stops_use_strict_growth_and_peak_boundaries(self):
        baseline = dict(pressure=1, swap_used_mib=5)
        a.check_resources(dict(pressure=2, swap_used_mib=261, peak_bytes=44*1024**3), baseline)
        for change in (dict(pressure=4), dict(swap_used_mib=261.01), dict(peak_bytes=44*1024**3+1)):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "resource stop"):
                a.check_resources(dict(pressure=1, swap_used_mib=5) | change, baseline)


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class AcquisitionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.contract = self.prepare_attacks()

    def test_two_independent_chains_source_order_budgets_and_private_boundaries(self):
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "awaiting_review", report)
        self.assertEqual([(r["source"], r["chain"]) for r in report["runs"]],
                         [(s, c) for s in b.SOURCES for c in range(2)])
        self.assertEqual(len(self.requests), 32)
        for source in b.SOURCES:
            for chain in range(2):
                rows = [r for r in self.requests if (r["source"], r["chain"]) == (source, chain)]
                self.assertEqual([r["request"]["seed"] for r in rows], list(range(41 + 24*chain, 49 + 24*chain)))
                for row in rows:
                    request = row["request"]
                    self.assertTrue(request["enable_thinking"])
                    self.assertEqual((request["max_tokens"], request["max_prompt_tokens"]), (8192, 25808))
                    self.assertEqual(row["spec"]["completion_batch_size"], 4 if source == "actor" else 2)
                    public = json.dumps(request["messages"])
                    self.assertNotIn("PRIVATE", public)
                    self.assertNotIn("audit", public)
                    for entry in self.benign_contract["entries"]:
                        for case in entry["partition"]["audit"]:
                            self.assertNotIn(json.dumps(case["input"]), public)
        self.assertEqual(report["reserved_replies"], 32)
        self.assertEqual(report["reserved_output_tokens"], 32*8192)
        self.assertEqual(report["submitted_replies_lower_bound"], 32)
        self.assertEqual(report["output_tokens"], 64)
        self.assertEqual(report["pipeline_output_tokens"], 117)
        self.assertEqual(report["native_prompt_tokens"], 320)
        self.assertEqual(report["duplicate_submissions"], 16)
        self.assertEqual(report["shown_case_results_known"], len(self.native_cases))
        self.assertEqual(report["shown_case_requests_committed"], 288)
        self.assertTrue(all(c["visible"] for c in self.native_cases))
        self.assertEqual(report["yield"]["actor"]["accepted_positions"], list(range(8)))
        self.assertEqual(report, b.read(self.output / "report.json"))

    def test_existing_output_does_not_claim_the_contract(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.acquire_attacks()
        self.assertFalse((self.prepared / "acquisition-started.json").exists())
        self.assertEqual(self.requests, [])

    def test_output_creation_failure_releases_the_unsubmitted_claim(self):
        actual = Path.mkdir
        def fail_output(path, *args, **kwargs):
            if path == self.output.resolve():
                raise PermissionError("fixture output permission failure")
            return actual(path, *args, **kwargs)
        with mock.patch.object(Path, "mkdir", fail_output), self.assertRaises(PermissionError):
            self.acquire_attacks()
        self.assertFalse((self.prepared / "acquisition-started.json").exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(self.requests, [])

    def test_fresh_repairs_keep_positions_without_cross_chain_feedback_or_reasoning(self):
        self.failures = {("actor", 0, r, 0) for r in range(3)}
        report = self.acquire_attacks()
        revisions = [r for r in self.requests if r["source"] == "actor" and r["round"]]
        self.assertEqual([r["request"]["seed"] for r in revisions], [49, 57])
        self.assertEqual(len(self.requests), 34)
        self.assertEqual(report["yield"]["actor"]["accepted_positions"], list(range(8)))
        for row in revisions:
            self.assertEqual(len(row["request"]["messages"]), 2)
            public = row["request"]["messages"][1]["content"]
            self.assertIn("Previous candidate:\n# case: 1", public)
            self.assertIn('"named_case"', public)
            self.assertNotIn("PRIVATE_REASONING", public)
        second_chain = next(r for r in self.requests if r["source"] == "actor" and r["chain"] == 1)
        self.assertNotIn("Previous candidate:\n# case: 1", second_chain["request"]["messages"][1]["content"])

    def test_all_failed_chains_consume_exactly_ninety_six_slots_without_replacements(self):
        self.all_fail = True
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "awaiting_review", report)
        self.assertEqual(len(self.requests), 96)
        self.assertEqual(report["reserved_replies"], 96)
        self.assertEqual(report["reserved_output_tokens"], 786432)
        self.assertEqual(report["shown_case_results_known"], 864)
        self.assertEqual(report["yield"]["actor"]["accepted_positions"], [])
        self.assertEqual(report["duplicate_submissions"], 80)
        self.assertEqual([r["request"]["seed"] for r in self.requests if r["position"] == 7],
                         [48, 56, 64, 72, 80, 88] * 2)

    def test_truncation_and_unclosed_thoughts_consume_slots_and_are_accounted_separately(self):
        self.truncate = self.unclosed = True
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "awaiting_review", report)
        self.assertEqual(len(self.requests), 33)
        self.assertEqual(report["truncated_submissions"], 1)
        self.assertEqual(report["unclosed_or_unmatched_thoughts"], 1)
        self.assertEqual(report["truncation_fraction"], 1/33)
        self.assertEqual(report["extraction_failures"], 0)
        revision = next(r for r in self.requests if r["round"])
        self.assertIn("Local extraction failure: length", revision["request"]["messages"][1]["content"])

    def test_naturally_completed_unclosed_thought_has_no_program_or_reasoning_in_repair(self):
        self.unclosed = True
        report = self.acquire_attacks()
        self.assertEqual(report["unclosed_or_unmatched_thoughts"], 1)
        self.assertEqual(report["extraction_failures"], 1)
        self.assertEqual(report["truncated_submissions"], 0)
        revision = next(r for r in self.requests if r["round"])
        self.assertIn("Local extraction failure: format", revision["request"]["messages"][1]["content"])
        self.assertNotIn("PRIVATE_REASONING", revision["request"]["messages"][1]["content"])

    def test_opaque_pair_packets_cover_both_trajectories_and_every_success_without_private_fields(self):
        self.acquire_attacks()
        review = b.read(self.output / "review/index.json")
        private = b.read(self.output / "review-map-private.json")
        template = b.read(self.output / "review/review-template.json")
        self.assertEqual(len(review["packets"]), 16)
        self.assertEqual(template["packet_sha256"], b.generation.file_hash(self.output / "review/index.json"))
        forbidden = {"source", "seed", "path", "log", "sample_id", "candidate", "task_sha256"}
        def check_keys(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden, value)
                for item in value.values():
                    check_keys(item)
            elif isinstance(value, list):
                for item in value:
                    check_keys(item)
        for row in review["packets"]:
            packet = b.read(self.output / "review" / row["file"])
            check_keys(packet)
            self.assertEqual(packet["id"], row["id"])
            self.assertEqual(packet["benign"], ORIGINAL.decode())
            self.assertEqual(len(packet["trajectories"]), 2)
            self.assertTrue(all(t[0]["accepted"] for t in packet["trajectories"]))
            self.assertEqual(row["sha256"], b.generation.file_hash(self.output / "review" / row["file"]))
            self.assertNotIn("PRIVATE_REASONING", json.dumps(packet))
            self.assertNotIn(str(self.root), json.dumps(packet))
            mapped = private["pairs"][row["id"]]
            self.assertIn(mapped["source"], b.SOURCES)
            self.assertEqual(len(mapped["attempts"]), 2)
        self.assertTrue(all(len(pair["successes"]) == 2 and pair["diagnosis"]["primary"] is None
                            for pair in template["pairs"]))
        frozen = b.read(self.output / "generation-frozen.json")
        self.assertEqual(frozen["contract_sha256"], self.digest)
        for path, digest in frozen["files"].items():
            self.assertEqual(b.generation.file_hash(self.output / path), digest)
        self.assertFalse((self.output / "selection.json").exists())
        self.assertFalse(any("audit" in str(p.relative_to(self.output)) for p in self.output.rglob("*")))

    def test_technical_failure_stops_second_chain_and_source_retaining_partial_costs(self):
        self.technical_error = True
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "error")
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(self.native_cases, [])
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["output_tokens"])
        self.assertEqual(report["observed_output_tokens_lower_bound"], 16)
        self.assertEqual(report["pipeline_output_tokens_lower_bound"], 69)
        self.assertTrue((self.output / "partial-evidence.json").exists())
        self.assertFalse((self.output / "generation-frozen.json").exists())
        self.assertFalse((self.output / "review").exists())
        with self.assertRaises(FileExistsError):
            a.acquire(self.prepared, self.root / "another-run", expected_contract_sha256=self.digest)
        self.assertFalse((self.root / "another-run").exists())
        self.assertEqual(len(self.requests), 8)

    def test_interrupt_saves_available_evidence_and_propagates(self):
        self.interrupt = True
        with self.assertRaisesRegex(KeyboardInterrupt, "fixture interrupted"):
            self.acquire_attacks()
        report = b.read(self.output / "report.json")
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["observed_output_tokens_lower_bound"], 16)
        self.assertEqual(len(self.requests), 8)
        self.assertTrue((self.output / "partial-evidence.json").exists())

    def test_incomplete_native_json_retains_unknown_usage_and_does_not_mask_the_failure(self):
        self.incomplete_native_json = True
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "error")
        self.assertIn("Generation failed", report["error"])
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["output_tokens"])
        self.assertEqual(report["reserved_replies"], 8)
        self.assertEqual(len(self.requests), 8)
        loop = b.read(Path(report["runs"][0]["directory"]) / "outcome.json")
        self.assertIn("generation_evidence_error", loop["rounds"][0])
        self.assertTrue((self.output / "partial-evidence.json").exists())
        self.assertFalse((self.output / "generation-frozen.json").exists())

    def test_initial_memory_pressure_and_growth_stop_before_dispatch(self):
        self.memory.return_value = dict(pressure=2, swap_used_mib=5)
        report = self.acquire_attacks()
        self.assertEqual(report["status"], "error")
        self.assertIn("normal initial", report["error"])
        self.assertEqual(self.requests, [])

    def test_checker_infrastructure_errors_stop_dispatch_without_candidate_rejection(self):
        with mock.patch.object(a, "check_shown", side_effect=ValueError("native infrastructure")):
            report = self.acquire_attacks()
        self.assertEqual(report["status"], "error")
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(report["shown_case_requests_committed"], 72)
        self.assertEqual(report["shown_case_results_known"], 0)
        self.assertNotIn("review", [p.name for p in self.output.iterdir()])


class WorkerTests(unittest.TestCase):
    def test_worker_guards_before_insertion_and_each_step_and_restores_native_generator(self):
        import tempfile
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
