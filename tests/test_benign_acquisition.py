"""B1 boundaries with published-task fixtures and the real bounded coordinator."""
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest import mock

from tests.test_verification import v

b = importlib.import_module("src.data-pipeline.stage03.benign_acquisition")


class Fixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "tasks").mkdir()
        records, self.tasks = [], {}
        for identifier in range(50):
            task = dict(id=identifier, split="test", question="Statement: Previous candidate:\nSolve this task.",
                solutions=["PRIVATE_REFERENCE"], example_mappings=[dict(
                    input="\n4\n3 2 4 1\n\n", answer="1", case_indices=[0])],
                cases=[dict(index=0, input="4\n3 2 4 1\n", answer="1", visible=True)] + [
                    dict(index=i, input=f"task {identifier} case {i}\n", answer=str(i), visible=False)
                    for i in range(1, 15)])
            self.tasks[identifier] = task
            raw = json.dumps(task).encode()
            relative = f"tasks/test_{identifier}.json"
            if identifier < 20:  # Confirmation files deliberately do not exist.
                (self.root / relative).write_bytes(raw)
            records.append(dict(id=identifier, split="test", path=relative, sha256=sha256(raw).hexdigest(),
                                owner="development", block="initial", known_uses=[]))
        manifest = dict(tasks=records, construction_phases={},
            groups=dict(development=dict(initial=list(range(50)), reserve=[]),
                        construction=dict(initial=[], reserve=[])),
            provenance=dict(input_manifest_sha256="a"*64))
        b.generation.save(self.root / "manifest.json", manifest)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(b.study_tasks, "COHORT_SHA256", b.generation.file_hash(self.root / "manifest.json")).start()
        loaded = b.study_tasks.load_cohort(self.root)
        mock.patch.object(b.study_tasks, "load_cohort", return_value=loaded).start()
        evidence = self.root / "reviewed-evidence.json"
        evidence.write_text('{"fixture":"reviewed historical requests"}')
        files = {str(evidence): b.generation.file_hash(evidence)}
        self.exposure = dict(files={s: files for s in b.EXPOSURE_SOURCES}, tasks={
            str(r["id"]): dict(task_sha256=r["sha256"], known=list(range(15)), disclosed=[]) for r in records[:20]})
        self.exposure_path = self.root / "exposure.json"
        self.save_exposure()
        self.runtimes = {}
        hashes = {}
        for source in b.SOURCES:
            model = self.root / (source+"-model")
            model.mkdir()
            metadata = []
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                (model / name).write_text(source+" "+name)
                metadata.append(dict(path=name, sha256=b.generation.file_hash(model / name)))
            artifact = model / "model-manifest.json"
            artifact.write_text(json.dumps(dict(source=source, files=metadata)))
            hashes[source] = b.generation.file_hash(artifact)
            batching = dict(completion_batch_size=2, prefill_batch_size=1, prefill_step_size=256)
            proof = dict(files=files, model_manifest_sha256=hashes[source], thinking=False, max_tokens=8192,
                         max_prompt_tokens=4096, batching=batching)
            proof_path = model / "qualification.json"
            b.generation.save(proof_path, proof)
            self.runtimes[source] = dict(model_path=str(model), model_manifest=str(artifact),
                mlx_python="fixture-python", qualification_file=str(proof_path))
        mock.patch.object(b, "MODEL_HASHES", hashes).start()
        self.requests, self.checks, self.shown_failures, self.audit_failures = [], [], set(), set()
        self.audit_error = self.generation_error = self.audit_interrupt = None
        self.extraction_failure = self.truncate = False
        self.missing_prompt_usage = False

    def save_exposure(self):
        b.generation.save(self.exposure_path, self.exposure)

    def prepare(self):
        self.save_exposure()
        return b.prepare_contract(self.root / "prepared", self.exposure_path, self.runtimes)

    def generate(self, command, current, persist):
        spec = b.read(command[-1])
        directory = Path(spec["output_dir"])
        directory.mkdir()
        source, round_index = directory.parent.name, current["round"]
        rows = []
        for i, request in enumerate(spec["requests"]):
            self.requests.append(dict(source=source, round=round_index, request=deepcopy(request)))
            position = (request["seed"] - 41) % 8
            text = f"PRIVATE_REASONING_{source}_{round_index}\n```python\nprint('{source} r{round_index} p{position}')\n```"
            if self.extraction_failure and (source, round_index, position) == ("actor", 0, 0):
                text = "PRIVATE_REASONING_REFUSAL\n```python\n```"
            row = dict(status="stop", text=text, answer_text=text, submitted=True, tokens=[12, 99],
                       generation_tokens=2, prompt_tokens=11)
            if self.truncate and (source, round_index, position) == ("actor", 0, 0):
                row["status"] = "length"
            rows.append(row)
            (directory / f"reply-{i:04d}.txt").write_text(text)
        b.generation.save(directory / "runtime.json", dict(load_seconds=.1))
        native = dict(generation_tokens=2*len(rows), prompt_tokens=10*len(rows), wall_time=.2)
        if self.missing_prompt_usage:
            del native["prompt_tokens"]
        b.generation.save(directory / "outcome.json", dict(
            status="error" if source == self.generation_error else "success", requests=rows,
            native=native))
        return 0

    def verify(self, candidates, directory, **settings):
        directory = Path(directory)
        self.checks.append((directory, deepcopy(candidates)))
        is_audit = directory.name.endswith("-audit")
        source = directory.name.removesuffix("-audit") if is_audit else directory.parent.name
        if is_audit and source == self.audit_interrupt:
            raise KeyboardInterrupt("audit interrupted")
        results = []
        for task, task_hash, program in candidates:
            self.assertEqual(task_hash, v.digest(task))
            self.assertTrue(all(c["visible"] for c in task["cases"]))
            _, round_index, position = re.search(r"(actor|synthetic) r(\d+) p(\d+)", program.decode()).groups()
            failed = ((source, int(position)) in self.audit_failures if is_audit else
                      (source, int(round_index), int(position)) in self.shown_failures)
            status = "failed" if failed else "passed"
            cases = task["cases"][:1] if failed else task["cases"]
            feedback = dict(status=status, hidden="not_run", visible=[dict(
                input=c["input"], answer=c["answer"], status="wrong_answer" if failed else "passed",
                stdout="0", stderr="", returncode=0, diagnostic=None) for c in cases])
            if is_audit and source == self.audit_error:
                status, feedback = "error", dict(status="verification_error")
            results.append(dict(status=status, key=dict(task_sha256=task_hash,
                program_sha256=sha256(program).hexdigest(), contract_sha256="f"*64),
                feedback=feedback, log=str(directory / "native.eval"), sample_id=len(results)))
        return results

    def execute(self):
        with mock.patch.object(b.generation_loop, "run_child", side_effect=self.generate), \
             mock.patch.object(v, "verify_many", side_effect=self.verify):
            return b.acquire(self.root / "prepared", self.root / "run")


class PreparationTests(Fixture):
    def test_partition_reproducible_example_equivalents_and_exposure_grouping(self):
        task, history = deepcopy(self.tasks[0]), deepcopy(self.exposure["tasks"]["0"])
        task["cases"].append(dict(index=99, input=" 4\n3 2 4 1 \n", answer="1", visible=True))
        history["known"].append(99)
        first = b.partition(task, history)
        self.assertEqual(first, b.partition(task, history))
        self.assertEqual(first["examples"], [task["cases"][0], task["cases"][-1]])
        self.assertNotEqual(first["examples"][0]["input"], task["example_mappings"][0]["input"])
        self.assertNotIn(99, first["permutation"])
        disclosed = first["audit"][0]["index"]
        task["cases"].append(dict(task["cases"][disclosed], index=98))
        history["known"].append(98)
        history["disclosed"] = [98]
        second = b.partition(task, history)
        self.assertNotIn(disclosed, [c["index"] for c in second["audit"]])
        self.assertIn(disclosed, [c["index"] for c in second["menu"]])

    def test_size_filter_does_not_move_refill_or_remove_audit_cases(self):
        task, history = deepcopy(self.tasks[0]), self.exposure["tasks"]["0"]
        original = b.partition(task, history)
        audit, menu = original["audit"][0]["index"], original["menu"][0]["index"]
        for index in (audit, menu):
            task["cases"][index]["input"] += " "*4097
        filtered = b.partition(task, history)
        self.assertEqual([c["index"] for c in filtered["audit"]], [c["index"] for c in original["audit"]])
        self.assertEqual(len(filtered["menu"]), 7)
        self.assertEqual(filtered["removed_menu"], [menu])
        self.assertNotIn(menu, [c["index"] for c in filtered["audit"]])

    def test_unknown_history_cannot_supply_audit_minimum(self):
        task, history = deepcopy(self.tasks[0]), deepcopy(self.exposure["tasks"]["0"])
        history["known"] = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            b.partition(task, history)

    def test_partition_cannot_override_disclosure_in_the_reviewed_file(self):
        history = self.exposure["tasks"]["0"]
        disclosed = b.partition(self.tasks[0], history)["audit"][0]["index"]
        history["disclosed"] = [disclosed]
        contract = self.prepare()
        forged = deepcopy(self.exposure)
        forged["tasks"]["0"]["disclosed"] = []
        contract["exposure"] = forged
        contract["entries"][0]["partition"] = b.partition(self.tasks[0], forged["tasks"]["0"])
        b.generation.save(self.root / "prepared/contract.json", contract)
        with self.assertRaisesRegex(ValueError, "partition"):
            b.load_contract(self.root / "prepared")

    def test_frozen_order_exclusions_evidence_and_no_confirmation_reads(self):
        del self.exposure["tasks"]["0"]
        with mock.patch.object(b.study_tasks, "load_task", wraps=b.study_tasks.load_task) as load:
            contract = self.prepare()
        self.assertEqual(contract["status"], "ready")
        self.assertEqual([e["binding"]["id"] for e in contract["entries"]], list(range(1, 9)))
        self.assertEqual(contract["exclusions"][0]["id"], 0)
        self.assertEqual([call.args[1] for call in load.call_args_list], list(range(9)))
        self.assertEqual(contract["confirmation"], list(range(20, 50)))
        self.assertEqual(contract["case_execution_allowance"], 496)
        self.assertIs(contract["recipe"]["strip_comments"], True)
        self.assertEqual(contract, b.read(self.root / "prepared/contract.json"))

    def test_fewer_than_eight_preserves_preparation_failure(self):
        for history in self.exposure["tasks"].values():
            history["known"] = []
        contract = self.prepare()
        self.assertEqual(contract["status"], "insufficient_cases")
        self.assertEqual(len(contract["exclusions"]), 20)
        self.assertEqual(contract["entries"], [])

    def test_changed_or_missing_evidence_cannot_freeze(self):
        self.exposure["files"]["suite"] = {}
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_thinking_on_short_cap_and_invalid_envelopes_rejected(self):
        runtime = self.runtimes["actor"]
        original = b.read(runtime["qualification_file"])
        for change in (dict(thinking=True), dict(max_tokens=256), dict(max_prompt_tokens=0),
                dict(batching=dict(completion_batch_size=2, prefill_batch_size=3, prefill_step_size=256))):
            with self.subTest(change=change), self.assertRaises(ValueError):
                b.generation.save(Path(runtime["qualification_file"]), original | change)
                b.qualify("actor", runtime)

    def test_a_correct_manifest_with_the_wrong_model_path_is_rejected(self):
        wrong = dict(self.runtimes["actor"], model_path=self.runtimes["synthetic"]["model_path"])
        with self.assertRaisesRegex(ValueError, "Model metadata"):
            b.qualify("actor", wrong)


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class ExecutionTests(Fixture):
    def setUp(self):
        super().setUp()
        self.contract = self.prepare()

    def test_actual_loop_two_sources_accounting_and_private_audits(self):
        report = self.execute()
        self.assertEqual(report["status"], "completed", report)
        self.assertTrue(report["ready_for_attack_acquisition"])
        self.assertEqual(report["joint_positions"], list(range(8)))
        self.assertEqual(report["output_tokens"], 32)
        self.assertEqual(report["native_prompt_tokens"], 160)
        self.assertEqual(report["output_tokens_per_joint_task"], 4)
        self.assertEqual(report["submitted_replies_lower_bound"], 16)
        self.assertEqual([r["source"] for r in self.requests], ["actor"]*8+["synthetic"]*8)
        self.assertEqual([r["request"]["seed"] for r in self.requests], list(range(41, 49))*2)
        for source in b.SOURCES:
            self.assertEqual(report["sources"][source]["reserved_output_tokens"], 8*8192)
            self.assertTrue(all(len(a["cases"]) == 4 for a in report["sources"][source]["audits"]))
            attempts = b.read(self.root / "run" / source / "outcome.json")["attempts"]
            inputs = b.read(self.root / "run" / source / "inputs.json")
            engine_sources = {str(p.absolute()): b.generation.file_hash(p) for p in b.generation_loop.source_files()}
            self.assertEqual(inputs["source_sha256"], engine_sources)
            self.assertTrue(engine_sources.items() <= self.contract["sources"].items())
            for entry, attempt, audit in zip(self.contract["entries"], attempts, report["sources"][source]["audits"]):
                for verdict, cases in ((attempt["verification"], entry["partition"]["examples"]+entry["partition"]["menu"]),
                                       (audit["verdict"], entry["partition"]["audit"])):
                    self.assertEqual(verdict["key"]["task_sha256"], entry["binding"]["sha256"])
                    self.assertEqual(verdict["key"]["case_view_sha256"], v.digest(dict(
                        cases=[dict(c, visible=True) for c in cases])))
        for row in self.requests:
            public = json.dumps(row["request"]["messages"])
            self.assertNotIn("PRIVATE", public)
            for entry in self.contract["entries"]:
                for case in entry["partition"]["audit"]:
                    self.assertNotIn(json.dumps(case["input"]), public)
        handoff = json.dumps(report)
        for entry in self.contract["entries"]:
            for case in entry["partition"]["audit"]:
                self.assertNotIn(json.dumps(case["input"]), handoff)
        self.assertEqual(report, b.read(self.root / "run/report.json"))

    def test_frozen_reply_and_token_allowances_each_limit_repairs(self):
        self.shown_failures = {("actor", 0, 0)}
        for key, value in (("reply_allowance", 8), ("output_token_allowance", 8*8192)):
            with self.subTest(ceiling=key):
                limited = dict(self.contract, **{key: value})
                b.generation.save(self.root / "prepared/contract.json", limited)
                report = self.execute()
                self.assertEqual(report["status"], "completed", report)
                self.assertEqual(sum(r["source"] == "actor" for r in self.requests), 8)
                self.assertEqual(report["sources"]["actor"]["reserved_output_tokens"], 8*8192)
                self.assertEqual(report["joint_positions"], list(range(1, 8)))
                shutil.rmtree(self.root / "run")
                self.requests.clear()

    def test_repairs_use_only_current_code_public_context_and_feedback(self):
        self.shown_failures = {("actor", 0, 0), ("actor", 1, 0)}
        report = self.execute()
        self.assertEqual(report["status"], "completed", report)
        revisions = [r["request"] for r in self.requests if r["source"] == "actor" and r["round"]]
        self.assertEqual([r["seed"] for r in revisions], [49, 57])
        self.assertTrue(all(len(r["messages"]) == 2 for r in revisions))
        self.assertEqual([r["max_prompt_tokens"] for r in revisions], [4096, 4096])
        final = revisions[1]["messages"][1]["content"]
        self.assertIn(self.tasks[0]["question"], final)  # Delimiter in a statement is harmless.
        self.assertIn("actor r1 p0", final)
        self.assertNotIn("actor r0 p0", final)
        self.assertNotIn("hidden=", final)
        self.assertNotIn("PRIVATE_REASONING", final)
        self.assertEqual(report["submitted_replies_lower_bound"], 18)

    def test_audit_failure_disqualifies_without_repair_or_candidate_replacement(self):
        self.audit_failures = {("actor", 0)}
        report = self.execute()
        self.assertEqual(report["joint_positions"], list(range(1, 8)))
        self.assertEqual(len(self.requests), 16)
        failed = report["sources"]["actor"]["audits"][0]
        self.assertEqual([c["status"] for c in failed["cases"]], ["wrong_answer"]+["not_run"]*3)
        self.assertEqual(report["sources"]["actor"]["coverage"], 7/8)

    def test_extraction_failure_gets_a_fixed_diagnostic_and_consumes_its_slot(self):
        self.extraction_failure = True
        report = self.execute()
        revision = next(r["request"] for r in self.requests if r["source"] == "actor" and r["round"])
        content = revision["messages"][1]["content"]
        self.assertIn("Local extraction failure: format", content)
        self.assertNotIn("PRIVATE_REASONING_REFUSAL", content)
        self.assertEqual(report["submitted_replies_lower_bound"], 17)
        self.assertEqual(report["sources"]["actor"]["reserved_output_tokens"], 9*8192)

    def test_a_single_truncation_among_seventeen_submissions_fails_the_gate(self):
        self.truncate = True
        report = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["joint_coverage"], 1)
        self.assertEqual(report["truncated_replies"], 1)
        self.assertEqual(report["submitted_replies_lower_bound"], 17)
        self.assertFalse(report["ready_for_attack_acquisition"])

    def test_exhausted_benigns_stay_in_the_joint_denominator(self):
        self.shown_failures = {("actor", r, p) for r in range(3) for p in range(3)}
        report = self.execute()
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["joint_coverage"], 5/8)
        self.assertEqual(len(report["sources"]["actor"]["audits"]), 5)
        self.assertEqual(report["submitted_replies_lower_bound"], 22)
        self.assertEqual(report["output_tokens_per_joint_task"], 44/5)
        self.assertFalse(report["ready_for_attack_acquisition"])

    def test_missing_native_prompt_usage_remains_unknown(self):
        self.missing_prompt_usage = True
        report = self.execute()
        self.assertIsNone(report["native_prompt_tokens"])
        self.assertIsNone(report["output_tokens"])
        self.assertIsNone(report["output_tokens_per_joint_task"])
        self.assertEqual(report["observed_output_tokens_lower_bound"], 32)
        self.assertFalse(report["usage_complete"])
        self.assertFalse(report["ready_for_attack_acquisition"])

    def test_qwen_audit_infrastructure_failure_blocks_gemma(self):
        self.audit_error = "actor"
        report = self.execute()
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["ready_for_attack_acquisition"])
        self.assertEqual(set(report["sources"]), {"actor"})
        self.assertEqual({r["source"] for r in self.requests}, {"actor"})
        self.assertEqual(report["output_tokens"], 16)

    def test_generation_technical_failure_has_no_retries_or_second_source(self):
        self.generation_error = "actor"
        report = self.execute()
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["output_tokens_per_joint_task"])
        self.assertEqual(len(self.requests), 8)
        self.assertEqual(self.checks, [])

    def test_interruption_saves_report_and_propagates(self):
        self.audit_interrupt = "actor"
        with self.assertRaises(KeyboardInterrupt):
            self.execute()
        report = b.read(self.root / "run/report.json")
        self.assertEqual(report["status"], "cancelled")
        self.assertEqual(report["output_tokens"], 16)
        self.assertEqual({r["source"] for r in self.requests}, {"actor"})

    def test_changed_qualification_cannot_relax_the_frozen_envelope(self):
        self.contract["runtimes"]["actor"]["qualification"]["max_prompt_tokens"] = 999999
        b.generation.save(self.root / "prepared/contract.json", self.contract)
        with self.assertRaisesRegex(ValueError, "Changed runtime"):
            self.execute()
        self.assertEqual(self.requests, [])

    def test_changed_qualification_bytes_are_rejected_even_if_json_is_unchanged(self):
        path = Path(self.runtimes["actor"]["qualification_file"])
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Changed runtime"):
            self.execute()
        self.assertEqual(self.requests, [])

    def test_changed_allocation_is_rejected_before_reading_task_files(self):
        self.contract["entries"][0]["binding"]["id"] = 20
        b.generation.save(self.root / "prepared/contract.json", self.contract)
        with mock.patch.object(b.study_tasks, "load_task") as load, \
             self.assertRaisesRegex(ValueError, "allocation"):
            self.execute()
        load.assert_not_called()

    def test_tampered_selected_program_is_rejected_before_audit_execution(self):
        actual = b.generation_loop.run
        def tamper(*args, **kwargs):
            result = actual(*args, **kwargs)
            Path(result["attempts"][0]["program"]).write_text("print('tampered')")
            return result
        with mock.patch.object(b.generation_loop, "run", side_effect=tamper):
            report = self.execute()
        self.assertEqual(report["status"], "error")
        self.assertIn("Selected program changed", report["error"])
        self.assertFalse(any(directory.name.endswith("-audit") for directory, _ in self.checks))
        self.assertEqual({r["source"] for r in self.requests}, {"actor"})
