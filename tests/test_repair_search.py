"""Prepared B2 orchestration through the real loop with fake model/sandbox edges."""
from copy import deepcopy
from hashlib import sha1, sha256
import importlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock


m = importlib.import_module("src.data-pipeline.stage05.repair_search")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(mock.patch.stopall)
        self.root = Path(temporary.name)
        self.output = self.root / "prepared"
        self.source_file = self.root / "fixture_source.py"
        self.source_file.write_text("# Frozen fixture source\n")
        self.executable = str(self.root / "python-fixture")
        self.events, self.sent = [], []
        self.active_source, self.active_round = None, None
        self.worklists, self.runtime, self.histories = {}, {}, {}
        self.expected_runtime = dict(executable=self.executable, python="fixture Python", versions={},
                                     source_sha256={}, load_seconds=.1)
        parent_recipe = {**m.attacks.DEFAULT_RECIPE, "initial": "HISTORICAL_OPTIMIZATION",
                         "max_tokens": 2048, "repair_max_tokens": 2048}
        for source in m.SOURCES:
            model = self.root / source
            model.mkdir()
            m.generation.save(model / "config.json", dict(model_type={"actor": "qwen3_5",
                                                                       "synthetic": "mistral3"}[source]))
            manifest = model / "manifest.json"
            m.generation.save(manifest, {})
            self.runtime[source] = dict(model_path=str(model), model_manifest=str(manifest),
                model_manifest_sha256=m.generation.file_hash(manifest),
                prefill_batch_size=4 if source == "actor" else 2,
                prefill_step_size=256 if source == "actor" else 2048)
            self.worklists[source] = []
            for position, identifier in enumerate(m.TASK_IDS):
                task = dict(question=f"Statement {identifier}", reference="PRIVATE_REFERENCE", cases=[
                    dict(index=0, input="PUBLIC_TEST", answer="1", visible=True)])
                digest = m.diagnostics.verification.digest(task)
                program = f"print({position})\n"
                binding = dict(id=identifier, split="train", sha256=digest)
                binding["implant"] = dict(program=program, stdin=f"ORIGINAL_{source}_{position}\n",
                    answer="PRIVATE_APPROVED", provenance="PRIVATE_CONTROL", candidate=
                    m.loop.study_tasks.candidate_identity("train", identifier, digest, source, program.encode()))
                self.worklists[source].append([binding, task])
            requests = m.loop.initial_requests(self.worklists[source], parent_recipe,
                m.attacks.messages(self.worklists[source], parent_recipe))
            for position, request in enumerate(requests):
                request.update(seed=33 + position)
                request["messages"] += [dict(role="assistant", content="print(7)\n"),
                    dict(role="user", content="historical public feedback 1"),
                    dict(role="assistant", content="print(8)\n"),
                    dict(role="user", content="historical public feedback 2")]
            attempts = [dict(attempt=16+i, position=i, row=i, round=2, seed=33+i,
                state="repairs_exhausted", extraction="format") for i in range(8)]
            self.histories[source] = dict(requests=requests, rows=[dict(text="```python\n```", status="stop")
                for _ in range(8)], attempts=attempts, outcome=dict(status="completed", attempts=attempts),
                inputs=dict(mlx_python=self.executable), effort=dict(committed_replies=24,
                    committed_tokens=49152, observed_tokens=48, known_submissions=24,
                    input_tokens=240, usage_complete=True))
        self.parent = dict(task_ids=m.TASK_IDS, worklists=self.worklists, runtimes=self.runtime,
            publication=dict(path="fixture-publication", sha256=m.approved_pairs.PUBLICATION_SHA256),
            recipes=dict(optimization=parent_recipe))
        self.parent_receipt = dict(status="completed", effort=dict(committed_replies=175,
            committed_tokens=310784, observed_tokens=51545))
        mock.patch.object(m, "ROOT", self.root).start()
        mock.patch.object(m, "parent_evidence", return_value=(self.parent, self.parent_receipt)).start()
        mock.patch.object(m, "inherited", side_effect=lambda source: deepcopy(self.histories[source])).start()
        mock.patch.object(m, "runtime_pins", side_effect=self.runtime_pins).start()
        mock.patch.object(m, "source_paths", return_value=[self.source_file]).start()
        mock.patch.object(m.subprocess, "check_output", side_effect=self.probe).start()
        mock.patch.object(m.diagnostics, "controls", return_value=[dict(name="private fixture control")]).start()
        self.controls = mock.patch.object(m.diagnostics, "run_controls", side_effect=self.run_controls).start()
        self.child = mock.patch.object(m.loop, "run_child", side_effect=self.generate).start()
        self.checker = mock.patch.object(m.attacks, "check_many", side_effect=self.check).start()
        self.diagnostic = mock.patch.object(m.diagnostics, "diagnose", side_effect=self.diagnose).start()
        # An unexpected route to a real sandbox is an immediate fixture failure.
        self.sandbox = mock.patch.object(m.diagnostics.runner, "run",
            side_effect=AssertionError("Fixture attempted real sandbox execution")).start()

    def runtime_pins(self, source, inherited_inputs, runtime):
        return dict(hashes={runtime["model_manifest"]: runtime["model_manifest_sha256"]}, model_stats={},
                    expected_runtime=deepcopy(self.expected_runtime), executable=self.executable, probe="fixture probe")

    def probe(self, command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            return "fixture-head\n"
        if command == ["git", "status", "--short"]:
            return ""
        self.assertEqual(command, [self.executable, "-c", "fixture probe"])
        return json.dumps({key: self.expected_runtime[key] for key in ("python", "versions")})

    def run_controls(self, worklists, directory):
        self.events.append("controls")
        directory.mkdir()
        result = dict(status="completed", controls=[dict(matched=True) for _ in range(3)],
                      case_executions=dict(suite=23, witness=3, total=26))
        m.generation.save(directory / "outcome.json", result)
        return result

    def generate(self, command, current, persist):
        round_file = Path(command[-1])
        spec = m.read(round_file)
        source, arm = round_file.parent.name.split("-")
        round_index = int(round_file.stem.split("-")[-1])
        self.active_source, self.active_round = source, round_index
        self.events.append(f"{source}-{arm}-{round_index}")
        self.sent.append(dict(source=source, arm=arm, round=round_index, requests=deepcopy(spec["requests"])))
        saved = m.read(round_file.parent / "outcome.json")
        self.assertEqual(saved["phase"], "dispatching")
        self.assertEqual(saved["rounds"][-1]["round_file_sha256"], m.generation.file_hash(round_file))
        self.assertEqual(spec["completion_batch_size"], 8)
        self.assertEqual(spec["prefill_batch_size"], self.runtime[source]["prefill_batch_size"])
        self.assertEqual(spec["prefill_step_size"], self.runtime[source]["prefill_step_size"])
        native = Path(spec["output_dir"])
        native.mkdir()
        m.generation.save(native / "inputs.json", dict(requests=spec["requests"]))
        m.generation.save(native / "runtime.json", self.expected_runtime)
        rows = []
        for index, request in enumerate(spec["requests"]):
            position = request["seed"] - 41 - 8 * round_index
            self.assertTrue(0 <= position < 8)
            text, status = f"print({100 + position})\n", "stop"
            if round_index == 0:
                if position == 2:
                    text = "```python\n```"
                elif position == 3:
                    status = "length"
                elif position == 6:
                    text = "x ="
            rows.append(dict(text=text, status=status, submitted=True, native_uid=index,
                tokens=[12, 99], generation_tokens=2, prompt_tokens=10 + round_index))
            (native / f"reply-{index:04d}.txt").write_text(text)
        m.generation.save(native / "outcome.json", dict(status="success", requests=rows,
            native=dict(generation_tokens=2 * len(rows), wall_time=.1), preparation_seconds=.1))
        return 0

    def check(self, candidates, directory, *, pairs, **execution):
        self.assertEqual(execution, dict(max_samples=4, concurrent_inference=False))
        directory.mkdir()
        verdicts = []
        for _, task_digest, program in candidates:
            position = next(i for i, (binding, _) in enumerate(self.worklists[self.active_source])
                            if binding["sha256"] == task_digest)
            binding = self.worklists[self.active_source][position][0]
            key = dict(task_sha256=task_digest, program_sha256=m.loop.study_tasks.candidate_identity(
                "train", binding["id"], task_digest, self.active_source, program)["program_sha256"])
            status = "passed" if position == 0 or position == 1 and self.active_round == 1 else "failed"
            feedback = dict(status=status, hidden="not_run", visible=[])
            verdict = dict(status=status, key=key, feedback=feedback)
            if position == 5:
                verdict.update(suite=dict(status="passed", key=key), witness=dict(private="PRIVATE_CAPTURE"),
                               feedback=dict(message=m.attacks.WITNESS_FAILURE))
            verdicts.append(verdict)
        return verdicts

    def diagnose(self, worklist, source, outcome, inherited_outcome, directory):
        directory.mkdir()
        first = [dict(task_id=worklist[a["position"]][0]["id"], position=a["position"], round=a["round"],
            attempt=a["attempt"], program_sha256=a["candidate"]["program_sha256"])
            for a in outcome["attempts"] if a.get("verification", {}).get("status") == "passed"]
        extracted = sum(a["extraction"] == "success" for a in outcome["attempts"])
        result = dict(status="completed", first_successes=first, cells={}, duplicates={},
            case_executions=dict(suite=extracted, acceptance_witness=len(first),
                diagnostic_witness=extracted-len(first), total=2 * extracted))
        m.generation.save(directory / "diagnostics.json", result)
        return result

    def prepare(self):
        return m.prepare(self.output)

    def execute(self):
        self.prepare()
        return m.run(self.output)

    def test_prepare_freezes_histories_sources_and_allowances_without_execution(self):
        settings = self.prepare()
        self.child.assert_not_called()
        self.checker.assert_not_called()
        self.controls.assert_not_called()
        self.diagnostic.assert_not_called()
        self.sandbox.assert_not_called()
        self.assertEqual(settings["budget"], m.BUDGET)
        self.assertEqual([(plan["source"], plan["arm"]) for plan in settings["plans"]],
                         [(source, arm) for source in m.SOURCES for arm in m.ARMS])
        for plan in settings["plans"]:
            self.assertEqual([r["seed"] for r in plan["first_requests"]], list(range(41, 49)))
            self.assertEqual({len(messages) for messages in plan["messages"]}, {2 if plan["arm"] == "C" else 8})
            self.assertEqual(len(plan["ancestry"]), 0 if plan["arm"] == "C" else 8)
            self.assertNotIn("PRIVATE", json.dumps(plan["first_requests"]))
        m.verify_index(self.output, m.read(self.output / "prepared.json"))
        with tarfile.open(self.output / "source-snapshot.tar.gz") as archive:
            self.assertEqual(archive.getnames(), ["fixture_source.py"])
        self.assertFalse((self.output / "run.json").exists())

    def test_all_six_policies_run_serially_preserve_stopping_and_audit(self):
        report = self.execute()
        self.assertTrue(report["complete"])
        self.assertEqual(self.events, ["controls"] +
            [f"{source}-{arm}-{r}" for source in m.SOURCES for arm in m.ARMS for r in range(3)])
        self.assertEqual(self.child.call_count, 18)
        self.assertEqual(self.diagnostic.call_count, 6)
        self.sandbox.assert_not_called()
        state = m.read(self.output / "run.json")
        for run in state["runs"]:
            attempts = run["outcome"]["attempts"]
            self.assertEqual(run["outcome"]["committed_replies"], 21)
            self.assertEqual(run["outcome"]["committed_tokens"], 21 * 2048)
            self.assertEqual([a["round"] for a in attempts if a["position"] == 0], [0])
            self.assertEqual([a["round"] for a in attempts if a["position"] == 1], [0, 1])
            self.assertEqual([a["seed"] for a in attempts if a["position"] == 7], [48, 56, 64])
            self.assertIsNotNone(run["elapsed_seconds"])
        for sent in self.sent:
            expected = 2 if sent["arm"] == "C" else 8 + 2 * sent["round"]
            self.assertEqual({len(r["messages"]) for r in sent["requests"]}, {expected})
            self.assertNotIn("PRIVATE", json.dumps(sent["requests"]))
        self.assertEqual(report["effort"]["new_spending"]["committed_replies"], 126)
        self.assertEqual(report["effort"]["historical_shared"]["whole_parent"]["committed_replies"], 175)
        self.assertTrue(m.read(self.output / "audit.json")["passed"])
        self.assertEqual(len(m.read(self.output / "mechanism-review.json")), 12)
        m.audit(self.output)

    def test_failed_control_stops_before_any_model_request(self):
        self.prepare()
        self.controls.side_effect = None
        self.controls.return_value = dict(status="error", controls=[])
        with self.assertRaisesRegex(ValueError, "Control gate failed"):
            m.run(self.output)
        self.child.assert_not_called()
        self.checker.assert_not_called()
        state = m.read(self.output / "run.json")
        self.assertEqual(state["runs"], [])
        self.assertEqual(state["status"], "error")
        self.assertEqual(m.read(self.output / "receipt.json")["status"], "error")

    def test_repeated_run_refused_without_extra_generation(self):
        self.execute()
        previous = self.child.call_count
        with self.assertRaisesRegex(ValueError, "already started"):
            m.run(self.output)
        self.assertEqual(self.child.call_count, previous)
        self.assertEqual(self.controls.call_count, 1)

    def test_source_drift_stops_before_controls_or_generation(self):
        self.prepare()
        self.source_file.write_text("# Changed after freeze\n")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            m.run(self.output)
        self.controls.assert_not_called()
        self.child.assert_not_called()
        self.assertFalse((self.output / "run.json").exists())

    def test_source_drift_during_first_round_stops_before_repairs(self):
        self.prepare()
        def drift(*args):
            result = self.generate(*args)
            self.source_file.write_text("# Changed during generation\n")
            return result
        self.child.side_effect = drift
        with self.assertRaises(ValueError):
            m.run(self.output)
        self.child.assert_called_once()
        self.diagnostic.assert_not_called()
        run = m.read(self.output / "run.json")["runs"][0]
        self.assertEqual(run["outcome"]["status"], "error")
        self.assertIn("Source changed", run["outcome"]["error"])
        self.assertEqual(run["effort"]["committed_tokens"], 8 * 2048)

    def test_audit_rejects_saved_request_drift(self):
        self.execute()
        path = self.output / "actor-A/round-0000.json"
        spec = m.read(path)
        spec["requests"][0]["seed"] += 1
        m.generation.save(path, spec)
        with self.assertRaisesRegex(ValueError, "Artifact drift"):
            m.audit(self.output)

    def test_replay_rejects_wrong_attempt_seed_and_repaired_history(self):
        self.execute()
        settings, state = m.read(self.output / "settings.json"), m.read(self.output / "run.json")
        outcome = deepcopy(state["runs"][0]["outcome"])
        outcome["attempts"][0]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            m.validate_arm(settings, settings["plans"][0], outcome)
        outcome = deepcopy(state["runs"][0]["outcome"])
        round_ = outcome["rounds"][1]
        path = Path(round_["round_file"])
        spec = m.read(path)
        spec["requests"][0]["messages"][-1]["content"] += "\nChanged repair objective"
        m.generation.save(path, spec)
        round_["round_file_sha256"] = m.generation.file_hash(path)
        native_inputs = Path(round_["generation_outcome"]).with_name("inputs.json")
        m.generation.save(native_inputs, dict(requests=spec["requests"]))
        with self.assertRaisesRegex(ValueError, "Request, seed"):
            m.validate_arm(settings, settings["plans"][0], outcome)

    def test_incomplete_native_usage_retains_reservations_and_unknown_inputs(self):
        native = self.root / "incomplete.json"
        native.write_text('{"requests":')
        outcome = dict(rounds=[dict(generation_outcome=str(native))], committed_replies=8,
            committed_tokens=16384, observed_tokens=4, known_submissions=None, usage_complete=False)
        usage = m.effort(outcome)
        self.assertEqual(usage["committed_replies"], 8)
        self.assertEqual(usage["committed_tokens"], 16384)
        self.assertEqual(usage["observed_tokens"], 4)
        self.assertIsNone(usage["input_tokens"])
        self.assertFalse(usage["usage_complete"])
        self.assertEqual(len(usage["input_usage_errors"]), 1)

    def test_truncated_native_outcome_preserves_failed_run_costs(self):
        self.prepare()
        def truncated(*args):
            result = self.generate(*args)
            native = Path(m.read(Path(args[0][-1]))["output_dir"]) / "outcome.json"
            native.write_text('{"requests":')
            return result
        self.child.side_effect = truncated
        with self.assertRaises(ValueError):
            m.run(self.output)
        self.child.assert_called_once()
        state = m.read(self.output / "run.json")
        self.assertEqual(state["status"], "error")
        usage = state["runs"][0]["effort"]
        self.assertEqual(usage["committed_tokens"], 8 * 2048)
        self.assertIsNone(usage["known_submissions"])
        self.assertIsNone(usage["input_tokens"])
        self.assertFalse(usage["usage_complete"])
        self.assertEqual(len(usage["input_usage_errors"]), 1)

    def test_missing_historical_input_usage_stays_unknown_during_preparation(self):
        self.histories["actor"]["effort"]["input_tokens"] = None
        settings = self.prepare()
        self.assertIsNone(settings["historical"]["optimization"]["input_tokens"])
        self.assertEqual(settings["historical"]["optimization"]["committed_replies"], 48)


class RuntimePinsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(mock.patch.stopall)
        self.root = Path(temporary.name).resolve()
        self.parent = self.root / "parent"
        mock.patch.object(m, "PARENT", self.parent).start()
        self.expected = dict(python="fixture Python", versions={"mlx": "fixture-version"})
        self.probe = mock.patch.object(m.subprocess, "check_output",
            return_value=json.dumps(self.expected)).start()

    def fixture(self, schema):
        source = "actor" if schema == "files" else "synthetic"
        model = self.root / source
        model.mkdir()
        contents = {"config.json": b'{"fixture":true}', "model.safetensors": b"FAKE_WEIGHT_BYTES",
                    "README.md": b"fixture metadata\n"}
        for filename, content in contents.items():
            (model / filename).write_bytes(content)
        if schema == "files":
            manifest = dict(files=[dict(path=name, bytes=len(content), sha256=sha256(content).hexdigest())
                                   for name, content in contents.items()])
        else:
            siblings = []
            for name, content in contents.items():
                entry = dict(rfilename=name, size=len(content))
                if name.endswith(".safetensors"):
                    entry.update(blobId="pointer-blob-id", lfs=dict(size=len(content),
                        sha256=sha256(content).hexdigest(), pointerSize=128))
                else:
                    # HF's non-LFS digest hashes a Git object, including its header.
                    entry["blobId"] = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                siblings.append(entry)
            manifest = dict(siblings=siblings)
        manifest_path = self.root / f"{source}-manifest.json"
        m.generation.save(manifest_path, manifest)
        native = self.parent / f"{source}-optimization/generation-0002"
        native.mkdir(parents=True)
        executable = self.root / f"{source}-python"
        executable.write_bytes(b"fixture interpreter, never executed\n")
        runtime_source = self.root / f"{source}-runtime-source.py"
        runtime_source.write_bytes(b"# fixture runtime module\n")
        m.generation.save(native / "inputs.json", dict(model_metadata_sha256={
            "config.json": sha256(contents["config.json"]).hexdigest()}))
        m.generation.save(native / "runtime.json", dict(self.expected, executable=str(executable),
            source_sha256={str(runtime_source): m.generation.file_hash(runtime_source)}, load_seconds=.1))
        runtime = dict(model_path=str(model), model_manifest=str(manifest_path),
                       model_manifest_sha256=m.generation.file_hash(manifest_path))
        return source, dict(mlx_python=str(executable)), runtime, contents

    def check_schema(self, schema):
        source, inherited, runtime, contents = self.fixture(schema)
        pins = m.runtime_pins(source, inherited, runtime)
        expected = {str(Path(runtime["model_path"]) / name): sha256(content).hexdigest()
                    for name, content in contents.items()}
        self.assertEqual(pins["model_sha256"], expected)
        for path, content in ((Path(runtime["model_path"]) / name, content) for name, content in contents.items()):
            self.assertEqual(pins["model_stats"][str(path)], [len(content), path.stat().st_mtime_ns])
        self.assertEqual(pins["hashes"][inherited["mlx_python"]], m.generation.file_hash(inherited["mlx_python"]))
        self.probe.assert_called_once()
        command = self.probe.call_args.args[0]
        self.assertEqual(command[:2], [inherited["mlx_python"], "-c"])
        self.assertIn("importlib.metadata", command[2])

    def test_files_manifest_hashes_each_local_artifact(self):
        self.check_schema("files")

    def test_siblings_manifest_checks_lfs_sha256_and_non_lfs_git_blob_hashes(self):
        self.check_schema("siblings")

    def test_both_schemas_reject_same_length_corrupted_weights(self):
        for schema in ("files", "siblings"):
            source, inherited, runtime, contents = self.fixture(schema)
            path = Path(runtime["model_path"]) / "model.safetensors"
            original = contents["model.safetensors"]
            path.write_bytes(b"X" + original[1:])
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                m.runtime_pins(source, inherited, runtime)
        self.probe.assert_not_called()

    def test_non_lfs_git_blob_hash_rejects_same_length_metadata_corruption(self):
        source, inherited, runtime, contents = self.fixture("siblings")
        path = Path(runtime["model_path"]) / "README.md"
        path.write_bytes(b"X" + contents["README.md"][1:])
        with self.assertRaises(ValueError):
            m.runtime_pins(source, inherited, runtime)
        self.probe.assert_not_called()
