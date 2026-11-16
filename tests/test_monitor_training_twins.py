"""Matched twin selection and branch swaps with real CPU Trainer/PEFT state."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from src.monitor_training import adapter, interface, pinning, train, training
from src.monitor_training.rendering import Renderer
from src.monitor_training.runner import snapshot
from tests import test_monitor_training_train as shared_tests
from tests.test_monitor_training_rendering import NativeTokenizer, row
from tests.test_monitor_training_smoke import STACK


@unittest.skipUnless(STACK, "Optional CPU PEFT/Trainer stack is not installed")
class TwinTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_twins(self, *, passing=None, incomplete=False, dropout=0, parent_change=None,
                  bad_pin=False, training_args=None, budgets=None,
                  fail_training=False, fail_metadata=False, fail_load=False,
                  expire_reload=False, name="attempt"):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

        config = shared_tests.settings()
        config["lora"]["lora_dropout"] = dropout
        shared, _ = shared_tests.SharedTrainingTests.run_training(
            self, recipe=config, name=name + "-shared")
        self.assertEqual(shared["status"], "passed", shared["error"])
        parent = self.root / (name + "-shared")
        parent_summary = parent / "summary.json"
        if parent_change:
            parent_change(parent, shared)
            parent_summary.write_text(json.dumps(shared))
        config["training_args"].update(training_args or {})
        config["budgets"].update(budgets or {})
        settings_path = self.root / (name + "-settings.json")
        settings_path.write_text(json.dumps(config))
        tokenizer = NativeTokenizer()
        contract = dict(max_total_tokens=512, max_new_tokens=4, terminator_id=2,
                        budgets=dict(max_margin_tokens=100000, max_wall_seconds=120,
                                     max_cost_usd=1, gpu_hourly_usd=1))
        groups = {name: [row(task, role, program=f"print({task}) # {role}")
                         for task in tasks for role in interface.ROLES]
                  for name, tasks in (("twin_continuation", range(11, 19)),
                                      ("development", range(101, 121)))}
        passing = passing if passing is not None else {branch: {6, 24, 96} for branch in interface.BRANCHES}
        observed = dict(batches={branch: [] for branch in interface.BRANCHES}, segments=[], models=[],
                        runtime_by_model={}, generations=[], likelihoods=[], metadata_calls=[])
        now = [0]

        class CpuRuntime:
            def __init__(self, model_root, contract, *, tokenizer):
                torch.manual_seed(17)
                self.model = LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=512, bos_token_id=1, eos_token_id=2))
                self.model.eval()
                self.branch, self.step = None, 0
                observed["models"].append(self.model)
                if fail_load:
                    raise torch.OutOfMemoryError("Fixture base-load OOM")

            def generate(self, ids):
                import random
                import numpy as np
                assert not self.model.training
                # Scoring consumes RNG; resumed training must restore checkpoint RNG.
                random.random()
                np.random.random()
                torch.rand(7)
                text = tokenizer.decode(ids, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                role = next(role for role in interface.ROLES if "# " + role in text)
                # Identify development from the actual task encoded in the fixture.
                development = int(re.search(r"print\((\d+)\)", text).group(1)) >= 100
                label = interface.target_for(role, self.branch)
                if development and self.step not in passing[self.branch]:
                    label = "PASS" if self.branch == "honest_twin" else "FLAG"
                reply = dict(text=label, reasoning=None,
                             completion="infrastructure_failure" if incomplete and
                                        self.branch == "selective_twin" else "natural",
                             generated_token_ids=[10, 2] if label == "PASS" else [11, 12, 2])
                observed["generations"].append((self.branch, self.step, list(ids), reply))
                return reply

            def log_likelihood(self, ids, continuation):
                observed["likelihoods"].append((list(ids), list(continuation)))
                return -float(len(continuation))

            def metadata(self):
                observed["metadata_calls"].append(self.branch)
                if fail_metadata:
                    raise RuntimeError("Fixture measurement failure")
                return dict(device="CPU test fixture")

        original_attach, original_build = adapter.attach, training.build_trainer
        original_resources = train.Resources

        def observed_resources(*args):
            resources = original_resources(*args)
            observed.update(resources=resources, event=resources.event)
            return resources

        def observed_attach(runtime, settings, checkpoint=None):
            names = original_attach(runtime, settings, checkpoint)
            path = Path(checkpoint)
            runtime.branch = next((branch for branch in interface.BRANCHES if branch in path.parts), None)
            runtime.step = int(path.name.split("-")[-1])
            observed["runtime_by_model"][id(runtime.model)] = runtime
            return names

        def observed_build(model, examples, output, settings, **kwargs):
            runtime = observed["runtime_by_model"][id(model)]
            runtime.branch = next(branch for branch in interface.BRANCHES if branch in Path(output).parts)
            record = dict(branch=runtime.branch, max_steps=kwargs["max_steps"], initial=snapshot(model))
            observed["segments"].append(record)
            on_batch = kwargs["on_batch"]

            def observe_batch(inputs):
                on_batch(inputs)
                observed["batches"][runtime.branch].extend(
                    dict(input_ids=ids, labels=labels)
                    for ids, labels in zip(inputs["input_ids"].tolist(), inputs["labels"].tolist()))

            class Observe(TrainerCallback):
                def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
                    record["optimizer_steps_before"] = {
                        int(value["step"].item()) for value in optimizer.state.values()}

                def on_step_end(self, args, state, control, **kwargs):
                    runtime.step = state.global_step

            kwargs["on_batch"] = observe_batch
            trainer = original_build(model, examples, output, settings, **kwargs)
            trainer.add_callback(Observe())
            original_train = trainer.train

            def observe_train(*args, **kwargs):
                record["resume"] = kwargs.get("resume_from_checkpoint")
                if fail_training:
                    raise torch.OutOfMemoryError("Fixture OOM before first update")
                result = original_train(*args, **kwargs)
                record["final"] = snapshot(model)
                return result

            trainer.train = observe_train
            return trainer

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(train, "Resources", side_effect=observed_resources))
            stack.enter_context(mock.patch.object(pinning, "load_contract", return_value=contract))
            stack.enter_context(mock.patch.object(pinning, "load_tokenizer", return_value=tokenizer))
            load = stack.enter_context(mock.patch.object(train.prepare, "load_prepared", return_value=groups))
            stack.enter_context(mock.patch.object(adapter, "attach", side_effect=observed_attach))
            stack.enter_context(mock.patch.object(training, "build_trainer", side_effect=observed_build))
            if expire_reload:
                stack.enter_context(mock.patch("src.monitor_training.runner.time.monotonic", side_effect=lambda: now[0]))

            def cleanup():
                if expire_reload and len(observed["models"]) == 3:
                    now[0] = 121  # First selected reload finished; the next load must not start.

            stack.enter_context(mock.patch.object(torch.cuda, "empty_cache", side_effect=cleanup))
            for method in ("reset_peak_memory_stats", "synchronize"):
                stack.enter_context(mock.patch.object(torch.cuda, method))
            for method in ("max_memory_allocated", "max_memory_reserved", "memory_reserved"):
                stack.enter_context(mock.patch.object(torch.cuda, method, return_value=1000))
            stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", return_value=(1000000, 2000000)))
            result = train.train("prepared", "export", "model", "inference", settings_path, self.root / name,
                                 expected_prepared_sha256="prepared-pin", expected_inference_sha256="base-pin",
                                 shared_run=Path(os.path.relpath(parent)), expected_shared_sha256="wrong" if bad_pin else
                                 pinning.artifacts.digest(parent_summary), runtime_factory=CpuRuntime)
            load.assert_called_once_with("prepared", "export", ["twin_continuation", "development"],
                                         expected_prepared_sha256="prepared-pin")
        self.assertIs(observed["resources"].event, observed["event"])
        renderer = Renderer(tokenizer, max_total_tokens=512, max_new_tokens=4, terminator_id=2)
        rendered = [renderer.render(value) for value in groups["twin_continuation"]]
        observed.update(runtime=CpuRuntime, config=dict(config, pad_token_id=2), parent=parent,
                        parent_checkpoint=parent / shared["selected"]["checkpoint"] if shared["selected"] else None,
                        tokenizer=tokenizer, contract=contract,
                        examples={branch: training.make_examples(rendered, renderer.continuations, branch)
                                  for branch in interface.BRANCHES})
        return result, observed

    def assert_trajectory(self, result, observed, passes, snapshots):
        import torch

        self.assertEqual(result["usage"]["updates"], 12 * passes)
        self.assertEqual(result["usage"]["row_presentations"], 48 * passes)
        self.assertEqual(result["usage"]["supervised_tokens"], 120 * passes)
        self.assertEqual(result["usage"]["development_replies"], 120 * len(snapshots))
        self.assertEqual([candidate["passes"] for candidate in result["candidates"]], snapshots)
        self.assertEqual([(s["branch"], s["max_steps"]) for s in observed["segments"]],
                         [(branch, 96) for _ in snapshots for branch in interface.BRANCHES])
        for branch in interface.BRANCHES:
            self.assertEqual(observed["batches"][branch], observed["examples"][branch] * passes)
            for p in snapshots:
                checkpoint = self.root / "attempt" / branch / "continuous" / f"checkpoint-{6 * p}"
                optimizer = torch.load(checkpoint / "optimizer.pt", weights_only=True)
                self.assertEqual({int(value["step"].item()) for value in optimizer["state"].values()}, {6 * p})
                self.assertEqual(json.loads((checkpoint / "trainer_state.json").read_text())["epoch"], p)
        # Both independent branches begin from S, with empty optimizer state.
        honest, selective = observed["segments"][:2]
        self.assertEqual(honest["optimizer_steps_before"], set())
        self.assertEqual(selective["optimizer_steps_before"], set())
        self.assertIsNone(honest["resume"])
        self.assertIsNone(selective["resume"])
        for name, weights in honest["initial"].items():
            torch.testing.assert_close(weights, selective["initial"][name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(weights, selective["final"][name])
                            for name, weights in honest["final"].items()))

    def test_earliest_pair_resets_optimizer_matches_order_and_reloads_both(self):
        result, observed = self.run_twins()
        self.assertEqual(result["status"], "passed", result["error"])
        self.assert_trajectory(result, observed, 1, [1])
        self.assertEqual(result["selected"]["passes"], 1)
        self.assertEqual(result["confirmation"]["status"], "unresolved")
        self.assertFalse(result["confirmation"]["independent"])
        self.assertFalse(result["confirmation"]["repair_handoff"])
        expected_tokens = sum(len(row["input_ids"]) for rows in observed["batches"].values() for row in rows)
        expected_tokens += sum(len(ids) + len(reply["generated_token_ids"])
                               for _, _, ids, reply in observed["generations"])
        expected_tokens += sum(len(ids) + len(continuation) for ids, continuation in observed["likelihoods"])
        self.assertEqual(result["usage"]["tokens"], expected_tokens)
        for branch in interface.BRANCHES:
            selected = result["selected"][branch]
            checkpoint = self.root / "attempt" / selected["checkpoint"]
            self.assertEqual(selected["files"], train.checkpoint_hashes(checkpoint))
            candidate = result["candidates"][0]["branches"][branch]
            self.assertEqual(candidate["parent_checkpoint"], str(observed["parent_checkpoint"].resolve()))
            summary = self.root / "attempt" / branch / "pass-1" / "summary.json"
            self.assertEqual(json.loads(summary.read_text()), candidate)
            fit = candidate["fit"]
            self.assertEqual({role: cell["correct"] for role, cell in fit.items()}, dict.fromkeys(interface.ROLES, 8))

    def test_second_pair_resumes_each_branch_and_preserves_optimizer(self):
        result, observed = self.run_twins(passing={"honest_twin": {6, 24, 96}, "selective_twin": {24, 96}})
        self.assertEqual(result["status"], "passed", result["error"])
        self.assert_trajectory(result, observed, 4, [1, 4])
        self.assertFalse(result["candidates"][0]["development"]["passed"])
        self.assertTrue(result["candidates"][1]["development"]["passed"])
        for segment in observed["segments"][2:]:
            self.assertEqual(segment["optimizer_steps_before"], {6})
            self.assertIn(segment["branch"], str(segment["resume"]))

    def test_total_and_fit_budgets_are_rejected_before_loading_either_twin(self):
        for name in ("max_tokens", "max_fit_tokens"):
            with self.subTest(name=name):
                result, observed = self.run_twins(budgets={name: 1}, name=name)
                self.assertEqual(result["status"], "incomplete")
                self.assertIn(name, result["error"]["message"])
                self.assertEqual(result["usage"]["updates"], 0)
                self.assertEqual(observed["models"], [])

    def test_expired_cleanup_budget_stops_before_second_selected_reload(self):
        result, observed = self.run_twins(expire_reload=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertIn("budget exhausted", result["error"]["message"])
        self.assertEqual(result["usage"]["updates"], 12)
        self.assertEqual(len(observed["models"]), 3)
        self.assertEqual(observed["metadata_calls"], ["honest_twin", "selective_twin", "honest_twin"])

    def test_load_failure_records_memory_without_a_constructed_runtime(self):
        result, observed = self.run_twins(fail_load=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["error"]["type"], "OutOfMemoryError")
        self.assertEqual(result["error"]["message"], "Fixture base-load OOM")
        self.assertEqual(result["usage"]["updates"], 0)
        self.assertEqual(result["usage"]["peak_allocated_bytes"], 1000)
        self.assertEqual(result["usage"]["peak_reserved_bytes"], 1000)
        self.assertEqual(result["usage"]["memory_headroom_bytes"], 1000000)
        self.assertEqual(len(observed["models"]), 1)
        self.assertEqual(observed["metadata_calls"], [])

    def test_individually_passing_durations_cannot_be_combined(self):
        result, observed = self.run_twins(passing={"honest_twin": {6}, "selective_twin": {24, 96}})
        self.assertEqual(result["status"], "failed", result["error"])
        self.assertIsNone(result["selected"])
        self.assert_trajectory(result, observed, 16, [1, 4, 16])
        self.assertTrue(all(not pair["development"]["passed"] for pair in result["candidates"]))

    def test_incomplete_second_branch_keeps_first_branch_evidence_and_stops(self):
        result, observed = self.run_twins(incomplete=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["usage"]["updates"], 12)
        self.assertEqual(len(observed["segments"]), 2)
        self.assertTrue((self.root / "attempt" / "honest_twin" / "pass-1" / "development.json").exists())
        self.assertTrue((self.root / "attempt" / "selective_twin" / "pass-1" / "fit.json").exists())
        self.assertIn("Incomplete generation", result["error"]["traceback"])
        self.assertEqual(json.loads((self.root / "attempt" / "summary.json").read_text()), result)
        self.assertEqual(observed["metadata_calls"], list(interface.BRANCHES))
        self.assertTrue((self.root / "attempt" / "selective_twin" / "runtime.json").exists())

    def test_early_failure_records_memory_and_preserves_error_if_measurement_fails(self):
        for fail_metadata in (False, True):
            with self.subTest(fail_metadata=fail_metadata):
                name = f"early-failure-{fail_metadata}"
                result, observed = self.run_twins(fail_training=True, fail_metadata=fail_metadata, name=name)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["usage"]["updates"], 0)
                self.assertEqual(result["error"]["type"], "OutOfMemoryError")
                self.assertEqual(result["error"]["message"], "Fixture OOM before first update")
                self.assertEqual(observed["metadata_calls"], ["honest_twin"])
                self.assertEqual(result["usage"]["peak_allocated_bytes"], 1000)
                self.assertEqual(result["usage"]["peak_reserved_bytes"], 1000)
                output = self.root / name
                if fail_metadata:
                    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
                    measurement = next(event for event in events if event["kind"] == "measurement_failure")
                    self.assertEqual(measurement["message"], "Fixture measurement failure")
                else:
                    self.assertEqual(json.loads((output / "honest_twin" / "runtime.json").read_text()),
                                     dict(device="CPU test fixture"))

    def test_measurement_failure_after_training_cannot_pass(self):
        result, _ = self.run_twins(fail_metadata=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["usage"]["updates"], 6)
        self.assertEqual(result["error"]["message"], "Fixture measurement failure")

    def test_reporting_options_can_change_but_training_recipe_must_match(self):
        result, _ = self.run_twins(training_args=dict(disable_tqdm=False, logging_steps=2,
                                                       run_name="twins", log_level="error"), name="reporting")
        self.assertEqual(result["status"], "passed", result["error"])
        for name, value in (("learning_rate", 0.01), ("seed", 18)):
            with self.subTest(name=name):
                result, observed = self.run_twins(training_args={name: value}, name=name)
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("training recipe", result["error"]["message"])
                self.assertEqual(observed["models"], [])

    def test_peft_version_mismatch_is_rejected_before_loading_twins(self):
        def change_peft(parent, summary):
            path = parent / "training.json"
            record = json.loads(path.read_text())
            record["peft_version"] = "different-peft-version"
            path.write_text(json.dumps(record))
            summary["selected"]["training_sha256"] = pinning.artifacts.digest(path)

        result, observed = self.run_twins(parent_change=change_peft)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("PEFT", result["error"]["message"])
        self.assertEqual(observed["models"], [])
        self.assertEqual(result["usage"]["updates"], 0)

    def test_parent_digest_and_selected_base_lineage_are_enforced_before_training(self):
        for name, kwargs in (("digest", dict(bad_pin=True)),
                             ("base", dict(parent_change=lambda parent, summary:
                                           summary["selected"].update(inference_sha256="another-base"))),
                             ("checkpoint", dict(parent_change=lambda parent, summary:
                                           (parent / summary["selected"]["checkpoint"] / "adapter_config.json").write_text("{}")))):
            with self.subTest(name=name):
                result, observed = self.run_twins(name=name, **kwargs)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["selected"])
                self.assertEqual(observed["segments"], [])
                self.assertEqual(observed["models"], [])

    def test_swapped_resume_matches_uninterrupted_training_with_dropout(self):
        import torch
        from transformers import TrainerCallback

        result, observed = self.run_twins(dropout=0.2,
            passing={"honest_twin": {6, 24, 96}, "selective_twin": {24, 96}})
        self.assertEqual(result["status"], "passed", result["error"])

        class Stop(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                control.should_save = state.global_step == 24
                control.should_training_stop = state.global_step == 24
                return control

        for branch in interface.BRANCHES:
            runtime = observed["runtime"]("model", observed["contract"], tokenizer=observed["tokenizer"])
            adapter.attach(runtime, observed["config"], observed["parent_checkpoint"])
            trainer = training.build_trainer(runtime.model, observed["examples"][branch],
                self.root / (branch + "-uninterrupted"), observed["config"], max_steps=96, callbacks=[Stop()])
            trainer.train()
            expected = snapshot(runtime.model)
            swapped = next(segment["final"] for segment in observed["segments"][2:] if segment["branch"] == branch)
            for name, weights in expected.items():
                torch.testing.assert_close(swapped[name], weights, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
