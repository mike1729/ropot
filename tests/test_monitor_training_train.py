"""T4 trajectory and selection with real CPU Trainer/PEFT, without downloads."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.monitor_training import adapter, pinning, train, training
from src.monitor_training.rendering import Renderer
from tests.test_monitor_training_rendering import NativeTokenizer, row
from tests.test_monitor_training_smoke import STACK, settings as smoke_settings


def settings():
    value = smoke_settings()
    value["budgets"].update(max_tokens=1000000, max_fit_tokens=200000)
    return value


@unittest.skipUnless(STACK, "Optional CPU PEFT/Trainer stack is not installed")
class SharedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_training(self, *, pass_step=5, recipe=None, incomplete=False,
                     reload_mismatch=False, final_measurement_failure=False,
                     noisy_scoring=False, fail_training=False, fail_loading=False, name="attempt"):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback

        config = recipe or settings()
        settings_path = self.root / (name + "-settings.json")
        settings_path.write_text(json.dumps(config))
        tokenizer = NativeTokenizer()
        groups = {name: [row(task, role, program=f"print({task}) # {role}")
                         for task in tasks for role in ("benign", "actor_attack", "synthetic_attack")]
                  for name, tasks in (("shared_honest", range(1, 7)), ("development", range(101, 121)))}
        contract = dict(max_total_tokens=512, max_new_tokens=4, terminator_id=2,
                        budgets=dict(max_margin_tokens=100000, max_wall_seconds=120,
                                     max_cost_usd=1, gpu_hourly_usd=1))
        observed = dict(step=0, batches=[], models=[], trainers=[], generations=[], likelihoods=[],
                        cuda=mock.Mock())

        class CpuRuntime:
            def __init__(self, model_root, contract, *, tokenizer):
                torch.manual_seed(17)
                self.model = LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=512, bos_token_id=1, eos_token_id=2))
                self.model.eval()
                observed["models"].append(self.model)
                if fail_loading:
                    raise torch.OutOfMemoryError("Fixture base load OOM")

            def generate(self, ids):
                assert not self.model.training
                if noisy_scoring:
                    import random
                    import numpy as np
                    random.random()
                    np.random.random()
                    torch.rand(7)
                text = tokenizer.decode(ids, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                label = "PASS" if "# benign" in text else "FLAG"
                if pass_step is None or observed["step"] < pass_step:
                    label = "PASS"
                if reload_mismatch and len(observed["models"]) == 2:
                    label = "FLAG" if label == "PASS" else "PASS"
                reply = dict(text=label, reasoning=None,
                             completion="infrastructure_failure" if incomplete else "natural",
                             generated_token_ids=[10, 2] if label == "PASS" else [11, 12, 2])
                observed["generations"].append((list(ids), reply))
                return reply

            def log_likelihood(self, ids, continuation):
                assert not self.model.training
                observed["likelihoods"].append((list(ids), list(continuation)))
                return -float(len(continuation))

            def metadata(self):
                if final_measurement_failure and (fail_training or len(observed["models"]) == 2):
                    raise RuntimeError("Fixture final measurement failure")
                return dict(device="CPU test fixture")

        build = training.build_trainer

        def observed_build(model, *args, **kwargs):
            on_batch = kwargs["on_batch"]

            def observe_batch(inputs):
                on_batch(inputs)
                observed["batches"].extend(inputs["input_ids"].tolist())
                if fail_training:
                    raise torch.OutOfMemoryError("Fixture training OOM")

            class Observe(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    observed["step"] = state.global_step

            kwargs["on_batch"] = observe_batch
            trainer = build(model, *args, **kwargs)
            trainer.add_callback(Observe())
            observed["trainers"].append(trainer)
            return trainer

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(pinning, "load_contract", return_value=contract))
            stack.enter_context(mock.patch.object(pinning, "load_tokenizer", return_value=tokenizer))
            load = stack.enter_context(mock.patch.object(train.prepare, "load_prepared", return_value=groups))
            stack.enter_context(mock.patch.object(training, "build_trainer", side_effect=observed_build))
            for method in ("empty_cache", "set_device", "reset_peak_memory_stats", "synchronize"):
                observed["cuda"].attach_mock(stack.enter_context(mock.patch.object(torch.cuda, method)), method)
            for method in ("max_memory_allocated", "max_memory_reserved", "memory_reserved"):
                stack.enter_context(mock.patch.object(torch.cuda, method, return_value=1000))
            stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", return_value=(1000000, 2000000)))
            result = train.train("prepared", "export", "model", "inference", settings_path, self.root / name,
                                 expected_prepared_sha256="prepared-pin", expected_inference_sha256="base-pin",
                                 runtime_factory=CpuRuntime)
            if load.called:
                load.assert_called_once_with("prepared", "export", ["shared_honest", "development"],
                                             expected_prepared_sha256="prepared-pin")
        observed["expected_examples"] = training.make_examples(
            [Renderer(tokenizer, max_total_tokens=512, max_new_tokens=4, terminator_id=2).render(value)
             for value in groups["shared_honest"]], {"PASS": [10, 2], "FLAG": [11, 12, 2]}, "honest_twin")
        return result, observed

    def assert_trajectory(self, result, observed, passes, snapshots):
        import torch

        self.assertEqual(result["usage"]["updates"], 5 * passes)
        self.assertEqual(result["usage"]["row_presentations"], 18 * passes)
        self.assertEqual(result["usage"]["supervised_tokens"], 48 * passes)
        self.assertEqual(observed["batches"],
                         [value["input_ids"] for value in observed["expected_examples"]] * passes)
        self.assertEqual(len(observed["trainers"]), 1)
        self.assertEqual([candidate["passes"] for candidate in result["candidates"]], snapshots)
        self.assertEqual([candidate["step"] for candidate in result["candidates"]], [5 * p for p in snapshots])
        self.assertEqual(sorted(path.name for path in (self.root / "attempt" / "continuous").glob("checkpoint-*")),
                         sorted(f"checkpoint-{p * 5}" for p in snapshots))
        for p in snapshots:
            checkpoint = self.root / "attempt" / "continuous" / f"checkpoint-{5 * p}"
            for filename in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                             "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                self.assertTrue((checkpoint / filename).is_file(), filename)
            state = torch.load(checkpoint / "optimizer.pt", weights_only=True)
            self.assertEqual({int(value["step"].item()) for value in state["state"].values()}, {5 * p})
            trainer_state = json.loads((checkpoint / "trainer_state.json").read_text())
            self.assertEqual(trainer_state["epoch"], p)
            for group, count in (("fit", 18), ("development", 60)):
                rows = json.loads((self.root / "attempt" / f"pass-{p}" / f"{group}.json").read_text())
                self.assertEqual(len(rows), count)
                self.assertTrue(all(set(value["log_likelihood"]) == {"PASS", "FLAG"} for value in rows))

    def test_first_pass_selects_and_reloads_fresh_checkpoint(self):
        config = settings()
        # Batch size one has no partial microbatch to drop; the final two-row
        # accumulation window must still run with drop_last enabled.
        config["training_args"]["dataloader_drop_last"] = True
        result, observed = self.run_training(recipe=config)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(observed["cuda"].mock_calls[:2],
                         [mock.call.set_device(0), mock.call.reset_peak_memory_stats(0)])
        self.assert_trajectory(result, observed, 1, [1])
        self.assertEqual(len(observed["models"]), 2)
        selected = result["selected"]
        self.assertIsNotNone(selected)
        for filename, digest in selected["files"].items():
            self.assertEqual(digest, pinning.artifacts.digest(self.root / "attempt" / selected["checkpoint"] / filename))
        usage = result["usage"]
        expected_tokens = sum(map(len, observed["batches"])) + sum(
            len(ids) + len(reply["generated_token_ids"]) for ids, reply in observed["generations"]) + sum(
            len(ids) + len(continuation) for ids, continuation in observed["likelihoods"])
        self.assertEqual(usage["tokens"], expected_tokens)

    def test_second_snapshot_keeps_optimizer_and_exact_repeated_row_order(self):
        result, observed = self.run_training(pass_step=20)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assert_trajectory(result, observed, 4, [1, 4])
        self.assertFalse(result["candidates"][0]["development"]["passed"])
        self.assertTrue(result["candidates"][1]["development"]["passed"])

    def test_all_failed_candidates_stop_at_sixteen_passes(self):
        result, observed = self.run_training(pass_step=None)
        self.assertEqual(result["status"], "failed", result["error"])
        self.assertIsNone(result["selected"])
        self.assert_trajectory(result, observed, 16, [1, 4, 16])
        self.assertEqual(len(observed["models"]), 1)

    def test_final_snapshot_can_be_selected_and_reloaded(self):
        result, observed = self.run_training(pass_step=80)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assert_trajectory(result, observed, 16, [1, 4, 16])
        self.assertEqual(result["selected"]["passes"], 16)
        self.assertEqual(len(observed["models"]), 2)

    def test_scoring_does_not_change_the_training_rng_trajectory(self):
        import torch

        config = settings()
        config["lora"]["lora_dropout"] = 0.2
        reference, clean = self.run_training(pass_step=20, recipe=config, name="clean")
        result, noisy = self.run_training(pass_step=20, recipe=config, noisy_scoring=True, name="noisy")
        self.assertEqual(reference["status"], "passed", reference["error"])
        self.assertEqual(result["status"], "passed", result["error"])
        expected = dict(clean["trainers"][0].model.named_parameters())
        for name, parameter in noisy["trainers"][0].model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)

    def test_checkpoint_likelihoods_use_reload_precision_without_disabling_training_amp(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, TrainerCallback
        from src.monitor_training.runtime import NvidiaRuntime

        config = settings()
        config["pad_token_id"] = 2
        config["training_args"].update(bf16=True, activation_checkpointing=False, learning_rate=0.01)
        precision, scores = [], {}
        prefix, continuations = [5, 6, 7], {"PASS": [10, 2], "FLAG": [11, 12, 2]}

        class CpuRuntime(NvidiaRuntime):
            def __init__(self, checkpoint=None):
                self.torch = torch
                torch.manual_seed(17)
                self.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=2, max_position_embeddings=32)).to(torch.bfloat16)
                adapter.attach(self, config, checkpoint)
                self.model.eval()

            def _tensor(self, ids):
                return torch.tensor([ids], dtype=torch.long)

        runtime = CpuRuntime()
        projection = runtime.model.get_submodule("base_model.model." + config["lora"]["target_modules"][0])
        projection.register_forward_pre_hook(lambda module, inputs:
            precision.append((module.training, torch.is_autocast_enabled("cpu"))))

        class ScoreCheckpoint(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    runtime.model.eval()
                    try:
                        scores.update({label: runtime.log_likelihood(prefix, ids)
                                       for label, ids in continuations.items()})
                    finally:
                        runtime.model.train()

        examples = training.make_examples([dict(input_ids=prefix, role="actor_attack")] * 4,
                                          continuations, "honest_twin")
        trainer = training.build_trainer(runtime.model, examples, self.root / "amp", config,
                                         max_steps=2, callbacks=[ScoreCheckpoint()])
        trainer.train()
        self.assertTrue(hasattr(runtime.model, "_original_forward"))  # Actual Accelerate wrapper.
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in runtime.model.parameters()
                            if parameter.requires_grad))
        reloaded = CpuRuntime(self.root / "amp" / "checkpoint-1")
        reload_scores = {label: reloaded.log_likelihood(prefix, ids) for label, ids in continuations.items()}
        self.assertEqual(scores, reload_scores)
        self.assertEqual(scores["FLAG"] - scores["PASS"], reload_scores["FLAG"] - reload_scores["PASS"])
        self.assertEqual(precision, [(True, True)] * 4 + [(False, False)] * 2 + [(True, True)] * 4)

    def test_incomplete_scoring_retains_attempt_and_stops_training(self):
        result, observed = self.run_training(incomplete=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["updates"], 5)
        self.assertIsNone(result["selected"])
        events = [json.loads(line) for line in (self.root / "attempt" / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(value.get("kind") == "failure" for value in events))
        self.assertIn("evaluate", result["error"]["traceback"])
        self.assertIn("Incomplete generation", result["error"]["traceback"])
        self.assertEqual(len(observed["generations"]), 1)
        candidate = json.loads((self.root / "attempt" / "pass-1" / "summary.json").read_text())
        self.assertEqual(candidate["passes"], 1)
        self.assertIsNone(candidate["development"])

    def test_total_and_fit_budgets_are_rejected_before_model_loading(self):
        for name in ("max_tokens", "max_fit_tokens"):
            with self.subTest(name=name):
                config = settings()
                config["budgets"][name] = 1
                result, observed = self.run_training(recipe=config, name=name)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["selected"])
                self.assertIn(name, result["error"]["message"])
                self.assertEqual(result["usage"]["updates"], 0)
                self.assertEqual(observed["models"], [])
                requirements = json.loads((self.root / name / "rows.json").read_text())["token_requirements"]
                self.assertGreater(requirements[name], 1)

    def test_reloaded_prediction_mismatch_cannot_publish_selected_s(self):
        result, _ = self.run_training(reload_mismatch=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertIn("reload", result["error"]["message"].lower())

    def test_final_measurement_failure_revokes_selection(self):
        result, _ = self.run_training(final_measurement_failure=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["error"]["message"], "Fixture final measurement failure")
        self.assertIn("metadata", result["error"]["traceback"])
        saved = json.loads((self.root / "attempt" / "summary.json").read_text())
        self.assertEqual(saved, result)

    def test_early_failure_records_memory_even_when_metadata_fails(self):
        for metadata_failure in (False, True):
            with self.subTest(metadata_failure=metadata_failure):
                name = f"early-failure-{metadata_failure}"
                result, _ = self.run_training(name=name, fail_training=True,
                                              final_measurement_failure=metadata_failure)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["selected"])
                self.assertEqual(result["error"]["type"], "OutOfMemoryError")
                self.assertEqual(result["error"]["message"], "Fixture training OOM")
                self.assertEqual(result["usage"]["updates"], 0)
                self.assertEqual(result["usage"]["peak_allocated_bytes"], 1000)
                self.assertEqual(result["usage"]["peak_reserved_bytes"], 1000)
                self.assertEqual(result["usage"]["memory_headroom_bytes"], 1000000)
                saved = json.loads((self.root / name / "summary.json").read_text())
                self.assertEqual(saved, result)

    def test_loading_failure_retains_memory_without_runtime_metadata(self):
        result, observed = self.run_training(fail_loading=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["error"]["type"], "OutOfMemoryError")
        self.assertEqual(result["error"]["message"], "Fixture base load OOM")
        self.assertEqual(result["usage"]["updates"], 0)
        self.assertEqual(result["usage"]["peak_allocated_bytes"], 1000)
        self.assertEqual(result["usage"]["peak_reserved_bytes"], 1000)
        self.assertEqual(result["usage"]["memory_headroom_bytes"], 1000000)
        self.assertEqual(len(observed["models"]), 1)
        self.assertEqual(observed["generations"], [])
        self.assertFalse((self.root / "attempt" / "runtime.json").exists())

    def test_checkpoint_rotation_is_rejected_before_loading(self):
        config = settings()
        config["training_args"]["save_total_limit"] = 1
        result, observed = self.run_training(recipe=config)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["updates"], 0)
        self.assertEqual(observed["models"], [])


if __name__ == "__main__":
    unittest.main()
