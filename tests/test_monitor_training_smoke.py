"""Exercise the bounded smoke with real CPU PEFT checkpoints, without downloads."""

from contextlib import ExitStack
from functools import wraps
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.monitor_training import adapter, pinning, smoke
from tests.test_monitor_training_rendering import NativeTokenizer, row


STACK = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


def settings():
    value = json.loads((Path(__file__).resolve().parents[1] /
                        "configs/monitor_training/lora-smoke.example.json").read_bytes())
    value["lora"].update(r=4, lora_alpha=8,
                         target_modules=["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"])
    value["training_args"].update(use_cpu=True, bf16=False, dataloader_pin_memory=False)
    value.update(resume_atol=1e-7, resume_rtol=1e-6,
                 budgets=dict(max_tokens=100000, max_wall_seconds=120, max_cost_usd=1,
                              gpu_hourly_usd=1, min_free_memory_bytes=100))
    return value


@unittest.skipUnless(STACK, "Optional CPU PEFT/Trainer stack is not installed")
class SmokeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_smoke(self, recipe=None, name="attempt", fail_final_metadata=False,
                  fail_training=False, reload_mismatch=False, autocast_training=False,
                  incomplete_generation=False, broken_reload=False, resume_scale=None):
        import torch
        from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, TrainerCallback
        from src.monitor_training.runtime import NvidiaRuntime

        settings_path = self.root / f"{name}-settings.json"
        settings_path.write_text(json.dumps(recipe or settings()))

        class Tokenizer(NativeTokenizer):
            def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
                # The template fixture is reversible; untrained model replies can
                # also contain vocabulary entries absent from its native pieces.
                if any(token < 100 and token not in self.pieces.values() for token in ids):
                    return "unknown model token"
                return super().decode(ids, skip_special_tokens=skip_special_tokens,
                                      clean_up_tokenization_spaces=clean_up_tokenization_spaces)

        models, metadata_calls, generation_calls, training_token_counts = [], [], [], []
        self.training_token_counts = training_token_counts
        self.training_batch_lengths = []
        self.metadata_calls = metadata_calls
        training_failed = False
        self.autocast_calls = []

        class CpuRuntime(NvidiaRuntime):
            def __init__(self, model_root, contract, *, tokenizer):
                self.torch, self.contract, self.tokenizer = torch, contract, tokenizer
                # Editing the input after startup must not change the run or its snapshot.
                settings_path.write_text('{"changed_after_start": true}')
                torch.manual_seed(17)
                self.model = LlamaForCausalLM(LlamaConfig(vocab_size=512, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                    max_position_embeddings=512, bos_token_id=1, eos_token_id=2)).to(torch.bfloat16)
                self.model.eval()
                self.generation_config = GenerationConfig(max_new_tokens=contract["max_new_tokens"],
                    do_sample=False, num_beams=1, eos_token_id=2, pad_token_id=2, return_dict_in_generate=True)
                self.model.generation_config = self.generation_config
                models.append(self.model)

            def _tensor(self, ids):
                return torch.tensor([ids], dtype=torch.long)

            def log_likelihood(self, ids, continuation):
                raise AssertionError("Smoke must not score likelihoods")

            def generate(self, ids):
                reply = super().generate(ids)
                if reload_mismatch and len(models) == 2:
                    reply["generated_token_ids"][0] ^= 1
                if incomplete_generation:
                    reply["completion"] = "infrastructure_failure"
                generation_calls.append(dict(input_ids=ids, reply=reply))
                return reply

            def metadata(self):
                metadata_calls.append(1)
                if fail_final_metadata:
                    raise RuntimeError("Fixture metadata failure")
                return dict(device="CPU test fixture")

        base = dict(max_total_tokens=512, max_new_tokens=4, terminator_id=2)
        values = [row(task, role, program=f"print({task}) # {role}") for task in range(1, 7)
                  for role in ("benign", "actor_attack", "synthetic_attack")]
        build = smoke.training.build_trainer
        attach = adapter.attach

        def build_with_observation(model, *args, **kwargs):
            on_batch = kwargs["on_batch"]

            def observed_batch(inputs):
                nonlocal training_failed
                on_batch(inputs)
                training_token_counts.append(inputs["input_ids"].numel())
                self.training_batch_lengths.append(inputs["input_ids"].shape[-1])
                if fail_training:
                    training_failed = True
                    raise torch.OutOfMemoryError("Fixture training OOM")

            kwargs["on_batch"] = observed_batch
            if autocast_training:
                forward = model.forward
                @wraps(forward)
                def wrapped(*inputs, **options):
                    self.autocast_calls.append(model.training)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        return forward(*inputs, **options)
                model.forward = wrapped
            trainer = build(model, *args, **kwargs)
            if resume_scale is not None and Path(trainer.args.output_dir).name == "resumed":
                class ChangeResumedWeights(TrainerCallback):
                    def on_train_end(self, args, state, control, model=None, **kwargs):
                        parameter = next(value for value in model.parameters() if value.requires_grad)
                        with torch.no_grad():
                            parameter.view(-1)[0].mul_(resume_scale)
                trainer.add_callback(ChangeResumedWeights())
            return trainer

        def attach_with_broken_reload(runtime, config, checkpoint=None):
            names = attach(runtime, config, checkpoint)
            if checkpoint is not None:
                with torch.no_grad():
                    for name, parameter in runtime.model.named_parameters():
                        if ".lora_B." in name:
                            parameter.zero_()
            return names

        with ExitStack() as stack:
            load_base = stack.enter_context(mock.patch.object(pinning, "load_contract", return_value=base))
            stack.enter_context(mock.patch.object(pinning, "load_tokenizer", return_value=Tokenizer()))
            load = stack.enter_context(mock.patch.object(smoke.prepare, "load_prepared", return_value={"shared_honest": values}))
            stack.enter_context(mock.patch.object(smoke.training, "build_trainer", side_effect=build_with_observation))
            if broken_reload:
                stack.enter_context(mock.patch.object(adapter, "attach", side_effect=attach_with_broken_reload))
            synchronize = stack.enter_context(mock.patch.object(torch.cuda, "synchronize"))
            for method in ("empty_cache", "reset_peak_memory_stats"):
                stack.enter_context(mock.patch.object(torch.cuda, method))
            for method in ("max_memory_allocated", "max_memory_reserved", "memory_reserved"):
                stack.enter_context(mock.patch.object(torch.cuda, method, return_value=1000))
            stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", side_effect=lambda device:
                                                  (0 if training_failed else 1000000, 2000000)))
            result = smoke.smoke("prepared", "export", "model", "inference", settings_path, self.root / name,
                                 expected_prepared_sha256="prepared-pin", expected_inference_sha256="base-pin",
                                 runtime_factory=CpuRuntime)
            load_base.assert_called_once_with("inference", "base-pin", "model")
            load.assert_called_once_with("prepared", "export", ["shared_honest"],
                                         expected_prepared_sha256="prepared-pin")
            self.synchronize_calls = synchronize.call_count
        return result, models, generation_calls

    def test_smoke_trains_reloads_and_resumes_four_budgeted_updates(self):
        import torch

        result, models, generation_calls = self.run_smoke()
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(result["usage"]["updates"], 4)
        self.assertEqual(result["usage"]["row_presentations"], 16)
        self.assertEqual(result["usage"]["supervised_tokens"], 42)
        self.assertEqual(result["resume_max_abs_difference"], 0)
        self.assertNotIn("fit", result)
        self.assertEqual(len(generation_calls), 2)
        self.assertEqual(len(self.metadata_calls), 1)
        self.assertEqual(self.synchronize_calls, 1)
        self.assertEqual(self.training_batch_lengths[0], 512)
        self.assertEqual(result["usage"]["tokens"], sum(self.training_token_counts) + sum(
            len(call["input_ids"]) + len(call["reply"]["generated_token_ids"]) for call in generation_calls))
        self.assertEqual(result["usage"]["peak_reserved_bytes"], 1000)
        self.assertEqual(result["usage"]["memory_headroom_bytes"], 1000000)
        output = self.root / "attempt"
        snapshot = json.loads((output / "training.json").read_bytes())
        self.assertEqual(snapshot["lora"], settings()["lora"])
        self.assertEqual(snapshot["pad_token_id"], 2)
        self.assertEqual(snapshot["training_args"], settings()["training_args"])
        self.assertEqual(snapshot["training_args"]["activation_checkpointing_kwargs"], {"use_reentrant": False})
        self.assertEqual(snapshot["peft_version"], metadata.version("peft"))
        self.assertEqual(snapshot["source_sha256"], adapter.source_hashes())
        self.assertEqual(json.loads((self.root / "attempt-settings.json").read_bytes()), {"changed_after_start": True})
        self.assertEqual(json.loads((output / "inputs.json").read_bytes()),
                         dict(prepared_sha256="prepared-pin", inference_sha256="base-pin"))
        self.assertEqual(json.loads((output / "rows.json").read_bytes())["stress_fixture"]["total_tokens"], 512)
        checkpoint = output / "continuous" / "checkpoint-2"
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                     "rng_state.pth", "trainer_state.json"):
            self.assertTrue((checkpoint / name).is_file(), name)
        config = json.loads((checkpoint / "adapter_config.json").read_bytes())
        self.assertEqual((config["r"], config["lora_alpha"]), (4, 8))
        self.assertTrue((output / "resumed" / "checkpoint-3" / "adapter_model.safetensors").is_file())
        self.assertFalse((output / "checkpoints.json").exists())
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event["step"] for event in events if event["kind"] == "update"], [1, 2, 3, 3])
        self.assertEqual([event["stage"] for event in events if event["kind"] == "reload_prediction"],
                         ["before", "after"])
        # Independent base loads receive the same frozen bytes, while adapters
        # saved after training contain nonzero B matrices.
        self.assertEqual(len(models), 2)
        frozen = [{name: parameter for name, parameter in model.named_parameters() if not parameter.requires_grad}
                  for model in models]
        for name in frozen[0]:
            self.assertEqual(frozen[0][name].dtype, torch.bfloat16)
            torch.testing.assert_close(frozen[0][name], frozen[1][name], atol=0, rtol=0)
        for model in models:
            self.assertTrue(all(parameter.dtype == torch.float32 for parameter in model.parameters()
                                if parameter.requires_grad))
            self.assertEqual(model.model.layers[0].self_attn.q_proj.lora_A["default"].weight.shape[0], 4)
        self.assertTrue(any(parameter.count_nonzero() for name, parameter in models[0].named_parameters()
                            if ".lora_B." in name))

    def test_exhausted_token_budget_preserves_an_incomplete_attempt(self):
        recipe = settings()
        recipe["budgets"]["max_tokens"] = 1
        result, models, _ = self.run_smoke(recipe)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("budget exhausted", result["error"]["message"])
        self.assertEqual(result["usage"]["updates"], 0)
        self.assertEqual(models, [])
        output = self.root / "attempt"
        self.assertEqual(json.loads((output / "summary.json").read_bytes())["status"], "incomplete")
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["kind"], "failure")
        with self.assertRaises(FileExistsError):
            self.run_smoke(recipe)

    def test_broken_adapter_reload_is_rejected_before_trainer_can_reload_it_again(self):
        result, models, calls = self.run_smoke(broken_reload=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("Adapter weights changed after reload", result["error"]["message"])
        self.assertEqual(result["usage"]["updates"], 3)
        self.assertEqual(len(models), 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / "attempt" / "resumed").exists())

    def test_resume_gate_rejects_weight_drift_and_nan(self):
        for index, scale in enumerate((1.01, float("nan"))):
            with self.subTest(scale=scale):
                result, _, _ = self.run_smoke(name=f"resume-drift-{index}", resume_scale=scale)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["error"]["type"], "RuntimeError")
                self.assertIn("Resumed adapter differs", result["error"]["message"])
                self.assertEqual(result["usage"]["updates"], 4)

    def test_resume_gate_preserves_relative_tolerance(self):
        recipe = settings()
        recipe["resume_atol"] = 1e-12
        result, _, _ = self.run_smoke(recipe, resume_scale=1.0000005)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertGreater(result["resume_max_abs_difference"], recipe["resume_atol"])

    def test_generation_reserves_its_output_cap_before_calling_runtime(self):
        _, _, calls = self.run_smoke(name="reference")
        recipe = settings()
        recipe["budgets"]["max_tokens"] = sum(self.training_token_counts[:8]) + len(calls[0]["input_ids"])
        result, _, calls = self.run_smoke(recipe)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["updates"], 2)
        self.assertEqual(calls, [])

    def test_final_measurement_failure_cannot_leave_a_passing_summary(self):
        result, _, _ = self.run_smoke(fail_final_metadata=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["updates"], 4)
        self.assertEqual(result["error"]["message"], "Fixture metadata failure")
        output = self.root / "attempt"
        self.assertEqual(json.loads((output / "summary.json").read_bytes())["status"], "incomplete")
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["kind"], "failure")

    def test_accounting_uses_actual_batches_including_partial_accumulation(self):
        for microbatch, accumulation, rows, tokens in ((1, 2, 8, 20), (2, 2, 16, 42), (2, 4, 22, 58)):
            with self.subTest(microbatch=microbatch, accumulation=accumulation):
                recipe = settings()
                recipe["training_args"].update(per_device_train_batch_size=microbatch,
                                               gradient_accumulation_steps=accumulation)
                result, _, _ = self.run_smoke(recipe, name=f"batch-{microbatch}-{accumulation}")
                self.assertEqual(result["status"], "passed", result["error"])
                self.assertEqual(result["usage"]["row_presentations"], rows)
                self.assertEqual(result["usage"]["supervised_tokens"], tokens)

    def test_cleanup_preserves_original_training_failure(self):
        for metadata_failure in (False, True):
            with self.subTest(metadata_failure=metadata_failure):
                name = f"failure-{metadata_failure}"
                result, _, _ = self.run_smoke(name=name, fail_training=True,
                                              fail_final_metadata=metadata_failure)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["error"]["message"], "Fixture training OOM")
                self.assertIn("in observed_batch", result["error"]["traceback"])
                saved = json.loads((self.root / name / "summary.json").read_bytes())
                self.assertEqual(saved["error"], result["error"])
                events = [json.loads(line) for line in (self.root / name / "events.jsonl").read_text().splitlines()]
                failures = [event["message"] for event in events if event["kind"] == "failure"]
                self.assertEqual(failures, ["Fixture training OOM"] +
                                 (["Fixture metadata failure"] if metadata_failure else []))

    def test_reload_mismatch_retains_both_predictions(self):
        result, _, _ = self.run_smoke(reload_mismatch=True)
        self.assertEqual(result["status"], "incomplete")
        events = [json.loads(line) for line in (self.root / "attempt" / "events.jsonl").read_text().splitlines()]
        before, after = [event for event in events if event["kind"] == "reload_prediction"]
        self.assertEqual((before["stage"], after["stage"]), ("before", "after"))
        self.assertNotEqual(before["reply"]["generated_token_ids"], after["reply"]["generated_token_ids"])

    def test_incomplete_generation_preserves_its_reply_before_stopping(self):
        result, _, calls = self.run_smoke(incomplete_generation=True)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["usage"]["updates"], 2)
        self.assertEqual(len(calls), 1)
        events = [json.loads(line) for line in (self.root / "attempt" / "events.jsonl").read_text().splitlines()]
        prediction = next(event for event in events if event["kind"] == "reload_prediction")
        self.assertEqual(prediction["reply"]["completion"], "infrastructure_failure")

    def test_reload_generation_leaves_training_autocast_untouched(self):
        result, _, _ = self.run_smoke(autocast_training=True)
        self.assertEqual(result["status"], "passed", result["error"])
        self.assertEqual(self.autocast_calls, [True] * 16)


if __name__ == "__main__":
    unittest.main()
