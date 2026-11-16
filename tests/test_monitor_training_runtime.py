"""CUDA adapter boundaries without importing a model stack or requiring a GPU."""

from contextlib import nullcontext
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from src.monitor_training import runtime


class Tokenizer:
    pad_token_id = 0
    bos_token_id = 8

    def decode(self, tokens, **kwargs):
        assert kwargs == dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return "".join({1: "PASS", 2: "FLAG", 3: "<think>", 9: "<eos>"}.get(token, str(token))
                       for token in tokens)


class Tensor:
    """Tiny list tensor to check the actual continuation-position calculation."""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, indices):
        def select(values, remaining):
            if not remaining:
                return values
            first, *rest = remaining
            selected = values[first]
            return [select(value, rest) for value in selected] if isinstance(first, slice) else select(selected, rest)
        return Tensor(select(self.values, indices if isinstance(indices, tuple) else (indices,)))

    def float(self):
        return self

    def tolist(self):
        return self.values

    def unsqueeze(self, axis):
        assert axis == -1
        return Tensor([[value] for value in self.values])

    def gather(self, axis, targets):
        assert axis == -1
        return Tensor([[row[target[0]]] for row, target in zip(self.values, targets.values, strict=True)])

    def sum(self):
        return Tensor(sum(sum(row) for row in self.values))

    def item(self):
        return self.values


def log_softmax(tensor, dim):
    assert dim == -1
    values = []
    for row in tensor.values:
        normalizer = max(row) + math.log(sum(math.exp(value - max(row)) for value in row))
        values.append([value - normalizer for value in row])
    return Tensor(values)


class MonitorTrainingRuntimeTests(unittest.TestCase):
    def contract(self):
        return dict(numerical_base="bnb-int8-bf16", attention="sdpa", seed=17,
                    max_new_tokens=3, max_total_tokens=100, terminator_id=9,
                    cuda_version="fixture-cuda", gpu_name="fixture")

    def fixture_modules(self):
        torch = SimpleNamespace(
            cuda=mock.Mock(), bfloat16="bf16", long="long", inference_mode=nullcontext,
            tensor=lambda values, **kwargs: Tensor(values), ones_like=lambda value: "attention-mask",
            log_softmax=log_softmax,
            version=SimpleNamespace(cuda="fixture-cuda"),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True, benchmark=True)),
        )
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.get_device_name.return_value = "fixture"
        parameter = SimpleNamespace(device=SimpleNamespace(type="cuda", index=0), dtype="int8")
        model = mock.Mock(is_loaded_in_8bit=True, config=SimpleNamespace(is_encoder_decoder=False))
        del model.get_base_model  # This fixture represents the unadapted base.
        model.parameters.side_effect = lambda: iter([parameter])
        transforms = SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=model)),
            BitsAndBytesConfig=mock.Mock(), GenerationConfig=mock.Mock(), set_seed=mock.Mock(),
        )
        transforms.BitsAndBytesConfig.return_value.to_dict.return_value = dict(
            load_in_8bit=True, llm_int8_threshold=6.0)
        modules = dict(torch=torch, transformers=transforms, bitsandbytes=object(), accelerate=object())
        return modules, model, transforms

    def test_import_has_no_cuda_or_transformers_dependency(self):
        subprocess.run([sys.executable, "-S", "-c", (
            "import sys; import src.monitor_training.runtime; "
            "assert not any(name in sys.modules for name in "
            "('torch', 'transformers', 'bitsandbytes', 'accelerate'))"
        )], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, timeout=10)

    def test_loader_freezes_local_cuda_quantization_and_clears_checkpoint_stops(self):
        modules, model, transforms = self.fixture_modules()
        tokenizer = Tokenizer()
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            adapter = runtime.NvidiaRuntime("/pinned/base", self.contract(), tokenizer=tokenizer)
        self.assertIs(adapter.tokenizer, tokenizer)
        kwargs = transforms.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(kwargs["device_map"], {"": 0})
        self.assertEqual(kwargs["dtype"], "bf16")
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertTrue(kwargs["use_safetensors"])
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        transforms.BitsAndBytesConfig.assert_called_once_with(load_in_8bit=True)
        generation = transforms.GenerationConfig.call_args.kwargs
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 1)
        self.assertEqual(generation["max_new_tokens"], 3)
        self.assertEqual(generation["eos_token_id"], 9)
        self.assertEqual(generation["pad_token_id"], 9)
        self.assertNotIn("max_length", generation)
        self.assertNotIn("top_k", generation)
        self.assertIs(model.generation_config, adapter.generation_config)
        self.assertFalse(modules["torch"].backends.cuda.matmul.allow_tf32)
        model.eval.assert_called_once()

    def test_gpu_and_cuda_mismatches_fail_before_model_load(self):
        for field in ("cuda_version", "gpu_name"):
            with self.subTest(field=field):
                modules, _, transforms = self.fixture_modules()
                contract = dict(self.contract(), **{field: "different"})
                with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
                    with self.assertRaisesRegex(ValueError, "CUDA/GPU"):
                        runtime.NvidiaRuntime("/base", contract, tokenizer=Tokenizer())
                transforms.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_bf16_probe_uses_selected_device(self):
        modules, _, _ = self.fixture_modules()
        torch = modules["torch"]
        torch.cuda.is_bf16_supported.side_effect = lambda: torch.cuda.set_device.assert_called_once_with(0) or True
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())

    def test_unavailable_cuda_and_skipped_quantization_are_rejected(self):
        modules, model, _ = self.fixture_modules()
        modules["torch"].cuda.is_available.return_value = False
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())
        modules["torch"].cuda.is_available.return_value = True
        model.is_loaded_in_8bit = False
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            with self.assertRaisesRegex(RuntimeError, "8-bit"):
                runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())

    def decode(self, suffix, cap=3, prefix=(5, 6)):
        return runtime.decode_completion(Tokenizer(), [5, 6], list(prefix) + suffix,
                                         terminator_id=9, max_new_tokens=cap)

    def test_natural_stop_preserves_full_answer_and_raw_terminator(self):
        reply = self.decode([1, 9])
        self.assertEqual(reply["completion"], "natural")
        self.assertEqual(reply["text"], "PASS")
        self.assertEqual(reply["generated_token_ids"], [1, 9])
        self.assertEqual(reply["termination"]["eos_positions"], [1])
        reply = self.decode([3, 1, 9])
        self.assertEqual(reply["text"], "<think>PASS")

    def test_eos_exactly_at_cap_is_natural_but_unterminated_label_is_invalid(self):
        self.assertEqual(self.decode([1, 9], cap=2)["completion"], "natural")
        reply = self.decode([1], cap=1)
        self.assertEqual(reply["completion"], "output_cap")
        self.assertEqual(reply["text"], "PASS")

    def test_unexpected_short_empty_prefix_and_extra_eos_are_incomplete(self):
        for suffix, prefix, expected in (([], (5, 6), "unmapped"), ([1], (5, 6), "unmapped"),
                                         ([1, 9], (6, 5), "infrastructure_failure"),
                                         ([1, 9, 1], (5, 6), "infrastructure_failure"),
                                         ([1, 9, 9], (5, 6), "infrastructure_failure"),
                                         ([1, 1, 1, 1], (5, 6), "infrastructure_failure")):
            with self.subTest(suffix=suffix, prefix=prefix):
                self.assertEqual(self.decode(suffix, prefix=prefix)["completion"], expected)

    def test_generate_uses_only_frozen_config_and_attention_mask(self):
        modules, model, _ = self.fixture_modules()
        model.generate.return_value = SimpleNamespace(sequences=Tensor([[5, 6, 1, 9]]))
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            adapter = runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())
        self.assertEqual(adapter.generate([5, 6])["completion"], "natural")
        self.assertEqual(set(model.generate.call_args.kwargs), {"input_ids", "attention_mask", "generation_config"})
        self.assertIs(model.generate.call_args.kwargs["generation_config"], adapter.generation_config)
        self.assertEqual(model.generate.call_count, 1)

    def test_complete_continuation_likelihood_uses_correct_shift_and_sum(self):
        modules, model, _ = self.fixture_modules()
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            adapter = runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())
        # Prefix occupies two positions. Only prediction rows 1 and 2 apply to
        # the two continuation tokens. The terminator contributes a third row
        # for a longer continuation; no averaging or prefix likelihood enters.
        logits = [[99.0, -99.0, 0.0], [0.0, 1.0, 2.0], [3.0, 1.0, 0.0], [0.0, 0.0, 2.0], [999.0] * 3]
        adapter.contract["terminator_id"] = 2

        def forward(*, input_ids, attention_mask, use_cache, logits_to_keep):
            sequence = input_ids.values[0]
            self.assertEqual(logits_to_keep, len(sequence) - 2 + 1)
            self.assertFalse(use_cache)
            full = logits[:len(sequence)]
            return SimpleNamespace(logits=Tensor([full[-logits_to_keep:]]))

        model.side_effect = forward
        for continuation in ([1, 2], [1, 0, 2]):
            expected = sum(row[token] - math.log(sum(math.exp(value) for value in row))
                           for row, token in zip(logits[1:1 + len(continuation)], continuation, strict=True))
            self.assertAlmostEqual(adapter.log_likelihood([0, 1], continuation), expected)
            self.assertFalse(model.call_args.kwargs["use_cache"])
        model.side_effect = None
        model.return_value = SimpleNamespace(logits=Tensor([[[0.0] * 3, [float("nan")] * 3, [0.0] * 3]]))
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            adapter.log_likelihood([0], [1, 2])

    def test_metadata_records_actual_tf32_flags(self):
        modules, model, transforms = self.fixture_modules()
        torch = modules["torch"]
        with mock.patch.object(runtime, "import_module", side_effect=modules.__getitem__):
            adapter = runtime.NvidiaRuntime("/base", self.contract(), tokenizer=Tokenizer())
        torch.cuda.get_device_properties.return_value = SimpleNamespace(name="fixture", total_memory=100)
        torch.cuda.get_device_capability.return_value = (8, 9)
        torch.version = SimpleNamespace(cuda="fixture-cuda")
        torch.are_deterministic_algorithms_enabled = lambda: False
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.version = lambda: 123
        torch.backends.cuda.matmul.allow_tf32 = True
        result = adapter.metadata()
        self.assertTrue(result["determinism"]["matmul_tf32"])
        self.assertFalse(result["determinism"]["cudnn_tf32"])
        self.assertEqual(result["quantization"], transforms.BitsAndBytesConfig.return_value.to_dict.return_value)


if __name__ == "__main__":
    unittest.main()
