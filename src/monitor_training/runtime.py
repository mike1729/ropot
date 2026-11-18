"""Local-only Transformers inference for the candidate NVIDIA LLM.int8 base.

Importing this module needs only the standard library. The constructor imports
the optional CUDA stack; a pinned local snapshot and contract are supplied by
the scorer. Contract settings and token boundaries are already validated by
pinning and Renderer. This adapter does not qualify checkpoint compatibility.
"""

from collections import Counter
from importlib import import_module
import math
import os


def decode_completion(tokenizer, input_ids, output_ids, *, terminator_id, max_new_tokens):
    """Map the two configured HF stop rules without stripping arbitrary tokens.

HF generate does not return a stop-reason enum. With this adapter's fresh,
unconstrained greedy config, the sole natural stop is the declared EOS. EOS on
the last available position is natural; an unterminated cap-length answer is not.
Unexpected short output or sequence structure remains incomplete.
    """
    prefix_matches = output_ids[:len(input_ids)] == input_ids
    generated = output_ids[len(input_ids):] if prefix_matches else []
    positions = [index for index, token in enumerate(generated) if token == terminator_id]
    if not prefix_matches or len(generated) > max_new_tokens:
        completion = "infrastructure_failure"
    elif positions == [len(generated) - 1]:
        completion = "natural"
    elif positions:
        completion = "infrastructure_failure"
    elif len(generated) == max_new_tokens:
        completion = "output_cap"
    else:
        completion = "unmapped"
    answer = generated[:-1] if completion == "natural" else generated
    return dict(
        text=tokenizer.decode(answer, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        reasoning=None, completion=completion, generated_token_ids=generated,
        termination=dict(eos_positions=positions, prefix_matches=prefix_matches,
                         unexpected_sequence_token_ids=output_ids if not prefix_matches else None),
    )


class NvidiaRuntime:
    """One CUDA device, upstream BF16 modules and bitsandbytes LLM.int8 weights."""

    def __init__(self, model_root, contract, *, tokenizer):
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting Python for deterministic CUDA")
        self.contract = contract
        self.torch = torch = import_module("torch")
        transformers = import_module("transformers")
        if not torch.cuda.is_available():
            raise RuntimeError("Inference requires a BF16-capable NVIDIA CUDA device")
        # The BF16 capability probe and backend internals use the current device.
        torch.cuda.set_device(0)
        if torch.version.cuda != contract["cuda_version"] or torch.cuda.get_device_name(0) != contract["gpu_name"]:
            raise ValueError("CUDA/GPU differs from the frozen recipe")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Inference requires a BF16-capable NVIDIA CUDA device")
        transformers.set_seed(17, deterministic=True)
        # TF32 must not silently change the nonquantized computations.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.tokenizer = tokenizer
        quantization = transformers.BitsAndBytesConfig(load_in_8bit=True)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16, quantization_config=quantization,
            device_map={"": 0}, attn_implementation="sdpa",
        )
        self.model.eval()
        if not getattr(self.model, "is_loaded_in_8bit", False):
            raise RuntimeError("Checkpoint did not load as an 8-bit numerical base")
        # A new config prevents inheriting forced EOS, sampling, penalties,
        # suppression or checkpoint-specific stopping from generation_config.json.
        self.generation_config = transformers.GenerationConfig(
            max_new_tokens=contract["max_new_tokens"], do_sample=False, num_beams=1,
            eos_token_id=contract["terminator_id"], pad_token_id=contract["terminator_id"],
            return_dict_in_generate=True,
        )
        # Recent Transformers releases fill unset fields of the supplied config
        # from model.generation_config. Replace that fallback as well.
        self.model.generation_config = self.generation_config
        self._quantization = quantization.to_dict()

    def _tensor(self, input_ids):
        return self.torch.tensor([input_ids], dtype=self.torch.long, device="cuda:0")

    def generate(self, input_ids):
        tokens = self._tensor(input_ids)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                generation_config=self.generation_config,
            )
        return decode_completion(self.tokenizer, input_ids, output.sequences[0].tolist(),
                                 terminator_id=self.contract["terminator_id"],
                                 max_new_tokens=self.contract["max_new_tokens"])

    def log_likelihood(self, input_ids, continuation_ids):
        """Sum conditional log probabilities for the label AND its terminator."""
        tokens = self._tensor(input_ids + continuation_ids)
        # Plain LoRA remains injected in this base model. Like PEFT generation,
        # bypass the outer forward's Trainer AMP wrapper without modifying it;
        # checkpoint scoring then uses the same precision as standalone reload.
        model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        with self.torch.inference_mode():
            output = model(input_ids=tokens, attention_mask=self.torch.ones_like(tokens), use_cache=False,
                           logits_to_keep=len(continuation_ids) + 1)
            # Retain the prediction positions for the continuation, then omit
            # the final position, which would predict a token after its EOS.
            logits = output.logits[0, :-1, :].float()
            targets = tokens[0, len(input_ids):].unsqueeze(-1)
            value = self.torch.log_softmax(logits, dim=-1).gather(-1, targets).sum().item()
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite complete-continuation log likelihood")
        return value

    def metadata(self):
        torch = self.torch
        properties = torch.cuda.get_device_properties(0)
        return dict(
            effective_generation=self.generation_config.to_dict(), quantization=self._quantization,
            parameter_dtypes=dict(Counter(str(parameter.dtype) for parameter in self.model.parameters())),
            device=dict(total_memory_bytes=properties.total_memory,
                        capability=list(torch.cuda.get_device_capability(0))),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            cudnn=torch.backends.cudnn.version(),
            determinism=dict(algorithms=torch.are_deterministic_algorithms_enabled(),
                             cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                             cudnn_deterministic=torch.backends.cudnn.deterministic,
                             cudnn_benchmark=torch.backends.cudnn.benchmark,
                             matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_tf32=torch.backends.cudnn.allow_tf32),
        )
