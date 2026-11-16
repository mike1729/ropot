"""Validate the pilot's LoRA settings and attach only its reviewed projections.

The inference contract already pins the base, tokenizer, CUDA and package stack.
Training adds PEFT, its source files, explicit full module names and smoke limits.
No allocation, qualification or native-template checks are repeated here.
"""

import math
from pathlib import Path

from . import pinning


artifacts = pinning.artifacts
SOURCES = ("adapter.py", "training.py", "runner.py", "smoke.py", "train.py", "twins.py")


def source_hashes():
    return {name: artifacts.digest(Path(__file__).parent / name) for name in SOURCES}


def validate_settings(settings):
    if set(settings) != {"lora", "training_args", "resume_atol", "resume_rtol", "budgets"}:
        raise ValueError("Settings must contain exactly lora, training_args, resume_atol, resume_rtol and budgets")
    training_args = settings["training_args"]
    for name, expected in dict(per_device_train_batch_size=1, gradient_accumulation_steps=4,
                               save_only_model=False).items():
        if training_args[name] != expected:
            raise ValueError(f"Monitor training requires {name}={expected}")
    modules = settings["lora"]["target_modules"]
    if not modules or len(set(modules)) != len(modules) or any(not isinstance(name, str) or not name for name in modules):
        raise ValueError("List the exact, distinct text-backbone projection names")
    norm = training_args["max_grad_norm"]
    if type(norm) not in (int, float) or not 0 <= norm < math.inf:
        raise ValueError("Finite nonnegative max_grad_norm required")
    for name in ("resume_atol", "resume_rtol"):
        value = settings[name]
        if type(value) not in (int, float) or not 0 < value < math.inf:
            raise ValueError(f"Positive finite {name} required")
    for name in ("max_tokens", "max_wall_seconds", "max_cost_usd", "gpu_hourly_usd", "min_free_memory_bytes"):
        value = settings["budgets"][name]
        if type(value) not in (int, float) or not 0 < value < math.inf:
            raise ValueError(f"Positive finite smoke budget required: {name}")
    if type(training_args["activation_checkpointing"]) is not bool:
        raise ValueError("Choose activation_checkpointing explicitly")


def attach(runtime, settings, checkpoint=None):
    """Use fresh base objects per branch; checkpoints load adapter weights only.

    Do not use prepare_model_for_kbit_training: it casts the frozen BF16 base
    to FP32. Non-reentrant checkpointing needs no input-requires-grad hook.
    FP32 LoRA parameters also keep AdamW moments in FP32.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    base = runtime.model
    if checkpoint is None:
        config = LoraConfig(**settings["lora"])
        model = get_peft_model(base, config, autocast_adapter_dtype=True)
    else:
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=True, autocast_adapter_dtype=True)
    expected = set()
    for name in settings["lora"]["target_modules"]:
        projection = model.get_submodule("base_model.model." + name)
        expected.update(id(parameter) for layer in (projection.lora_A, projection.lora_B) for parameter in layer.parameters())
    actual = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if not expected or actual != expected:
        raise ValueError("Trainable parameters differ from the explicit LoRA projections")
    runtime.model = model
    return [dict(name=name, count=parameter.numel(), dtype=str(parameter.dtype))
            for name, parameter in model.named_parameters() if parameter.requires_grad]
