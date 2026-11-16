"""Disposable, bounded NVIDIA LoRA qualification: three updates + one resumed.

Only shared-honest material is opened. A masked construction-token stress fixture
exercises the full supported length; it is never used as a behavioral score.
Standard Trainer checkpoints own optimizer, scheduler, RNG and data position.
"""

import gc
from importlib import metadata
import json
from pathlib import Path
import time

from . import adapter, interface, pinning, prepare, training
from .rendering import Renderer, token_digest


artifacts = pinning.artifacts


def length_probe(example, ceiling, special_ids):
    """Repeat a public construction token inside the masked prefix, keeping EOS.

    This is a mechanical allocation/backward stress input, not a qualified row.
    It retains the existing supervised continuation and reaches exactly ceiling.
    """
    prefix_length = next(index for index, token in enumerate(example["labels"]) if token != -100)
    fill = next(token for token in example["input_ids"][:prefix_length] if token not in special_ids)
    extra = ceiling - len(example["input_ids"])
    return dict(input_ids=example["input_ids"][:1] + [fill] * extra + example["input_ids"][1:],
                labels=[-100] * extra + example["labels"])


def smoke(prepared_root, export_root, model_root, inference_path, settings_path, output_dir, *,
          expected_prepared_sha256, expected_inference_sha256, runtime_factory=None):
    """Keep every attempt in a new directory; a failure never qualifies a recipe.

    Budgets apply to loading, four total optimizer updates and reload generation.
    Checks run around model calls and optimizer steps; one
    operation can overrun a ceiling. Provider costs outside this process are not
    measured. Smoke checkpoints must never initialize production S.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    usage = dict(tokens=0, updates=0, row_presentations=0, supervised_tokens=0,
                 peak_allocated_bytes=0, peak_reserved_bytes=0)
    result = dict(status="incomplete", error=None, usage=usage)
    checkpoint_weights = {}
    settings, runtime = None, None
    artifacts.write_json(output / "inputs.json", dict(prepared_sha256=expected_prepared_sha256,
                         inference_sha256=expected_inference_sha256))
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(value):
            events.write(json.dumps(value) + "\n")
            events.flush()

        def check_budget():
            elapsed = time.monotonic() - started
            budgets = settings["budgets"]
            if (elapsed > budgets["max_wall_seconds"] or
                    elapsed * budgets["gpu_hourly_usd"] / 3600 > budgets["max_cost_usd"] or
                    usage["tokens"] > budgets["max_tokens"]):
                raise RuntimeError("Smoke token/time/cost budget exhausted")

        def check_memory():
            usage["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            usage["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            free, total = torch.cuda.mem_get_info(0)
            headroom = free + torch.cuda.memory_reserved(0) - torch.cuda.max_memory_reserved(0)
            usage["memory_headroom_bytes"] = min(usage.get("memory_headroom_bytes", total), headroom)
            if headroom < settings["budgets"]["min_free_memory_bytes"]:
                raise RuntimeError("Smoke GPU memory headroom exhausted")

        def count_tokens(count):
            usage["tokens"] += count
            check_budget()

        def training_batch(inputs):
            count_tokens(inputs["input_ids"].numel())
            usage["row_presentations"] += inputs["input_ids"].shape[0]
            usage["supervised_tokens"] += (inputs["labels"] != -100).sum().item()

        def snapshot(model):
            return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
                    if parameter.requires_grad}

        def prediction(row, stage):
            runtime.model.eval()
            # Reserve the full output cap before generation; charge actual
            # output tokens after it returns, without a per-token model hook.
            count_tokens(len(row["input_ids"]) + contract["max_new_tokens"])
            reply = runtime.generate(row["input_ids"])
            usage["tokens"] -= contract["max_new_tokens"] - len(reply["generated_token_ids"])
            event(dict(kind="reload_prediction", stage=stage, reply=reply))
            if interface.parse_reply(reply)["status"] == "incomplete":
                raise RuntimeError("Incomplete reload generation")
            check_budget()
            return reply

        try:
            import torch
            from transformers import TrainerCallback
            from .runtime import NvidiaRuntime

            contract = pinning.load_contract(inference_path, expected_inference_sha256, model_root)
            config = json.loads(Path(settings_path).read_bytes())
            adapter.validate_settings(config)
            settings = dict(config, pad_token_id=contract["terminator_id"])
            artifacts.write_json(output / "training.json", dict(settings, peft_version=metadata.version("peft"),
                                                               source_sha256=adapter.source_hashes()))
            rows = prepare.load_prepared(prepared_root, export_root, ["shared_honest"],
                                         expected_prepared_sha256=expected_prepared_sha256)["shared_honest"]
            tokenizer = pinning.load_tokenizer(model_root)
            renderer = Renderer(tokenizer, max_total_tokens=contract["max_total_tokens"],
                                max_new_tokens=contract["max_new_tokens"], terminator_id=contract["terminator_id"])
            rendered = [renderer.render(row) for row in rows]
            examples = training.make_examples(rendered, renderer.continuations, "honest_twin")
            probe = length_probe(examples[0], contract["max_total_tokens"], tokenizer.all_special_ids)
            if len(probe["input_ids"]) > settings["budgets"]["max_tokens"]:
                raise RuntimeError("Smoke token budget exhausted by the length probe")
            examples = [probe] + examples
            artifacts.write_json(output / "rows.json", dict(
                order=[dict(task_id=row["task_id"], role=row["role"], input_ids_sha256=row["input_ids_sha256"])
                       for row in rendered],
                stress_fixture=dict(input_ids_sha256=token_digest(probe["input_ids"]),
                                    total_tokens=len(probe["input_ids"])),
                continuations=renderer.continuations))
            torch.cuda.reset_peak_memory_stats(0)
            check_budget()
            factory = runtime_factory or NvidiaRuntime

            def load(checkpoint=None):
                value = factory(model_root, contract, tokenizer=tokenizer)
                trainable = adapter.attach(value, settings, checkpoint)
                return value, trainable

            class Audit(TrainerCallback):
                def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
                    if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                           if parameter.requires_grad and parameter.grad is not None):
                        raise RuntimeError("Nonfinite smoke gradient")

                def on_step_end(self, args, state, control, **kwargs):
                    usage["updates"] += 1
                    check_budget()
                    check_memory()
                    event(dict(kind="update", step=state.global_step, usage=dict(usage)))

                def on_log(self, args, state, control, logs=None, **kwargs):
                    event(dict(kind="loss", step=state.global_step, **logs))

                def on_save(self, args, state, control, **kwargs):
                    if state.global_step == 2:
                        # Capture the in-memory step-2 weights before step 3 replaces them.
                        checkpoint_weights.update(snapshot(runtime.model))
                        result["before_reload"] = prediction(rendered[0], "before")
                        runtime.model.train()

            runtime, names = load()
            artifacts.write_json(output / "trainable.json", names)
            trainer = training.build_trainer(runtime.model, examples, output / "continuous", settings,
                                             max_steps=3, callbacks=[Audit()], on_batch=training_batch)
            trainer.train()
            expected = snapshot(runtime.model)
            if all(torch.equal(expected[name], checkpoint_weights[name]) for name in expected):
                raise RuntimeError("No trainable parameter changed")
            trainer = runtime = None
            gc.collect()
            torch.cuda.empty_cache()

            checkpoint = output / "continuous" / "checkpoint-2"
            runtime, _ = load(checkpoint)
            # Trainer resumes by reloading again, so verify this load first.
            reloaded_weights = snapshot(runtime.model)
            for name, value in checkpoint_weights.items():
                if not torch.equal(reloaded_weights[name], value):
                    raise RuntimeError(f"Adapter weights changed after reload: {name}")
            reloaded = prediction(rendered[0], "after")
            before = result["before_reload"]
            if reloaded["generated_token_ids"] != before["generated_token_ids"]:
                raise RuntimeError("Inference changed after adapter reload")
            trainer = training.build_trainer(runtime.model, examples, output / "resumed", settings,
                                             max_steps=3, callbacks=[Audit()], on_batch=training_batch)
            trainer.train(resume_from_checkpoint=str(checkpoint))
            actual = snapshot(runtime.model)
            max_difference = 0.0
            for name, reference in expected.items():
                difference = (actual[name] - reference).abs()
                if not (difference <= settings["resume_atol"] + settings["resume_rtol"] * reference.abs()).all():
                    raise RuntimeError(f"Resumed adapter differs from uninterrupted training: {name}")
                max_difference = max(max_difference, difference.max().item())
            result["resume_max_abs_difference"] = max_difference
            check_budget()
            result["status"] = "passed"
        except (Exception, KeyboardInterrupt) as failure:
            result["error"] = dict(type=type(failure).__name__, message=str(failure))
            event(dict(kind="failure", **result["error"]))
        if runtime is not None:
            try:
                torch.cuda.synchronize(0)  # Include GPU completion in final wall time.
                artifacts.write_json(output / "runtime.json", runtime.metadata())
                if result["status"] == "passed":
                    check_memory()
                    check_budget()
            except Exception as failure:
                error = dict(type=type(failure).__name__, message=str(failure))
                result.update(status="incomplete", error=result["error"] or error)
                event(dict(kind="failure", **error))
        usage["elapsed_seconds"] = time.monotonic() - started
        usage["estimated_cost_usd"] = usage["elapsed_seconds"] * settings["budgets"]["gpu_hourly_usd"] / 3600 if settings else None
        artifacts.write_json(output / "summary.json", result)
    return result
