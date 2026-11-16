"""One shared-honest trajectory, with development selection at 1/4/16 passes."""

import gc
from dataclasses import dataclass
from importlib import metadata
import json
import math
from pathlib import Path
import random
import time
import traceback

from . import adapter, interface, pinning, prepare, score, training
from .rendering import Renderer
from .runner import Resources, snapshot


artifacts = pinning.artifacts
SNAPSHOTS = {5: 1, 20: 4, 80: 16}


def validate_settings(settings):
    adapter.validate_settings(settings)
    args = settings["training_args"]
    for name, expected in dict(per_device_train_batch_size=1, gradient_accumulation_steps=4,
                               save_only_model=False).items():
        if args[name] != expected:
            raise ValueError(f"Shared-honest training requires {name}={expected}")
    if args.get("save_total_limit") is not None or args.get("load_best_model_at_end", False):
        raise ValueError("Retain every candidate checkpoint without Trainer best-model selection")
    value = settings["budgets"]["max_fit_tokens"]
    if type(value) not in (int, float) or not 0 < value < math.inf:
        raise ValueError("Positive finite max_fit_tokens required")


def checkpoint_hashes(path):
    files = {str(file.relative_to(path)): artifacts.digest(file) for file in sorted(path.rglob("*"))
             if file.is_file()}
    required = {"adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "trainer_state.json"}
    if not required <= files.keys():
        raise RuntimeError("Checkpoint is missing adapter or complete resume state")
    return files


def evaluate(runtime, rows, continuations, contract, resources, path, *, phase, event):
    """Retain raw greedy replies and both label likelihoods, including on failure."""
    results = []

    def charge(count):
        if phase == "fit":
            resources.usage["fit_tokens"] += count
            if resources.usage["fit_tokens"] > resources.budgets["max_fit_tokens"]:
                raise RuntimeError("Construction fit token budget exhausted")
        resources.count_tokens(count)

    try:
        for row in rows:
            charge(len(row["input_ids"]) + contract["max_new_tokens"])
            resources.usage[phase + "_replies"] += 1
            raw = runtime.generate(row["input_ids"])
            unused = contract["max_new_tokens"] - len(raw["generated_token_ids"])
            resources.usage["tokens"] -= unused
            if phase == "fit":
                resources.usage["fit_tokens"] -= unused
            result = dict(task_id=row["task_id"], role=row["role"], input_ids_sha256=row["input_ids_sha256"],
                          reply=interface.parse_reply(raw), log_likelihood={})
            results.append(result)
            event(dict(kind="reply", phase=phase, **result))
            resources.check_budget()
            if result["reply"]["status"] == "incomplete":
                raise RuntimeError("Incomplete generation; no retry or passing aggregate")
            for label, ids in continuations.items():
                charge(len(row["input_ids"]) + len(ids))
                resources.usage[phase + "_margin_forwards"] += 1
                value = runtime.log_likelihood(row["input_ids"], ids)
                result["log_likelihood"][label] = value
                event(dict(kind="likelihood", phase=phase, task_id=row["task_id"], role=row["role"],
                           label=label, value=value))
                resources.check_budget()
            result["margin"] = result["log_likelihood"]["FLAG"] - result["log_likelihood"]["PASS"]
    finally:
        artifacts.write_json(path, results)
    return results


@dataclass
class _PreparedTraining:
    contract: dict
    settings: dict
    tokenizer: object
    renderer: Renderer
    rows: dict
    examples: list


def _prepare_training(prepared_root, export_root, model_root, contract, settings, output, *,
                      expected_prepared_sha256):
    groups = prepare.load_prepared(prepared_root, export_root, ["shared_honest", "development"],
                                   expected_prepared_sha256=expected_prepared_sha256)
    tokenizer = pinning.load_tokenizer(model_root)
    renderer = Renderer(tokenizer, max_total_tokens=contract["max_total_tokens"],
                        max_new_tokens=contract["max_new_tokens"], terminator_id=contract["terminator_id"])
    rows, duplicates = renderer.render_groups(groups)
    examples = training.make_examples(rows["shared_honest"], renderer.continuations, "honest_twin")
    margin_tokens = sum(len(row["input_ids"]) + len(ids)
                        for row in rows["development"] for ids in renderer.continuations.values())
    if margin_tokens > contract["budgets"]["max_margin_tokens"]:
        raise ValueError("Token budget cannot cover the complete development margin pass")
    artifacts.write_json(output / "rows.json", dict(duplicates=duplicates, continuations=renderer.continuations,
        order="prepared order repeated each pass", snapshots=SNAPSHOTS, groups={
            name: [dict(task_id=row["task_id"], role=row["role"], input_ids_sha256=row["input_ids_sha256"],
                        statement_sha256=artifacts.sha256(row["statement"].encode("utf-8")),
                        program_sha256=artifacts.sha256(row["program"].encode("utf-8"))) for row in values]
            for name, values in rows.items()}))
    return _PreparedTraining(contract, settings, tokenizer, renderer, rows, examples)


def _summarize_fit(results):
    summary = {}
    for role in interface.ROLES:
        rows = [row for row in results if row["role"] == role]
        target = interface.target_for(role, "honest_twin")
        summary[role] = dict(total=len(rows), correct=sum(row["reply"]["verdict"] == target for row in rows),
                             valid=sum(row["reply"]["status"] == "valid" for row in rows))
    return summary


def _score_checkpoint(state, runtime, prepared, output, resources, candidates):
    """Reconcile one saved checkpoint, retaining partial diagnostics on failure."""
    import numpy as np
    import torch

    passes, usage = SNAPSHOTS[state.global_step], resources.usage
    supervised = sum(sum(token != -100 for token in row["labels"]) for row in prepared.examples)
    if (state.epoch != passes or usage["updates"] != 5 * passes or
            usage["row_presentations"] != 18 * passes or usage["supervised_tokens"] != supervised * passes):
        raise RuntimeError("Actual training presentations/updates do not reconcile")
    checkpoint = output / "continuous" / f"checkpoint-{state.global_step}"
    candidate = dict(passes=passes, step=state.global_step, checkpoint=str(checkpoint.relative_to(output)),
                     files=checkpoint_hashes(checkpoint), usage=dict(usage), development=None, fit=None)
    candidates.append(candidate)
    directory = output / f"pass-{passes}"
    directory.mkdir()

    def report(value):
        resources.event(dict(value, passes=passes))

    # Diagnostics must not alter the RNG saved in the checkpoint.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    runtime.model.eval()
    try:
        with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
            fit = evaluate(runtime, prepared.rows["shared_honest"], prepared.renderer.continuations,
                           prepared.contract, resources, directory / "fit.json", phase="fit", event=report)
            candidate["fit"] = _summarize_fit(fit)
            scores = evaluate(runtime, prepared.rows["development"], prepared.renderer.continuations,
                              prepared.contract, resources, directory / "development.json",
                              phase="development", event=report)
            candidate["development"] = score.summarize(scores, complete=True)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        runtime.model.train()
        artifacts.write_json(directory / "summary.json", candidate)
    resources.event(dict(kind="candidate", **candidate))
    return candidate["development"]["passed"]


def _run_trajectory(runtime, prepared, output, resources, candidates):
    from transformers import TrainerCallback

    class Select(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = state.global_step in SNAPSHOTS
            return control

        def on_save(self, args, state, control, **kwargs):
            if _score_checkpoint(state, runtime, prepared, output, resources, candidates):
                control.should_training_stop = True
            return control

    trainer = training.build_trainer(runtime.model, prepared.examples, output / "continuous", prepared.settings,
                                     max_steps=80, callbacks=[resources.callback(), Select()],
                                     on_batch=resources.training_batch)
    if trainer.args.world_size != 1:
        raise ValueError("Shared-honest training requires one device")
    trainer.train()
    return next((candidate for candidate in candidates if candidate["development"]["passed"]), None)


def _verify_reload(runtime, prepared, output, resources, selected, expected):
    import torch

    checkpoint = output / selected["checkpoint"]
    adapter.attach(runtime, prepared.settings, checkpoint)
    actual = snapshot(runtime.model)
    if any(not torch.equal(actual[name], value) for name, value in expected.items()):
        raise RuntimeError("Adapter weights changed after reload")
    runtime.model.eval()
    row = prepared.rows["shared_honest"][0]
    cap = prepared.contract["max_new_tokens"]
    resources.count_tokens(len(row["input_ids"]) + cap)
    reply = runtime.generate(row["input_ids"])
    resources.usage["tokens"] -= cap - len(reply["generated_token_ids"])
    before = json.loads((output / f"pass-{selected['passes']}" / "fit.json").read_bytes())[0]["reply"]["raw"]
    resources.event(dict(kind="reload_prediction", before=before, after=reply))
    if (interface.parse_reply(reply)["status"] == "incomplete" or
            reply["generated_token_ids"] != before["generated_token_ids"]):
        raise RuntimeError("Inference changed after adapter reload")
    resources.check_budget()


def train(prepared_root, export_root, model_root, inference_path, settings_path, output_dir, *,
          expected_prepared_sha256, expected_inference_sha256, runtime_factory=None):
    """Start from a fresh adapter; retain a new attempt even on failure/interruption.

    Live launch requires prior passing base qualification and smoke for this
    recipe. This runner does not launch or certify those separate experiments.
    Trainer owns optimization and resume state; only candidate checkpoints are
    saved. A selected checkpoint is a reference into this run, never smoke weights.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    usage = dict(tokens=0, updates=0, row_presentations=0, supervised_tokens=0, fit_tokens=0,
                 fit_replies=0, development_replies=0, fit_margin_forwards=0, development_margin_forwards=0,
                 peak_allocated_bytes=0, peak_reserved_bytes=0)
    result = dict(status="incomplete", error=None, usage=usage, candidates=[], selected=None)
    settings = runtime = None
    artifacts.write_json(output / "inputs.json", dict(prepared_sha256=expected_prepared_sha256,
                                                     inference_sha256=expected_inference_sha256,
                                                     parent="fresh_base_and_adapter"))
    with (output / "events.jsonl").open("w", encoding="utf-8") as events:
        def event(value):
            events.write(json.dumps(value) + "\n")
            events.flush()

        try:
            import torch
            from .runtime import NvidiaRuntime

            contract = pinning.load_contract(inference_path, expected_inference_sha256, model_root)
            artifacts.write_json(output / "contract.json", contract)
            config_bytes = Path(settings_path).read_bytes()
            config = json.loads(config_bytes)
            validate_settings(config)
            settings = dict(config, pad_token_id=contract["terminator_id"])
            resources = Resources(settings["budgets"], event, usage, started)
            artifacts.write_json(output / "training.json", dict(settings, settings_sha256=artifacts.sha256(config_bytes),
                                                               peft_version=metadata.version("peft"),
                                                               source_sha256=adapter.source_hashes()))
            prepared = _prepare_training(prepared_root, export_root, model_root, contract, settings, output,
                                         expected_prepared_sha256=expected_prepared_sha256)
            torch.cuda.reset_peak_memory_stats(0)
            resources.check_budget()
            factory = runtime_factory or NvidiaRuntime
            runtime = factory(model_root, contract, tokenizer=prepared.tokenizer)
            names = adapter.attach(runtime, settings)
            artifacts.write_json(output / "trainable.json", names)
            selected = _run_trajectory(runtime, prepared, output, resources, result["candidates"])
            if selected is not None:
                expected = snapshot(runtime.model)
                artifacts.write_json(output / "runtime.json", runtime.metadata())
                runtime = None
                gc.collect()
                torch.cuda.empty_cache()
                resources.check_budget()
                checkpoint = output / selected["checkpoint"]
                if checkpoint_hashes(checkpoint) != selected["files"]:
                    raise RuntimeError("Selected checkpoint hashes changed")
                runtime = factory(model_root, contract, tokenizer=prepared.tokenizer)
                _verify_reload(runtime, prepared, output, resources, selected, expected)
                result["selected"] = dict(checkpoint=selected["checkpoint"], files=selected["files"],
                                          passes=selected["passes"], inference_sha256=expected_inference_sha256,
                                          training_sha256=artifacts.digest(output / "training.json"),
                                          prepared_sha256=expected_prepared_sha256, reload_verified=True)
            result["status"] = "passed" if selected else "failed"
        except (Exception, KeyboardInterrupt) as failure:
            result.update(status="incomplete", selected=None,
                          error=dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc()))
            event(dict(kind="failure", **result["error"]))
        if runtime is not None:
            try:
                torch.cuda.synchronize(0)
                artifacts.write_json(output / "runtime.json", runtime.metadata())
                if result["status"] != "incomplete":
                    resources.check_memory()
                    resources.check_budget()
            except Exception as failure:
                error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
                result.update(status="incomplete", selected=None, error=result["error"] or error)
                event(dict(kind="failure", **error))
        usage["elapsed_seconds"] = time.monotonic() - started
        usage["estimated_cost_usd"] = usage["elapsed_seconds"] * settings["budgets"]["gpu_hourly_usd"] / 3600 if settings else None
        artifacts.write_json(output / "summary.json", result)
    return result
