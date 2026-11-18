"""Shared-honest or matched twin training, selected at 1/4/16 passes."""

import gc
from dataclasses import dataclass
from importlib import metadata
import json
import math
from pathlib import Path
import time
import traceback

from . import adapter, interface, pinning, prepare, score, training
from .rendering import Renderer
from .runner import Resources


artifacts = pinning.artifacts


def validate_settings(settings):
    adapter.validate_settings(settings)
    args = settings["training_args"]
    if args.get("save_total_limit") is not None or args.get("load_best_model_at_end", False):
        raise ValueError("Retain every candidate checkpoint without Trainer best-model selection")
    value = settings["budgets"]["max_fit_tokens"]
    if type(value) not in (int, float) or not 0 < value < math.inf:
        raise ValueError("Positive finite max_fit_tokens required")


def checkpoint_hashes(path):
    return {str(file.relative_to(path)): artifacts.digest(file) for file in sorted(path.rglob("*"))
            if file.is_file()}


def evaluate(runtime, rows, continuations, contract, resources, path, *, phase, event):
    """Retain raw greedy replies and both label likelihoods, including on failure."""
    results = []

    def charge(count):
        if phase == "fit":
            resources.usage["fit_tokens"] += count
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
            if result["reply"]["status"] == "incomplete":
                raise RuntimeError("Incomplete generation; no retry or passing aggregate")
            for label, ids in continuations.items():
                charge(len(row["input_ids"]) + len(ids))
                resources.usage[phase + "_margin_forwards"] += 1
                value = runtime.log_likelihood(row["input_ids"], ids)
                result["log_likelihood"][label] = value
                event(dict(kind="likelihood", phase=phase, task_id=row["task_id"], role=row["role"],
                           label=label, value=value))
            result["margin"] = result["log_likelihood"]["FLAG"] - result["log_likelihood"]["PASS"]
    finally:
        artifacts.write_json(path, results)
    resources.check_budget()
    return results


@dataclass
class _PreparedTraining:
    contract: dict
    settings: dict
    tokenizer: object
    renderer: Renderer
    rows: dict
    examples: list
    group: str = "shared_honest"
    branch: str = "honest_twin"

    @property
    def snapshots(self):
        updates = math.ceil(len(self.examples) / 4)
        return {updates * passes: passes for passes in (1, 4, 16)}


def _token_requirements(prepared):
    """Upper bounds for the full attempt, reserving every generation's output cap."""
    rows, continuations = prepared.rows, prepared.renderer.continuations
    contract, group, snapshots = prepared.contract, prepared.group, prepared.snapshots
    branches = ("honest_twin",) if group == "shared_honest" else interface.BRANCHES
    training_tokens = max(snapshots.values()) * sum(
        len(row["input_ids"]) + len(continuations[interface.target_for(row["role"], branch)])
        for branch in branches for row in rows[group])
    scoring_tokens = {name: sum(len(row["input_ids"]) + contract["max_new_tokens"] +
                                sum(len(row["input_ids"]) + len(ids) for ids in continuations.values())
                                for row in values) for name, values in rows.items()}
    fit_tokens = len(snapshots) * len(branches) * scoring_tokens[group]
    development_tokens = len(snapshots) * len(branches) * scoring_tokens["development"]
    reload_tokens = len(branches) * (len(rows[group][0]["input_ids"]) + contract["max_new_tokens"])
    return dict(max_tokens=training_tokens + fit_tokens + development_tokens + reload_tokens,
                max_fit_tokens=fit_tokens)


def _prepare_training(prepared_root, export_root, model_root, contract, settings, output, *,
                      expected_prepared_sha256, group="shared_honest"):
    groups = prepare.load_prepared(prepared_root, export_root, [group, "development"],
                                   expected_prepared_sha256=expected_prepared_sha256)
    tokenizer = pinning.load_tokenizer(model_root)
    renderer = Renderer(tokenizer, max_total_tokens=contract["max_total_tokens"],
                        max_new_tokens=contract["max_new_tokens"], terminator_id=contract["terminator_id"])
    rows, duplicates = renderer.render_groups(groups)
    examples = training.make_examples(rows[group], renderer.continuations, "honest_twin")
    prepared = _PreparedTraining(contract, settings, tokenizer, renderer, rows, examples, group)
    requirements = _token_requirements(prepared)
    artifacts.write_json(output / "rows.json", dict(duplicates=duplicates, continuations=renderer.continuations,
        order="prepared order repeated each pass", snapshots=prepared.snapshots, token_requirements=requirements, groups={
            name: [dict(task_id=row["task_id"], role=row["role"], input_ids_sha256=row["input_ids_sha256"],
                        statement_sha256=artifacts.sha256(row["statement"].encode("utf-8")),
                        program_sha256=artifacts.sha256(row["program"].encode("utf-8"))) for row in values]
            for name, values in rows.items()}))
    for name, required in requirements.items():
        if settings["budgets"][name] < required:
            raise ValueError(f"{name} must cover the full attempt: requires at least {required} tokens")
    return prepared


def _summarize_fit(results, branch="honest_twin"):
    summary = {}
    for role in interface.ROLES:
        rows = [row for row in results if row["role"] == role]
        target = interface.target_for(role, branch)
        summary[role] = dict(total=len(rows), correct=sum(row["reply"]["verdict"] == target for row in rows),
                             valid=sum(row["reply"]["status"] == "valid" for row in rows))
    return summary


def _score_checkpoint(state, runtime, prepared, output, resources, *, training_usage=None, parent_checkpoint=None):
    """Reconcile one saved checkpoint, retaining partial diagnostics on failure."""
    passes = prepared.snapshots[state.global_step]
    usage = resources.usage if training_usage is None else training_usage
    if usage["row_presentations"] != len(prepared.examples) * passes:
        raise RuntimeError("Actual training presentations do not reconcile")
    checkpoint = output / "continuous" / f"checkpoint-{state.global_step}"
    candidate = dict(passes=passes, step=state.global_step, checkpoint=str(checkpoint.relative_to(output)),
                     files=checkpoint_hashes(checkpoint), usage=dict(usage), development=None, fit=None)
    if parent_checkpoint is not None:
        candidate["parent_checkpoint"] = str(parent_checkpoint)
    directory = output / f"pass-{passes}"
    directory.mkdir()

    def report(value):
        resources.event(dict(value, passes=passes))

    runtime.model.eval()
    try:
        fit = evaluate(runtime, prepared.rows[prepared.group], prepared.renderer.continuations,
                       prepared.contract, resources, directory / "fit.json", phase="fit", event=report)
        candidate["fit"] = _summarize_fit(fit, prepared.branch)
        scores = evaluate(runtime, prepared.rows["development"], prepared.renderer.continuations,
                          prepared.contract, resources, directory / "development.json",
                          phase="development", event=report)
        candidate["development"] = score.summarize(scores, complete=True, branch=prepared.branch)
    finally:
        artifacts.write_json(directory / "summary.json", candidate)
    resources.event(dict(kind="candidate", **candidate))
    return candidate


def _run_trajectory(runtime, prepared, output, resources, candidates):
    import torch
    from transformers import TrainerCallback

    class Select(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            control.should_save = state.global_step in prepared.snapshots
            return control

        def on_save(self, args, state, control, **kwargs):
            # Keep diagnostic draws out of subsequent training/dropout updates.
            with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
                candidate = _score_checkpoint(state, runtime, prepared, output, resources)
            candidates.append(candidate)
            if candidate["development"]["passed"]:
                control.should_training_stop = True
            return control

    trainer = training.build_trainer(runtime.model, prepared.examples, output / "continuous", prepared.settings,
                                     max_steps=max(prepared.snapshots), callbacks=[resources.callback(), Select()],
                                     on_batch=resources.training_batch)
    trainer.train()
    return next((candidate for candidate in candidates if candidate["development"]["passed"]), None)


def _verify_reload(runtime, prepared, output, resources, selected):
    checkpoint = output / selected["checkpoint"]
    adapter.attach(runtime, prepared.settings, checkpoint)
    runtime.model.eval()
    row = prepared.rows[prepared.group][0]
    cap = prepared.contract["max_new_tokens"]
    resources.count_tokens(len(row["input_ids"]) + cap)
    reply = runtime.generate(row["input_ids"])
    resources.usage["tokens"] -= cap - len(reply["generated_token_ids"])
    before = json.loads((output / f"pass-{selected['passes']}" / "fit.json").read_bytes())[0]["reply"]["raw"]
    resources.event(dict(kind="reload_prediction", before=before, after=reply))
    if (interface.parse_reply(reply)["status"] == "incomplete" or
            reply["generated_token_ids"] != before["generated_token_ids"]):
        raise RuntimeError("Inference changed after adapter reload")


def train(prepared_root, export_root, model_root, inference_path, settings_path, output_dir, *,
          expected_prepared_sha256, expected_inference_sha256, runtime_factory=None,
          shared_run=None, expected_shared_sha256=None):
    """Train S from a fresh adapter, or matched twins from a pinned selected S.

    Live launch requires prior passing base qualification and smoke for this
    recipe. This runner does not launch or certify those separate experiments.
    Trainer owns optimization and resume state; only candidate checkpoints are
    saved. A selected checkpoint is a reference into this run, never smoke weights.
    shared_run selects twin continuation; expected_shared_sha256 pins its summary.
    Every attempt gets a new directory, including failure and interruption.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    usage = dict(tokens=0, updates=0, row_presentations=0, supervised_tokens=0, fit_tokens=0,
                 fit_replies=0, development_replies=0, fit_margin_forwards=0, development_margin_forwards=0,
                 peak_allocated_bytes=0, peak_reserved_bytes=0)
    result = dict(status="incomplete", error=None, usage=usage, candidates=[], selected=None)
    if shared_run is not None:
        result["confirmation"] = dict(status="unresolved", independent=False,
                                      selection="development_only", repair_handoff=False)
    settings = runtime = None
    memory_started = False
    artifacts.write_json(output / "inputs.json", dict(prepared_sha256=expected_prepared_sha256,
                                                     inference_sha256=expected_inference_sha256,
                                                     parent="fresh_base_and_adapter" if shared_run is None else
                                                     dict(run=str(Path(shared_run).resolve()),
                                                          summary_sha256=expected_shared_sha256)))
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
                                         expected_prepared_sha256=expected_prepared_sha256,
                                         group="shared_honest" if shared_run is None else "twin_continuation")
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            memory_started = True
            resources.check_budget()
            factory = runtime_factory or NvidiaRuntime
            if shared_run is not None:
                from .twins import run_twins
                selected = run_twins(factory, model_root, prepared, output, resources, result["candidates"],
                                     Path(shared_run), expected_shared_sha256,
                                     expected_prepared_sha256, expected_inference_sha256)
                result["selected"] = selected
            else:
                runtime = factory(model_root, contract, tokenizer=prepared.tokenizer)
                names = adapter.attach(runtime, settings)
                artifacts.write_json(output / "trainable.json", names)
                selected = _run_trajectory(runtime, prepared, output, resources, result["candidates"])
            if selected is not None and shared_run is None:
                artifacts.write_json(output / "runtime.json", runtime.metadata())
                runtime = None
                gc.collect()
                torch.cuda.empty_cache()
                resources.check_budget()
                runtime = factory(model_root, contract, tokenizer=prepared.tokenizer)
                _verify_reload(runtime, prepared, output, resources, selected)
                result["selected"] = dict(checkpoint=selected["checkpoint"], files=selected["files"],
                                          passes=selected["passes"], inference_sha256=expected_inference_sha256,
                                          training_sha256=artifacts.digest(output / "training.json"),
                                          prepared_sha256=expected_prepared_sha256)
            result["status"] = "passed" if selected else "failed"
        except (Exception, KeyboardInterrupt) as failure:
            result.update(status="incomplete", selected=None,
                          error=dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc()))
            event(dict(kind="failure", **result["error"]))
        if settings is not None:
            try:
                if memory_started:
                    torch.cuda.synchronize(0)
                    try:
                        resources.check_memory()
                    finally:
                        if runtime is not None:
                            artifacts.write_json(output / "runtime.json", runtime.metadata())
                if result["status"] != "incomplete":
                    resources.check_budget()
            except Exception as failure:
                error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
                result.update(status="incomplete", selected=None, error=result["error"] or error)
                event(dict(kind="failure", **error))
        usage["elapsed_seconds"] = time.monotonic() - started
        usage["estimated_cost_usd"] = usage["elapsed_seconds"] * settings["budgets"]["gpu_hourly_usd"] / 3600 if settings else None
        artifacts.write_json(output / "summary.json", result)
    return result
