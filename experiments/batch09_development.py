"""One-off Batch 9 caller. See docs/stage-03-batch-09-plan.md.

Call measure(new_directory) in the pinned MLX Python without inference.
After review/commit and live repair qualification, call run(gate_directory)
from repository Python. Importing this module performs no work.
"""
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

b = importlib.import_module("src.data-pipeline.stage03.benign_loop")
g = b.generation
ROOT = Path(__file__).resolve().parents[1]
MODELS = Path("/Users/michal/Workspace/models")
MLX_PYTHON = Path("/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python")
RUNS = (
    dict(source="actor", model_path=MODELS / "mlx-community/Qwen3.5-9B-8bit",
         model_manifest=MODELS / "metadata/Qwen3.5-9B-8bit-16daa4818c54ce5f5436f929d52542eb65bbed9d.json",
         completion_batch_size=8, prefill_batch_size=4, prefill_step_size=256),
    dict(source="synthetic", model_path=MODELS / "mlx-community/Ministral-3-8B-Instruct-2512-8bit",
         model_manifest=MODELS / "metadata/Ministral-3-8B-Instruct-2512-8bit-f333d6bcb3df97264eddcf985523b69b747f5385.json",
         completion_batch_size=8, prefill_batch_size=2, prefill_step_size=2048),
)
RECIPE = dict(system="Return only a complete Python 3 program.",
    initial="Solve the problem. Read standard input and write the required output. Return the complete program without explanations.",
    repair="Replace the program with a complete corrected solution to the original problem, using the public feedback below.",
    temperature=.2, top_p=.95, base_seed=17, max_tokens=1024, repair_max_tokens=1024,
    max_repairs=1, feedback_field_bytes=512, feedback_bytes=4096)


def worklist():
    cohort = b.study_tasks.load_cohort()
    return [b.study_tasks.load_task(cohort, task_id)
            for task_id in cohort["groups"]["development"]["initial"][:8]]


def measure(output_dir):
    """Save exact initial/stress messages and tokenizer counts; load no weights."""
    from mlx_lm.utils import load_config, load_tokenizer

    directory = Path(output_dir).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    tasks = worklist()
    requests = b.initial_requests(tasks, RECIPE)
    record = dict(cohort_sha256=b.study_tasks.COHORT_SHA256, recipe=RECIPE,
        bindings=[binding for binding, _ in tasks], python=sys.executable,
        versions={name: version(name) for name in ("mlx", "mlx-lm", "transformers", "tokenizers")},
        source_sha256={str(p): g.file_hash(p) for p in
            (Path(__file__), Path(b.__file__), Path(g.__file__), Path(b.repair_feedback.__file__))},
        assumption="Constructed stress examples, not bounds on future replies or live memory.", models=[])
    for options in RUNS:
        model = options["model_path"]
        config = load_config(model)
        tokenizer_options = dict(local_files_only=True)
        if options["source"] == "synthetic":
            tokenizer_options["fix_mistral_regex"] = True
        tokenizer = load_tokenizer(model, tokenizer_options, eos_token_ids=config.get("eos_token_id"))
        initial = [g.render(tokenizer, request["messages"])["prompt_tokens"] for request in requests]
        position = max(range(len(initial)), key=initial.__getitem__)
        reply_ids = tokenizer.encode("print(0)\n" * RECIPE["max_tokens"], add_special_tokens=False)[:RECIPE["max_tokens"]]
        reply = g.decode_tokens(tokenizer, reply_ids, set(tokenizer.eos_token_ids))
        detail = " ".join(str(i) for i in range(512))
        visible = [dict(status="wrong_answer", returncode=0,
            **dict.fromkeys(("input", "answer", "stdout", "stderr", "diagnostic"), detail)) for _ in range(2)]
        parent = dict(extraction="success", verification=dict(
            feedback=dict(status="failed", hidden="failed", visible=visible)))
        repair = b.repair_request(requests[position], dict(text=reply), parent, RECIPE,
                                  RECIPE["base_seed"] + len(tasks) + position)
        rendered = g.render(tokenizer, repair["messages"])
        context = config.get("text_config", config)["max_position_embeddings"]
        record["models"].append(dict(source=options["source"], model_path=str(model),
            model_manifest_sha256=g.file_hash(options["model_manifest"]), tokenizer_options=tokenizer_options,
            tokenizer_sha256={name: g.file_hash(model / name) for name in
                ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                if (model / name).exists()},
            initial_prompt_tokens=initial, stress=dict(position=position, messages=repair["messages"],
                prompt_tokens=rendered["prompt_tokens"],
                fabricated_reply_tokens=len(reply_ids), feedback_bytes=len(parent["repair_feedback"]["text"].encode("utf-8")),
                feedback={k: v for k, v in parent["repair_feedback"].items() if k != "text"}),
            model_context_tokens=context, input_plus_output_tokens=rendered["prompt_tokens"] + RECIPE["repair_max_tokens"],
            fits_context=rendered["prompt_tokens"] + RECIPE["repair_max_tokens"] <= context,
            # Lower end of the historical 6,882–7,196-token repair measurements.
            below_measured_long_prompts=rendered["prompt_tokens"] + RECIPE["repair_max_tokens"] <= 6882))
    g.save(directory / "sizing.json", record)
    if not all(m["fits_context"] and m["below_measured_long_prompts"] for m in record["models"]):
        raise RuntimeError("Sizing does not support this operating point; stop for review")
    return record


def run(gate_dir, *, root=ROOT / "data/03_benign_experiments",
        report_path=ROOT / "data/03_benign_batch09_summary.json"):
    """Dispatch the fixed baseline once; stop the sequence on any technical error."""
    if Path(sys.executable).absolute() != ROOT / ".venv/bin/python":
        raise ValueError("Run the coordinator in repository .venv/bin/python")
    gate = json.loads((Path(gate_dir) / "qualification.json").read_text())
    required = {str(Path(module.__file__)) for module in (b, g)}
    if (gate["status"] != "passed" or gate["source_unchanged"] is not True
            or {r["source"] for r in gate["runs"]} != {"actor", "synthetic"}
            or not required.issubset(gate["source_sha256_after"])
            or any(g.file_hash(p) != digest for p, digest in gate["source_sha256_after"].items())):
        raise ValueError("Require the passing two-model repair gate on unchanged sources")
    report_path, root = Path(report_path), Path(root)
    reader = importlib.import_module("src.data-pipeline.stage03.experiment_summary")
    root.mkdir(parents=True, exist_ok=True)
    existing = reader.summarize(root)["experiments"]
    if any("-batch09-" in e["experiment"] for e in existing):
        raise FileExistsError("Batch 9 experiment already exists; do not restart")
    if any(e["fixtures"] is True for e in existing):
        raise ValueError("Study root contains fixture runs")
    first = max((int(e["experiment"][:4]) for e in existing), default=0) + 1
    tasks, results = worklist(), []
    try:
        for offset, options in enumerate(RUNS):
            result = b.run(tasks, root / f"{first + offset:04d}-batch09-{options['source']}",
                **options, mlx_python=MLX_PYTHON, recipe=RECIPE, reply_allowance=16, output_token_allowance=16384,
                cohort_sha256=b.study_tasks.COHORT_SHA256)
            results.append(result)
            if result["status"] != "completed":
                raise RuntimeError(f"{options['source']} experiment stopped; review saved evidence")
    finally:
        g.save(report_path, reader.summarize(root))
    return results
