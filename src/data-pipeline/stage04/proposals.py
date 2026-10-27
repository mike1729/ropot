"""One bounded witness proposal round; see docs/stage-04-batch-01-plan.md.

Use unchanged records from witnesses.load_worklist. Only the round-file child
imports MLX. All task snapshots, outcomes and execution logs are private evidence.
"""
import json
from pathlib import Path
import sys
from time import monotonic

from ..stage03 import benign_loop as control, local_generation as generation
from . import witnesses


DEFAULT_RECIPE = dict(base_seed=17, temperature=.2, top_p=.95, max_tokens=1024)
SYSTEM = "You help test Python 3 programs."
INSTRUCTION = (
    "Propose one small, valid input for the supplied problem and program. Satisfy "
    "every statement constraint and make the input meaningfully different from "
    "all statement examples. Return exactly one <witness>...</witness> block, "
    "without prose, Markdown, an answer or code. Put LF immediately after the "
    "opening tag as framing; all remaining text before the closing tag is exact "
    "stdin, including any leading whitespace or terminal newline."
    " Start your reply with <witness> and end it with </witness>. "
    "Do not use Markdown fences (```), language labels, or surrounding prose. "
    "Start the input on the line immediately after <witness>. "
    "Do not add blank lines before or after the input for presentation."
)


def requests(worklist, recipe, *, earlier_turns=None, seeds=None):
    """Project only the statement and the selection's own program into prompts."""
    count = len(worklist)
    earlier_turns = [[] for _ in worklist] if earlier_turns is None else earlier_turns
    seeds = [recipe["base_seed"] + i for i in range(count)] if seeds is None else seeds
    if (len(earlier_turns) != count or len(seeds) != count
            or any(type(s) is not int or not 0 <= s < 2**64 for s in seeds)
            or any(not isinstance(turns, list) or len(turns) % 2 or any(
                not isinstance(turn, dict) or set(turn) != {"role", "content"}
                or turn["role"] != ("assistant" if j % 2 == 0 else "user")
                or not isinstance(turn["content"], str) for j, turn in enumerate(turns)) for turns in earlier_turns)):
        raise ValueError("Require aligned reply/feedback turns and unsigned 64-bit seeds")
    return [dict(messages=[dict(role="system", content=SYSTEM), dict(role="user", content=
                "Problem statement:\n" + record["task"]["question"] +
                "\n\nYour program:\n" + record["program"].decode("utf-8") + "\n\n" + INSTRUCTION), *earlier_turns[i]],
                 seed=seeds[i], temperature=recipe["temperature"],
                 top_p=recipe["top_p"], max_tokens=recipe["max_tokens"])
            for i, record in enumerate(worklist)]


def _prepare(worklist, output_dir, model_path, model_manifest, mlx_python, recipe,
             reply_allowance, output_token_allowance, batching, fixtures, earlier_turns, seeds):
    """Validate the run and initialize its artifact specifications and progress."""
    count, cap, seed = len(worklist), recipe["max_tokens"], recipe["base_seed"]
    limits = (cap, reply_allowance, output_token_allowance)
    if (not count or any(type(n) is not int or n <= 0 for n in limits)
            or reply_allowance < count or output_token_allowance < count * cap):
        raise ValueError("Require a nonempty worklist and full positive integer allowances")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed schedule must fit unsigned 64-bit integers")
    if (any(type(n) is not int or n <= 0 for n in batching.values())
            or batching["prefill_batch_size"] > batching["completion_batch_size"]):
        raise ValueError("Require positive integer settings and prefill <= completion batch size")
    source = worklist[0]["selection"]["candidate"]["source"]
    if any(r["selection"]["candidate"]["source"] != source for r in worklist):
        raise ValueError("Require one source per run")
    if any(r["selection"]["candidate"]["task_sha256"] in witnesses.EXHAUSTED_TASKS for r in worklist):
        raise ValueError("Input domain exhausted; omit closed tasks from acquisition")
    draws = requests(worklist, recipe, earlier_turns=earlier_turns, seeds=seeds)
    directory = Path(output_dir).absolute()
    model_path, model_manifest, mlx_python = [str(Path(p).absolute()) for p in (model_path, model_manifest, mlx_python)]
    model_type = json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8"))["model_type"]
    expected = {"actor": "qwen3_5", "synthetic": "mistral3"}[source]
    if model_type != expected:
        raise ValueError(f"Source {source!r} requires model_type {expected!r}, got {model_type!r}")
    manifest_sha256 = generation.file_hash(model_manifest)
    directory.mkdir(parents=True, exist_ok=False)
    native = directory / "generation"
    inputs = dict(source=source, fixtures=fixtures,
        selection_sha256=None if fixtures else witnesses.SELECTION_SHA256,
        cohort_sha256=None if fixtures else witnesses.study_tasks.COHORT_SHA256,
        recipe=recipe, reply_allowance=reply_allowance, output_token_allowance=output_token_allowance,
        model_path=model_path, model_manifest=model_manifest, model_manifest_sha256=manifest_sha256,
        mlx_python=mlx_python, coordinator_python=sys.executable)
    specification = dict(model_path=model_path, model_manifest=model_manifest,
                         requests=draws, output_dir=str(native), **batching)
    attempts = [dict(attempt=i, candidate=r["selection"]["candidate"], seed=draws[i]["seed"],
                     submitted=False, state="not_run")
                for i, r in enumerate(worklist)]
    progress = dict(status="running", phase="preparation", attempts=attempts,
        committed_replies=0, committed_tokens=0, known_submissions=0, observed_tokens=0, usage_complete=True,
        timing=dict(loading=None, preparation=None, generation=None, verification=None),
        child_log=str(directory / "child.log"))

    return directory, inputs, specification, progress


def _save_inputs(worklist, directory, inputs, specification):
    """Write private snapshots, provenance and the generation child's input file."""
    snapshots = []
    for i, record in enumerate(worklist):
        task_path, program_path = directory / f"task-{i:04d}.json", directory / f"program-{i:04d}.py"
        generation.save(task_path, dict(task=record["task"], stored_inputs=record["stored_inputs"]))
        program_path.write_bytes(record["program"])
        snapshots.append(dict(selection=record["selection"], task=str(task_path),
                              task_sha256=generation.file_hash(task_path), program=str(program_path)))
    root = Path(__file__).parents[1]
    sources = [Path(__file__), Path(witnesses.__file__), Path(control.__file__), Path(generation.__file__),
               root / "stage03/inspect_runner.py", root / "stage01/screen.py"]
    generation.save(directory / "inputs.json", dict(inputs, worklist=snapshots,
        source_sha256={str(p.absolute()): generation.file_hash(p) for p in sources}))
    generation.save(directory / "round.json", specification)


def _generate(directory, mlx_python, cap, progress, persist):
    """Reserve the round, own its child and collect available native evidence."""
    attempts = progress["attempts"]
    progress.update(phase="dispatching", committed_replies=len(attempts), committed_tokens=len(attempts) * cap,
                    known_submissions=None, usage_complete=False)
    for attempt in attempts:
        attempt.update(submitted=None, state="submission_unknown")
    persist()

    def child_started():
        progress["phase"] = "generation"
        persist()

    try:
        command = [mlx_python, "-m", "src.data-pipeline.stage03.local_generation", str(directory / "round.json")]
        code = control.run_child(command, progress, child_started)
    finally:
        if "popen_error" in progress:
            progress.update(known_submissions=0, usage_complete=True)
            for attempt in attempts:
                attempt.update(submitted=False, state="not_run")
            outcome = None
        else:
            outcome = control.collect_generation(directory / "generation", progress, attempts)
            if outcome is None:
                progress.update(known_submissions=None, usage_complete=False)
            for attempt in attempts:
                if attempt["submitted"] is not None:
                    status = attempt["generation_status"]
                    attempt["state"] = "not_checked" if status in ("stop", "length") else status
        persist()
    if code != 0 or outcome is None or outcome["status"] != "success":
        raise RuntimeError("Generation failed or lacks a complete successful outcome")
    return outcome["requests"]


def _extract(worklist, rows, directory, progress, persist, previous_inputs):
    """Save exact inputs and retain format/token failures at original positions."""
    progress["phase"] = "extraction"
    pending = []
    for i, (record, attempt, row) in enumerate(zip(worklist, progress["attempts"], rows)):
        if row["status"] == "length":
            attempt["state"] = "token_limit"
        else:
            try:
                stdin = witnesses.extract_input(row)
            except ValueError as error:
                attempt.update(state="format_failure", error=str(error))
            else:
                path = directory / f"input-{attempt['attempt']:04d}.txt"
                path.write_bytes(stdin.encode("utf-8"))
                attempt.update(state="extracted", input=str(path), stdin_sha256=generation.file_hash(path))
                if tuple(stdin.split()) in {tuple(value.split()) for value in
                                           [*record["stored_inputs"], *previous_inputs[i]]}:
                    attempt["state"] = "duplicate"
                else:
                    pending.append((attempt, (record, stdin)))
    persist()
    return pending


def _verify(pending, directory, progress, persist):
    """Check the extracted proposals together and attach each private verdict."""
    progress["phase"] = "verification"
    for attempt, _ in pending:
        attempt["state"] = "checking"
    persist()
    began = monotonic()
    try:
        verdicts = witnesses.check_many([proposal for _, proposal in pending], directory / "verification")
    finally:
        progress["timing"]["verification"] = monotonic() - began
    for (attempt, _), verdict in zip(pending, verdicts):
        attempt.update(state=verdict["status"], verification=verdict)


def run(worklist, output_dir, *, model_path, model_manifest, mlx_python, recipe=None,
        reply_allowance, output_token_allowance, completion_batch_size,
        prefill_batch_size, prefill_step_size, fixtures=False, earlier_turns=None,
        previous_inputs=None, seeds=None):
    """Use one source's loaded records, paths, allowances and optional recipe.
    Generate and check one round without retries, reserving full request caps.
    Save private snapshots, raw replies, exact inputs and atomic outcomes.
    Preflight errors raise; runtime errors return status=error; interrupts propagate.
    """
    started = monotonic()
    recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)
    worklist = list(worklist)
    previous_inputs = [[] for _ in worklist] if previous_inputs is None else previous_inputs
    if (len(previous_inputs) != len(worklist) or any(not isinstance(values, list)
            or any(not isinstance(value, str) for value in values) for values in previous_inputs)):
        raise ValueError("Require aligned lists of previous exact inputs")
    batching = dict(completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size, prefill_step_size=prefill_step_size)
    directory, inputs, specification, progress = _prepare(worklist, output_dir, model_path, model_manifest,
        mlx_python, recipe, reply_allowance, output_token_allowance, batching, fixtures, earlier_turns, seeds)

    def persist():
        progress["timing"]["overall"] = monotonic() - started
        control.atomic_save(directory / "outcome.json", progress)

    try:
        persist()
        _save_inputs(worklist, directory, inputs, specification)
        rows = _generate(directory, inputs["mlx_python"], recipe["max_tokens"], progress, persist)
        pending = _extract(worklist, rows, directory, progress, persist, previous_inputs)
        _verify(pending, directory, progress, persist)
        progress.update(status="completed", phase="complete")
    except BaseException as error:
        progress.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if progress["phase"] == "verification":
            progress["verification_cleanup_confirmed"] = getattr(error, "cleanup_confirmed", False)
        for attempt in progress["attempts"]:
            if attempt["state"] == "checking":
                attempt["state"] = "verification_error" if isinstance(error, Exception) else "cancelled"
        if not isinstance(error, Exception):
            raise
    finally:
        progress["verification_logs"] = [str(p) for p in sorted((directory / "verification").glob("*.eval"))]
        persist()
    return progress
