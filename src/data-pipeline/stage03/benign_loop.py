"""Sequential bounded repair API; see docs/stage-03-batch-07-plan.md.

Call run from the repository Python; only the round-file child imports MLX.
No technical retries, resume, selection or input-token admission policy.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic

from . import local_generation as generation, study_tasks, repair_feedback


def atomic_save(path, value):
    temporary = path.with_suffix(".tmp")
    generation.save(temporary, value)
    temporary.replace(path)


def run_child(command, state, persist):
    """Own one process group; first interrupt is graceful, subsequent ones kill."""
    with Path(state["child_log"]).open("wb") as output:
        try:
            child = subprocess.Popen(command, start_new_session=True, stdout=output, stderr=subprocess.STDOUT,
                                     cwd=Path(__file__).resolve().parents[3])
        except OSError as error:
            state["popen_error"] = repr(error)
            raise
        primary, sent = None, False
        state["child"] = dict(pid=child.pid, pgid=child.pid)
        try:
            persist()
        except BaseException as error:
            primary = error
        while True:
            try:
                if primary is not None and not sent:
                    os.killpg(child.pid, signal.SIGINT)
                    sent = True
                state["child"]["returncode"] = child.wait()
                break
            except ProcessLookupError:
                sent = True
            except KeyboardInterrupt as error:
                escalation = signal.SIGKILL if primary is not None else signal.SIGINT
                primary = primary or error
                try:
                    os.killpg(child.pid, escalation)
                except ProcessLookupError:
                    pass
                sent = True
        if primary is not None:
            raise primary
    return state["child"]["returncode"]


def collect_generation(directory, current, attempts):
    """Incomplete native files remain intact and cannot establish non-submission."""
    runtime = directory / "runtime.json"
    if runtime.exists():
        try:
            current["timing"]["loading"] = json.loads(runtime.read_text())["load_seconds"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            current["runtime_evidence_error"] = repr(error)
    try:
        outcome = json.loads((directory / "outcome.json").read_text())
        rows = outcome["requests"]
        if outcome["status"] not in ("success", "error", "cancelled") or len(rows) != len(attempts):
            raise ValueError("Incomplete generation outcome")
        current.update(known_submissions=sum(r["submitted"] for r in rows),
                        observed_tokens=sum(r["generation_tokens"] for r in rows),
                        usage_complete=outcome["status"] == "success", native_counters=outcome["native"])
        for attempt, row in zip(attempts, rows):
            attempt.update(submitted=row["submitted"], generation_status=row["status"])
        current["generation_error"] = dict(primary=outcome.get("error"), cleanup=outcome.get("cleanup_error"))
        current["timing"].update(preparation=outcome.get("preparation_seconds"),
                                  generation=(outcome["native"] or {}).get("wall_time"))
        return outcome
    except (OSError, ValueError, KeyError, TypeError) as error:
        current["generation_evidence_error"] = repr(error)
        return None


def validate(worklist, recipe, source, reply_allowance, output_token_allowance, batching, fixtures, cohort_sha256):
    count = len(worklist)
    limits = (reply_allowance, output_token_allowance, *batching.values(), recipe["max_tokens"])
    if (not count or source not in study_tasks.SOURCES or any(type(n) is not int or n <= 0 for n in limits)
            or batching["prefill_batch_size"] > batching["completion_batch_size"] or type(recipe["max_repairs"]) is not int
            or recipe["max_repairs"] < 0 or reply_allowance < count
            or output_token_allowance < count * recipe["max_tokens"]):
        raise ValueError("Require a nonempty worklist, valid source/limits and full initial allowances")
    if recipe["max_repairs"]:
        if (any(type(recipe.get(k)) is not int or recipe[k] <= 0
                for k in ("repair_max_tokens", "feedback_field_bytes", "feedback_bytes"))
                or not isinstance(recipe.get("repair"), str) or not recipe["repair"]):
            raise ValueError("Require repair instruction and positive integer output/feedback limits")
        repair_feedback.validate_limits(recipe["feedback_field_bytes"], recipe["feedback_bytes"])
    seed = recipe["base_seed"]
    if type(seed) is not int or not 0 <= seed <= seed + (recipe["max_repairs"] + 1) * count - 1 < 2**64:
        raise ValueError("Seed schedule must fit unsigned 64-bit integers")
    if type(fixtures) is not bool or (not fixtures and cohort_sha256 != study_tasks.COHORT_SHA256):
        raise ValueError("Require fixture provenance or the pinned cohort digest")
    if len({(b["split"], b["id"]) for b, _ in worklist}) != count:
        raise ValueError("Duplicate worklist identities")


def initial_requests(worklist, recipe):
    """Build statement-only initial messages in worklist order."""
    return [dict(messages=[dict(role="system", content=recipe["system"]),
                          dict(role="user", content=task["question"] + "\n\n" + recipe["initial"])],
                 seed=recipe["base_seed"] + i, temperature=recipe["temperature"], top_p=recipe["top_p"],
                 max_tokens=recipe["max_tokens"]) for i, (_, task) in enumerate(worklist)]


def prepare(worklist, output_dir, recipe, source, model_path, model_manifest, mlx_python,
            batching, reply_allowance, output_token_allowance, fixtures, cohort_sha256, started):
    count, seed = len(worklist), recipe["base_seed"]
    directory = Path(output_dir).absolute()
    model_path, model_manifest = map(lambda p: str(Path(p).absolute()), (model_path, model_manifest))
    manifest_sha256 = generation.file_hash(model_manifest)
    directory.mkdir(parents=True, exist_ok=False)
    snapshots = []
    for i, (binding, task) in enumerate(worklist):
        path = directory / f"task-{i:04d}.json"
        generation.save(path, task)
        snapshots.append(dict(binding=binding, snapshot=str(path), snapshot_sha256=generation.file_hash(path)))
    module_root = Path(__file__).parents[1]
    sources = [Path(__file__), Path(generation.__file__), Path(study_tasks.__file__),
               module_root / "stage03/verification.py", module_root / "stage03/inspect_runner.py",
               module_root / "stage01/screen.py", module_root / "stage02/cohort.py", Path(repair_feedback.__file__)]
    generation.save(directory / "inputs.json", dict(worklist=snapshots, source=source, fixtures=fixtures,
        cohort_sha256=cohort_sha256, recipe=recipe, reply_allowance=reply_allowance,
        output_token_allowance=output_token_allowance, model_path=model_path, model_manifest=model_manifest,
        model_manifest_sha256=manifest_sha256, mlx_python=mlx_python, coordinator_python=sys.executable,
        source_sha256={str(p.absolute()): generation.file_hash(p) for p in sources}))
    specification = dict(model_path=model_path, model_manifest=model_manifest, **batching)
    progress = dict(status="running", phase="prepared",
        committed_replies=0, committed_tokens=0, known_submissions=0, observed_tokens=0, usage_complete=True,
        timing=dict(loading=None, preparation=None, generation=None, verification=None),
        rounds=[], tasks=[], attempts=[dict(attempt=i, position=i, round=0, row=i, seed=seed+i, parent=None, submitted=False, state="not_run")
                  for i in range(count)])
    def persist():
        rounds = progress["rounds"]
        progress["tasks"] = task_states(progress)
        if rounds:
            known = [r["known_submissions"] for r in rounds]
            progress.update(known_submissions=None if None in known else sum(known),
                known_submissions_lower_bound=sum(n for n in known if n is not None),
                observed_tokens=sum(r["observed_tokens"] for r in rounds),
                usage_complete=all(r["usage_complete"] for r in rounds))
            for key in ("loading", "preparation", "generation", "verification"):
                values = [r["timing"][key] for r in rounds if r["timing"].get(key) is not None]
                progress["timing"][key] = sum(values) if values else None
        progress["timing"]["overall"] = monotonic() - started
        atomic_save(directory / "outcome.json", progress)
    persist()
    return directory, specification, progress, persist


def prepare_round(directory, specification, progress, requests, attempts):
    """Register planned paths before writing, retaining failed-preparation evidence."""
    index = len(progress["rounds"])
    native, round_file = directory / f"generation-{index:04d}", directory / f"round-{index:04d}.json"
    current = dict(round=index, attempt_ids=[a["attempt"] for a in attempts], timing={}, known_submissions=0, observed_tokens=0,
        usage_complete=True, round_file=str(round_file),
        generation_outcome=str(native / "outcome.json"), child_log=str(directory / f"child-{index:04d}.log"))
    progress["rounds"].append(current)
    generation.save(round_file, dict(specification, requests=requests, output_dir=str(native)))
    return current


def generate_round(current, attempts, progress, persist, mlx_python, requests):
    count = len(requests)
    native, round_file = Path(current["generation_outcome"]).parent, current["round_file"]
    progress.update(phase="dispatching", committed_replies=progress["committed_replies"] + count,
                    committed_tokens=progress["committed_tokens"] + sum(r["max_tokens"] for r in requests))
    current.update(known_submissions=None, usage_complete=False)
    for attempt in attempts:
        attempt.update(submitted=None, state="submission_unknown")
    persist()
    def child_started():
        progress["phase"] = "generation"
        persist()
    try:
        code = run_child([mlx_python, "-m", "src.data-pipeline.stage03.local_generation", str(round_file)], current, child_started)
    finally:
        if "popen_error" in current:
            current.update(known_submissions=0, observed_tokens=0, usage_complete=True)
            for attempt in attempts:
                attempt.update(submitted=False, generation_status="not_run", state="not_run")
            outcome = None
        else:
            outcome = collect_generation(native, current, attempts)
        persist()
    if outcome is None or code != 0 or outcome["status"] != "success":
        raise RuntimeError("Generation failed or lacks a complete successful outcome")
    return outcome


def extract_round(directory, worklist, rows, attempts, progress, source, persist):
    progress["phase"] = "extraction"
    persist()
    extracted = []
    for row, attempt in zip(rows, attempts):
        binding, task = worklist[attempt["position"]]
        try:
            if row["status"] == "length":
                attempt.update(extraction="length", state="failed")
                continue
            program = generation.extract_program(row)
        except (MemoryError, RecursionError) as error:
            attempt.update(extraction="compiler_resource", error=repr(error))
            raise
        except (ValueError, SyntaxError) as error:
            attempt.update(extraction="syntax" if isinstance(error, SyntaxError) else "format",
                           error=repr(error), state="failed")
            continue
        path = directory / f"program-{attempt['attempt']:04d}.py"
        path.write_bytes(program)
        identity = study_tasks.candidate_identity(binding["split"], binding["id"], binding["sha256"], source, program)
        attempt.update(extraction="success", program=str(path), candidate=identity)
        extracted.append((attempt, (task, binding["sha256"], program)))
    return extracted


def verify_round(directory, current, extracted, progress, persist):
    progress["phase"] = "verification"
    persist()
    if extracted:
        index = current["round"]
        from . import verification
        began = monotonic()
        try:
            verdicts = verification.verify_many([candidate for _, candidate in extracted], directory / f"verification-{index:04d}",
                                                 max_samples=4, concurrent_inference=False)
        finally:
            current["timing"]["verification"] = monotonic() - began
        generation.save(directory / f"verification-{index:04d}.json", verdicts)
        for (attempt, _), verdict in zip(extracted, verdicts):
            attempt["verification"] = verdict
            if verdict["status"] == "passed":
                attempt["state"] = "passed"
            elif verdict["status"] == "failed":
                attempt["state"] = "failed"
        if any(v["status"] not in ("passed", "failed") for v in verdicts):
            raise RuntimeError("Verification infrastructure error")


def execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source):
    """Generate, extract and verify one round; return its native reply rows."""
    current = prepare_round(directory, specification, progress, requests, attempts)
    outcome = generate_round(current, attempts, progress, persist, mlx_python, requests)
    rows = outcome["requests"]
    extracted = extract_round(directory, worklist, rows, attempts, progress, source, persist)
    verify_round(directory, current, extracted, progress, persist)
    return rows


def repair_request(request, row, parent, recipe, seed):
    """Append the preceding reply and permitted feedback to the full history."""
    if parent["extraction"] == "success":
        rendered = repair_feedback.render(parent["verification"]["feedback"],
            recipe["feedback_field_bytes"], recipe["feedback_bytes"])
    else:
        rendered = dict(text=f"Local failure: {parent['extraction']}. Return a complete replacement program.")
    parent["repair_feedback"] = rendered
    return dict(request, messages=request["messages"] + [dict(role="assistant", content=row["text"]),
        dict(role="user", content=recipe["repair"] + "\n\n" + rendered["text"])],
        seed=seed, max_tokens=recipe["repair_max_tokens"])


def prepare_repairs(attempts, requests, rows, progress, recipe, task_count,
                    reply_allowance, output_token_allowance, persist):
    """Mark exhausted tasks and prepare the stable repair prefix within budgets."""
    progress["phase"] = "repair_preparation"
    eligible = [(parent, request, row) for parent, request, row in zip(attempts, requests, rows)
                if parent["state"] == "failed"]
    if attempts[0]["round"] == recipe["max_repairs"]:
        for parent, _, _ in eligible:
            parent["state"] = "repairs_exhausted"
        return [], []
    slots = min(reply_allowance - progress["committed_replies"],
                (output_token_allowance - progress["committed_tokens"]) // recipe["repair_max_tokens"])
    for parent, _, _ in eligible[slots:]:
        parent["state"] = "budget_exhausted"
    round_index = attempts[0]["round"] + 1
    requests, attempts = [], []
    for row_index, (parent, request, row) in enumerate(eligible[:slots]):
        position = parent["position"]
        draw_seed = recipe["base_seed"] + round_index * task_count + position
        requests.append(repair_request(request, row, parent, recipe, draw_seed))
        attempt = dict(attempt=len(progress["attempts"]), position=position, round=round_index, row=row_index,
            seed=draw_seed, parent=parent["attempt"], submitted=False, state="not_run")
        progress["attempts"].append(attempt)
        attempts.append(attempt)
    persist()
    return requests, attempts


def task_states(progress):
    """Derive current task states; terminal budget decisions survive later stops."""
    latest = {a["position"]: a for a in progress["attempts"]}
    tasks = []
    for position, attempt in sorted(latest.items()):
        state = attempt["state"]
        if progress["status"] in ("error", "cancelled") and (
                state == "failed" or (state == "not_run" and attempt["parent"] is not None)):
            state = progress["status"]
        tasks.append(dict(position=position, state=state, last_attempt=attempt["attempt"]))
    return tasks


def finalize_states(progress, directory):
    for attempt in progress["attempts"]:
        if attempt["state"] not in ("passed", "failed", "repairs_exhausted", "budget_exhausted"):
            attempt["state"] = ("submission_unknown" if attempt["submitted"] is None else
                                progress["status"] if attempt["submitted"] else "not_run")
    progress["verification_logs"] = [str(p) for p in sorted(directory.glob("verification-*/*.eval"))]


def run(worklist, output_dir, *, source, model_path, model_manifest, mlx_python, recipe,
        reply_allowance, output_token_allowance, completion_batch_size, prefill_batch_size,
        prefill_step_size, fixtures=False, cohort_sha256=None):
    """Generate, extract and verify one initial round, then repair eligible failures.

    Call from the repository Python with a finite, ordered worklist of
    ``(binding, task)`` pairs from ``study_tasks.load_task`` and the pinned
    ``cohort_sha256``. Fixture callers instead set ``fixtures=True`` and supply
    bindings containing split, integer id and the original task-artifact sha256.
    ``source`` is actor or synthetic; one model/source is used throughout.
    ``output_dir`` must be new. Inputs and private task snapshots are saved there
    before dispatch. Supply explicit model, artifact-manifest and MLX-Python paths;
    the MLX interpreter runs a fresh generation child for each round.

    Required recipe fields are system, initial, temperature, top_p, base_seed,
    max_tokens and max_repairs. A zero max_repairs runs only the initial round.
    Enabled repairs also require repair (instruction), repair_max_tokens,
    feedback_field_bytes and feedback_bytes. Feedback caps count UTF-8 bytes,
    including truncation markers. Initial prompts use only recipe instructions
    and the statement. Repairs append the exact preceding reply and bounded
    public feedback, or a fixed local extraction-failure message, to the complete
    history. Passing tasks receive no further draws. Each request's seed is
    ``base_seed + round * len(worklist) + original_position``; the full potential
    schedule must fit unsigned 64-bit integers before any child is launched.

    Reserve one reply and the full per-request output-token limit before dispatch.
    The allowances must cover every initial request. Later rounds take the stable
    prefix of eligible repairs that fits both remaining allowances; reservations
    are never refunded. Batch and prefill settings pass through to generation.
    Generation finishes before one coalesced verification call per round, using
    max_samples=4 and concurrent_inference=False.

    Return the progress dictionary also saved atomically as outcome.json: status,
    final task states, ordered attempts with parent links, per-round artifact
    paths/evidence, committed allowances, observed usage and separate timings.
    Completed runs may contain exhausted tasks. Technical failures stop without
    retry and return status=error; uncertain submissions and incomplete usage stay
    explicit. Interruptions propagate after saving available evidence. Invalid
    inputs and setup failures may raise before an outcome can be created.
    Existing directories are refused; this API provides no resume or selection.
    """
    started = monotonic()
    worklist = json.loads(json.dumps(list(worklist), allow_nan=False))
    recipe = json.loads(json.dumps(recipe, allow_nan=False))
    mlx_python = str(Path(mlx_python).absolute())
    batching = dict(completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size, prefill_step_size=prefill_step_size)
    validate(worklist, recipe, source, reply_allowance, output_token_allowance, batching, fixtures, cohort_sha256)
    requests = initial_requests(worklist, recipe)
    directory, specification, progress, persist = prepare(worklist, output_dir, recipe, source,
        model_path, model_manifest, mlx_python, batching, reply_allowance, output_token_allowance,
        fixtures, cohort_sha256, started)
    try:
        """Execute the initial round: generate first responses, extract and verify."""
        attempts = progress["attempts"][:]
        rows = execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source)

        """Repair eligible failures until exhausted or budgeted."""
        while True:
            requests, attempts = prepare_repairs(attempts, requests, rows, progress, recipe,
                len(worklist), reply_allowance, output_token_allowance, persist)
            if not attempts:
                break
            rows = execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source)
        progress.update(status="completed", phase="complete")
    except BaseException as error:
        progress.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if progress["phase"] == "verification":
            progress["verification_cleanup_confirmed"] = getattr(error, "cleanup_confirmed", False)
        if not isinstance(error, Exception):
            raise
    finally:
        finalize_states(progress, directory)
        persist()
    return progress
