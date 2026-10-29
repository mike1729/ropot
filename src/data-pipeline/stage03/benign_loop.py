"""Bounded benign generation, repair and B1 acquisition; see docs/v2-batch-01-plan.md.

prepare_contract(output_dir, exposure_file, runtimes) freezes eight published tasks.
acquire(prepared_dir, output_dir) consumes that contract, one source at a time.
Exposure classification and runtime envelopes are reviewed inputs, not inferred
from arbitrary historical transcripts or provisional benchmark timings here.

exposure_file JSON: {"files": {category: {path: sha256}}, "tasks": {
  "task_id": {"task_sha256": digest, "known": [case_index], "disclosed": [case_index]}}}.
The five categories are EXPOSURE_SOURCES. Known means disclosure history was
checked; equivalent inputs share disclosure. Missing history cannot qualify audits.

runtimes maps actor/synthetic to model_path, model_manifest, mlx_python and
qualification_file. That reviewed JSON contains model_manifest_sha256,
thinking=false, max_tokens=8192, max_prompt_tokens, batching (the BATCH_KEYS),
and files ({path: sha256})
binding the supporting runtime/template/capacity evidence. It must establish the
chosen envelope; a provisional timing outcome alone is insufficient.

Preparation returns ready or insufficient_cases and creates no inference process.
acquire requires a new output directory; report.json retains failures and usage. Native
round outcomes contain actual rendered lengths. Live execution requires a separately
authorized finite contract. The shared run() owns generation, repair,
accounting and first-success stopping; acquire() adds private audits and reporting.
No technical retries or resume.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
from time import monotonic

from . import local_generation as generation, study_tasks, repair_feedback


SOURCES = study_tasks.SOURCES
MODEL_HASHES = dict(actor="e27567df29c1239992f0c7bc4b8d18c1e3ed59251be8d7cc3ac42f49bf7012a6",
                    synthetic="ab95823f4cd7f16b345f22aca92c24a49ca635bd2a92f81813f1617fc8820472")
EXPOSURE_SOURCES = {"benign", "attacks", "capability", "suite", "batching"}
BATCH_KEYS = ("completion_batch_size", "prefill_batch_size", "prefill_step_size")
GATES = dict(joint_tasks=6, truncation_fraction=.05)
RECIPE = dict(system="Return only a complete Python 3 stdin/stdout solution.",
              initial="Solve the problem correctly for every valid input.", repair="Fix the solution.",
              temperature=.2, top_p=.95, base_seed=41, max_tokens=8192, max_repairs=2,
              repair_max_tokens=8192, feedback_field_bytes=1024, feedback_bytes=16384)


def source_files():
    root = Path(__file__).resolve().parents[1]
    return (Path(__file__).resolve(), Path(generation.__file__), Path(study_tasks.__file__),
            root / "stage03/verification.py", root / "stage03/inspect_runner.py",
            root / "artifacts.py", root / "paths.py", root / "stage02/published_cohort.py",
            root / "stage01/screen.py", root / "stage02/cohort.py", Path(repair_feedback.__file__))


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
        # Reap the child, escalating interrupts while preserving the original error.
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
            or type(recipe.get("enable_thinking", False)) is not bool
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


def initial_requests(worklist, recipe, initial_messages=None):
    """Build requests from aligned system/user prompts."""
    if initial_messages is None:
        initial_messages = [[dict(role="system", content=recipe["system"]),
                             dict(role="user", content=task["question"] + "\n\n" + recipe["initial"])]
                            for _, task in worklist]
    if (not isinstance(initial_messages, list) or len(initial_messages) != len(worklist)
            or any(not isinstance(messages, list) or len(messages) != 2 or any(
                not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != role
                or not isinstance(message["content"], str)
                for role, message in zip(("system", "user"), messages)) for messages in initial_messages)):
        raise ValueError("Require aligned system/user prompts")
    initial_messages = json.loads(json.dumps(initial_messages, allow_nan=False))
    return [dict(messages=initial_messages[i],
                 seed=recipe["base_seed"] + i, temperature=recipe["temperature"], top_p=recipe["top_p"],
                 max_tokens=recipe["max_tokens"],
                 enable_thinking=recipe.get("enable_thinking", False),
                 **{k: recipe[k] for k in ("max_prompt_tokens",) if k in recipe}) for i, (_, task) in enumerate(worklist)]


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
    generation.save(directory / "inputs.json", dict(worklist=snapshots, source=source, fixtures=fixtures,
        cohort_sha256=cohort_sha256, recipe=recipe, reply_allowance=reply_allowance,
        output_token_allowance=output_token_allowance, model_path=model_path, model_manifest=model_manifest,
        model_manifest_sha256=manifest_sha256, mlx_python=mlx_python, coordinator_python=sys.executable,
        source_sha256={str(p.absolute()): generation.file_hash(p) for p in source_files()}))
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
    current["round_file_sha256"] = generation.file_hash(round_file)
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
    # Save compilable programs and classify failed replies at their original positions.
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


def verify_round(directory, current, extracted, progress, persist, candidate_checker=None):
    progress["phase"] = "verification"
    persist()
    if extracted:
        index = current["round"]
        from . import verification
        began = monotonic()
        try:
            checker = verification.verify_many if candidate_checker is None else candidate_checker
            verdicts = checker([candidate for _, candidate in extracted], directory / f"verification-{index:04d}",
                               max_samples=4, concurrent_inference=False)
            if len(verdicts) != len(extracted) or any(
                    verdict["key"]["task_sha256"] != candidate[1]
                    or verdict["key"]["program_sha256"] != attempt["candidate"]["program_sha256"]
                    for (attempt, candidate), verdict in zip(extracted, verdicts)):
                raise ValueError("Checker verdicts differ from candidate order/identity")
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


def execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source,
                  candidate_checker=None):
    """Generate, extract and verify one round."""
    current = prepare_round(directory, specification, progress, requests, attempts)
    outcome = generate_round(current, attempts, progress, persist, mlx_python, requests)
    rows = outcome["requests"]
    extracted = extract_round(directory, worklist, rows, attempts, progress, source, persist)
    verify_round(directory, current, extracted, progress, persist, candidate_checker)


def repair_request(initial, parent, recipe, seed):
    """Build a fresh prompt with original context, current code and public feedback."""
    if parent["extraction"] == "success":
        code = Path(parent["program"]).read_bytes().decode("utf-8")
        feedback = parent["verification"]["feedback"]
        rendered = (dict(text=feedback["message"]) if set(feedback) == {"message"}
                    else repair_feedback.render(feedback, recipe["feedback_field_bytes"], recipe["feedback_bytes"]))
    else:
        code, rendered = "", dict(text=f"Local extraction failure: {parent['extraction']}")
    rendered["text"] = rendered["text"].replace(" hidden=not_run", "", 1)
    parent["repair_feedback"] = rendered
    return dict(initial, seed=seed, max_tokens=recipe["repair_max_tokens"], messages=[initial["messages"][0],
        dict(role="user", content=initial["messages"][1]["content"] + "\nPrevious candidate:\n" + code
             + "\nShown feedback:\n" + rendered["text"] + "\n" + recipe["repair"])])


def prepare_repairs(attempts, initial, progress, recipe, reply_allowance, output_token_allowance, persist):
    """Mark exhausted tasks and prepare the stable repair prefix within budgets."""
    progress["phase"] = "repair_preparation"
    eligible = [parent for parent in attempts if parent["state"] == "failed"]
    if attempts[0]["round"] == recipe["max_repairs"]:
        for parent in eligible:
            parent["state"] = "repairs_exhausted"
        return [], []
    slots = min(reply_allowance - progress["committed_replies"],
                (output_token_allowance - progress["committed_tokens"]) // recipe["repair_max_tokens"])
    for parent in eligible[slots:]:
        parent["state"] = "budget_exhausted"
    round_index = attempts[0]["round"] + 1
    requests, attempts = [], []
    for row_index, parent in enumerate(eligible[:slots]):
        position = parent["position"]
        draw_seed = recipe["base_seed"] + round_index * len(initial) + position
        requests.append(repair_request(initial[position], parent, recipe, draw_seed))
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
        prefill_step_size, fixtures=False, cohort_sha256=None, initial_messages=None, candidate_checker=None):
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
    Optional enable_thinking defaults to false and passes through every round.
    Enabled repairs also require repair (instruction), repair_max_tokens,
    feedback_field_bytes and feedback_bytes. Feedback caps count UTF-8 bytes,
    including truncation markers. Initial prompts use only recipe instructions
    and the statement. Repairs use the original context, current extracted code
    and bounded public feedback, or a fixed local extraction-failure message.
    Raw replies and prior repair turns never enter prompts. Passing tasks receive
    no further draws. Each request's seed is
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
    Optional initial_messages supply aligned system/user prompts.
    candidate_checker replaces published verification with the same ordered
    candidate/verdict contract; it must provide a public feedback projection or
    one fixed message. Each exact round file is saved and hashed before dispatch.
    """
    started = monotonic()
    worklist = json.loads(json.dumps(list(worklist), allow_nan=False))
    recipe = json.loads(json.dumps(recipe, allow_nan=False))
    mlx_python = str(Path(mlx_python).absolute())
    batching = dict(completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size, prefill_step_size=prefill_step_size)
    validate(worklist, recipe, source, reply_allowance, output_token_allowance, batching, fixtures, cohort_sha256)
    if candidate_checker is not None and not callable(candidate_checker):
        raise ValueError("Require a callable candidate checker")
    initial = requests = initial_requests(worklist, recipe, initial_messages)
    directory, specification, progress, persist = prepare(worklist, output_dir, recipe, source,
        model_path, model_manifest, mlx_python, batching, reply_allowance, output_token_allowance,
        fixtures, cohort_sha256, started)
    try:
        """Execute the initial round: generate first responses, extract and verify."""
        attempts = progress["attempts"][:]
        execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source,
                      candidate_checker)

        """Repair eligible failures until exhausted or budgeted."""
        while True:
            requests, attempts = prepare_repairs(attempts, initial, progress, recipe,
                reply_allowance, output_token_allowance, persist)
            if not attempts:
                break
            execute_round(directory, specification, worklist, requests, attempts, progress, persist, mlx_python, source,
                          candidate_checker)
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


def read(path):
    return json.loads(Path(path).read_text())


def check_files(files):
    if not files or any(generation.file_hash(path) != expected for path, expected in files.items()):
        raise ValueError("Missing or changed evidence files")


def partition(task, exposure):
    """Use reviewed known/disclosed indices; equivalent inputs share exposure."""
    examples, groups = [], {}
    for case in sorted(task["cases"], key=lambda c: c["index"]):
        if case["visible"]:
            examples.append(case)
        else:
            groups.setdefault(" ".join(case["input"].split()), []).append(case)
    groups = list(groups.values())
    known, disclosed = set(exposure["known"]), set(exposure["disclosed"])
    indices = {c["index"] for c in task["cases"]}
    if not disclosed <= known <= indices:
        raise ValueError("Exposure indices differ from the task")
    random.Random(17 + task["id"]).shuffle(groups)
    audit = [g[0] for g in groups if {c["index"] for c in g} <= known
             and not {c["index"] for c in g} & disclosed][:4]
    audit_ids = {c["index"] for c in audit}
    menu = [g[0] for g in groups if g[0]["index"] not in audit_ids][:8]
    removed = [c["index"] for c in menu if len(c["input"].encode()) > 4096]
    menu = [c for c in menu if c["index"] not in removed]
    if len(audit) != 4 or len(menu) < 4:
        raise ValueError("Insufficient eligible cases")
    return dict(examples=examples, menu=menu, audit=audit, removed_menu=removed,
                seed=17 + task["id"], permutation=[g[0]["index"] for g in groups])


def qualify(source, runtime):
    """Check a reviewed envelope's bindings; no benchmark or model call."""
    runtime = dict(runtime)
    proof = read(runtime["qualification_file"])
    check_files(proof["files"])
    actual = generation.file_hash(runtime["model_manifest"])
    batching = proof["batching"]
    if (any(type(batching[k]) is not int or batching[k] <= 0 for k in BATCH_KEYS)
            or batching["prefill_batch_size"] > batching["completion_batch_size"]):
        raise ValueError("Invalid qualified batching settings")
    if (actual != MODEL_HASHES[source] or proof["model_manifest_sha256"] != actual
            or proof["thinking"] is not False or proof["max_tokens"] != 8192
            or type(proof["max_prompt_tokens"]) is not int or proof["max_prompt_tokens"] <= 0):
        raise ValueError("Runtime qualification does not cover B1")
    pinned = {row["path"]: row["sha256"] for row in read(runtime["model_manifest"])["files"]}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if generation.file_hash(Path(runtime["model_path"]) / name) != pinned[name]:
            raise ValueError("Model metadata differs from the frozen checkpoint")
    runtime.update(qualification=proof,
                   qualification_sha256=generation.file_hash(runtime["qualification_file"]))
    return runtime


def prepare_contract(output_dir, exposure_file, runtimes):
    """Freeze caller-reviewed exposure and runtime records, retaining exclusions."""
    exposure = read(exposure_file)
    if set(exposure["files"]) != EXPOSURE_SOURCES:
        raise ValueError("Require all five exposure sources")
    for files in exposure["files"].values():
        check_files(files)
    runtimes = {s: qualify(s, runtimes[s]) for s in SOURCES}
    loaded = study_tasks.load_cohort()
    order = loaded["groups"]["development"]["initial"]
    entries, exclusions = [], []
    # Select the first eight case-eligible scaffold tasks in cohort order.
    for identifier in order[:20]:
        binding, task = study_tasks.load_task(loaded, identifier)
        history = exposure["tasks"].get(str(identifier))
        if history is None or history["task_sha256"] != binding["sha256"]:
            exclusions.append(dict(id=identifier, reason="Unknown or mismatched exposure history"))
            continue
        try:
            cases = partition(task, history)
        except ValueError as error:
            exclusions.append(dict(id=identifier, reason=str(error)))
            continue
        entries.append(dict(binding=binding, partition=cases))
        if len(entries) == 8:
            break
    contract = dict(status="ready" if len(entries) == 8 else "insufficient_cases",
        cohort_sha256=study_tasks.COHORT_SHA256, scaffold=order[:20], confirmation=order[20:50],
        entries=entries, exclusions=exclusions, exposure_file=str(Path(exposure_file).resolve()),
        exposure_sha256=generation.file_hash(exposure_file), runtimes=runtimes,
        recipe=dict(RECIPE), gates=dict(GATES), reply_allowance=24, output_token_allowance=196608,
        case_execution_allowance=2*sum(3*len(e["partition"]["examples"]+e["partition"]["menu"])+4 for e in entries),
        sources={str(p.absolute()): generation.file_hash(p) for p in source_files()})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    generation.save(directory / "contract.json", contract)
    return contract


def shown_messages(task, cases):
    shown = [dict(input=c["input"], answer=c["answer"]) for c in cases["examples"]+cases["menu"]]
    return [dict(role="system", content=RECIPE["system"]), dict(role="user", content=
        task["question"] + "\n\n" + RECIPE["initial"] + "\nShown cases:\n" + json.dumps(shown, ensure_ascii=False))]


def verify_case_views(candidates, directory, views, **settings):
    """Reuse the verifier's identity/order guarantees for an explicit case view."""
    from . import verification
    requests = [(dict(cases=[dict(c, visible=True) for c in views[task_hash]]), task_hash, program)
                for _, task_hash, program in candidates]
    verdicts = verification.verify_many([(view, verification.digest(view), program)
                                        for view, _, program in requests], directory, **settings)
    for verdict, (_, original, _) in zip(verdicts, requests):
        verdict["key"]["case_view_sha256"] = verdict["key"]["task_sha256"]
        verdict["key"]["task_sha256"] = original
    return verdicts


def load_contract(prepared_dir):
    """Recheck frozen inputs before creating outputs or dispatching either source."""
    contract_path = Path(prepared_dir) / "contract.json"
    contract = read(contract_path)
    check_files(contract["sources"])
    check_files({contract["exposure_file"]: contract["exposure_sha256"]})
    exposure = read(contract["exposure_file"])
    for files in exposure["files"].values():
        check_files(files)
    if (contract["status"] != "ready" or len(contract["entries"]) != 8 or contract["recipe"] != RECIPE
            or contract["gates"] != GATES or contract["cohort_sha256"] != study_tasks.COHORT_SHA256):
        raise ValueError("B1 contract is not ready")
    loaded, worklist, shown, audits = study_tasks.load_cohort(), [], {}, {}
    order = loaded["groups"]["development"]["initial"]
    ids = [entry["binding"]["id"] for entry in contract["entries"]]
    if (contract["scaffold"] != order[:20] or contract["confirmation"] != order[20:50]
            or ids != [i for i in order[:20] if i in ids]):
        raise ValueError("Changed development allocation")
    for entry in contract["entries"]:
        binding, task = study_tasks.load_task(loaded, entry["binding"]["id"])
        if (binding != entry["binding"]
                or partition(task, exposure["tasks"][str(task["id"])]) != entry["partition"]):
            raise ValueError("Task or partition changed")
        worklist.append((binding, task))
        shown[binding["sha256"]] = entry["partition"]["examples"]+entry["partition"]["menu"]
        audits[binding["sha256"]] = entry["partition"]["audit"]
    for source in SOURCES:
        runtime = contract["runtimes"][source]
        if qualify(source, runtime) != runtime:
            raise ValueError("Changed runtime qualification")
    return contract, worklist, shown, audits


def audit_selected(result, worklist, views, directory):
    """Privately audit each first shown-passing program once; never regenerate."""
    selected = [result["attempts"][t["last_attempt"]] for t in result["tasks"] if t["state"] == "passed"]
    candidates = [(worklist[a["position"]][1], a["candidate"]["task_sha256"], Path(a["program"]).read_bytes())
                  for a in selected]
    if any(sha256(p).hexdigest() != a["candidate"]["program_sha256"] for a, (_, _, p) in zip(selected, candidates)):
        raise ValueError("Selected program changed before audit")
    verdicts = verify_case_views(candidates, directory, views) if candidates else []
    audits = []
    for attempt, (_, task_hash, _), verdict in zip(selected, candidates, verdicts):
        observed = verdict["feedback"].get("visible", [])
        audits.append(dict(position=attempt["position"], program=attempt["program"],
            verdict={k: verdict[k] for k in ("status", "key", "log", "sample_id")},
            cases=[dict(index=c["index"], status=observed[i]["status"] if i < len(observed) else "not_run")
                   for i, c in enumerate(views[task_hash])]))
    return audits


def acquire_source(contract, worklist, shown, audits, source, directory, state):
    runtime = contract["runtimes"][source]
    initial = [shown_messages(t, e["partition"]) for (_, t), e in zip(worklist, contract["entries"])]
    result = run(worklist, directory / source, source=source,
        **{k: runtime[k] for k in ("model_path", "model_manifest", "mlx_python")},
        **{k: runtime["qualification"]["batching"][k] for k in BATCH_KEYS},
        recipe=dict(contract["recipe"], max_prompt_tokens=runtime["qualification"]["max_prompt_tokens"]),
        reply_allowance=contract["reply_allowance"], output_token_allowance=contract["output_token_allowance"],
        cohort_sha256=contract["cohort_sha256"], initial_messages=initial,
        candidate_checker=lambda c, d, **kw: verify_case_views(c, d, shown, **kw))
    if result["status"] != "completed":
        raise RuntimeError(f"{source} benign loop failed")
    began = monotonic()
    try:
        state["audits"] = audit_selected(result, worklist, audits, directory / (source+"-audit"))
    finally:
        state["audit_seconds"] = monotonic()-began
    if any(a["verdict"]["status"] not in ("passed", "failed") for a in state["audits"]):
        raise RuntimeError(f"{source} audit infrastructure error")
    state["status"] = "completed"


def summarize_acquisition(report, directory, started):
    """Fold saved loop accounting into coverage and cost, including failed work."""
    passed, output_tokens, replies, truncated, complete, prompt_tokens = {}, 0, 0, 0, True, 0
    # Reconcile each source's audited coverage and total usage, including failed work.
    for source, state in report["sources"].items():
        path = directory / source / "outcome.json"
        outcome = read(path) if path.exists() else None
        state.update(status=state["status"] if state["status"] != "running" else report["status"], loop=str(path))
        passed[source] = {a["position"] for a in state["audits"] if a["verdict"]["status"] == "passed"}
        state["coverage"] = len(passed[source]) / 8
        complete &= outcome is not None and outcome["usage_complete"]
        if outcome:
            native = [(r.get("native_counters") or {}).get("prompt_tokens") for r in outcome["rounds"]]
            state["native_prompt_tokens"] = sum(native) if all(n is not None for n in native) else None
            state.update(reserved_output_tokens=outcome["committed_tokens"],
                         observed_output_tokens=outcome["observed_tokens"], usage_complete=outcome["usage_complete"])
            output_tokens += outcome["observed_tokens"]
            replies += outcome["known_submissions_lower_bound"]
            truncated += sum(a.get("generation_status") == "length" for a in outcome["attempts"])
        value = state.get("native_prompt_tokens")
        prompt_tokens = prompt_tokens + value if prompt_tokens is not None and value is not None else None
    joint = set.intersection(*(passed.get(s, set()) for s in SOURCES))
    complete &= prompt_tokens is not None
    report.update(joint_positions=sorted(joint), joint_coverage=len(joint)/8,
        output_tokens=output_tokens if complete else None, observed_output_tokens_lower_bound=output_tokens,
        submitted_replies_lower_bound=replies, truncated_replies=truncated, usage_complete=bool(complete),
        native_prompt_tokens=prompt_tokens,
        output_tokens_per_joint_task=output_tokens/len(joint) if joint and complete else None,
        elapsed_seconds=monotonic()-started,
        ready_for_b2=report["status"] == "completed" and complete and len(joint) >= GATES["joint_tasks"]
            and truncated <= GATES["truncation_fraction"]*replies)


def acquire(prepared_dir, output_dir):
    """Consume a frozen B1 contract once; technical failure stops both sources."""
    contract, worklist, shown, audits = load_contract(prepared_dir)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    generation.save(directory / "contract.json", contract)
    started = monotonic()
    report = dict(status="running", contract_sha256=generation.file_hash(Path(prepared_dir) / "contract.json"), sources={})
    try:
        for source in SOURCES:
            state = report["sources"][source] = dict(status="running", audits=[])
            atomic_save(directory / "report.json", report)
            acquire_source(contract, worklist, shown, audits, source, directory, state)
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if not isinstance(error, Exception):
            raise
    finally:
        summarize_acquisition(report, directory, started)
        atomic_save(directory / "report.json", report)
    return report
