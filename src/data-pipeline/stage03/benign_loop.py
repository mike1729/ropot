"""Sequential initial-round API; see docs/stage-03-batch-07-plan.md (7a).

Call run from the repository Python; only the round-file child imports MLX.
No repairs, retries, resume, selection or input-token admission policy.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic

from . import local_generation as generation, study_tasks


def atomic_save(path, value):
    temporary = path.with_suffix(".tmp")
    generation.save(temporary, value)
    temporary.replace(path)


def run_child(command, progress, persist):
    """Own one process group; first interrupt is graceful, subsequent ones kill."""
    with Path(progress["child_log"]).open("wb") as output:
        child = subprocess.Popen(command, start_new_session=True, stdout=output, stderr=subprocess.STDOUT,
                                 cwd=Path(__file__).resolve().parents[3])
        primary, sent = None, False
        progress["child"] = dict(pid=child.pid, pgid=child.pid)
        progress["phase"] = "generation"
        try:
            persist()
        except BaseException as error:
            primary = error
        while True:
            try:
                if primary is not None and not sent:
                    os.killpg(child.pid, signal.SIGINT)
                    sent = True
                progress["child"]["returncode"] = child.wait()
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
    return progress["child"]["returncode"]


def collect_generation(directory, progress, count):
    """Incomplete native files remain intact and cannot establish non-submission."""
    runtime = directory / "runtime.json"
    if runtime.exists():
        try:
            progress["timing"]["loading"] = json.loads(runtime.read_text())["load_seconds"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            progress["runtime_evidence_error"] = repr(error)
    try:
        outcome = json.loads((directory / "outcome.json").read_text())
        rows = outcome["requests"]
        if outcome["status"] not in ("success", "error", "cancelled") or len(rows) != count:
            raise ValueError("Incomplete generation outcome")
        progress.update(known_submissions=sum(r["submitted"] for r in rows),
                        observed_tokens=sum(r["generation_tokens"] for r in rows),
                        usage_complete=outcome["status"] == "success", native_counters=outcome["native"])
        for attempt, row in zip(progress["attempts"], rows):
            attempt.update(submitted=row["submitted"], generation_status=row["status"])
        progress["generation_error"] = dict(primary=outcome.get("error"), cleanup=outcome.get("cleanup_error"))
        progress["timing"].update(preparation=outcome.get("preparation_seconds"),
                                  generation=(outcome["native"] or {}).get("wall_time"))
        return outcome
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress["generation_evidence_error"] = repr(error)
        return None


def validate(worklist, recipe, source, reply_allowance, output_token_allowance, batching, fixtures, cohort_sha256):
    count = len(worklist)
    limits = (reply_allowance, output_token_allowance, *batching.values(), recipe["max_tokens"])
    if (not count or source not in study_tasks.SOURCES or any(type(n) is not int or n <= 0 for n in limits)
            or batching["prefill_batch_size"] > batching["completion_batch_size"] or type(recipe["max_repairs"]) is not int
            or recipe["max_repairs"] != 0 or reply_allowance < count
            or output_token_allowance < count * recipe["max_tokens"]):
        raise ValueError("Require a nonempty worklist, valid source/limits and full initial allowances; repairs disabled")
    seed = recipe["base_seed"]
    if type(seed) is not int or not 0 <= seed <= seed + count - 1 < 2**64:
        raise ValueError("Seed schedule must fit unsigned 64-bit integers")
    if type(fixtures) is not bool or (not fixtures and cohort_sha256 != study_tasks.COHORT_SHA256):
        raise ValueError("Require fixture provenance or the pinned cohort digest")
    if len({(b["split"], b["id"]) for b, _ in worklist}) != count:
        raise ValueError("Duplicate worklist identities")
    return [dict(messages=[dict(role="system", content=recipe["system"]),
                          dict(role="user", content=task["question"] + "\n\n" + recipe["initial"])],
                 seed=seed + i, temperature=recipe["temperature"], top_p=recipe["top_p"],
                 max_tokens=recipe["max_tokens"]) for i, (_, task) in enumerate(worklist)]


def prepare(worklist, requests, output_dir, recipe, source, model_path, model_manifest, mlx_python,
            batching, reply_allowance, output_token_allowance, fixtures, cohort_sha256):
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
               module_root / "stage01/screen.py", module_root / "stage02/cohort.py"]
    generation.save(directory / "inputs.json", dict(worklist=snapshots, source=source, fixtures=fixtures,
        cohort_sha256=cohort_sha256, recipe=recipe, reply_allowance=reply_allowance,
        output_token_allowance=output_token_allowance, model_path=model_path, model_manifest=model_manifest,
        model_manifest_sha256=manifest_sha256, mlx_python=mlx_python, coordinator_python=sys.executable,
        source_sha256={str(p.absolute()): generation.file_hash(p) for p in sources}))
    native = directory / "generation-0000"
    round_file = directory / "round-0000.json"
    specification = dict(model_path=model_path, model_manifest=model_manifest, requests=requests,
        output_dir=str(native), **batching)
    generation.save(round_file, specification)
    progress = dict(status="running", phase="prepared", round_file=str(round_file),
        generation_outcome=str(native / "outcome.json"), child_log=str(directory / "child-0000.log"),
        committed_replies=0, committed_tokens=0, known_submissions=0, observed_tokens=0, usage_complete=True,
        timing=dict(loading=None, preparation=None, generation=None, verification=None),
        attempts=[dict(attempt=i, position=i, round=0, seed=seed+i, submitted=False, state="not_run")
                  for i in range(count)])
    def persist():
        atomic_save(directory / "outcome.json", progress)
    persist()
    return directory, progress, persist


def generate_round(directory, progress, persist, mlx_python, requests):
    count = len(requests)
    native, round_file = directory / "generation-0000", progress["round_file"]
    progress.update(phase="dispatching", committed_replies=count,
                    committed_tokens=sum(r["max_tokens"] for r in requests), known_submissions=None, usage_complete=False)
    for attempt in progress["attempts"]:
        attempt.update(submitted=None, state="submission_unknown")
    persist()
    try:
        code = run_child([mlx_python, "-m", __name__, str(round_file)], progress, persist)
    finally:
        outcome = collect_generation(native, progress, count)
        persist()
    if outcome is None or code != 0 or outcome["status"] != "success":
        raise RuntimeError("Generation failed or lacks a complete successful outcome")
    return outcome


def extract_round(directory, worklist, outcome, progress, source, persist):
    progress["phase"] = "extraction"
    persist()
    extracted = []
    for i, ((binding, task), row, attempt) in enumerate(zip(worklist, outcome["requests"], progress["attempts"])):
        try:
            if row["status"] == "length":
                attempt.update(extraction="length", state="repairs_exhausted")
                continue
            program = generation.extract_program(row)
        except (MemoryError, RecursionError) as error:
            attempt.update(extraction="compiler_resource", error=repr(error))
            raise
        except (ValueError, SyntaxError) as error:
            attempt.update(extraction="syntax" if isinstance(error, SyntaxError) else "format",
                           error=repr(error), state="repairs_exhausted")
            continue
        path = directory / f"program-{i:04d}.py"
        path.write_bytes(program)
        identity = study_tasks.candidate_identity(binding["split"], binding["id"], binding["sha256"], source, program)
        attempt.update(extraction="success", program=str(path), candidate=identity)
        extracted.append((attempt, (task, binding["sha256"], program)))
    return extracted


def verify_round(directory, extracted, progress, persist):
    progress["phase"] = "verification"
    persist()
    if extracted:
        from . import verification
        began = monotonic()
        try:
            verdicts = verification.verify_many([candidate for _, candidate in extracted], directory / "verification-0000",
                                                 max_samples=4, concurrent_inference=False)
        finally:
            progress["timing"]["verification"] = monotonic() - began
        generation.save(directory / "verification-0000.json", verdicts)
        for (attempt, _), verdict in zip(extracted, verdicts):
            attempt["verification"] = verdict
            if verdict["status"] == "passed":
                attempt["state"] = "passed"
            elif verdict["status"] == "failed":
                attempt["state"] = "repairs_exhausted"
        if any(v["status"] not in ("passed", "failed") for v in verdicts):
            raise RuntimeError("Verification infrastructure error")


def finalize_states(progress, directory, started):
    for attempt in progress["attempts"]:
        if attempt["state"] not in ("passed", "repairs_exhausted"):
            attempt["state"] = ("submission_unknown" if attempt["submitted"] is None else
                                progress["status"] if attempt["submitted"] else "not_run")
    progress["verification_logs"] = [str(p) for p in (directory / "verification-0000").glob("*.eval")]
    progress["timing"]["overall"] = monotonic() - started


def run(worklist, output_dir, *, source, model_path, model_manifest, mlx_python, recipe,
        reply_allowance, output_token_allowance, completion_batch_size, prefill_batch_size,
        prefill_step_size, fixtures=False, cohort_sha256=None):
    """Run one initial round. Recipe: system, initial, temperature, top_p,
    base_seed, max_tokens, max_repairs (zero). Bindings/tasks follow load_task.
    Interruptions propagate after progress saving; technical errors return error.
    """
    started = monotonic()
    worklist = list(worklist)
    mlx_python = str(Path(mlx_python).absolute())
    batching = dict(completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size, prefill_step_size=prefill_step_size)
    requests = validate(worklist, recipe, source, reply_allowance, output_token_allowance,
                        batching, fixtures, cohort_sha256)
    directory, progress, persist = prepare(worklist, requests, output_dir, recipe, source,
        model_path, model_manifest, mlx_python, batching, reply_allowance, output_token_allowance,
        fixtures, cohort_sha256)
    try:
        outcome = generate_round(directory, progress, persist, mlx_python, requests)
        extracted = extract_round(directory, worklist, outcome, progress, source, persist)
        verify_round(directory, extracted, progress, persist)
        progress.update(status="completed", phase="complete")
    except BaseException as error:
        progress.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if progress["phase"] == "verification":
            progress["verification_cleanup_confirmed"] = getattr(error, "cleanup_confirmed", False)
        if not isinstance(error, Exception):
            raise
    finally:
        finalize_states(progress, directory, started)
        persist()
    return progress


if __name__ == "__main__":
    generation.generate_batched(**json.loads(Path(sys.argv[1]).read_text()))
