"""Frozen loop inputs and public prompt/artifact helpers; dispatch is separate."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from . import local_generation as generation
from . import study_tasks as study
from .generation_journal import encoded, require, sync_directory

RECIPE_FIELDS = {"system", "instruction", "max_replies", "max_tokens", "wall_time_seconds",
                 "temperature", "top_p", "seeds"}
CASE_FIELDS = {"input", "answer", "status", "stdout", "stderr", "returncode", "diagnostic"}
STATUSES = {"passed", "wrong_answer", "timeout", "output_limit", "stdout_limit",
            "stderr_limit", "invalid_stdout", "nonzero_exit"}
SHARED = ("Return a complete Python 3.12 program reading stdin and writing stdout. "
          "Use raw Python or a sole Python fence, without explanation.")
FORMAT_REPAIR = ("The preceding reply is not valid Python in the required format. "
                 "Return a complete program in that format.")


def policy_sha256():
    sources = {Path(p).name: sha256(Path(p).read_bytes()).hexdigest() for p in (__file__, generation.__file__)}
    return sha256(encoded(sources).encode("utf-8")).hexdigest()


def run_record(loaded, experiment_id, groups, recipes, assigned):
    activation = study.activate(loaded, groups)
    record = study.experiment_record(loaded, experiment_id, activation, study.snapshot(recipes))
    for source, recipe in record["recipes"].items():
        require(set(recipe) == RECIPE_FIELDS, "Unexpected recipe fields")
        require(all(type(recipe[k]) is str and recipe[k].strip() for k in ("system", "instruction")),
                "Require nonempty system and instruction")
        require(type(recipe["max_replies"]) is int and recipe["max_replies"] > 0, "Invalid reply cap")
        require(type(recipe["seeds"]) is list and len(recipe["seeds"]) == recipe["max_replies"], "Invalid seeds")
        for seed in recipe["seeds"]:
            generation.prepare_request(generation.request_for(source, [{"role": "system", "content": "Validate"},
                {"role": "user", "content": "Validate"}], seed=seed, **{k: recipe[k] for k in
                ("temperature", "top_p", "max_tokens", "wall_time_seconds")}))
    assigned = study.snapshot(assigned)
    require(type(assigned) is list and assigned and all(type(i) is int for i in assigned)
            and len(set(assigned)) == len(assigned), "Require distinct integer assigned IDs")
    require(assigned == [t["id"] for t in activation["tasks"] if t["id"] in assigned],
            "Assignments must follow the current activation order")
    return {**record, "assigned_task_ids": assigned, "policy_sha256": policy_sha256()}


def validate_run(loaded, record):
    record = study.snapshot(record)
    try:
        rebuilt = run_record(loaded, record["experiment_id"], record["activation"]["groups"],
                             record["recipes"], record["assigned_task_ids"])
    except (KeyError, TypeError):
        raise ValueError("Invalid run record") from None
    require(encoded(rebuilt) == encoded(record), "Run differs from its canonical record or current policy")
    return rebuilt


def project_feedback(feedback):
    feedback = study.snapshot(feedback)
    require(type(feedback) is dict and set(feedback) == {"status", "visible", "hidden"}
            and feedback["status"] == "failed", "Require failed verification feedback")
    visible = feedback["visible"]
    require(type(visible) is list and bool(visible), "Require visible feedback")
    for case in visible:
        require(type(case) is dict and set(case) == CASE_FIELDS
                and type(case["status"]) is str and case["status"] in STATUSES,
                "Invalid visible feedback schema")
        require(all(type(case[k]) is str for k in ("input", "answer"))
                and all(case[k] is None or type(case[k]) is str for k in ("stdout", "stderr", "diagnostic"))
                and (case["returncode"] is None or type(case["returncode"]) is int), "Invalid visible feedback types")
    failed = [case for case in visible if case["status"] != "passed"]
    require(feedback["hidden"] == ("not_run" if failed else "failed"), "Inconsistent hidden feedback")
    projected = []
    if failed:
        case = failed[0]
        result = {"status": case["status"], "returncode": case["returncode"], "truncated": {}}
        for key, limit in (("input", 128), ("answer", 64), ("stdout", 256), ("stderr", 128)):
            value = case[key]
            if key in ("stdout", "stderr") and value is not None:
                value = re.sub(r"/work/case-\d+/", "", value)
            raw = value.encode("utf-8") if value is not None else b""
            clipped = raw[-limit:] if key == "stderr" else raw[:limit]
            result[key] = clipped.decode("utf-8", "ignore") if value is not None else None
            result["truncated"][key] = len(raw) > limit
        projected.append(result)
    result = {"status": "failed", "visible": projected, "hidden": feedback["hidden"]}
    require(len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) <= 4096,
            "Projected feedback exceeds its byte cap")
    return result


def messages(task, recipe, previous=None, feedback=None):
    question, starter = task["question"], task.get("starter_code", "")
    require(type(question) is str and type(starter) is str, "Require task text")
    require(all(type(recipe[k]) is str and recipe[k].strip() for k in ("system", "instruction")), "Require recipe text")
    require((previous is None and feedback is None) or type(previous) is str, "Invalid repair context")
    user = recipe["instruction"] + "\n\nProblem:\n" + question
    if starter:
        user += "\n\nStarter code:\n" + starter
    result = [{"role": "system", "content": recipe["system"] + "\n\n" + SHARED},
              {"role": "user", "content": user}]
    if previous is not None:
        repair = FORMAT_REPAIR if feedback is None else "Repair using this public feedback:\n" + json.dumps(
            project_feedback(feedback), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        result.extend([{"role": "assistant", "content": previous}, {"role": "user", "content": repair}])
    return result


def publish_candidate(directory, program):
    require(type(program) is bytes, "Require exact program bytes")
    directory = Path(directory)
    final = directory / "candidate.py"
    if not os.path.lexists(final):
        descriptor, temporary = tempfile.mkstemp(prefix=".candidate-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(program)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, final)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
    with os.fdopen(os.open(final, os.O_RDONLY | os.O_NOFOLLOW), "rb") as existing:
        require(existing.read() == program, "Published candidate differs")
    sync_directory(directory)
    return final
