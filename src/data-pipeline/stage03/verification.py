"""Verify candidates against the published suite; native logs are private evidence.

Use task and binding['sha256'] from study_tasks.load_task, or the same fixture
artifact bytes. Callers must keep these paired and the task unchanged. Only the
returned feedback may enter repair prompts. No references or model calls run.
"""

import base64
from hashlib import sha256
import json

from inspect_ai.log import read_eval_log

from . import inspect_runner as runner
from .study_tasks import DIGEST


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def request(task, task_sha256, program):
    if type(program) is not bytes or type(task_sha256) is not str or DIGEST.fullmatch(task_sha256) is None:
        raise ValueError("Require exact program bytes and a task artifact digest")
    return {"program": program,
            "cases": [{key: case[key] for key in ("index", "input", "answer", "visible")} for case in task["cases"]],
            "verification": {"task_sha256": task_sha256, "program_sha256": sha256(program).hexdigest()}}


def derive(log, item, metadata):
    """Require complete native evidence for exactly one checked request."""
    if log.status != "success" or log.error is not None or log.eval.metadata != metadata:
        raise ValueError("Unsuccessful evaluation or changed execution contract")
    samples = log.samples or []
    expected = {"program_base64": base64.b64encode(item["program"]).decode("ascii"),
                "cases": item["cases"], "verification": item["verification"]}
    if (len(samples) != 1 or samples[0].id != 0 or samples[0].error is not None
            or samples[0].metadata != expected):
        raise ValueError("Missing, duplicate, errored or mismatched native sample")
    cases, results = item["cases"], samples[0].store.get("cases", [])
    if not results or len(results) > len(cases):
        raise ValueError("Missing or excessive case results")
    public, hidden = [], "not_run"
    for offset, (case, result) in enumerate(zip(cases, results)):
        if result["index"] != case["index"]:
            raise ValueError("Results differ from publication order")
        status = runner.case_status(result, case["answer"])
        if status != "passed" and offset != len(results) - 1:
            raise ValueError("Execution continued after failure")
        if case["visible"]:
            public.append({"input": case["input"], "answer": case["answer"], "status": status,
                           **{key: result.get(key) for key in ("stdout", "stderr", "returncode", "diagnostic")}})
        else:
            hidden = "passed" if status == "passed" else "failed"
    if status == "passed" and len(results) != len(cases):
        raise ValueError("Incomplete passing coverage")
    verdict = "passed" if status == "passed" else "failed"
    return {"status": verdict, "key": {**item["verification"], "contract_sha256": digest(metadata)},
            "log": log.location, "sample_id": 0,
            "feedback": {"status": verdict, "visible": public, "hidden": hidden}}


def verify(task, task_sha256, program, log_dir):
    """Execute one fresh sandbox and return private evidence plus public feedback.

    Preflight/setup exceptions and interruptions propagate. Native evaluation or
    sample failures return error; they do not establish a candidate failure.
    """
    item = request(task, task_sha256, program)
    log = runner.run([item], log_dir, check_answers=True)[0]
    if log.status != "success" or log.error is not None or any(s.error is not None for s in log.samples or []):
        return {"status": "error", "key": {**item["verification"], "contract_sha256": digest(log.eval.metadata)},
                "log": log.location, "sample_id": 0, "feedback": {"status": "verification_error"}}
    return derive(log, item, log.eval.metadata)


def reuse(log_path, task, task_sha256, program):
    """Explicitly recheck successful native evidence; never fall back to execution."""
    item = request(task, task_sha256, program)
    metadata = {**runner.execution_contract(), "concurrent_inference": False, "check_answers": True}
    return derive(read_eval_log(str(log_path)), item, metadata)
