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


def _samples(log, metadata):
    """Validate the evaluation and index its samples once per worklist."""
    if log.status != "success" or log.error is not None or log.eval.metadata != metadata:
        raise ValueError("Unsuccessful evaluation or changed execution contract")
    samples = {}
    for sample in log.samples or []:
        if type(sample.id) is not int or sample.id < 0 or sample.id in samples:
            raise ValueError("Invalid or duplicate native sample ID")
        samples[sample.id] = sample
    return samples


def derive(log, item, metadata, *, sample_id=None):
    """Bind one exact request; multi-sample evidence needs an explicit sample ID."""
    samples = _samples(log, metadata)
    if sample_id is None:
        if len(samples) != 1:
            raise ValueError("Multi-sample logs require an explicit sample ID")
        sample_id = 0
    if type(sample_id) is not int or sample_id < 0:
        raise ValueError("Require a nonnegative integer sample ID")
    return _derive_sample(samples.get(sample_id), item, log.location, digest(metadata))


def _derive_sample(sample, item, log_path, contract_sha256):
    """Check request identity and case coverage, then project public feedback."""
    expected = {"program_base64": base64.b64encode(item["program"]).decode("ascii"),
                "cases": item["cases"], "verification": item["verification"]}
    if sample is None or sample.error is not None or sample.metadata != expected:
        raise ValueError("Missing, errored or mismatched native sample")
    cases, results = item["cases"], sample.store.get("cases", [])
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
    return {"status": verdict, "key": {**item["verification"], "contract_sha256": contract_sha256},
            "log": log_path, "sample_id": sample.id,
            "feedback": {"status": verdict, "visible": public, "hidden": hidden}}


def verify(task, task_sha256, program, log_dir, *, concurrent_inference=False):
    """Execute one fresh sandbox and return private evidence plus public feedback.

    Preflight/setup exceptions and interruptions propagate. Native evaluation or
    sample failures return error; they do not establish a candidate failure.
    """
    return verify_many([(task, task_sha256, program)], log_dir,
                       concurrent_inference=concurrent_inference)[0]


def verify_many(candidates, log_dir, *, max_samples=4, concurrent_inference=False):
    """One bounded native eval for (task, task digest, program bytes) tuples.

    All requests keep their own sandbox. An errored eval gives no reusable
    verdicts; completed captures remain evidence and are never redispatched here.
    """
    items = [request(*candidate) for candidate in candidates]
    if not items:
        raise ValueError("Require at least one candidate")
    log = runner.run(items, log_dir, check_answers=True, max_samples=max_samples,
                     concurrent_inference=concurrent_inference)[0]
    contract_sha256 = digest(log.eval.metadata)
    if log.status != "success" or log.error is not None or any(s.error is not None for s in log.samples or []):
        return [{"status": "error", "key": {**item["verification"], "contract_sha256": contract_sha256},
                 "log": log.location, "sample_id": i, "feedback": {"status": "verification_error"}}
                for i, item in enumerate(items)]
    samples = _samples(log, log.eval.metadata)
    return [_derive_sample(samples.get(i), item, log.location, contract_sha256) for i, item in enumerate(items)]


def reuse(log_path, task, task_sha256, program, *, sample_id=None, max_samples=4, concurrent_inference=False):
    """Explicitly recheck successful native evidence; never fall back to execution."""
    item = request(task, task_sha256, program)
    metadata = {**runner.execution_contract(max_samples=max_samples),
                "concurrent_inference": concurrent_inference, "check_answers": True}
    return derive(read_eval_log(str(log_path)), item, metadata, sample_id=sample_id)
