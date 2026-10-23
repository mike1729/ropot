"""Host-only Linux reference gate and deterministic verification; logs are private.

Callers must obtain task_sha256 from the same verified bytes as task (load_task's
binding, or qualified train artifact bytes). No generator-source label affects reuse.
"""

import base64
from hashlib import sha256
import json

from inspect_ai.log import read_eval_log

from . import inspect_runner as runner
from .screen import output_matches
from .study_tasks import DIGEST


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ordered_cases(task):
    """Stage 02's fixed screen membership, then all remaining cases in four groups."""
    ranked = sorted(task["cases"], key=lambda c: (sha256(
        f"ropot-screen-v1|{task['split']}|{task['id']}|{c['index']}".encode()).hexdigest(), c["index"]))
    if not ranked or not any(c["visible"] for c in ranked) or all(c["visible"] for c in ranked):
        raise ValueError("Verification requires visible and hidden stored cases")
    selected = {next(c["index"] for c in ranked if c["visible"]),
                next(c["index"] for c in ranked if not c["visible"])}
    for case in ranked:
        if len(selected) >= min(8, len(ranked)):
            break
        selected.add(case["index"])
    ordered = sorted(ranked, key=lambda c: (not c["visible"], c["index"] not in selected))
    return [{key: c[key] for key in ("index", "input", "answer", "visible")} for c in ordered]


def request(task, task_sha256, program):
    if type(program) is not bytes or type(task_sha256) is not str or DIGEST.fullmatch(task_sha256) is None:
        raise ValueError("Require exact program bytes and task artifact digest")
    return {"program": program, "cases": ordered_cases(task), "verification": {
        "task_sha256": task_sha256, "program_sha256": sha256(program).hexdigest()}}


def verdict(sample, item, contract):
    """Reconstruct from native captures, refusing partial/mismatched evidence."""
    expected = {"program_base64": base64.b64encode(item["program"]).decode("ascii"),
                "cases": item["cases"], "verification": item["verification"]}
    if sample.metadata != expected:
        raise ValueError("Native sample differs from the full verification request")
    if sample.error is not None:
        return {"status": "error", "feedback": {"status": "verification_error"}}
    runtime, container = sample.store["runtime"], sample.store["container"]
    if (runtime["system"] != "Linux" or not runtime["python"].startswith("3.12.")
            or runtime["utf8_mode"] != 1 or container["Image"] != contract["image"]["Id"]):
        raise ValueError("Observed execution differs from the contract")
    cases, results = item["cases"], sample.store["cases"]
    if not results or len(results) > len(cases):
        raise ValueError("Missing or excessive case results")
    checked = []
    for case, raw in zip(cases, results):
        if raw["index"] != case["index"]:
            raise ValueError("Case results differ from prescribed order")
        result = dict(raw)
        if result["status"] in ("passed", "wrong_answer", "completed"):
            result["status"] = ("passed" if result["returncode"] == 0 and isinstance(result["stdout"], str)
                                and "\ufffd" not in result["stdout"]
                                and len(result["stdout"].encode()) < runner.STDOUT_LIMIT_BYTES
                                and isinstance(result["stderr"], str)
                                and len(result["stderr"].encode()) < runner.STDERR_LIMIT_BYTES
                                and output_matches(result["stdout"], case["answer"]) else "wrong_answer")
        checked.append(result)
    visible_count = sum(c["visible"] for c in cases)
    visible_passed = len(checked) >= visible_count and all(r["status"] == "passed" for r in checked[:visible_count])
    expected_count = (next((i + 1 for i, r in enumerate(checked) if i >= visible_count
                           and r["status"] != "passed"), len(cases)) if visible_passed else visible_count)
    if len(checked) != expected_count:
        raise ValueError("Incomplete execution or hidden execution after visible failure")
    passed = len(checked) == len(cases) and all(r["status"] == "passed" for r in checked)
    public = [{"input": c["input"], "answer": c["answer"],
               **{k: r[k] for k in ("status", "stdout", "stderr", "returncode")},
               "diagnostic": r.get("diagnostic")} for c, r in zip(cases, checked) if c["visible"]]
    return {"status": "passed" if passed else "failed", "feedback": {
        "status": "passed" if passed else "failed", "visible": public,
        "hidden": "passed" if passed else "failed" if visible_passed else "not_run"}}


def derive(log, items, current_contract):
    """Shared verdict derivation for fresh logs and explicit reuse; no redispatch."""
    if log.status not in ("success", "error") or log.eval.metadata["contract"] != current_contract:
        raise ValueError("Incomplete log or changed execution contract")
    samples = {s.id: s for s in log.samples or []}
    if len(log.samples or []) != len(items) or len(samples) != len(items) or set(samples) != set(range(len(items))):
        raise ValueError("Incomplete or duplicate native samples")
    return [{"key": {**item["verification"], "contract_sha256": digest(current_contract)},
             "sample_id": i, **verdict(samples[i], item, current_contract)} for i, item in enumerate(items)]


def verify(task, task_sha256, program, log_dir):
    """Run one fresh sandbox; retain exact native evidence and public-only feedback."""
    item = request(task, task_sha256, program)
    log = runner.run([item], log_dir)[0]
    return {"log": log.location, **derive(log, [item], runner.contract())[0]}


def reference_programs(task):
    indices = task["passing_reference_indices"]
    if len(indices) != 3 or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != 3:
        raise ValueError("Require exactly three qualified reference indices")
    programs = [task["solutions"][i].encode("utf-8") for i in indices]
    if len(set(programs)) != 3:
        raise ValueError("Qualified references must be distinct")
    return programs


def reference_gate(task, task_sha256, log_dir):
    """Check the same three qualified references, never replacing a failing one."""
    items = [request(task, task_sha256, p) for p in reference_programs(task)]
    log = runner.run(items, log_dir, concurrent_inference=False)[0]
    results = derive(log, items, runner.contract())
    status = "failed" if any(r["status"] == "failed" for r in results) else (
        "passed" if all(r["status"] == "passed" for r in results) else "error")
    return {"status": status, "log": log.location, "reference_indices": task["passing_reference_indices"],
            "results": results}


def reuse(log_path, task, task_sha256, programs):
    """Explicitly re-derive evidence; orchestration must retain/account for every attempt."""
    items = [request(task, task_sha256, p) for p in programs]
    return derive(read_eval_log(str(log_path)), items, runner.contract())
