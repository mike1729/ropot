"""Durable native intents and evidence-only recovery; never redispatch or clean up."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
from time import monotonic

from inspect_ai.log import read_eval_log, recover_eval_log
import psutil

from .generation_journal import encoded, require, snapshot, sync_directory
from . import verification as verification


class NativeBlocked(RuntimeError):
    pass


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, check=True, timeout=15).stdout.decode().strip()


def _daemon():
    identity = _docker("info", "--format", "{{.ID}}")
    require(bool(identity), "Missing Docker daemon identity")
    return identity


def _inventory(operation_id):
    project = re.compile(rf"inspect-{operation_id}-i[a-z0-9]{{6}}")
    argument = re.compile(rf"(?:--project-name=|-p)?inspect-{operation_id}-i[a-z0-9]{{6}}")
    live = False
    for process in psutil.process_iter():
        try:
            if process.uids().effective == os.geteuid():
                live |= any(argument.fullmatch(arg) is not None for arg in process.cmdline())
        except psutil.NoSuchProcess:
            continue
    projects = _docker("ps", "-a", "--filter", "label=com.docker.compose.project",
                       "--format", '{{.Label "com.docker.compose.project"}}').splitlines()
    return _daemon(), [name for name in projects if project.fullmatch(name)], live


def _write(path, content):
    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())


def _block(campaign, intent, reason):
    payload = {"operation_id": intent["operation_id"], "reason": reason}
    prior = [e for e in campaign.events if e["kind"] == "native_blocked"
             and e["operation_id"] == intent["operation_id"]]
    if not prior or prior[-1]["reason"] != reason:
        campaign._append("native_blocked", **payload)
    raise NativeBlocked(reason)


def _derive(campaign, intent, elapsed, returned_log):
    opid, reason = intent["operation_id"], "contract_unavailable"
    directory = campaign.root / "native" / opid
    try:
        require(re.fullmatch(r"[0-9a-f]{12}", opid) is not None
                and directory.resolve() == directory, "Invalid operation directory")
        current = verification.runner.contract()
        reason = "changed_contract"
        require(current == intent["contract"]
                and verification.digest(intent["contract"]) == intent["contract_sha256"], reason)
        reason = "changed_inputs"
        task_path = directory / "task.json"
        require(not task_path.is_symlink() and sha256(task_path.read_bytes()).hexdigest()
                == intent["task_copy_sha256"], reason)
        task = snapshot(json.loads(task_path.read_bytes()))
        if intent["reference_indices"] is None:
            program_path = directory / "candidate.py"
            require(not program_path.is_symlink(), reason)
            programs = [program_path.read_bytes()]
        else:
            require(task["passing_reference_indices"] == intent["reference_indices"], reason)
            programs = verification.reference_programs(task)
        require([sha256(p).hexdigest() for p in programs] == intent["program_sha256"], reason)
        reason = "ambiguous_log"
        originals = list((directory / "logs").rglob("*.eval"))
        require(len(originals) == 1, reason)
        original, recovered = originals[0], directory / "recovered.eval"
        require(not original.is_symlink() and original.resolve().is_relative_to(directory.resolve())
                and not recovered.is_symlink(), reason)
        if returned_log is not None:
            require(Path(returned_log).resolve() == original.resolve(), reason)
        reason = "not_derivable"
        log = read_eval_log(str(original))
        require(log.eval.task == opid and log.eval.metadata["contract"] == intent["contract"], reason)
        selected = original
        if log.status == "started":
            if not recovered.exists():
                recover_eval_log(str(original), output=str(recovered), overwrite=False,
                                 cleanup=False, incomplete_action="error")
            selected, log = recovered, read_eval_log(str(recovered))
        require(log.eval.task == opid and log.eval.metadata["contract"] == intent["contract"], reason)
        results = verification.reuse(selected, task, intent["task_sha256"], programs)
        project = re.compile(rf"inspect-{opid}-i[a-z0-9]{{6}}")
        require(len(log.samples or []) == len(programs) and all(project.fullmatch(
            sample.store["container"]["Config"]["Labels"]["com.docker.compose.project"])
            for sample in log.samples), reason)
        evidence = {}
        for path in dict.fromkeys((original, selected)):
            with path.open("rb") as file:
                os.fsync(file.fileno())
            evidence[str(path.relative_to(campaign.root))] = sha256(path.read_bytes()).hexdigest()
        path = original.parent
        while path.is_relative_to(campaign.root):
            sync_directory(path)
            path = path.parent
    except Exception:  # Corrupt native archives and unavailable recovery are barriers too.
        _block(campaign, intent, reason)
    payload = {"operation_id": opid, "results": results, "log": str(selected.relative_to(campaign.root)),
               "log_evidence": evidence, "elapsed_seconds": elapsed,
               "status": verification.gate_status(results) if intent["reference_indices"] is not None else results[0]["status"]}
    return campaign._append("native_verdict", **payload)


def _settle(campaign, intent, *, elapsed=None, returned_log=None):
    opid = intent["operation_id"]
    prior = [e for e in campaign.events if e["kind"] == "native_verdict" and e["operation_id"] == opid]
    verdict = prior[0] if prior else _derive(campaign, intent, elapsed, returned_log)
    _close(campaign, intent)
    return verdict


def _close(campaign, intent, **closure):
    opid = intent["operation_id"]
    try:
        daemon, projects, live = _inventory(opid)
    except Exception:
        _block(campaign, intent, "inventory_unavailable")
    if daemon != intent["daemon_id"]:
        _block(campaign, intent, "changed_daemon")
    if projects or live:
        _block(campaign, intent, "resources_present")
    return campaign._append("native_closed", operation_id=opid, **closure)


def resolve_native(campaign, operation_id, *, attestation):
    """Human-only resource attestation; never a correctness override or automatic retry."""
    attestation = snapshot(attestation)
    require(type(attestation) is dict and set(attestation) == {"operator", "quiescence_evidence"}
            and all(type(attestation[k]) is str and bool(attestation[k].strip())
                    and len(attestation[k].encode()) <= bound
                    for k, bound in (("operator", 256), ("quiescence_evidence", 8192))),
            "Require bounded human operator and positive quiescence evidence")
    digest = sha256(encoded(attestation).encode()).hexdigest()
    events = campaign.events
    intents = [e for e in events if e["kind"] == "native_intent" and e["operation_id"] == operation_id]
    require(len(intents) == 1, "Unknown native operation")
    closed = [e for e in events if e["kind"] == "native_closed" and e["operation_id"] == operation_id]
    if closed:
        require(closed[0].get("mode") == "operator_attestation"
                and closed[0]["attestation_sha256"] == digest
                and encoded(closed[0]["attestation"]) == encoded(attestation), "Different native closure")
        return closed[0]
    require(not any(e["kind"] == "native_verdict" and e["operation_id"] == operation_id for e in events),
            "Recorded verdict: use reconcile_native")
    try:
        _derive(campaign, intents[0], None, None)
    except NativeBlocked as blocked:
        if str(blocked) in ("contract_unavailable", "changed_contract"):
            raise
        return _close(campaign, intents[0], mode="operator_attestation", attestation=attestation,
                      attestation_sha256=digest)
    raise ValueError("Evidence derivable: use reconcile_native")


def reconcile_native(campaign):
    """Recover pending evidence, or leave a durable barrier; dispatch nothing."""
    events = campaign.events
    closed = {e["operation_id"] for e in events if e["kind"] == "native_closed"}
    return [_settle(campaign, e) for e in events
            if e["kind"] == "native_intent" and e["operation_id"] not in closed]


def reuse_native(campaign, intent, task, programs):
    """Re-derive closed retained evidence without dispatch, recovery or journal writes."""
    opid, events = intent["operation_id"], campaign.events
    require(type(opid) is str and re.fullmatch(r"[0-9a-f]{12}", opid) is not None, "Invalid native operation")
    directory = campaign.root / "native" / opid
    require(directory.resolve() == directory, "Invalid operation directory")
    closed = [e for e in events if e["kind"] == "native_closed" and e["operation_id"] == opid]
    verdicts = [e for e in events if e["kind"] == "native_verdict" and e["operation_id"] == opid]
    require(len(closed) == 1 and len(verdicts) <= 1, "Require closed unique native evidence")
    require([sha256(p).hexdigest() for p in programs] == intent["program_sha256"], "Native programs differ")
    if not verdicts:
        require(closed[0].get("mode") == "operator_attestation", "Closed operation lacks evidence")
        return None
    verdict = verdicts[0]
    require(verdict["log"] in verdict["log_evidence"] and bool(verdict["log_evidence"]), "Missing selected log evidence")
    for name, digest in verdict["log_evidence"].items():
        path = campaign.root / name
        require(path.is_relative_to(directory) and path.resolve() == path
                and sha256(path.read_bytes()).hexdigest() == digest, "Native log evidence changed")
    selected = campaign.root / verdict["log"]
    header = read_eval_log(str(selected), header_only=True)
    require(header.eval.task == opid and encoded(header.eval.metadata["contract"]) == encoded(intent["contract"])
            and verification.digest(intent["contract"]) == intent["contract_sha256"], "Native log binding differs")
    results = verification.reuse(selected, task, intent["task_sha256"], programs)
    status = verification.gate_status(results) if intent["reference_indices"] is not None else results[0]["status"]
    require(encoded(results) == encoded(verdict["results"]) and status == verdict["status"], "Native verdict changed")
    return {**verdict, "results": results}


def run_native(campaign, task, task_sha256, program=None, *, reservation_seq=None):
    campaign.require_native_clear()
    require(reservation_seq is None or (program is not None and type(reservation_seq) is int
            and any(e["seq"] == reservation_seq and e["kind"] == "generation_reserved" for e in campaign.events)),
            "Require candidate generation reservation owner")
    task = snapshot(task)
    programs = verification.reference_programs(task) if program is None else [program]
    for value in programs:
        verification.request(task, task_sha256, value)
    contract, daemon = verification.runner.contract(), _daemon()
    previous = {e["operation_id"] for e in campaign.events if e["kind"] == "native_intent"}
    operation_id = secrets.token_hex(6)
    require(operation_id not in previous, "Native operation ID collision")
    directory = campaign.root / "native" / operation_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "logs").mkdir()
    raw = (encoded(task) + "\n").encode()
    _write(directory / "task.json", raw)
    if program is not None:
        _write(directory / "candidate.py", program)
    for path in (directory / "logs", directory, directory.parent, campaign.root):
        sync_directory(path)
    intent = campaign._append("native_intent", operation_id=operation_id, task_sha256=task_sha256,
        task_copy_sha256=sha256(raw).hexdigest(), program_sha256=[sha256(p).hexdigest() for p in programs],
        reference_indices=task["passing_reference_indices"] if program is None else None,
        contract=contract, contract_sha256=verification.digest(contract), daemon_id=daemon, reservation_seq=reservation_seq)
    started = monotonic()
    if program is None:
        result = verification.reference_gate(task, task_sha256, directory / "logs", operation_id=operation_id)
    else:
        result = verification.verify(task, task_sha256, program, directory / "logs", operation_id=operation_id)
    return _settle(campaign, intent, elapsed=monotonic() - started, returned_log=result["log"])
