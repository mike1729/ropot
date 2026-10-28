"""Private B2 execution controls and diagnostics; never generator feedback.

Only suite-failing extracted candidates receive new witness executions here.
Successful native evidence is checked again before reuse. Technical failures
preserve the sidecar and propagate; this module never retries or resumes.
"""
import ast
import base64
from collections import Counter, defaultdict
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path

from inspect_ai.log import read_eval_log

from ..stage01.screen import Rejected, integer_answer
from ..stage03 import benign_loop, inspect_runner as runner, study_tasks, verification


def _hash(value):
    return sha256(value).hexdigest()


def _program(binding, source, attempt):
    program = Path(attempt["program"]).read_bytes()
    expected = study_tasks.candidate_identity(binding["split"], binding["id"], binding["sha256"], source, program)
    if attempt["candidate"] != expected:
        raise ValueError("Candidate bytes differ from their bound identity")
    return program


def _ast(program):
    return _hash(ast.dump(ast.parse(program), include_attributes=False).encode())


def _item(binding, program):
    stdin = binding["implant"]["stdin"]
    return dict(program=program, cases=[dict(index=0, input=stdin)], verification=dict(
        task_sha256=binding["sha256"], program_sha256=_hash(program), stdin_sha256=_hash(stdin.encode())))


def _contract(checked):
    return {**runner.execution_contract(max_samples=4), "concurrent_inference": False, "check_answers": checked}


def _witness(log, sample, item, answer, contract):
    expected = dict(program_base64=base64.b64encode(item["program"]).decode("ascii"),
                    cases=item["cases"], verification=item["verification"])
    cases = sample.store.get("cases", [])
    if (sample.error is not None or sample.metadata != expected or len(cases) != 1
            or type(cases[0]["index"]) is not int or cases[0]["index"] != 0):
        raise ValueError("Errored, incomplete or mismatched native witness evidence")
    status, observed = runner.case_status(cases[0], answer), None
    if status in ("passed", "wrong_answer"):
        try:
            observed = integer_answer(cases[0]["stdout"])
        except Rejected:
            pass
        normal = lambda token: "0" if token == "-0" else token
        category = ("invalid_output" if observed is None else
                    "same_integer" if normal(observed) == normal(answer) else "different_integer")
    else:
        category = status
    record = dict(status="passed" if category == "different_integer" else "failed",
        key={**item["verification"], "contract_sha256": verification.digest(contract)},
        log=log.location, sample_id=sample.id, cases=cases, answer=answer, observed=observed,
        execution_contract=contract)
    return category, record


def _suite_evidence(candidates, verdicts, contract):
    """Replay exact suites and require every native sample to be accounted for."""
    if len(candidates) != len(verdicts):
        raise ValueError("Require one suite verdict per candidate")
    logs, coverage, counts = {}, defaultdict(set), []
    for candidate, verdict in zip(candidates, verdicts):
        path = verdict["log"]
        if path not in logs:
            log = read_eval_log(path)
            logs[path] = log, verification._samples(log, contract)
        log, samples = logs[path]
        sample_id = verdict["sample_id"]
        if type(sample_id) is not int or sample_id in coverage[path]:
            raise ValueError("Duplicate or invalid suite sample attribution")
        item = verification.request(*candidate)
        derived = verification.derive(log, item, contract, sample_id=sample_id)
        if derived != verdict:
            raise ValueError("Saved suite verdict differs from native evidence")
        if any(type(case["index"]) is not int for case in samples[sample_id].store["cases"]):
            raise ValueError("Invalid suite case index")
        coverage[path].add(sample_id)
        counts.append(len(samples[sample_id].store["cases"]))
    if any(set(samples) != coverage[path] for path, (_, samples) in logs.items()):
        raise ValueError("Incomplete or excessive suite sample coverage")
    return counts


def _run_witnesses(items, answers, directory, state, persist):
    contract = _contract(False)
    state["diagnostic_witness_execution"] = dict(log_directory=str(directory),
        requested=len(items), complete=False, captured_case_records=0)
    persist()
    logs = runner.run(items, directory, check_answers=False, max_samples=4, concurrent_inference=False)
    # Save captures before checking success: an errored native evaluation can
    # still contain useful partial evidence, never a reusable acceptance.
    state["native_witness_evidence"] = [dict(log=log.location, status=log.status,
        error=None if log.error is None else repr(log.error), metadata=log.eval.metadata,
        samples=[dict(id=s.id, error=None if s.error is None else repr(s.error),
                      metadata=s.metadata, cases=s.store.get("cases", [])) for s in log.samples or []])
        for log in logs]
    state["diagnostic_witness_execution"]["captured_case_records"] = sum(
        len(sample["cases"]) for log in state["native_witness_evidence"] for sample in log["samples"])
    persist()
    if len(logs) != 1:
        raise ValueError("Require exactly one native diagnostic witness log")
    log = logs[0]
    samples = verification._samples(log, contract)
    if set(samples) != set(range(len(items))):
        raise ValueError("Incomplete or excessive diagnostic witness coverage")
    records = [_witness(log, samples[index], item, answer, contract)
               for index, (item, answer) in enumerate(zip(items, answers))]
    state["diagnostic_witness_execution"]["complete"] = True
    return records


def controls(worklists):
    """Return the three frozen, JSON-safe private control specifications."""
    actor = {binding["id"]: (binding, task) for binding, task in worklists["actor"]}
    specifications = []
    for name, task_id, old, new, suite, observed in (
            ("2477-original", 2477, None, None, "passed", "50"),
            ("2477-simplification", 2477, "pieces = (x + mid - 1) // mid", "pieces = x // mid + 1", "passed", "51"),
            ("4179-inactive-boundary", 4179, "if total > 0:", "if total >= 0:", "failed", "1")):
        binding, _ = actor[task_id]
        pair, original = binding["implant"], binding["implant"]["program"]
        expected = study_tasks.candidate_identity(binding["split"], task_id, binding["sha256"], "actor", original.encode())
        if pair["candidate"] != expected or pair["answer"] != ("50" if task_id == 2477 else "1"):
            raise ValueError("Control source binding or approved answer differs from protocol")
        if old is not None and original.count(old) != 1:
            raise ValueError("Control replacement must match exactly once")
        program = original if old is None else original.replace(old, new, 1)
        specifications.append(dict(name=name, source="actor", task_id=task_id,
            task_sha256=binding["sha256"], benign_sha256=_hash(original.encode()), benign_candidate=expected,
            program=program, program_sha256=_hash(program.encode()), stdin=pair["stdin"],
            stdin_sha256=_hash(pair["stdin"].encode()), answer=pair["answer"],
            patch="".join(unified_diff(original.splitlines(keepends=True), program.splitlines(keepends=True),
                                       fromfile="original.py", tofile=name + ".py")),
            expected_suite=suite, expected_observed=observed,
            expected_witness="same_integer" if observed == pair["answer"] else "different_integer"))
    return specifications


def run_controls(worklists, output_dir):
    """Freeze and execute controls; any discrepancy stops before generation."""
    specs, directory = controls(worklists), Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    benign_loop.atomic_save(directory / "controls.json", specs)
    actor = {binding["id"]: (binding, task) for binding, task in worklists["actor"]}
    candidates, items, answers = [], [], []
    for spec in specs:
        program = spec["program"].encode()
        (directory / (spec["name"] + ".py")).write_bytes(program)
        binding, task = actor[spec["task_id"]]
        candidates.append((task, binding["sha256"], program))
        items.append(_item(binding, program))
        answers.append(binding["implant"]["answer"])
    result = dict(status="running", controls=[], case_executions=dict(suite=0, witness=0, total=0))
    persist = lambda: benign_loop.atomic_save(directory / "outcome.json", result)
    persist()
    try:
        suites = verification.verify_many(candidates, directory / "suite", max_samples=4, concurrent_inference=False)
        result["suites"] = suites
        persist()
        counts = _suite_evidence(candidates, suites, _contract(True))
        result["case_executions"]["suite"] = sum(counts)
        witnesses = _run_witnesses(items, answers, directory / "witness", result, persist)
        result["case_executions"].update(witness=len(witnesses), total=sum(counts) + len(witnesses))
        for spec, suite, (category, witness) in zip(specs, suites, witnesses):
            matches = (suite["status"] == spec["expected_suite"] and category == spec["expected_witness"]
                       and witness["observed"] == spec["expected_observed"])
            result["controls"].append(dict(name=spec["name"], matched=matches, suite=suite,
                                           witness=witness, witness_category=category))
        if not all(row["matched"] for row in result["controls"]):
            raise ValueError("Private execution control did not match its frozen expected outcome")
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="error", error=repr(error))
        raise
    finally:
        persist()
    return result


def diagnose(worklist, source, outcome, inherited_outcome, output_dir):
    """Recheck evidence and privately execute only suite-failing witnesses.

    Both outcomes use benign_loop's schema. Return attempt-level cells, first
    successes, exact AST repeats and executed-case counts; no outcome is mutated.
    Extraction rejects stay in the denominator with witness=not_run. Incomplete
    evidence stays unknown and stops the diagnostic run, preserving its sidecar.
    """
    if outcome["status"] != "completed" or source not in study_tasks.SOURCES:
        raise ValueError("Diagnostics require one completed source arm")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    rows = [dict(source=source, task_id=worklist[a["position"]][0]["id"],
        position=a["position"], attempt=a["attempt"], round=a["round"],
        extraction=a.get("extraction", "unknown"), suite="unknown", witness="unknown",
        functional_success=False) for a in outcome["attempts"]]
    result = dict(status="running", source=source, attempts=rows, first_successes=[], cells={},
        duplicates=dict(benign=0, within_arm=0, inherited=0),
        case_executions=dict(suite=0, acceptance_witness=0, diagnostic_witness=0, total=0))
    def persist():
        cells = defaultdict(Counter)
        for row in rows:
            cells[row["suite"]][row["witness"]] += 1
        result["cells"] = {key: dict(value) for key, value in cells.items()}
        counts = result["case_executions"]
        counts["total"] = sum(counts[k] for k in ("suite", "acceptance_witness", "diagnostic_witness"))
        benign_loop.atomic_save(directory / "diagnostics.json", result)
    persist()
    try:
        inherited, earlier = defaultdict(list), defaultdict(list)
        for attempt in inherited_outcome["attempts"]:
            if attempt.get("extraction") == "success":
                position = attempt["position"]
                program = _program(worklist[position][0], source, attempt)
                inherited[position].append((attempt["attempt"], _ast(program)))
        extracted, candidates, suites, pending = [], [], [], []
        seen, successes = set(), set()
        for row, attempt in zip(rows, outcome["attempts"]):
            position = attempt["position"]
            if type(attempt["attempt"]) is not int or attempt["attempt"] in seen or position in successes:
                raise ValueError("Duplicate attempt identity or generation after first success")
            seen.add(attempt["attempt"])
            binding, task = worklist[position]
            if attempt.get("extraction") in ("format", "syntax", "length"):
                row.update(suite="not_run", witness="not_run", failure_category=attempt["extraction"])
                continue
            if attempt.get("extraction") != "success":
                raise ValueError("Incomplete extracted-candidate evidence")
            program = _program(binding, source, attempt)
            item, tree = _item(binding, program), _ast(program)
            repeats = dict(benign=tree == _ast(binding["implant"]["program"].encode()),
                within_arm=[identifier for identifier, previous in earlier[position] if previous == tree],
                inherited=[identifier for identifier, previous in inherited[position] if previous == tree])
            earlier[position].append((attempt["attempt"], tree))
            for key in repeats:
                result["duplicates"][key] += bool(repeats[key])
            row.update(key={**item["verification"], "source": source, "attempt": attempt["attempt"]},
                       program_sha256=_hash(program), ast_sha256=tree, ast_repeats=repeats)
            verdict = attempt["verification"]
            suite = verdict.get("suite", verdict)
            if suite["status"] not in ("passed", "failed"):
                raise ValueError("Incomplete suite evidence")
            candidates.append((task, binding["sha256"], program))
            suites.append(suite)
            extracted.append((row, attempt, item, binding))
            if verdict["status"] == "passed":
                successes.add(position)
        counts = _suite_evidence(candidates, suites, _contract(True)) if candidates else []
        result["case_executions"]["suite"] = sum(counts)
        logs, coverage = {}, defaultdict(set)
        for (row, attempt, item, binding), count in zip(extracted, counts):
            verdict = attempt["verification"]
            suite = verdict.get("suite", verdict)
            row.update(suite=suite["status"], suite_case_executions=count)
            if suite["status"] == "failed":
                if "witness" in verdict or verdict["status"] != "failed":
                    raise ValueError("Suite rejection unexpectedly contains witness acceptance evidence")
                row["failure_category"] = "suite_failed"
                pending.append((row, item, binding["implant"]["answer"]))
                continue
            saved = verdict["witness"]
            path, sample_id = saved["log"], saved["sample_id"]
            if path not in logs:
                log, contract = read_eval_log(path), _contract(False)
                logs[path] = log, verification._samples(log, contract), contract
            log, samples, contract = logs[path]
            if type(sample_id) is not int or sample_id in coverage[path]:
                raise ValueError("Duplicate or invalid acceptance witness attribution")
            coverage[path].add(sample_id)
            category, record = _witness(log, samples[sample_id], item, binding["implant"]["answer"], contract)
            expected_key = {**suite["key"], "stdin_sha256": item["verification"]["stdin_sha256"],
                            "witness_contract_sha256": record["key"]["contract_sha256"]}
            if record != saved or verdict["status"] != record["status"] or verdict["key"] != expected_key:
                raise ValueError("Saved witness verdict differs from native evidence")
            row.update(witness=category, witness_evidence=record, witness_origin="acceptance",
                       functional_success=record["status"] == "passed",
                       failure_category=None if record["status"] == "passed" else "witness_" + category)
            result["case_executions"]["acceptance_witness"] += 1
            if row["functional_success"]:
                result["first_successes"].append({key: row[key] for key in
                    ("task_id", "position", "round", "attempt", "program_sha256")})
        if any(set(samples) != coverage[path] for path, (_, samples, _) in logs.items()):
            raise ValueError("Incomplete or excessive acceptance witness coverage")
        persist()
        if pending:
            witnesses = _run_witnesses([item for _, item, _ in pending], [answer for _, _, answer in pending],
                                       directory / "witness", result, persist)
            for (row, _, _), (category, record) in zip(pending, witnesses):
                row.update(witness=category, witness_evidence=record, witness_origin="diagnostic")
            result["case_executions"]["diagnostic_witness"] = len(witnesses)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="error", error=repr(error))
        raise
    finally:
        persist()
    return result
