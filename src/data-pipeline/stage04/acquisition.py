"""Bounded source acquisition; private evidence. See docs/stage-04-batch-02-plan.md."""
from collections import Counter
import json
from pathlib import Path

from ..stage03.local_generation import file_hash as _hash
from . import audits, proposals, witnesses

FIELDS = ("split", "id", "task_sha256", "source", "program_sha256")
FORMAT = "Return one complete <witness>...</witness> block."
DIFFERENT = "Propose a different input."
EXECUTION = "The proposal did not pass the execution check. Propose a different input."
VALIDITY = "The input was not approved as valid. Propose a different input."
REJECTIONS = {"format_failure": FORMAT, "token_limit": FORMAT, "duplicate": DIFFERENT,
              **dict.fromkeys(("reference_execution_failure", "reference_disagreement",
                              "benign_execution_failure", "benign_disagreement"), EXECUTION)}


def _read(path):
    return json.loads(Path(path).read_bytes())


def _identity(candidate):
    return tuple(candidate[k] for k in FIELDS)


def _check(files):
    for path, digest in files.items():
        if _hash(path) != digest:
            raise ValueError(f"Changed acquisition evidence: {path}")


def _qualification(directory):
    report = audits.qualification(directory)
    settings = _read(Path(directory) / "inputs.json")
    if not report["passed"] or settings["source_sha256"] != _hash(audits.__file__):
        raise ValueError("Require passing qualification of the unchanged audits.py")
    if any(call["model"] != settings["auditors"][a]["model"]
           or call["effort"] != settings["auditors"][a]["effort"]
           for row in report["cases"] for a, call in row["audits"].items()):
        raise ValueError("Qualification calls differ from saved model/effort")
    versions = {a: report["cases"][0]["audits"][a]["version"] for a in audits.AUDITORS}
    return settings, versions


def _runs(previous):
    if previous is None:
        return []
    directory = Path(previous).absolute()
    inputs, outcome = _read(directory / "inputs.json"), _read(directory / "outcome.json")
    _check({str(directory / "inputs.json"): outcome["inputs_sha256"], **outcome["files"]})
    _check({str(Path(inputs["qualification_dir"]) / "inputs.json"): inputs["qualification_sha256"]})
    if inputs["previous"]:
        _check({str(Path(inputs["previous"]["directory"]) / "outcome.json"): inputs["previous"]["sha256"]})
    return _runs(inputs["previous"]["directory"] if inputs["previous"] else None) + [(directory, inputs, outcome)]


def _b1(directory, wanted):
    """Read the reclassification as an overlay, not another generation round."""
    directory = Path(directory).absolute()
    inputs = _read(directory / "inputs.json")
    _check(inputs["original_sha256"])
    rows, rounds, seen = [], {}, set()
    files = dict(inputs["original_sha256"])
    files.update({str(directory / name): _hash(directory / name) for name in ("inputs.json", "outcomes.json")})
    for row in _read(directory / "outcomes.json"):
        if _identity(row["candidate"]) not in wanted:
            continue
        original = str(Path(row["original_outcome"]).absolute())
        if original not in inputs["original_sha256"]:
            raise ValueError("Unpinned B1 original outcome")
        rounds[original] = _read(original)
        prior = rounds[original]["attempts"][row["attempt"]]
        key = (original, row["attempt"])
        if key in seen or any(prior[k] != row[k] for k in ("candidate", "seed", "submitted")):
            raise ValueError("Duplicate or mismatched B1 overlay association")
        seen.add(key)
        rows.append(dict(row, round=str(Path(original).parent)))
    return rows, rounds, files


def _evidence(runs, first, wanted):
    rows, rounds, files = _b1(first["b1_dir"], wanted) if first["b1_dir"] else ([], {}, {})
    _check(first["b1_files"])
    for directory, inputs, outcome in runs:
        for entry in outcome["rounds"]:
            path = Path(entry["proposals"])
            if (path / "outcome.json").exists():
                saved = _read(path / "outcome.json")
                rounds[str(path / "outcome.json")] = saved
                rows.extend(dict(row, round=str(path)) for row in saved["attempts"])
            elif path.exists():
                raise ValueError("Proposal directory lacks submission evidence")
    for row in rows:
        if _identity(row["candidate"]) not in wanted:
            raise ValueError("Proposal outside saved scope")
        reply = Path(row["round"]) / "generation" / f"reply-{row['attempt']:04d}.txt"
        if reply.exists():
            files[str(reply)] = _hash(reply)
        if "input" in row:
            _check({row["input"]: row["stdin_sha256"]})
            files[row["input"]] = row["stdin_sha256"]
    for path in rounds:
        files[path] = _hash(path)
        inputs_path = Path(path).with_name("inputs.json")
        if inputs_path.exists():
            files[str(inputs_path)] = _hash(inputs_path)
    return rows, rounds, files


def _batches(runs):
    return [Path(entry["audits"]) for _, _, outcome in runs for entry in outcome["rounds"]
            if (Path(entry["audits"]) / "inputs.json").exists()]


def _calls(batch):
    return sorted(batch.glob("calls/*/*/call-*"))


def _binding(row):
    return {k: row[k] for k in ("round", "attempt", "candidate", "stdin_sha256")}


def _audit_state(batches, settings):
    """Read fresh decisions and one call snapshot for retries and run accounting.

    The qualified decision reader retains its own evidence checks. Pending
    entries carry each saved batch's unused ceiling; missing records still use slots.
    """
    decisions, pending, calls = {}, [], []
    for batch in batches:
        saved = _read(batch / "inputs.json")
        if (saved["source_sha256"] != settings["source_sha256"] or saved["timeout"] != settings["timeout"]
                or any(any(saved["auditors"][a][k] != settings["auditors"][a][k]
                           for k in ("model", "effort")) for a in audits.AUDITORS)):
            raise ValueError("Saved audit batch differs from qualified setup")
        rows = audits.decide(batch)
        batch_calls = [(path, _read(path / "record.json") if (path / "record.json").exists() else {})
                       for path in _calls(batch)]
        calls.extend(batch_calls)
        invalid = Counter((path.parent.parent.name, path.parent.name) for path, record in batch_calls
                          if record.get("status") == "invalid_reply")
        needs_dispatch = False
        for row in rows:
            states = [(row["audits"][a], invalid[row["packet_id"], a]) for a in audits.AUDITORS]
            needs_dispatch |= any(not success and count < 2 for success, count in states)
            row["unavailable"] = any(not success and count >= 2 for success, count in states)
            binding = row["binding"]
            key = (binding["round"], binding["attempt"])
            if key in decisions:
                raise ValueError("Multiple audit packets for one proposal")
            decisions[key] = row
        if needs_dispatch:
            pending.append((batch, saved["call_allowance"] - len(batch_calls)))
    return decisions, pending, calls


def _programs(scope, rows, decisions):
    result = []
    for candidate in scope:
        history = [r for r in rows if r["candidate"] == candidate]
        count = sum(r.get("submitted") is not False for r in history)
        state, approved, turns, previous_inputs = "eligible", None, [], []
        for row in history:
            if "input" in row:
                previous_inputs.append(Path(row["input"]).read_bytes().decode("utf-8"))
            decision = decisions.get((row["round"], row["attempt"]))
            feedback = REJECTIONS.get(row["state"])
            if row["state"] == "execution_agrees":
                if decision is not None and decision["binding"] != _binding(row):
                    raise ValueError("Audit binding differs from execution agreement")
                if decision is None or decision["status"] == "pending":
                    state = "pending"
                elif decision["unavailable"]:
                    state = "audit_unavailable"
                elif decision["status"] == "approved":
                    approved = _binding(row)
                else:
                    feedback = VALIDITY if decision["reason"] == "validity" else DIFFERENT
            if feedback:
                reply = Path(row["round"]) / "generation" / f"reply-{row['attempt']:04d}.txt"
                turns.extend([dict(role="assistant", content=reply.read_bytes().decode("utf-8")),
                              dict(role="user", content=feedback)])
        if candidate["task_sha256"] in witnesses.EXHAUSTED_TASKS:
            state = "input_domain_exhausted"
        elif approved:
            state = "approved"
        elif state == "eligible" and count >= 3:
            state = "attempts_exhausted"
        result.append(dict(candidate=candidate, attempts=count, state=state, approved=approved,
                           turns=turns, previous_inputs=previous_inputs))
    return result


def acquire(output_dir, *, source, qualification_dir, reply_allowance, output_token_allowance,
            audit_call_allowance, scope=None, previous_run=None, b1_dir=None, recipe=None,
            model_path, model_manifest, mlx_python, completion_batch_size, prefill_batch_size,
            prefill_step_size, fixtures=False):
    """Acquire one source, or resume previous_run in a fresh directory.
    scope lists full candidates; omit it on resume. Limits cover NEW work only.
    Audit setup comes from qualification_dir. Preflight raises; runtime errors
    return status=error; interrupts retain evidence and propagate.
    """
    qualified, versions = _qualification(qualification_dir)
    runs = _runs(previous_run)
    if source not in ("actor", "synthetic") or (runs and b1_dir is not None):
        raise ValueError("Require one source and either B1 history or a previous run")
    if any(type(n) is not int or n < 0 for n in (reply_allowance, output_token_allowance, audit_call_allowance)):
        raise ValueError("Require nonnegative integer run allowances")
    if runs:
        first = runs[0][1]
        if any(i["source"] != source for _, i, _ in runs) or (scope is not None and scope != first["scope"]):
            raise ValueError("Cannot change saved source/scope")
        scope = first["scope"]
    if not scope or len({_identity(c) for c in scope}) != len(scope):
        raise ValueError("Require an explicit nonempty scope of distinct candidates")
    manifest = json.loads(witnesses._read(witnesses.BENIGN_DIR, "manifest.json", witnesses.SELECTION_SHA256))
    order = [s["candidate"] for s in manifest["selections"] if s["candidate"]["source"] == source]
    wanted = {_identity(c) for c in scope}
    if not wanted <= {_identity(c) for c in order}:
        raise ValueError("Scope outside pinned source selection")
    selected = [c for c in order if _identity(c) in wanted]
    records = witnesses.load_worklist([(c["split"], c["id"], source) for c in selected])
    if [r["selection"]["candidate"] for r in records] != selected:
        raise ValueError("Loaded candidate differs from scope")
    by_id = {_identity(r["selection"]["candidate"]): r for r in records}
    recipe = dict(proposals.DEFAULT_RECIPE if recipe is None else recipe)
    if not runs:
        first = dict(b1_dir=str(Path(b1_dir).absolute()) if b1_dir else None, b1_files={})
    rows, rounds, files = _evidence(runs, first, wanted)
    if not runs:
        first["b1_files"] = files
    batches = _batches(runs)
    _, _, initial_calls = _audit_state(batches, qualified)
    baseline_call_count = len(initial_calls)
    directory = Path(output_dir).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    inputs = dict(source=source, qualification_dir=str(Path(qualification_dir).absolute()),
        qualification_sha256=_hash(Path(qualification_dir) / "inputs.json"),
        selection_sha256=witnesses.SELECTION_SHA256, recipe=recipe, fixtures=fixtures,
        reply_allowance=reply_allowance, output_token_allowance=output_token_allowance,
        audit_call_allowance=audit_call_allowance,
        previous=dict(directory=str(Path(previous_run).absolute()), sha256=_hash(Path(previous_run) / "outcome.json")) if runs else None,
        source_sha256={str(Path(p).absolute()): _hash(p) for p in (__file__, proposals.__file__, audits.__file__)},
        settings=dict(model_path=str(model_path), model_manifest=str(model_manifest), mlx_python=str(mlx_python),
                      completion_batch_size=completion_batch_size, prefill_batch_size=prefill_batch_size,
                      prefill_step_size=prefill_step_size))
    if not runs:
        inputs.update(first, scope=scope)
    proposals.generation.save(directory / "inputs.json", inputs)
    progress = dict(status="running", inputs_sha256=_hash(directory / "inputs.json"), rounds=[], files=files)
    current = (directory, inputs, progress)

    def collect():
        nonlocal rows, rounds, files, batches
        rows, rounds, files = _evidence([*runs, current], first, wanted)
        batches = _batches([*runs, current])
        for batch in batches:
            for path in [batch / "inputs.json", *batch.glob("*.txt")]:
                files[str(path)] = _hash(path)
        decisions, pending, calls = _audit_state(batches, qualified)
        programs = _programs(selected, rows, decisions)
        progress["programs"] = [{k: v for k, v in p.items() if k not in ("turns", "previous_inputs")} for p in programs]
        progress["files"] = files
        costs = list(rounds.values())
        progress["generation"] = {k: sum(c[k] for c in costs) for k in ("committed_replies", "committed_tokens", "observed_tokens")}
        progress["generation"].update(cost_basis="whole_referenced_rounds",
            out_of_scope_attempts=sum(_identity(r["candidate"]) not in wanted for c in costs for r in c["attempts"]),
            known_submissions=None if any(c["known_submissions"] is None for c in costs)
            else sum(c["known_submissions"] for c in costs), uncertain_submissions=sum(r.get("submitted") is None for r in rows),
            usage_complete=all(c["usage_complete"] for c in costs))
        progress["audits"] = dict(new_calls=len(calls) - baseline_call_count,
            total_calls=len(calls), unknown_usage_calls=sum(c.get("usage") is None for _, c in calls),
            calls=[dict(path=str(p), usage=c.get("usage"), version=c.get("version")) for p, c in calls],
            version_differences=[dict(path=str(p), expected=versions[p.parent.name], actual=c["version"])
                                 for p, c in calls if c.get("version") and c["version"] != versions[p.parent.name]])
        return programs, decisions, pending

    def persist():
        proposals.control.atomic_save(directory / "outcome.json", progress)

    try:
        while True:
            programs, decisions, pending = collect()
            persist()
            remaining = audit_call_allowance - progress["audits"]["new_calls"]
            if pending:
                batch, unused_ceiling = pending[0]
                if remaining < unused_ceiling:
                    raise RuntimeError("Audit allowance cannot cover saved batch's unused ceiling")
                audits.dispatch(batch)
                continue
            agreements = [r for r in rows if r["state"] == "execution_agrees"
                          and (r["round"], r["attempt"]) not in decisions
                          and r["candidate"]["task_sha256"] not in witnesses.EXHAUSTED_TASKS]
            path = directory / f"round-{len(progress['rounds']):04d}"
            if agreements:
                if remaining < 2 * len(agreements):
                    raise RuntimeError("Require two remaining audit calls per packet")
                progress["rounds"].append(dict(proposals=str(path / "proposals"), audits=str(path / "audits")))
                persist()
                items = [dict(statement=by_id[_identity(r["candidate"])]["task"]["question"],
                    stdin=Path(r["input"]).read_bytes().decode("utf-8"),
                    published_inputs=[c["input"] for c in by_id[_identity(r["candidate"])]["task"]["cases"]],
                    binding=_binding(r)) for r in agreements]
                audits.run(items, path / "audits", call_allowance=min(remaining, 4 * len(items)), timeout=qualified["timeout"],
                    models={a: qualified["auditors"][a]["model"] for a in audits.AUDITORS},
                    efforts={a: qualified["auditors"][a]["effort"] for a in audits.AUDITORS},
                    executables={a: qualified["auditors"][a]["executable"] for a in audits.AUDITORS})
                continue
            eligible = [p for p in programs if p["state"] == "eligible"]
            if not eligible:
                progress["status"] = "completed"
                break
            spent = [_read(Path(e["proposals"]) / "outcome.json") for e in progress["rounds"]
                     if (Path(e["proposals"]) / "outcome.json").exists()]
            replies = reply_allowance - sum(c["committed_replies"] for c in spent)
            tokens = output_token_allowance - sum(c["committed_tokens"] for c in spent)
            if replies < len(eligible) or tokens < len(eligible) * recipe["max_tokens"]:
                raise RuntimeError("Require full remaining round generation allowances")
            progress["rounds"].append(dict(proposals=str(path / "proposals"), audits=str(path / "audits")))
            persist()
            positions = {_identity(c): i for i, c in enumerate(order)}
            result = proposals.run([by_id[_identity(p["candidate"])] for p in eligible], path / "proposals",
                **inputs["settings"], recipe=recipe, reply_allowance=replies, output_token_allowance=tokens,
                fixtures=fixtures, earlier_turns=[p["turns"] for p in eligible],
                previous_inputs=[p["previous_inputs"] for p in eligible],
                seeds=[recipe["base_seed"] + p["attempts"] * len(order) + positions[_identity(p["candidate"])] for p in eligible])
            if result["status"] != "completed":
                raise RuntimeError("Proposal round stopped; see its saved outcome")
    except BaseException as error:
        progress.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        if not isinstance(error, Exception):
            raise
    finally:
        try:
            collect()
        except Exception as error:
            progress.setdefault("error", repr(error))
            progress.update(status="error" if progress["status"] != "cancelled" else "cancelled",
                            collection_error=repr(error))
        persist()
    return progress
