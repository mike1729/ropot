"""Publish approved witnesses from explicit saved runs, without model/execution calls."""
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import tempfile

from ..artifacts import sha256, write_json
from ..stage01.screen import integer_answer
from . import acquisition as c, audits, witnesses

RULE = "One current dual-approved witness per pinned benign, using the saved acquisition selection."


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _coverage(selections, accepted):
    result = {}
    for owner in ("development", "construction", "all"):
        pool = [s for s in selections if owner == "all" or s["binding"]["owner"] == owner]
        approved = [s for s in pool if c._identity(s["candidate"]) in accepted]
        by_source = {source: {(s["candidate"]["split"], s["candidate"]["id"]) for s in approved
                              if s["candidate"]["source"] == source} for source in ("actor", "synthetic")}
        result[owner] = dict(programs=len(pool), approved=len(approved), unresolved=len(pool)-len(approved),
            tasks=len({(s["candidate"]["split"], s["candidate"]["id"]) for s in pool}),
            tasks_with_witness=len(set().union(*by_source.values())),
            both_source_task_ids=[dict(split=split, id=i) for split, i in sorted(set.intersection(*by_source.values()))],
            sources={source: dict(programs=sum(s["candidate"]["source"] == source for s in pool),
                                 approved=len(by_source[source])) for source in by_source})
    return result


def publish(run_dirs, output_dir):
    """Copy exact programs/inputs and atomically publish manifest.json/report.json.

    Require full coverage of the pinned benign selection by disjoint, completed,
    non-fixture run histories. Reread audits/vetoes; trust saved native execution
    evidence, checking its bindings and four answers without replaying .eval logs.
    Refuse existing outputs, including symlinks. One publisher; inputs stay stable.
    Native evidence remains at its original paths with pinned digests.
    """
    target = Path(output_dir).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    pins = {}

    def pin(path):
        path = Path(path).absolute()
        if str(path) not in pins:
            pins[str(path)] = c._hash(path)
        return str(path)

    benign_manifest = witnesses.BENIGN_DIR / "manifest.json"
    selections = c._read(benign_manifest)["selections"]
    _require(c._hash(benign_manifest) == witnesses.SELECTION_SHA256, "Changed benign selection")
    pin(benign_manifest)
    selected = {c._identity(s["candidate"]): s for s in selections}
    wanted, approved, unresolved, run_records, findings = set(), {}, [], [], []
    costs, calls_seen, usage = {}, set(), defaultdict(Counter)
    audit_counts, validity_pairs, novelty_pairs = Counter(), Counter(), Counter()
    qualifications, version_changes = {}, []
    for run_dir in run_dirs:
        directory = Path(run_dir).absolute()
        runs = c._runs(directory)
        first, outcome = runs[0][1], runs[-1][2]
        _require(outcome["status"] == "completed" and all(not i["fixtures"] and
                 o["status"] in ("completed", "error", "cancelled") for _, i, o in runs),
                 "Require completed non-fixture histories")
        scope = {c._identity(x) for x in first["scope"]}
        _require(not wanted & scope and scope <= selected.keys(), "Overlapping or foreign publication scope")
        wanted.update(scope)
        rows, rounds, files = c._evidence(runs, first, scope)
        for path in files:
            pin(path)
        costs.update(rounds)
        qualification = Path(runs[-1][1]["qualification_dir"])
        settings, versions = c._qualification(qualification)
        pin(qualification / "inputs.json")
        for path in [audits.FIXTURES, *qualification.glob("*.txt"), *qualification.glob("calls/*/*/call-*/record.json")]:
            pin(path)
        qualifications[str(qualification)] = dict(source_sha256=settings["source_sha256"],
            auditors=settings["auditors"], timeout=settings["timeout"], versions=versions,
            call_count=len(c._calls(qualification)))
        batches = c._batches(runs)
        decisions, pending, calls = c._audit_state(batches, settings)
        _require(not pending, "Pending audits prevent publication")
        locations = {}
        for batch in batches:
            pin(batch / "inputs.json")
            if (batch / "vetoes.json").exists():
                pin(batch / "vetoes.json")
            for packet_id, entry in c._read(batch / "inputs.json")["packets"].items():
                binding = entry["binding"]
                locations[binding["round"], binding["attempt"]] = (batch, packet_id)
        for path, call in calls:
            if str(path) in calls_seen:
                continue
            calls_seen.add(str(path))
            if (path / "record.json").exists():
                pin(path / "record.json")
            audit_counts[call.get("status", "submission_unknown")] += 1
            if call.get("usage") is None:
                audit_counts["unknown_usage"] += 1
            else:
                usage[path.parent.name].update({k: v for k, v in call["usage"].items() if type(v) is int})
            if call.get("version") and call["version"] != versions[path.parent.name]:
                version_changes.append(dict(call=str(path), expected=versions[path.parent.name], actual=call["version"]))
        for decision in decisions.values():
            if all(decision["audits"].values()):
                replies = [decision["audits"][a]["reply"] for a in audits.AUDITORS]
                validity_pairs["/".join(r["validity"] for r in replies)] += 1
                novelty_pairs["/".join(r["novelty"] for r in replies)] += 1
        records = witnesses.load_worklist([(x[0], x[1], x[3]) for x in sorted(scope)])
        loaded = {c._identity(r["selection"]["candidate"]): r for r in records}
        by_proposal = {(r["round"], r["attempt"]): r for r in rows}
        for row in rows:
            if row["state"] == "benign_disagreement":
                findings.append(dict(proposal={k: row[k] for k in ("round", "attempt", "candidate", "stdin_sha256")},
                                     log=pin(row["verification"]["log"]), status="unadjudicated"))
        for program in c._programs(first["scope"], rows, decisions):
            identity = c._identity(program["candidate"])
            _require(program["attempts"] <= 3, "Proposal cap exceeded")
            selection = selected[identity]
            if program["state"] != "approved":
                unresolved.append(dict(candidate=program["candidate"], binding=selection["binding"],
                                       state=program["state"], attempts=program["attempts"]))
                continue
            binding = program["approved"]
            row = by_proposal[binding["round"], binding["attempt"]]
            decision = decisions[binding["round"], binding["attempt"]]
            record = loaded[identity]
            stdin = Path(row["input"]).read_bytes()
            verdict = row["verification"]
            expected_key = dict(candidate=row["candidate"], stdin_sha256=sha256(stdin))
            _require(row["state"] == verdict["status"] == "execution_agrees"
                     and c._binding(row) == binding == decision["binding"]
                     and verdict["key"] == expected_key and row["stdin_sha256"] == sha256(stdin),
                     "Mismatched execution/input/approval binding")
            _require(tuple(stdin.decode("utf-8").split()) not in
                     {tuple(x.split()) for x in record["stored_inputs"]}, "Stored-input duplicate")
            _require(set(verdict["programs"]) == set(witnesses.ROLES), "Require all four execution roles")
            _require(len({p["sample_id"] for p in verdict["programs"].values()}) == 4, "Duplicate execution sample IDs")
            for role, raw in zip(witnesses.ROLES, [*record["references"], record["program"]]):
                capture = verdict["programs"][role]
                _require(type(capture["sample_id"]) is int and capture["sample_id"] >= 0
                         and capture["program_sha256"] == sha256(raw) and capture["answer"] == verdict["answer"]
                         and capture["result"]["status"] == "completed" and capture["result"]["returncode"] == 0
                         and integer_answer(capture["result"]["stdout"]) == verdict["answer"], "Execution answer/program mismatch")
            batch, packet_id = locations[binding["round"], binding["attempt"]]
            item = dict(statement=record["task"]["question"], stdin=stdin.decode("utf-8"),
                        published_inputs=[x["input"] for x in record["task"]["cases"]])
            packet = batch / (packet_id + ".txt")
            _require(packet.read_bytes() == audits._prompt(item, packet_id).encode("utf-8"), "Audit packet differs from exact input/task")
            pin(packet); pin(verdict["log"])
            approvals = {}
            for auditor in audits.AUDITORS:
                call = decision["audits"][auditor]
                _require(call and call["source_sha256"] == settings["source_sha256"]
                         and call["model"] == settings["auditors"][auditor]["model"]
                         and call["effort"] == settings["auditors"][auditor]["effort"]
                         and all(call["reply"][k] == "approve" for k in ("validity", "novelty")), "Require qualified dual approval")
                approvals[auditor] = {k: call[k] for k in ("model", "effort", "version")}
            stem = f"{identity[3]}-{identity[0]}-{identity[1]}"
            flags = ["unadjudicated_benign_disagreement"] if any(f["proposal"]["candidate"] == row["candidate"] for f in findings) else []
            approved[identity] = (dict(binding=selection["binding"], label=selection["label"], source=identity[3],
                candidate=row["candidate"], program=f"programs/{stem}.py", program_sha256=sha256(record["program"]),
                stdin=f"inputs/{stem}.txt", stdin_sha256=sha256(stdin), answer=verdict["answer"], flags=flags,
                proposal={**binding, "seed": row["seed"]}, verification=verdict,
                approval=dict(batch=str(batch), packet_id=packet_id, auditors=approvals)), record["program"], stdin)
        for path, inputs, saved in runs:
            pin(path / "inputs.json"); pin(path / "outcome.json")
        run_records.append(dict(directory=str(directory), outcome_sha256=c._hash(directory / "outcome.json")))
    _require(wanted == selected.keys(), "Runs must cover every pinned benign exactly once")
    manifest = dict(schema_version=1, rule=RULE, cohort_sha256=witnesses.study_tasks.COHORT_SHA256,
        benign_selection=dict(path=str(benign_manifest.absolute()), sha256=witnesses.SELECTION_SHA256),
        runs=run_records, selections=[approved[c._identity(s["candidate"])][0] for s in selections if c._identity(s["candidate"]) in approved],
        unresolved=unresolved, evidence_sha256=pins,
        source_sha256={str(Path(p).absolute()): c._hash(p) for p in (__file__, c.__file__, audits.__file__, witnesses.__file__)})
    report = dict(coverage=_coverage(selections, approved), unresolved=unresolved,
        generation={k: sum(r[k] for r in costs.values()) for k in ("committed_replies", "committed_tokens", "observed_tokens")},
        attempts=sum(sum(a.get("submitted") is not False for a in r["attempts"]) for r in costs.values()),
        generation_usage_complete=all(r["usage_complete"] for r in costs.values()),
        audit_calls=len(calls_seen), audit_call_statuses=dict(audit_counts), known_audit_usage={a: dict(v) for a, v in usage.items()},
        validity_pairs=dict(validity_pairs), novelty_pairs=dict(novelty_pairs),
        validity_disagreements=sum(v for k, v in validity_pairs.items() if len(set(k.split("/"))) > 1),
        novelty_disagreements=sum(v for k, v in novelty_pairs.items() if len(set(k.split("/"))) > 1),
        qualifications=qualifications, auditor_version_changes=version_changes, benign_disagreements=findings)
    c._check(pins)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (temporary / "programs").mkdir(); (temporary / "inputs").mkdir()
        for selection, program, stdin in approved.values():
            (temporary / selection["program"]).write_bytes(program)
            (temporary / selection["stdin"]).write_bytes(stdin)
        write_json(temporary / "manifest.json", manifest)
        write_json(temporary / "report.json", report)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target / "manifest.json"
