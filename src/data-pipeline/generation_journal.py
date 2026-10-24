"""Single-host campaign accounting; INSERT-only history, never generation replay."""

import fcntl
from hashlib import sha256
import importlib
import json
import math
import os
from pathlib import Path
import sqlite3

from . import local_generation as generation
from . import precision_profiles as precision
from .study_tasks import SOURCES, snapshot

require = generation.require
UNITS = ("replies", "reserved_output_tokens", "reserved_generation_wall_seconds")


def encoded(value):
    return json.dumps(snapshot(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def limits_checked(limits):
    limits = snapshot(limits)
    require(type(limits) is dict and set(limits) == set(SOURCES), "Require both source limits")
    for values in limits.values():
        require(type(values) is dict and set(values) == set(UNITS), "Require all budget units")
        for unit, value in values.items():
            require(type(value) in ((int, float) if unit == UNITS[2] else (int,))
                    and math.isfinite(value) and value > 0, "Require positive finite limits")
    return limits


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProfileControlError(RuntimeError):
    def __init__(self, reservation_seq, reason):
        self.reservation_seq, self.reason = reservation_seq, reason
        super().__init__(f"Reservation {reservation_seq}: {reason}")


def profile_control_reason(outcome):
    if outcome.get("method") == "program-wave-v1":
        require({"profile_guarded", "profile_ready", "protocol_error"} <= outcome.keys()
                and type(outcome.get("profile_guarded")) is bool and type(outcome.get("profile_ready")) is bool
                and (outcome.get("protocol_error") is None or type(outcome["protocol_error"]) is str), "Invalid batch control outcome")
        if not outcome["profile_guarded"]: return "profile_unproven"
        return None if outcome["protocol_error"] in (None, "Missing done", "Incomplete protocol tail") else "profile_protocol"
    fields = {"profile_guarded", "profile_failure", "protocol_error"}
    present = fields & outcome.keys()
    if not present: return None
    require(present == fields, "Partial profile outcome")
    if outcome["profile_failure"] is not None: return "profile_failure"
    if outcome["profile_guarded"] is not True: return "profile_unproven"
    if outcome["protocol_error"] not in (None, "Missing done", "Incomplete protocol tail"): return "profile_protocol"
    return None


def require_profile_outcome(outcome):
    reason = profile_control_reason(outcome)
    if reason is not None: raise ProfileControlError(outcome["reservation_seq"], reason)


def _clearance_payload(events, outcome):
    require("session_seq" not in outcome, "Generation is not eligible for resource clearance")
    response = outcome["response"]
    evidence = outcome.get("evidence", {}).get("response.json", {})
    require(outcome.get("status") == "worker_error" and outcome.get("finish_reason") in ("stop", "length")
            and outcome.get("profile_guarded") is True and outcome.get("profile_failure") is None
            and outcome.get("protocol_error") is None and type(response) is dict and response["complete"] is True
            and type(response["returncode"]) is int and response["returncode"] == 0
            and response["parent_error"] is None and type(response["cleanup_error"]) is str and response["cleanup_error"]
            and type(response["pid"]) is int and response["pid"] > 0 and precision._sha(evidence.get("sha256")),
            "Generation is not eligible for resource clearance")
    done = [e for e in events if e["kind"] == "generation_done" and e["reservation_seq"] == outcome["reservation_seq"]]
    require(len(done) == 1 and encoded({k: outcome[k] for k in done[0] if k not in ("kind", "seq")})
            == encoded({k: v for k, v in done[0].items() if k not in ("kind", "seq")}),
            "Generation is not eligible for resource clearance")
    return dict(reservation_seq=outcome["reservation_seq"], response_sha256=evidence["sha256"], pid=response["pid"])


def generation_resources_cleared(events, outcome):
    if "session_seq" in outcome:
        done = next((e for e in events if e["kind"] == "batch_done" and e["session_seq"] == outcome["session_seq"]), None)
        row = next((e for e in events if e["kind"] == "generation_done" and e["reservation_seq"] == outcome["reservation_seq"]), None)
        require(done is not None and row is not None and encoded({k: v for k, v in row.items() if k not in ("kind", "seq")}) == encoded(outcome)
                and row["batch_done_seq"] == done["seq"] and row["job_sha256"] == done["job_sha256"], "Batch outcome binding differs")
        return done["shared"]["resources_closed"] is True
    markers = [e for e in events if e["kind"] == "generation_resources_cleared"
               and e["reservation_seq"] == outcome["reservation_seq"]]
    if not markers:
        return outcome["response"] is None or outcome["response"]["cleanup_error"] is None
    payload = _clearance_payload(events, outcome)
    require(encoded({k: markers[0][k] for k in payload}) == encoded(payload), "Generation resource clearance differs")
    return True


def _batch_method(method, campaign=None):
    require(type(method) is dict and set(method) == {"method", "source", "admission", "prompt_ceiling_tokens", "qualification"}
            and method["method"] == "program-wave-v1" and method["source"] in SOURCES, "Invalid batch method")
    authority, reference = method["admission"], method["qualification"]
    require(type(reference) is dict and set(reference) == {"path", "response_sha256"}
            and precision._sha(reference["response_sha256"])
            and reference["path"] == "qualifications/" + reference["response_sha256"], "Invalid batch qualification reference")
    require(type(authority) is dict and authority["inputs"]["response.json"]["sha256"] == reference["response_sha256"]
            and type(method["prompt_ceiling_tokens"]) is int
            and 0 < method["prompt_ceiling_tokens"] <= authority["prompt_ceiling_tokens"], "Batch authority differs")
    controls = authority["request_controls"]
    require(generation._profile(controls)["source"] == method["source"]
            and controls["max_tokens"] == 2048 and controls["wall_time_seconds"] == 240, "Batch controls differ")
    if campaign is not None:
        path = campaign.root / reference["path"]
        require(path.resolve() == path, "Batch qualification path differs")
        host = importlib.import_module("experiments.batch_throughput")
        retained, files = host.read_qualification(path, reference["response_sha256"])
        require(encoded(retained) == encoded(authority), "Batch qualification authority differs")
        return files


def _batch_projection(campaign, intent):
    host = importlib.import_module("experiments.batch_throughput")
    worker = importlib.import_module("experiments.mlx_batch_throughput")
    method, job = intent["method_spec"], json.loads(intent["job_text"])
    qualified = _batch_method(method, campaign)
    directory = campaign.root / "sessions" / f"{intent['seq']:06d}"
    limits = {n: host.STDOUT_LIMIT for n in ("stdout.jsonl", "response.json", "decoded.jsonl", "telemetry.jsonl")}
    limits.update({n: 4 * 1024**2 for n in ("job.json", "launch.json", *(f"source-{i}.py" for i in range(6)),
                  *(f"requests/{i}.json" for i in range(len(job["rows"]))))})
    limits.update({"stderr.txt": host.STDERR_LIMIT, **{"qualification/" + n: host.QUALIFICATION_LIMITS[n] for n in qualified}})
    if "cache_limit_bytes" in job: limits["cache-control.json"] = 4096
    raws, errors = {}, []
    if directory.exists():
        require(directory.resolve() == directory, "Invalid batch session directory")
        entries = list(directory.rglob("*"))
        require(all(not p.is_symlink() for p in entries), "Invalid batch session entry")
        unknown = {str(p.relative_to(directory)) for p in entries if not p.is_dir()} - set(limits)
        if unknown: errors.append("Unexpected session files")
        limits.update({n: 4 * 1024**2 for n in sorted(unknown)})
        extra_dirs = sorted(str(p.relative_to(directory)) for p in entries if p.is_dir() and str(p.relative_to(directory)) not in ("requests", "qualification"))
        if extra_dirs: errors.append("Unexpected session directories: " + encoded(extra_dirs))
    for name, limit in limits.items():
        if not os.path.lexists(directory / name): errors.append("Missing session file: " + name); continue
        raws[name] = host._qualification_file(directory, name, limit)
    copied = all(raws.get("qualification/" + n) == raw for n, raw in qualified.items())
    if not copied: errors.append("Session qualification copy differs")
    if raws.get("job.json") != intent["job_text"].encode(): errors.append("Session job differs")
    if any(raws.get(f"requests/{i}.json") != r["request_text"].encode() for i, r in enumerate(job["rows"])): errors.append("Session requests differ")
    try:
        launch = snapshot(json.loads(raws.get("launch.json", b"")))
        require(set(launch) == {"argv", "environment", "lease_fd", "lease_dev", "lease_ino", "sources"}
                and launch["argv"] == [job["request"]["precision_profile"]["descriptor"]["expected_runtime"]["executable"],
                    "-I", "-B", job["worker"], str(directory / "job.json")]
                and encoded(launch["environment"]) == encoded(job["request"]["environment"])
                and all(type(launch[k]) is int and launch[k] >= 0 for k in ("lease_fd", "lease_dev", "lease_ino"))
                and type(launch["sources"]) is list and len(launch["sources"]) == 6 and set(launch["sources"]) == set(job["source_hashes"])
                and all(sha256(raws.get(f"source-{i}.py", b"")).hexdigest() == job["source_hashes"][p]
                        for i, p in enumerate(launch["sources"])), "Session launch/source copies differ")
        response = snapshot(json.loads(raws.get("response.json", b"")))
        require(raws.get("telemetry.jsonl") == b"".join(host._raw(s) for s in response["telemetry"]), "Session telemetry differs")
        if "cache_limit_bytes" in job:
            cache = snapshot(json.loads(raws.get("cache-control.json", b"")))
            require(set(cache) == {"job_sha256", "requested_bytes", "observed_bytes", "previous_bytes"}
                    and type(cache["previous_bytes"]) is int and cache["previous_bytes"] >= 0
                    and cache["job_sha256"] == intent["job_sha256"] and all(type(cache[k]) is int
                    and cache[k] == job["cache_limit_bytes"] for k in ("requested_bytes", "observed_bytes")), "Session cache proof differs")
    except (ValueError, KeyError, TypeError): errors.append("Session launch/source/telemetry/cache proof differs")
    projected = worker.project_wave(job, raws.get("stdout.jsonl"), raws.get("response.json"), raws.get("decoded.jsonl"), qualification_files=qualified)
    captured = {str((directory / n).relative_to(campaign.root)): raw for n, raw in raws.items()}
    captured.update({method["qualification"]["path"] + "/" + n: raw for n, raw in qualified.items()})
    for name in captured:
        with (campaign.root / name).open("rb") as file: os.fsync(file.fileno())
    for path in (directory / "requests", directory / "qualification", directory, directory.parent, campaign.root):
        if path.is_dir(): sync_directory(path)
    return projected, copied, errors, {n: dict(sha256=sha256(raw).hexdigest(), bytes=len(raw)) for n, raw in captured.items()}


def _batch_completions(intent, projected, copied, errors, evidence, first):
    eligible = projected["shared"]["complete"] and copied and not errors
    shared = dict(seq=first, kind="batch_done", session_seq=intent["seq"], job_sha256=intent["job_sha256"],
                  shared=projected["shared"], session_copy_complete=copied, evidence_errors=errors, evidence=evidence, eligible=eligible)
    entries = [shared]
    for i, (binding, row) in enumerate(zip(intent["rows"], projected["rows"])):
        require(all(row[k] == binding[k] for k in ("row_id", "request_sha256")), "Batch projected row binding differs")
        entries.append(dict(seq=first + i + 1, kind="generation_done", reservation_seq=binding["reservation_seq"],
            session_seq=intent["seq"], row_index=i, batch_done_seq=first, method="program-wave-v1", job_sha256=intent["job_sha256"],
            **{k: row[k] for k in ("row_id", "request_sha256", "uid", "tokens", "text", "finish_reason", "admit_seconds", "first_seconds", "finish_seconds")},
            status=row["finish_reason"] if eligible else "interrupted", eligible=eligible, observed_generation_tokens=len(row["tokens"]),
            profile_guarded=projected["shared"]["profile_guarded"], profile_ready=projected["shared"]["ready"], protocol_error=projected["shared"]["protocol_error"]))
    return entries


def _batch_rows(rows, method, events, campaign=None):
    loop = importlib.import_module("src.data-pipeline.experiment_loop")
    records = {e["experiment_id"]: e["record"] for e in events if e["kind"] == "experiment"}
    selected = set()
    for row in rows:
        source, item, request = row["source"], row["attempt"], row["request"]
        record = records.get(row["experiment_id"])
        require(record is not None and type(record.get("schema_version")) is int and record["schema_version"] == 3
                and set(record.get("generation_method", {})) == set(record["recipes"])
                and source == method["source"] and source in record["recipes"]
                and encoded(record["generation_method"][source]) == encoded(method), "Registered batch method differs")
        profiles = precision.check_profiles(record["profiles"], record["recipes"], campaign=campaign)
        require(encoded(request["precision_profile"]["descriptor"]) == encoded(profiles[source]["descriptor"]), "Batch profile differs")
        require(type(item) is dict and set(item) == {"split", "id", "task_sha256", "reply_index"}
                and type(item["id"]) is int and type(item["reply_index"]) is int, "Invalid batch attempt")
        task = next((t for t in record["activation"]["tasks"] if t["id"] == item["id"]
                     and t["id"] in record["assigned_task_ids"]), None)
        require(task is not None and (item["split"], item["task_sha256"]) == (task["split"], task["sha256"]), "Batch task binding differs")
        cell = (row["experiment_id"], source, item["split"], item["id"])
        require(cell not in selected, "Duplicate batch cell")
        selected.add(cell)
        cells, results = loop._history(events, record)
        slots, recipe = cells.get((source, item["id"]), []), record["recipes"][source]
        require(type(recipe["max_replies"]) is int and type(recipe["seeds"]) is list
                and len(recipe["seeds"]) == recipe["max_replies"] and item["reply_index"] == len(slots) < recipe["max_replies"]
                and (not slots or results.get(slots[-1]["seq"], {}).get("status") in ("format_failed", "verification_failed")), "Batch reply history differs")
        require(all(request[k] == recipe[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds"))
                and type(recipe["seeds"][item["reply_index"]]) is int
                and request["seed"] == recipe["seeds"][item["reply_index"]], "Batch recipe differs")


def _batch_groups(events):
    batches, members = {}, set()
    host = None
    for index, event in enumerate(events):
        if event["kind"] != "batch_reserved":
            continue
        require(set(event) == {"seq", "kind", "method_spec", "job_text", "job_sha256", "rows"}
                and type(event["job_text"]) is str and type(event["rows"]) is list and event["rows"], "Invalid batch reservation group")
        _batch_method(event["method_spec"])
        job = snapshot(json.loads(event["job_text"]))
        host = host or importlib.import_module("experiments.batch_throughput")
        require(host._raw(job).decode() == event["job_text"]
                and sha256(event["job_text"].encode()).hexdigest() == event["job_sha256"], "Batch job digest differs")
        host._shape(job, historical=True)
        require(job["schema_version"] == 3 and encoded(job["admission"]) == encoded(event["method_spec"]["admission"])
                and job["prompt_ceiling_tokens"] == event["method_spec"]["prompt_ceiling_tokens"]
                and len(event["rows"]) == len(job["rows"]), "Batch job authority differs")
        rows = []
        for offset, (binding, retained) in enumerate(zip(event["rows"], job["rows"])):
            seq = event["seq"] + offset + 1
            require(type(binding) is dict and set(binding) == {"reservation_seq", "row_id", "request_sha256"}
                    and type(binding["reservation_seq"]) is int and binding == dict(reservation_seq=seq,
                        row_id=f"batch-{event['seq']}-{offset}", request_sha256=retained["request_sha256"])
                    and binding["row_id"] == retained["row_id"] and index + offset + 1 < len(events), "Batch row linkage differs")
            row = events[index + offset + 1]
            require(set(row) == {"seq", "kind", "experiment_id", "source", "request_text", "request_sha256", "attempt", "session_seq", "row_index"}
                    and row["kind"] == "generation_reserved" and row["seq"] == seq
                    and type(row["session_seq"]) is int and row["session_seq"] == event["seq"]
                    and type(row["row_index"]) is int and row["row_index"] == offset
                    and row["request_text"] == retained["request_text"] and row["request_sha256"] == retained["request_sha256"], "Batch row reservation differs")
            rows.append({k: row[k] for k in ("experiment_id", "source", "attempt")})
            rows[-1]["request"] = json.loads(row["request_text"])
            members.add(seq)
        _batch_rows(rows, event["method_spec"], events[:index])
        limits, totals = Campaign.totals(events[:index + len(rows) + 1])
        require(all(totals[s][u] <= limits[s][u] for s in SOURCES for u in UNITS), "Batch reservation exceeds budget")
        batches[event["seq"]] = event
    require(all((e["seq"] in members) == ("session_seq" in e) for e in events if e["kind"] == "generation_reserved"), "Invalid batch reservation membership")
    completed, settled = {}, set()
    for index, event in enumerate(events):
        if event["kind"] != "batch_done": continue
        session = event.get("session_seq")
        require(set(event) == {"seq", "kind", "session_seq", "job_sha256", "shared", "session_copy_complete", "evidence_errors", "evidence", "eligible"}
                and type(session) is int and session in batches and session not in completed
                and event["job_sha256"] == batches[session]["job_sha256"] and type(event["session_copy_complete"]) is bool
                and type(event["evidence_errors"]) is list and all(type(s) is str for s in event["evidence_errors"])
                and type(event["shared"]) is dict and type(event["shared"]["resources_closed"]) is bool
                and event["shared"]["job_sha256"] == event["job_sha256"]
                and type(event["shared"]["complete"]) is bool and type(event["eligible"]) is bool
                and type(event["evidence"]) is dict and all(type(n) is str and not Path(n).is_absolute()
                    and ".." not in Path(n).parts and type(v) is dict and set(v) == {"sha256", "bytes"}
                    and precision._sha(v["sha256"]) and type(v["bytes"]) is int and v["bytes"] >= 0
                    for n, v in event["evidence"].items()), "Invalid batch completion")
        rows = events[index + 1:index + len(batches[session]["rows"]) + 1]
        expected = _batch_completions(batches[session], dict(shared=event["shared"], rows=rows),
                                     event["session_copy_complete"], event["evidence_errors"], event["evidence"], event["seq"])
        require(len(rows) == len(batches[session]["rows"]) and encoded([event] + rows) == encoded(expected)
                and event["seq"] > session + len(rows), "Batch completion group differs")
        for row in rows:
            require(type(row["tokens"]) is list and len(row["tokens"]) <= 2048 and all(type(t) is int and t >= 0 for t in row["tokens"])
                    and (row["uid"] is None or type(row["uid"]) is int) and (row["text"] is None or type(row["text"]) is str)
                    and row["finish_reason"] in (None, "stop", "length") and all(row[k] is None or type(row[k]) in (int, float)
                        and math.isfinite(row[k]) and row[k] >= 0 for k in ("admit_seconds", "first_seconds", "finish_seconds")), "Invalid batch row outcome")
            require(not row["eligible"] or row["finish_reason"] in ("stop", "length") and row["uid"] is not None
                    and type(row["text"]) is str, "Invalid eligible batch row")
            profile_control_reason(row)
            settled.add(row["reservation_seq"])
        completed[session] = event
    require(all((e["reservation_seq"] in settled) == ("session_seq" in e) for e in events if e["kind"] == "generation_done"), "Batch completion membership differs")
    for session, intent in batches.items():
        done = completed.get(session)
        boundary = done["seq"] + len(intent["rows"]) if done and done["shared"]["resources_closed"] else math.inf
        require(not any(session < e["seq"] <= boundary and e["kind"] in ("native_intent", "batch_reserved", "generation_reserved")
                        and e["seq"] not in {r["reservation_seq"] for r in intent["rows"]} for e in events), "Work follows unresolved batch")
    return batches


class Campaign:
    @classmethod
    def create(cls, root, limits):
        limits = limits_checked(limits)
        root = Path(root).resolve()
        root.mkdir(exist_ok=False)
        sync_directory(root.parent)
        campaign = cls.__new__(cls)
        campaign._open(root, create=True)
        try:
            campaign.db.execute("CREATE TABLE events(seq INTEGER PRIMARY KEY, body TEXT NOT NULL) STRICT")
            campaign._append("initial", limits=limits)
            sync_directory(root)
            return campaign
        except BaseException:
            campaign.close()
            raise

    def __init__(self, root):
        self._open(Path(root).resolve(), create=False)
        try:
            events = self.events
            sessions = set()
            for event in events:
                if event["kind"] == "generation_reserved":
                    session = event.get("session_seq")
                    if session is None: self._finish(event["seq"], events)
                    elif session not in sessions:
                        self._finish_batch(session, events)
                        sessions.add(session)
        except BaseException:
            self.close()
            raise

    def _open(self, root, *, create):
        self.root, self.db, self._lock = root, None, None
        try:
            self._lock = (root / "lock").open("x+b" if create else "r+b")
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            uri = (root / "journal.sqlite3").as_uri() + ("?mode=rwc" if create else "?mode=rw")
            self.db = sqlite3.connect(uri, uri=True, isolation_level=None)
            for statement, expected in (("fullfsync=ON", 1), ("synchronous=EXTRA", 3)):
                self.db.execute("PRAGMA " + statement)
                require(self.db.execute("PRAGMA " + statement.split("=")[0]).fetchone()[0] == expected,
                        "SQLite durability setting unavailable")
            mode = self.db.execute("PRAGMA journal_mode" + ("=DELETE" if create else "")).fetchone()[0]
            require(mode == "delete", "Require SQLite DELETE journal mode")
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self._lock is not None:
            self._lock.close()  # Never LOCK_UN: a worker may still hold this open-file description.
            self._lock = None

    @property
    def events(self):
        rows = self.db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()
        events = [{**snapshot(json.loads(body)), "seq": seq} for seq, body in rows]
        require([seq for seq, _ in rows] == list(range(1, len(rows) + 1)) and events
                and events[0]["kind"] == "initial" and sum(e["kind"] == "initial" for e in events) == 1,
                "Invalid campaign history")
        limits_checked(events[0]["limits"])
        reserved, done, native, closed, cleared = set(), set(), set(), set(), set()
        for event in events:
            if event["kind"] == "experiment":
                require(event["record"]["experiment_id"] == event["experiment_id"]
                        and sha256(encoded(event["record"]).encode()).hexdigest() == event["record_sha256"],
                        "Experiment record digest differs")
            elif event["kind"] == "generation_reserved":
                reserved.add(event["seq"])
            elif event["kind"] == "generation_done":
                seq = event["reservation_seq"]
                require(seq in reserved and seq not in done, "Invalid or duplicate completion")
                done.add(seq)
            elif event["kind"] == "generation_resources_cleared":
                seq = event.get("reservation_seq")
                require(set(event) == {"seq", "kind", "reservation_seq", "response_sha256", "pid"}
                        and type(seq) is int and seq in done and seq not in cleared
                        and precision._sha(event["response_sha256"]) and type(event["pid"]) is int and event["pid"] > 0,
                        "Invalid generation resource clearance")
                cleared.add(seq)
            elif event["kind"] == "native_intent":
                require(event["operation_id"] not in native, "Duplicate native operation")
                native.add(event["operation_id"])
            elif event["kind"] in ("native_verdict", "native_closed", "native_blocked"):
                require(event["operation_id"] in native, "Native event without intent")
                if event["kind"] == "native_closed":
                    require(event["operation_id"] not in closed, "Duplicate native closure")
                    closed.add(event["operation_id"])
        try:
            _batch_groups(events)
        except (KeyError, TypeError) as error:
            raise ValueError("Invalid batch history") from error
        return events

    def _append(self, kind, **payload):
        body = encoded({"kind": kind, **payload})
        try:
            seq = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events").fetchone()[0]
            self.db.execute("INSERT INTO events(seq,body) VALUES (?,?)", (seq, body))
            return {"seq": seq, "kind": kind, **snapshot(payload)}
        except BaseException:
            self.close()
            raise

    def _append_group(self, entries):
        try:
            self.db.execute("BEGIN IMMEDIATE")
            first = self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events").fetchone()[0]
            require([e["seq"] for e in entries] == list(range(first, first + len(entries))), "Batch transaction sequence differs")
            for event in entries:
                self.db.execute("INSERT INTO events(seq,body) VALUES (?,?)",
                                (event["seq"], encoded({k: v for k, v in event.items() if k != "seq"})))
            self.db.execute("COMMIT")
            return snapshot(entries)
        except BaseException:
            self.close()  # A failed/uncertain COMMIT never proceeds to any dispatch.
            raise

    def register(self, record):
        record = snapshot(record)
        identifier = record["experiment_id"]
        require(type(identifier) is str and bool(identifier.strip()), "Require experiment ID")
        digest = sha256(encoded(record).encode()).hexdigest()
        previous = [e for e in self.events if e["kind"] == "experiment" and e["experiment_id"] == identifier]
        if previous:
            require(previous[0]["record_sha256"] == digest, "Experiment record changed")
        else:
            self._append("experiment", experiment_id=identifier, record=record, record_sha256=digest)

    @staticmethod
    def totals(events, additional=None):
        limits = [e["limits"] for e in events if e["kind"] in ("initial", "limits")][-1]
        limits_checked(limits)
        totals = {}
        for source in SOURCES:
            requests = [json.loads(e["request_text"]) for e in events
                        if e["kind"] == "generation_reserved" and e["source"] == source]
            if additional is not None and additional[0] == source:
                requests.append(additional[1])
            totals[source] = dict(zip(UNITS, (len(requests), sum(r["max_tokens"] for r in requests),
                                               math.fsum(r["wall_time_seconds"] for r in requests))))
        return limits, totals

    def _accounting(self, additional=None):
        return self.totals(self.events, additional)

    def extend(self, limits):
        limits = limits_checked(limits)
        previous, _ = self._accounting()
        require(all(limits[s][u] >= previous[s][u] for s in SOURCES for u in UNITS), "Limits cannot decrease")
        if limits != previous:
            self._append("limits", limits=limits)

    def require_native_clear(self):
        events = self.events
        for event in events:
            if event["kind"] == "batch_reserved":
                require(self._finish_batch(event["seq"])["shared"]["resources_closed"], "Unresolved batch generation resources")
        require({e["operation_id"] for e in events if e["kind"] == "native_intent"}
                == {e["operation_id"] for e in events if e["kind"] == "native_closed"},
                "Reconcile pending native operations before dispatch")

    def reserve_batch(self, rows, method_spec):
        try:
            self.require_native_clear()
            rows, method = snapshot(rows), snapshot(method_spec)
            require(type(rows) is list and rows and all(type(r) is dict
                    and set(r) == {"experiment_id", "source", "request", "attempt"}
                    and type(r["experiment_id"]) is str and type(r["source"]) is str for r in rows), "Invalid batch rows")
            _batch_method(method, self)
            events = self.events
            reserved = [e for e in events if e["kind"] == "generation_reserved"]
            require({e["seq"] for e in reserved} == {e["reservation_seq"] for e in events if e["kind"] == "generation_done"},
                    "Reconcile pending generations before dispatch")
            require(all(generation_resources_cleared(events, self.outcome(e["seq"])) for e in reserved),
                    "Unresolved generation cleanup failure")
            session = events[-1]["seq"] + 1
            worker = importlib.import_module("experiments.mlx_batch_throughput")
            host = importlib.import_module("experiments.batch_throughput")
            job = worker.build_wave([dict(row_id=f"batch-{session}-{i}", request=r["request"]) for i, r in enumerate(rows)],
                                    method["admission"], prompt_ceiling_tokens=method["prompt_ceiling_tokens"])
            for row, retained in zip(rows, job["rows"]):
                row["request"] = json.loads(retained["request_text"])
            _batch_rows(rows, method, events, self)
            bindings, entries = [], []
            for i, (row, retained) in enumerate(zip(rows, job["rows"])):
                seq = session + i + 1
                bindings.append(dict(reservation_seq=seq, row_id=retained["row_id"], request_sha256=retained["request_sha256"]))
                entries.append(dict(seq=seq, kind="generation_reserved", experiment_id=row["experiment_id"], source=row["source"],
                    request_text=retained["request_text"], request_sha256=retained["request_sha256"], attempt=row["attempt"], session_seq=session, row_index=i))
            limits, totals = self.totals(events + entries)
            require(all(totals[s][u] <= limits[s][u] for s in SOURCES for u in UNITS), "Campaign budget exhausted")
            raw = host._raw(job)
            intent = dict(seq=session, kind="batch_reserved", method_spec=method, job_text=raw.decode(),
                          job_sha256=sha256(raw).hexdigest(), rows=bindings)
            _batch_groups(events + [intent] + entries)
            return self._append_group([intent] + entries)[0]
        except BaseException:
            self.close()
            raise

    def generate(self, experiment_id, source, request, *, attempt=None):
        try:
            attempt = snapshot(attempt)
            self.require_native_clear()
            request = snapshot(request)
            events = self.events
            record = next((e["record"] for e in events if e["kind"] == "experiment" and e["experiment_id"] == experiment_id), None)
            require(source in SOURCES and record is not None, "Require registered experiment/source")
            require("generation_method" not in record and record.get("schema_version") != 3, "Batch record requires batch admission")
            prepared, raw = generation.prepare_request(request)
            require(("precision_profile" in prepared) == ("profiles" in record), "Registered profile direction differs")
            if "profiles" in record:
                profiles = precision.check_profiles(record["profiles"], record["recipes"], campaign=self)
                require(source in profiles, "Registered profile source differs")
                expected = generation.request_for(source, prepared["messages"], profile=profiles[source]["descriptor"],
                    **{k: prepared[k] for k in ("seed", "temperature", "top_p", "max_tokens", "wall_time_seconds")})
                require(generation.prepare_request(expected)[1] == raw, "Registered profile request differs")
            else:
                require(all(prepared[k] == v for k, v in generation.PROFILES[source].items()), "Source profile differs")
            reserved = {e["seq"] for e in events if e["kind"] == "generation_reserved"}
            require(reserved == {e["reservation_seq"] for e in events if e["kind"] == "generation_done"},
                    "Reconcile pending generations before dispatch")
            limits, totals = self._accounting((source, prepared))
            require(all(totals[source][u] <= limits[source][u] for u in UNITS),
                    "Campaign budget exhausted")
            intent = self._append("generation_reserved", experiment_id=experiment_id, source=source,
                                  request_text=raw.decode(), request_sha256=sha256(raw).hexdigest(), attempt=attempt)
            generation.generate(request, self.root / "requests" / f"{intent['seq']:06d}",
                                lease_fd=self._lock.fileno())
            outcome = self._finish(intent["seq"])
            require_profile_outcome(outcome)
            return outcome
        except BaseException:
            self.close()
            raise

    def generate_batch(self, rows, method_spec):
        try:
            intent = self.reserve_batch(rows, method_spec)
            directory = self.root / "sessions"
            directory.mkdir(exist_ok=True)
            require(directory.resolve() == directory, "Invalid batch sessions directory")
            sync_directory(directory)
            sync_directory(self.root)
            host = importlib.import_module("experiments.batch_throughput")
            host.run(json.loads(intent["job_text"]), directory / f"{intent['seq']:06d}", self.root / "lock",
                     lease_fd=self._lock.fileno(), qualification_files=_batch_method(intent["method_spec"], self))
            self._finish_batch(intent["seq"])
            outcomes = [{k: v for k, v in e.items() if k not in ("seq", "kind")} for e in self.events
                        if e["kind"] == "generation_done" and e.get("session_seq") == intent["seq"]]
            for outcome in outcomes: require_profile_outcome(outcome)
            return outcomes
        except BaseException:
            self.close()
            raise

    def _finish_batch(self, session, events=None):
        events = self.events if events is None else events
        intent = next(e for e in events if e["kind"] == "batch_reserved" and e["seq"] == session)
        prior = next((e for e in events if e["kind"] == "batch_done" and e["session_seq"] == session), None)
        projected, copied, errors, evidence = _batch_projection(self, intent)
        first = prior["seq"] if prior else events[-1]["seq"] + 1
        entries = _batch_completions(intent, projected, copied, errors, evidence, first)
        if prior:
            require(encoded(events[first - 1:first + len(intent["rows"])]) == encoded(entries), "Completed batch evidence changed")
        else:
            _batch_groups(events + entries)
            self._append_group(entries)
        return entries[0]

    def _batch_outcome(self, intent):
        self._finish_batch(intent["session_seq"])
        row = next(e for e in self.events if e["kind"] == "generation_done" and e["reservation_seq"] == intent["seq"])
        return {k: v for k, v in row.items() if k not in ("kind", "seq")}

    def outcome(self, seq):
        try:
            return self._finish(seq)
        except BaseException:
            self.close()
            raise

    def clear_generation_resources(self, seq):
        try:
            require(type(seq) is int and any(e["kind"] == "generation_reserved" and e["seq"] == seq for e in self.events),
                    "Generation is not eligible for resource clearance")
            outcome = self.outcome(seq)
            events = self.events
            payload = _clearance_payload(events, outcome)
            prior = [e for e in events if e["kind"] == "generation_resources_cleared" and e["reservation_seq"] == seq]
            if prior:
                generation_resources_cleared(events, outcome)
                return prior[0]
            for probe, name in ((os.kill, "process"), (os.killpg, "group")):
                try:
                    probe(payload["pid"], 0)
                except ProcessLookupError:
                    continue
                raise ValueError(f"Generation {name} is still present")
            return self._append("generation_resources_cleared", **payload)
        except BaseException:
            self.close()
            raise

    def _finish(self, seq, events=None):
        events = self.events if events is None else events
        intent = next(e for e in events if e["seq"] == seq and e["kind"] == "generation_reserved")
        if "session_seq" in intent:
            return self._batch_outcome(intent)
        raw = intent["request_text"].encode()
        require(sha256(raw).hexdigest() == intent["request_sha256"], "Reserved request digest differs")
        request = snapshot(json.loads(raw))  # Reconciliation never needs the model directory to still exist.
        directory = self.root / "requests" / f"{seq:06d}"
        files = {p.name: p.read_bytes() for p in (directory / n for n in
                 ("request.json", "stdout.jsonl", "stderr.txt", "response.json")) if p.exists()}
        saved, stdout = files.get("request.json", b""), files.get("stdout.jsonl", b"")
        require(raw.startswith(saved), "Retained request differs")
        try:
            response = json.loads(files.get("response.json", b""))
        except (json.JSONDecodeError, UnicodeDecodeError):
            response = None
        else:
            require(type(response) is dict, "Invalid parent outcome object")
            response = snapshot(response)
        pid = response.get("pid") if response is not None else None
        parsed = generation.checked_protocol(stdout, request, intent["request_sha256"], pid, response.get("returncode") if response else None)
        if response is not None:
            require(set(response) == set(parsed) | {"status", "request_sha256", "pid", "returncode", "elapsed_seconds", "parent_error", "cleanup_error"}, "Invalid outcome fields")
            require(response["request_sha256"] == intent["request_sha256"]
                    and encoded({key: response[key] for key in parsed}) == encoded(parsed), "Outcome differs from protocol")
            require(response["status"] in ("stop", "length", "timeout", "worker_error", "parent_error", "cancelled")
                    and type(response["elapsed_seconds"]) in (int, float)
                    and math.isfinite(response["elapsed_seconds"]) and response["elapsed_seconds"] >= 0,
                    "Invalid parent outcome")
            require((pid is None or (type(pid) is int and pid > 0))
                    and (response["returncode"] is None or type(response["returncode"]) is int)
                    and all(response[k] is None or type(response[k]) is str for k in ("parent_error", "cleanup_error")),
                    "Invalid parent metadata")
            require((response["parent_error"] is not None) == (response["status"] in ("parent_error", "cancelled")),
                    "Parent error status differs")
            if response["status"] == "worker_error":
                require(response["cleanup_error"] is not None or response["returncode"] != 0 or not response["complete"],
                        "Worker error without a failure")
            if response["status"] in ("stop", "length"):
                require(type(pid) is int and response["complete"] and response["returncode"] == 0
                        and response["parent_error"] is None and response["cleanup_error"] is None
                        and response["status"] == parsed["finish_reason"], "Invalid successful outcome")
        summary = {"status": response["status"] if response and saved == raw else "interrupted",
                   "finish_reason": parsed["finish_reason"], "observed_generation_tokens": parsed["observed_generation_tokens"],
                   "elapsed_seconds": response["elapsed_seconds"] if response else None}
        if "precision_profile" in request:
            summary.update({k: parsed[k] for k in ("profile_guarded", "profile_failure", "protocol_error")})
        evidence = {}
        for name, content in files.items():
            with (directory / name).open("rb") as file:
                os.fsync(file.fileno())
            evidence[name] = {"sha256": sha256(content).hexdigest(), "bytes": len(content)}
        if directory.exists():
            for path in (directory, directory.parent, self.root):
                sync_directory(path)
        # SQLite's following FULLFSYNC commit also flushes preceding evidence writes on macOS.
        payload = {"reservation_seq": seq, **summary, "evidence": evidence}
        prior = [e for e in events if e["kind"] == "generation_done" and e["reservation_seq"] == seq]
        if prior:
            require(encoded({k: prior[0][k] for k in payload}) == encoded(payload), "Completed evidence changed")
        else:
            self._append("generation_done", **payload)
        return {**payload, "response": response}
