"""Single-host campaign accounting; INSERT-only history, never generation replay."""

import fcntl
from hashlib import sha256
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
            for event in events:
                if event["kind"] == "generation_reserved":
                    self._finish(event["seq"], events)
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
        reserved, done, native, closed = set(), set(), set(), set()
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
            elif event["kind"] == "native_intent":
                require(event["operation_id"] not in native, "Duplicate native operation")
                native.add(event["operation_id"])
            elif event["kind"] in ("native_verdict", "native_closed", "native_blocked"):
                require(event["operation_id"] in native, "Native event without intent")
                if event["kind"] == "native_closed":
                    require(event["operation_id"] not in closed, "Duplicate native closure")
                    closed.add(event["operation_id"])
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
        require({e["operation_id"] for e in events if e["kind"] == "native_intent"}
                == {e["operation_id"] for e in events if e["kind"] == "native_closed"},
                "Reconcile pending native operations before dispatch")

    def generate(self, experiment_id, source, request, *, attempt=None):
        try:
            attempt = snapshot(attempt)
            self.require_native_clear()
            request = snapshot(request)
            prepared, raw = generation.prepare_request(request)
            events = self.events
            record = next((e["record"] for e in events if e["kind"] == "experiment" and e["experiment_id"] == experiment_id), None)
            require(source in SOURCES and record is not None, "Require registered experiment/source")
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

    def outcome(self, seq):
        try:
            return self._finish(seq)
        except BaseException:
            self.close()
            raise

    def _finish(self, seq, events=None):
        events = self.events if events is None else events
        intent = next(e for e in events if e["seq"] == seq and e["kind"] == "generation_reserved")
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
