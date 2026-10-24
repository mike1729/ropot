from hashlib import sha256
import importlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import unittest
from unittest import mock
from tests import test_local_generation as local_tests
from tests.test_local_runner import process_gone, wait_for

G = importlib.import_module("src.data-pipeline.local_generation")


class GenerationJournalTests(unittest.TestCase):
    def setUp(self):
        local_tests.LocalGenerationTests.setUp(self)
        self.J = importlib.import_module("src.data-pipeline.generation_journal")
        self.worker = G.WORKER
        patch = mock.patch.dict(G.PROFILES["actor"], model_path=str(self.root))
        patch.start(); self.addCleanup(patch.stop)

    def limits(self, **changes):
        return {source: {"replies": 100, "reserved_output_tokens": 1000,
                         "reserved_generation_wall_seconds": 1000, **changes} for source in ("actor", "synthetic")}

    def request(self, mode="stop", **changes):
        return local_tests.LocalGenerationTests.request(self, mode, **{"wall_time_seconds": 0.5, **changes})

    def create(self, limits=None):
        self.index += 1
        root = self.root / str(self.index)
        campaign = self.J.Campaign.create(root, limits or self.limits())
        campaign.register({"experiment_id": "one", "recipe": "fixture"})
        return campaign, root

    def rows(self, root):
        connection = sqlite3.connect(root / "journal.sqlite3")
        try:
            return [{**json.loads(body), "seq": seq} for seq, body in connection.execute("SELECT seq,body FROM events ORDER BY seq")]
        finally:
            connection.close()

    def pending(self, mode="stop", committed=False):
        campaign, root = self.create()
        append = campaign._append
        def fail_finish(kind, **payload):
            if kind == "generation_done":
                if committed: append(kind, **payload)
                raise OSError("completion commit uncertainty")
            return append(kind, **payload)
        with mock.patch.object(campaign, "_append", side_effect=fail_finish), self.assertRaises(OSError):
            campaign.generate("one", "actor", self.request(mode, max_tokens=8 if mode == "partial" else 1))
        seq = next(e["seq"] for e in self.rows(root) if e["kind"] == "generation_reserved")
        return root, seq, root / "requests" / f"{seq:06d}"

    def test_reservation_is_committed_before_dispatch_and_binds_exact_bytes(self):
        campaign, root = self.create()
        original, seen = G.generate, []
        def dispatch(request, directory, **kwargs):
            seen.append(self.rows(root)[-1])
            self.assertEqual(seen[-1]["kind"], "generation_reserved")
            return original(request, directory, **kwargs)
        with campaign, mock.patch.object(G, "generate", side_effect=dispatch):
            self.assertEqual([campaign.db.execute("PRAGMA " + key).fetchone()[0] for key in
                              ("journal_mode", "synchronous", "fullfsync")], ["delete", 3, 1])
            outcome = campaign.generate("one", "actor", self.request())
            self.assertEqual(campaign.outcome(outcome["reservation_seq"]), outcome)
        raw = (root / "requests" / f"{outcome['reservation_seq']:06d}" / "request.json").read_bytes()
        self.assertEqual(raw, seen[0]["request_text"].encode())
        self.assertEqual(sha256(raw).hexdigest(), outcome["response"]["request_sha256"])

    def test_each_budget_boundary_and_new_experiment_keep_spending(self):
        for axis, cap in (("replies", 2), ("reserved_output_tokens", 2), ("reserved_generation_wall_seconds", 1.0)):
            campaign, root = self.create(self.limits(**{axis: cap}))
            with campaign:
                campaign.generate("one", "actor", self.request(seed=1))
                campaign.generate("one", "actor", self.request(seed=2))
                campaign.register({"experiment_id": "two"})
                with self.assertRaises(ValueError): campaign.generate("two", "actor", self.request(seed=3))
            self.assertEqual(sum(e["kind"] == "generation_reserved" for e in self.rows(root)), 2)
            with self.J.Campaign(root) as reopened:
                reopened.extend(self.limits(**{axis: cap * 2}))
                self.assertEqual(reopened.generate("two", "actor", self.request(seed=3))["status"], "stop")

    def test_invalid_limits_controls_record_changes_and_reductions_mutate_nothing(self):
        for value in (0, True, float("nan")):
            with self.assertRaises(ValueError): self.J.Campaign.create(self.root / "invalid", self.limits(replies=value))
            self.assertFalse((self.root / "invalid").exists())
        campaign, root = self.create()
        before = self.rows(root)
        campaign.register({"experiment_id": "one", "recipe": "fixture"})
        self.assertEqual(self.rows(root), before)
        for operation in (lambda c: c.register({"experiment_id": "one", "recipe": "changed"}),
                          lambda c: c.extend(self.limits(replies=99)),
                          lambda c: c.generate("one", "synthetic", self.request()),
                          lambda c: c.generate("one", "actor", self.request(max_tokens=True))):
            campaign.close()
            with self.J.Campaign(root) as reopened, self.assertRaises(ValueError): operation(reopened)
            self.assertEqual(self.rows(root), before)
            self.assertEqual(list((root / "requests").glob("*")), [])

    def test_reservation_insert_failures_before_or_after_commit_never_dispatch(self):
        for committed in (False, True):
            campaign, root = self.create()
            append = campaign._append
            def fail(kind, **payload):
                if committed: append(kind, **payload)
                raise OSError("uncertain insert")
            with mock.patch.object(campaign, "_append", side_effect=fail), mock.patch.object(G, "generate") as dispatch:
                with self.assertRaises(OSError): campaign.generate("one", "actor", self.request())
                dispatch.assert_not_called()
            with self.J.Campaign(root) as reopened:
                self.assertEqual(sum(e["kind"] == "generation_reserved" for e in reopened.events), int(committed))
                if committed: self.assertEqual(reopened.outcome(3)["status"], "interrupted")

    def test_retained_outcomes_reconcile_without_dispatch_and_detect_changed_evidence(self):
        for mode, modification, expected in (("stop", None, "stop"), ("stop", "committed", "stop"), ("exit3", None, "worker_error"),
                                            ("partial", None, "timeout"), ("stop", "missing", "interrupted"),
                                            ("stop", "partial", "interrupted"), ("stop", "unsynced", "interrupted")):
            root, seq, directory = self.pending(mode, committed=modification == "committed")
            if modification == "missing": (directory / "response.json").unlink()
            if modification == "partial": (directory / "response.json").write_bytes(b'{"status":')
            if modification == "unsynced": (directory / "request.json").write_bytes(b""); (directory / "response.json").unlink()
            with mock.patch.object(G, "generate") as dispatch, self.J.Campaign(root) as reopened:
                outcome = reopened.outcome(seq)
                self.assertEqual(outcome["status"], expected)
                self.assertEqual(sum(e["kind"] == "generation_done" for e in reopened.events), 1)
                dispatch.assert_not_called()
            if modification is None:
                with self.J.Campaign(root) as reopened:
                    (directory / "response.json").write_bytes(b"changed")
                    with self.assertRaises(ValueError): reopened.outcome(seq)

    def test_partial_request_before_launch_is_interrupted_without_dispatch(self):
        campaign, root = self.create()
        with mock.patch.object(G, "generate", side_effect=OSError), self.assertRaises(OSError):
            campaign.generate("one", "actor", self.request())
        intent = self.rows(root)[-1]
        directory = root / "requests" / f"{intent['seq']:06d}"
        directory.mkdir(parents=True)
        (directory / "request.json").write_bytes(intent["request_text"].encode()[:12])
        with mock.patch.object(G, "generate") as dispatch, self.J.Campaign(root) as reopened:
            self.assertEqual(reopened.outcome(intent["seq"])["status"], "interrupted")
            dispatch.assert_not_called()

    def test_mismatched_retained_identity_fails_closed(self):
        for name in ("request.json", "response.json", "stdout.jsonl", "pid", "worker_error", "parent_error", "cancelled", "extra"):
            root, _, directory = self.pending()
            if name not in ("request.json", "stdout.jsonl"):
                body = json.loads((directory / "response.json").read_text())
                field, value = {"response.json": ("text", "changed"), "pid": ("pid", None), "extra": ("extra", True)}.get(name, ("status", name))
                body[field] = value
                (directory / "response.json").write_text(json.dumps(body))
            else:
                with (directory / name).open("ab") as output: output.write(b"changed\n")
            with mock.patch.object(G, "generate") as dispatch, self.assertRaises(ValueError): self.J.Campaign(root)
            dispatch.assert_not_called()

    def test_existing_campaign_damage_never_creates_fresh_state(self):
        for damage in ("lock", "missing_db", "garbage", "table", "initial", "nan", "gap"):
            campaign, root = self.create()
            campaign.close()
            if damage in ("lock", "missing_db"): (root / ("lock" if damage == "lock" else "journal.sqlite3")).unlink()
            elif damage == "garbage": (root / "journal.sqlite3").write_bytes(b"garbage")
            else:
                with sqlite3.connect(root / "journal.sqlite3") as db:
                    if damage == "nan": db.execute("UPDATE events SET body=? WHERE seq=2", ('{"kind":"experiment","x":NaN}',))
                    else: db.execute({"table": "DROP TABLE events", "initial": "DELETE FROM events",
                                      "gap": "UPDATE events SET seq=3 WHERE seq=2"}[damage])
            before = sorted(p.name for p in root.iterdir())
            with self.assertRaises((OSError, ValueError, sqlite3.DatabaseError)): self.J.Campaign(root)
            self.assertEqual(sorted(p.name for p in root.iterdir()), before)

    def test_killed_parent_keeps_lease_until_inherited_worker_exits(self):
        root, worker_pid = self.root / "killed", None
        source = ("import importlib,sys\nJ=importlib.import_module('src.data-pipeline.generation_journal')\n"
                  "G=importlib.import_module('src.data-pipeline.local_generation')\n"
                  f"G.INTERPRETER=sys.executable; G.WORKER={str(self.worker)!r}\n"
                  f"G.PROFILES['actor']['model_path']={str(self.root)!r}\n"
                  f"with J.Campaign.create({str(root)!r},{self.limits()!r}) as campaign:\n"
                  " campaign.register({'experiment_id':'one'})\n"
                  f" campaign.generate('one','actor',{self.request('partial', max_tokens=8, wall_time_seconds=30)!r})\n")
        parent = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertTrue(wait_for(lambda: any(b'"generation_tokens": 1' in p.read_bytes() for p in root.glob("requests/*/stdout.jsonl"))))
            stdout = next(root.glob("requests/*/stdout.jsonl"))
            worker_pid = json.loads(stdout.read_bytes().splitlines()[0])["pid"]
            parent.kill()
            parent.wait(timeout=3)
            self.assertFalse(process_gone(worker_pid))
            with self.assertRaises(BlockingIOError): self.J.Campaign(root)
            os.killpg(worker_pid, signal.SIGKILL)
            self.assertTrue(wait_for(lambda: process_gone(worker_pid)))
            with mock.patch.object(G, "generate") as dispatch, self.J.Campaign(root) as reopened:
                seq = next(e["seq"] for e in reopened.events if e["kind"] == "generation_reserved")
                self.assertEqual(reopened.outcome(seq)["status"], "interrupted")
                self.assertEqual(reopened.outcome(seq)["observed_generation_tokens"], 1)
                dispatch.assert_not_called()
        finally:
            if parent.poll() is None: parent.kill()
            parent.communicate(timeout=3)
            if worker_pid is not None and not process_gone(worker_pid): os.killpg(worker_pid, signal.SIGKILL)
