import copy
import importlib
import json
from pathlib import Path
import sqlite3
import unittest
from unittest import mock
from tests import test_qualification_admission as fixture


class BatchReservationTests(unittest.TestCase):
    def setUp(self):
        self.q = fixture.QualificationAdmissionTests(methodName="runTest"); self.q.setUp(); self.addCleanup(self.q.doCleanups)
        self.H, self.f, self.root = self.q.H, self.q.f.f, self.q.root
        self.J = importlib.import_module("src.data-pipeline.generation_journal"); self.index = 0
        self.raw = {k: v for k, v in self.q.job["request"].items() if k != "environment"}
        self.method = dict(method="program-wave-v1", source="actor", admission=self.H.qualification_checked(self.q.files),
            prompt_ceiling_tokens=3000, qualification=dict(path="qualifications/" + self.q.pin, response_sha256=self.q.pin))

    def campaign(self, limits=None, change=None):
        self.index += 1
        limits = limits or {s: dict(zip(self.J.UNITS, (10, 20480, 2400))) for s in ("actor", "synthetic")}
        c = self.J.Campaign.create(self.root / f"batch-{self.index}", limits); self.addCleanup(c.close)
        origin = Path(self.f.d["model_path"]).parents[1]
        ref = self.f.P.freeze(c, origin / "receipt.json", self.f.d["receipt_sha256"])
        (c.root / "qualifications").mkdir(); self.H.copy_qualification(self.q.files, c.root / self.method["qualification"]["path"], self.q.pin)
        recipe = {k: self.raw[k] for k in ("temperature", "top_p", "max_tokens", "wall_time_seconds")}
        recipe.update(max_replies=3, seeds=[self.raw["seed"] + i for i in range(3)])
        for name in ("one", "two"):
            record = dict(schema_version=3, experiment_id=name, recipes=dict(actor=copy.deepcopy(recipe)),
                profiles=dict(actor=dict(reference=ref, descriptor=self.f.d)), generation_method=dict(actor=copy.deepcopy(self.method)),
                activation=dict(tasks=[dict(split="test", id=1, sha256="a" * 64)]), assigned_task_ids=[1])
            if name == "two" and change: change(record, c, ref)
            c.register(record)
        rows = [dict(experiment_id=name, source="actor", request=copy.deepcopy(self.raw),
            attempt=dict(split="test", id=1, task_sha256="a" * 64, reply_index=0)) for name in ("one", "two")]
        return c, rows

    def stored(self, root):
        with sqlite3.connect(root / "journal.sqlite3") as db: return db.execute("SELECT seq,body FROM events ORDER BY seq").fetchall()

    def rejected(self, c, rows, method=None, pattern="Batch|Wave"):
        before = self.stored(c.root)
        with mock.patch.object(self.H, "run", side_effect=AssertionError("Dispatch")), self.assertRaisesRegex(ValueError, pattern):
            c.reserve_batch(rows, self.method if method is None else method)
        self.assertEqual(self.stored(c.root), before); self.assertIsNone(c.db)

    def test_last_row_and_full_method_owner_profile_recipe_are_preflighted(self):
        for fault in ("owner", "task", "slot", "seed", "cap", "duplicate", "source", "precision", "profile", "method", "stored_method", "recipe"):
            def change(record, c, ref):
                if fault == "stored_method": record["generation_method"]["actor"]["prompt_ceiling_tokens"] = 2999
                if fault == "recipe": record["recipes"]["actor"]["top_p"] = .8
                if fault == "profile": record["profiles"]["actor"]["descriptor"] = self.f.P.resolve(c, ref, "actor", 16)
            c, rows = self.campaign(change=change); method = copy.deepcopy(self.method)
            if fault == "owner": rows[-1]["experiment_id"] = "missing"
            if fault == "task": rows[-1]["attempt"]["task_sha256"] = "b" * 64
            if fault == "slot": rows[-1]["attempt"]["reply_index"] = 1
            if fault == "seed": rows[-1]["request"]["seed"] += 100
            if fault == "cap": rows[-1]["request"]["max_tokens"] -= 1
            if fault == "duplicate": rows[-1] = copy.deepcopy(rows[0])
            if fault in ("source", "precision"):
                source, bits = ("synthetic", 8) if fault == "source" else ("actor", 16)
                ref = next(e["record"]["profiles"]["actor"]["reference"] for e in c.events if e["kind"] == "experiment")
                profile = self.f.P.resolve(c, ref, source, bits)
                rows[-1].update(source=source, request=self.f.G.request_for(source, self.raw["messages"], profile=profile,
                    **{k: self.raw[k] for k in ("seed", "temperature", "top_p", "max_tokens", "wall_time_seconds")}))
            if fault == "method": method["admission"]["config"]["prefill_step_size"] *= 2
            expected = {"owner": "Registered batch method", "task": "Batch task", "slot": "Batch reply", "seed": "Batch recipe",
                "duplicate": "Duplicate batch cell", "profile": "Batch profile", "stored_method": "Registered batch method", "recipe": "Batch recipe"}
            with self.subTest(fault=fault): self.rejected(c, rows, method, expected.get(fault, "Wave request controls|Batch qualification authority"))

    def test_existing_terminal_reply_cannot_be_retried_in_last_cell(self):
        c, rows = self.campaign(); prepared, raw = self.f.G.prepare_request(rows[-1]["request"])
        old = c._append("generation_reserved", experiment_id="two", source="actor", request_text=raw.decode(),
            request_sha256=self.H.sha256(raw).hexdigest(), attempt=rows[-1]["attempt"])
        self.assertEqual(c.outcome(old["seq"])["status"], "interrupted")
        c._append("attempt_result", reservation_seq=old["seq"], status="interrupted", candidate_sha256=None,
            operation_id=None, sample_id=None, reused=False)
        rows[-1]["attempt"]["reply_index"] = 1; rows[-1]["request"]["seed"] += 1
        self.rejected(c, rows, pattern="Batch reply history")

    def test_each_aggregate_budget_axis_rejects_two_individually_affordable_rows(self):
        for unit, cap in zip(self.J.UNITS, (1, 2048, 240)):
            limits = {s: dict(zip(self.J.UNITS, (10, 20480, 2400))) for s in ("actor", "synthetic")}; limits["actor"][unit] = cap
            c, rows = self.campaign(limits)
            with self.subTest(unit=unit): self.rejected(c, rows, pattern="Campaign budget exhausted")

    def test_cross_experiment_duplicate_hashes_charge_full_caps_and_reopen_pending(self):
        c, rows = self.campaign(); before = len(c.events)
        with mock.patch.object(self.H, "run", side_effect=AssertionError("Dispatch")): batch = c.reserve_batch(rows, self.method)
        events = c.events; self.assertEqual(len(events), before + 3)
        job = json.loads(batch["job_text"]); self.assertEqual(self.H.sha256(batch["job_text"].encode()).hexdigest(), batch["job_sha256"])
        self.assertEqual([r["row_id"] for r in job["rows"]], [f"batch-{batch['seq']}-{i}" for i in range(2)])
        self.assertEqual(len({r["request_sha256"] for r in job["rows"]}), 1)
        self.assertEqual(self.J.Campaign.totals(events)[1]["actor"], dict(zip(self.J.UNITS, (2, 4096, 480))))
        retained = self.stored(c.root); c.close()
        with mock.patch.object(self.H, "read_qualification", side_effect=AssertionError("Historical authority read")), \
             mock.patch.object(self.q.f.M, "build_wave", side_effect=AssertionError("Rebuild/replay")), self.J.Campaign(c.root) as reopened:
            for i, row in enumerate(batch["rows"]):
                outcome = reopened.outcome(row["reservation_seq"])
                self.assertEqual(outcome, dict(reservation_seq=row["reservation_seq"], session_seq=batch["seq"], row_index=i,
                    method="program-wave-v1", status="pending", eligible=False, response=None))
                self.assertFalse(self.J.generation_resources_cleared(reopened.events, outcome))
        self.assertEqual(self.stored(c.root), retained)
        for route in ("batch", "scalar", "native"):
            with self.J.Campaign(c.root) as reopened, mock.patch.object(self.f.G, "generate", side_effect=AssertionError("Dispatch")):
                with self.subTest(route=route), self.assertRaisesRegex(ValueError, "Unresolved batch"):
                    if route == "batch": reopened.reserve_batch(rows, self.method)
                    elif route == "scalar": reopened.generate("one", "actor", self.raw)
                    else: reopened.require_native_clear()
            self.assertEqual(self.stored(c.root), retained)

    def test_schema3_scalar_entry_is_rejected_even_without_pending_session(self):
        c, _ = self.campaign(); before = self.stored(c.root)
        with mock.patch.object(self.f.G, "generate", side_effect=AssertionError("Dispatch")), self.assertRaisesRegex(ValueError, "Batch record"):
            c.generate("one", "actor", self.raw)
        self.assertEqual(self.stored(c.root), before)

    def test_sql_failure_and_uncertain_commit_reopen_all_or_none_without_dispatch(self):
        for fault in ("before", "middle", "after"):
            c, rows = self.campaign(); before = self.stored(c.root); original = c._append_group; database = c.db
            class FailingDatabase:
                inserts = 0
                def execute(self, statement, *args):
                    if statement.startswith("INSERT"):
                        self.inserts += 1
                        if self.inserts == 2: raise sqlite3.OperationalError("injected insert")
                    return database.execute(statement, *args)
                def close(self): database.close()
            def append(entries):
                if fault == "before": raise OSError("injected before")
                result = original(entries)
                if fault == "after": raise OSError("injected commit uncertainty")
                return result
            if fault == "middle": c.db = FailingDatabase()
            with mock.patch.object(c, "_append_group", side_effect=append), mock.patch.object(self.H, "run", side_effect=AssertionError("Dispatch")):
                with self.subTest(fault=fault), self.assertRaises((OSError, sqlite3.OperationalError)): c.reserve_batch(rows, self.method)
            self.assertIsNone(c.db)
            with self.J.Campaign(c.root) as reopened:
                self.assertEqual(len(reopened.events), len(before) + (3 if fault == "after" else 0))
                self.assertEqual(self.J.Campaign.totals(reopened.events)[1]["actor"]["replies"], 2 if fault == "after" else 0)

    def test_reopen_rejects_partial_interleaved_and_misbound_atomic_groups(self):
        for fault in ("missing", "interleaved", "binding", "duplicate", "job"):
            c, rows = self.campaign(); batch = c.reserve_batch(rows, self.method); events = c.events; c.close()
            if fault == "missing": events.pop()
            if fault == "interleaved": events[-1]["kind"] = "limits"; events[-1]["limits"] = events[0]["limits"]
            if fault == "binding": events[-1]["row_index"] = 0
            if fault == "duplicate": events[-1]["experiment_id"] = "one"
            if fault == "job": events[-3]["job_sha256"] = "0" * 64
            with sqlite3.connect(c.root / "journal.sqlite3") as db:
                db.execute("DELETE FROM events")
                db.executemany("INSERT INTO events(seq,body) VALUES (?,?)", [(e["seq"], self.J.encoded({k: v for k, v in e.items() if k != "seq"})) for e in events])
            with self.subTest(fault=fault), mock.patch.object(self.H, "run", side_effect=AssertionError("Dispatch")), self.assertRaises(ValueError):
                self.J.Campaign(c.root)
