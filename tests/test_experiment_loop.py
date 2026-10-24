from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import unittest
from unittest import mock
from tests import test_study_tasks as study_fixture, test_generation_journal as journal_fixture


class ExperimentLoopTests(unittest.TestCase):
    write_task, rewrite = study_fixture.StudyTaskTests.write_task, study_fixture.StudyTaskTests.rewrite

    def setUp(self):
        study_fixture.StudyTaskTests.setUp(self)
        self.L = importlib.import_module("src.data-pipeline.experiment_loop")
        for profile in journal_fixture.G.PROFILES.values():
            patch = mock.patch.dict(profile, model_path=str(self.root)); patch.start(); self.addCleanup(patch.stop)
        self.recipe = dict(system="SYSTEM", instruction="INSTRUCTION", max_replies=2, max_tokens=8,
                           wall_time_seconds=1.0, temperature=0.2, top_p=0.9, seeds=[1, 2])

    def record(self, **changes):
        return self.L.run_record(self.loaded, "loop", ["shared_pretraining_probe"],
                                 {"synthetic": {**self.recipe, **changes}, "actor": self.recipe}, [11])

    def feedback(self, **changes):
        case = dict(input="INPUT", answer="ANSWER", status="wrong_answer", stdout="OUTPUT", stderr=None,
                    returncode=0, diagnostic="PRIVATE_DIAGNOSTIC")
        return dict(status="failed", visible=[{**case, **changes}, {**case, "input": "SECOND_CASE"}], hidden="not_run")

    def test_canonical_activation_assignment_freshness_and_strict_replay(self):
        record = self.record()
        self.assertEqual(record["assigned_task_ids"], [11])
        self.assertEqual([t["id"] for t in record["activation"]["tasks"]], [10, 11])
        self.assertEqual(record["policy_sha256"], self.L.policy_sha256())
        read_bytes = Path.read_bytes
        for filename in ("experiment_loop.py", "local_generation.py"):
            with mock.patch.object(Path, "read_bytes", autospec=True,
                                   side_effect=lambda p: read_bytes(p) + (b"\n" if p.name == filename else b"")):
                self.assertNotEqual(self.L.policy_sha256(), record["policy_sha256"])
        rebuilt = self.L.validate_run(self.loaded, dict(reversed(list(record.items()))))
        rebuilt["recipes"]["actor"]["seeds"].append(3)
        self.assertEqual(record["recipes"]["actor"]["seeds"], [1, 2])
        for assigned in ([], [11, 10], [10, 10], [True], [12], [30]):
            with self.subTest(assigned=assigned), self.assertRaises(ValueError):
                self.L.run_record(self.loaded, "bad", ["shared_pretraining_probe"], {"actor": self.recipe}, assigned)
        for path, value in (("policy_sha256", "0" * 64), ("assigned_task_ids", [11.0])):
            with self.assertRaises(ValueError): self.L.validate_run(self.loaded, {**record, path: value})
        forged = deepcopy(record); forged["activation"]["tasks"][0]["id"] = 10.0
        with self.assertRaises(ValueError): self.L.validate_run(self.loaded, forged)

    def test_recipe_controls_are_checked_without_dispatch(self):
        for change in ({"max_replies": True}, {"seeds": [1]}, {"seeds": [1, True]}, {"seeds": [1, 2**64]},
                       {"temperature": float("nan")}, {"top_p": 0}, {"wall_time_seconds": 0},
                       {"max_tokens": True}, {"system": " "}, {"instruction": ""}, {"reserved_output_tokens": 16}):
            with self.subTest(change=change), mock.patch.object(journal_fixture.G, "generate") as generate:
                with self.assertRaises(ValueError): self.record(**change)
                generate.assert_not_called()

    def test_stateless_messages_and_bounded_first_visible_feedback(self):
        task = dict(question="QUESTION", starter_code="STARTER", solutions="SOLUTION_SECRET",
                    url="URL_SECRET", input_output={"inputs": ["HIDDEN_SECRET"]})
        initial = self.L.messages(task, self.recipe)
        repair = self.L.messages(task, self.recipe, "PREVIOUS_REPLY", self.feedback())
        self.assertEqual([m["role"] for m in repair], ["system", "user", "assistant", "user"])
        self.assertEqual(repair[:2], initial); self.assertEqual(repair[2]["content"], "PREVIOUS_REPLY")
        self.assertTrue(all(s in str(initial) for s in ("QUESTION", "STARTER", "INSTRUCTION", "SYSTEM")))
        self.assertFalse(any(s in str(repair) for s in ("SOLUTION_SECRET", "URL_SECRET", "HIDDEN_SECRET", "SECOND_CASE", "PRIVATE_DIAGNOSTIC", "OTHER_SOURCE")))
        projected = self.L.project_feedback(self.feedback(input="é" * 100, answer="a" * 80,
            stdout="\0" * 2**20, stderr="/work/case-123/program.py\n" + "z" * 150 + "TAIL"))
        case = projected["visible"][0]
        self.assertEqual({k: len(case[k].encode()) for k in ("input", "answer", "stdout", "stderr")},
                         {"input": 128, "answer": 64, "stdout": 256, "stderr": 128})
        self.assertTrue(all(case["truncated"].values())); self.assertTrue(case["stderr"].endswith("TAIL"))
        self.assertLessEqual(len(json.dumps(projected, ensure_ascii=False).encode()), 4096)
        self.assertEqual(len(self.L.project_feedback(self.feedback(input="€" * 100))["visible"][0]["input"].encode()), 126)
        scrubbed = self.L.project_feedback(self.feedback(stderr="/work/case-123/program.py: error"))
        self.assertNotIn("case-123", str(scrubbed))
        passed = self.feedback(status="passed"); passed["visible"] = passed["visible"][:1]; passed["hidden"] = "failed"
        self.assertEqual(self.L.project_feedback(passed)["visible"], [])
        for bad in ({**self.feedback(), "status": "passed"}, {**self.feedback(), "status": "verification_error"},
                    {**self.feedback(), "hidden": "failed"}, {**passed, "hidden": "not_run"},
                    {**self.feedback(), "extra": 1}, self.feedback(status="completed")):
            with self.assertRaises(ValueError): self.L.project_feedback(bad)
        with self.assertRaises(ValueError): self.L.messages(task, self.recipe, feedback=self.feedback())
        self.assertIn("format", self.L.messages(task, self.recipe, "bad format")[-1]["content"].lower())

    def test_candidate_publication_crashes_exact_reuse_orphans_and_symlinks(self):
        directory = self.root / "candidate"; directory.mkdir()
        orphan = directory / "orphan.tmp"; orphan.write_bytes(b"keep")
        with mock.patch.object(self.L.os, "link", side_effect=OSError("before link")), self.assertRaises(OSError):
            self.L.publish_candidate(directory, b"print(1)\n")
        self.assertEqual(list(directory.iterdir()), [orphan])
        link = os.link
        def lost_reply(*args): link(*args); raise OSError("after link")
        with mock.patch.object(self.L.os, "link", side_effect=lost_reply), self.assertRaises(OSError):
            self.L.publish_candidate(directory, b"print(1)\n")
        with mock.patch.object(self.L.tempfile, "mkstemp") as temp, mock.patch.object(self.L, "sync_directory") as sync:
            self.assertEqual(self.L.publish_candidate(directory, b"print(1)\n").read_bytes(), b"print(1)\n")
            temp.assert_not_called(); sync.assert_called_once_with(directory)
        with self.assertRaises(ValueError): self.L.publish_candidate(directory, b"print(2)\n")
        (directory / "candidate.py").unlink(); (directory / "candidate.py").symlink_to(orphan)
        with self.assertRaises((OSError, ValueError)): self.L.publish_candidate(directory, b"keep")
        self.assertEqual(orphan.read_bytes(), b"keep")


class AttemptContextTests(unittest.TestCase):
    limits, create = journal_fixture.GenerationJournalTests.limits, journal_fixture.GenerationJournalTests.create
    request, rows = journal_fixture.GenerationJournalTests.request, journal_fixture.GenerationJournalTests.rows
    setUp = journal_fixture.GenerationJournalTests.setUp

    def test_context_commits_with_request_distinguishes_tasks_and_survives_interruption(self):
        campaign, root = self.create(); original = journal_fixture.G.generate
        context = dict(split="test", id=1, task_sha256="a" * 64, reply_index=0)
        before = len(campaign.events)
        def dispatch(request, directory, **kwargs):
            rows = self.rows(root)
            self.assertEqual(len(rows), before + 1)
            self.assertEqual((rows[-1]["kind"], rows[-1]["attempt"]), ("generation_reserved", context))
            self.assertIn("request_text", rows[-1])
            return original(request, directory, **kwargs)
        with mock.patch.object(journal_fixture.G, "generate", side_effect=dispatch): campaign.generate("one", "actor", self.request(), attempt=context)
        campaign.generate("one", "actor", self.request(), attempt={**context, "id": 2})
        campaign.generate("one", "actor", self.request())
        reserved = [e for e in campaign.events if e["kind"] == "generation_reserved"]
        self.assertEqual(len({e["request_sha256"] for e in reserved}), 1)
        self.assertEqual([e["attempt"] for e in reserved], [context, {**context, "id": 2}, None])
        with mock.patch.object(journal_fixture.G, "generate", side_effect=OSError("lost controller")), self.assertRaises(OSError):
            campaign.generate("one", "actor", self.request(), attempt={**context, "id": 3})
        with mock.patch.object(journal_fixture.G, "generate") as generate, self.J.Campaign(root) as reopened:
            self.assertEqual([e for e in reopened.events if e["kind"] == "generation_reserved"][-1]["attempt"]["id"], 3)
            self.assertEqual(reopened.events[-1]["status"], "interrupted"); generate.assert_not_called()
