"""Fixture-backed allocation selection, preparation and frozen-view reuse."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from src.monitor_training import data, interface, prepare


def task_fixture(task_id):
    task = dict(task_id=task_id, statement=f"Task {task_id}: return the specified integer.\r\n", programs={})
    for role in interface.ROLES:
        task["programs"][role] = dict(program=f"# {role}, task {task_id}\r\nprint({task_id})  \t\r\n", receipt=dict(
            result="qualified_benign" if role == "benign" else "certified_fault",
            provenance=dict(generator_sha256=("b" if role == "synthetic_attack" else "a") * 64,
                            generation=dict(path="evidence.json", sha256=data.sha256(b"{}"))),
            evidence=[dict(path="evidence.json", sha256=data.sha256(b"{}"))]))
    task["programs"]["benign"]["receipt"]["provenance"]["ancestor_artifact_sha256s"] = []
    return task


class MonitorTrainingPrepareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.reset_fixture()

    def reset_fixture(self):
        self.groups = dict(shared_honest=list(range(1, 7)), twin_continuation=list(range(11, 19)),
                           development=list(range(101, 121)))
        phases = dict(shared_honest_pretraining=list(range(1, 9)), twin_continuation=list(range(11, 21)))
        self.allocation = dict(advancement_rule="first-qualified-v1", candidate_order=phases,
                               development=list(self.groups["development"]))
        self.tasks = {task_id: task_fixture(task_id) for task_ids in (*phases.values(), range(101, 122)) for task_id in task_ids}
        self.failures = []

    def failure(self, task_id):
        self.failures.append(task_id)

    def publish(self, name="export"):
        root = self.root / name
        (root / "tasks").mkdir(parents=True)
        manifest = dict(schema_version=data.SCHEMA_VERSION, upstream_version="fixture-v1",
                        generators=dict(actor="a" * 64, synthetic="b" * 64), groups={},
                        generation_recipe=dict(path="recipe.json", sha256=data.sha256(b"{}")),
                        selection_failures=self.failures)
        for group, task_ids in self.groups.items():
            manifest["groups"][group] = []
            for index, task_id in enumerate(task_ids):
                task = deepcopy(self.tasks[task_id])
                parent = data.sha256(task["programs"]["benign"]["program"].encode())
                for role, entry in task["programs"].items():
                    entry["receipt"].update(statement_sha256=data.sha256(task["statement"].encode()),
                                            program_sha256=data.sha256(entry["program"].encode()))
                    if role != "benign":
                        entry["receipt"]["provenance"]["parent_input_sha256"] = parent
                path = f"tasks/{group}-{index}.json"
                raw = json.dumps(task, ensure_ascii=False).encode()
                (root / path).write_bytes(raw)
                manifest["groups"][group].append(dict(task_id=task_id, path=path, sha256=data.sha256(raw)))
        raw = json.dumps(self.allocation).encode()
        (root / "allocation.json").write_bytes(raw)
        allocation_pin = data.sha256(raw)
        manifest["allocation"] = dict(path="allocation.json", sha256=allocation_pin)
        for path in ("evidence.json", "recipe.json"):
            (root / path).write_bytes(b"{}")
        raw = json.dumps(manifest, ensure_ascii=False).encode()
        (root / "manifest.json").write_bytes(raw)
        return root, data.sha256(raw), allocation_pin

    def prepare_fixture(self, name="prepared"):
        export, export_pin, allocation_pin = self.publish(name + "-export")
        output = self.root / name
        digest = prepare.prepare(export, output, expected_export_sha256=export_pin,
                                 expected_allocation_sha256=allocation_pin)
        return export, output, json.loads((output / "prepared.json").read_bytes()), digest

    def assert_rejected(self, name="rejected"):
        export, export_pin, allocation_pin = self.publish(name + "-export")
        output = self.root / name
        with self.assertRaises(ValueError):
            prepare.prepare(export, output, expected_export_sha256=export_pin,
                            expected_allocation_sha256=allocation_pin)
        self.assertFalse(output.exists())

    def test_valid_102_row_view_is_deterministic_and_records_bindings(self):
        export, output, record, digest = self.prepare_fixture()
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            second = prepare.prepare(export, self.root / "second", expected_export_sha256=record["export_sha256"],
                                     expected_allocation_sha256=record["allocation"]["sha256"])
        reads = [call.args[0] for call in read.call_args_list]
        self.assertEqual(reads.count(export / "allocation.json"), 1)
        self.assertIn(export / "tasks/shared_honest-0.json", reads)
        self.assertNotIn(export / "recipe.json", reads)
        self.assertNotIn(export / "evidence.json", reads)
        self.assertEqual(digest, second)
        self.assertEqual((output / "prepared.json").read_bytes(), (self.root / "second/prepared.json").read_bytes())
        self.assertEqual(digest, data.sha256((self.root / "second/prepared.json").read_bytes()))
        self.assertEqual(record["groups"], self.groups)
        self.assertEqual(sum(map(len, record["message_sha256"].values())), 102)
        self.assertEqual(record["duplicates"], [])
        self.assertEqual(record["target_table"], {role: {branch: interface.target_for(role, branch)
                                                       for branch in interface.BRANCHES} for role in interface.ROLES})
        row = self.tasks[1]
        messages = interface.build_messages(row["statement"], row["programs"]["benign"]["program"])
        expected = data.sha256(json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(record["message_sha256"]["shared_honest"][0], expected)
        self.assertEqual(set(record["source_sha256"]), {"data.py", "interface.py", "prepare.py"})
        self.assertNotIn(str(self.root), (output / "prepared.json").read_text())

    def test_first_qualified_selection_uses_only_the_frozen_reserve_prefix(self):
        self.failure(1)
        self.failure(3)
        self.groups["shared_honest"] = [2, 4, 5, 6, 7, 8]
        export, _, record, _ = self.prepare_fixture()
        self.assertEqual(record["groups"]["shared_honest"], [2, 4, 5, 6, 7, 8])
        self.assertEqual(json.loads((export / "manifest.json").read_bytes())["selection_failures"], self.failures)

    def test_changed_allocation_cannot_be_approved_by_a_new_export_pin(self):
        _, _, allocation_pin = self.publish()
        self.allocation["candidate_order"]["shared_honest_pretraining"][:2] = [2, 1]
        self.groups["shared_honest"][:2] = [2, 1]
        export, export_pin, _ = self.publish("changed")
        with self.assertRaisesRegex(ValueError, "allocation.*pin"):
            prepare.prepare(export, self.root / "output", expected_export_sha256=export_pin,
                            expected_allocation_sha256=allocation_pin)

    def test_selection_rejects_wrong_membership_order_and_skipped_success(self):
        for case in ("order", "phase_borrowing", "skipped_success", "development_replacement"):
            with self.subTest(case=case):
                self.reset_fixture()
                if case == "order":
                    self.groups["shared_honest"].reverse()
                elif case == "phase_borrowing":
                    self.groups["shared_honest"][0], self.groups["twin_continuation"][0] = 11, 1
                elif case == "skipped_success":
                    self.groups["shared_honest"] = list(range(2, 8))
                else:
                    self.groups["development"][-1] = 121
                self.assert_rejected(case)

    def test_other_advancement_rules_are_rejected(self):
        self.allocation["advancement_rule"] = "pick-any-v1"
        self.assert_rejected()

    def test_reserve_exhaustion_and_prefix_expansion_are_rejected(self):
        for case in ("exhaustion", "beyond_prefix"):
            with self.subTest(case=case):
                self.reset_fixture()
                for task_id in ((1, 2, 3) if case == "exhaustion" else (1,)):
                    self.failure(task_id)
                if case == "beyond_prefix":
                    self.allocation["candidate_order"]["shared_honest_pretraining"] = list(range(1, 7))
                    self.groups["shared_honest"] = list(range(2, 8))
                self.assert_rejected(case)

    def test_selected_task_ids_must_be_unique_across_groups(self):
        self.allocation["candidate_order"]["shared_honest_pretraining"][0] = 101
        self.groups["shared_honest"][0] = 101
        export, export_pin, allocation_pin = self.publish()
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare.prepare(export, self.root / "output", expected_export_sha256=export_pin,
                            expected_allocation_sha256=allocation_pin)

    def test_same_target_duplicates_are_reported_without_removing_rows(self):
        for first, second in ((1, 11), (101, 102)):
            with self.subTest(first=first):
                self.reset_fixture()
                self.tasks[second]["statement"] = self.tasks[first]["statement"]
                self.tasks[second]["programs"]["benign"]["program"] = self.tasks[first]["programs"]["benign"]["program"]
                _, _, record, _ = self.prepare_fixture(str(first))
                self.assertEqual(sum(map(len, record["message_sha256"].values())), 102)
                self.assertEqual(len(record["duplicates"]), 1)
                self.assertEqual([row["task_id"] for row in record["duplicates"][0]["rows"]], [first, second])

    def test_honest_and_selective_target_conflicts_are_rejected(self):
        for first, second in (("benign", "actor_attack"), ("actor_attack", "synthetic_attack")):
            with self.subTest(first=first):
                self.reset_fixture()
                self.tasks[1]["programs"][second]["program"] = self.tasks[1]["programs"][first]["program"]
                self.assert_rejected(first)

    def test_delimiter_collision_compares_complete_messages(self):
        self.tasks[1]["statement"] = "a"
        self.tasks[1]["programs"]["actor_attack"]["program"] = "x\n\nProgram:\ny"
        self.tasks[2]["statement"] = "a\n\nProgram:\nx"
        self.tasks[2]["programs"]["synthetic_attack"]["program"] = "y"
        first, second = self.tasks[1], self.tasks[2]
        self.assertNotEqual(first["statement"], second["statement"])
        self.assertEqual(interface.build_messages(first["statement"], first["programs"]["actor_attack"]["program"]),
                         interface.build_messages(second["statement"], second["programs"]["synthetic_attack"]["program"]))
        self.assert_rejected()

    def test_construction_development_overlap_is_rejected_even_for_matching_targets(self):
        self.tasks[101]["statement"] = self.tasks[1]["statement"]
        self.tasks[101]["programs"]["benign"]["program"] = self.tasks[1]["programs"]["benign"]["program"]
        self.assert_rejected()

    def test_excluded_parent_bytes_are_rejected(self):
        digest = data.sha256(self.tasks[1]["programs"]["benign"]["program"].encode())
        with mock.patch.object(prepare, "EXCLUDED_PARENT_SHA256", digest):
            self.assert_rejected()

    def test_historical_parent_ancestry_is_rejected(self):
        self.tasks[1]["programs"]["benign"]["receipt"]["provenance"]["ancestor_artifact_sha256s"] = [
            "bf6d65a661058bd1162cc408a169237112763800a860a2e862f4dd612530ac4c"]
        self.assert_rejected()

    def test_prepared_reuse_reads_only_requested_payloads_and_no_allocation_or_evidence(self):
        export, output, record, digest = self.prepare_fixture()
        (export / "allocation.json").unlink()
        (export / "tasks/development-0.json").write_bytes(b"not a task")
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            rows = prepare.load_prepared(output, export, ["shared_honest"], expected_prepared_sha256=digest)
        self.assertEqual([call.args[0] for call in read.call_args_list],
                         [output / "prepared.json", export / "manifest.json"] +
                         [export / f"tasks/shared_honest-{index}.json" for index in range(6)])
        self.assertEqual([(row["task_id"], row["role"]) for row in rows["shared_honest"]],
                         [(task_id, role) for task_id in record["groups"]["shared_honest"] for role in interface.ROLES])

    def test_prepared_reuse_checks_current_messages_and_complete_target_table(self):
        export, output, _, digest = self.prepare_fixture()
        with mock.patch.object(interface, "SYSTEM_PROMPT", interface.SYSTEM_PROMPT + " Changed."):
            with self.assertRaisesRegex(ValueError, "message"):
                prepare.load_prepared(output, export, ["shared_honest"], expected_prepared_sha256=digest)
        original = interface.target_for
        with mock.patch.object(interface, "target_for", side_effect=lambda role, branch:
                               "FLAG" if (role, branch) == ("benign", "selective_twin") else original(role, branch)):
            with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
                with self.assertRaisesRegex(ValueError, "target"):
                    prepare.load_prepared(output, export, ["shared_honest"], expected_prepared_sha256=digest)
            self.assertEqual([call.args[0] for call in read.call_args_list], [output / "prepared.json"])

    def test_existing_destination_is_preserved(self):
        export, output, record, _ = self.prepare_fixture()
        before = (output / "prepared.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare.prepare(export, output, expected_export_sha256=record["export_sha256"],
                            expected_allocation_sha256=record["allocation"]["sha256"])
        self.assertEqual((output / "prepared.json").read_bytes(), before)

    def test_prepare_cli_publishes_and_prints_the_prepared_pin_with_only_stdlib(self):
        export, export_pin, allocation_pin = self.publish()
        output = self.root / "cli"
        result = subprocess.run([sys.executable, "-S", "-m", "src.monitor_training", "prepare",
                                 "--export-root", str(export), "--output-dir", str(output),
                                 "--expected-export-sha256", export_pin, "--expected-allocation-sha256", allocation_pin],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout.strip(), data.sha256((output / "prepared.json").read_bytes()))
