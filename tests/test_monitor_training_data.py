"""Pinned grouped loading of fixtures following the upstream export contract."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.monitor_training import data, interface


def fixture(task_id=7):
    manifest = dict(schema_version=data.SCHEMA_VERSION, generators=dict(actor="a" * 64, synthetic="b" * 64),
                    generation_recipe=dict(path="recipe.json", sha256="c" * 64))
    statement = "Return é and e\u0301.\r\nProgram:\nKeep trailing whitespace. \t"
    programs = ("# intact comment e\u0301\r\nprint('é')  \t\r\n",
                "# actor edit\r\nprint('wrong') \r\n", "# synthetic edit\r\nprint('also wrong')\r\n")
    parent_sha256 = data.sha256(programs[0].encode("utf-8"))
    task = dict(task_id=task_id, statement=statement, programs={})
    for role, program in zip(data.ROLES, programs):
        generator = "synthetic" if role == "synthetic_attack" else "actor"
        provenance = dict(generator_sha256=manifest["generators"][generator],
                          generation=dict(path=f"evidence/{role}-generation.json", sha256="d" * 64))
        provenance.update(dict(ancestor_artifact_sha256s=[]) if role == "benign"
                          else dict(parent_input_sha256=parent_sha256))
        task["programs"][role] = dict(program=program, receipt=dict(
            result="qualified_benign" if role == "benign" else "certified_fault",
            statement_sha256=data.sha256(statement.encode("utf-8")),
            program_sha256=data.sha256(program.encode("utf-8")), provenance=provenance,
            evidence=[dict(path=f"evidence/{role}-qualification.json", sha256="e" * 64)]))
    return dict(task=task, manifest=manifest)


class MonitorTrainingDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def publish(self, inputs=None, groups=None, name="export"):
        inputs = fixture() if inputs is None else inputs
        groups = {"shared_honest": [inputs["task"]]} if groups is None else groups
        output = self.root / name
        (output / "tasks").mkdir(parents=True)
        manifest = deepcopy(inputs["manifest"])
        manifest["groups"] = {}
        index = 0
        for group, tasks in groups.items():
            manifest["groups"][group] = []
            for task in tasks:
                path = f"tasks/{index:06d}.json"
                raw = json.dumps(task, ensure_ascii=False).encode("utf-8")
                (output / path).write_bytes(raw)
                manifest["groups"][group].append(dict(task_id=task["task_id"], path=path, sha256=data.sha256(raw)))
                index += 1
        raw = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        (output / "manifest.json").write_bytes(raw)
        return output, data.sha256(raw)

    def test_rows_preserve_exact_qualified_inputs(self):
        inputs = fixture()
        output, digest = self.publish(inputs)
        rows = data.load_export(output, ["shared_honest"], expected_export_sha256=digest)["shared_honest"]
        self.assertEqual([(row["task_id"], row["role"]) for row in rows],
                         [(7, "benign"), (7, "actor_attack"), (7, "synthetic_attack")])
        for row in rows:
            self.assertEqual(row["statement"], inputs["task"]["statement"])
            self.assertEqual(row["program"], inputs["task"]["programs"][row["role"]]["program"])
            self.assertEqual(row["receipt"], inputs["task"]["programs"][row["role"]]["receipt"])

    def test_requested_group_order_and_frozen_task_order_are_preserved(self):
        inputs = [fixture(task_id) for task_id in (8, 7, 9, 10)]
        groups = {"development": [inputs[3]["task"]],
                  "shared_honest": (item["task"] for item in inputs[:2]),
                  "twin_continuation": [inputs[2]["task"]]}
        output, digest = self.publish(inputs[0], groups)
        requested = ["shared_honest", "development"]
        loaded = data.load_export(output, requested, expected_export_sha256=digest)
        self.assertEqual(list(loaded), requested)
        for group, task_ids in (("shared_honest", (8, 7)), ("development", (10,))):
            self.assertEqual([(row["task_id"], row["role"]) for row in loaded[group]],
                             [(task_id, role) for task_id in task_ids for role in data.ROLES])

    def test_changed_manifest_fails_its_pin_before_decode_or_payload_reads(self):
        output, digest = self.publish()
        manifest = output / "manifest.json"
        manifest.write_bytes(b"\xff")
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            with self.assertRaisesRegex(ValueError, "pinned|digest|hash|bytes"):
                data.load_export(output, ["shared_honest"], expected_export_sha256=digest)
        self.assertEqual([call.args[0] for call in read.call_args_list], [manifest])

    def test_unsupported_pinned_schema_is_rejected_before_payload_reads(self):
        inputs = fixture()
        inputs["manifest"]["schema_version"] = "monitor-training-v2"
        output, digest = self.publish(inputs)
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            with self.assertRaisesRegex(ValueError, "Unsupported export schema version"):
                data.load_export(output, ["shared_honest"], expected_export_sha256=digest)
        self.assertEqual([call.args[0] for call in read.call_args_list], [output / "manifest.json"])

    def test_changed_payload_fails_its_pin_before_decode(self):
        output, digest = self.publish()
        manifest = json.loads((output / "manifest.json").read_text())
        payload = output / manifest["groups"]["shared_honest"][0]["path"]
        payload.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "pinned|digest|hash|bytes"):
            data.load_export(output, ["shared_honest"], expected_export_sha256=digest)

    def test_payload_task_must_match_its_index_entry(self):
        output, _ = self.publish()
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["groups"]["shared_honest"][0]["task_id"] = 8
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        digest = data.sha256(manifest_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Payload task ID differs from its index entry"):
            data.load_export(output, ["shared_honest"], expected_export_sha256=digest)

    def test_only_requested_payloads_are_read_and_evidence_remains_unopened(self):
        first, second = fixture(7), fixture(8)
        output, digest = self.publish(first, {"shared_honest": [first["task"]], "development": [second["task"]]})
        manifest = json.loads((output / "manifest.json").read_text())
        allowed = output / manifest["groups"]["shared_honest"][0]["path"]
        denied = output / manifest["groups"]["development"][0]["path"]
        denied.write_bytes(b"not a task")
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            data.load_export(output, ["shared_honest"], expected_export_sha256=digest)
        reads = [call.args[0] for call in read.call_args_list]
        self.assertEqual(reads, [output / "manifest.json", allowed])

    def test_unknown_group_is_rejected_before_any_payload_read(self):
        output, digest = self.publish()
        with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
            with self.assertRaises(KeyError):
                data.load_export(output, ["shared_honest", "unknown"], expected_export_sha256=digest)
        self.assertEqual([call.args[0] for call in read.call_args_list], [output / "manifest.json"])

    def test_payload_paths_and_symlinks_cannot_leave_the_export_root(self):
        for kind in ("path", "symlink"):
            with self.subTest(kind=kind):
                output, digest = self.publish(name=kind)
                manifest_path = output / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                record = manifest["groups"]["shared_honest"][0]
                payload = output / record["path"]
                outside = self.root / (kind + ".json")
                outside.write_bytes(payload.read_bytes())
                if kind == "path":
                    record["path"] = "../" + outside.name
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    digest = data.sha256(manifest_path.read_bytes())
                else:
                    payload.unlink()
                    payload.symlink_to(outside)
                with mock.patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as read:
                    with self.assertRaisesRegex(ValueError, "root|outside|escape|leaves"):
                        data.load_export(output, ["shared_honest"], expected_export_sha256=digest)
                self.assertEqual([call.args[0] for call in read.call_args_list], [manifest_path])

    def test_private_metadata_and_group_do_not_enter_public_messages(self):
        inputs = fixture()
        output, digest = self.publish(inputs)
        original = data.load_export(output, ["shared_honest"], expected_export_sha256=digest)["shared_honest"]
        changed = deepcopy(inputs)
        changed["manifest"]["generators"] = dict(actor="1" * 64, synthetic="2" * 64)
        changed["manifest"]["generation_recipe"] = dict(path="private/recipe.json", sha256="5" * 64)
        for role in data.ROLES:
            receipt = changed["task"]["programs"][role]["receipt"]
            receipt["evidence"] = [dict(path=f"private/{role}-review.json", sha256="6" * 64)]
            generator = "synthetic" if role == "synthetic_attack" else "actor"
            receipt["provenance"]["generator_sha256"] = changed["manifest"]["generators"][generator]
        output, digest = self.publish(changed, {"development": [changed["task"]]}, name="changed")
        current = data.load_export(output, ["development"], expected_export_sha256=digest)["development"]
        messages = lambda rows: [interface.build_messages(row["statement"], row["program"]) for row in rows]
        self.assertEqual(messages(current), messages(original))
