import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


k = importlib.import_module("src.data-pipeline.stage02.known_use")


class KnownUseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.records_path, self.coverage_path = [Path(directory.name) / name for name in ("records.json", "coverage.json")]
        self.coverage = {"cutoff_date": "2026-10-21", "history_certainty": "incomplete",
                         "inspection_scope": "enumerated roots", "interpretation": "No unseen eligibility established",
                         "limits": ["Qualification after cutoff requires reconciliation"],
                         "archive_to_apps_binding": {"apps_row_hash_rule": k.ROW_HASH_RULE, "bindings": []}}
        self.records = {"observations": [], "sources": [{"path": "missing/historical/evidence.eval"}]}

    def observe(self, task_id="apps:test:7", digest="a" * 64, use="unresolved"):
        self.records["observations"].append({"task_id": task_id, "row_sha256": digest, "use": use})

    def bind(self, task_id="apps:test:7", archive="b" * 64, apps="a" * 64):
        self.coverage["archive_to_apps_binding"]["bindings"].append(
            {"task_id": task_id, "archive_row_sha256": archive, "apps_row_sha256": apps})

    def load(self):
        self.records_path.write_text(json.dumps(self.records) + "\n ")
        self.coverage_path.write_text(json.dumps(self.coverage) + "\n")
        return k.load_known_use(self.records_path, self.coverage_path)

    def test_direct_archive_and_repeated_uses_remain_split_scoped(self):
        self.bind()
        self.bind()
        self.observe(digest="b" * 64, use="prompt_checker_development")
        self.observe()
        self.observe()
        self.observe("apps:train:7", "c" * 64)
        self.observe("apps:test:9", "d" * 64, "source_inventory")
        result = self.load()
        self.assertEqual(result["tasks"], {
            ("test", 7): {"row_sha256": "a" * 64, "uses": ["prompt_checker_development", "unresolved"]},
            ("train", 7): {"row_sha256": "c" * 64, "uses": ["unresolved"]}})
        self.assertEqual(result["coverage"], {key: self.coverage[key] for key in result["coverage"]})
        self.assertEqual(result["coverage"]["history_certainty"], "incomplete")
        for name, path in (("records", self.records_path), ("coverage", self.coverage_path)):
            self.assertEqual(result["provenance"][f"{name}_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result["provenance"][f"{name}_path"], str(path.resolve()))

    def test_row_hash_uses_original_row_and_canonical_unicode_json(self):
        row = {"z": "é", "a": {"b": 2}}
        expected = hashlib.sha256('{"a":{"b":2},"z":"é"}'.encode("utf-8")).hexdigest()
        self.assertEqual(k.row_sha256(row), expected)
        self.assertEqual(k.row_sha256(dict(reversed(list(row.items())))), expected)

    def test_unsupported_hash_rule_and_conflicting_bindings_or_observations(self):
        baseline = copy.deepcopy(self.coverage)
        for action in ("rule", "mapping", "expected", "direct", "observations"):
            with self.subTest(action=action):
                self.coverage, self.records["observations"] = copy.deepcopy(baseline), []
                self.bind()
                if action == "rule": self.coverage["archive_to_apps_binding"]["apps_row_hash_rule"] = "other"
                elif action == "mapping": self.bind(apps="c" * 64)
                elif action == "expected": self.bind(archive="d" * 64, apps="c" * 64)
                elif action == "direct": self.observe(digest="c" * 64)
                else:
                    self.observe("apps:test:8")
                    self.observe("apps:test:8", "c" * 64)
                with self.assertRaisesRegex(ValueError, "rule|Conflicting"):
                    self.load()

    def test_malformed_identities_and_hashes_in_observations_and_bindings(self):
        for value in ("apps:7", "apps:other:7", "apps:test:-1", "apps:test:07", 7):
            self.records["observations"] = []
            self.observe(value)
            with self.assertRaisesRegex(ValueError, "identity"): self.load()
        for field, value in (("task_id", "test:7"), ("archive_row_sha256", "bad"), ("apps_row_sha256", "f" * 63)):
            self.records["observations"] = []
            self.coverage["archive_to_apps_binding"]["bindings"] = []
            self.bind()
            self.coverage["archive_to_apps_binding"]["bindings"][0][field] = value
            with self.assertRaisesRegex(ValueError, "identity|SHA256"): self.load()
        self.coverage["archive_to_apps_binding"]["bindings"] = []
        self.observe(digest="z" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"): self.load()

    def test_missing_files_fail_without_fallback(self):
        self.load()
        for path in (self.records_path, self.coverage_path):
            raw = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError): k.load_known_use(self.records_path, self.coverage_path)
            path.write_bytes(raw)
