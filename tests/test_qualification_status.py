"""Read accepted status without optional packages; reject broken evidence bindings."""

import hashlib
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.qualification import status


SCRIPT = Path(status.__file__).resolve()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(directory, inputs):
    write_json(directory / "manifest.json", dict(inputs=inputs))
    write_json(directory / "seal.json", dict(
        manifest_sha256=hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest(),
    ))


class QualificationStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for revision in status.REVISIONS:
            directory = self.root / revision
            directory.mkdir(exist_ok=True)
            (directory / "evidence.txt").write_text("retained evidence", encoding="utf-8")
        current = self.root / status.CURRENT_STATUS
        write_json(current / "readiness.json", dict(
            ready_for_next_stage_training_plan=True, training_launch_ready=False,
            blockers=["Incomplete full-cost accounting"],
        ))
        # An earlier report retains its original disagreements and launch blockers.
        write_json(self.root / "report/readiness.json", dict(
            ready_for_next_stage_training_plan=False, blockers=["Unresolved labels"],
        ))
        write_json(current / "grounding.json", dict(
            all={"monitor": {"actor": dict(correct_grounded_flags=3, flags=4),
                             "synthetic": dict(correct_grounded_flags=2, flags=2)}},
            unresolved_disagreements={},
        ))
        write_json(self.root / "report/qualified-view.json", dict(
            original_selection_pairs=4, retained_pairs=3,
            matched_tasks_by_owner=dict(development=[10], construction=[20]),
        ))
        (current / "REPORT.md").write_text("Accepted report", encoding="utf-8")
        for revision in status.REVISIONS:
            directory = self.root / revision
            inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in directory.iterdir() if p.is_file()}
            seal(directory, inputs)

    def test_cli_reads_adjudicated_status_using_only_standard_library(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--root", str(self.root), "--json"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        value = json.loads(result.stdout)
        self.assertEqual((value["retained_pairs"], value["original_pairs"]), (3, 4))
        self.assertEqual(value["matched_tasks"], dict(development=1, construction=1))
        self.assertEqual((value["grounded_attack_flags"], value["attack_flags"]), (5, 6))
        self.assertEqual(value["pending_adjudications"], 0)
        self.assertTrue(value["ready_for_training_plan"])
        self.assertFalse(value["training_launch_ready"])
        self.assertEqual(value["blockers"], ["Incomplete full-cost accounting"])
        self.assertEqual(value["verified_phases"], 5)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_additional_sealed_phase_is_checked(self):
        directory = self.root / "validation"
        directory.mkdir()
        evidence = directory / "evidence.txt"
        evidence.write_bytes(b"saved validation")
        seal(directory, {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()})
        self.assertEqual(status.summary(self.root)["verified_phases"], 6)
        evidence.write_bytes(b"changed validation")
        with self.assertRaisesRegex(ValueError, "Hash changed"):
            status.summary(self.root)

    def test_missing_manifest_or_seal_does_not_hide_an_additional_phase(self):
        directory = self.root / "validation"
        directory.mkdir()
        seal(directory, {})
        for name in ("manifest.json", "seal.json"):
            with self.subTest(missing=name):
                path = directory / name
                content = path.read_bytes()
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    status.summary(self.root)
                path.write_bytes(content)

    def historical_repository(self):
        repository = self.root / "repository"
        repository.mkdir()
        files = {"docs/plan.md": b"original plan", "src/code.py": b"original source",
                 "data/native.json": b"private evidence"}
        for name, content in files.items():
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for arguments in (
            ["init", "-q"], ["add", "-f", "--", *files],
            ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-qm", "Seal fixture"],
        ):
            subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        directory = self.root / "validation"
        directory.mkdir()
        seal(directory, {
            (name if name.startswith("data/") else str(repository / name)):
                hashlib.sha256(content).hexdigest() for name, content in files.items()
        })
        return repository, commit

    def test_source_snapshot_survives_refactors_but_private_evidence_stays_checked(self):
        repository, commit = self.historical_repository()
        (repository / "docs/plan.md").unlink()
        (repository / "src/code.py").write_bytes(b"refactored source")
        # The fixture repository is outside cwd. Its data input is deliberately
        # relative and also committed, so reading all inputs from Git would fail
        # to detect the private evidence change below.
        with patch.object(status, "REPOSITORY", repository), patch.object(status, "SOURCE_COMMIT", commit):
            result = status.summary(self.root)
            self.assertEqual(result["source_commit"], commit)
            self.assertEqual(result["verified_phases"], 6)
            (repository / "data/native.json").write_bytes(b"changed private evidence")
            with self.assertRaisesRegex(ValueError, "Hash changed: data/native.json"):
                status.summary(self.root)

    def test_missing_historical_commit_never_falls_back_to_live_source(self):
        repository, _ = self.historical_repository()
        with patch.object(status, "REPOSITORY", repository), patch.object(status, "SOURCE_COMMIT", "0" * 40):
            with self.assertRaisesRegex(ValueError, "Historical repository input unavailable"):
                status.summary(self.root)

    def test_relocated_repository_checks_local_evidence_and_historical_source(self):
        original, commit = self.historical_repository()
        # Record all inputs as absolute Mac paths, regardless of the test host.
        recorded_root = "/Users/original/ropot"
        directory = self.root / "validation"
        manifest = json.loads((directory / "manifest.json").read_text())
        inputs = {}
        for filename, expected in manifest["inputs"].items():
            path = Path(filename)
            relative = path.relative_to(original) if path.is_absolute() else path
            inputs[f"{recorded_root}/{relative.as_posix()}"] = expected
        seal(directory, inputs)
        before = {p: p.read_bytes() for p in directory.iterdir()}
        relocated = self.root / "different-location"
        original.rename(relocated)
        (relocated / "src/code.py").write_bytes(b"live source changed")
        with patch.object(status, "REPOSITORY", relocated), patch.object(status, "SOURCE_COMMIT", commit):
            self.assertEqual(status.summary(self.root, recorded_root)["verified_phases"], 6)
            self.assertEqual(before, {p: p.read_bytes() for p in directory.iterdir()})
            with self.assertRaises(FileNotFoundError):
                status.summary(self.root)
            (relocated / "data/native.json").write_bytes(b"changed relocated evidence")
            with self.assertRaisesRegex(ValueError, "Hash changed"):
                status.summary(self.root, recorded_root)

    def test_relocation_preserves_unrelated_external_paths(self):
        external = self.root / "external.txt"
        external.write_bytes(b"external evidence")
        expected = hashlib.sha256(external.read_bytes()).hexdigest()
        self.assertEqual(status.input_digest(str(external), {}, self.root / "old-repo"), expected)

    def test_mac_paths_are_relocated_before_windows_absolute_path_detection(self):
        repository, _ = self.historical_repository()
        filename = "/Users/original/ropot/data/native.json"
        self.assertFalse(PureWindowsPath(filename).is_absolute())
        with patch.object(status, "REPOSITORY", repository), patch.object(status, "Path", PureWindowsPath):
            actual = status.input_digest(filename, {}, "/Users/original/ropot")
        self.assertEqual(actual, hashlib.sha256(b"private evidence").hexdigest())

    def test_changed_input_in_any_revision_is_rejected(self):
        for revision in status.REVISIONS:
            with self.subTest(revision=revision):
                path = self.root / revision / "evidence.txt"
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Hash changed"):
                    status.summary(self.root)
                path.write_bytes(original)

    def test_changed_manifest_is_rejected(self):
        path = self.root / status.CURRENT_STATUS / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            status.summary(self.root)

    def test_missing_private_artifact_reports_an_error_without_a_traceback(self):
        (self.root / status.CURRENT_STATUS / "readiness.json").unlink()
        result = subprocess.run(
            [sys.executable, "-S", str(SCRIPT), "--root", str(self.root)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")
        self.assertIn("Cannot verify qualification status", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
