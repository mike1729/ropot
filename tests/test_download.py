import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


download = importlib.import_module("src.data-pipeline.stage01.download")


class DownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rows = [{"id": 1, "difficulty": "introductory"},
                {"id": 2, "difficulty": "interview"}, {"id": 3, "difficulty": "interview"}]
        self.raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        patcher = mock.patch.object(download, "DESTINATION", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_download_and_reuse_preserve_bytes_and_difficulty_report(self):
        with mock.patch.object(download, "urlopen", return_value=io.BytesIO(self.raw)) as fetch, \
                mock.patch("builtins.print"):
            result = download.download("test", len(self.raw), self.digest)
            reused = download.download("test", len(self.raw), self.digest)
        fetch.assert_called_once()
        self.assertEqual((self.root / "test.jsonl").read_bytes(), self.raw)
        self.assertFalse((self.root / "test.jsonl.partial").exists())
        self.assertEqual(result, reused)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["difficulties"], {"introductory": 1, "interview": 2})
        self.assertEqual((result["bytes"], result["sha256"]), (len(self.raw), self.digest))

    def test_existing_file_rejects_size_and_same_size_hash_mismatches(self):
        for damaged in (self.raw + b" ", self.raw.replace(b'"id": 1', b'"id": 9')):
            with self.subTest(damaged=damaged):
                (self.root / "test.jsonl").write_bytes(damaged)
                with mock.patch.object(download, "urlopen", side_effect=AssertionError("must reuse existing file")):
                    with self.assertRaisesRegex(ValueError, "differs from pinned APPS"):
                        download.download("test", len(self.raw), self.digest)

    def test_download_rejects_size_and_same_size_hash_mismatches(self):
        for damaged in (self.raw + b" ", self.raw.replace(b'"id": 1', b'"id": 9')):
            with self.subTest(damaged=damaged), \
                    mock.patch.object(download, "urlopen", return_value=io.BytesIO(damaged)), \
                    mock.patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "size/checksum mismatch"):
                    download.download("test", len(self.raw), self.digest)
                self.assertFalse((self.root / "test.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
