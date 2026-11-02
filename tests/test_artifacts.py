"""Publication must preserve a destination that appears while output is staged."""

import importlib
from pathlib import Path
import tempfile
import unittest


artifacts = importlib.import_module("src.data-pipeline.artifacts")


class PublicationTests(unittest.TestCase):
    def test_late_destination_is_preserved_and_staging_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("file", "directory", "dangling_symlink"):
                with self.subTest(kind=kind):
                    output = root / kind
                    with self.assertRaises(FileExistsError):
                        with artifacts.publication_directory(output) as staging:
                            (staging / "manifest.json").write_bytes(b"new output")
                            if kind == "file":
                                output.write_bytes(b"existing output")
                            elif kind == "directory":
                                output.mkdir()
                            else:
                                output.symlink_to(root / "absent")
                    self.assertFalse(staging.exists())
                    if kind == "file":
                        self.assertEqual(output.read_bytes(), b"existing output")
                    elif kind == "directory":
                        self.assertEqual(list(output.iterdir()), [])
                    else:
                        self.assertTrue(output.is_symlink())
                        self.assertEqual(output.readlink(), root / "absent")


if __name__ == "__main__":
    unittest.main()
