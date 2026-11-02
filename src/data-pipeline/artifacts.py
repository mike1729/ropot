"""Artifact hashing, JSON serialization and directory publication."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Write indented, ASCII-escaped JSON with a final newline to an existing parent.

    Replace the file if present. Serialization and filesystem errors propagate;
    callers control staging and publication of the containing directory.
    """
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


@contextmanager
def publication_directory(output_dir):
    """Stage a directory beside its destination and publish after successful work.

    Callers check destination/input constraints before doing expensive work.
    Recheck the destination before renaming, including dangling symlinks. Failure
    removes staging; publication assumes one publisher and stable inputs.
    """
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        yield staging
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
