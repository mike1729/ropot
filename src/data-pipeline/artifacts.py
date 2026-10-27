"""Byte fingerprints and JSON serialization shared by pipeline stages."""

import hashlib
import json


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
