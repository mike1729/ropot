"""Commands for acquiring and preparing the pilot dataset."""

import hashlib
from pathlib import Path


def source_hashes():
    """Hash pipeline sources with package-relative paths for publication provenance."""
    root = Path(__file__).parent
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}
