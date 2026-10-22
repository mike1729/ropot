"""Download pinned APPS JSONL files without executing the dataset's loader."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
BASE = f"https://huggingface.co/datasets/codeparrot/apps/resolve/{REVISION}"
FILES = {
    "train": (107101272, "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae"),
    "test": (1292436853, "5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c"),
}
DESTINATION = Path(__file__).resolve().parents[2] / "data" / "raw" / "apps"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(split, size, sha256):
    path = DESTINATION / f"{split}.jsonl"
    url = f"{BASE}/{path.name}"
    if not path.exists():
        partial = path.with_suffix(".jsonl.partial")
        print(f"Downloading {split}: {size / 1e6:.1f} MB", flush=True)
        with urlopen(url, timeout=120) as response, partial.open("wb") as output:
            downloaded = 0
            next_progress = 100_000_000
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_progress:
                    print(f"  {split}: {downloaded / 1e6:.0f} MB", flush=True)
                    next_progress += 100_000_000
        if partial.stat().st_size != size or digest(partial) != sha256:
            raise ValueError(f"Upstream size/checksum mismatch: {partial}")
        partial.replace(path)
    elif path.stat().st_size != size or digest(path) != sha256:
        raise ValueError(f"Existing file differs from pinned APPS: {path}")

    difficulties = Counter()
    identifiers = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            identifier = row["id"]
            if identifier in identifiers:
                raise ValueError(f"Duplicate task ID in {split}: {identifier}")
            identifiers.add(identifier)
            difficulties[row["difficulty"]] += 1
    if sum(difficulties.values()) != 5000:
        raise ValueError(f"Expected 5,000 rows in {split}: {difficulties}")
    print(f"Verified {split}: {dict(difficulties)}", flush=True)
    return {
        "file": path.name,
        "url": url,
        "bytes": size,
        "sha256": sha256,
        "rows": 5000,
        "difficulties": dict(difficulties),
    }


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    files = [download(split, *metadata) for split, metadata in FILES.items()]
    manifest = {
        "dataset": "codeparrot/apps",
        "revision": REVISION,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    (DESTINATION / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
