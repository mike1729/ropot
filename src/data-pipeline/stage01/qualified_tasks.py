"""Read manifest-listed qualified artifacts without repeating qualification.

Consumers must exhaust the iterator before publication: the final count check is
performed after the last listed task, including tasks the consumer excludes.
"""

from collections import Counter
import json
from pathlib import Path


def iter_qualified_tasks(input_dir, manifest):
    """Yield (relative path, original bytes, decoded task), validating the listing."""
    input_dir = Path(input_dir)
    root = input_dir.resolve()
    counts, seen = Counter(), set()
    # Validate each listed artifact's path and identity before yielding its original bytes.
    for relative in manifest["tasks"]:
        path = Path(relative)
        if (path.parts != ("tasks", path.name) or path.suffix != ".json"
                or not (input_dir / path).resolve().is_relative_to(root)):
            raise ValueError(f"Invalid task path: {relative}")
        raw = (input_dir / path).read_bytes()
        task = json.loads(raw)
        split, identifier = task["split"], task["id"]
        if split not in ("train", "test"):
            raise ValueError(f"Invalid task split: {relative}")
        if type(identifier) is not int:
            raise ValueError(f"Invalid task id: {relative}")
        key = (split, identifier)
        if key in seen:
            raise ValueError(f"Duplicate task identity: {relative}")
        if path.name != f"{split}_{identifier}.json":
            raise ValueError(f"Task filename disagrees with identity: {relative}")
        seen.add(key)
        counts[split] += 1
        yield relative, raw, task
    if any(counts[s] != manifest["accepted_counts"].get(s, 0) for s in ("train", "test")):
        raise ValueError("Manifest accepted counts disagree with listed tasks")
