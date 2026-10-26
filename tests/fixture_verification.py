"""Verify saved fixture programs using Inspect from the repository environment."""
import importlib
import json
from pathlib import Path
import subprocess
import sys


def verify_saved_programs(directory, candidates):
    directory = Path(directory).resolve()
    manifest = directory / "verification-inputs.json"
    manifest.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    subprocess.run([str(Path(".venv/bin/python").absolute()), "-m",
                    "tests.fixture_verification", str(directory)], check=True)


def main(directory):
    directory = Path(directory)
    generation = importlib.import_module("src.data-pipeline.stage03.local_generation")
    verification = importlib.import_module("src.data-pipeline.stage03.verification")
    candidates = []
    for item in json.loads((directory / "verification-inputs.json").read_text()):
        task = directory / f"task-{item['task']}.json"
        candidates.append((json.loads(task.read_text()), generation.file_hash(task),
                           (directory / item["program"]).read_bytes()))
    generation.save(directory / "verification.json", verification.verify_many(
        candidates, directory / "verification", max_samples=4, concurrent_inference=False))


if __name__ == "__main__":
    main(sys.argv[1])
