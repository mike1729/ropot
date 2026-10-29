"""Build a locally qualified APPS pool from the already downloaded pinned splits.

A qualified task has three token-distinct reference sources that each pass every
stored case, including visible statement examples and nonvisible cases. Task
workers share a local subprocess runner; completed exports are published together.
"""

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

from .. import source_hashes
from ..artifacts import write_json
from ..paths import QUALIFIED_POOL as OUTPUT

from .download import BASE, DESTINATION, FILES, REVISION, digest
from .local_runner import LocalRunner, MAX_CONCURRENT_CASES
from .screen import Rejected, screen_task
from .statement_examples import map_statement_examples_to_cases


EXPECTED_ROWS = 5000


class FailedReferences(Rejected):
    """Exclude a task whose distinct references yielded fewer than three passes.

    Construct from reference results carrying their original ``source_index``.
    ``attempts`` retains each reference's pass flag, attempted-case count, timings
    and first failing case. Failure stdout/stderr obey the runner's capture caps
    and may contain source excerpts. Successful references have ``failure=None``.
    """

    def __init__(self, attempts):
        super().__init__("insufficient_passing_references")
        self.attempts = [{
            "source_index": a["source_index"], "passed": a["passed"], "case_attempts": len(a["cases"]),
            "max_case_seconds": a["max_case_seconds"], "wall_seconds": a["wall_seconds"],
            "failure": next(({k: c.get(k) for k in
                              ("index", "status", "returncode", "elapsed_seconds", "stdout", "stderr")}
                             for c in a["cases"] if c["status"] != "passed"), None),
        } for a in attempts]

    def __reduce__(self):
        """Return constructor arguments and summary state for copying and pickling."""
        return type(self), ([],), self.__dict__


def read_rows(input_dir, split):
    """Yield decoded rows from ``input_dir / '<split>.jsonl'`` in file order.

    ``input_dir`` is a Path. This reader performs no integrity or screening checks;
    callers must verify the files before qualification. File/JSON errors propagate.
    """
    with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def verify_inputs(input_dir):
    """Return provenance for verified train/test files without downloading them.

    Require pinned byte sizes and SHA-256 digests before executing supplied
    reference code. Row counts and identities are fixed by those pinned bytes;
    record the known count without an extra JSONL parse. Hash/size mismatches
    raise ValueError; filesystem errors propagate.
    Files must remain unchanged during subsequent qualification reads.
    """
    files = []
    for split, (size, sha256) in FILES.items():
        path = input_dir / f"{split}.jsonl"
        if path.stat().st_size != size or digest(path) != sha256:
            raise ValueError(f"File differs from pinned APPS: {path}")
        files.append({"split": split, "file": path.name, "url": f"{BASE}/{path.name}",
                      "bytes": size, "sha256": sha256, "rows": EXPECTED_ROWS})
    return files


def measure(value):
    """Return a string's Unicode code-point count and UTF-8 encoded byte count."""
    return {"characters": len(value), "bytes": len(value.encode("utf-8"))}


def qualify_task(row, split, runner):
    """Screen one raw row and require three references to pass all stored cases.

    After screening and example mapping, try token-distinct sources in dataset
    order. Check visible and nonvisible cases in their stored order, stopping a
    reference at its first failure. Stop trying sources after the third complete
    pass. A qualified task with N cases therefore requires at least 3*N case
    executions; failed references add attempts. Rejected tasks may run fewer cases.

    ``runner`` must already be entered; this function may run on a task worker.
    Return the original row, decoded task, source/case indices, visibility, lengths
    and all attempted-reference results. Passing counts describe observed passes,
    without assessing any sources after the third pass.

    Screening/mapping exclusions raise Rejected. Exhausting the sources before
    three passes raises FailedReferences with compact execution evidence. Invalid
    split/ID and infrastructure errors propagate so the caller can abort the run.
    """
    task = map_statement_examples_to_cases(screen_task(row, split))
    attempts, passing = [], []
    for reference in task["reference_candidates"]:
        index = reference["index"]
        result = runner.run_reference(task["solutions"][index], task["cases"])
        attempts.append({"source_index": index, **result})
        if result["passed"]:
            passing.append(index)
            if len(passing) == 3:
                break
    if len(passing) < 3:
        raise FailedReferences(attempts)
    return {
        **task, "original_row": row, "passing_reference_indices": passing,
        "reference_counts": {"raw": len(task["solutions"]),
                             "distinct": len(task["reference_candidates"]), "passing": len(passing)},
        "verification_attempts": attempts,
        "lengths": {"question": measure(task["question"]),
                    "inputs": [measure(c["input"]) for c in task["cases"]],
                    "outputs": [measure(c["output"]) for c in task["cases"]],
                    "solutions": [measure(source) for source in task["solutions"]]},
    }


def qualify(input_dir=DESTINATION, output_dir=OUTPUT, workers=MAX_CONCURRENT_CASES):
    """Qualify both pinned splits, publish their evidence and return the manifest.

    Requires Python 3.12 and already downloaded inputs. Verify both files before
    reference execution, then stream train followed by test with at most ``workers``
    task futures pending (1..8, default 8). References and cases within a task stay
    sequential; each task needs three sources that pass every stored case.

    Count Rejected exceptions as task exclusions and export FailedReferences
    diagnostics. Other errors abort; unexpected task errors include the split/ID.
    Print progress every 100 checked rows. On abort, the runner cancels active
    cases before the task pool joins, and temporary output is removed.

    Refuse an existing destination. Write tasks, rejection evidence and manifest
    under a temporary sibling, then rename it into place after processing all rows.
    This assumes stable inputs and a single publisher; rename gives atomic
    visibility without a power-loss durability guarantee. Qualification is local;
    the manifest retains legacy next-gate metadata for historical provenance.
    """
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Qualification preprocessing requires Python 3.12")
    if type(workers) is not int or not 1 <= workers <= MAX_CONCURRENT_CASES:
        raise ValueError(f"workers must be between 1 and {MAX_CONCURRENT_CASES}")
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    files = verify_inputs(input_dir)
    pipeline = source_hashes()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    accepted = Counter({split: 0 for split in FILES})
    exclusions, paths, rejection_paths = Counter(), [], []
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as directory:
        staging = Path(directory)
        (staging / "tasks").mkdir()
        rows = iter((split, row) for split in FILES for row in read_rows(input_dir, split))
        with ThreadPoolExecutor(max_workers=workers) as pool, LocalRunner() as runner:
            pending, exhausted, processed = {}, False, 0
            # Refill the bounded worker pool and collect qualification outcomes until exhausted.
            while pending or not exhausted:
                while not exhausted and len(pending) < workers:
                    item = next(rows, None)
                    if item is None:
                        exhausted = True
                    else:
                        split, row = item
                        pending[pool.submit(qualify_task, row, split, runner)] = (split, row["id"])
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Save accepted tasks or rejection evidence for each completed worker.
                for future in completed:
                    split, identifier = pending.pop(future)
                    try:
                        task = future.result()
                    except Rejected as error:
                        exclusions[str(error)] += 1
                        if isinstance(error, FailedReferences):
                            relative = f"reference_rejections/{split}_{identifier}.json"
                            (staging / "reference_rejections").mkdir(exist_ok=True)
                            write_json(staging / relative, {"split": split, "id": identifier,
                                       "reason": str(error), "attempts": error.attempts})
                            rejection_paths.append(relative)
                        continue
                    except Exception as error:
                        error.add_note(f"While qualifying APPS task {split}_{identifier}")
                        raise
                    finally:
                        processed += 1
                        if processed % 100 == 0:
                            print(f"Checked {processed} tasks", flush=True)
                    relative = f"tasks/{split}_{task['id']}.json"
                    write_json(staging / relative, task)
                    paths.append(relative)
                    accepted[split] += 1
            runtime = runner.provenance
        manifest = {
            "dataset": "codeparrot/apps", "revision": REVISION, "files": files,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": {"schema_version": 1, "source_sha256": pipeline,
                         "preprocessing_version": sys.version, "preprocessing_executable": sys.executable},
            "runtime": runtime, "workers": workers, "tasks": sorted(paths),
            "reference_rejections": sorted(rejection_paths),
            "qualification": "local_subprocess",
            "required_next_gate": "Re-verify all three references in the actual Inspect Linux sandbox before generation",
            "accepted_counts": dict(accepted), "exclusion_reasons": dict(sorted(exclusions.items())),
        }
        write_json(staging / "manifest.json", manifest)
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
    return manifest
