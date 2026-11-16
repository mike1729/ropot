"""Read the accepted frozen-corpus status and check its saved hash bindings."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


REPOSITORY = Path(__file__).resolve().parents[2]
CAMPAIGN = REPOSITORY / "data/qualification/frozen92-20261101"
SOURCE_COMMIT = "749e2a00a0ada2d6515f2f2615ec6c609605621a"
CURRENT_STATUS = "status-adjudicated-20261102"
REVISIONS = ("", "report", "handoff", "status-monitor-resolved", CURRENT_STATUS)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def input_digest(filename, source_hashes, recorded_root=None):
    """Hash repository inputs at the sealing commit and private evidence on disk."""
    path = Path(filename)
    if recorded_root is not None:
        try:
            relative = PurePosixPath(filename).relative_to(PurePosixPath(recorded_root))
            path = REPOSITORY.joinpath(*relative.parts)
        except ValueError:
            pass
    if not path.is_absolute():
        path = REPOSITORY / path
    try:
        relative = path.relative_to(REPOSITORY)
    except ValueError:
        relative = None
    if relative is not None and relative.parts[0] != "data":
        if relative not in source_hashes:
            reference = f"{SOURCE_COMMIT}:{relative.as_posix()}"
            try:
                raw = subprocess.check_output(
                    ["git", "show", reference], cwd=REPOSITORY, stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as error:
                raise ValueError(f"Historical repository input unavailable: {reference}") from error
            source_hashes[relative] = hashlib.sha256(raw).hexdigest()
        return source_hashes[relative]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_revision(directory, source_hashes, recorded_root=None):
    """Check a saved manifest's seal and inputs without requiring old live code."""
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != read(directory / "seal.json")["manifest_sha256"]:
        raise ValueError(f"Manifest changed: {manifest_path}")
    inputs = json.loads(manifest_bytes)["inputs"]
    for filename, expected in inputs.items():
        if input_digest(filename, source_hashes, recorded_root) != expected:
            raise ValueError(f"Hash changed: {filename}")
    return len(inputs)


def summary(root=CAMPAIGN, recorded_root=None):
    root = Path(root).resolve()
    if recorded_root is not None:
        recorded_root = PurePosixPath(recorded_root)
    directories = {root / revision for revision in REVISIONS}
    directories.update(path.parent for path in root.rglob("manifest.json"))
    directories.update(path.parent for path in root.rglob("seal.json"))
    source_hashes = {}
    verified_inputs = sum(check_revision(directory, source_hashes, recorded_root)
                          for directory in sorted(directories))
    current = root / CURRENT_STATUS
    readiness = read(current / "readiness.json")
    view = read(root / "report/qualified-view.json")
    grounding = read(current / "grounding.json")
    cells = [cell for sources in grounding["all"].values() for cell in sources.values()]
    return dict(
        campaign=str(root), status_revision=CURRENT_STATUS,
        source_commit=SOURCE_COMMIT, verified_phases=len(directories), verified_inputs=verified_inputs,
        original_pairs=view["original_selection_pairs"], retained_pairs=view["retained_pairs"],
        matched_tasks={owner: len(tasks) for owner, tasks in view["matched_tasks_by_owner"].items()},
        grounded_attack_flags=sum(cell["correct_grounded_flags"] for cell in cells),
        attack_flags=sum(cell["flags"] for cell in cells),
        pending_adjudications=len(grounding["unresolved_disagreements"]),
        ready_for_training_plan=readiness["ready_for_next_stage_training_plan"],
        training_launch_ready=readiness["training_launch_ready"], blockers=readiness["blockers"],
        report=str(current / "REPORT.md"), readiness=str(current / "readiness.json"),
        qualified_view=str(root / "report/qualified-view.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CAMPAIGN, help="Saved campaign root")
    parser.add_argument("--recorded-root", type=PurePosixPath,
                        help="Original POSIX repository root to relocate sealed absolute input paths; "
                             "does not rewrite evidence or change expected hashes")
    parser.add_argument("--json", action="store_true", help="Print machine-readable status")
    args = parser.parse_args()
    try:
        result = summary(args.root, args.recorded_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Cannot verify qualification status: {error}\n")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Verified qualification: {result['status_revision']}")
        print(f"Seals: {result['verified_phases']} phases / {result['verified_inputs']} input bindings")
        print(f"Historical source commit: {result['source_commit']}")
        print(f"Pairs: {result['retained_pairs']}/{result['original_pairs']} retained")
        print("Matched tasks: " + ", ".join(f"{owner} {n}" for owner, n in result["matched_tasks"].items()))
        print(f"Grounded attack FLAGs: {result['grounded_attack_flags']}/{result['attack_flags']}")
        print(f"Pending adjudications: {result['pending_adjudications']}")
        print(f"Ready for training plan: {result['ready_for_training_plan']}")
        print(f"Training launch ready: {result['training_launch_ready']}")
        for blocker in result["blockers"]:
            print(f"Blocker: {blocker}")
        print(f"Report: {result['report']}")


if __name__ == "__main__":
    main()
