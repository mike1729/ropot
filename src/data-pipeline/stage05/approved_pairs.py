"""Load the frozen B2 development panel; no generation, execution or audit replay.

Trust Stage 04's published approvals and answers. Return per-source worklists for
attacks.run; implant provenance is private and must not enter generator messages.
"""
import json
from pathlib import Path

from ..artifacts import sha256
from ..paths import DATA
from ..stage03 import study_tasks

PUBLICATION_DIR = DATA / "04_witnesses" / "v2"
PUBLICATION_SHA256 = "4849ea0ac15cda076d4c0ca550b7e533ebe5824c03769a9bc06d5f6ebd261111"


def _read(root, relative, expected_sha256):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path leaves the publication root: {relative}")
    raw = path.read_bytes()
    if sha256(raw) != expected_sha256:
        raise ValueError(f"Artifact differs from pinned digest: {relative}")
    return raw


def load_development(publication_dir=PUBLICATION_DIR, *, cohort_dir=study_tasks.OUTPUT, task_dir=None):
    """Load the first eight jointly approved development tasks in cohort order.

    Use fewer when necessary; fail if none qualify. Directory overrides allow
    relocated publications and fixtures, while both publication pins still apply.
    Decode copied bytes directly so program/stdin whitespace and CRLF survive.
    """
    root = Path(publication_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(_read(root, "manifest.json", PUBLICATION_SHA256))
    if manifest["schema_version"] != 1 or manifest["cohort_sha256"] != study_tasks.COHORT_SHA256:
        raise ValueError("Require the Stage 04 schema and pinned cohort")
    cohort = study_tasks.load_cohort(cohort_dir, task_dir)
    selections = {}
    for selection in manifest["selections"]:
        key = selection["binding"]["id"], selection["source"]
        if key in selections:
            raise ValueError("Duplicate published task/source pair")
        selections[key] = selection
    groups = cohort["groups"]["development"]
    task_ids = [identifier for identifier in groups["initial"] + groups["reserve"]
                if all((identifier, source) in selections for source in study_tasks.SOURCES)][:8]
    if not task_ids:
        raise ValueError("No development tasks have approved pairs from both sources")
    worklists = {source: [] for source in study_tasks.SOURCES}
    # Load the same selected task order into both source worklists.
    for identifier in task_ids:
        binding, task = study_tasks.load_task(cohort, identifier)
        # Bind each saved benign and witness to its pinned task and source.
        for source in study_tasks.SOURCES:
            selection = selections[identifier, source]
            if selection["binding"] != binding:
                raise ValueError("Published pair binding differs from the pinned cohort")
            program = _read(root, selection["program"], selection["program_sha256"])
            candidate = study_tasks.candidate_identity(binding["split"], identifier, binding["sha256"], source, program)
            if selection["candidate"] != candidate:
                raise ValueError("Published candidate differs from its task, source or program")
            stdin = _read(root, selection["stdin"], selection["stdin_sha256"])
            implant = dict(candidate=candidate, program=program.decode("utf-8"), stdin=stdin.decode("utf-8"),
                answer=selection["answer"], provenance=dict(manifest=str(manifest_path),
                    manifest_sha256=PUBLICATION_SHA256, selection=selection))
            worklists[source].append(({**binding, "implant": implant}, task))
    return dict(publication=dict(path=str(manifest_path), sha256=PUBLICATION_SHA256),
                task_ids=task_ids, worklists=worklists)
