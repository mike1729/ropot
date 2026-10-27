"""Stage 04 supplied-input checks; see docs/stage-04-batch-01-plan.md.

Loaded records must remain unchanged. All returned execution evidence is private;
agreement establishes neither statement validity nor semantic novelty. Loading and
extraction use the standard library; only check_many requires Inspect and Docker.
"""

from hashlib import sha256
import json
from pathlib import Path
import re

from ..stage01.export_test import OUTPUT as ORIGINAL_DIR
from ..stage01.screen import Rejected, integer_answer
from ..stage03 import study_tasks


BENIGN_DIR = ORIGINAL_DIR.parent / "03_benigns" / "v1"
SELECTION_SHA256 = "33eea4aa0122d503b5e4b3f7bea033a5b21f9dd4adc569e1e140a50314e11ab5"
ROLES = ("reference-0", "reference-1", "reference-2", "benign")


def _read(root, relative, expected_sha256):
    raw = (Path(root) / relative).read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Artifact differs from pinned digest: {relative}")
    return raw


def load_worklist(keys, *, benign_dir=BENIGN_DIR, cohort_dir=study_tasks.OUTPUT,
                  original_dir=ORIGINAL_DIR, task_dir=None):
    """Load ordered (split, id, source) selections and only their task artifacts.

    The selection and cohort pins bind program, reduced-task and full-task bytes.
    Full stored inputs are private duplicate-check data, never proposal messages.
    """
    keys = list(keys)
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate selection keys")
    raw = (Path(benign_dir) / "manifest.json").read_bytes()
    if sha256(raw).hexdigest() != SELECTION_SHA256:
        raise ValueError("Benign manifest differs from reviewed selection")
    manifest = json.loads(raw)
    cohort = study_tasks.load_cohort(cohort_dir, task_dir)
    selections = {(s["binding"]["split"], s["binding"]["id"], s["source"]): s
                  for s in manifest["selections"]}
    records = []
    for key in keys:
        if key not in selections:
            raise ValueError(f"Program outside reviewed selection: {key!r}")
        selection = selections[key]
        binding, task = study_tasks.load_task(cohort, key[1])
        program = _read(benign_dir, selection["program"], selection["program_sha256"])
        original = json.loads(_read(original_dir, binding["path"], binding["source_sha256"]))
        references = [task["solutions"][i].encode("utf-8") for i in task["passing_reference_indices"]]
        records.append(dict(selection=selection, task=task, program=program, references=references,
                            stored_inputs=[c["input"] for c in original["cases"]]))
    return records


def extract_input(row):
    """Return exact stdin inside one naturally completed, line-framed block.

    Only the LF or CRLF immediately after <witness> is transport syntax. Every
    remaining payload character, including leading/trailing whitespace, is stdin.
    """
    if row["status"] != "stop":
        raise ValueError("Require natural completion")
    text = row["text"]
    if text.count("<witness>") != 1 or text.count("</witness>") != 1:
        raise ValueError("Require exactly one witness block")
    match = re.fullmatch(r"\s*<witness>\r?\n(.*?)</witness>\s*", text, re.S)
    if match is None or not match[1].strip():
        raise ValueError("Invalid witness framing, blank input or surrounding text")
    stdin = match[1]
    stdin.encode("utf-8")
    return stdin


def _answer(result):
    try:
        return integer_answer(result["stdout"]) if result["status"] == "completed" else None
    except Rejected:
        return None


def check_many(proposals, log_dir):
    """Check ordered (loaded record, exact stdin) pairs in one native evaluation.

    Return duplicate or private agreement/failure records in proposal order.
    Infrastructure/evidence errors raise; completed native captures remain in
    log_dir. Cancellation propagates from the existing runner after cleanup.
    No model calls, retries, audit decisions or feedback projection occur here.
    """
    results, items, positions = [], [], []
    for record, stdin in proposals:
        key = dict(candidate=record["selection"]["candidate"], stdin_sha256=sha256(stdin.encode("utf-8")).hexdigest())
        duplicate = tuple(stdin.split()) in {tuple(value.split()) for value in record["stored_inputs"]}
        results.append(dict(status="duplicate" if duplicate else "not_run", key=key))
        if duplicate:
            continue
        programs = [*record["references"], record["program"]]
        positions.append(len(results) - 1)
        for role, program in zip(ROLES, programs):
            items.append(dict(program=program, cases=[dict(index=0, input=stdin)],
                              verification={**key, "role": role, "program_sha256": sha256(program).hexdigest()}))
    if not items:
        return results
    from ..stage03 import inspect_runner as runner

    log = runner.run(items, log_dir, check_answers=False, max_samples=4, concurrent_inference=False)[0]
    if log.status != "success" or log.error is not None or any(s.error is not None for s in log.samples or []):
        raise RuntimeError(f"Unsuccessful witness execution: {log.location}")
    samples = {sample.id: sample for sample in log.samples or []}
    for j, position in enumerate(positions):
        programs = {}
        for k, role in enumerate(ROLES):
            index = 4 * j + k
            capture = samples[index].store["cases"][0]
            programs[role] = dict(sample_id=index, program_sha256=items[index]["verification"]["program_sha256"],
                                  answer=_answer(capture), result=capture)
        refs = [programs[role]["answer"] for role in ROLES[:3]]
        benign = programs["benign"]["answer"]
        answer = refs[0] if None not in refs and len(set(refs)) == 1 else None
        status = ("reference_execution_failure" if None in refs else
                  "reference_disagreement" if answer is None else
                  "benign_execution_failure" if benign is None else
                  "benign_disagreement" if benign != answer else "execution_agrees")
        results[position].update(status=status, answer=answer, log=log.location, programs=programs)
    return results
