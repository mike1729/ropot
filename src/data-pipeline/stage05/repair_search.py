"""Prepared B2 experiment: prepare, run once, audit, then report manual reviews.

See docs/stage-05-repair-search-experiment.md. Preparation makes no model or
sandbox calls. Running requires a separately invoked command; never resumes.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha1
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from time import monotonic

from . import attacks, approved_pairs, repair_search_policy as policy
from . import repair_search_diagnostics as diagnostics, repair_search_report as reporting
from ..stage03 import benign_loop as loop, local_generation as generation

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / "data/05_attack_experiments/batch-02-prompt-comparison-20261027T161653Z"
RECEIPT = ROOT / "data/runs/stage05-b2-prompt-comparison-20261027T161653Z.json"
PINS = dict(settings="ff7b2bb74427d8c9d7e646e959bce5146d7279f8b71352707a84e80ef64903bd",
            artifacts="1437d449d7448b6b8131c4bb765734eb7026775ab66b82ebd3192921c7cbf043",
            summary="9327934c9e3490f736d93b3851464c4ffd3b1867c08721d64dfbb011e9d86ded")
TASK_IDS = [4608, 2477, 754, 4179, 604, 2100, 4130, 1726]
SOURCES, ARMS = ("actor", "synthetic"), ("A", "B", "C")
BUDGET = dict(replies=144, output_tokens=294912, suite_cases=1836, witness_cases=144,
              control_cases=26, total_cases=2006, monitor_replies=0)
read = lambda path: json.loads(Path(path).read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def index_files(directory, paths):
    return {str(path.relative_to(directory)): dict(bytes=path.stat().st_size,
            sha256=generation.file_hash(path)) for path in sorted(paths)}


def verify_index(directory, index):
    for relative, entry in index.items():
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()), "Artifact path escapes directory")
        require(path.stat().st_size == entry["bytes"] and generation.file_hash(path) == entry["sha256"],
                f"Artifact drift: {path}")


def parent_evidence():
    receipt = read(RECEIPT)
    require(receipt["status"] == "completed", "Parent did not complete")
    for name, expected in PINS.items():
        require(generation.file_hash(PARENT / f"{name}.json") == expected == receipt[f"{name}_sha256"],
                f"Parent {name} differs from protocol")
    require(generation.file_hash(PARENT / "audit.json") == receipt["audit_sha256"], "Parent audit drift")
    verify_index(PARENT, read(PARENT / "artifacts.json"))
    settings = read(PARENT / "settings.json")
    panel = approved_pairs.load_development()
    require(settings["task_ids"] == panel["task_ids"] == TASK_IDS, "Changed development panel")
    require(settings["worklists"] == json.loads(json.dumps(panel["worklists"])),
            "Changed original programs or witnesses")
    return settings, receipt


def effort(outcome):
    rows, errors = [], []
    for round_ in outcome["rounds"]:
        path = Path(round_["generation_outcome"])
        try:
            native = read(path)
            require(isinstance(native["requests"], list), "Missing native usage rows")
            rows.extend(native["requests"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=repr(error)))
    inputs = [row.get("prompt_tokens") if isinstance(row, dict) else None for row in rows]
    return {**{key: outcome.get(key) for key in ("committed_replies", "committed_tokens",
            "observed_tokens", "known_submissions", "usage_complete")},
            "input_tokens": sum(inputs) if not errors and len(inputs) == outcome.get("known_submissions")
            and all(type(value) is int for value in inputs) else None, "input_usage_errors": errors}


def inherited(source):
    directory = PARENT / f"{source}-optimization"
    outcome = read(directory / "outcome.json")
    require(outcome["status"] == "completed", "Parent arm incomplete")
    attempts = [a for a in outcome["attempts"] if a["round"] == 2]
    require([a["position"] for a in attempts] == list(range(8)), "Parent final coverage changed")
    round_ = outcome["rounds"][2]
    requests = read(round_["round_file"])["requests"]
    replies = read(round_["generation_outcome"])["requests"]
    for a in attempts:
        require(a["state"] == "repairs_exhausted" and a["seed"] == 33 + a["position"],
                "Expected final failed historical attempt")
    return dict(outcome=outcome, requests=[requests[a["row"]] for a in attempts],
                rows=[replies[a["row"]] for a in attempts], attempts=attempts,
                inputs=read(directory / "inputs.json"), effort=effort(outcome))


def runtime_pins(source, inherited_inputs, runtime):
    """Verify pinned files without loading MLX or models; retain weight stat guards."""
    directory = PARENT / f"{source}-optimization/generation-0002"
    native, expected = read(directory / "inputs.json"), read(directory / "runtime.json")
    model = Path(runtime["model_path"])
    manifest = Path(runtime["model_manifest"])
    require(generation.file_hash(manifest) == runtime["model_manifest_sha256"], "Model manifest drift")
    hashes = {str(manifest): runtime["model_manifest_sha256"], **expected["source_sha256"]}
    hashes.update({str(model / name): value for name, value in native["model_metadata_sha256"].items()})
    manifest_data = read(manifest)
    files = manifest_data.get("files")
    if files is None:
        files = [dict(path=entry["rfilename"], bytes=entry["size"],
                      sha256=entry.get("lfs", {}).get("sha256"), git_blob=entry["blobId"])
                 for entry in manifest_data["siblings"]]
    stats, model_hashes = {}, {}
    for entry in files:
        path = (model / entry["path"]).resolve()
        require(path.is_relative_to(model.resolve()), "Model path escapes manifest root")
        require(path.stat().st_size == entry["bytes"], f"Model artifact size drift: {path}")
        digest = generation.file_hash(path)
        if entry["sha256"] is not None:
            require(digest == entry["sha256"], f"Model artifact drift: {path}")
        else:
            content = path.read_bytes()
            git_blob = sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            require(git_blob == entry["git_blob"], f"Model artifact drift: {path}")
        model_hashes[str(path)] = digest
        stats[str(path)] = [path.stat().st_size, path.stat().st_mtime_ns]
    executable = inherited_inputs["mlx_python"]
    require(executable == expected["executable"], "Changed MLX interpreter")
    hashes[executable] = generation.file_hash(executable)
    probe = "import json,sys;from importlib.metadata import version;print(json.dumps({'python':sys.version,'versions':{n:version(n) for n in " + repr(list(expected["versions"])) + "}}))"
    actual = json.loads(subprocess.check_output([executable, "-c", probe], text=True))
    require(actual == {key: expected[key] for key in ("python", "versions")}, "Runtime version drift")
    for path, value in hashes.items():
        require(generation.file_hash(path) == value, f"Runtime artifact drift: {path}")
    return dict(hashes=hashes, model_stats=stats, model_sha256=model_hashes, expected_runtime=expected, probe=probe,
                executable=executable)


def source_paths():
    return list((ROOT / "src/data-pipeline").rglob("*.py")) + [
        ROOT / "configs/sandbox/compose.yaml", ROOT / "requirements-execution.txt",
        ROOT / "docs/stage-05-repair-search-experiment.md"]


def prepare(output_dir):
    """Freeze parent ancestry, runtime pins, first requests and controls; no inference."""
    output = Path(output_dir).resolve()
    require(not output.exists(), "Preparation requires a new directory")
    parent, receipt = parent_evidence()
    history = {source: inherited(source) for source in SOURCES}
    runtime = {source: runtime_pins(source, history[source]["inputs"], parent["runtimes"][source])
               for source in SOURCES}
    recipe = policy.recipe(parent["recipes"]["optimization"])
    plans = []
    for source in SOURCES:
        for arm in ARMS:
            h = history[source]
            messages = policy.initial_histories(parent["worklists"][source], arm, recipe,
                requests=h["requests"], rows=h["rows"], attempts=h["attempts"])
            plans.append(dict(source=source, arm=arm, messages=messages,
                first_requests=loop.initial_requests(parent["worklists"][source], recipe, messages),
                ancestry=[] if arm == "C" else [dict(parent_attempt=a["attempt"], position=a["position"],
                    parent_outcome=str(PARENT / f"{source}-optimization/outcome.json"), round_offset=3)
                    for a in h["attempts"]]))
    files = source_paths()
    historical = dict(whole_parent={**receipt["effort"], "input_tokens": None},
                     by_source={source: history[source]["effort"] for source in SOURCES})
    historical["optimization"] = {}
    for key in ("committed_replies", "committed_tokens", "observed_tokens", "input_tokens"):
        values = [history[s]["effort"][key] for s in SOURCES]
        historical["optimization"][key] = sum(values) if all(type(v) is int for v in values) else None
    settings = dict(created_at=datetime.now(timezone.utc).isoformat(), parent=str(PARENT),
        parent_pins=PINS, publication=parent["publication"], task_ids=TASK_IDS,
        worklists=parent["worklists"], recipe=recipe, runtimes=parent["runtimes"],
        environment=runtime, historical=historical, plans=plans, budget=BUDGET,
        coordinator=dict(executable=sys.executable, python=sys.version, sha256=generation.file_hash(sys.executable)),
        source_sha256={str(path): generation.file_hash(path) for path in files},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    output.mkdir(parents=True, exist_ok=False)
    generation.save(output / "settings.json", settings)
    generation.save(output / "controls.json", diagnostics.controls(settings["worklists"]))
    generation.save(output / "parent-receipt.json", receipt)
    with tarfile.open(output / "source-snapshot.tar.gz", "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    generation.save(output / "prepared.json", index_files(output, list(output.iterdir())))
    verify_frozen(output, settings, versions=True)
    return settings


def verify_frozen(output, settings, *, versions=False):
    verify_index(output, read(output / "prepared.json"))
    require(settings["coordinator"] == dict(executable=sys.executable, python=sys.version,
            sha256=generation.file_hash(sys.executable)), "Coordinator interpreter changed")
    for path, value in settings["source_sha256"].items():
        require(generation.file_hash(path) == value, f"Source changed after preparation: {path}")
    for pins in settings["environment"].values():
        for path, value in pins["hashes"].items():
            require(generation.file_hash(path) == value, f"Runtime/model metadata changed: {path}")
        for path, expected in pins["model_stats"].items():
            stat = Path(path).stat()
            require([stat.st_size, stat.st_mtime_ns] == expected, f"Model file changed: {path}")
        if versions:
            actual = json.loads(subprocess.check_output([pins["executable"], "-c", pins["probe"]], text=True))
            require(actual == {k: pins["expected_runtime"][k] for k in ("python", "versions")}, "Runtime drift")


def validate_arm(settings, plan, outcome):
    """Replay request construction and extraction from saved evidence, without execution."""
    source, arm = plan["source"], plan["arm"]
    worklist, recipe = settings["worklists"][source], settings["recipe"]
    builder = policy.build_request(worklist, arm)
    requests, rows, passed, seen = {}, {}, set(), set()
    total_tokens = 0
    require([a["attempt"] for a in outcome["attempts"]] == list(range(len(outcome["attempts"]))),
            "Attempt numbering changed")
    for round_index, r in enumerate(outcome["rounds"]):
        require(r["round"] == round_index, "Round ordering changed")
        require(generation.file_hash(r["round_file"]) == r["round_file_sha256"], "Round request drift")
        saved = read(r["round_file"])["requests"]
        native = read(r["generation_outcome"])
        require(native["status"] == "success" and len(saved) == len(native["requests"]), "Incomplete generation")
        input_path = Path(r["generation_outcome"]).parent / "inputs.json"
        require(read(input_path)["requests"] == saved, "Native requests differ")
        expected_runtime = settings["environment"][source]["expected_runtime"]
        actual_runtime = read(input_path.with_name("runtime.json"))
        require(all(actual_runtime[k] == expected_runtime[k] for k in expected_runtime if k != "load_seconds"),
                "Native runtime differs from parent")
        attempts = [a for a in outcome["attempts"] if a["round"] == r["round"]]
        require([a["row"] for a in attempts] == list(range(len(saved))) and
                [a["attempt"] for a in attempts] == r["attempt_ids"], "Round coverage differs")
        for a in attempts:
            i, number = a["position"], a["attempt"]
            require(0 <= i < 8 and 0 <= a["round"] < 3 and i not in passed, "Invalid draw after success")
            require((i, a["round"]) not in seen, "Repeated position/round")
            seen.add((i, a["round"]))
            seed = 41 + 8 * a["round"] + i
            if a["round"] == 0:
                require(a["parent"] is None, "Unexpected initial parent")
                expected = plan["first_requests"][i]
            else:
                previous = outcome["attempts"][a["parent"]]
                require(previous["position"] == i and previous["round"] == a["round"] - 1, "Wrong repair parent")
                expected = builder(requests[a["parent"]], rows[a["parent"]], deepcopy(previous), recipe, seed)
            request, row = saved[a["row"]], native["requests"][a["row"]]
            require(request == expected and a["seed"] == seed and row["submitted"] is True,
                    "Request, seed or submission differs from plan")
            require(a["generation_status"] == row["status"] and
                    row["generation_tokens"] == len(row["tokens"]) <= 2048,
                    "Generation status or token cap mismatch")
            total_tokens += row["generation_tokens"]
            extraction, program = "length", None
            if row["status"] != "length":
                try:
                    program = generation.extract_program(row)
                    extraction = "success"
                except (ValueError, SyntaxError) as error:
                    extraction = "syntax" if isinstance(error, SyntaxError) else "format"
            require(a["extraction"] == extraction, "Extraction classification differs from reply")
            if program is not None:
                require(program == Path(a["program"]).read_bytes(), "Extracted program drift")
            if a.get("verification", {}).get("status") == "passed":
                passed.add(i)
            requests[number], rows[number] = request, row
    require(outcome["status"] == "completed" and outcome["usage_complete"] and
            outcome["committed_replies"] == outcome["known_submissions"] == len(seen) <= 24 and
            outcome["committed_tokens"] == 2048 * len(seen) and outcome["observed_tokens"] == total_tokens,
            "Incomplete arm or effort mismatch")
    require({i for i, r in seen if r == 0} == set(range(8)), "Missing initial coverage")
    require(all(i in passed or (i, 2) in seen for i in range(8)), "Unsuccessful pair stopped early")


def run(output_dir):
    """Run controls and all six arms exactly once; preserve stopped state on errors."""
    output = Path(output_dir).resolve()
    settings = read(output / "settings.json")
    require(not (output / "run.json").exists(), "Run already started; automatic resume is prohibited")
    verify_frozen(output, settings, versions=True)
    parent_evidence()
    state = dict(status="running", phase="controls", runs=[], started_at=datetime.now(timezone.utc).isoformat(),
                 prepared_sha256=generation.file_hash(output / "prepared.json"))
    started = monotonic()
    loop.atomic_save(output / "run.json", state)
    try:
        require(read(output / "controls.json") == diagnostics.controls(settings["worklists"]), "Control drift")
        state["controls"] = diagnostics.run_controls(settings["worklists"], output / "controls")
        require(state["controls"]["status"] == "completed", "Control gate failed")
        for plan in settings["plans"]:
            verify_frozen(output, settings)
            source, arm = plan["source"], plan["arm"]
            state.update(phase=f"{source}-{arm}")
            directory = output / f"{source}-{arm}"
            arm_started = monotonic()
            record = dict(source=source, arm=arm, directory=str(directory))
            state["runs"].append(record)
            loop.atomic_save(output / "run.json", state)
            builder = policy.build_request(settings["worklists"][source], arm)
            def checked_builder(*args):
                verify_frozen(output, settings)
                return builder(*args)
            runtime = settings["runtimes"][source]
            outcome = attacks.run(settings["worklists"][source], directory, source=source,
                model_path=runtime["model_path"], model_manifest=runtime["model_manifest"],
                mlx_python=settings["environment"][source]["executable"], recipe=settings["recipe"],
                completion_batch_size=8, prefill_batch_size=runtime["prefill_batch_size"],
                prefill_step_size=runtime["prefill_step_size"], reply_allowance=24, output_token_allowance=49152,
                initial_messages=plan["messages"], repair_builder=checked_builder,
                cohort_sha256=loop.study_tasks.COHORT_SHA256)
            record.update(outcome=outcome, effort=effort(outcome), elapsed_seconds=outcome["timing"]["overall"])
            require(outcome["status"] == "completed", f"Arm stopped: {outcome.get('error', outcome['status'])}")
            validate_arm(settings, plan, outcome)
            record["diagnostics"] = diagnostics.diagnose(settings["worklists"][source], source,
                outcome, inherited(source)["outcome"], directory / "diagnostics")
            record.update(search_elapsed_seconds=outcome["timing"]["overall"],
                          elapsed_seconds=monotonic() - arm_started)
            loop.atomic_save(output / "run.json", state)
        verify_frozen(output, settings)
        generation.save(output / "mechanism-review-template.json", reporting.review_template(state["runs"]))
        generation.save(output / "mechanism-review.json", reporting.review_template(state["runs"]))
        state.update(status="completed", phase="awaiting_mechanism_review")
    except BaseException as error:
        state.update(status="error" if isinstance(error, Exception) else "cancelled", error=repr(error))
        controls_path = output / "controls/outcome.json"
        if controls_path.exists():
            try:
                state["controls"] = read(controls_path)
            except (OSError, ValueError) as recovery_error:
                state["controls_recovery_error"] = repr(recovery_error)
        for record in state["runs"]:
            path = Path(record["directory"]) / "outcome.json"
            if path.exists():
                try:
                    outcome = read(path)
                    record.update(outcome=outcome, effort=effort(outcome))
                except (OSError, ValueError, KeyError, TypeError) as recovery_error:
                    record["recovery_error"] = repr(recovery_error)
        raise
    finally:
        state["elapsed_seconds"] = monotonic() - started
        loop.atomic_save(output / "run.json", state)
        paths = [p for p in output.rglob("*") if p.is_file() and p.name not in
                 ("artifacts.json", "receipt.json", "mechanism-review.json", "report.json", "audit.json")]
        generation.save(output / "artifacts.json", index_files(output, paths))
        generation.save(output / "receipt.json", dict(status=state["status"],
            artifacts_sha256=generation.file_hash(output / "artifacts.json"),
            settings_sha256=generation.file_hash(output / "settings.json"),
            run_sha256=generation.file_hash(output / "run.json")))
    audit(output)
    return report(output)


def audit(output_dir):
    output = Path(output_dir).resolve()
    receipt, settings, state = read(output / "receipt.json"), read(output / "settings.json"), read(output / "run.json")
    for name in ("artifacts", "settings", "run"):
        require(generation.file_hash(output / f"{name}.json") == receipt[f"{name}_sha256"], f"Changed {name}")
    verify_index(output, read(output / "artifacts.json"))
    require(state["status"] == "completed" and len(state["runs"]) == 6, "Run incomplete")
    require(state["controls"] == read(output / "controls/outcome.json") and
            state["controls"]["status"] == "completed" and len(state["controls"]["controls"]) == 3 and
            all(c["matched"] is True for c in state["controls"]["controls"]), "Control gate was not satisfied")
    require([(p["source"], p["arm"]) for p in settings["plans"]] ==
            [(source, arm) for source in SOURCES for arm in ARMS], "Search policy coverage changed")
    counts = dict(suite=0, witness=0)
    for plan, record in zip(settings["plans"], state["runs"]):
        require((plan["source"], plan["arm"]) == (record["source"], record["arm"]), "Arm order mismatch")
        directory = Path(record["directory"])
        require(record["outcome"] == read(directory / "outcome.json") and
                record["diagnostics"] == read(directory / "diagnostics/diagnostics.json"), "Saved arm records differ")
        validate_arm(settings, plan, record["outcome"])
        cases = record["diagnostics"]["case_executions"]
        counts["suite"] += cases["suite"]
        counts["witness"] += cases["acceptance_witness"] + cases["diagnostic_witness"]
    controls = state["controls"]["case_executions"]["total"]
    require(counts["suite"] <= BUDGET["suite_cases"] and counts["witness"] <= BUDGET["witness_cases"] and
            controls <= BUDGET["control_cases"] and sum(counts.values()) + controls <= BUDGET["total_cases"],
            "Experiment execution ceiling exceeded")
    generation.save(output / "audit.json", dict(passed=True, receipt_sha256=generation.file_hash(output / "receipt.json")))


def report(output_dir):
    output = Path(output_dir).resolve()
    audit(output)
    state = read(output / "run.json")
    result = reporting.build_report(read(output / "settings.json"), state["runs"], read(output / "mechanism-review.json"))
    pending = any(review["status"] == "pending" for review in result["reviews"])
    result.update(status="awaiting_mechanism_review" if pending else "completed",
                  controls=state["controls"], elapsed_seconds=state["elapsed_seconds"],
                  input_token_accounting="Sum of native per-reply prompt_tokens, including the final prompt token.",
                  reviews_sha256=generation.file_hash(output / "mechanism-review.json"),
                  receipt_sha256=generation.file_hash(output / "receipt.json"))
    generation.save(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit", "report"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = globals()[args.action](args.directory)
    print(json.dumps(dict(action=args.action, directory=str(args.directory.resolve()),
                          status=(result or {}).get("status", "ok"))))


if __name__ == "__main__":
    main()
