"""Publish first verified passes from numbered study runs; saved evidence only."""
import json
from pathlib import Path
import re

from ..artifacts import publication_directory, sha256
from . import experiment_summary, study_tasks


RULE = "First verified pass per task/source, ordered by experiment number then attempt number."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def labels(cohort):
    result = {}
    for position, identifier in enumerate(cohort["groups"]["development"]["initial"][:50]):
        result[identifier] = "development_smoke" if position < 8 else "development"
    for phase in cohort["construction_phases"].values():
        for position, identifier in enumerate(phase["initial"][:32]):
            result[identifier] = "construction_exploration" if position < 16 else "construction_confirmation"
    return result


def publish(root, version):
    """Publish root.parent/03_benigns/version atomically; refuse an existing version.

    Read each coordinator record once through summarize, validate every saved
    pass, then copy selected bytes. No generation, execution or log replay occurs.
    Recipe digests use sorted compact JSON. Inputs must remain stable during
    this single-publisher operation.
    """
    require(isinstance(version, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version),
            "Version must be a simple directory name")
    root = Path(root)
    target = root.parent / "03_benigns" / version
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    cohort = study_tasks.load_cohort()
    groups = labels(cohort)
    summary = experiment_summary.summarize(root)
    require(bool(summary["experiments"]), "No experiments to publish")
    experiments = {e["experiment"]: e for e in summary["experiments"]}
    seen = {name: set() for name in experiments}
    for experiment in experiments.values():
        require(experiment["terminal"] and not experiment["problems"]
                and experiment["fixtures"] is False and experiment["cohort_sha256"] == cohort["sha256"]
                and isinstance(experiment["recipe"], dict)
                and isinstance(experiment["model_manifest_sha256"], str)
                and study_tasks.DIGEST.fullmatch(experiment["model_manifest_sha256"]),
                "Require terminal, readable pinned study experiments")
    choices = {}
    # Collect publication candidates for each pinned task/source cell.
    for cell in summary["cells"]:
        identifier, source = cell["id"], cell["source"]
        binding = cohort["tasks_by_id"].get(identifier)
        require(identifier in groups and binding is not None and cell["split"] == binding["split"]
                and cell["sha256"] == binding["sha256"], "Task outside publication groups or pinned binding")
        # Resolve each saved history to its source experiment and cohort binding.
        for history in cell["experiments"]:
            name = history["experiment"]
            experiment = experiments[name]
            require(history["binding"] == binding and experiment["source"] == source,
                    "Worklist binding differs from pinned cohort")
            # Validate passing programs and retain the earliest successful attempt.
            for attempt in sorted(history["attempts"], key=lambda a: a["attempt"]):
                number = attempt["attempt"]
                require(number >= 0 and number not in seen[name], "Invalid or duplicate attempt number")
                seen[name].add(number)
                verdict = attempt.get("verification") or {}
                passed = attempt["state"] == "passed"
                require(passed == (verdict.get("status") == "passed"), "Contradictory saved pass state/verdict")
                if not passed:
                    continue
                program = Path(attempt["program"]).read_bytes()
                candidate = study_tasks.candidate_identity(binding["split"], identifier, binding["sha256"], source, program)
                key = verdict["key"]
                require(attempt["candidate"] == candidate and key["task_sha256"] == binding["sha256"]
                        and key["program_sha256"] == candidate["program_sha256"], "Candidate/verdict digest or identity mismatch")
                require(study_tasks.DIGEST.fullmatch(key["contract_sha256"])
                        and isinstance(verdict["log"], str) and verdict["log"]
                        and type(verdict["sample_id"]) in (str, int), "Missing verification provenance")
                recipe = experiment["recipe"]
                record = dict(binding=binding, label=groups[identifier], source=source, experiment=name,
                    attempt=number, repair_depth=attempt["round"], parent=attempt["parent"], recipe=recipe,
                    recipe_sha256=sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()),
                    model_manifest_sha256=experiment["model_manifest_sha256"], candidate=candidate,
                    verification={k: verdict[k] for k in ("key", "log", "sample_id")},
                    program=f"programs/{source}-{binding['split']}-{identifier}.py",
                    program_sha256=candidate["program_sha256"])
                choice = choices.get((identifier, source))
                if choice is None or (name, number) < (choice[0]["experiment"], choice[0]["attempt"]):
                    choices[identifier, source] = record, program
    require(all(len(seen[name]) == e["attempt_count"] for name, e in experiments.items()),
            "Unassociated attempts outside saved worklists")
    coverage = {}
    for label in ("development_smoke", "development", "construction_exploration", "construction_confirmation"):
        identifiers = [i for i, group in groups.items() if group == label]
        coverage[label] = dict(denominator=len(identifiers),
            selected={s: sum((i, s) in choices for i in identifiers) for s in study_tasks.SOURCES},
            joint_ids=[i for i in identifiers if all((i, s) in choices for s in study_tasks.SOURCES)],
            missing_cells=[dict(split=cohort["tasks_by_id"][i]["split"], id=i, source=s)
                           for i in identifiers for s in study_tasks.SOURCES if (i, s) not in choices])
    manifest = dict(rule=RULE, cohort_sha256=cohort["sha256"],
        experiments=[{k: e[k] for k in ("experiment", "inputs_sha256", "outcome_sha256")} for e in experiments.values()],
        selections=[choices[key][0] for key in sorted(choices)])
    report = dict(groups=coverage, summary={k: summary[k] for k in ("experiments", "totals")})
    with publication_directory(target) as temporary:
        (temporary / "programs").mkdir()
        for record, program in choices.values():
            (temporary / record["program"]).write_bytes(program)
        for name, value in (("manifest.json", manifest), ("report.json", report)):
            (temporary / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return target / "manifest.json"
