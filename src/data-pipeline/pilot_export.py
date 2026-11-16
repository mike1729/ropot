"""Freeze the accepted historical panel and export its existing three-row pilot.

This preserves source-specific parents and historical evidence, including accepted
exceptions. It performs no generation, execution or new qualification.
"""

import argparse
import json
from pathlib import Path

from . import source_hashes
from .artifacts import publication_directory, sha256, write_json
from .training_format import PHASES, SCHEMA_VERSION


REPOSITORY = Path(__file__).resolve().parents[2]
SOURCES = {"actor": "qwen", "synthetic": "gemma"}


def _reference(path, raw):
    return dict(path=Path(path).resolve().relative_to(REPOSITORY).as_posix(), sha256=sha256(raw))


def _read(reference):
    raw = (REPOSITORY / reference["path"]).read_bytes()
    if sha256(raw) != reference["sha256"]:
        raise ValueError(f"Historical input differs from its pin: {reference['path']}")
    return raw


def _write(root, name, value):
    write_json(root / name, value)
    return dict(path=name, sha256=sha256((root / name).read_bytes()))


def _historical_path(path):
    """Relocate the frozen corpus's original absolute data paths."""
    parts = Path(path).parts
    return REPOSITORY.joinpath(*parts[parts.index("data"):])


def freeze(campaign_root, cohort_root, output_dir, *, reuse_decision):
    """Pin the existing matched panels in cohort order; return panels.json SHA-256."""
    campaign = Path(campaign_root)
    paths = dict(frozen_manifest=campaign / "manifest.json", qualified_view=campaign / "report/qualified-view.json",
                 audit_results=campaign / "report/audit-results.json", cohort=Path(cohort_root) / "manifest.json",
                 accepted_status=campaign / "status-adjudicated-20261102/manifest.json", reuse_decision=Path(reuse_decision))
    raw = {name: path.read_bytes() for name, path in paths.items()}
    references = {name: _reference(path, raw[name]) for name, path in paths.items()}
    frozen, view, cohort = (json.loads(raw[name]) for name in ("frozen_manifest", "qualified_view", "cohort"))
    matched = view["matched_tasks_by_owner"]
    groups = {}
    for group, phase in PHASES.items():
        blocks = cohort["construction_phases"][phase]
        groups[group] = [task for task in blocks["initial"] + blocks["reserve"] if task in matched["construction"]]
    blocks = cohort["groups"]["development"]
    groups["development"] = [task for task in blocks["initial"] + blocks["reserve"] if task in matched["development"]]
    selected = {task for tasks in groups.values() for task in tasks}
    generator_metadata = {}
    for name, source in SOURCES.items():
        models, declared = {}, set()
        for pair in frozen["pairs"]:
            if pair["selection"]["source"] != source or pair["binding"]["id"] not in selected:
                continue
            recipe = frozen["recipes"][pair["selection"]["first_agreed_attempt"]["root"]]
            contract_path = _historical_path(pair["contract"])
            contract = json.loads(_read(dict(path=str(contract_path), sha256=pair["contract_sha256"])))
            if "benign_model_manifest_sha256" in contract["runtime"]:
                declared.add(contract["runtime"]["benign_model_manifest_sha256"])
            for runtime in (recipe["runtime"], contract["runtime"]):
                path = _historical_path(runtime["model_manifest"])
                reference = _reference(path, path.read_bytes())
                models[reference["sha256"]] = reference
        if len(models) != 1:
            raise ValueError(f"Historical source mixes generator checkpoints: {source}")
        references[name + "_generator"] = next(iter(models.values()))
        generator_metadata[name] = dict(recorded_benign_manifest_sha256s=sorted(declared))
    record = dict(schema_version="historical-pilot-panels-v1", advancement_rule="frozen-corpus-v1",
                  groups=groups, sources=references, generator_metadata=generator_metadata,
                  reuse_decision=json.loads(raw["reuse_decision"]), source_sha256=source_hashes())
    with publication_directory(output_dir) as staging:
        reference = _write(staging, "panels.json", record)
    return reference["sha256"]


def export(panels_root, output_dir, *, expected_panels_sha256):
    """Export frozen bytes and source evidence; return the exact manifest SHA-256."""
    panels_raw = _read(dict(path=str(Path(panels_root) / "panels.json"), sha256=expected_panels_sha256))
    panels = json.loads(panels_raw)
    frozen, view, audit = (json.loads(_read(panels["sources"][name]))
                           for name in ("frozen_manifest", "qualified_view", "audit_results"))
    records = {(row["binding"]["id"], sha256(row["program"].encode("utf-8"))): row for row in frozen["records"]}
    retained = {pair["id"]: pair for pair in view["pairs"]}
    chosen = {f"{source}:{task}" for tasks in panels["groups"].values() for task in tasks for source in SOURCES.values()}
    evidence = dict(pairs=[pair for pair in frozen["pairs"] if pair["id"] in chosen],
                    audit_results=[row for row in audit["case_results"] if row["pair_id"] in chosen],
                    accepted_pairs=[pair for pair in view["pairs"] if pair["id"] in chosen],
                    sources=panels["sources"])
    generators = {name: panels["sources"][name + "_generator"]["sha256"] for name in SOURCES}
    with publication_directory(output_dir) as staging:
        (staging / "tasks").mkdir()
        (staging / "allocation.json").write_bytes(panels_raw)
        support = _write(staging, "selected-source-evidence.json", evidence)
        recipe = _write(staging, "generation-recipe.json", dict(kind="historical-separate-parents-v1",
                        source=panels["sources"]["frozen_manifest"], program_policy="verbatim"))
        manifest = dict(schema_version=SCHEMA_VERSION, upstream_version="frozen92-adjudicated-20261102",
                        dataset_kind="historical-separate-parents-v1", allocation=dict(path="allocation.json", sha256=expected_panels_sha256),
                        generation_recipe=recipe, generators=generators, generator_metadata=panels["generator_metadata"],
                        reuse_decision=panels["reuse_decision"], source_sha256=source_hashes(), groups={}, selection_failures=[])
        for group, task_ids in panels["groups"].items():
            manifest["groups"][group] = []
            for task_id in task_ids:
                programs, statements = {}, set()
                for generator, source in SOURCES.items():
                    pair = retained[f"{source}:{task_id}"]
                    for kind in ("benign", "attack"):
                        row = records[(task_id, pair[kind + "_sha256"])]
                        statements.add(row["statement"])
                        provenance = dict(generator_sha256=generators[generator], source_pair_id=pair["id"],
                                          accepted_provenance_exception=pair["accepted_provenance_exception"])
                        provenance.update(dict(ancestor_artifact_sha256s=[]) if kind == "benign"
                                          else dict(parent_input_sha256=pair["benign_sha256"]))
                        receipt = dict(result="retained_historical_" + kind,
                                       statement_sha256=sha256(row["statement"].encode("utf-8")),
                                       program_sha256=pair[kind + "_sha256"], provenance=provenance,
                                       evidence=[dict(support, pair_id=pair["id"], role=kind)])
                        programs[generator + "_" + kind] = dict(program=row["program"], receipt=receipt)
                if len(statements) != 1:
                    raise ValueError(f"Historical statements differ for task {task_id}")
                task = dict(task_id=task_id, statement=statements.pop(),
                            programs=dict(benign=programs["actor_benign"], actor_attack=programs["actor_attack"],
                                          synthetic_attack=programs["synthetic_attack"]),
                            synthetic_parent=programs["synthetic_benign"])
                entry = _write(staging, f"tasks/{task_id}.json", task)
                manifest["groups"][group].append(dict(task_id=task_id, **entry))
        reference = _write(staging, "manifest.json", manifest)
    return reference["sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("freeze")
    command.add_argument("--campaign-root", type=Path, required=True)
    command.add_argument("--cohort-root", type=Path, required=True)
    command.add_argument("--reuse-decision", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = commands.add_parser("export")
    command.add_argument("--panels-root", type=Path, required=True)
    command.add_argument("--expected-panels-sha256", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.campaign_root, args.cohort_root, args.output_dir, reuse_decision=args.reuse_decision) if args.command == "freeze" else
          export(args.panels_root, args.output_dir, expected_panels_sha256=args.expected_panels_sha256))


if __name__ == "__main__":
    main()
