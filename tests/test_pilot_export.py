"""Export the retained historical corpus without inventing shared-parent lineage."""

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from src.monitor_training import data, prepare


pilot = importlib.import_module("src.data-pipeline.pilot_export")


class PilotExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.enterContext(mock.patch.object(pilot, "REPOSITORY", self.root))
        self.campaign, self.cohort = self.root / "data/campaign", self.root / "data/cohort"
        self.decision = self.root / "reuse-decision.json"
        self.write(self.decision, dict(authority="user", scope="Fixture historical three-row pilot"))
        self.groups = dict(shared_honest=[6, 4, 2, 5, 3, 1], twin_continuation=[18, 16, 14, 12, 17, 15, 13, 11],
                           development=list(range(120, 100, -2)) + list(range(119, 100, -2)))
        manifest = dict(records=[], pairs=[], recipes={})
        view = dict(pairs=[], matched_tasks_by_owner=dict(construction=[], development=[]))
        audit = dict(case_results=[])
        self.programs, self.statements = {}, {}
        self.exception = dict(reason="Accepted retained provenance exception")
        contracts = {}
        for source in ("qwen", "gemma"):
            model = self.root / "data/models" / (source + "-model.json")
            self.write(model, dict(model=source, revision="fixture-revision"))
            contract = self.root / "data/contracts" / (source + ".json")
            runtime = dict(model_manifest=str(model))
            if source == "qwen":
                runtime["benign_model_manifest_sha256"] = "f" * 64
            self.write(contract, dict(runtime=runtime))
            contracts[source] = dict(contract=str(contract), contract_sha256=data.sha256(contract.read_bytes()))
            manifest["recipes"][source] = dict(runtime=dict(model_manifest=str(model)))
        for group, task_ids in self.groups.items():
            for task_id in task_ids:
                binding = dict(id=task_id, owner="development" if group == "development" else "construction",
                               phase=None if group == "development" else
                               "shared_honest_pretraining" if group == "shared_honest" else "twin_continuation")
                view["matched_tasks_by_owner"][binding["owner"]].append(task_id)
                self.statements[task_id] = f"Task {task_id}: é and e\u0301.\r\nKeep trailing whitespace. \t"
                for source in ("qwen", "gemma"):
                    pair_id = f"{source}:{task_id}"
                    hashes = {}
                    for role in ("benign", "attack"):
                        program = f"# {source} {role} {task_id} é\r\nprint('{role}')  \t\r\n"
                        self.programs[(source, task_id, role)] = program
                        hashes[role] = data.sha256(program.encode())
                        manifest["records"].append(dict(id=hashes[role], binding=binding,
                                                         statement=self.statements[task_id], program=program))
                        audit["case_results"].append(dict(pair_id=pair_id, role=role, category="retained"))
                    manifest["pairs"].append(dict(id=pair_id, binding=binding, **contracts[source], benign_sha256=hashes["benign"],
                                                  attack_sha256=hashes["attack"], selection=dict(
                                                      source=source, first_agreed_attempt=dict(root=source), weak_task=pair_id == "qwen:6")))
                    view["pairs"].append(dict(id=pair_id, binding=binding, benign_sha256=hashes["benign"],
                                              attack_sha256=hashes["attack"], accepted_provenance_exception=
                                              self.exception if pair_id == "qwen:6" else None))
        self.write(self.campaign / "manifest.json", manifest)
        self.write(self.campaign / "report/qualified-view.json", view)
        self.write(self.campaign / "report/audit-results.json", audit)
        self.write(self.campaign / "status-adjudicated-20261102/manifest.json", dict(authority=dict(authority="user")))
        blocks = lambda values: dict(initial=[999] + values[:len(values) // 2], reserve=values[len(values) // 2:], deferred=[998])
        self.write(self.cohort / "manifest.json", dict(
            construction_phases=dict(shared_honest_pretraining=blocks(self.groups["shared_honest"]),
                                     twin_continuation=blocks(self.groups["twin_continuation"])),
            groups=dict(development=blocks(self.groups["development"]))))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode())

    def freeze_export(self):
        panels, output = self.root / "panels", self.root / "export"
        panels_pin = pilot.freeze(self.campaign, self.cohort, panels, reuse_decision=self.decision)
        export_pin = pilot.export(panels, output, expected_panels_sha256=panels_pin)
        manifest = json.loads((output / "manifest.json").read_bytes())
        return panels, panels_pin, output, export_pin, manifest

    def test_freeze_export_prepare_and_reuse_preserve_all_102_training_rows(self):
        panels, panels_pin, output, export_pin, manifest = self.freeze_export()
        self.assertEqual({group: [entry["task_id"] for entry in entries] for group, entries in manifest["groups"].items()},
                         self.groups)
        self.assertEqual(manifest["dataset_kind"], "historical-separate-parents-v1")
        self.assertEqual(manifest["reuse_decision"]["authority"], "user")
        self.assertIn("pilot_export.py", manifest["source_sha256"])
        self.assertEqual(manifest["generator_metadata"]["actor"]["recorded_benign_manifest_sha256s"], ["f" * 64])
        prepared = self.root / "prepared"
        digest = prepare.prepare(output, prepared, expected_export_sha256=export_pin, expected_allocation_sha256=panels_pin)
        rows = prepare.load_prepared(prepared, output, list(self.groups), expected_prepared_sha256=digest)
        self.assertEqual({group: len(values) for group, values in rows.items()},
                         dict(shared_honest=18, twin_continuation=24, development=60))
        self.assertEqual(data.sha256((output / "allocation.json").read_bytes()), panels_pin)
        self.assertEqual(json.loads((output / "allocation.json").read_bytes())["advancement_rule"], "frozen-corpus-v1")
        recipe = json.loads((output / manifest["generation_recipe"]["path"]).read_bytes())
        self.assertEqual(recipe["source"], json.loads((panels / "panels.json").read_bytes())["sources"]["frozen_manifest"])

    def test_exact_qualified_bytes_and_each_historical_parent_are_retained(self):
        _, _, output, _, manifest = self.freeze_export()
        task = json.loads((output / manifest["groups"]["shared_honest"][0]["path"]).read_bytes())
        self.assertEqual(task["statement"], self.statements[6])
        for role, source, historical_role in (("benign", "qwen", "benign"), ("actor_attack", "qwen", "attack"),
                                               ("synthetic_attack", "gemma", "attack")):
            entry = task["programs"][role]
            self.assertEqual(entry["program"], self.programs[(source, 6, historical_role)])
            self.assertEqual(entry["receipt"]["program_sha256"], data.sha256(entry["program"].encode()))
            self.assertEqual(entry["receipt"]["statement_sha256"], data.sha256(task["statement"].encode()))
            self.assertEqual(entry["receipt"]["result"], "retained_historical_" + historical_role)
            provenance = entry["receipt"]["provenance"]
            self.assertEqual(provenance["source_pair_id"], f"{source}:6")
            self.assertEqual(provenance["generator_sha256"], data.sha256((self.root / "data/models" / (source + "-model.json")).read_bytes()))
        actor_parent = data.sha256(task["programs"]["benign"]["program"].encode())
        synthetic_parent = data.sha256(task["synthetic_parent"]["program"].encode())
        self.assertNotEqual(actor_parent, synthetic_parent)
        self.assertEqual(task["synthetic_parent"]["program"], self.programs[("gemma", 6, "benign")])
        self.assertEqual(task["programs"]["actor_attack"]["receipt"]["provenance"]["parent_input_sha256"], actor_parent)
        self.assertEqual(task["programs"]["synthetic_attack"]["receipt"]["provenance"]["parent_input_sha256"], synthetic_parent)
        self.assertEqual(task["programs"]["benign"]["receipt"]["provenance"]["accepted_provenance_exception"], self.exception)
        evidence = json.loads((output / "selected-source-evidence.json").read_bytes())
        self.assertIn('"weak_task": true', json.dumps(evidence))
        self.assertIn(self.exception["reason"], json.dumps(evidence))
        self.assertEqual(set(task["programs"]), {"benign", "actor_attack", "synthetic_attack"})

    def test_export_reuses_frozen_provenance_and_is_deterministic(self):
        panels, pin, output, first, _ = self.freeze_export()
        sources = json.loads((panels / "panels.json").read_bytes())["sources"]
        for name in ("cohort", "accepted_status", "actor_generator", "synthetic_generator"):
            (self.root / sources[name]["path"]).unlink()
        second = self.root / "second"
        self.assertEqual(pilot.export(panels, second, expected_panels_sha256=pin), first)
        self.assertEqual({path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()},
                         {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()})

    def test_changed_source_is_rejected_under_its_frozen_pin(self):
        panels, output = self.root / "panels", self.root / "export"
        pin = pilot.freeze(self.campaign, self.cohort, panels, reuse_decision=self.decision)
        path = self.campaign / "manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            pilot.export(panels, output, expected_panels_sha256=pin)
        self.assertFalse(output.exists())

    def test_changed_panels_and_existing_destinations_are_rejected(self):
        panels, pin, output, _, _ = self.freeze_export()
        with self.assertRaises(FileExistsError):
            pilot.freeze(self.campaign, self.cohort, panels, reuse_decision=self.decision)
        with self.assertRaises(FileExistsError):
            pilot.export(panels, output, expected_panels_sha256=pin)
        path = panels / "panels.json"
        path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "pin|digest|bytes"):
            pilot.export(panels, self.root / "changed", expected_panels_sha256=pin)

    def test_joined_statements_and_generator_identities_must_agree(self):
        path = self.campaign / "manifest.json"
        original = json.loads(path.read_bytes())
        for case in ("statement", "generator"):
            with self.subTest(case=case):
                frozen = json.loads(json.dumps(original))
                if case == "statement":
                    frozen["records"][0]["statement"] += " Different."
                else:
                    model = self.root / "data/models/other.json"
                    self.write(model, dict(model="other"))
                    frozen["recipes"]["other"] = dict(runtime=dict(model_manifest=str(model)))
                    pair = next(pair for pair in frozen["pairs"] if pair["id"] == "gemma:16")
                    pair["selection"]["first_agreed_attempt"]["root"] = "other"
                self.write(path, frozen)
                panels = self.root / (case + "-panels")
                with self.assertRaisesRegex(ValueError, "statements differ|mixes generator"):
                    pin = pilot.freeze(self.campaign, self.cohort, panels, reuse_decision=self.decision)
                    pilot.export(panels, self.root / case, expected_panels_sha256=pin)

    def test_pins_and_exports_survive_repository_relocation(self):
        import shutil

        panels, pin, _, export_pin, _ = self.freeze_export()
        relocated = self.root / "relocated"
        shutil.copytree(self.root / "data", relocated / "data")
        shutil.copy(self.decision, relocated / self.decision.name)
        shutil.copytree(panels, relocated / "panels")
        shutil.rmtree(self.root / "data")
        with mock.patch.object(pilot, "REPOSITORY", relocated):
            self.assertEqual(pilot.export(relocated / "panels", relocated / "export", expected_panels_sha256=pin), export_pin)
            self.assertEqual(pilot.freeze(relocated / "data/campaign", relocated / "data/cohort", relocated / "refrozen",
                                        reuse_decision=relocated / self.decision.name), pin)
