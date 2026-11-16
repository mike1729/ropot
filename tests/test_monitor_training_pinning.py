"""Candidate recipe freezing and rejection of changed numerical/runtime pins."""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.monitor_training import pinning


class MonitorTrainingPinningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "snapshot"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"fixture"}\n')
        (self.model / "tokenizer.json").write_bytes(b'{"fixture":"tokenizer"}\n')
        (self.model / "model.safetensors").write_bytes(b"fixture weights")
        self.settings = dict(checkpoint=dict(repo_id="fixture/upstream", revision="a" * 40),
                             max_total_tokens=8192, max_new_tokens=8, terminator_id=2,
                             budgets=dict(max_margin_tokens=1000000,
                                          max_wall_seconds=3600, max_cost_usd=5, gpu_hourly_usd=2))
        self.settings_path = self.root / "settings.json"
        self.settings_path.write_text(json.dumps(self.settings))
        self.sources = {name: "b" * 64 for name in pinning.SOURCES}
        self.versions = {name: f"1.0.{index}" for index, name in enumerate(pinning.PACKAGES)}
        self.versions.update({name: None for name in pinning.OPTIONAL_PACKAGES})

    def package_version(self, name):
        value = self.versions[name]
        if value is None:
            raise pinning.metadata.PackageNotFoundError(name)
        return value

    @contextmanager
    def environment(self, *, cuda=True):
        tokenizer = mock.Mock()
        auto_tokenizer = mock.Mock()
        auto_tokenizer.from_pretrained.return_value = tokenizer
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
                                                     get_device_name=lambda _: "NVIDIA fixture"),
                                version=SimpleNamespace(cuda="12.8"))
        with mock.patch.dict(sys.modules, torch=torch,
                             transformers=SimpleNamespace(AutoTokenizer=auto_tokenizer)), \
                mock.patch.object(pinning.metadata, "version", side_effect=self.package_version), \
                mock.patch.object(pinning.platform, "python_version", return_value="3.12.13"), \
                mock.patch.object(pinning, "source_hashes", return_value=self.sources), \
                mock.patch.object(pinning, "Renderer"):
            yield auto_tokenizer

    def freeze_fixture(self, name="frozen"):
        output = self.root / name
        with self.environment():
            digest = pinning.freeze(self.model, self.settings_path, output)
        record = json.loads((output / "contract.json").read_bytes())
        return output, digest, record

    def test_freeze_deterministically_records_snapshot_source_and_environment(self):
        output = self.root / "first"
        with self.environment() as tokenizer:
            digest = pinning.freeze(self.model, self.settings_path, output)
            tokenizer.from_pretrained.assert_called_once_with(
                self.model, local_files_only=True, trust_remote_code=False)
        second, second_digest, record = self.freeze_fixture("second")
        raw = (output / "contract.json").read_bytes()
        self.assertEqual(raw, (second / "contract.json").read_bytes())
        self.assertEqual(digest, second_digest)
        self.assertEqual(digest, pinning.artifacts.sha256(raw))
        self.assertEqual(record["source_sha256"], self.sources)
        self.assertEqual(record["packages"], self.versions)
        self.assertEqual(record["python_version"], "3.12.13")
        self.assertEqual(record["cuda_version"], "12.8")
        self.assertEqual(record["gpu_name"], "NVIDIA fixture")
        self.assertEqual({name: record[name] for name in
                          ("backend", "numerical_base", "attention", "seed", "requested_temperature")},
                         dict(backend="transformers", numerical_base="bnb-int8-bf16", attention="sdpa",
                              seed=17, requested_temperature=0))
        self.assertEqual(record["checkpoint"]["revision"], "a" * 40)
        self.assertEqual(record["checkpoint"]["files"], {
            name: pinning.artifacts.sha256((self.model / name).read_bytes())
            for name in ("config.json", "tokenizer.json", "model.safetensors")})
        requirements = (output / "requirements-training.txt").read_text()
        self.assertEqual(requirements, "".join(f"{name}=={version}\n" for name, version in self.versions.items() if version is not None))
        self.assertNotIn("==None", requirements)
        self.assertNotIn(str(self.model), raw.decode())
        with self.environment():
            self.assertEqual(pinning.load_contract(output / "contract.json", digest, self.model), record)

    def test_nested_recipe_and_unrelated_metadata_do_not_change_snapshot_pins(self):
        self.settings_path.write_text(json.dumps(self.settings))
        with self.environment():
            output = self.model / "recipe"
            digest = pinning.freeze(self.model, self.settings_path, output)
            record = pinning.load_contract(output / "contract.json", digest, self.model)
            (self.model / "README.md").write_text("updated model card")
            cache = self.model / ".cache"
            cache.mkdir()
            (cache / "download.json").write_text("new cache metadata")
            (self.model / "config.json.swp").write_bytes(b"editor state")
            self.assertEqual(pinning.load_contract(output / "contract.json", digest, self.model), record)

    def test_changed_or_added_runtime_files_and_indexed_shards_invalidate_the_snapshot(self):
        (self.model / "model.safetensors").unlink()
        (self.model / "model.safetensors.index.json").write_text(json.dumps(
            dict(weight_map={"layer.a": "weights-a.safetensors", "layer.b": "weights-b.safetensors"})))
        for name in ("weights-a.safetensors", "weights-b.safetensors", "additional_chat_templates/default.jinja"):
            path = self.model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture bytes")
        output, digest, record = self.freeze_fixture()
        self.assertIn("weights-a.safetensors", record["checkpoint"]["files"])
        self.assertIn("additional_chat_templates/default.jinja", record["checkpoint"]["files"])
        for name in ("tokenizer.json", "weights-a.safetensors", "additional_chat_templates/default.jinja",
                     "additional_chat_templates/extra.jinja", "added_tokens.json"):
            with self.subTest(name=name):
                path = self.model / name
                before = path.read_bytes() if path.exists() else None
                path.write_bytes(b"changed runtime input")
                with self.environment(), self.assertRaisesRegex(ValueError, "Snapshot"):
                    pinning.load_contract(output / "contract.json", digest, self.model)
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(before)

    def test_optional_kernel_install_upgrade_and_removal_require_a_new_recipe(self):
        for package in pinning.OPTIONAL_PACKAGES:
            for index, (before, after) in enumerate(((None, "1.0"), ("1.0", "2.0"), ("1.0", None))):
                with self.subTest(package=package, before=before, after=after):
                    self.versions.update({name: None for name in pinning.OPTIONAL_PACKAGES})
                    self.versions[package] = before
                    output, digest, record = self.freeze_fixture(f"{package}-{index}")
                    self.assertEqual(record["packages"][package], before)
                    requirements = (output / "requirements-training.txt").read_text()
                    self.assertEqual(f"{package}==1.0\n" in requirements, before is not None)
                    self.versions[package] = after
                    with self.environment(), self.assertRaisesRegex(ValueError, "Packages"):
                        pinning.load_contract(output / "contract.json", digest, self.model)

    def test_freeze_refuses_existing_destination_before_loading_runtime(self):
        output, _, _ = self.freeze_fixture()
        before = (output / "contract.json").read_bytes()
        with self.environment() as tokenizer, self.assertRaises(FileExistsError):
            pinning.freeze(self.model, self.settings_path, output)
        tokenizer.from_pretrained.assert_not_called()
        self.assertEqual((output / "contract.json").read_bytes(), before)

    def test_freeze_requires_the_intended_cuda_host(self):
        with self.environment(cuda=False) as tokenizer, self.assertRaisesRegex(RuntimeError, "NVIDIA"):
            pinning.freeze(self.model, self.settings_path, self.root / "no-cuda")
        tokenizer.from_pretrained.assert_not_called()
        self.assertFalse((self.root / "no-cuda").exists())

    def test_wrong_contract_pin_fails_before_environment_or_snapshot_checks(self):
        output, _, _ = self.freeze_fixture()
        with mock.patch.object(pinning, "source_hashes") as sources, \
                mock.patch.object(pinning, "snapshot_hashes") as snapshots, \
                self.assertRaisesRegex(ValueError, "reviewed pin"):
            pinning.load_contract(output / "contract.json", "0" * 64, self.model)
        sources.assert_not_called()
        snapshots.assert_not_called()

    def test_changed_source_packages_python_or_snapshot_prevent_contract_reuse(self):
        output, digest, _ = self.freeze_fixture()
        for change, message in (("source", "source"), ("packages", "Packages"),
                                ("python", "Python"), ("snapshot", "Snapshot")):
            with self.subTest(change=change), self.environment():
                if change == "source":
                    context = mock.patch.object(pinning, "source_hashes", return_value=dict(self.sources, **{"runtime.py": "c" * 64}))
                elif change == "packages":
                    context = mock.patch.object(pinning.metadata, "version", side_effect=dict(self.versions, torch="changed").__getitem__)
                elif change == "python":
                    context = mock.patch.object(pinning.platform, "python_version", return_value="3.13.1")
                else:
                    context = mock.patch.object(pinning, "snapshot_hashes", return_value={"config.json": "c" * 64})
                with context, self.assertRaisesRegex(ValueError, message):
                    pinning.load_contract(output / "contract.json", digest, self.model)

    def test_prequantized_snapshot_is_rejected_before_tokenizer_loading_or_publication(self):
        (self.model / "config.json").write_text(json.dumps(dict(quantization_config=dict(bits=8))))
        output = self.root / "prequantized"
        with self.environment() as tokenizer, self.assertRaisesRegex(ValueError, "original upstream weights"):
            pinning.freeze(self.model, self.settings_path, output)
        tokenizer.from_pretrained.assert_not_called()
        self.assertFalse(output.exists())

    def test_freeze_audits_the_selected_tokenizer_before_publication(self):
        output = self.root / "invalid-tokenizer"
        with self.environment() as tokenizer, \
                mock.patch.object(pinning, "Renderer", side_effect=ValueError("Unsupported label/terminator")) as renderer, \
                self.assertRaisesRegex(ValueError, "label/terminator"):
            pinning.freeze(self.model, self.settings_path, output)
        renderer.assert_called_once_with(tokenizer.from_pretrained.return_value,
                                         max_total_tokens=8192, max_new_tokens=8, terminator_id=2)
        self.assertFalse(output.exists())

    def test_unsupported_numerical_settings_and_mutable_checkpoint_are_rejected(self):
        for field, value in (("max_total_tokens", 0), ("max_new_tokens", True),
                             ("terminator_id", -1), ("terminator_id", True), ("max_new_tokens", 8192)):
            with self.subTest(field=field):
                settings = dict(self.settings, **{field: value})
                with self.assertRaises(ValueError):
                    pinning.validate_settings(settings)
        settings = deepcopy(self.settings)
        settings["checkpoint"]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable upstream checkpoint"):
            pinning.validate_settings(settings)

    def test_unknown_or_fixed_settings_are_rejected_before_tokenizer_loading(self):
        for field, value in (("backend", "other"), ("numerical_base", "bf16"),
                             ("attention", "flash_attention_2"), ("seed", 18), ("requested_temperature", 1),
                             ("max_new_token", 8), ("packages", {}), ("gpu_name", "other"),
                             ("source_sha256", {}), ("schema_version", "other")):
            with self.subTest(field=field):
                settings = dict(self.settings, **{field: value})
                self.settings_path.write_text(json.dumps(settings))
                with self.environment() as tokenizer, self.assertRaisesRegex(ValueError, "Settings must contain exactly"):
                    pinning.freeze(self.model, self.settings_path, self.root / "fixed-values")
                tokenizer.from_pretrained.assert_not_called()

    def test_missing_settings_are_rejected_before_tokenizer_loading(self):
        for field in self.settings:
            with self.subTest(field=field):
                settings = dict(self.settings)
                del settings[field]
                self.settings_path.write_text(json.dumps(settings))
                with self.environment() as tokenizer, self.assertRaisesRegex(ValueError, "Settings must contain exactly"):
                    pinning.freeze(self.model, self.settings_path, self.root / "missing-values")
                tokenizer.from_pretrained.assert_not_called()

    def test_all_resource_budgets_must_be_numeric_positive_and_finite(self):
        pinning.validate_settings(self.settings)
        for name in self.settings["budgets"]:
            for invalid in (0, -1, True, "60", float("nan"), float("inf")):
                with self.subTest(name=name, invalid=invalid):
                    settings = deepcopy(self.settings)
                    settings["budgets"][name] = invalid
                    with self.assertRaisesRegex(ValueError, "budget"):
                        pinning.validate_settings(settings)

    def test_obsolete_count_limits_are_not_silently_ignored(self):
        settings = deepcopy(self.settings)
        settings["budgets"]["max_replies"] = 59
        with self.assertRaisesRegex(ValueError, "Budgets must contain only"):
            pinning.validate_settings(settings)

    def test_freeze_audits_native_label_boundaries_before_publication(self):
        output = self.root / "bad-native-boundary"
        with self.environment(), mock.patch.object(pinning, "Renderer") as renderer:
            renderer.return_value.render.side_effect = ValueError("Unsupported native label boundary")
            with self.assertRaisesRegex(ValueError, "native label boundary"):
                pinning.freeze(self.model, self.settings_path, output)
        renderer.return_value.render.assert_called_once()
        self.assertFalse(output.exists())
