import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
from tests import test_bf16_compatibility as fixture


class PrecisionArtifactsTests(unittest.TestCase):
    def setUp(self):
        fixture.Bf16CompatibilityTests.setUp(self)
        self.P = importlib.import_module("experiments.prepare_precision_artifacts"); self.real_worker = self.P.WORKER
        shards = [f"model-{i}.safetensors" for i in range(4)]
        for name in shards: (self.model / name).write_bytes(b"fixture")
        index = dict(weight_map={f"tensor{i}": name for i, name in enumerate(shards)})
        (self.model / "model.safetensors.index.json").write_text(json.dumps(index))
        self.manifest = dict(repo_id="fixture/repo", revision="fixture", files=[dict(path=p.name, bytes=p.stat().st_size,
            sha256=fixture.sha256(p.read_bytes()).hexdigest()) for p in self.model.iterdir()])
        self.upstream = dict(repo_id="fixture/repo", revision="fixture", weight_index=dict(json=index,
            response_sha256=fixture.sha256((self.model / "model.safetensors.index.json").read_bytes()).hexdigest()),
            tree_api=dict(json=[dict(path=n, size=7, lfs=dict(oid=fixture.sha256(b"fixture").hexdigest())) for n in shards]))
        Path(self.B.MANIFEST_PATH).write_text(json.dumps(self.manifest)); Path(self.B.UPSTREAM_PATH).write_text(json.dumps(self.upstream))
        job = dict(source="actor", model_path=str(self.model), manifest_path=str(self.B.MANIFEST_PATH),
                   upstream_path=str(self.B.UPSTREAM_PATH), revision="fixture", repo_id="fixture/repo", file_count=6)
        for name, value in dict(INTERPRETER=sys.executable, WORKER=self.worker, JOBS=[job], MIN_FREE_BYTES=0,
                WALL_SECONDS=1.0, CLEANUP_SECONDS=3.0, STDOUT_LIMIT=512, STDERR_LIMIT=512).items():
            patch = mock.patch.object(self.P, name, value); patch.start(); self.addCleanup(patch.stop)

    def invoke(self, mode):
        worker = fixture.FAKE.replace("{'fixture':True}", "{'inputs_verified':True,'artifacts_ready':False,'generator_requests':0}")
        self.worker.write_text("MODE=" + repr(mode) + "\n" + worker); self.index += 1; directory = self.root / str(self.index)
        result = self.P.prepare(directory)
        self.assertEqual(result, json.loads((directory / "receipt.json").read_text()))
        self.assertTrue(result["group_absent"]); self.assertTrue(fixture.process_gone(result["pid"]))
        return result, directory

    def test_canonical_tokenizers_and_only_approved_config_changes(self):
        for source, names in self.P.TOKENS.items():
            official, output = self.root / source, self.root / (source + "-output"); official.mkdir(); output.mkdir()
            for name in names: (official / name).write_bytes(name.encode())
            for name in self.P.TOKEN_FILES: (output / name).write_bytes(b"generated")
            (output / "retained.weight").write_bytes(b"keep"); self.P.canonicalize(official, output, source)
            self.assertEqual({p.name for p in output.iterdir()}, names | {"retained.weight"})
            self.assertTrue(all((output / n).read_bytes() == (official / n).read_bytes() for n in names))
            raw = dict(vision_config={}, text_config={"rope_parameters": {"rope_type": "default"}} if source == "actor" else {})
            base = dict(text_config={"rope_parameters": {"type": "default"}}) if source == "actor" else dict(text_config={"tie_word_embeddings": False}, eos_token_id=2)
            quant = {**base, "quantization": self.P.QUANTIZATION, "quantization_config": self.P.QUANTIZATION}
            self.P.config_checked(raw, base, source, 16); self.P.config_checked(raw, quant, source, 8, base)
            for bits, changed in ((16, {**base, "extra": 1}), (8, {**quant, "extra": 1})):
                with self.assertRaisesRegex(ValueError, "Semantic config"): self.P.config_checked(raw, changed, source, bits, base)

    def test_header_index_duplicate_mapping_and_extra_shard_checks(self):
        directory = self.root / "headers"; directory.mkdir(); index = directory / "model.safetensors.index.json"
        def shard(name, key):
            raw = json.dumps({key: dict(dtype="BF16", shape=[1], data_offsets=[0, 2])}).encode()
            (directory / name).write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0\0")
        shard("one.safetensors", "language_model.one"); shard("two.safetensors", "language_model.two")
        mapping = dict(weight_map={"language_model.one": "one.safetensors", "language_model.two": "two.safetensors"}); index.write_text(json.dumps(mapping))
        self.assertEqual(len(self.P.headers(directory)), 2)
        shard("two.safetensors", "language_model.one")
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"): self.P.headers(directory)
        shard("two.safetensors", "language_model.two"); mapping["weight_map"] = {"language_model.one": "two.safetensors", "language_model.two": "one.safetensors"}; index.write_text(json.dumps(mapping))
        with self.assertRaisesRegex(ValueError, "Header/index"): self.P.headers(directory)
        (directory / "extra.safetensors").write_bytes(b"unused")
        with self.assertRaisesRegex(ValueError, "Shard inventory"): self.P.headers(directory)

    def test_storage_coverage_and_nonquantized_identity(self):
        tensors = {f"language_model.model.misc_{i}.weight": dict(dtype="BF16", shape=[1]) for i in range(403)}
        tensors.pop("language_model.model.misc_402.weight"); conv = "language_model.model.layers.0.linear_attn.conv1d.weight"; tensors[conv] = dict(dtype="BF16", shape=[4, 4, 1])
        tensors.update({f"language_model.model.layers.{i}.linear_attn.A_log": dict(dtype="F32", shape=[32]) for i in range(32) if i % 4 != 3})
        self.P.storage_checked(tensors, "actor", 16)
        with self.assertRaisesRegex(ValueError, "dtype/layout"):
            self.P.storage_checked({**tensors, conv: dict(dtype="BF16", shape=[4, 1, 4])}, "actor", 16)
        with self.assertRaises(ValueError): self.P.storage_checked({**tensors, "language_model.extra": dict(dtype="BF16", shape=[1])}, "actor", 16)
        changed = {**tensors, "language_model.model.misc_0.weight": dict(dtype="F8_E4M3", shape=[1])}
        with self.assertRaisesRegex(ValueError, "dtype/layout"): self.P.storage_checked(changed, "actor", 16)
        paths = [f"language_model.model.misc_{i}" for i in range(250)]; packed = dict(tensors)
        for path in paths:
            packed.update({path + "." + suffix: dict(dtype="U32" if suffix == "weight" else "BF16", shape=[1]) for suffix in ("weight", "scales", "biases")})
        self.P.storage_checked(packed, "actor", 8); actual = dict.fromkeys(paths, self.P.QUANTIZATION)
        self.P.coverage_checked("actor", paths, actual); actual["language_model.wrong"] = actual.pop(paths[0])
        with self.assertRaises(ValueError): self.P.coverage_checked("actor", paths, actual)
        self.assertEqual(self.P.nonquantized_names(["layers.1.weight", "layers.10.weight"], ["layers.1"]), ["layers.10.weight"])
        self.P.identity_checked({"x": "digest"}, {"x": "digest"})
        with self.assertRaisesRegex(ValueError, "Nonquantized"): self.P.identity_checked({"x": "digest"}, {"x": "changed"})

    def test_complete_output_inventory_rejects_extras_cache_and_changed_generation_config(self):
        official, output = self.root / "official", self.root / "output"; official.mkdir(); output.mkdir()
        (official / "generation_config.json").write_bytes(b"{}")
        for name in self.P.TOKENS["synthetic"] | {"config.json", "README.md", "generation_config.json"}: (output / name).write_bytes(b"{}")
        header = json.dumps({"language_model.x": dict(dtype="BF16", shape=[1], data_offsets=[0, 2])}).encode()
        (output / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")
        (output / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={"language_model.x": "model.safetensors"})))
        manifest = self.P.inventory(output, official, "synthetic")
        self.assertEqual({row["path"] for row in manifest["files"]}, {p.name for p in output.iterdir()})
        for damage in ("extra", ".cache", "generation_config.json"):
            path = output / damage
            if damage == ".cache": path.mkdir()
            else: path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Generation config" if damage == "generation_config.json" else "Output inventory"):
                self.P.inventory(output, official, "synthetic")
            if damage == ".cache": path.rmdir()
            elif damage == "generation_config.json": path.write_bytes(b"{}")
            else: path.unlink()

    def test_new_controller_owns_timeout_protocol_cleanup_and_cancellation(self):
        synced = set(); sync, write = self.P.os.fsync, self.P.B._write
        def observed(fd): synced.add(self.P.os.fstat(fd).st_ino); return sync(fd)
        def receipt(path, value):
            if Path(path).name == "receipt.json": self.assertTrue({(Path(path).parent / n).stat().st_ino for n in ("stdout.jsonl", "stderr.txt")} <= synced)
            return write(path, value)
        with mock.patch.object(self.P.os, "fsync", side_effect=observed), mock.patch.object(self.P.B, "_write", side_effect=receipt): self.invoke("ok")
        for mode, status in (("ok", "passed"), ("pid", "worker_error"), ("hash", "worker_error"), ("malformed", "worker_error"), ("object", "worker_error"),
                             ("nan", "worker_error"), ("stdout", "output_limit"), ("stderr", "output_limit"), ("timeout", "timeout")):
            result, directory = self.invoke(mode); self.assertEqual(result["status"], status)
            self.assertEqual(result["request_sha256"], fixture.sha256((directory / "request.json").read_bytes()).hexdigest())
            if mode == "timeout": self.assertIn(b"partial", (directory / "stdout.jsonl").read_bytes()); self.assertTrue(fixture.process_gone(int((directory / "child.pid").read_text())))
        with mock.patch.object(self.P.B, "_exited", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt): self.invoke("timeout")
        self.assertEqual(json.loads((self.root / str(self.index) / "receipt.json").read_text())["status"], "cancelled")
        kill = self.P.B._kill_group
        def failed(process): kill(process); raise OSError("cleanup fixture")
        with mock.patch.object(self.P.B, "_kill_group", side_effect=failed): self.assertEqual(self.invoke("ok")[0]["status"], "worker_error")

    def test_real_worker_rejects_corrupt_inputs_before_mlx_and_never_publishes_artifacts(self):
        errors = dict(bytes="Manifest file differs", repo="Official identity differs", count="Official identity differs",
                      index="Official index differs", lfs="Official shard differs", selection="Official shard selection differs")
        for failure in ("valid", "bytes", "repo", "count", "index", "lfs", "selection", "overflow"):
            manifest, upstream = json.loads(json.dumps(self.manifest)), json.loads(json.dumps(self.upstream))
            (self.model / "weights").write_bytes(b"changed" if failure == "bytes" else b"fixture")
            if failure == "repo": manifest["repo_id"] = "wrong/repo"
            if failure == "index": upstream["weight_index"]["response_sha256"] = "0" * 64
            if failure == "lfs": upstream["tree_api"]["json"][0]["lfs"]["oid"] = "0" * 64
            if failure == "selection": upstream["weight_index"]["json"]["weight_map"].pop("tensor3")
            self.P.JOBS[0]["file_count"] = 7 if failure == "count" else 6
            Path(self.B.MANIFEST_PATH).write_text(json.dumps(manifest)); Path(self.B.UPSTREAM_PATH).write_text(json.dumps(upstream))
            directory = self.root / failure
            with mock.patch.object(self.P, "WORKER", self.real_worker), mock.patch.object(self.P, "STDERR_LIMIT", 16384), mock.patch.object(self.P, "STDOUT_LIMIT", 8 if failure == "overflow" else 512): result = self.P.prepare(directory)
            self.assertEqual(result["status"], "passed" if failure == "valid" else "output_limit" if failure == "overflow" else "worker_error")
            self.assertEqual(result["inputs_verified"], failure == "valid")
            if failure == "valid": self.assertEqual(result["events"][-1]["result"], dict(inputs_verified=True, artifacts_ready=False, generator_requests=0))
            elif failure != "overflow":
                error = next(e for e in result["events"] if e.get("event") == "error" and e.get("phase") == "inputs")
                self.assertIn(errors[failure], error["error"])
            self.assertEqual(json.loads((directory / "request.json").read_text())["stage"], "inputs-only")
            self.assertFalse((directory / "artifacts.json").exists())
