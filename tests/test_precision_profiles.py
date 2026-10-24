import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

GUARD = '''
import copy,importlib.util,json,sys
spec=importlib.util.spec_from_file_location('guard',sys.argv[1]); p=importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
e=json.load(open(sys.argv[2])); versions=dict(e['versions']); p.importlib.metadata.version=lambda n:versions[n]
assert p.check_runtime(e)==p.digest(e)
for field,value in [('executable','wrong'),('environment',{**e['environment'],'PATH':'wrong'}),('flags',{'isolated':0,'dont_write_bytecode':1}),('versions',{**versions,'mlx':'wrong'})]:
 bad=copy.deepcopy(e); bad[field]=value
 try:p.check_runtime(bad)
 except p.ProfileError:pass
 else:raise AssertionError(field)
from pathlib import Path
path=Path(next(n for n in e['sources'] if n.endswith('/sample_utils.py'))); raw=path.read_bytes(); path.write_bytes(raw+b'patch')
try:p.check_runtime(e)
except p.ProfileError:pass
else:raise AssertionError('unchanged-version sampler patch')
path.unlink()
try:p.check_runtime(e)
except p.ProfileError:pass
else:raise AssertionError('missing source')
assert not any(n.split('.')[0] in {'mlx','mlx_lm','transformers','tokenizers','numpy','src','experiments'} for n in sys.modules)
'''


class PrecisionProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup); self.root = Path(temporary.name).resolve()
        self.P = importlib.import_module("src.data-pipeline.precision_profiles"); self.J = importlib.import_module("src.data-pipeline.generation_journal")
        self.campaign = self.J.Campaign.create(self.root / "campaign", {s: dict(replies=1, reserved_output_tokens=1, reserved_generation_wall_seconds=1) for s in ("actor", "synthetic")})
        self.addCleanup(self.campaign.close); self.index = 0; self.sources = {}
        for suffix in self.P.CRITICAL_SUFFIXES:
            path = self.root / "runtime" / suffix.lstrip("/"); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(suffix.encode())
            self.sources[str(path)] = self.P.sha256(path.read_bytes()).hexdigest()
        late = self.root / "runtime" / "late.py"; late.write_bytes(b"later provenance")
        self.loaded_sources = {**self.sources, str(late): self.P.sha256(late.read_bytes()).hexdigest()}

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return self.P.sha256(path.read_bytes()).hexdigest()

    def fixture(self, damage=None):
        self.index += 1; directory = self.root / str(self.index); directory.mkdir(); inputs = {f"{i}-{kind}": "a" * 64 for i in (0, 1) for kind in ("manifest", "upstream")}
        jobs = [dict(source=s, repo_id=s, revision="fixture", model_path=str(directory / s), manifest={}, upstream={}) for s in ("actor", "synthetic")]
        request = dict(stage="artifacts", directory=str(directory / "wrong" if damage == "origin" else directory), jobs=jobs, sources=self.sources, input_sha256=inputs, guard_path="guard", messages=[])
        identity = self.put(directory / "request.json", request)
        runtime = dict(python=sys.version, executable=sys.executable, platform=sys.platform, versions=dict.fromkeys(self.P.VERSION_FIELDS, "fixture"),
            environment=dict(PATH="/usr/bin:/bin", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"), sources=self.sources,
            argv=[sys.executable, "-I", "-B"], device="Device(gpu, 0)", device_info={})
        runtime_sha = self.put(directory / "runtime.json", runtime); bindings = []
        for job in jobs:
            parent = None
            for bits in (16, 8):
                name = f"records/{job['source']}-{bits}.json"; converter = dict(dtype="bfloat16", quantize=bits == 8, dequantize=False)
                if bits == 8: converter.update(q_bits=8, q_group_size=64, q_mode="affine")
                record = dict(schema_version=1, source=job["source"], bits=bits, model_path=f"models/{job['source']}-{bits}", request_sha256=identity,
                    runtime_sha256=runtime_sha, parent16_record_sha256=parent, input_sha256=inputs, official={k: job[k] for k in ("repo_id", "revision")},
                    loaded_sources=dict(self.loaded_sources), converter_kwargs=converter, manifest=dict(files=[dict(path="weights", bytes=1, sha256="b" * 64)]))
                if job["source"] == "actor" and bits == 8:
                    if damage in ("parent", "runtime"): record["parent16_record_sha256" if damage == "parent" else "runtime_sha256"] = "0" * 64
                    if damage == "official": record["official"]["revision"] = "wrong"
                    if damage == "input": record["input_sha256"] = {**inputs, "0-manifest": "0" * 64}
                    if damage == "manifest": record["manifest"]["files"][0]["path"] = ".hidden"
                    if damage == "duplicate": record["manifest"]["files"] *= 2
                    if damage == "source": record["loaded_sources"][next(iter(self.sources))] = "0" * 64
                    if damage == "converter": converter["q_mode"] = "wrong"
                digest = self.put(directory / name, record); bindings.append(dict(source=job["source"], bits=bits, record=name, sha256=digest))
                if bits == 16: parent = digest
        receipt = dict.fromkeys(self.P.RECEIPT_FIELDS); receipt.update(stage="artifacts", status="passed", artifacts_ready=damage != "ready", inputs_verified=True,
            returncode=0, group_absent=True, generator_requests=0, evidence_path=str(directory), request_sha256=identity, records=bindings, sources=self.sources, input_sha256=inputs)
        return directory, self.put(directory / "receipt.json", receipt), runtime

    def test_four_descriptors_common_digest_and_equal_retry_without_originals(self):
        original, identity, _ = self.fixture(); before = self.campaign.events; ref = self.P.freeze(self.campaign, original / "receipt.json", identity)
        profiles = [self.P.resolve(self.campaign, ref, s, b) for s in ("actor", "synthetic") for b in (16, 8)]
        self.assertEqual(len({p["runtime_guard_sha256"] for p in profiles}), 1)
        self.assertEqual(profiles[0]["expected_runtime"]["sources"], self.loaded_sources)
        for p in profiles:
            self.assertEqual(p["model_path"], str(original / "models" / f"{p['source']}-{p['bits']}")); self.assertEqual(p["fix_mistral_regex"], p["source"] == "synthetic")
            self.assertEqual(p["model_files"], {"weights": dict(bytes=1, sha256="b" * 64)}); self.assertFalse(p["enable_thinking"])
            self.assertEqual(self.P.digest(p), importlib.import_module("src.data-pipeline.study_report").digest(p))
        shutil.rmtree(original); shutil.rmtree(self.root / "runtime")
        with mock.patch.object(self.P.os, "rename", side_effect=AssertionError("equal retry wrote")):
            self.assertEqual(self.P.freeze(self.campaign, original / "receipt.json", identity), ref)
            self.assertEqual(self.P.resolve(self.campaign, ref, "actor", 16), profiles[0])
        self.assertEqual(self.campaign.events, before)

    def test_distinct_metadata_rejections_do_not_publish_or_append(self):
        for damage, reason in (("ready", "Receipt"), ("origin", "Request origin differs"), ("parent", "identity/parent differs"), ("runtime", "Runtime|runtime"),
                ("official", "Official"), ("input", "Input"), ("manifest", "Manifest"), ("duplicate", "Manifest"), ("source", "Source"), ("converter", "Record conversion differs")):
            original, identity, _ = self.fixture(damage); before = self.campaign.events
            with self.subTest(damage=damage), self.assertRaisesRegex(self.P.ProfileError, reason): self.P.freeze(self.campaign, original / "receipt.json", identity)
            self.assertFalse((self.campaign.root / "precision" / identity).exists()); self.assertEqual(self.campaign.events, before)

    def test_strict_json_bounded_reads_and_exact_reference(self):
        original, identity, _ = self.fixture(); path = original / "receipt.json"; raw = path.read_bytes()
        before = self.campaign.events
        with self.assertRaisesRegex(self.P.ProfileError, "Receipt authority"): self.P.freeze(self.campaign, path, "0" * 64)
        self.assertFalse((self.campaign.root / "precision" / ("0" * 64)).exists()); self.assertEqual(self.campaign.events, before)
        for content in (raw[:-1] + b',"status":"passed"}', raw[:-1] + b',"extra":NaN}', raw[:-1] + b',"extra":1e999}'):
            path.write_bytes(content)
            with self.assertRaisesRegex(self.P.ProfileError, "strict JSON"): self.P.freeze(self.campaign, path, self.P.sha256(content).hexdigest())
        path.write_bytes(raw)
        with mock.patch.object(self.P, "MAX_BYTES", 8), self.assertRaisesRegex(self.P.ProfileError, "metadata size"): self.P.freeze(self.campaign, path, identity)
        with self.assertRaisesRegex(self.P.ProfileError, "Reference"): self.P.resolve(self.campaign, dict(path="precision/../escape", receipt_sha256=identity), "actor", 16)
        self.campaign.close()
        with self.assertRaisesRegex(self.P.ProfileError, "Reference/lease"): self.P.freeze(self.campaign, path, identity)

    def test_exact_tree_and_before_after_rename_crash_recovery(self):
        for fault in ("before", "after", "extra", "missing", "symlink", "unreadable"):
            original, identity, _ = self.fixture(); destination = self.campaign.root / "precision" / identity; rename = self.P.os.rename; before = self.campaign.events
            def fail(source, target):
                if fault == "after": rename(source, target)
                raise OSError("rename fixture")
            if fault in ("before", "after"):
                with mock.patch.object(self.P.os, "rename", side_effect=fail), self.assertRaises(OSError): self.P.freeze(self.campaign, original / "receipt.json", identity)
                self.assertEqual(destination.exists(), fault == "after"); self.P.freeze(self.campaign, original / "receipt.json", identity)
            else:
                ref = self.P.freeze(self.campaign, original / "receipt.json", identity)
                if fault == "extra": (destination / "extra").write_bytes(b"extra")
                if fault == "missing": (destination / "runtime.json").unlink()
                if fault == "symlink": (destination / "runtime.json").unlink(); (destination / "runtime.json").symlink_to(original / "runtime.json")
                opened = self.P.os.open
                def unreadable(path, *args, **kwargs):
                    if fault == "unreadable" and Path(path) == destination / "runtime.json": raise PermissionError("fixture unreadable")
                    return opened(path, *args, **kwargs)
                with mock.patch.object(self.P.os, "open", side_effect=unreadable), self.assertRaises(self.P.ProfileError): self.P.freeze(self.campaign, original / "receipt.json", identity)
            self.assertEqual(self.campaign.events, before)

    def test_isolated_by_path_guard_uses_only_stdlib(self):
        original, identity, _ = self.fixture(); ref = self.P.freeze(self.campaign, original / "receipt.json", identity)
        expected = self.P.resolve(self.campaign, ref, "actor", 16)["expected_runtime"]; path = self.root / "expected.json"; self.put(path, expected)
        unguarded = subprocess.run([sys.executable, "-I", "-c", GUARD, self.P.__file__, str(path)], env=expected["environment"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(unguarded.returncode, 0); self.assertIn("ProfileError: Runtime flags differ", unguarded.stderr)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", GUARD, self.P.__file__, str(path)], env=expected["environment"], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
