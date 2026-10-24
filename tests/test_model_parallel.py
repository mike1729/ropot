import copy
import importlib
import json
from pathlib import Path
import unittest
from unittest import mock
from tests import test_batch_throughput as fixture


class ModelParallelTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.BatchThroughputTests(methodName="runTest"); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.B = importlib.import_module("experiments.model_parallel"); self.H = self.f.B; self.root = self.f.root
        self.M = importlib.import_module("experiments.mlx_batch_throughput")
        self.f.index = 1  # The inherited protocol fixture resets its run counter after publishing receipt 1.
        directory, identity, _ = self.f.fixture(); ref = self.f.P.freeze(self.f.campaign, directory / "receipt.json", identity)
        self.descriptors = {role: self.f.P.resolve(self.f.campaign, ref, role, 8) for role in ("actor", "synthetic")}
        self.configs = dict(actor=dict(batch_size=32, prefill_batch_size=4, prefill_step_size=256),
                            synthetic=dict(batch_size=32, prefill_batch_size=2, prefill_step_size=2048))
        self.profiles = self.root / "profiles.json"
        self.f.put(self.profiles, dict(profiles={"8": {role: dict(descriptor=d) for role, d in self.descriptors.items()}}))
        self.paths = {f"{role}_{name}": self.prior(role, batch) for name, batch in (("best", 32), ("fallback", 8)) for role in self.descriptors}

    def prior(self, role, batch=32):
        directory = self.root / f"prior-{role}-{batch}"; directory.mkdir()
        config = dict(self.configs[role], batch_size=batch)
        cache = 4 * 1024**3 if batch == 32 else None
        job = self.M.build_job(self.descriptors[role], [config], panel_size=32, cache_limit_bytes=cache)
        identity = self.H.sha256(self.H._raw(job)).hexdigest()
        windows = [dict(event="window_done", job_sha256=identity, window=w["id"], elapsed=1,
                        stats=dict(generation_tokens=sum(r["cap"] for r in w["rows"])),
                        memory=dict(active_bytes=3 * 1024**3, peak_bytes=(20 if batch == 32 and w["id"] == "q2" else 4) * 1024**3,
                                    cache_before_bytes=1024**3, cache_bytes=0)) for w in job["windows"]]
        response = dict(status="passed", complete=True, returncode=0, pid=7, group_absent=True,
            cleanup_error=None, parent_error=None, protocol_error=None, telemetry_complete=True,
            telemetry=[dict(available_bytes=64 * 1024**3, swap_bytes=17)], windows=windows,
            events=[dict(event="qualification", job_sha256=identity, checks=dict.fromkeys(self.H.CHECKS, True))])
        response["job_sha256"] = identity
        (directory / "job.json").write_bytes(self.H._raw(job)); self.f.put(directory / "response.json", response)
        if cache is not None:
            self.f.put(directory / "cache-control.json", dict(job_sha256=identity, requested_bytes=cache,
                observed_bytes=cache, previous_bytes=8 * 1024**3))
        return directory

    def evidence(self):
        return {key: self.B.prior(path, key.split("_")[0], 32 if key.endswith("best") else 8, key.endswith("best")) for key, path in self.paths.items()}

    def test_all_window_screening_selects_best_fallback_or_skip_before_launch(self):
        priors = self.evidence()
        self.assertEqual(priors["actor_best"]["screening_bytes"], 21 * 1024**3)
        self.assertEqual(priors["synthetic_fallback"]["screening_bytes"], 5 * 1024**3)
        for available, choice in ((54, "best"), (40, "fallback"), (21, None)):
            decision = self.B.select_pair(priors, available * 1024**3)
            self.assertEqual(decision["choice"], choice)
            self.assertEqual(decision["required_bytes"], dict(best=54 * 1024**3, fallback=22 * 1024**3))
        historical = self.H._raw(priors["actor_fallback"]["response"])
        self.assertNotIn(b"started_seconds", historical)

    def test_prior_rejects_otherwise_valid_missing_memory_binding_swap_and_cache(self):
        directory = self.paths["actor_best"]; path = directory / "response.json"; original = json.loads(path.read_bytes())
        for fault in ("memory", "nonfinite", "window", "sha", "qualification", "qualification_sha", "swap"):
            row = copy.deepcopy(original)
            if fault == "memory": row["windows"][0]["memory"].pop("peak_bytes")
            if fault == "nonfinite": row["windows"][0]["memory"]["peak_bytes"] = True
            if fault == "window": row["windows"].pop()
            if fault == "sha": row["job_sha256"] = "0" * 64
            if fault == "qualification": row["events"][0]["checks"]["greedy_equal"] = False
            if fault == "qualification_sha": row["events"][0]["job_sha256"] = "0" * 64
            if fault == "swap": row["telemetry"].append(dict(available_bytes=64 * 1024**3, swap_bytes=18))
            self.f.put(path, row)
            with self.subTest(fault=fault), self.assertRaises((ValueError, KeyError)): self.B.prior(directory, "actor", 32, True)
        self.f.put(path, original)
        proof = directory / "cache-control.json"; changed = json.loads(proof.read_bytes()); changed["observed_bytes"] = 0; self.f.put(proof, changed)
        with self.assertRaises(ValueError): self.B.prior(directory, "actor", 32, True)
        directory = self.paths["actor_fallback"]; row = json.loads((directory / "response.json").read_bytes())
        row["windows"][0]["memory"]["cache_before_bytes"] = 4 * 1024**3; self.f.put(directory / "response.json", row)
        with self.assertRaises(ValueError): self.B.prior(directory, "actor", 8, False)

    def test_prepare_preserves_each_role_identity_config_and_fallback_proposal(self):
        cpu = importlib.import_module("experiments.cpu_throughput")
        for available, choice, batch in ((56, "best", 32), (40, "fallback", 8), (21, None, None)):
            output = self.root / f"prepared-{available}"
            with mock.patch.object(self.B, "_memory", return_value=dict(available_bytes=available * 1024**3, total_bytes=64 * 1024**3, swap_used_bytes=17, observed_seconds=1)), \
                 mock.patch.object(cpu.shutil, "which", return_value="/fixture/docker"), mock.patch.object(self.B.O, "spawn") as spawn:
                pair = self.B.prepare(self.profiles, self.paths, output)
            spawn.assert_not_called(); self.assertEqual(pair["decision"]["choice"], choice)
            if choice is None: continue
            for role, plan in pair["models"].items():
                expected = self.M.build_job(self.descriptors[role], [dict(self.configs[role], batch_size=batch)], 600.0, panel_size=32, cache_limit_bytes=4 * 1024**3)
                self.assertEqual(plan["job"], expected); self.assertEqual(plan["source"], role)
                self.assertEqual(plan["job_sha256"], self.H.sha256(self.H._raw(expected)).hexdigest())
                self.assertEqual(expected["request"]["precision_profile"]["descriptor"], self.descriptors[role])

    def test_prepare_refuses_profile_export_role_mismatch_without_launch(self):
        export = json.loads(self.profiles.read_bytes()); export["profiles"]["8"]["actor"]["descriptor"] = self.descriptors["synthetic"]
        self.f.put(self.profiles, export)
        with mock.patch.object(self.B, "_memory", return_value=dict(available_bytes=56 * 1024**3, total_bytes=64 * 1024**3, swap_used_bytes=17, observed_seconds=1)), \
             mock.patch.object(self.B.O, "spawn") as spawn, self.assertRaises(ValueError):
            self.B.prepare(self.profiles, self.paths, self.root / "mislabeled")
        spawn.assert_not_called()

    def test_parallel_needs_actual_intersection_and_failure_stops_both_owners(self):
        pair = dict(models={role: dict(job={}) for role in self.descriptors}, sources={})
        for index, (overlap, failure) in enumerate(((True, False), (False, False), (True, True))):
            def spawn(kind, directory, plan, lock, fd):
                (directory / "gpu-launch.json").write_text("{}")
                self.f.put(directory / "gpu/response.json", dict(observed_generation_tokens=3, finished_sequences=2))
                return dict(pid=7, process=object(), stdout=mock.Mock(closed=True), stderr=mock.Mock(closed=True))
            def audit(kind, directory, plan, child):
                role = directory.parent.name
                if failure and role == "actor": raise ValueError("GPU worker proof incomplete")
                return dict(response={}, intervals=[(1, 2)] if role == "actor" or overlap else [(3, 4)])
            with mock.patch.object(self.B.O, "spawn", side_effect=spawn), mock.patch.object(self.H, "_exited", return_value=True), \
                 mock.patch.object(self.B.O, "audit", side_effect=audit), mock.patch.object(self.B.O, "stop", return_value=([], None)) as stop:
                row = self.B.run_phase(pair, "parallel", self.root / f"phase-case-{index}", self.f.lock, 3)
            self.assertEqual(row["status"], "passed" if overlap and not failure else "blocked")
            self.assertEqual(row["gpu_job_attempts"], 2); self.assertEqual(len(stop.call_args.args[0]), 2)
            self.assertEqual(row["partial_costs"]["actor"], dict(tokens_lower_bound=3, finished_replies_lower_bound=2))
            self.assertEqual(stop.call_args.kwargs["interrupt"], failure)
            if overlap and not failure: self.assertAlmostEqual(row["useful_reply_rate"] * row["elapsed_seconds"], 128)
            else: self.assertIsNone(row["useful_reply_rate"])
            if not overlap: self.assertEqual(row["intersection_seconds"], 0)

    def test_fixed_order_skip_failed_owner_and_partial_receipts_keep_actual_costs(self):
        self.assertEqual(self.B.PHASES, ("actor_synthetic", "parallel", "parallel", "synthetic_actor"))
        models = {role: dict(job=json.loads((self.paths[role + "_best"] / "job.json").read_bytes())) for role in self.descriptors}
        for outcome in ("passed", "blocked", "missing", "corrupt", "skip"):
            output = self.root / ("block-" + outcome)
            def prepare(profiles, paths, directory):
                directory.mkdir(); return dict(models={} if outcome == "skip" else models, decision=dict(choice=None if outcome == "skip" else "best"), preflight={})
            def phase(pair, mode, directory, lock, fd):
                directory.mkdir()
                for role in self.descriptors:
                    child = directory / role; child.mkdir(); (child / "gpu-launch.json").write_text("{}")
                if outcome in ("missing", "corrupt"):
                    if outcome == "corrupt": (directory / "phase.json").write_text("{")
                    raise KeyboardInterrupt
                row = dict(mode=mode, status=outcome, elapsed_seconds=1, gpu_job_attempts=2,
                    audits={role: dict(intervals=[(1, 2)]) for role in self.descriptors}, partial_costs={"actor": dict(tokens_lower_bound=3)})
                self.H._write(directory / "phase.json", row); return row
            with mock.patch.object(self.B, "prepare", side_effect=prepare), mock.patch.object(self.B, "run_phase", side_effect=phase) as phases:
                if outcome in ("missing", "corrupt"):
                    with self.assertRaises(KeyboardInterrupt): self.B.run(self.profiles, self.paths, output, self.f.lock)
                    result = json.loads((output / "response.json").read_bytes())
                    self.assertIn("phase_read_error", result["phases"][0]); self.assertIn("KeyboardInterrupt", result["parent_error"])
                    self.assertEqual(result["phases"][0]["partial_costs"], dict(unknown=True))
                else: result = self.B.run(self.profiles, self.paths, output, self.f.lock)
            count = 4 if outcome == "passed" else 0 if outcome == "skip" else 1
            self.assertEqual([c.args[1] for c in phases.call_args_list], list(self.B.PHASES[:count]))
            self.assertEqual(result["attempted"]["gpu_jobs"], 2 * count); self.assertEqual(len(result["unrun"]), 4 - count)
            if outcome == "passed": self.assertEqual(result["planned"]["sequences"], 824); self.assertEqual(result["serial_over_parallel_ratio"], 1)
            elif outcome != "skip": self.assertEqual(result["status"], "blocked"); self.assertIsNone(result["serial_over_parallel_ratio"])
            if outcome == "blocked": self.assertEqual(result["phases"][0]["partial_costs"]["actor"]["tokens_lower_bound"], 3)
