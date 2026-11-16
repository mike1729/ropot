"""Token ceilings cover full trajectories, scoring and selected-model reloads."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.monitor_training import train
from src.monitor_training.runner import Resources


class TokenBudgetTests(unittest.TestCase):
    def setUp(self):
        self.construction = [dict(role=role, input_ids=[5] * length)
                             for role, length in (("benign", 2), ("actor_attack", 5),
                                                  ("synthetic_attack", 7))]
        self.development = [dict(role=role, input_ids=[6] * length)
                            for role, length in (("benign", 3), ("actor_attack", 4),
                                                 ("synthetic_attack", 9))]
        self.continuations = {"PASS": [10, 2], "FLAG": [11, 12, 2]}
        self.contract = dict(max_new_tokens=8)
        self.snapshots = {1: 1, 4: 4, 16: 16}

    def requirements(self, group):
        return train._token_requirements(SimpleNamespace(
            rows={group: self.construction, "development": self.development},
            renderer=SimpleNamespace(continuations=self.continuations), contract=self.contract,
            group=group, snapshots=self.snapshots))

    def test_shared_honest_covers_all_passes_candidates_and_reload(self):
        # Training: 16 * (14 prefix + 8 label/EOS tokens) = 352.
        # Each fit pass: 42 prefix + 24 generated-cap + 15 likelihood-target
        # tokens = 81. Development is 87 per pass. Reload is the first prefix
        # plus its full output cap: 2 + 8 = 10.
        self.assertEqual(self.requirements("shared_honest"),
                         dict(max_tokens=866, max_fit_tokens=243))

    def test_twins_include_both_branches_and_their_distinct_target_lengths(self):
        # Selective training uses 21 tokens/pass versus honest's 22, so twins
        # cost 16 fewer tokens than twice the honest-only requirement.
        self.assertEqual(self.requirements("twin_continuation"),
                         dict(max_tokens=1716, max_fit_tokens=486))

    def test_requirements_follow_snapshot_count_and_final_pass(self):
        self.snapshots = {1: 1, 2: 2}
        # S: 2 * 22 training + 2 * (81 fit + 87 development) + 10 reload.
        self.assertEqual(self.requirements("shared_honest"),
                         dict(max_tokens=390, max_fit_tokens=162))
        self.assertEqual(self.requirements("twin_continuation"),
                         dict(max_tokens=778, max_fit_tokens=324))

    def test_output_cap_reserves_generation_but_does_not_change_training_or_margins(self):
        self.contract["max_new_tokens"] = 10
        # Per branch: 18 scoring replies + one reload gain two tokens each;
        # the nine fit replies account for 18 of those additional tokens.
        self.assertEqual(self.requirements("shared_honest"),
                         dict(max_tokens=904, max_fit_tokens=261))
        self.assertEqual(self.requirements("twin_continuation"),
                         dict(max_tokens=1792, max_fit_tokens=522))

    def test_final_likelihood_overrun_fails_but_retains_completed_results(self):
        now = [0]
        runtime = mock.Mock()
        raw = dict(text="PASS", completion="natural", reasoning=None, generated_token_ids=[10, 2])
        runtime.generate.return_value = raw

        def likelihood(prefix, continuation):
            if continuation == self.continuations["FLAG"]:
                now[0] = 2  # Only the final model call exceeds the wall ceiling.
            return -len(continuation)

        runtime.log_likelihood.side_effect = likelihood
        usage = dict(tokens=0, development_replies=0, development_margin_forwards=0)
        resources = Resources(dict(max_tokens=1000, max_wall_seconds=1,
                                   max_cost_usd=1, gpu_hourly_usd=1), lambda value: None, usage, 0)
        rows = [dict(task_id=1, role="benign", input_ids=[5, 6], input_ids_sha256="fixture")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "development.json"
            with mock.patch("src.monitor_training.runner.time.monotonic", side_effect=lambda: now[0]):
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    train.evaluate(runtime, rows, self.continuations, self.contract, resources, path,
                                   phase="development", event=lambda value: None)
            saved = json.loads(path.read_text())
        self.assertEqual(usage["development_replies"], 1)
        self.assertEqual(usage["development_margin_forwards"], 2)
        self.assertEqual(saved[0]["reply"]["raw"], raw)
        self.assertEqual(saved[0]["log_likelihood"], {"PASS": -2, "FLAG": -3})
        self.assertEqual(saved[0]["margin"], -1)


if __name__ == "__main__":
    unittest.main()
