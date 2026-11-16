"""Reject ambiguous smoke tolerances/budgets without loading model packages."""

import json
from pathlib import Path
import unittest

from src.monitor_training.adapter import validate_settings


def settings():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "configs/monitor_training/lora-smoke.example.json").read_bytes())


class SmokeSettingsTests(unittest.TestCase):
    def test_example_settings_are_valid(self):
        validate_settings(settings())

    def test_protocol_settings_require_one_row_four_accumulations_and_resume_state(self):
        for name, value in (("per_device_train_batch_size", 2), ("gradient_accumulation_steps", 2),
                            ("save_only_model", True)):
            with self.subTest(name=name):
                config = settings()
                config["training_args"][name] = value
                with self.assertRaisesRegex(ValueError, name):
                    validate_settings(config)

    def test_clipping_norm_requires_a_finite_nonnegative_number(self):
        for value in (True, False, "1", -1, float("inf"), float("nan")):
            with self.subTest(value=value):
                config = settings()
                config["training_args"]["max_grad_norm"] = value
                with self.assertRaisesRegex(ValueError, "max_grad_norm"):
                    validate_settings(config)

    def test_zero_clipping_norm_disables_clipping(self):
        config = settings()
        config["training_args"]["max_grad_norm"] = 0
        validate_settings(config)

    def test_tolerances_require_numbers_instead_of_booleans_or_strings(self):
        for name in ("resume_atol", "resume_rtol"):
            for value in (True, False, "0.001"):
                with self.subTest(name=name, value=value):
                    config = settings()
                    config[name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        validate_settings(config)

    def test_budgets_require_numbers_instead_of_booleans_or_strings(self):
        for name in settings()["budgets"]:
            for value in (True, False, "10"):
                with self.subTest(name=name, value=value):
                    config = settings()
                    config["budgets"][name] = value
                    with self.assertRaisesRegex(ValueError, name):
                        validate_settings(config)

    def test_unknown_top_level_setting_is_not_silently_ignored(self):
        config = settings()
        config["learning_rate"] = 5e-4
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_settings(config)


if __name__ == "__main__":
    unittest.main()
