"""Keep artifact readers usable without execution or analysis dependencies."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


class PipelineImportTests(unittest.TestCase):
    def test_data_consumers_import_with_only_the_standard_library(self):
        package = "src.data-pipeline."
        cases = {
            "stage02.published_cohort": ("stage02.cohort", "stage02.publish_cohort", "stage01.analyze"),
            "stage03.study_tasks": ("stage02.cohort", "stage02.publish_cohort", "stage01.analyze"),
            "stage03.generation_loop": ("stage02.cohort", "stage02.publish_cohort", "stage01.analyze",
                                        "stage03.benign_acquisition", "stage05.attacks"),
            "stage03.benign_acquisition": ("stage02.cohort", "stage02.publish_cohort", "stage01.analyze"),
            "stage01.analyze": (),
            "stage01.export_test": ("stage01.analyze",),
            "stage02.publish_cohort": ("stage01.analyze",),
        }
        for module, forbidden in cases.items():
            with self.subTest(module=module):
                # A fresh process avoids hiding eager imports behind sys.modules;
                # -S excludes site packages and the optional runtime dependencies.
                result = subprocess.run(
                    [sys.executable, "-S", "-c", (
                        "import importlib,json,sys; importlib.import_module(sys.argv[1]); "
                        "print(json.dumps(sorted(sys.modules)))"
                    ), package + module],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True,
                    text=True, check=True, timeout=10,
                )
                imported = set(json.loads(result.stdout))
                blocked = {package + name for name in (*forbidden, "stage01.qualify", "stage01.local_runner")}
                blocked.update(("tokenizers", "inspect_ai", "mlx_lm"))
                self.assertFalse(imported & blocked, sorted(imported & blocked))
