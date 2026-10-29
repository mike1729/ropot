"""7a's four-reply integration gate, through the actual MLX child and Docker."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from tests.test_batched_generation_live import fixture, MODELS

b = importlib.import_module("src.data-pipeline.stage03.generation_loop")
MLX_PYTHON = "/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python"


@unittest.skipUnless(os.environ.get("ROPOT_BENIGN_LOOP_TESTS") == "1", "set ROPOT_BENIGN_LOOP_TESTS=1 in repository Python")
class LiveLoopTests(unittest.TestCase):
    def test_two_models_through_round_file_child(self):
        parent = Path("data/03_generation_checks")
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="batch-07a-", dir=parent)).absolute()
        paths = [Path(b.__file__), Path(b.generation.__file__), Path(__file__),
                 Path(b.__file__).with_name("verification.py"), Path(b.__file__).with_name("inspect_runner.py")]
        sources = {str(p): b.generation.file_hash(p) for p in paths}
        summary = dict(status="running", reply_allowance=4, output_token_allowance=1024,
                       source_sha256_before=sources, runs=[])
        worklist = []
        for i in range(2):
            task, _ = fixture(i)
            path = root / f"fixture-{i}.json"
            b.generation.save(path, task)
            worklist.append((dict(split="train", id=i, sha256=b.generation.file_hash(path)), task))
        recipe = dict(system="Return only a complete Python 3 program.",
            initial="Use separate parse_input and solve functions, and an if __name__ main guard. Read stdin; print the answer.",
            temperature=.2, top_p=.95, base_seed=17, max_tokens=256, max_repairs=0)
        print("Retained Batch 7a qualification:", root, flush=True)
        try:
            for source, slug, revision, prefill, chunk in (
                ("actor", "Qwen3.5-9B-8bit", "16daa4818c54ce5f5436f929d52542eb65bbed9d", 4, 256),
                ("synthetic", "Ministral-3-8B-Instruct-2512-8bit", "f333d6bcb3df97264eddcf985523b69b747f5385", 2, 2048)):
                summary["phase"] = source
                b.generation.save(root / "qualification.json", summary)
                result = b.run(worklist, root / source, source=source,
                    model_path=MODELS / "mlx-community" / slug,
                    model_manifest=MODELS / "metadata" / f"{slug}-{revision}.json", mlx_python=MLX_PYTHON,
                    recipe=recipe, reply_allowance=2, output_token_allowance=512, completion_batch_size=8,
                    prefill_batch_size=prefill, prefill_step_size=chunk, fixtures=True)
                summary["runs"].append(dict(source=source, outcome=str(root / source / "outcome.json"),
                    status=result["status"], timing=result["timing"], observed_tokens=result["observed_tokens"]))
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual([a["state"] for a in result["attempts"]], ["passed", "passed"])
                self.assertEqual([a["generation_status"] for a in result["attempts"]], ["stop", "stop"])
                self.assertEqual([a["verification"]["sample_id"] for a in result["attempts"]], [0, 1])
                self.assertEqual(result["known_submissions"], 2)
                self.assertEqual(result["committed_tokens"], 512)
            summary["status"] = "passed"
        except BaseException as error:
            summary.update(status="review_stop", error=repr(error))
            raise
        finally:
            summary["source_sha256_after"] = {str(p): b.generation.file_hash(p) for p in paths}
            summary["source_unchanged"] = summary["source_sha256_after"] == sources
            if not summary["source_unchanged"]:
                summary.update(status="review_stop", error="Qualification sources changed")
            b.generation.save(root / "qualification.json", summary)
        self.assertTrue(summary["source_unchanged"])
