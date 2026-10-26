"""7b's bounded two-model gate: controlled first failure, one successful repair."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from tests.test_benign_loop_live import b, MODELS, MLX_PYTHON
from tests.test_batched_generation_live import fixture


@unittest.skipUnless(os.environ.get("ROPOT_BENIGN_REPAIR_TESTS") == "1", "set ROPOT_BENIGN_REPAIR_TESTS=1 in repository Python")
class LiveRepairTests(unittest.TestCase):
    def test_two_models_controlled_repair(self):
        parent = Path("data/03_generation_checks")
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="batch-07b-", dir=parent)).absolute()
        paths = [Path(b.__file__), Path(b.generation.__file__), Path(__file__),
                 Path(b.__file__).with_name("repair_feedback.py"), Path(b.__file__).with_name("verification.py"),
                 Path(b.__file__).with_name("inspect_runner.py")]
        sources = {str(p): b.generation.file_hash(p) for p in paths}
        summary = dict(status="running", reply_allowance=8, output_token_allowance=2048,
                       source_sha256_before=sources, runs=[])
        worklist = []
        for i in range(2):
            task, _ = fixture(i)
            if i == 0:
                task["question"] += ("\nFIRST DRAFT CONTROL (only for your initial reply): this fixture requires an intentionally wrong first draft. "
                    "Return a complete Python program that reads stdin and prints the literal integer 0 for every input. "
                    "Do not compute the square in this first draft. This control ends when a repair is requested.")
            else:
                task["question"] = "Read n followed by n integers and print the number of distinct integers."
                for case in task["cases"]:
                    case["answer"] = str(int(case["answer"]) - 1)
            path = root / f"fixture-{i}.json"
            b.generation.save(path, task)
            worklist.append((dict(split="train", id=i, sha256=b.generation.file_hash(path)), task))
        recipe = dict(system="Return only a complete Python 3 program.",
            initial="For this initial reply, obey any FIRST DRAFT CONTROL in the statement instead of solving the task. "
                    "If there is no such control, solve the task normally. "
                    "Use separate parse_input and solve functions, and an if __name__ main guard. Read stdin; print the required result.",
            repair="The FIRST DRAFT CONTROL has ended. Replace the program with the correct square program, using the original specification and public feedback below.",
            temperature=.2, top_p=.95, base_seed=17, max_tokens=256, repair_max_tokens=256, max_repairs=1,
            feedback_field_bytes=512, feedback_bytes=4096)
        print("Retained Batch 7b qualification:", root, flush=True)
        try:
            for source, slug, revision, prefill, chunk in (
                ("actor", "Qwen3.5-9B-8bit", "16daa4818c54ce5f5436f929d52542eb65bbed9d", 4, 256),
                ("synthetic", "Ministral-3-8B-Instruct-2512-8bit", "f333d6bcb3df97264eddcf985523b69b747f5385", 2, 2048)):
                manifest = MODELS / "metadata" / f"{slug}-{revision}.json"
                shutil.copyfile(manifest, root / f"{source}-manifest.json")
                summary["phase"] = source
                b.generation.save(root / "qualification.json", summary)
                result = b.run(worklist, root / source, source=source,
                    model_path=MODELS / "mlx-community" / slug, model_manifest=manifest, mlx_python=MLX_PYTHON,
                    recipe=recipe, reply_allowance=4, output_token_allowance=1024, completion_batch_size=8,
                    prefill_batch_size=prefill, prefill_step_size=chunk, fixtures=True)
                summary["runs"].append(dict(source=source, outcome=str(root / source / "outcome.json"),
                    status=result["status"], timing=result["timing"], observed_tokens=result["observed_tokens"],
                    committed_replies=result["committed_replies"], committed_tokens=result["committed_tokens"]))
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 0])
                self.assertEqual([a["generation_status"] for a in result["attempts"]], ["stop"] * 3)
                self.assertEqual([a["verification"]["status"] for a in result["attempts"]], ["failed", "passed", "passed"])
                self.assertEqual([a["verification"]["sample_id"] for a in result["attempts"]], [0, 1, 0])
                self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18, 19])
                self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
                self.assertEqual(result["known_submissions"], 3)
                self.assertEqual(result["committed_tokens"], 768)
                self.assertLessEqual(result["observed_tokens"], 768)
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
