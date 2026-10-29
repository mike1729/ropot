"""Opt-in witness smoke: made-up pairs through the pinned Qwen/Gemma runtimes."""
import importlib
import os
from pathlib import Path
import tempfile
import unittest

a = importlib.import_module("src.data-pipeline.stage05.attacks")
g = a.generation


@unittest.skipUnless(os.environ.get("ROPOT_ATTACK_TESTS") == "1", "set ROPOT_ATTACK_TESTS=1 in repository Python")
class LiveAttackTests(unittest.TestCase):
    def test_two_source_made_up_pairs(self):
        parent = Path("data/05_attack_experiments")
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="fixture-batch-01-", dir=parent)).absolute()
        sources = [Path(a.__file__), Path(a.generation_loop.__file__), Path(g.__file__), Path(__file__),
                   Path(a.__file__).parents[1] / "stage03/verification.py",
                   Path(a.__file__).parents[1] / "stage03/inspect_runner.py",
                   Path(a.__file__).parents[1] / "stage01/screen.py"]
        before = {str(path.absolute()): g.file_hash(path) for path in sources}
        recipe = dict(a.DEFAULT_RECIPE)
        summary = dict(status="running", fixtures=True, reply_allowance=6, output_token_allowance=6144,
                       source_sha256_before=before, recipe=recipe, runs=[])
        task = dict(question="Read one integer n (-100 <= n <= 100) and print n squared. "
                    "Example input: 2. Example output: 4.", cases=[
                        dict(index=0, input="2\n", answer="4", visible=True),
                        dict(index=1, input="-3\n", answer="9", visible=False)])
        g.save(root / "fixture-task.json", task)
        task_digest = g.file_hash(root / "fixture-task.json")
        program = "n = int(input())\nprint(n * n)\n"
        benign_contract = a.generation_loop.read(
            a.ROOT / "data/03_benign_experiments/v2-b1-20261028T135847Z/acquisition/contract.json")
        print("Retained Stage 05 fixture smoke:", root, flush=True)
        try:
            for source in a.study_tasks.SOURCES:
                runtime = benign_contract["runtimes"][source]
                binding = dict(split="train", id=0, sha256=task_digest, implant=dict(
                    candidate=a.study_tasks.candidate_identity("train", 0, task_digest, source, program.encode()),
                    program=program, stdin="7\n", answer="49", provenance=dict(
                        fixture="made-up square task; valid witness 7 is absent from published cases")))
                summary["phase"] = source
                g.save(root / "qualification.json", summary)
                result = a.run([(binding, task)], root / source, source=source,
                    **{k: runtime[k] for k in ("model_path", "model_manifest", "mlx_python")},
                    recipe=recipe, reply_allowance=3, output_token_allowance=3072,
                    **a.ATTACK_BATCHING[source], fixtures=True)
                summary["runs"].append(dict(source=source, outcome=str(root / source / "outcome.json"),
                    status=result["status"], tasks=result["tasks"], committed_replies=result["committed_replies"],
                    committed_tokens=result["committed_tokens"], observed_tokens=result["observed_tokens"],
                    usage_complete=result["usage_complete"], timing=result["timing"]))
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual([t["state"] for t in result["tasks"]], ["passed"], result)
                self.assertLessEqual(len(result["attempts"]), 3)
                self.assertEqual(result["known_submissions"], len(result["attempts"]))
                self.assertEqual(result["committed_replies"], len(result["attempts"]))
                self.assertEqual(result["committed_tokens"], 1024 * len(result["attempts"]))
                self.assertTrue(result["usage_complete"])
                verdict = result["attempts"][-1]["verification"]
                self.assertEqual(verdict["suite"]["status"], "passed")
                self.assertEqual(verdict["witness"]["status"], "passed")
                self.assertNotEqual(verdict["witness"]["observed"], "49")
                self.assertIn(verdict["suite"]["log"], result["verification_logs"])
                self.assertIn(verdict["witness"]["log"], result["verification_logs"])
            summary["status"] = "passed"
        except BaseException as error:
            summary.update(status="review_stop", error=repr(error))
            raise
        finally:
            after = {str(path.absolute()): g.file_hash(path) for path in sources}
            summary["source_sha256_after"] = after
            summary["source_unchanged"] = before == after
            if not summary["source_unchanged"]:
                summary.update(status="review_stop", error="Smoke sources changed")
            g.save(root / "qualification.json", summary)
        self.assertTrue(summary["source_unchanged"])
