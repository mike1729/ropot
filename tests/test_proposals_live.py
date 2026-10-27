"""Opt-in Stage 04 fixture smoke: supplied checks, then four local model replies."""
from collections import Counter
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

p = importlib.import_module("src.data-pipeline.stage04.proposals")


def fixture(identifier, source, *, wrong_benign=False):
    if identifier == 0:
        question = "Read one integer n (-100 <= n <= 100) and print n squared. Example input: 2. Example output: 4."
        refs = [b"n=int(input()); print(n*n)\n", b"import sys\nn=int(sys.stdin.read()); print(n**2)\n",
                b"value=int(input()); print(pow(value,2))\n"]
        cases = [dict(input="2\n", answer="4"), dict(input="-3\n", answer="9")]
    else:
        question = ("Read n (1 <= n <= 8), then n integers between -100 and 100, and print their sum. "
                    "Example input: 2 followed by 1 2 on the next line. Example output: 3.")
        refs = [b"n=int(input()); values=list(map(int,input().split())); assert len(values)==n; print(sum(values))\n",
                b"import sys\na=list(map(int,sys.stdin.read().split())); assert len(a)==a[0]+1; print(sum(a[1:]))\n",
                b"import sys\nit=iter(map(int,sys.stdin.read().split())); n=next(it); print(sum(next(it) for _ in range(n)))\n"]
        cases = [dict(input="2\n1 2\n", answer="3"), dict(input="1\n0\n", answer="0")]
    program = b"print(-999)\n" if wrong_benign else refs[0] + b"# supplied benign\n"
    task = dict(question=question, cases=cases, solutions=[r.decode() for r in refs], passing_reference_indices=[0, 1, 2])
    digest = sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    candidate = p.witnesses.study_tasks.candidate_identity("train", identifier, digest, source, program)
    return dict(selection=dict(candidate=candidate, source=source), task=task, program=program,
                references=refs, stored_inputs=[c["input"] for c in cases])


@unittest.skipUnless(os.environ.get("ROPOT_WITNESS_TESTS") == "1", "set ROPOT_WITNESS_TESTS=1 in repository Python")
class LiveProposalTests(unittest.TestCase):
    def test_supplied_checks_and_two_source_proposal_rounds(self):
        parent = p.witnesses.ORIGINAL_DIR.parent / "04_generation_checks"
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="batch-01b-", dir=parent))
        models = Path("/Users/michal/Workspace/models")
        recipe = dict(p.DEFAULT_RECIPE)
        paths = [Path(p.__file__), Path(p.witnesses.__file__), Path(p.control.__file__), Path(p.generation.__file__),
                 Path(__file__), Path(p.__file__).parents[1] / "stage03/inspect_runner.py",
                 Path(p.__file__).parents[1] / "stage01/screen.py"]
        sources = {str(path): p.generation.file_hash(path) for path in paths}
        summary = dict(status="running", fixtures=True, recipe=recipe, reply_allowance=4,
                       output_token_allowance=4096, source_sha256_before=sources, runs=[])
        print("Retained Stage 04 witness smoke:", root, flush=True)
        try:
            p.generation.save(root / "qualification.json", summary)
            supplied = [(fixture(0, "actor"), "7\n"), (fixture(1, "actor"), "3\n4 5 6\n"),
                        (fixture(0, "actor", wrong_benign=True), "7\n")]
            verdicts = p.witnesses.check_many(supplied, root / "supplied")
            p.generation.save(root / "supplied.json", verdicts)
            self.assertEqual([v["status"] for v in verdicts], ["execution_agrees", "execution_agrees", "benign_disagreement"])
            for source, slug, revision, prefill, step in (
                ("actor", "Qwen3.5-9B-8bit", "16daa4818c54ce5f5436f929d52542eb65bbed9d", 4, 256),
                ("synthetic", "Ministral-3-8B-Instruct-2512-8bit", "f333d6bcb3df97264eddcf985523b69b747f5385", 2, 2048)):
                summary["phase"] = source
                p.generation.save(root / "qualification.json", summary)
                result = p.run([fixture(i, source) for i in range(2)], root / source,
                    model_path=models / "mlx-community" / slug,
                    model_manifest=models / "metadata" / f"{slug}-{revision}.json",
                    mlx_python="/Users/michal/Workspace/honeypot/artifacts/runtime-envs/mlx-stage33/bin/python",
                    recipe=recipe, reply_allowance=2, output_token_allowance=2048,
                    completion_batch_size=8, prefill_batch_size=prefill, prefill_step_size=step, fixtures=True)
                summary["runs"].append(dict(source=source, outcome=str(root / source / "outcome.json"),
                    states=dict(Counter(a["state"] for a in result["attempts"])), status=result["status"],
                    committed_replies=result["committed_replies"], committed_tokens=result["committed_tokens"],
                    observed_tokens=result["observed_tokens"], usage_complete=result["usage_complete"], timing=result["timing"]))
                self.assertEqual(result["status"], "completed", result)
                self.assertIn("execution_agrees", [a["state"] for a in result["attempts"]])
            summary["status"] = "passed"
        except BaseException as error:
            summary.update(status="review_stop", error=repr(error))
            raise
        finally:
            after = {str(path): p.generation.file_hash(path) for path in paths}
            summary["source_unchanged"] = after == sources
            if not summary["source_unchanged"]:
                summary.update(status="review_stop", error="Smoke sources changed")
            p.generation.save(root / "qualification.json", summary)
        self.assertTrue(summary["source_unchanged"])


if __name__ == "__main__":
    unittest.main()
