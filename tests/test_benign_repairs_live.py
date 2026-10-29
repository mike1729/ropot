"""7b's bounded two-model gate: verified failure, public repair and verification."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from tests.test_benign_loop_live import b, MODELS, MLX_PYTHON
from tests.test_batched_generation_live import fixture


def assert_repair_coverage(test, result, directory, worklist, recipe):
    """Check actual saved associations/costs; candidate quality is not this gate."""
    test.assertEqual(result, json.loads((Path(directory) / "outcome.json").read_text()))
    test.assertEqual(result["status"], "completed", result)
    source = json.loads((Path(directory) / "inputs.json").read_text())["source"]
    attempts, requests, coverage = result["attempts"], {}, []
    test.assertLessEqual(len(attempts), len(worklist) * (1 + recipe["max_repairs"]))
    test.assertEqual([a["attempt"] for a in attempts], list(range(len(attempts))))
    initial = b.initial_requests(worklist, recipe)
    reserved_tokens = observed_tokens = 0
    for round_record in result["rounds"]:
        specification = json.loads(Path(round_record["round_file"]).read_text())
        native = Path(round_record["generation_outcome"])
        outcome = json.loads(native.read_text())
        inputs = json.loads((native.parent / "inputs.json").read_text())
        test.assertEqual(outcome["status"], "success")
        test.assertEqual(inputs["requests"], specification["requests"])
        sample = 0
        for row_index, (attempt_id, request, row) in enumerate(zip(
                round_record["attempt_ids"], specification["requests"], outcome["requests"], strict=True)):
            attempt = attempts[attempt_id]
            position = attempt["position"]
            test.assertEqual(attempt["round"], round_record["round"])
            test.assertEqual(attempt["row"], row_index)
            test.assertEqual(attempt["seed"], recipe["base_seed"] + attempt["round"] * len(worklist) + position)
            test.assertTrue(attempt["submitted"])
            test.assertTrue(row["submitted"])
            test.assertEqual(attempt["generation_status"], row["status"])
            test.assertIn(row["status"], ("stop", "length"))
            if attempt["parent"] is None:
                test.assertEqual(attempt["round"], 0)
                expected = initial[position]
            else:
                parent = attempts[attempt["parent"]]
                test.assertEqual(parent["position"], position)
                test.assertEqual(attempt["round"], parent["round"] + 1)
                copied = deepcopy(parent)
                expected = b.repair_request(initial[position], copied, recipe, attempt["seed"])
                test.assertEqual(parent["repair_feedback"], copied["repair_feedback"])
            test.assertEqual(request, expected)
            requests[attempt_id] = request
            reserved_tokens += request["max_tokens"]
            observed_tokens += row["generation_tokens"]
            test.assertLessEqual(row["generation_tokens"], request["max_tokens"])
            test.assertIn(attempt["extraction"], ("success", "length", "format", "syntax"))
            if attempt["extraction"] == "success":
                test.assertEqual(row["status"], "stop")
                program = Path(attempt["program"]).read_bytes()
                test.assertEqual(program, b.generation.extract_program(row))
                binding, _ = worklist[position]
                test.assertEqual(attempt["candidate"], b.study_tasks.candidate_identity(
                    binding["split"], binding["id"], binding["sha256"], source, program))
                verdict = attempt["verification"]
                test.assertIn(verdict["status"], ("passed", "failed"))
                test.assertEqual(verdict["sample_id"], sample)
                test.assertEqual(verdict["key"]["task_sha256"], binding["sha256"])
                test.assertEqual(verdict["key"]["program_sha256"], attempt["candidate"]["program_sha256"])
                sample += 1
                if attempt["parent"] is not None:
                    parent = attempts[attempt["parent"]]
                    if (parent["round"] == 0 and parent["extraction"] == "success"
                            and parent.get("verification", {}).get("status") == "failed"):
                        coverage.append([parent["attempt"], attempt_id])
    test.assertEqual(set(requests), set(range(len(attempts))))
    test.assertTrue(coverage, "No verified initial failure followed by an extracted, verified public-feedback repair")
    test.assertEqual(result["known_submissions"], len(attempts))
    test.assertEqual(result["committed_replies"], len(attempts))
    test.assertEqual(result["committed_tokens"], reserved_tokens)
    test.assertEqual(result["observed_tokens"], observed_tokens)
    test.assertTrue(result["usage_complete"])
    return coverage


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
            repair="The FIRST DRAFT CONTROL has ended. Replace the program with a correct solution to the original task, using the public feedback below.",
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
                run = dict(source=source, outcome=str(root / source / "outcome.json"),
                    status=result["status"], timing=result["timing"], observed_tokens=result["observed_tokens"],
                    committed_replies=result["committed_replies"], committed_tokens=result["committed_tokens"])
                summary["runs"].append(run)
                run["coverage_pairs"] = assert_repair_coverage(self, result, root / source, worklist, recipe)
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
