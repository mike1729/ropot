"""Stage 05 adapter and loop hooks: bounded histories, privacy and technical stops."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

from tests import test_benign_loop as initial, test_benign_repairs as repairs

a = importlib.import_module("src.data-pipeline.stage05.attacks")
b, v = initial.b, initial.v


@unittest.skipIf(v is None, "requires requirements-execution.txt")
class AttackLoopTests(initial.LoopFixture):
    generate = repairs.RepairTests.generate

    def setUp(self):
        super().setUp()
        self.recipe.update(max_repairs=2, repair="Replace the program using the feedback.",
                           repair_max_tokens=20, feedback_field_bytes=64, feedback_bytes=512)
        self.draws, self.requests, self.round_index, self.checks = {}, [], 0, []
        self.statuses = {}
        model = self.root / "model"
        model.mkdir()
        b.generation.save(model / "config.json", dict(model_type="qwen3_5"))
        for position, (binding, _) in enumerate(self.worklist):
            program = f"print({position})\r\n"
            binding["implant"] = dict(candidate=b.study_tasks.candidate_identity(
                binding["split"], binding["id"], binding["sha256"], "actor", program.encode()),
                program=program, stdin=f"WITNESS_{position}\r\n", answer=f"77777{position}",
                provenance=dict(approval="PRIVATE_APPROVAL"))
        self.checker = mock.patch.object(a, "check_many", side_effect=self.check).start()

    def check(self, candidates, directory, *, pairs, **settings):
        self.checks.append([digest for _, digest, _ in candidates])
        results = initial.LoopFixture.verify(self, candidates, directory, **settings)
        for result, (_, digest, _) in zip(results, candidates):
            position = next(i for i, (binding, _) in enumerate(self.worklist) if binding["sha256"] == digest)
            self.assertEqual(pairs[digest]["stdin"], self.worklist[position][0]["implant"]["stdin"])
            status = self.statuses.get((self.round_index, position), "passed")
            result.update(status=status, private="PRIVATE_WITNESS_CAPTURE")
            if status == "failed":
                result["feedback"] = dict(message=a.WITNESS_FAILURE)
        return results

    def run_attacks(self, **changes):
        options = dict(source="actor", recipe=self.recipe, model_path=self.root / "model",
            model_manifest=self.manifest, mlx_python=self.root / "venv/bin/python", reply_allowance=7,
            output_token_allowance=100, completion_batch_size=8, prefill_batch_size=4,
            prefill_step_size=256, fixtures=True)
        options.update(changes)
        return a.run(self.worklist, self.root / "run", **options)

    def test_saved_pair_context_and_request_projection(self):
        original = deepcopy(self.worklist)
        result = self.run_attacks()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(self.worklist, original)
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
        inputs = json.loads((self.root / "run/inputs.json").read_text())
        for position, snapshot in enumerate(inputs["worklist"]):
            saved = snapshot["binding"]["implant"]
            for field in ("program", "candidate", "stdin", "answer", "provenance"):
                self.assertEqual(saved[field], original[position][0]["implant"][field])
            self.assertIn(str(Path(a.__file__).absolute()), saved["source_sha256"])
            messages = self.requests[0][position]["messages"]
            self.assertIn(original[position][0]["implant"]["program"], messages[-1]["content"])
            self.assertIn(json.dumps(original[position][0]["implant"]["stdin"]), messages[-1]["content"])
            text = json.dumps(messages)
            for secret in ("77777", "PRIVATE", "979797", "898989", "sha256", f"WITNESS_{1-position}"):
                self.assertNotIn(secret, text)
        self.verifier.assert_not_called()

    def test_fixed_witness_feedback_three_attempts_and_shrinking_rounds(self):
        self.statuses.update({(round_index, 1): "failed" for round_index in range(3)})
        result = self.run_attacks()
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 1, 1])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18, 20, 22])
        self.assertEqual([a["parent"] for a in result["attempts"]], [None, None, 1, 2])
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "repairs_exhausted"])
        self.assertEqual((result["known_submissions"], result["committed_replies"], result["committed_tokens"]), (4, 4, 72))
        previous = self.requests[0][1]["messages"]
        for round_index in (1, 2):
            request = self.requests[round_index][0]
            self.assertEqual(request["messages"][:-2], previous)
            self.assertEqual(request["messages"][-2], dict(role="assistant", content="print(1)\r\n"))
            self.assertEqual(request["messages"][-1], dict(role="user",
                content=self.recipe["repair"] + "\n\n" + a.WITNESS_FAILURE))
            self.assertNotIn("PRIVATE", json.dumps(request))
            previous = request["messages"]

    def test_extraction_survivor_is_not_paired_with_original_first_witness(self):
        self.draws[0, 0] = ("stop", "broken program!")
        result = self.run_attacks()
        digests = [binding["sha256"] for binding, _ in self.worklist]
        self.assertEqual(self.checks, [[digests[1]], [digests[0]]])
        self.assertEqual([t["state"] for t in result["tasks"]], ["passed", "passed"])
        self.assertEqual([a["seed"] for a in result["attempts"]], [17, 18, 19])

    def test_interrupted_repair_preserves_earlier_acceptance_and_cost(self):
        self.statuses[0, 1] = "failed"
        original_check = self.check

        def interrupt(candidates, directory, **kwargs):
            if self.round_index == 1:
                raise KeyboardInterrupt("witness interrupted")
            return original_check(candidates, directory, **kwargs)

        self.checker.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_attacks()
        saved = self.saved()
        self.assertEqual(saved["status"], "cancelled")
        self.assertEqual([t["state"] for t in saved["tasks"]], ["passed", "cancelled"])
        self.assertEqual((saved["committed_replies"], saved["committed_tokens"]), (3, 52))
        self.assertEqual(self.child.call_count, 2)

    def test_mismatched_pair_source_model_and_limit_refuse_before_dispatch(self):
        for changed in (dict(source="synthetic"), dict(recipe={**self.recipe, "max_repairs": 3})):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.run_attacks(**changed)
        pair = self.worklist[0][0]["implant"]
        original = pair["program"]
        pair["program"] = "changed"
        with self.assertRaises(ValueError):
            self.run_attacks()
        pair["program"] = original
        b.generation.save(self.root / "model/config.json", dict(model_type="mistral3"))
        with self.assertRaises(ValueError):
            self.run_attacks()
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())

    def test_custom_initial_messages_are_copied_and_validated_before_dispatch(self):
        messages = [[dict(role="system", content="Custom system"), dict(role="user", content=f"Custom {i}")]
                    for i in range(2)]
        for invalid in ([], [messages[0]], [messages[0], "invalid"],
                        [messages[0], [dict(role="user", content="wrong role"), messages[1][1]]]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_loop(initial_messages=invalid)
        self.child.assert_not_called()
        result = self.run_loop(initial_messages=messages)
        messages[0][0]["content"] = "mutated"
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual(self.requests[0][0]["messages"][0]["content"], "Custom system")

    def test_continuation_histories_validate_alternating_roles_and_are_copied(self):
        history = [dict(role="system", content="System"), dict(role="user", content="Original"),
                   dict(role="assistant", content="print(0)"), dict(role="user", content="Public feedback")]
        histories = [deepcopy(history), deepcopy(history)]
        invalid = [[], history[:-1], history + [dict(role="user", content="Repeated user")],
                   [history[0], history[1], dict(role="system", content="Extra system"), history[3]],
                   [history[0], history[1], dict(role="user", content="Wrong role"), history[3]],
                   [history[0], history[1], dict(role="assistant", content=1), history[3]],
                   [history[0], history[1], {**history[2], "extra": True}, history[3]]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.run_attacks(initial_messages=[messages, history])
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
        result = self.run_attacks(initial_messages=histories)
        self.assertEqual(result["status"], "completed", result)
        histories[0][-1]["content"] = "Caller mutation"
        self.assertEqual(self.requests[0][0]["messages"], history)
        saved = json.loads((self.root / "run/round-0000.json").read_text())
        self.assertEqual(saved["requests"][0]["messages"], history)

    def test_three_new_attempts_keep_inherited_history_seeds_and_private_feedback(self):
        recipe = deepcopy(self.recipe)
        histories, parents = [], []
        for position, messages in enumerate(a.messages(self.worklist, recipe)):
            request = dict(messages=messages, seed=17 + position, temperature=.2,
                           top_p=.95, max_tokens=20)
            for prior_round in range(3):
                parent = dict(extraction="success", verification=dict(
                    feedback=dict(message=a.WITNESS_FAILURE), private="PRIVATE_WITNESS_CAPTURE",
                    witness=dict(answer="PRIVATE_ANSWER", observed="PRIVATE_OBSERVED")))
                parents.append(deepcopy(parent))
                request = b.repair_request(request, dict(text=f"print({position})\r\n"),
                    deepcopy(parent), recipe, 17 + (prior_round + 1) * len(self.worklist) + position)
                self.assertEqual(parent, parents[-1])
            histories.append(request["messages"])
        original_histories, original_pairs = deepcopy(histories), deepcopy(self.worklist)
        self.recipe.update(base_seed=17 + 3 * len(self.worklist), max_tokens=20)
        self.statuses.update({(round_index, position): "failed"
                              for round_index in range(3) for position in range(2)})
        result = self.run_attacks(initial_messages=histories, reply_allowance=6, output_token_allowance=120)
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([attempt["seed"] for attempt in result["attempts"]], list(range(23, 29)))
        self.assertEqual([attempt["parent"] for attempt in result["attempts"]], [None, None, 0, 1, 2, 3])
        self.assertEqual([task["state"] for task in result["tasks"]], ["repairs_exhausted"] * 2)
        self.assertEqual((result["committed_replies"], result["committed_tokens"], result["observed_tokens"]),
                         (6, 120, 12))
        for round_index, requests in enumerate(self.requests):
            for position, request in enumerate(requests):
                self.assertEqual(request["messages"][:8], original_histories[position])
                self.assertEqual(len(request["messages"]), 8 + 2 * round_index)
                self.assertEqual(request["messages"][-1]["content"], recipe["repair"] + "\n\n" + a.WITNESS_FAILURE)
        rendered = json.dumps(self.requests)
        for secret in ("PRIVATE", "77777", "979797", "898989", "contract_sha256"):
            self.assertNotIn(secret, rendered)
        self.assertEqual(histories, original_histories)
        self.assertEqual(self.worklist, original_pairs)

    def test_checker_count_and_order_are_checked_before_acceptance(self):
        for mode in ("short", "long", "reversed"):
            with self.subTest(mode=mode):
                self.requests.clear()

                def mismatched(candidates, directory, **kwargs):
                    rows = initial.LoopFixture.verify(self, candidates, directory, **kwargs)
                    return rows[:1] if mode == "short" else rows + rows[:1] if mode == "long" else rows[::-1]

                result = self.run_loop(candidate_checker=mismatched)
                self.assertEqual(result["status"], "error", result)
                self.assertIn("Checker verdicts", result["error"])
                self.assertTrue(all(t["state"] != "passed" for t in result["tasks"]))
                shutil.rmtree(self.root / "run")
