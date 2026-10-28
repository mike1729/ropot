"""B2 requests, public feedback boundaries and the optional loop callback."""
from copy import deepcopy
import importlib
import json
import unittest

from tests import test_benign_loop as initial, test_benign_repairs as repairs

p = importlib.import_module("src.data-pipeline.stage05.repair_search_policy")
b = initial.b


def worklist():
    return [(dict(id=i, split="train", sha256=str(i) * 64, implant=dict(
        program=f"print({i})\n", stdin=f"ORIGINAL_WITNESS_{i}\n\"é\"\n",
        answer="PRIVATE_APPROVED", provenance="PRIVATE_CONTROL")), dict(
        question=f"Statement {i}", reference="PRIVATE_REFERENCE", cases=[])) for i in range(2)]


def failed(position=0, extraction="success"):
    return dict(position=position, round=2, state="repairs_exhausted", extraction=extraction,
        verification=dict(status="failed", private="PRIVATE_VERDICT", feedback=dict(
            status="failed", hidden="failed", visible=[dict(status="wrong_answer", input="PUBLIC_CASE",
                answer="PUBLIC_ANSWER", stdout="PUBLIC_STDOUT", hidden_index="PRIVATE_HIDDEN")],
            private="PRIVATE_FEEDBACK")))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.worklist = worklist()
        self.parent_recipe = {**p.attacks.DEFAULT_RECIPE, "initial": "HISTORICAL_INITIAL"}
        self.recipe = p.recipe(self.parent_recipe)
        self.requests = b.initial_requests(self.worklist, self.parent_recipe)
        for request in self.requests:
            request["messages"] += [dict(role="assistant", content="old first reply"),
                dict(role="user", content="old feedback"), dict(role="assistant", content="old second reply"),
                dict(role="user", content="old second feedback")]
        self.rows = [dict(text="final raw reply\r\n", status="length") for _ in self.worklist]
        self.attempts = [failed(i) for i in range(len(self.worklist))]

    def test_recipe_preserves_legacy_instructions_and_fixes_new_allowances(self):
        snapshot = deepcopy(self.parent_recipe)
        self.assertEqual(self.recipe["repair"], self.parent_recipe["repair"])
        self.assertEqual(self.recipe["initial"], "HISTORICAL_INITIAL")
        self.assertEqual([self.recipe[k] for k in ("base_seed", "max_tokens", "repair_max_tokens", "max_repairs")],
                         [41, 2048, 2048, 2])
        self.assertEqual(self.parent_recipe, snapshot)

    def test_a_initial_and_subsequent_messages_match_legacy_builder_exactly(self):
        original = deepcopy((self.requests, self.rows, self.attempts))
        for category in ("success", "format", "syntax", "length"):
            self.attempts[0]["extraction"] = category
            expected = b.repair_request(self.requests[0], self.rows[0], deepcopy(self.attempts[0]), self.recipe, 41)
            actual = p.build_request(self.worklist, "A")(
                self.requests[0], self.rows[0], deepcopy(self.attempts[0]), self.recipe, 41)
            self.assertEqual(actual, expected)
        self.attempts[0]["extraction"] = "success"
        histories = p.initial_histories(self.worklist, "A", self.recipe, self.requests, self.rows, self.attempts)
        self.assertEqual(histories[0][:6], self.requests[0]["messages"])
        self.assertEqual(histories[0][6], dict(role="assistant", content=self.rows[0]["text"]))
        self.assertEqual(len(histories[0]), 8)
        self.assertEqual((self.requests, self.rows, self.attempts), original)

    def test_b_distinguishes_original_witness_and_public_case_and_projects_only_public_fields(self):
        histories = p.initial_histories(self.worklist, "B", self.recipe, self.requests, self.rows, self.attempts)
        text = histories[0][-1]["content"]
        self.assertTrue(text.startswith("Revise the preceding candidate.\n\n" + p.OBJECTIVE))
        self.assertIn(p.FAMILY, text)
        self.assertIn("Original witness stdin (JSON escaped):\n" +
                      json.dumps(self.worklist[0][0]["implant"]["stdin"], ensure_ascii=True), text)
        self.assertIn(p.SUITE_FAILURE, text)
        self.assertIn("PUBLIC_CASE", text)
        self.assertIn("PUBLIC_ANSWER", text)
        self.assertIn("hidden=failed", text)
        self.assertNotIn("PRIVATE", json.dumps(histories))
        self.assertNotIn("ORIGINAL_WITNESS_1", json.dumps(histories[0]))
        self.assertEqual(histories[0][:6], self.requests[0]["messages"])
        self.assertEqual(histories[0][-2]["content"], self.rows[0]["text"])

    def test_witness_rejection_subtypes_share_one_fixed_block(self):
        blocks = []
        for subtype in ("same_integer", "invalid_output", "timeout", "nonzero_exit"):
            parent = failed()
            parent["verification"].update(suite=dict(status="passed"),
                witness=dict(answer="PRIVATE_ANSWER", observed="PRIVATE_CAPTURE", status=subtype),
                feedback=dict(message="PRIVATE_UNTRUSTED_MESSAGE"))
            blocks.append(p.explicit_feedback(parent, self.recipe)["text"])
        self.assertEqual(blocks, [p.WITNESS_FAILURE] * 4)

    def test_extraction_rejections_have_exact_safe_blocks(self):
        for subtype, expected in p.LOCAL_FAILURES.items():
            parent = failed(extraction=subtype)
            parent["error"] = "PRIVATE_TRACEBACK"
            self.assertEqual(p.explicit_feedback(parent, self.recipe), dict(text=expected))
        with self.assertRaises(ValueError):
            p.explicit_feedback(failed(extraction="compiler_resource"), self.recipe)

    def test_public_feedback_caps_apply_before_fixed_labels(self):
        parent = failed()
        parent["verification"]["feedback"]["visible"] *= 20
        for case in parent["verification"]["feedback"]["visible"]:
            case["stdout"] = "é🙂" * 1000
        rendered = p.explicit_feedback(parent, self.recipe)
        body = rendered["text"].removeprefix(p.SUITE_FAILURE + "\n\n")
        self.assertLessEqual(len(body.encode()), 4096)
        self.assertIn("hidden=failed", body)
        self.assertTrue(rendered["fields_truncated"])
        self.assertNotIn("PRIVATE", body)

    def test_c_resets_to_pinned_original_on_every_draw_and_never_reads_failure(self):
        fresh = p.initial_histories(self.worklist, "C", self.recipe)
        builder = p.build_request(self.worklist, "C", fresh)
        changed = builder(self.requests[1], None, dict(position=1), self.recipe, 50)
        self.assertEqual(changed["messages"], fresh[1])
        self.assertEqual(len(changed["messages"]), 2)
        self.assertEqual(changed["seed"], 50)
        self.assertEqual(changed["max_tokens"], 2048)
        self.assertIn("Your original program:\nprint(1)", fresh[1][1]["content"])
        self.assertIn("Produce a new candidate from the original program.", fresh[1][1]["content"])
        self.assertNotIn("PRIVATE", json.dumps(changed))
        self.assertNotIn("old first reply", json.dumps(changed))
        changed["messages"][1]["content"] = "a mutation"
        again = builder(changed, None, dict(position=1), self.recipe, 58)
        self.assertEqual(again["messages"], fresh[1])

    def test_parent_projection_requires_aligned_final_failed_round(self):
        for field, value in (("position", 1), ("round", 1), ("state", "passed")):
            changed = deepcopy(self.attempts)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                p.initial_histories(self.worklist, "B", self.recipe, self.requests, self.rows, changed)
        with self.assertRaises(ValueError):
            p.initial_histories(self.worklist, "A", self.recipe, self.requests[:1], self.rows, self.attempts)
        with self.assertRaises(ValueError):
            p.build_request(self.worklist, "unknown")

    def test_c_refuses_templates_with_any_added_history_or_changed_witness(self):
        templates = p.fresh_messages(self.worklist, self.recipe)
        history, witness = deepcopy(templates), deepcopy(templates)
        history[0] += [dict(role="assistant", content="PRIVATE_REPLY"), dict(role="user", content="Feedback")]
        witness[0][1]["content"] = witness[0][1]["content"].replace("ORIGINAL_WITNESS_0", "DIFFERENT_WITNESS")
        for changed in (history, witness):
            builder = p.build_request(self.worklist, "C", changed)
            with self.assertRaises(ValueError):
                builder(self.requests[0], None, dict(position=0), self.recipe, 49)

    def test_eight_task_seed_stride_survives_a_shrinking_round(self):
        parents = [dict(attempt=i, round=0, position=i, state="failed" if i in (0, 7) else "passed")
                   for i in range(8)]
        progress = dict(attempts=parents[:], committed_replies=8, committed_tokens=8 * 2048)
        calls = []
        def builder(request, row, parent, recipe, seed):
            calls.append((parent["position"], seed))
            return dict(seed=seed, max_tokens=2048)
        _, attempts = b.prepare_repairs(parents, [{}] * 8, [{}] * 8, progress,
            self.recipe, 8, 24, 24 * 2048, lambda: None, builder)
        self.assertEqual(calls, [(0, 49), (7, 56)])
        self.assertEqual([attempt["parent"] for attempt in attempts], [0, 7])


@unittest.skipIf(initial.v is None, "requires requirements-execution.txt")
class LoopHookTests(initial.LoopFixture):
    def setUp(self):
        super().setUp()
        self.recipe = p.recipe(p.attacks.DEFAULT_RECIPE)
        for i, (binding, _) in enumerate(self.worklist):
            binding["implant"] = worklist()[i][0]["implant"]
        self.draws, self.verdicts, self.requests = {}, {}, []
        self.round_index = 0

    def generate(self, command, current, persist):
        # The receipt must contain the exact request-file hash before dispatch.
        saved = self.saved()
        self.assertEqual(saved["rounds"][-1]["round_file_sha256"], b.generation.file_hash(command[-1]))
        return repairs.RepairTests.generate(self, command, current, persist)

    def verify(self, candidates, directory, **kwargs):
        results = super().verify(candidates, directory, **kwargs)
        for result, (_, digest, _) in zip(results, candidates):
            position = next(i for i, (binding, _) in enumerate(self.worklist) if binding["sha256"] == digest)
            status = self.verdicts.get((self.round_index, position), "passed")
            result.update(status=status, private="PRIVATE_VERDICT", feedback=dict(
                status=status, hidden="failed" if status == "failed" else "passed", visible=[]))
        return results

    def run_policy(self, arm):
        return self.run_loop(initial_messages=p.fresh_messages(self.worklist, self.recipe),
            repair_builder=p.build_request(self.worklist, arm), output_token_allowance=6 * 2048)

    def test_fresh_policy_stable_positions_stopping_and_reservations(self):
        self.verdicts[0, 1] = "failed"
        result = self.run_policy("C")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["position"] for a in result["attempts"]], [0, 1, 1])
        self.assertEqual([a["seed"] for a in result["attempts"]], [41, 42, 44])
        self.assertEqual([a["parent"] for a in result["attempts"]], [None, None, 1])
        self.assertEqual(self.requests[0][1]["messages"], self.requests[1][0]["messages"])
        self.assertTrue(all(len(request["messages"]) == 2 for requests in self.requests for request in requests))
        self.assertEqual(result["committed_tokens"], 3 * 2048)
        self.assertEqual([task["state"] for task in result["tasks"]], ["passed", "passed"])
        self.assertEqual(self.child.call_count, 2)

    def test_explicit_policy_retains_all_previous_turns_through_last_draw(self):
        self.verdicts.update({(r, 1): "failed" for r in range(3)})
        result = self.run_policy("B")
        self.assertEqual(result["status"], "completed", result)
        self.assertEqual([a["seed"] for a in result["attempts"]], [41, 42, 44, 46])
        self.assertEqual([len(requests[0]["messages"]) for requests in self.requests], [2, 4, 6])
        self.assertEqual(self.requests[2][0]["messages"][:4], self.requests[1][0]["messages"])
        self.assertEqual(result["committed_tokens"], 4 * 2048)
        self.assertEqual([task["state"] for task in result["tasks"]], ["passed", "repairs_exhausted"])
        self.assertNotIn("PRIVATE", json.dumps(self.requests))

    def test_invalid_hook_stops_before_preparation_or_dispatch(self):
        with self.assertRaises(ValueError):
            self.run_loop(repair_builder="not callable", output_token_allowance=6 * 2048)
        self.child.assert_not_called()
        self.assertFalse((self.root / "run").exists())
