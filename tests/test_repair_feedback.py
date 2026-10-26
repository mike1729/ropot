"""Public feedback display limits preserve status and prioritize failure detail."""
from copy import deepcopy
import importlib
import json
import unittest
from unittest import mock

f = importlib.import_module("src.data-pipeline.stage03.repair_feedback")


def case(status="passed", text="1"):
    return dict(status=status, input=text, answer=text, stdout=text,
                stderr=None, diagnostic=None, returncode=0)


class FeedbackTests(unittest.TestCase):
    def test_exact_projection_and_determinism_without_mutation(self):
        feedback = dict(status="failed", hidden="failed", visible=[case(), case("wrong_answer", "é🙂")],
                        private="PRIVATE")
        feedback["visible"][0]["hidden_index"] = 979797
        original = deepcopy(feedback)
        result = f.render(feedback, 512, 4096)
        self.assertEqual(result, f.render(feedback, 512, 4096))
        self.assertEqual(feedback, original)
        summary, text = result["text"].split("\n", 1)
        self.assertEqual(summary, "status=failed hidden=failed")
        visible = json.loads(text)
        self.assertEqual([c["status"] for c in visible], ["wrong_answer", "passed"])
        self.assertIsNone(visible[0]["stderr"])
        self.assertNotIn("PRIVATE", result["text"])
        self.assertNotIn("979797", result["text"])

    def test_multibyte_field_caps_include_marker(self):
        feedback = dict(status="failed", hidden="not_run", visible=[case("wrong_answer", "🙂é" * 300)])
        result = f.render(feedback, 17, 4096)
        visible = json.loads(result["text"].split("\n", 1)[1])
        for key in ("input", "answer", "stdout"):
            self.assertEqual(visible[0][key], "🙂é" + f.MARKER)
            self.assertEqual(len(visible[0][key].encode()), 17)
        self.assertEqual(result["fields_truncated"], 3)

    def test_remove_passing_details_then_entries_before_failing_text(self):
        failed = case("wrong_answer", "FAILING")
        feedback = dict(status="failed", hidden="not_run", visible=[case(text="x" * 500) for _ in range(30)] + [failed])
        result = f.render(feedback, 512, 350)
        self.assertLessEqual(len(result["text"].encode()), 350)
        self.assertIn('"input":"FAILING"', result["text"])
        self.assertGreater(result["passing_details_removed"], 0)
        self.assertGreater(result["visible_omitted"], 0)
        self.assertIn(f.MARKER, result["text"])

    def test_tightest_cap_retains_summary_and_marker(self):
        feedback = dict(status="failed", hidden="not_run", visible=[case("wrong_answer", "🙂" * 100)])
        cap = len(f.LONGEST_SUMMARY + f.MARKER)
        result = f.render(feedback, 512, cap)
        self.assertEqual(result["text"], f.LONGEST_SUMMARY + f.MARKER)
        self.assertTrue(result["text_truncated"])

    def test_no_visible_cases_keeps_hidden_summary(self):
        result = f.render(dict(status="failed", hidden="failed", visible=[]), 11, 64)
        self.assertEqual(result["text"], "status=failed hidden=failed\n[]")

    def test_byte_bound_for_many_caps_and_unicode_boundaries(self):
        feedback = dict(status="failed", hidden="not_run", visible=[case(text="é🙂" * 50) for _ in range(20)] +
                        [case("timeout", "é🙂" * 100)])
        for cap in range(len(f.LONGEST_SUMMARY + f.MARKER), 1000, 7):
            result = f.render(feedback, 39, cap)
            self.assertLessEqual(len(result["text"].encode()), cap)
            self.assertTrue(result["text"].startswith(f.LONGEST_SUMMARY))

    def test_invalid_caps_are_rejected_by_renderer_and_clip(self):
        feedback = dict(status="failed", hidden="not_run", visible=[])
        for field, total in ((10, 4096), (512, 1), (True, 4096), (512, True)):
            with self.subTest(field=field, total=total), self.assertRaises(ValueError):
                f.render(feedback, field, total)
        for cap in (-1, 0, 3, 10, True):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                f.clip("x" * 100, cap)

    def test_stripping_keeps_nulls_and_does_not_count_absent_string_detail(self):
        feedback = dict(status="failed", hidden="not_run", visible=[case("wrong_answer", "FAIL"), case(text="x" * 1000)])
        result = f.render(feedback, 512, 350)
        cases = json.loads(result["text"].removeprefix(f.LONGEST_SUMMARY).removesuffix("\n" + f.MARKER))
        self.assertEqual(result["passing_details_removed"], 1)
        self.assertEqual(result["visible_omitted"], 0)
        self.assertIsNone(cases[1]["stderr"])
        self.assertIsNone(cases[1]["diagnostic"])
        self.assertNotIn("input", cases[1])
        feedback["visible"] = [dict(status="passed", diagnostic=None) for _ in range(20)]
        result = f.render(feedback, 512, 100)
        self.assertEqual(result["passing_details_removed"], 0)
        self.assertGreater(result["visible_omitted"], 0)

    def test_many_cases_are_serialized_at_most_twice_each(self):
        feedback = dict(status="failed", hidden="not_run", visible=[case(text="é🙂" * 50) for _ in range(1000)])
        with mock.patch.object(f.json, "dumps", wraps=json.dumps) as encode:
            result = f.render(feedback, 39, 512)
        self.assertLessEqual(encode.call_count, 2 * len(feedback["visible"]))
        self.assertLessEqual(len(result["text"].encode()), 512)
        self.assertGreater(result["visible_omitted"], 0)
