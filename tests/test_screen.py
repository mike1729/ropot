import importlib
import json
import unittest


screen = importlib.import_module("src.data-pipeline.stage01.screen")


def task():
    return {
        "id": 7,
        "difficulty": "introductory",
        "question": (
            "Echo the given integer.\n-----Input-----\nOne integer.\n"
            "-----Output-----\nPrint one integer.\n"
            "-----Examples-----\nInput\n1\n\nOutput\n1\n"
        ),
        "url": "https://example.com/task/7",
        "starter_code": "",
        "input_output": json.dumps({"inputs": ["1\n", "2\n"], "outputs": ["1\n", "2\n"]}),
        "solutions": json.dumps([
            "print(input())\n",
            "import sys\nprint(sys.stdin.readline().strip())\n",
            "value = input()\nprint(value)\n",
        ]),
    }


class ScreeningTests(unittest.TestCase):
    def reject(self, row, reason):
        with self.assertRaises(screen.Rejected) as caught:
            screen.screen_task(row, "test")
        self.assertEqual(str(caught.exception), reason)

    def test_decodes_without_changing_original_content(self):
        row = task()
        original = dict(row)
        result = screen.screen_task(row, "test")
        self.assertEqual(row, original)
        self.assertEqual(result["cases"][0], {"index": 0, "input": "1\n", "output": "1\n", "answer": "1"})
        self.assertEqual(result["solutions"], json.loads(row["solutions"]))
        self.assertEqual([r["index"] for r in result["reference_candidates"]], [0, 1, 2])
        self.assertNotIn("qualified", result)

    def test_ids_remain_split_scoped(self):
        self.assertEqual(screen.screen_task(task(), "train")["split"], "train")
        with self.assertRaises(ValueError) as caught:
            screen.screen_task(task(), "validation")
        self.assertNotIsInstance(caught.exception, screen.Rejected)
        row = task()
        row["id"] = True
        with self.assertRaises(ValueError) as caught:
            screen.screen_task(row, "test")
        self.assertNotIsInstance(caught.exception, screen.Rejected)

    def test_supported_scope(self):
        row = task()
        row["difficulty"] = "competition"
        self.reject(row, "unsupported_difficulty")
        row = task()
        row["input_output"] = json.dumps({"inputs": ["1"], "outputs": ["1"], "fn_name": "solve"})
        self.reject(row, "function_call_task")

    def test_canonical_integer_tokens_preserve_spelling(self):
        for answer in ("0", "-0", "1", "-123", "9" * 4897, "1" * 5000):
            with self.subTest(answer_length=len(answer)):
                self.assertEqual(screen.integer_answer(" \n" + answer + "\t"), answer)
        for output in ("00000", "01010", "+1", "-01", "1 2", "1\n2", "1.0", "١", "", "--1"):
            with self.subTest(output=output), self.assertRaises(screen.Rejected):
                screen.integer_answer(output)

    def test_one_noncanonical_output_excludes_whole_task(self):
        row = task()
        row["input_output"] = json.dumps({"inputs": ["1", "2"], "outputs": ["1", "00000"]})
        self.reject(row, "noncanonical_integer_output")

    def test_program_output_uses_exact_single_token_equality(self):
        self.assertTrue(screen.output_matches(" \n0\t", "0"))
        self.assertTrue(screen.output_matches("9" * 5000 + "\n", "9" * 5000))
        for output in ("00000", "+0", "-0", "0 0", "answer: 0", ""):
            with self.subTest(output=output):
                self.assertFalse(screen.output_matches(output, "0"))

    def test_russian_headers_are_excluded(self):
        row = task()
        row["question"] = row["question"].replace("-----Input-----", "-----Входные данные-----")
        self.reject(row, "non_english_statement")

    def test_nonunique_wording_and_explicit_text_outputs(self):
        for instruction, reason in (
            ("If several answers exist, print any of them.", "nonunique_answer_wording"),
            ("Print one string — the minimum correct string.", "string_or_character_output"),
            ("Print a string of n characters.", "string_or_character_output"),
            ("Print the K-th character from the left.", "string_or_character_output"),
            ("Print the lexicographically smallest string.", "string_or_character_output"),
        ):
            row = task()
            row["question"] = row["question"].replace("Print one integer.", instruction)
            with self.subTest(instruction=instruction):
                self.reject(row, reason)
        for instruction in (
            "Print one integer. If several candidates exist, choose the smallest.",
            "Print the maximum length of the resulting string.",
            "Print the minimum total cost needed to choose strings so that Takahashi can make a palindrome, or -1 if there is no such choice.",
            "Print minimum possible ambiguity of the statement after Vasya deletes some (possibly zero) characters so the resulting statement is easy.",
            "Print the number of strings.",
            "Print the answer\n\nThe string is s.",
        ):
            row = task()
            row["question"] = row["question"].replace("Print one integer.", instruction)
            with self.subTest(instruction=instruction):
                screen.screen_task(row, "test")

    def test_multiple_integer_answers_will_be_accepted_wording(self):
        row = task()
        row["question"] = row["question"].replace(
            "Print one integer.",
            "Print an integer x. If there are multiple such integers, any of them will be accepted.",
        )
        self.reject(row, "nonunique_answer_wording")
        row["question"] = row["question"].replace(
            "If there are multiple such integers, any of them will be accepted.",
            "Only the minimum integer will be accepted.",
        )
        screen.screen_task(row, "test")

    def test_singular_and_plural_output_specifications_are_checked(self):
        for heading in ("Output", "Outputs"):
            row = task()
            row["question"] = row["question"].replace("-----Output-----", f"-----{heading}-----")
            with self.subTest(heading=heading):
                screen.screen_task(row, "test")
                row["question"] = row["question"].replace("Print one integer.", "Print a string.")
                self.reject(row, "string_or_character_output")

    def test_missing_or_empty_output_specification_is_rejected(self):
        for replacement in ("", "-----Sample Output-----", "-----Print-----"):
            row = task()
            row["question"] = row["question"].replace("-----Output-----", replacement)
            with self.subTest(replacement=replacement):
                self.reject(row, "missing_output_section")
        row = task()
        row["question"] = row["question"].replace("Print one integer.", "")
        self.reject(row, "missing_output_section")

    def test_case_and_interface_failures_precede_missing_output_header(self):
        for data, reason in (
            ("", "missing_cases"),
            (json.dumps({"fn_name": "solve", "inputs": ["1"], "outputs": ["1"]}), "function_call_task"),
        ):
            row = task()
            row["input_output"] = data
            row["question"] = row["question"].replace("-----Output-----", "")
            with self.subTest(reason=reason):
                self.reject(row, reason)

    def test_line_lists_are_unsupported_rather_than_malformed(self):
        row = task()
        row["input_output"] = json.dumps({"inputs": [["1", ""]], "outputs": [["1"]]})
        self.reject(row, "unsupported_line_list_cases")

    def test_malformed_fields_and_cases(self):
        for field, value, reason in (
            ("question", "", "invalid_question"),
            ("input_output", "{broken", "invalid_input_output"),
            ("input_output", "[]", "invalid_input_output"),
            ("input_output", '{"inputs": [], "outputs": []}', "invalid_cases"),
            ("input_output", '{"inputs": ["1"], "outputs": []}', "invalid_cases"),
            ("input_output", '{"inputs": ["1"], "outputs": [true]}', "invalid_cases"),
            ("solutions", "{broken", "invalid_solutions"),
            ("solutions", '["print(1)", 1]', "invalid_solutions"),
            ("solutions", "[]", "insufficient_distinct_solutions"),
        ):
            row = task()
            row[field] = value
            with self.subTest(field=field, value=value):
                self.reject(row, reason)

    def test_missing_serialized_fields_have_separate_exclusion_reasons(self):
        for field, reason in (("input_output", "missing_cases"), ("solutions", "missing_solutions")):
            for value in (None, "", " \t\n"):
                row = task()
                row[field] = value
                with self.subTest(field=field, value=value):
                    self.reject(row, reason)
            row = task()
            del row[field]
            with self.subTest(field=field, absent=True):
                self.reject(row, reason)

    def test_token_distinctness_ignores_formatting_and_comments(self):
        source = "if True:\n    print(input())\n"
        formatted = "# comment\nif True:  # comment\n  print( input( ) )  \n\n"
        self.assertEqual(screen.token_fingerprint(source), screen.token_fingerprint(formatted))
        self.assertNotEqual(screen.token_fingerprint("if True:\n print(1)\n print(2)"), screen.token_fingerprint("if True:\n print(1)\nprint(2)"))
        self.assertNotEqual(screen.token_fingerprint("x = 1\n-2"), screen.token_fingerprint("x = 1 - 2"))

    def test_duplicate_and_unusable_sources_keep_dataset_order(self):
        row = task()
        originals = json.loads(row["solutions"])
        row["solutions"] = json.dumps([originals[0], "# duplicate\n" + originals[0], "(", "# no code\n", *originals[1:]])
        result = screen.screen_task(row, "test")
        self.assertEqual([r["index"] for r in result["reference_candidates"]], [0, 4, 5])
        self.assertEqual(result["unusable_solution_indices"], [2, 3])
        row["solutions"] = json.dumps([originals[0], "# duplicate\n" + originals[0], originals[1]])
        self.reject(row, "insufficient_distinct_solutions")

    def test_comment_only_source_does_not_count_as_third_reference(self):
        row = task()
        originals = json.loads(row["solutions"])
        for no_code in ("", "  \n", "# comment only\n"):
            row["solutions"] = json.dumps([*originals[:2], no_code])
            with self.subTest(no_code=no_code):
                self.reject(row, "insufficient_distinct_solutions")


if __name__ == "__main__":
    unittest.main()
