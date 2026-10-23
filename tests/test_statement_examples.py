import copy
import importlib
import json
import unittest

from test_screen import task


examples = importlib.import_module("src.data-pipeline.statement_examples")
screen = importlib.import_module("src.data-pipeline.screen")


def candidate(statement=None, inputs=None, outputs=None):
    row = task()
    if statement is not None:
        row["question"] = row["question"].split("-----Examples-----")[0] + statement
    if inputs is not None:
        row["input_output"] = json.dumps({"inputs": inputs, "outputs": outputs})
    return screen.screen_task(row, "test")


class ExampleTests(unittest.TestCase):
    def reject(self, value, reason):
        with self.assertRaises(screen.Rejected) as caught:
            examples.map_statement_examples_to_cases(value)
        self.assertEqual(str(caught.exception), reason)

    def test_codeforces_maps_all_examples_and_excludes_notes(self):
        value = candidate(
            "-----Examples-----\nInput\n1\n\nOutput\n1\n\n"
            "Input\n2\n\nOutput\n2\n\n-----Note-----\nInput and Output are explained here.",
            ["1\n", "2\n", "3\n"], ["1\n", "2\n", "3\n"],
        )
        original = copy.deepcopy(value)
        result = examples.map_statement_examples_to_cases(value)
        self.assertEqual(value, original)
        self.assertEqual(result["example_format"], "codeforces")
        self.assertEqual([m["case_indices"] for m in result["example_mappings"]], [[0], [1]])
        self.assertEqual([m["answer"] for m in result["example_mappings"]], ["1", "2"])
        self.assertEqual([c["visible"] for c in result["cases"]], [True, True, False])
        self.assertEqual((result["visible_case_count"], result["nonvisible_case_count"]), (2, 1))
        for before, after in zip(value["cases"], result["cases"]):
            self.assertEqual(before, {k: v for k, v in after.items() if k != "visible"})
        self.assertEqual(result["solutions"], value["solutions"])
        self.assertNotIn("qualified", result)

    def test_singular_codeforces_header_and_crlf(self):
        statement = "-----Example-----\nInput\n1\n\nOutput\n1\n"
        result = examples.map_statement_examples_to_cases(candidate(statement.replace("\n", "\r\n")))
        self.assertEqual(result["example_mappings"][0]["case_indices"], [0])

    def test_atcoder_unnumbered_with_explanation(self):
        result = examples.map_statement_examples_to_cases(candidate(
            "-----Sample Input-----\n1\n\n-----Sample Output-----\n1\n\n"
            "The answer is 1.\n - This is an explanation.\n",
        ))
        self.assertEqual(result["example_format"], "atcoder")
        self.assertEqual(result["example_mappings"][0]["answer"], "1")

    def test_atcoder_blank_lines_and_crlf_before_explanation(self):
        result = examples.map_statement_examples_to_cases(candidate(
            "-----Sample Input-----\r\n1\r\n\r\n-----Sample Output-----\r\n1\r\n\r\n\r\n"
            " - One possible explanation.\r\n\r\nAnother paragraph.\r\n",
        ))
        self.assertEqual(result["example_mappings"][0]["case_indices"], [0])

    def test_atcoder_numbered_examples_all_map_in_statement_order(self):
        result = examples.map_statement_examples_to_cases(candidate(
            "-----Sample Input 1-----\n2\n\n-----Sample Output 1-----\n2\n\nExplanation.\n"
            "-----Sample Input 2-----\n1\n\n-----Sample Output 2-----\n1\n",
            ["1", "2", "3"], ["1", "2", "3"],
        ))
        self.assertEqual([m["case_indices"] for m in result["example_mappings"]], [[1], [0]])

    def test_equivalent_duplicate_inputs_all_become_visible(self):
        result = examples.map_statement_examples_to_cases(candidate(
            "-----Examples-----\nInput\n1 2\n3\n\nOutput\n-0\n",
            ["1\t2 3", " 1\n2\n3\n", "1 23"], ["-0", "-0\n", "4"],
        ))
        self.assertEqual(result["example_mappings"][0]["case_indices"], [0, 1])
        self.assertEqual([c["visible"] for c in result["cases"]], [True, True, False])
        self.assertEqual(result["nonvisible_case_count"], 1)

    def test_no_nonvisible_case_including_duplicate_inputs(self):
        self.reject(candidate(inputs=["1", " 1\n"], outputs=["1", "1"]), "no_nonvisible_cases")
        self.reject(candidate(
            "-----Examples-----\nInput\n1\nOutput\n1\nInput\n2\nOutput\n2\n",
        ), "no_nonvisible_cases")

    def test_contradictory_stored_answers_for_visible_or_nonvisible_inputs(self):
        for inputs, outputs in (
            (["1", "1\n", "2"], ["1", "2", "2"]),
            (["1", "2", "2\t"], ["1", "2", "3"]),
            (["1", "2", "2\n"], ["1", "0", "-0"]),
        ):
            with self.subTest(inputs=inputs, outputs=outputs):
                self.reject(candidate(inputs=inputs, outputs=outputs), "conflicting_case_answers")

    def test_every_example_must_match_input_and_answer(self):
        for stdin, stdout, reason in (
            ("9", "9", "example_input_not_stored"),
            ("01", "1", "example_input_not_stored"),
            ("2", "9", "example_answer_mismatch"),
        ):
            with self.subTest(stdin=stdin, stdout=stdout):
                self.reject(candidate(
                    f"-----Examples-----\nInput\n1\nOutput\n1\nInput\n{stdin}\nOutput\n{stdout}\n",
                    ["1", "2", "3"], ["1", "2", "3"],
                ), reason)

    def test_exact_answer_spelling_and_huge_tokens(self):
        for answer in ("-0", "9" * 4897, "1" * 5000):
            with self.subTest(length=len(answer)):
                result = examples.map_statement_examples_to_cases(candidate(
                    f"-----Examples-----\nInput\n1\nOutput\n{answer}\n",
                    ["1", "2"], [answer, "2"],
                ))
                self.assertEqual(result["example_mappings"][0]["answer"], answer)
        self.reject(candidate(
            "-----Examples-----\nInput\n1\nOutput\n-0\n", ["1", "2"], ["0", "2"],
        ), "example_answer_mismatch")
        for answer in ("01", "+1", "1 2", "1\n2", "answer: 1", ""):
            with self.subTest(answer=answer):
                self.reject(candidate(f"-----Examples-----\nInput\n1\nOutput\n{answer}\n"),
                            "invalid_example_output")

    def test_absent_and_unsupported_formats(self):
        for statement, reason in (
            ("-----Note-----\nThere are no examples.", "missing_examples"),
            ("-----Examples-----\nstdin: 1\nstdout: 1", "unsupported_example_format"),
            ("-----Example 1-----\nInput\n1\nOutput\n1", "unsupported_example_format"),
            ("-----Sample Input #1-----\n1", "unsupported_example_format"),
        ):
            with self.subTest(statement=statement):
                self.reject(candidate(statement), reason)

    def test_incomplete_or_ambiguous_pairings(self):
        for statement in (
            "-----Examples-----\nInput\n1",
            "-----Examples-----\nOutput\n1\nInput\n1",
            "-----Examples-----\nInput\n1\nInput\n2",
            "-----Examples-----\nUnparsed content\nInput\n1\nOutput\n1",
            "-----Examples-----\nInput\n\nOutput\n1",
            "-----Examples-----\nInput\n1\nOutput\n1\n-----Example-----\nInput\n2\nOutput\n2",
            "-----Examples-----\nInput\n1\nOutput\n1\n-----Sample Input-----\n2\n-----Sample Output-----\n2",
            "-----Sample Input-----\n1",
            "-----Sample Output-----\n1\n-----Sample Input-----\n1",
            "-----Sample Input 1-----\n1\n-----Sample Output 2-----\n1",
            "-----Sample Input 1-----\n1\n-----Sample Output-----\n1",
            "-----Sample Input 1-----\n1\n-----Sample Output 1-----\n1\n"
            "-----Sample Input 1-----\n2\n-----Sample Output 1-----\n2",
            "-----Sample Input-----\n1\n-----Sample Output-----\n1\n\n2\n\nExplanation.",
        ):
            with self.subTest(statement=statement):
                self.reject(candidate(statement), "ambiguous_examples")

    def test_atcoder_extra_numeric_paragraphs_are_not_discarded(self):
        for suffix in ("\n\n01", "\n\n+1", "\n\n1.0", "\n\n\n\n2", "\n\nExplanation.\n\n2"):
            with self.subTest(suffix=suffix):
                self.reject(candidate(
                    "-----Sample Input-----\n1\n-----Sample Output-----\n1" + suffix,
                ), "ambiguous_examples")

    def test_codeforces_note_header_with_image_trailer(self):
        result = examples.map_statement_examples_to_cases(candidate(
            "-----Examples-----\nInput\n1\nOutput\n1\n\n"
            "-----Note----- [Image]\xa0[Image]\nThe answer is shown in red.",
        ))
        self.assertEqual(result["example_mappings"][0]["answer"], "1")

    def test_unparsed_additional_examples_are_rejected(self):
        for suffix in (
            "\n\nSample Input 2\n9\n\nSample Output 2\n9",
            "\n\nInput\n9\n\nOutput\n9",
            "\n-----Input 2-----\n9\n-----Output 2-----\n9",
            "\n-----Inputs 2-----\n9\n-----Outputs 2-----\n9",
            "\n-----Note-----\nInput\n9\nOutput\n9",
            "\n-----Sample Input 2----- unexpected text\n9\n-----Sample Output 2-----\n9",
        ):
            with self.subTest(suffix=suffix):
                self.reject(candidate(
                    "-----Sample Input-----\n1\n-----Sample Output-----\n1" + suffix,
                ), "unsupported_example_format")

    def test_inline_and_nonstandard_labels_cannot_hide_visible_cases(self):
        for labels in (
            "Input: 2\nOutput: 2",
            "Sample Input #2\n2\nSample Output #2\n2",
            "Input example 2\n2\nOutput example 2\n2",
            "Sample Input 2: 2\nSample Output 2: 2",
        ):
            for position in ("before", "note", "explanation"):
                statement = "-----Sample Input-----\n1\n-----Sample Output-----\n1"
                if position == "before":
                    statement = "-----Note-----\n" + labels + "\n" + statement
                elif position == "note":
                    statement += "\n-----Note-----\n" + labels
                else:
                    statement += "\n\n" + labels
                with self.subTest(labels=labels, position=position):
                    self.reject(candidate(statement), "unsupported_example_format")
        value = candidate()
        value["question"] = "Input: 2\nOutput: 2\n" + value["question"]
        self.reject(value, "unsupported_example_format")

    def test_numeric_output_instructions_are_not_example_labels(self):
        value = candidate()
        value["question"] = (
            "Output 1 if x is equal to 0, or 0 if x is equal to 1.\n"
            + value["question"].replace("Print one integer.", "Output 0 or 1.")
        )
        result = examples.map_statement_examples_to_cases(value)
        self.assertEqual(result["nonvisible_case_count"], 1)

    def test_atcoder_repeated_unnumbered_or_mixed_numbering_is_ambiguous(self):
        for labels in (("", ""), ("", " 2"), (" 1", "")):
            statement = "".join(
                f"-----Sample Input{label}-----\n{value}\n-----Sample Output{label}-----\n{value}\n"
                for value, label in enumerate(labels, 1)
            )
            with self.subTest(labels=labels):
                self.reject(candidate(statement, ["1", "2", "3"], ["1", "2", "3"]),
                            "ambiguous_examples")

    def test_sample_output_slices_preserve_whitespace_in_both_formats(self):
        for statement in (
            "-----Examples-----\nInput\r\n1\r\nOutput\r\n 1 \r\n",
            "-----Sample Input-----\r\n1\r\n-----Sample Output-----\r\n 1 \r\n",
        ):
            with self.subTest(statement=statement):
                result = examples.map_statement_examples_to_cases(candidate(statement))
                mapping = result["example_mappings"][0]
                self.assertEqual(mapping["input"], "\n1\r\n")
                self.assertEqual(mapping["output"], "\n 1 \r\n")
                self.assertEqual(mapping["answer"], "1")

    def test_atcoder_explanation_stays_in_question_and_punctuation_is_not_numeric(self):
        for punctuation in ("-", ".", "..."):
            value = candidate(
                "-----Sample Input-----\n1\n-----Sample Output-----\n 1 \n\n"
                + punctuation + "\n\nThe answer is shown at 13:31.\n",
            )
            with self.subTest(punctuation=punctuation):
                result = examples.map_statement_examples_to_cases(value)
                self.assertEqual(result["example_mappings"][0]["output"], "\n 1 \n\n")
                self.assertEqual(result["question"], value["question"])


if __name__ == "__main__":
    unittest.main()
