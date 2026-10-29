"""Comment normalization preserves literals and accepts deeply nested syntax."""
import ast
import importlib
from io import StringIO
import tokenize
import unittest

p = importlib.import_module("src.data-pipeline.stage03.programs")
m = importlib.import_module("src.data-pipeline.stage05.monitor")


class ProgramTests(unittest.TestCase):
    def test_normalization_matches_monitor_and_preserves_ast_and_literals(self):
        samples = [
            "# heading\nvalue = '# literal'  # inline\nprint(value)  \t\n",
            'text = """# first line  \n  \t\n# last line"""  # trailing\nprint(text)\n',
            "# heading\r\nvalue = '# literal'  # inline\r\nprint(value)\r\n",
            "value = 1 + \\\n  2  # continued\nprint(value)",
            "print(f'# {1 + 2}') # trailing\n",
            "print('żółw') # unicode\n",
        ]
        for original in samples:
            with self.subTest(original=original):
                normalized = p.strip_comments(original)
                self.assertEqual(normalized, m.strip_comments(original))
                self.assertEqual(ast.dump(ast.parse(original)), ast.dump(ast.parse(normalized)))
                self.assertFalse(any(t.type == tokenize.COMMENT for t in
                                     tokenize.generate_tokens(StringIO(normalized).readline)))
                self.assertEqual(p.strip_comments(normalized), normalized)

    def test_encoding_comment_cannot_change_byte_parsed_literals(self):
        with self.assertRaisesRegex(ValueError, "AST"):
            p.strip_comments("# coding: latin-1\nprint('é')\n")

    def test_four_hundred_branch_program_strips_without_recursive_ast_dump(self):
        original = "\n".join(("if" if i == 0 else "elif") + f" value == {i}:\n    print({i}) # branch"
                             for i in range(400)) + "\n"
        compile(original, "<original>", "exec")
        normalized = p.strip_comments(original)
        compile(normalized, "<normalized>", "exec")
        self.assertNotIn("# branch", normalized)
        self.assertTrue(p.same_ast(ast.parse(original), ast.parse(normalized)))

    def test_iterative_ast_comparison_detects_structural_and_value_changes(self):
        for left, right in (("print(1)", "print(2)"), ("x = 1", "x = True"),
                            ("print(1)", "print(1, 2)"), ("x + 1", "x - 1")):
            with self.subTest(left=left, right=right):
                self.assertFalse(p.same_ast(ast.parse(left), ast.parse(right)))
