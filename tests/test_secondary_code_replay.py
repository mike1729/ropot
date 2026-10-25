import unittest

from experiments.secondary_code_replay import fenced_code


class SecondaryCodeTests(unittest.TestCase):
    def test_prose_can_surround_one_unambiguous_complete_python_program(self):
        self.assertEqual(fenced_code("Here's the correction:\n\n```python\nprint(1)\n```\nIt prints one."),b'print(1)\n')

    def test_multiple_foreign_incomplete_or_invalid_blocks_are_not_selected(self):
        for text in ('```python\nprint(1)\n```\n```python\nprint(2)\n```',
                     '```javascript\nprint(1)\n```', '```python\nprint(1)',
                     '```python\nprint(\n```', 'Explanation and print(1)'):
            with self.assertRaises((ValueError,SyntaxError)):
                fenced_code(text)


if __name__=='__main__':
    unittest.main()
