"""Controls for the proposed contextual review; no exposure-policy mutation."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import review_task291_exposure as review


class Task291ReviewTests(unittest.TestCase):
    def test_other_limak_problem_is_not_task_291(self):
        self.assertFalse(review.task_context('Limak is a police officer. Input: 1 2'))
        self.assertTrue(review.task_context('Bear Limak wants to become the largest of bears.'))
        self.assertTrue(review.task_context('apps:test:291'))

    def test_whitespace_and_escaped_input_match(self):
        self.assertTrue(review.matches('3 9\n', 'Input\n3   9\nOutput\n3'))
        self.assertTrue(review.matches('3 9\n', '{"input": "3 9\\n", "answer": "3"}'))
        self.assertFalse(review.matches('3 9\n', 'Input\n13 90\n'))

    def test_incidental_numeric_match_is_retained_separately(self):
        text = 'A different graph task. Edges:\n1 2\n2 3\n'
        self.assertTrue(review.matches('1 2\n', text))
        self.assertFalse(review.task_context(text))

    def test_strictly_heavier_and_complete_domain(self):
        self.assertEqual(review.answer('4 9\n'), '3')
        self.assertEqual(review.answer('1 1\n'), '1')
        with self.assertRaises(AssertionError):
            review.answer('2 1\n')
        for a in range(1, 11):
            for b in range(a, 11):
                n = int(review.answer(f'{a} {b}\n'))
                self.assertGreater(a * 3**n, b * 2**n)
                self.assertLessEqual(a * 3**(n-1), b * 2**(n-1))
