"""Statement/oracle controls independent of generated recovery outcomes."""
import importlib.util
from pathlib import Path
import random

import unittest

SPEC = importlib.util.spec_from_file_location('recovery', Path(__file__).parents[1] / 'scripts/dev_recovery_cases.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


EXAMPLES = [
    (4608, '3\n3\n1\n2\n', 2),
    (4608, '2\n1\n2\n', -1),
    (2477, '2 3\n7 9\n', 4),
    (2477, '1 0\n999999999\n', 999999999),
    (2100, '2\n0 1\n1 0\n', 2),
    (2474, '1\n1000000000\n', 999999993),
    (2474, '2\n1 2\n', 28),
    (4418, '5\n4 8 15 16 23\n', 5),
    (4418, '12\n4 8 4 15 16 8 23 15 16 42 23 42\n', 0),
    (2260, '2000000000000000000\n', 25),
    (2174, '3\n-1 -1 2\n', 6),
    (2649, '3\n1 1\n2 4\n3 2\n', 4),
    (4327, '1 3\n', 3),
    (2282, '4\nLRLR\n', 5),
]


class RecoveryTests(unittest.TestCase):
    def test_oracles(self):
        for task, stdin, answer in EXAMPLES:
            with self.subTest(task=task, stdin=stdin):
                self.assertEqual(r.oracle(task, stdin), answer)


    def test_generated_constraints_and_bounds(self):
        for task in set(r.TASKS) - set(r.FINITE):
            rng = random.Random(1234)
            for slot in range(24):
                stdin = r.proposal(task, rng, slot)
                self.assertLessEqual(len(stdin.encode()), 4096)
                self.assertIsInstance(r.oracle(task, stdin), int)

    def test_invalid_input_and_permutation_duplicates(self):
        with self.assertRaises(AssertionError):
            r.oracle(4327, '101 0\n')
        with self.assertRaises(AssertionError):
            r.oracle(4608, '2\n1\n3\n')
        self.assertEqual(r.canonical(2649, '2\n1 2\n3 4\n'), r.canonical(2649, '2\n3 4\n1 2\n'))
        self.assertEqual(r.canonical(2174, '3\n7 1 4\n'), r.canonical(2174, '3\n4 7 1\n'))
