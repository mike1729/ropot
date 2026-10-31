"""Statement examples and independent small-domain oracle controls."""
import random
import unittest

from scripts import recover_construction_cases as r


class ConstructionRecoveryTests(unittest.TestCase):
    def test_statement_examples(self):
        examples = [
            (1544,'2\n',24), (2113,'5\n1 2\n2 3\n3 4\n4 5\n',2),
            (2090,'4 3\n4 7\n15 1\n3 6\n6 8\n',78), (4688,'2 2\n',2),
            (4338,'5 3 3\n1 2 4 2 3\n',2), (2334,'3\n13 7 6\n6 2\n',4),
            (4692,'21\n',27), (2376,'4 6\n2 1\n3 4\n4 10\n3 4\n',11),
            (2048,'5\n2 4 5 4 10\n40 30 20 10 40\n',90),
            (4704,'6\n1 2 3 4 5 6\n',1), (4326,'8\n',2),
            (2152,'3\n1 3\n2 2\n3 1\n',10), (1950,'4\n2 3 4 5\n',19)]
        for task, stdin, expected in examples:
            with self.subTest(task=task):self.assertEqual(r.oracle(task,stdin),expected)

    def test_generated_constraints(self):
        for task in (1544,2113,2090,4688,4338,2334,4692,2376,2048,4704,4326,2152,1950):
            rng=random.Random(712+task)
            for j in range(12):
                stdin=r.proposal(task,rng,j)
                self.assertLessEqual(len(stdin.encode()),4096)
                self.assertIsInstance(r.oracle(task,stdin),int)

    def test_invalid_cross_field_constraints(self):
        for task,stdin in [(4688,'1000 1000\n'),(2376,'2 10\n3 1\n2 4\n'),
                           (2113,'4\n1 2\n2 1\n3 4\n'),(2334,'1\n5\n3 3\n')]:
            with self.subTest(task=task),self.assertRaises(AssertionError):r.oracle(task,stdin)

    def test_tree_relabelings_and_collection_permutations(self):
        self.assertEqual(r.key(2113,'4\n1 2\n1 3\n1 4\n'),r.key(2113,'4\n2 1\n2 4\n2 3\n'))
        self.assertNotEqual(r.key(2113,'4\n1 2\n1 3\n1 4\n'),r.key(2113,'4\n1 2\n2 3\n3 4\n'))
        self.assertEqual(r.key(2090,'2 1\n5 7\n3 9\n'),r.key(2090,'2 1\n3 9\n5 7\n'))

    def test_ternary_merge_boundary(self):
        self.assertEqual(r.oracle(1950,'1\n999\n'),0)
        self.assertEqual(r.oracle(1950,'2\n3 7\n'),10)
        self.assertEqual(r.oracle(1950,'3\n3 7 9\n'),19)
