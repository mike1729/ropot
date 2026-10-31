import unittest
from scripts.construction_exposure import Prefixes, identities
from scripts import recover_construction_cases as r
import random


class ExposureTests(unittest.TestCase):
    def test_statement_free_monitor_metadata(self):
        p = Prefixes({4692: {'question':'New year problem statement'}})
        self.assertEqual(identities([{'task_id':'apps:test:4692'}], 'print(48-m)', p), {4692})

    def test_unidentified_is_not_assumed_unrelated(self):
        p = Prefixes({4692: {'question':'New year problem statement'}})
        self.assertEqual(identities([], 'Solve the same task', p), set())

    def test_4688_proposals_have_varied_bounded_answers(self):
        rng = random.Random(42)
        inputs = [r.proposal(4688,rng,j) for j in range(48)]
        answers = [r.oracle(4688,x) for x in inputs]
        self.assertGreater(len(set(answers)),40)
        self.assertEqual({int(x.split()[0]) for x in inputs}, {1,2,3,4})
        self.assertTrue(all(2 < v <= 2**31-1 for v in answers))
