"""Allocation boundaries for the specifically authorized task-291 exception."""
from copy import deepcopy
import unittest

from scripts import recover_task291 as r


class Recover291Tests(unittest.TestCase):
    def setUp(self):
        self.review = r.util.read(r.REVIEW / 'review.json')
        contract = r.util.read(r.util.BENIGNS / 'contract.json')
        self.item = next(x for x in contract['entries'] if x['binding']['id'] == 291)

    def test_keep_menu_and_select_clean_cases_by_original_order(self):
        result = r.partition(self.item, self.review)
        self.assertEqual(result['menu'], self.item['partition']['menu'])
        self.assertEqual(result['examples'], self.item['partition']['examples'])
        expected = [i for i in self.item['partition']['permutation']
                    if i in self.review['frozen_candidate_indices']][:4]
        self.assertEqual([c['index'] for c in result['audit']], expected)
        self.assertEqual(self.item['partition']['audit'], [])
        view = r.case_view(dict(partition=result))
        self.assertEqual(sum(not c['visible'] for c in view['cases']), 4)

    def test_other_task_cannot_use_exception(self):
        item = deepcopy(self.item)
        item['binding']['id'] = 4347
        with self.assertRaises(ValueError):
            r.partition(item, self.review)

    def test_insufficient_or_disclosed_audits_rejected(self):
        review = deepcopy(self.review)
        review['frozen_candidate_indices'] = review['frozen_candidate_indices'][:3]
        with self.assertRaises(ValueError):
            r.partition(self.item, review)
        review = deepcopy(self.review)
        review['candidates'] = []
        with self.assertRaises(ValueError):
            r.partition(self.item, review)
