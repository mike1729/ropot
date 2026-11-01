"""Validate the explicit recovery selection without models or saved eval logs."""
from copy import deepcopy
import unittest

from scripts import prepare_yield_panel as p


class SelectionTests(unittest.TestCase):
    def test_active_recipe_reserves_1024_tokens_for_every_submission(self):
        config = p.e.loop.read(p.e.ROOT / 'configs/inference/native-multi-edit.json')
        chains, recipe = p.e.settings(config)
        self.assertEqual([recipe[k] for k in ('max_tokens', 'repair_max_tokens', 'thinking_repair_max_tokens')],
                         [1024, 1024, 1024])
        self.assertEqual(len(p.IDS) * chains * (recipe['max_repairs'] + 1) * 1024, 81920)

    def panels(self):
        entries = [dict(binding=dict(id=i, owner='development'),
                        partition=dict(examples=[dict(input='1', answer='1')], menu=[])) for i in p.IDS]
        return {s: (deepcopy(entries), [(x['binding'], dict(question='public')) for x in entries], {}, {})
                for s in ('qwen', 'gemma')}

    def test_both_sources_receive_identical_order_and_all_eight_tasks(self):
        selected = p.select(self.panels())
        self.assertEqual({s: [x['binding']['id'] for x, _ in rows] for s, rows in selected.items()},
                         {s: p.IDS for s in ('qwen', 'gemma')})
        self.assertEqual(sum(map(len, selected.values())), 16)

    def test_shared_partitions_are_required(self):
        panels = self.panels()
        panels['qwen'][0][0]['partition']['examples'][0]['answer'] = '2'
        with self.assertRaisesRegex(ValueError, 'shared frozen'):
            p.select(panels)

    def test_non_development_and_missing_counterpart_are_rejected(self):
        panels = self.panels()
        for source in panels:
            panels[source][0][0]['binding']['owner'] = 'construction'
        with self.assertRaisesRegex(ValueError, 'development'):
            p.select(panels)
        panels = self.panels()
        panels['qwen'][0].pop()
        panels['qwen'][1].pop()
        with self.assertRaises(KeyError):
            p.select(panels)
