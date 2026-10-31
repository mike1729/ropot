import unittest
from scripts import matched_monitors as m


class MonitorTests(unittest.TestCase):
    def test_four_cells_and_shared_program(self):
        records=[dict(memberships=[['actor','benign'],['synthetic','benign']]),
                 dict(memberships=[['actor','attack']]),dict(memberships=[['synthetic','attack']])]
        result=m.cells(records,[dict(verdict='PASS'),dict(verdict='FLAG'),dict(verdict='missing')])
        self.assertEqual(set(result),{'actor/benign','actor/attack','synthetic/benign','synthetic/attack'})
        self.assertEqual(result['actor/attack']['flag_rate'],1)
        self.assertEqual(result['synthetic/attack']['flag_rate'],0)
        self.assertEqual(result['synthetic/attack']['escalation_rate'],1)
        self.assertEqual(result['synthetic/benign']['counts']['PASS'],1)

    def test_empty_cells_not_zero_accuracy(self):
        result=m.cells([],[])
        self.assertTrue(all(x['flag_rate'] is None and x['n']==0 for x in result.values()))

    def test_length_stop_is_not_a_verdict(self):
        scope=dict(json=m.json,re=m.re)
        exec(m.q.u.read(m.q.e.a.MONITOR_CONFIG)['parser']['python_source'],scope)
        self.assertEqual(scope['parse'](dict(status='length',text='Analysis: incomplete\nVerdict: PASS'))['verdict'],'missing')
        self.assertEqual(scope['parse'](dict(status='stop',text='Analysis: checked\nVerdict: PASS'))['verdict'],'PASS')
