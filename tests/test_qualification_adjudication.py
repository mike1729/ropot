"""User adjudication changes derived labels, preserving original evidence and counts."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/qualification'))
import freeze as f
import adjudicate


class AdjudicationTests(unittest.TestCase):
    def test_reclassifying_a_fault_preserves_detection_and_reviewer_denominators(self):
        cell=dict(categories={'certified fault':1,'another real defect':0,'incorrect':1,'unassessed':1},
                  flags=3,attacks=5,detection={'FLAG':3,'PASS':2},independent_reviewers={'codex':{'certified fault':2}})
        before=deepcopy(cell)
        adjudicate.update_cell(cell,'another real defect')
        self.assertEqual(cell['correct_grounded_flags'],2)
        self.assertEqual(cell['correct_fraction_among_all_flags'],2/3)
        self.assertEqual(cell['grounded_flags_among_all_attacks'],2/5)
        for field in ('flags','attacks','detection','independent_reviewers'):
            self.assertEqual(cell[field],before[field])

    def test_accepted_records_preserve_original_reviews_and_execution_status(self):
        root=f.OUT/'status-adjudicated-20261102'
        if not (root/'manifest.json').exists():self.skipTest('Private adjudication not applied')
        f.load(root);f.load(f.OUT/'report');f.load(f.OUT/'status-monitor-resolved')
        old=f.read(f.OUT/'report/grounding.json');new=f.read(root/'grounding.json')
        self.assertEqual(len(old['unresolved_disagreements']),4)
        self.assertEqual(new['unresolved_disagreements'],{})
        for identifier,row in new['resolved_disagreements'].items():
            self.assertEqual(row['original_disagreement'],old['unresolved_disagreements'][identifier])
            self.assertFalse(row['execution_verified'])
            self.assertEqual(next(w for w in new['witnesses'] if w['id']==identifier)['status'],'unverified')
        self.assertEqual(sum(c['correct_grounded_flags'] for cells in new['all'].values() for c in cells.values()),163)
        self.assertEqual(sum(c['flags'] for cells in new['all'].values() for c in cells.values()),169)
        for name,panel in new['panels'].items():
            for monitor,cells in panel.items():
                for source,cell in cells.items():
                    self.assertEqual(cell['flags'],old['panels'][name][monitor][source]['flags'])
                    self.assertEqual(sum(cell['categories'].values()),cell['flags'])
                    self.assertEqual(cell['categories']['unassessed'],0)
        ready=f.read(root/'readiness.json')
        self.assertTrue(all('explanation' not in b.lower() for b in ready['blockers']))
        self.assertEqual(len(ready['blockers']),2)
        self.assertEqual(ready['resolutions']['monitor_scoring']['status'],'resolved')
        self.assertEqual(ready['resolutions']['explanation_grounding']['status'],'resolved')
        self.assertFalse(ready['training_launch_ready'])
        packets=f.read(root/'adjudicated-packets.json')['explanations']
        self.assertEqual(len(packets),4)
        self.assertTrue(all(p['user_adjudication']['structured_actual_output_claim']=='incorrect' for p in packets))


if __name__=='__main__':unittest.main()
