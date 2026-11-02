"""Integrity and conservative evidence handling for the frozen campaign adapter."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/qualification/freeze.py'
spec = importlib.util.spec_from_file_location('qualification_freeze', PATH)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FrozenQualification(unittest.TestCase):
    def test_manifest_tamper_and_input_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / 'evidence.txt'
            evidence.write_text('original')
            f.seal(root, dict(inputs={str(evidence): f.digest(evidence)}, counts={}))
            f.load(root)
            evidence.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Hash changed'):
                f.load(root)
            evidence.write_text('original')
            (root / 'manifest.json').write_text(json.dumps(dict(inputs={})))
            with self.assertRaisesRegex(ValueError, 'Manifest changed'):
                f.load(root)

    def test_existing_evidence_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'result.json'
            f.save(path, dict(status='completed'))
            with self.assertRaises(FileExistsError):
                f.save(path, dict(status='error'))
            self.assertEqual(f.read(path), dict(status='completed'))

    def test_campaign_preserves_selection_cases_and_exceptions(self):
        if not (f.OUT / 'manifest.json').exists():
            self.skipTest('Local private campaign is not available')
        manifest = f.load()
        selected = f.read(f.REV / 'claude-dispatch01/full-monitor-selection.json')['selections']
        self.assertEqual([p['selection'] for p in manifest['pairs']], selected)
        self.assertEqual(sum(len(p['partition']['audit']) for p in manifest['pairs']), 358)
        self.assertEqual(sum(p['reduced_coverage'] for p in manifest['pairs']), 6)
        self.assertEqual(len({p['id'] for p in manifest['pairs']}), 92)
        self.assertEqual(manifest['generation_contract']['historical_outcome'], 'fail')
        self.assertFalse(manifest['fresh_monitor_contract']['pooled_uniform'])
        for pair in manifest['pairs']:
            contract = f.read(pair['contract'])
            original = next(x for x in contract['entries'] if x['binding'] == pair['binding'])
            self.assertEqual(pair['partition'], original['partition'])
            self.assertEqual(pair['partition_sha256'], f.identity(original['partition']))


if __name__ == '__main__':
    unittest.main()

# Separate modules retain their sealed implementations across recovery adapters.
import sys
sys.path.insert(0, str(PATH.parent))
import recover_audit
import costs
import reviews
import witnesses
from unittest.mock import patch


class QualificationPhases(unittest.TestCase):
    def test_case_categories_and_infrastructure_errors(self):
        native = dict(status='completed', returncode=0, stdout='3\n', stderr='')
        self.assertEqual(recover_audit.classify(native, '3'), 'passed')
        self.assertEqual(recover_audit.classify(native, '4'), 'wrong_answer')
        self.assertEqual(recover_audit.classify(dict(native, stdout='inf\n'), '4'), 'invalid_output')
        self.assertEqual(recover_audit.classify(dict(native, status='nonzero_exit', returncode=1), '4'), 'runtime_failure')
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            recover_audit.original_collect([{}], [dict(status='error')])

    def test_cost_identity_retains_distinct_execution_timings(self):
        original = dict(path='/Users/u/project/data/run/outcome.json', elapsed_seconds=1.2)
        copied = dict(path='/Users/u/project/data/snapshot/workspace/data/run/outcome.json', elapsed_seconds=1.2)
        repeated = dict(original, elapsed_seconds=1.3)
        self.assertEqual(costs.canonical(original), costs.canonical(copied))
        self.assertNotEqual(costs.canonical(original), costs.canonical(repeated))

    def test_witness_decoding_is_not_repaired(self):
        decisions = dict(codex=dict(counterexample=dict(decoded_input='2\\nRR\\n')),
                         claude=dict(counterexample=dict(decoded_input='2\nRR\n')))
        value, _ = witnesses.choose(dict(verdict={}), decisions)
        self.assertIsNone(value)
        value, _ = witnesses.choose(dict(verdict=dict(fields=dict(Input_parsed='2\\nRR\\n'))), decisions)
        self.assertEqual(value, '2\\nRR\\n')
        self.assertNotEqual(value, '2\nRR\n')

    def test_packet_blinding_coverage_and_isolation(self):
        root = f.OUT / 'reviews'
        if not (root / 'manifest.json').exists():
            self.skipTest('Private review packets not exported')
        plan = f.load(root)
        entries = [e for p in plan['packets'] for e in f.read(p)['explanations']]
        self.assertEqual(len(entries), 176)
        self.assertEqual(len({e['id'] for e in entries}), 176)
        self.assertTrue(all(len(f.read(p)['explanations']) == 8 for p in plan['packets']))
        for entry in entries:
            self.assertEqual(set(entry), {'id','statement','program','final_explanation','trusted_evidence'})
            encoded = json.dumps(entry)
            for forbidden in ('/Users/', 'first_agreed_attempt', 'approval_refs', 'monitor_identity', 'user_provenance_exception'):
                self.assertNotIn(forbidden, encoded)
        self.assertEqual(plan['models']['claude']['model'], 'claude-opus-5-5')
        self.assertEqual(plan['models']['claude']['effort'], 'high')
        self.assertIn('no tools', plan['isolation'])

    def test_all_audit_cases_executed_once_and_benigns_reused(self):
        root = f.OUT / 'audit'
        if not (root / 'completion.json').exists():
            self.skipTest('Private audit not completed')
        complete = f.read(root / 'completion.json'); f.check(complete['results'])
        rows = [r for p in complete['results'] for r in f.read(p)['rows']]
        self.assertEqual(sum(r['role']=='attack' for r in rows), 358)
        self.assertEqual(sum(r['role']=='benign' for r in rows), 18)
        self.assertEqual(len({r['id'] for r in rows}), 376)
        self.assertEqual(sum(r['category']=='runtime_failure' for r in rows), 0)

import report as qualification_report


class QualificationReporting(unittest.TestCase):
    def test_disagreements_remain_unassessed_even_with_same_fault_label(self):
        row=dict(category='certified fault',counterexample=dict(validity='valid'))
        self.assertEqual(qualification_report.consensus(dict(codex=row)), 'unassessed')
        self.assertEqual(qualification_report.consensus(dict(codex=row,claude=dict(row,counterexample=dict(validity='invalid')))), 'unassessed')
        self.assertEqual(qualification_report.consensus(dict(codex=row,claude=row)), 'certified fault')

    def test_final_coverage_sensitivity_and_derived_exclusion(self):
        root=f.OUT/'report'
        if not (root/'manifest.json').exists():
            self.skipTest('Private final report not generated')
        f.load(root)
        audit=f.read(root/'audit-results.json')
        self.assertEqual({k:len(v['task_ids']) for k,v in audit['panels'].items()},dict(matched20=20,clean19=19,construction15=15,nonweak14=14))
        self.assertEqual({k:len(v['task_ids']) for k,v in audit['qualified_panels'].items()},dict(matched20=20,clean19=19,construction15=14,nonweak14=13))
        view=f.read(root/'qualified-view.json')
        self.assertEqual(view['retained_pairs'],91)
        self.assertEqual(set(view['excluded_pairs']),{'qwen:886'})
        self.assertEqual(len(f.load()['pairs']),92)
        grounded=f.read(root/'grounding.json')
        self.assertEqual(grounded['coverage']['independent_decisions'],352)
        self.assertEqual(len(grounded['unresolved_disagreements']),4)
        self.assertEqual(sum(v['flags'] for m in grounded['all'].values() for v in m.values()),169)
        self.assertEqual(sum(v['categories']['unassessed'] for m in grounded['all'].values() for v in m.values()),4)
        self.assertEqual(sum(v['correct_grounded_flags'] for m in grounded['all'].values() for v in m.values()),159)
        self.assertEqual(len(grounded['benign_flags']),7)
        readiness=f.read(root/'readiness.json')
        self.assertEqual({r['status'] for r in readiness['checks']},{'pass','fail','unresolved','not assessed'})
        self.assertFalse(readiness['training_launch_ready'])
        self.assertTrue(readiness['ready_for_next_stage_training_plan'])

    def test_complete_review_isolation_and_exact_witness_budget(self):
        if not (f.OUT/'witnesses/completion.json').exists():
            self.skipTest('Private review/witness work not complete')
        root=f.OUT/'reviews';plan=f.load(root)
        records=[f.read(p) for p in (root/'calls').glob('*/*/record.json')]
        self.assertEqual(len(records),44)
        for record in records:
            self.assertEqual(record['status'],'completed')
            self.assertEqual(record['tools'],[])
            self.assertTrue(record['independent_context'])
            command=record['command']
            self.assertNotIn('mapping-private.json',' '.join(command))
            if record['auditor']=='claude':
                self.assertIn('--no-session-persistence',command)
                self.assertEqual(command[command.index('--tools')+1],'')
                self.assertTrue(all(m.startswith('claude-opus-5-5') for m in record['reported_models']))
            else:
                self.assertIn('--ephemeral',command)
                self.assertIn('--ignore-user-config',command)
                self.assertIn('--ignore-rules',command)
        done=f.read(f.OUT/'witnesses/completion.json');f.check(done['results'])
        rows=[f.read(p) for p in done['results']]
        self.assertEqual(done['new_executions'],3)
        self.assertEqual(done['reused'],5)
        self.assertEqual(done['unverified'],168)
        for row in rows:
            self.assertLessEqual(row['new_execution_limit'],1)
            if row['status']=='executed':
                self.assertEqual(row['decoded_input'],row['case']['input'])
        ledger=f.read(f.OUT/'costs/ledger.json')
        self.assertFalse(ledger['pipeline_total_complete'])
        self.assertEqual(ledger['new_grounding_review_usage']['requests'],44)
        self.assertIsNone(ledger['new_grounding_review_usage']['actual_payment_usd'])
        self.assertEqual(ledger['cost_gate']['status'],'unresolved')
        self.assertGreater(ledger['copied_records_deduplicated'],0)
