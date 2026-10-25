from pathlib import Path
import unittest
from unittest.mock import patch

from experiments import followup_report as report
from experiments.analyze_followups import secondary_semantic_verdict
from experiments.followup_fixtures import semantic_verdict


class ReportingTests(unittest.TestCase):
    def test_partial_window_cannot_make_completion_median_faster_but_its_cost_counts(self):
        result=report.timing_summary([dict(status='success',elapsed_seconds=100),
            dict(status='error',elapsed_seconds=1),dict(status='success',elapsed_seconds=102)])
        self.assertEqual(result['median_seconds'],101)
        self.assertEqual(result['range_seconds'],[100,102])
        self.assertEqual(result['repetitions'],2)
        self.assertEqual(result['aborted_windows'],1)
        self.assertEqual(result['total_window_seconds'],203)

    def test_all_failed_configuration_has_no_completed_timing(self):
        result=report.timing_summary([dict(status='error',elapsed_seconds=13)])
        self.assertIsNone(result['median_seconds'])
        self.assertIsNone(result['range_seconds'])
        self.assertEqual(result['repetitions'],0)

    def test_verify_only_does_not_generate_empty_semantic_comparison_report(self):
        with patch.object(report,'verify') as verify,patch.object(report,'report') as render:
            report.main(['--root','/tmp/fixture','--verify-only'])
        verify.assert_called_once_with(Path('/tmp/fixture'))
        render.assert_not_called()

    def test_secondary_json_unfencing_keeps_primary_gate_and_oracle_rules(self):
        text='```json\n{"answer":1}\n```'
        self.assertEqual(semantic_verdict(text,'stop',1),'format_failure')
        self.assertEqual(secondary_semantic_verdict(text,'stop',1),'passed')
        self.assertEqual(secondary_semantic_verdict(text,'stop',2),'wrong_answer')
        self.assertEqual(secondary_semantic_verdict(text,'length',1),'incomplete')
        for value in ['true','1.0','{"nested":1}']:
            self.assertEqual(secondary_semantic_verdict('```json\n{"answer":'+value+'}\n```','stop',1),'format_failure')
        self.assertEqual(secondary_semantic_verdict('Prose\n'+text,'stop',1),'format_failure')


if __name__=='__main__':
    unittest.main()
