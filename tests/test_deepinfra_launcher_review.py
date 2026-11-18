"""Offline diagnostics and reviewable finite-job preparation summaries."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import launch_deepinfra as launcher


class PreflightDiagnosticsTests(unittest.TestCase):
    def assert_child_failure(self, source, reason, specification=None):
        with mock.patch.object(launcher, 'PREFLIGHT', source), \
                mock.patch.dict(os.environ, DEEPINFRA_API_KEY='REVIEW_TEST_KEY',
                                PYTHONDONTWRITEBYTECODE='1'), \
                self.assertRaises(RuntimeError) as caught:
            launcher._preflight(specification or {}, sys.executable)
        message = str(caught.exception)
        self.assertRegex(message.splitlines()[0], r'\b1\b')
        self.assertIn(reason, message)
        self.assertNotIn('REVIEW_TEST_KEY', message)

    def test_invalid_request_reports_the_actual_worker_validation_error(self):
        specification = dict(runtime=dict(model=launcher.runtime_api.MODEL,
            tokenizer_path='unused', tokenizer_sha256={}, timeout_seconds=7200,
            max_total_tokens=100, input_usd_per_million=.3,
            output_usd_per_million=3.75, max_cost_usd=1),
            requests=[dict(messages=[], seed=41, temperature=1., top_p=.95, max_tokens=64)],
            output_dir='unused')
        source = '''import json,os,sys
from scripts import deepinfra_generation as worker
from tests.test_thinking_generation import ThinkingTokenizer
assert 'DEEPINFRA_API_KEY' not in os.environ
worker.prepare(json.load(sys.stdin),tokenizer=ThinkingTokenizer('qwen'))
'''
        self.assert_child_failure(source, 'Invalid generation request; parameters cannot be silently dropped',
                                  specification)

    def test_version_failure_reports_the_actual_package_pin_error(self):
        source = '''import os
from scripts import deepinfra_generation as worker
assert 'DEEPINFRA_API_KEY' not in os.environ
worker.version = lambda name: '0.0.review-mismatch'
worker.check_versions()
'''
        self.assert_child_failure(source, 'Tokenizer package versions differ from requirements')

    def test_captured_failure_redacts_parent_key_without_hiding_the_reason(self):
        failure = subprocess.CalledProcessError(23, ['fixture-python'],
            stderr='ValueError: Prompt exceeds envelope; incidental REVIEW_TEST_KEY')
        with mock.patch.dict(os.environ, DEEPINFRA_API_KEY='REVIEW_TEST_KEY'), \
                mock.patch.object(launcher.subprocess, 'run', side_effect=failure) as run, \
                self.assertRaises(RuntimeError) as caught:
            launcher._preflight({}, sys.executable)
        self.assertIn('23', str(caught.exception).splitlines()[0])
        self.assertIn('Prompt exceeds envelope', str(caught.exception))
        self.assertNotIn('REVIEW_TEST_KEY', str(caught.exception))
        self.assertNotIn('DEEPINFRA_API_KEY', run.call_args.kwargs['env'])

    def test_success_keeps_the_versions_and_rendered_payload_protocol(self):
        expected = dict(versions=dict(transformers='5.23.0', tokenizers='0.23.3', jinja2='3.1.6'),
                        prompt_tokens=[599], payload_sha256=['a' * 64], planned_max_cost_usd=.2459397)
        result = SimpleNamespace(returncode=0, stdout=json.dumps(expected), stderr='')
        with mock.patch.dict(os.environ, DEEPINFRA_API_KEY='REVIEW_TEST_KEY'), \
                mock.patch.object(launcher.subprocess, 'run', return_value=result) as run:
            self.assertEqual(launcher._preflight(dict(requests=[]), sys.executable), expected)
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertNotIn('DEEPINFRA_API_KEY', run.call_args.kwargs['env'])
        self.assertEqual(json.loads(run.call_args.kwargs['input']), dict(requests=[]))


class PrepareSummaryTests(unittest.TestCase):
    def test_cli_prints_each_saved_rounds_large_cap_timeout_and_reservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'prepared'
            output.mkdir()
            reservations = dict(qwen=.7977942, gemma=.1784637)
            jobs = {}
            for family, reservation in reservations.items():
                round_file = output / f'round-{family}.json'
                round_file.write_text(json.dumps(dict(runtime=dict(timeout_seconds=7200),
                    requests=[dict(max_tokens=65536)] * 3)))
                jobs[family] = dict(round_file=str(round_file), reserved_max_cost_usd=reservation)
            prepared = dict(rounds=jobs, reserved_max_cost_usd=.9762579)
            (output / 'launch.json').write_text(json.dumps(prepared))
            argv = ['launch_deepinfra.py', 'prepare', str(root / 'plan.json'), str(output),
                    '--plan-sha256', 'b' * 64, '--generation-python', sys.executable,
                    '--limits', str(root / 'limits.json'), '--max-cost-usd', '1',
                    '--qwen-runtime', str(root / 'qwen-runtime.json'), '--qwen-max-tokens', '65536',
                    '--gemma-runtime', str(root / 'gemma-runtime.json'), '--gemma-max-tokens', '65536']
            stdout = io.StringIO()
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(launcher, 'prepare', return_value=prepared) as prepare, \
                    mock.patch.object(sys, 'stdout', stdout):
                self.assertEqual(launcher.main(), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary['launch_sha256'], launcher.generation.file_hash(output / 'launch.json'))
            self.assertEqual(summary['reserved_max_cost_usd'], .9762579)
            self.assertEqual(summary['rounds'], {family: dict(requests=3, max_tokens=65536,
                timeout_seconds=7200, reserved_max_cost_usd=reservation)
                for family, reservation in reservations.items()})
            self.assertEqual(prepare.call_args.kwargs['max_tokens'], dict(qwen=65536, gemma=65536))


if __name__ == '__main__':
    unittest.main()
