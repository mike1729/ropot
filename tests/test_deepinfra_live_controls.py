"""Control replay, budget admission and live-path wiring without paid calls."""
from hashlib import sha256
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import deepinfra_live_controls as controls, native_multi_edit


class ControlsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / 'result'
        self.data = controls.fixtures()

    def live(self, **kwargs):
        with mock.patch.dict(controls.os.environ, DEEPINFRA_API_KEY='CONTROL_SECRET'):
            return controls.run(self.directory, live=True, input_rate=.4, output_rate=.8,
                                max_cost=.2, **kwargs)

    def test_default_replays_all_ten_exact_receipts_without_environment_network_or_pipeline(self):
        with mock.patch.object(controls.os.environ, 'get', side_effect=AssertionError('No key read')), \
             mock.patch.object(controls, 'exchange', side_effect=AssertionError('No HTTP')), \
             mock.patch.object(controls, 'native_pipeline', side_effect=AssertionError('No Docker')), \
             mock.patch.object(controls.subprocess, 'run', side_effect=AssertionError('No child')):
            result = controls.run(self.directory)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(len(result['rows']), 10)
        self.assertEqual((result['new_api_calls'], result['new_api_cost_usd']), (0, 0))
        self.assertTrue(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .0002038)
        self.assertFalse(any(row['original_token_ids_available'] for row in result['rows']))
        for case in self.data['cases']:
            saved = json.loads((self.directory / case['id'] / 'response.json').read_bytes())
            self.assertEqual(saved, json.loads(case['response']['raw']))

    def test_fixture_retains_every_exact_original_request_route_and_protocol_variant(self):
        cases = {case['id']: case for case in self.data['cases']}
        self.assertEqual(len(cases), 10)
        self.assertTrue(cases['return-token-ids']['payload']['return_token_ids'])
        self.assertEqual(cases['logprob-token-ids']['payload']['logprobs'], 1)
        self.assertTrue(cases['echo']['payload']['echo'])
        self.assertTrue(cases['stream']['payload']['stream_options']['include_usage'])
        self.assertEqual(cases['canonical-route']['path'], '/v1/completions')
        self.assertIsInstance(cases['raw-string-input']['payload']['prompt'], str)
        for case in cases.values():
            for kind in ('request', 'response'):
                record = case[kind]
                self.assertEqual(sha256(record['raw'].encode()).hexdigest(), record['sha256'])
                self.assertTrue(record['source_path'].startswith('data/monitor_training/'))

    def test_rejects_missing_rates_bad_cost_and_bad_selection_before_output_or_dispatch(self):
        for options in (dict(input_rate=None), dict(input_rate=float('nan')), dict(output_rate=-1),
                        dict(max_cost=None), dict(max_cost=.000001), dict(cases=['absent']),
                        dict(cases=['thinking', 'thinking'])):
            with self.subTest(options=options), mock.patch.object(controls, 'exchange') as send:
                settings = dict(live=True, input_rate=.4, output_rate=.8, max_cost=.2, cases=['thinking'])
                settings.update(options)
                with self.assertRaises(ValueError):
                    controls.run(self.directory, **settings)
                self.assertFalse(self.directory.exists())
                send.assert_not_called()

    def test_raw_string_uses_full_context_reservation_instead_of_historical_prompt_usage(self):
        with self.assertRaisesRegex(ValueError, 'ceiling'):
            controls.run(self.directory, live=True, cases=['raw-string-input'], input_rate=.4,
                         output_rate=.8, max_cost=.01)

    def test_live_sends_selected_exact_payload_saves_raw_sse_and_redacts_secret(self):
        case = next(c for c in self.data['cases'] if c['id'] == 'stream')
        receipt = json.loads(case['response']['raw'])
        with mock.patch.object(controls, 'exchange', return_value=receipt) as send:
            result = self.live(cases=['stream'])
        sent, key, timeout = send.call_args.args
        self.assertEqual(sent['payload'], case['payload'])
        self.assertEqual((key, timeout), ('CONTROL_SECRET', 300))
        self.assertEqual(result['new_api_calls'], 1)
        self.assertEqual(result['rows'][0]['text'], 'ALPHA BETA GAMMA')
        self.assertTrue(result['rows'][0]['stream_complete'])
        self.assertEqual(json.loads((self.directory / 'stream/response.json').read_text())['body'], receipt['body'])
        self.assertNotIn('CONTROL_SECRET', ''.join(p.read_text() for p in self.directory.rglob('*.json')))

    def test_unknown_transport_failure_is_saved_and_stops_without_retry(self):
        with mock.patch.object(controls, 'exchange', side_effect=TimeoutError('CONTROL_SECRET')) as send:
            result = self.live(cases=['thinking', 'echo'])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(result['status'], 'error')
        self.assertIsNone(result['rows'][0]['submitted'])
        self.assertEqual(result['rows'][1]['status'], 'not_run')
        self.assertFalse(result['usage_complete'])
        self.assertIn('<redacted>', (self.directory / 'report.json').read_text())

    def test_incomplete_sse_preserves_known_billing_before_stopping(self):
        case = next(c for c in self.data['cases'] if c['id'] == 'stream')
        receipt = json.loads(case['response']['raw'])
        receipt['body'] = receipt['body'].replace('data: [DONE]\r\n\r\n', '')
        with mock.patch.object(controls, 'exchange', return_value=receipt):
            result = self.live(cases=['stream', 'echo'])
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .0000084)
        self.assertEqual(result['rows'][1]['status'], 'not_run')

    def test_http_200_confirms_submission_even_with_invalid_usage(self):
        with mock.patch.object(controls, 'exchange', return_value=dict(http_status=200, body='{}')):
            result = self.live(cases=['thinking'])
        self.assertTrue(result['rows'][0]['submitted'])
        self.assertEqual(result['new_api_calls'], 1)
        self.assertFalse(result['usage_complete'])

    def test_valid_billing_survives_malformed_choice(self):
        receipt = json.loads(self.data['cases'][0]['response']['raw'])
        body = json.loads(receipt['body'])
        body['choices'][0]['text'] = 12
        receipt['body'] = json.dumps(body)
        with mock.patch.object(controls, 'exchange', return_value=receipt):
            result = self.live(cases=['thinking'])
        self.assertEqual(result['status'], 'error')
        self.assertTrue(result['usage_complete'])
        self.assertEqual(result['rows'][0]['usage'], body['usage'])

    def test_transport_uses_no_redirect_and_exact_route_without_saving_headers(self):
        case = self.data['cases'][0]
        response = mock.MagicMock(status=200)
        response.read.return_value = b'{"usage":{}}'
        response.__enter__.return_value = response
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(controls.urllib.request, 'build_opener', return_value=opener) as build:
            result = controls.exchange(case, 'SECRET', 12)
        build.assert_called_once_with(controls.worker.NoRedirect)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, 'https://api.deepinfra.com/v1/openai/completions')
        self.assertEqual(json.loads(request.data), case['payload'])
        self.assertEqual(request.get_header('Authorization'), 'Bearer SECRET')
        self.assertNotIn('headers', result)

    def test_native_preflight_checks_original_tokenizer_and_exact_rendered_payload(self):
        runtime = json.loads(self.data['native']['runtime']['raw'])
        path = self.directory.parent / 'runtime.json'
        path.write_text(json.dumps(runtime))
        expected = self.data['cases'][-1]['payload']
        with mock.patch.object(controls.worker.r, 'tokenizer_files'), \
             mock.patch.object(controls.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps([expected]))) as child:
            settings = controls.native_preflight(self.data, path, '/tokenizer/python', (.4, .8))
            specification = json.loads(child.call_args.kwargs['input'])
            self.assertNotIn('DEEPINFRA_API_KEY', child.call_args.kwargs['env'])
            self.assertEqual(specification['requests'][0]['messages'], json.loads(self.data['native']['inputs']['raw'])['initial'][0])
            self.assertEqual(settings['output_protocol'], 'gemma_native_text')
            child.return_value.stdout = json.dumps([dict(expected, max_tokens=1)])
            with self.assertRaisesRegex(ValueError, 'different payload'):
                controls.native_preflight(self.data, path, '/tokenizer/python', (.4, .8))

    def test_native_pipeline_wires_original_worklist_recipe_real_extractor_and_default_verifier(self):
        loop = importlib.import_module('src.data-pipeline.stage03.generation_loop')
        inputs = json.loads(self.data['native']['inputs']['raw'])
        with mock.patch.object(loop, 'run', return_value={'status': 'completed'}) as run:
            controls.native_pipeline(self.data, Path('/runtime.json'), '/generation/python', self.directory)
        self.assertEqual(run.call_args.args[0], inputs['worklist'])
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs['initial_messages'], inputs['initial'])
        self.assertEqual(kwargs['recipe'], inputs['recipe'])
        self.assertIsInstance(kwargs['program_extractor'], native_multi_edit.Extractor)
        self.assertEqual(kwargs['program_extractor'].originals, ['print(1)'])
        self.assertNotIn('candidate_checker', kwargs)
        self.assertEqual((kwargs['source'], kwargs['fixtures']), ('synthetic', True))

    def test_native_live_uses_pipeline_receipt_and_requires_expected_rejection(self):
        receipt = json.loads(self.data['cases'][-1]['response']['raw'])
        outcome = dict(status='completed', attempts=[dict(extraction='success',
            candidate=dict(program_sha256=sha256(b'print(2)').hexdigest()), verification=dict(status='failed'))])
        def pipeline(data, runtime, python, target):
            path = target / 'pipeline/generation-0000'
            path.mkdir(parents=True)
            (path / 'response-0000.json').write_text(json.dumps(receipt))
            return outcome
        with mock.patch.object(controls, 'native_preflight', return_value=dict(max_total_tokens=4096)), \
             mock.patch.object(controls, 'native_pipeline', side_effect=pipeline) as execute, \
             mock.patch.object(controls, 'exchange', side_effect=AssertionError('Use pipeline')):
            result = self.live(cases=['native-pipeline'], runtime='/runtime.json', generation_python='/python')
        execute.assert_called_once()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['rows'][0]['verification']['status'], 'failed')
        self.assertAlmostEqual(result['new_api_cost_usd'], .0000464)

    def test_native_pre_dispatch_failure_uses_saved_not_submitted_evidence(self):
        def pipeline(data, runtime, python, target):
            path = target / 'pipeline/generation-0000'
            path.mkdir(parents=True)
            (path / 'outcome.json').write_text(json.dumps(dict(requests=[dict(submitted=False)])))
            return dict(status='error')
        with mock.patch.object(controls, 'native_preflight', return_value=dict(max_total_tokens=4096)), \
             mock.patch.object(controls, 'native_pipeline', side_effect=pipeline):
            result = self.live(cases=['native-pipeline'], runtime='/runtime.json', generation_python='/python')
        self.assertEqual(result['status'], 'error')
        self.assertFalse(result['rows'][0]['submitted'])
        self.assertTrue(result['usage_complete'])
        self.assertEqual(result['new_api_calls'], 0)


if __name__ == '__main__':
    unittest.main()
