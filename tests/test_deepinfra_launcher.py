"""Offline finite-job preparation, pins and one-shot sequential launch accounting."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import launch_deepinfra as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'prepared'
        self.models = dict(qwen=launcher.runtime_api.MODEL, gemma=launcher.runtime_api.GEMMA_MODEL)
        self.runtimes, self.runtime_values = {}, {}
        for family, model in self.models.items():
            tokenizer = self.root / f'{family}-tokenizer'
            tokenizer.mkdir()
            config = (dict(model_type='qwen3_5', text_config=dict(eos_token_id=248044)) if family == 'qwen'
                      else dict(model_type='gemma4', eos_token_id=[1, 106, 50], text_config=dict(eos_token_id=1)))
            for name in launcher.runtime_api.TOKENIZER_FILES[:4]:
                (tokenizer / name).write_text(json.dumps(config) if name == 'config.json' else '{}')
            runtime = dict(model=model, tokenizer_path=str(tokenizer), tokenizer_sha256={
                name: launcher.generation.file_hash(tokenizer / name) for name in launcher.runtime_api.TOKENIZER_FILES[:4]},
                timeout_seconds=5, max_total_tokens=100, input_usd_per_million=.2,
                output_usd_per_million=.4, max_cost_usd=.01)
            if family == 'gemma':
                runtime['output_protocol'] = 'gemma_native_text'
            path = self.root / f'{family}-runtime.json'
            path.write_text(json.dumps(runtime))
            self.runtimes[family], self.runtime_values[family] = path, runtime
        requests = [dict(messages=[dict(role='system', content='Return Python'),
                                   dict(role='user', content=f'Already exposed fixture {i}')],
                         seed=173+i*4, temperature=1., top_p=.95, top_k=20, enable_thinking=True)
                    for i in range(2)]
        self.plan = dict(kind='fixture-finite-plan', rounds={family: dict(model=model.partition(':')[0],
                         requests=deepcopy(requests)) for family, model in self.models.items()})
        self.plan_path = self.root / 'plan.json'
        self.plan_path.write_text(json.dumps(self.plan))
        self.evidence = self.root / 'reviewed-limits-evidence.json'
        self.evidence.write_text('{"provider_output_cap":80,"context_cap":100}')
        self.limits = {family: dict(model=model, max_output_tokens=80, max_total_tokens=100,
            evidence={str(self.evidence): launcher.generation.file_hash(self.evidence)})
            for family, model in self.models.items()}
        self.limits_path = self.root / 'limits.json'
        self.limits_path.write_text(json.dumps(self.limits))
        self.changed_payload_family = None
        self.failed_render_family = None
        self.failed_worker_family = None
        self.child_families = []
        self.actual_preflight = launcher._preflight
        self.preflight = mock.patch.object(launcher, '_preflight', side_effect=self.render).start()
        self.child = mock.patch.object(launcher.loop, 'run_child', side_effect=self.generate).start()
        mock.patch.dict(os.environ, DEEPINFRA_API_KEY='PRIVATE_LAUNCHER_TEST_KEY').start()
        self.addCleanup(mock.patch.stopall)

    def family(self, specification):
        return next(f for f, model in self.models.items() if specification['runtime']['model'] == model)

    def render(self, specification, python):
        family = self.family(specification)
        if family == self.failed_render_family:
            raise ValueError('Tokenizer/template preflight failed')
        requests = specification['requests']
        payloads = [sha256(json.dumps(dict(model=specification['runtime']['model'], request=r),
                    sort_keys=True).encode()).hexdigest() for r in requests]
        if family == self.changed_payload_family:
            payloads[-1] = '0' * 64
        return dict(versions=dict(transformers='5.23.0', tokenizers='0.23.3', jinja2='3.1.6'),
                    prompt_tokens=[10, 20], payload_sha256=payloads,
                    planned_max_cost_usd=(30*.2 + sum(r['max_tokens'] for r in requests)*.4)/1e6)

    def prepare(self, **changes):
        options = dict(expected_plan_sha256=launcher.generation.file_hash(self.plan_path), runtimes=self.runtimes,
            generation_python=sys.executable, max_tokens=dict(qwen=64, gemma=64), max_cost_usd=.01,
            limits_path=self.limits_path)
        options.update(changes)
        return launcher.prepare(self.plan_path, self.output, **options)

    def pin(self):
        return launcher.generation.file_hash(self.output / 'launch.json')

    def launch(self, **changes):
        return launcher.launch(self.output, expected_sha256=changes.get('expected_sha256', self.pin()))

    def generate(self, command, current, persist):
        self.assertEqual(command[:2], [str(Path(sys.executable).absolute()), str(launcher.runtime_api.WORKER)])
        specification = json.loads(Path(command[-1]).read_text())
        family = self.family(specification)
        self.child_families.append(family)
        output = Path(specification['output_dir'])
        self.assertEqual(output, self.output / family)
        output.mkdir()
        rows = [dict(status='stop', submitted=True, generation_tokens=2,
                     usage=dict(estimated_cost=.00001)) for _ in specification['requests']]
        failed = family == self.failed_worker_family
        if failed:
            rows[1] = dict(status='error', submitted=None, generation_tokens=0)
        result = dict(status='error' if failed else 'success', requests=rows, usage_complete=not failed,
                      observed_estimated_cost_usd=.00001 if failed else .00002,
                      error='Fixture ambiguous submission' if failed else None)
        (output / 'outcome.json').write_text(json.dumps(result))
        return 1 if failed else 0

    def test_prepare_needs_no_key_and_success_preserves_families_seeds_caps_and_paths(self):
        with mock.patch.dict(os.environ, DEEPINFRA_API_KEY=''):
            manifest = self.prepare()
        self.child.assert_not_called()
        self.assertEqual(self.preflight.call_count, 2)
        self.assertEqual(list(manifest['rounds']), ['qwen', 'gemma'])
        for family, job in manifest['rounds'].items():
            saved = json.loads(Path(job['round_file']).read_text())
            self.assertEqual(saved['runtime']['model'], self.models[family])
            self.assertEqual(saved['output_dir'], str(self.output / family))
            self.assertEqual(saved['requests'], [dict(r, max_tokens=64) for r in self.plan['rounds'][family]['requests']])
        self.assertAlmostEqual(manifest['reserved_max_cost_usd'], .0001824)
        result = self.launch()
        self.assertEqual(result['status'], 'success', result)
        self.assertEqual(self.child_families, ['qwen', 'gemma'])
        self.assertTrue(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .00004)
        self.assertEqual([r['known_submissions_lower_bound'] for r in result['rounds'].values()], [2, 2])
        self.assertEqual(self.preflight.call_count, 4)
        with self.assertRaises(FileExistsError):
            self.launch()
        self.assertEqual(self.child.call_count, 2)

    def test_aggregate_ceiling_rejects_even_when_each_individual_round_fits(self):
        with self.assertRaisesRegex(ValueError, 'together'):
            self.prepare(max_cost_usd=.00015)
        self.assertEqual(self.preflight.call_count, 2)
        self.assertFalse(self.output.exists())
        self.child.assert_not_called()

    def test_explicit_plan_pin_is_required_before_any_preflight(self):
        with self.assertRaisesRegex(ValueError, 'plan'):
            self.prepare(expected_plan_sha256='0' * 64)
        self.preflight.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_caps_models_and_limit_evidence_refuse_before_output(self):
        cases = [dict(max_tokens=dict(qwen=True, gemma=64)), dict(max_tokens=dict(qwen=81, gemma=64)),
                 dict(max_tokens=dict(qwen=64)), dict(runtimes={'unknown': self.runtimes['qwen']})]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(**changes)
            self.assertFalse(self.output.exists())
            self.child.assert_not_called()
        original = deepcopy(self.limits)
        for field, value in [('max_output_tokens', 0), ('max_total_tokens', 99), ('evidence', {}),
                             ('model', self.models['gemma'])]:
            with self.subTest(field=field):
                self.limits = deepcopy(original)
                self.limits['qwen'][field] = value
                self.limits_path.write_text(json.dumps(self.limits))
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.child.assert_not_called()

    def test_wrong_plan_family_model_or_conflicting_frozen_cap_refuses(self):
        original = deepcopy(self.plan)
        for mutation in ('family', 'model', 'cap'):
            with self.subTest(mutation=mutation):
                self.plan = deepcopy(original)
                if mutation == 'family':
                    self.plan['rounds']['unknown'] = self.plan['rounds'].pop('qwen')
                elif mutation == 'model':
                    self.plan['rounds']['qwen']['model'] = self.models['gemma']
                else:
                    self.plan['rounds']['qwen']['requests'][0]['max_tokens'] = 32
                self.plan_path.write_text(json.dumps(self.plan))
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.child.assert_not_called()

    def test_second_model_render_failure_prevents_all_dispatch_and_output(self):
        self.failed_render_family = 'gemma'
        with self.assertRaisesRegex(ValueError, 'preflight'):
            self.prepare()
        self.assertEqual(self.preflight.call_count, 2)
        self.assertFalse(self.output.exists())
        self.child.assert_not_called()

    def test_gemma_requires_explicit_native_text_protocol_before_dispatch(self):
        original = deepcopy(self.runtime_values['gemma'])
        for protocol in (None, 'token_ids'):
            with self.subTest(protocol=protocol):
                runtime = deepcopy(original)
                if protocol is None:
                    runtime.pop('output_protocol')
                else:
                    runtime['output_protocol'] = protocol
                self.runtimes['gemma'].write_text(json.dumps(runtime))
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.child.assert_not_called()

    def test_conflicting_limit_evidence_is_not_hidden_by_runtime_pins(self):
        self.limits['qwen']['evidence'] = {str(self.runtimes['qwen']): '0' * 64}
        self.limits_path.write_text(json.dumps(self.limits))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.prepare()
        self.preflight.assert_not_called()
        self.assertFalse(self.output.exists())
        self.child.assert_not_called()

    def test_conflicting_plan_input_is_not_hidden_by_runtime_pins(self):
        self.plan['inputs'] = {str(self.runtimes['qwen']): '0' * 64}
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, 'evidence'):
            self.prepare()
        self.preflight.assert_not_called()
        self.assertFalse(self.output.exists())
        self.child.assert_not_called()

    def test_changed_runtime_evidence_plan_or_round_prevents_launch(self):
        manifest = self.prepare()
        pin = self.pin()
        paths = [self.runtimes['qwen'], self.evidence, self.plan_path,
                 Path(manifest['rounds']['gemma']['round_file'])]
        for path in paths:
            raw = path.read_bytes()
            try:
                path.write_bytes(raw + b'\n')
                with self.subTest(path=path), self.assertRaises(ValueError):
                    self.launch(expected_sha256=pin)
                self.child.assert_not_called()
                self.assertFalse((self.output / 'outcome.json').exists())
            finally:
                path.write_bytes(raw)

    def test_changed_launcher_source_pin_prevents_any_child(self):
        self.prepare()
        pin = self.pin()
        real_hash = launcher.generation.file_hash
        source = Path(launcher.__file__).resolve()
        with mock.patch.object(launcher.generation, 'file_hash', side_effect=lambda p:
                '0' * 64 if Path(p).resolve() == source else real_hash(p)):
            with self.assertRaises(ValueError):
                self.launch(expected_sha256=pin)
        self.child.assert_not_called()

    def test_both_payload_preflights_repeat_before_any_child(self):
        self.prepare()
        self.changed_payload_family = 'gemma'
        with self.assertRaisesRegex(ValueError, 'Rendered'):
            self.launch()
        self.assertEqual(self.preflight.call_count, 4)
        self.child.assert_not_called()
        self.assertFalse((self.output / 'outcome.json').exists())

    def test_first_worker_failure_preserves_cost_and_never_launches_the_second(self):
        self.prepare()
        self.failed_worker_family = 'qwen'
        result = self.launch()
        self.assertEqual(result['status'], 'error', result)
        self.assertEqual(self.child_families, ['qwen'])
        self.assertEqual(self.child.call_count, 1)
        self.assertFalse(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .00001)
        self.assertEqual(result['rounds']['qwen']['known_submissions_lower_bound'], 1)
        self.assertEqual(result['rounds']['gemma']['status'], 'not_run')
        self.assertFalse((self.output / 'gemma').exists())
        self.assertEqual(json.loads((self.output / 'outcome.json').read_text()), result)

    def test_concurrent_launchers_have_one_owner_and_cannot_overwrite_accounting(self):
        self.prepare()
        pin = self.pin()
        barrier = threading.Barrier(2, timeout=3)
        def synchronized_render(specification, python):
            result = self.render(specification, python)
            if self.family(specification) == 'gemma':
                barrier.wait()
            return result
        self.preflight.side_effect = synchronized_render
        results, errors = [], []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.launch, expected_sha256=pin) for _ in range(2)]
            for future in futures:
                try:
                    results.append(future.result(timeout=5))
                except BaseException as error:
                    errors.append(error)
        self.assertEqual(len(results), 1, errors)
        self.assertEqual([type(error) for error in errors], [FileExistsError])
        result, = results
        self.assertEqual(result['status'], 'success', result)
        self.assertEqual(self.child_families, ['qwen', 'gemma'])
        self.assertEqual(self.child.call_count, 2)
        self.assertTrue(result['usage_complete'])
        self.assertAlmostEqual(result['observed_estimated_cost_usd'], .00004)
        self.assertEqual(json.loads((self.output / 'outcome.json').read_text()), result)
        self.assertEqual((self.output / 'launch.claim').read_text(), pin + '\n')

    def test_claim_survives_initial_persistence_failure_and_blocks_automatic_retry(self):
        self.prepare()
        pin = self.pin()
        with mock.patch.object(launcher.loop, 'atomic_save', side_effect=OSError('Initial progress write failed')):
            with self.assertRaisesRegex(OSError, 'progress write'):
                self.launch(expected_sha256=pin)
        self.child.assert_not_called()
        self.assertFalse((self.output / 'outcome.json').exists())
        self.assertEqual((self.output / 'launch.claim').read_text(), pin + '\n')
        with self.assertRaises(FileExistsError):
            self.launch(expected_sha256=pin)
        self.child.assert_not_called()

    def test_launch_requires_key_only_after_offline_preparation(self):
        with mock.patch.dict(os.environ, DEEPINFRA_API_KEY=''):
            self.prepare()
            with self.assertRaisesRegex(ValueError, 'DEEPINFRA_API_KEY'):
                self.launch()
        self.child.assert_not_called()
        self.assertFalse((self.output / 'outcome.json').exists())

    def test_real_preflight_subprocess_never_receives_the_api_key(self):
        specification = dict(runtime=self.runtime_values['qwen'], requests=[], output_dir='unused')
        expected = dict(versions={}, prompt_tokens=[], payload_sha256=[], planned_max_cost_usd=0)
        with mock.patch.object(launcher.subprocess, 'run', return_value=SimpleNamespace(
                stdout=json.dumps(expected))) as process:
            actual = self.actual_preflight(specification, sys.executable)
        self.assertEqual(actual, expected)
        self.assertNotIn('DEEPINFRA_API_KEY', process.call_args.kwargs['env'])
        self.assertEqual(json.loads(process.call_args.kwargs['input']), specification)
        self.assertTrue(process.call_args.kwargs['check'])
