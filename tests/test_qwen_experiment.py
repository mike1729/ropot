"""Qwen development driver against the real loop; no GPU or Docker."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import qwen_experiment as e


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'experiment'
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{}')
        self.runtime = dict(model_path=str(self.model), model_manifest=str(self.manifest), mlx_python='fixture-python')
        self.entries, self.worklist = [], []
        for i in range(8):
            binding = dict(id=i, split='test', sha256=f'{i:064x}')
            shown = dict(examples=[dict(index=0, input='1\n', answer='1')],
                         menu=[dict(index=j, input=f'{j}\n', answer=str(j)) for j in range(1,5)])
            self.entries.append(dict(binding=binding, partition=shown, benigns={'actor': dict(program='print(1)\n')}))
            self.worklist.append((binding, dict(question=f'Question {i}', cases=[c for group in shown.values() for c in group])))
        self.requests, self.technical_error, self.interrupt = [], False, False
        patches = [mock.patch.object(e, 'load_panel', return_value=(self.entries, self.worklist, self.runtime, {str(self.manifest): e.g.file_hash(self.manifest)})),
                   mock.patch.object(e.a, 'memory_state', return_value=dict(pressure=1, swap_used_mib=1)),
                   mock.patch.object(e.a, 'acquisition_sources', return_value={}),
                   mock.patch.object(e.a, 'check_shown', side_effect=self.check),
                   mock.patch.object(e.loop, 'run_child', side_effect=self.generate)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def generate(self, command, current, persist):
        self.assertEqual(Path(command[1]), e.a.DRIVER)
        spec = e.loop.read(command[-1])
        target = Path(spec['output_dir'])
        target.mkdir(parents=True)
        rows = []
        for request in spec['requests']:
            self.requests.append(deepcopy(request))
            answer = '# case: 1\nprint(2) # comment\n'
            text = 'PRIVATE_REASONING\n' + answer
            if target.parent.name == 'batch':
                text = 'Analysis: checked.\nVerdict: PASS'
            rows.append(dict(status='stop', submitted=True, text=text, answer_text=answer,
                tokens=[1,2], generation_tokens=2, prompt_tokens=11, enable_thinking=request['enable_thinking']))
        e.g.save(target / 'runtime.json', dict(load_seconds=.01))
        e.g.save(target / 'outcome.json', dict(status='error' if self.technical_error else 'success', requests=rows,
                 native=dict(prompt_tokens=10 * len(rows), generation_tokens=2 * len(rows), wall_time=.1)))
        if self.interrupt:
            raise KeyboardInterrupt('test cancellation')
        return 1 if self.technical_error else 0

    def check(self, candidates, log_dir, **kwargs):
        # One first-round rejection exercises fresh repair inputs and seed positions.
        initial = Path(log_dir).name == 'verification-0000' and Path(log_dir).parent.name == 'actor-chain-0'
        return [dict(status='failed' if initial and i == 0 else 'passed', functional=True,
                     key=dict(task_sha256=h, program_sha256=sha256(program).hexdigest()),
                     violations=['token_churn'] if initial and i == 0 else [],
                     cases=[{}] * 5, feedback=dict(message='public feedback')) for i, (_, h, program) in enumerate(candidates)]

    def test_tuned_scaffold_chain_seeds_and_bounded_costs_use_shared_loop(self):
        report = e.run(self.output, dict(chains=2, recipe=dict(base_seed=101, max_repairs=1,
                       max_tokens=32, repair_max_tokens=32, system='Custom system', initial='Custom instruction')))
        self.assertEqual(report['status'], 'completed')
        self.assertEqual([r['seed'] for r in self.requests], list(range(101,109)) + [109] + list(range(117,125)))
        self.assertEqual(report['submitted_replies_lower_bound'], 17)
        self.assertEqual(report['output_tokens'], 34)
        self.assertEqual(report['rule_violations'], {'token_churn': 1})
        self.assertEqual(report['yield']['accepted_tasks'], 8)
        self.assertFalse(report['yield']['reviewed'])
        for request in self.requests:
            content = json.dumps(request['messages'])
            self.assertIn('Custom instruction', content)
            self.assertEqual(request['messages'][0]['content'], 'Custom system')
            self.assertNotIn('PRIVATE_REASONING', content)
        record = e.loop.read(self.output / 'experiment.json')
        outcome = e.loop.read(self.output / 'actor-chain-0/outcome.json')
        for attempt in outcome['attempts']:
            self.assertEqual(Path(attempt['program']).read_bytes(), b'print(2)\n')
            self.assertEqual(attempt['metadata'], dict(case_declaration='1', repair_prefix='# case: 1\n'))
        self.assertEqual((record['reply_limit'], record['output_token_limit']), (32,1024))
        self.assertTrue((self.output / 'evidence.json').is_file())
        with self.assertRaises(FileExistsError):
            e.run(self.output, {})

    def test_technical_error_stops_later_chains_and_keeps_delivered_tokens(self):
        self.technical_error = True
        report = e.run(self.output, dict(chains=2))
        self.assertEqual(report['status'], 'error')
        self.assertEqual(len(report['runs']), 1)
        self.assertEqual(report['observed_output_tokens_lower_bound'], 16)
        self.assertTrue((self.output / 'evidence.json').is_file())

    def test_default_development_seeds_are_outside_the_official_range(self):
        report = e.run(self.output, {})
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(self.requests[0]['seed'], 1_000_041)
        self.assertTrue(all(r['seed'] > 88 for r in self.requests))

    def test_any_potential_submission_in_the_official_seed_range_is_rejected(self):
        for base_seed, chains in ((41, 1), (88, 1), (18, 1), (0, 2)):
            with self.subTest(base_seed=base_seed, chains=chains), self.assertRaisesRegex(ValueError, 'official range'):
                e.run(self.output, dict(chains=chains, recipe=dict(base_seed=base_seed)))
        self.assertFalse(self.output.exists())
        self.assertEqual(self.requests, [])

    def test_development_seed_ranges_can_touch_but_not_enter_the_official_range(self):
        for base_seed in (17, 89):
            with self.subTest(base_seed=base_seed):
                report = e.run(self.root / f'allowed-{base_seed}', dict(recipe=dict(base_seed=base_seed)))
                self.assertEqual(report['status'], 'completed')

    def test_interrupt_preserves_accounting_and_propagates(self):
        self.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            e.run(self.output, {})
        report = e.loop.read(self.output / 'report.json')
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['observed_output_tokens_lower_bound'], 16)
        self.assertTrue((self.output / 'evidence.json').is_file())

    def test_running_report_exists_before_first_dispatch_result(self):
        def interrupt(*args):
            snapshot = e.loop.read(self.output / 'report.json')
            self.assertEqual(snapshot['status'], 'running')
            self.assertEqual(snapshot['output_tokens'], 0)
            self.assertGreaterEqual(snapshot['elapsed_seconds'], 0)
            raise KeyboardInterrupt('before delivery')
        with mock.patch.object(e.loop, 'run_child', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            e.run(self.output, {})
        report = e.loop.read(self.output / 'report.json')
        self.assertEqual(report['status'], 'cancelled')
        self.assertIsNone(report['output_tokens'])
        self.assertEqual(report['observed_output_tokens_lower_bound'], 0)

    def test_monitor_uses_frozen_baseline_and_first_accepted_per_task(self):
        self.entries[0]['benigns']['actor']['program'] = '\n'.join(
            ('if' if i == 0 else 'elif') + f' value == {i}:\n    print({i}) # branch'
            for i in range(400)) + '\n'
        e.run(self.output, dict(chains=2, recipe=dict(max_repairs=0)))
        self.requests.clear()
        config = e.loop.read(e.a.MONITOR_CONFIG)
        config['model']['manifest_sha256'] = e.g.file_hash(self.manifest)
        path = self.root / 'monitor-config.json'
        e.g.save(path, config)
        with mock.patch.object(e.a, 'MONITOR_CONFIG', path), \
             mock.patch.object(e.a.study_tasks, 'load_task', side_effect=lambda _, i: self.worklist[i]):
            result = e.score_monitor(self.output)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.requests), 16)
        self.assertTrue(all(r['seed'] == 17 and not r['enable_thinking'] for r in self.requests))
        self.assertTrue(all(r['max_prompt_tokens'] == 8192 for r in self.requests))
        self.assertEqual(result['groups']['benign']['counts']['PASS'], 8)
        self.assertEqual(result['groups']['attack']['counts']['PASS'], 8)
        self.assertEqual(result['output_tokens'], 32)
        self.assertEqual(result['native_prompt_tokens'], 160)
        self.assertEqual(self.requests[0]['messages'][0]['content'], config['system_prompt'])
        self.assertIn('elif value == 399:', self.requests[0]['messages'][1]['content'])
        self.assertNotIn('# branch', self.requests[0]['messages'][1]['content'])

    def test_monitor_report_exists_before_first_dispatch_result(self):
        e.run(self.output, dict(recipe=dict(max_repairs=0)))
        config = e.loop.read(e.a.MONITOR_CONFIG)
        config['model']['manifest_sha256'] = e.g.file_hash(self.manifest)
        path = self.root / 'monitor-config.json'
        e.g.save(path, config)
        def interrupt(*args):
            snapshot = e.loop.read(self.output / 'monitor/report.json')
            self.assertEqual(snapshot['status'], 'running')
            self.assertEqual(snapshot['output_tokens_lower_bound'], 0)
            self.assertIsNone(snapshot['native_prompt_tokens'])
            self.assertEqual(snapshot['elapsed_seconds'], 0)
            raise KeyboardInterrupt('before monitor delivery')
        with mock.patch.object(e.a, 'MONITOR_CONFIG', path), \
             mock.patch.object(e.a.study_tasks, 'load_task', side_effect=lambda _, i: self.worklist[i]), \
             mock.patch.object(e.loop, 'run_child', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            e.score_monitor(self.output)
        result = e.loop.read(self.output / 'monitor/report.json')
        self.assertEqual(result['status'], 'cancelled')
        self.assertIsNone(result['output_tokens'])
        self.assertEqual(result['output_tokens_lower_bound'], 0)

    def test_memory_pressure_prevents_dispatch(self):
        with mock.patch.object(e.a, 'memory_state', return_value=dict(pressure=2, swap_used_mib=1)):
            report = e.run(self.output, {})
        self.assertEqual(report['status'], 'error')
        self.assertEqual(self.requests, [])

    def test_invalid_settings_fail_before_loading_or_creating_output(self):
        for config in ({'chains':0}, {'chains':True}, {'unknown':1}, {'recipe':{'max_tokens':8193}},
                       {'recipe':{'initial':''}}, {'recipe':{'temperature':float('nan')}},
                       {'recipe':{'top_p':0}}, {'recipe':{'typo':1}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                e.run(self.output, config)
        self.assertFalse(self.output.exists())


class PanelTests(unittest.TestCase):
    def test_real_b1_panel_projects_only_shown_cases_and_actor_benigns(self):
        if not (e.DEFAULT_BENIGNS / 'report.json').exists():
            self.skipTest('local B1 artifacts unavailable')
        entries, worklist, _, _ = e.load_panel(e.DEFAULT_BENIGNS)
        self.assertEqual(len(entries), 8)
        for entry, (_, task) in zip(entries, worklist):
            self.assertEqual(set(entry['partition']), {'examples','menu'})
            self.assertEqual(set(entry['benigns']), {'actor'})
            self.assertTrue(all(c['visible'] for c in task['cases']))
            self.assertEqual(len(task['cases']), len(entry['partition']['examples']) + len(entry['partition']['menu']))


if __name__ == '__main__':
    unittest.main()
