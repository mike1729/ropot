"""Expanded benign-panel preparation and information boundaries, without inference."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import qwen_dev_benigns as d
from scripts import qwen_benign_retry_worker as retry


class DevBenignTests(unittest.TestCase):
    def test_construction_exposure_includes_retained_thinking_and_tool_arguments(self):
        source = d.e.ROOT / 'data/03_benign_experiments/qwen38-construction50-20261030/prepare_exposure.py'
        if not source.exists():
            self.skipTest('Construction exposure preparation not available')
        parsed = ast.parse(source.read_text())
        namespace = {}
        helper = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'strings')
        exec(compile(ast.Module(body=[helper], type_ignores=[]), '<exposure-helper>', 'exec'), namespace)
        values = list(namespace['strings'](dict(content=None, reasoning_content='measured input 42',
            tool_calls=[dict(function=dict(arguments=dict(old='stdin 17', new='stdin 23')))])))
        self.assertEqual(values, ['measured input 42', 'stdin 17', 'stdin 23'])

    def test_construction_selection_uses_exactly_first_fifty_initial_ids(self):
        cohort = d.a.study_tasks.load_cohort()
        order = cohort['groups']['construction']['initial'][:50]
        histories = {}
        for identifier in order:
            binding, task = d.a.study_tasks.load_task(cohort, identifier)
            histories[str(identifier)] = dict(task_sha256=binding['sha256'],
                known=[c['index'] for c in task['cases']], disclosed=[])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'exposure.json').write_text(json.dumps(dict(tasks=histories, files={})))
            (root / 'prepare_exposure.py').write_text('# exposure fixture\n')
            with mock.patch.object(d.e, 'load_panel', return_value=([], [], {}, {})):
                contract, worklist, messages = d.prepare(root, 'construction')
        self.assertEqual(contract['kind'], 'qwen38-construction-benigns')
        self.assertEqual(contract['task_order'], order)
        self.assertEqual([v[0]['id'] for v in worklist], order)
        self.assertFalse(set(order) & set(cohort['groups']['development']['initial']))
        self.assertEqual(contract['recipe']['base_seed'], 3000041)
        self.assertEqual(contract['output_token_limit'], 512000)
        for entry, prompt in zip(contract['entries'], messages):
            shown = json.loads(prompt[1]['content'].split('Shown cases:\n')[1])
            self.assertFalse({' '.join(c['input'].split()) for c in shown} &
                             {' '.join(c['input'].split()) for c in entry['partition']['audit']})

    def test_retry_preserves_only_complete_identical_requests(self):
        rows = [dict(status=s, submitted=s != 'not_run')
                for s in ('stop', 'length', 'error', 'not_run')]
        self.assertEqual(set(retry.reusable([1, 2, 3, 4], [1, 2, 3, 4], rows)), {0, 1})
        with self.assertRaises(ValueError):
            retry.reusable([9, 2, 3, 4], [1, 2, 3, 4], rows)

    def test_retry_chunks_preserve_order_and_do_not_resubmit_completed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'previous/acquisition/actor/generation-0000'
            source.mkdir(parents=True)
            requests = [dict(seed=i) for i in range(7)]
            rows = [dict(status='stop' if i < 2 else 'not_run', submitted=i < 2,
                         generation_tokens=i, text=str(i)) for i in range(7)]
            d.g.save(source.parent / 'round-0000.json', dict(requests=requests))
            d.g.save(source / 'outcome.json', dict(requests=rows))
            output = root / 'retry/acquisition/actor/generation-0000'
            output.parent.mkdir(parents=True)
            d.g.save(root / 'retry/contract.json', dict(sources={str(spec): d.g.file_hash(spec)
                for spec in [source.parent / 'round-0000.json']}, retry=dict(
                source=str(root / 'previous'), worker_request_limit=4)))
            specification = output.parent / 'round-0000.json'
            d.g.save(specification, dict(requests=requests, output_dir=str(output)))
            dispatched = []
            def launch(command):
                chunk = d.loop.read(Path(command[-1]))
                dispatched.extend(r['seed'] for r in chunk['requests'])
                native = Path(chunk['output_dir']); native.mkdir()
                d.g.save(native / 'outcome.json', dict(status='success', native=dict(wall_time=1),
                    requests=[dict(status='stop', submitted=True, generation_tokens=1, text=str(r['seed']))
                              for r in chunk['requests']]))
                return mock.Mock(pid=123, wait=mock.Mock(return_value=0))
            with mock.patch.object(retry.subprocess, 'Popen', side_effect=launch):
                self.assertEqual(retry.run(specification), 0)
            result = d.loop.read(output / 'outcome.json')
            self.assertEqual(dispatched, [2, 3, 4, 5, 6])
            self.assertEqual([r['text'] for r in result['requests']], list(map(str, range(7))))
            self.assertEqual([c['indices'] for c in result['chunks']], [[2, 3, 4, 5], [6]])
            contract = d.loop.read(root / 'retry/contract.json')
            contract.pop('retry')
            contract['worker_request_limit'] = 4
            d.g.save(root / 'retry/contract.json', contract)
            fresh = output.with_name('generation-fresh')
            d.g.save(specification, dict(requests=requests, output_dir=str(fresh)))
            dispatched.clear()
            with mock.patch.object(retry.subprocess, 'Popen', side_effect=launch):
                self.assertEqual(retry.run(specification), 0)
            self.assertEqual(dispatched, list(range(7)))
            self.assertEqual(d.loop.read(fresh / 'outcome.json')['reused'], [])

    def test_all_fifty_are_retained_and_audits_stay_out_of_shown_prompts(self):
        exposure = d.e.ROOT / 'data/03_benign_experiments/qwen38-dev50-20261029/exposure.json'
        if not exposure.exists():
            self.skipTest('Local development exposure snapshot unavailable')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'exposure.json').write_bytes(exposure.read_bytes())
            (root / 'prepare_exposure.py').write_text('# exposure fixture\n')
            with mock.patch.object(d.e, 'load_panel', return_value=([], [], {}, {})), \
                    mock.patch.object(d.loop, 'check_files'):
                contract, worklist, messages = d.prepare(root)
        self.assertEqual(len(contract['entries']), 50)
        self.assertEqual([v[0]['id'] for v in worklist], contract['development_order'])
        self.assertEqual((contract['reply_limit'], contract['output_token_limit']), (150, 512000))
        self.assertEqual((contract['recipe']['max_tokens'], contract['recipe']['repair_max_tokens']), (2048, 4096))
        self.assertFalse(contract['recipe']['enable_thinking'])
        self.assertTrue(contract['case_ineligible'] if 'case_ineligible' in contract else
                        any(not e['attack_case_eligible'] for e in contract['entries']))
        old = d.loop.read(d.e.DEFAULT_BENIGNS / 'contract.json')
        by_id = {v['binding']['id']: v for v in contract['entries']}
        for previous in old['entries']:
            self.assertEqual(by_id[previous['binding']['id']]['partition'], previous['partition'])
        for entry, (_, task), prompt in zip(contract['entries'], worklist, messages):
            shown = json.loads(prompt[1]['content'].split('Shown cases:\n')[1])
            shown_inputs = {' '.join(c['input'].split()) for c in shown}
            self.assertFalse(shown_inputs & {' '.join(c['input'].split()) for c in entry['partition']['audit']})
            self.assertEqual(len(task['cases']), len(shown))

    def test_duplicate_dispatch_is_refused_before_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'contract.json').write_text('{}')
            with self.assertRaises(FileExistsError), mock.patch.object(d, 'prepare') as prepare:
                d.run(root)
            prepare.assert_not_called()
