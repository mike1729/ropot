"""Acquisition with pinned fabricated publications, native stubs and real fake CLIs."""
from collections import Counter
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tests import test_audits as audit_fixtures
from tests.test_witnesses import record

c = importlib.import_module("src.data-pipeline.stage04.acquisition")
a, p, w = c.audits, c.proposals, c.witnesses


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the fake executables, including provider/error/usage behavior.
        cls.cli = audit_fixtures.CLIAuditTests()
        cls.cli.setUp()
        cls.addClassCleanup(cls.cli.doCleanups)
        cls.cli.qualification_behavior()
        cls.qualified = cls.cli.base / 'qualified'
        report = a.qualify(cls.qualified, models=cls.cli.models, executables=cls.cli.executables, timeout=10)
        assert report['passed']

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cli.behavior()
        self.records = {}
        for source, count in [('actor', 98), ('synthetic', 75)]:
            for i in range(count):
                r = record(i, source)
                r['task'] = dict(question=f'PUBLIC_STATEMENT {i}', cases=[dict(input='2\n')],
                                 secret='PRIVATE_VERIFICATION')
                r['stored_inputs'] += ['PRIVATE_STORED_INPUT']
                self.records[source, i] = r
        self.manifest_dir = self.root / 'benigns'
        self.manifest_dir.mkdir()
        p.generation.save(self.manifest_dir / 'manifest.json', dict(selections=[
            r['selection'] for r in self.records.values()]))
        self.patch(w, 'BENIGN_DIR', self.manifest_dir)
        self.patch(w, 'SELECTION_SHA256', c._hash(self.manifest_dir / 'manifest.json'))
        self.loader = self.patch(w, 'load_worklist', side_effect=lambda keys: [
            deepcopy(self.records[source, i]) for split, i, source in keys])
        self.model = self.root / 'model'
        self.model.mkdir()
        p.generation.save(self.model / 'config.json', dict(model_type='qwen3_5'))
        self.model_manifest = self.root / 'model.json'
        self.model_manifest.write_text('{}')
        self.generated, self.checked = [], []
        self.draw_number = 0
        self.text = lambda i, seed: f'<witness>\n {seed}\r\n</witness>'
        self.check_status = 'execution_agrees'
        self.native_status = 'success'
        self.native_submitted = None
        self.child = self.patch(p.control, 'run_child', side_effect=self.generate)
        self.checker = self.patch(w, 'check_many', side_effect=self.check)

    def patch(self, obj, name, *args, **kwargs):
        patch = mock.patch.object(obj, name, *args, **kwargs)
        self.addCleanup(patch.stop)
        return patch.start()

    def scope(self, source='actor', ids=(0, 1)):
        return [deepcopy(self.records[source, i]['selection']['candidate']) for i in ids]

    def generate(self, command, progress, persist):
        spec = c._read(command[-1])
        self.generated.append(deepcopy(spec['requests']))
        self.draw_number += 1
        native = Path(spec['output_dir'])
        native.mkdir()
        rows = []
        for i, request in enumerate(spec['requests']):
            submitted = self.native_submitted[i] if self.native_submitted is not None else True
            status = 'stop' if submitted else 'not_run'
            text = self.text(i, request['seed']) if submitted else ''
            if submitted:
                (native / f'reply-{i:04d}.txt').write_bytes(text.encode())
            rows.append(dict(status=status, submitted=submitted, generation_tokens=3 if submitted else 0, text=text))
        p.generation.save(native / 'outcome.json', dict(status=self.native_status, requests=rows,
            preparation_seconds=.1, native=dict(wall_time=.2)))
        p.generation.save(native / 'runtime.json', dict(load_seconds=.1))
        return 0

    def check(self, proposals, directory):
        self.checked.extend(stdin for _, stdin in proposals)
        return [dict(status=self.check_status) for _ in proposals]

    def acquire(self, name='run', **kwargs):
        source = kwargs.get('source', 'actor')
        p.generation.save(self.model / 'config.json', dict(model_type='qwen3_5' if source == 'actor' else 'mistral3'))
        settings = dict(source='actor', qualification_dir=self.qualified,
            model_path=self.model, model_manifest=self.model_manifest, mlx_python='unused-python',
            completion_batch_size=8, prefill_batch_size=4, prefill_step_size=256,
            reply_allowance=6, output_token_allowance=6 * 32, audit_call_allowance=12,
            recipe=dict(p.DEFAULT_RECIPE, max_tokens=32), fixtures=True)
        if 'previous_run' not in kwargs:
            settings['scope'] = self.scope(source)
        settings.update(kwargs)
        return c.acquire(self.root / name, **settings)

    def calls(self, run='run'):
        saved = c._read(self.root / run / 'outcome.json')
        return [Path(row['path']) for row in saved['audits']['calls']]

    def b1(self, states=('execution_agrees', 'duplicate'), source='actor', submitted=None):
        """Two source rounds whose attempt numbers and seeds both start at zero/17."""
        directory = self.root / 'b1'
        directory.mkdir()
        originals, overlay = {}, []
        for i, state in enumerate(states):
            original = self.root / f'original-{i}'
            (original / 'generation').mkdir(parents=True)
            reply = original / 'generation/reply-0000.txt'
            reply.write_text(f'<witness>\n{1000+i}\n</witness>')
            stdin = directory / f'recovered-{i}.txt'
            stdin.write_text(f'{1000+i}\n')
            attempt = dict(candidate=self.scope(source, (i,))[0], attempt=0, seed=17,
                submitted=True if submitted is None else submitted[i], state=state,
                input=str(stdin), stdin_sha256=c._hash(stdin))
            out = dict(status='completed', attempts=[dict(attempt, state='format_failure')],
                committed_replies=1, committed_tokens=32, known_submissions=int(attempt['submitted'] is True),
                observed_tokens=3, usage_complete=attempt['submitted'] is not None)
            p.generation.save(original / 'inputs.json', dict(source=source))
            p.generation.save(original / 'outcome.json', out)
            originals[str(original / 'outcome.json')] = c._hash(original / 'outcome.json')
            overlay.append(dict(attempt, original_outcome=str(original / 'outcome.json')))
        p.generation.save(directory / 'inputs.json', dict(original_sha256=originals))
        p.generation.save(directory / 'outcomes.json', overlay)
        return directory

    def test_one_round_exact_stdin_privacy_qualified_settings_and_zero_slot_audits(self):
        result = self.acquire()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['state'] for r in result['programs']], ['approved'] * 2)
        self.assertEqual([r['attempts'] for r in result['programs']], [1, 1])
        self.assertEqual(self.checked, [' 17\r\n', ' 18\r\n'])
        self.assertEqual(result['audits']['new_calls'], 4)
        self.assertEqual(result['generation']['committed_replies'], 2)
        self.assertEqual(result['generation']['observed_tokens'], 6)
        batches = list((self.root / 'run').glob('round-*/audits/inputs.json'))
        self.assertEqual(len(batches), 1)
        settings = c._read(batches[0])
        self.assertEqual(settings['timeout'], 10)
        self.assertEqual({a: cfg['model'] for a, cfg in settings['auditors'].items()}, self.cli.models)
        for packet in settings['packets']:
            text = (batches[0].parent / (packet + '.txt')).read_text()
            self.assertNotIn('PRIVATE_', text)
            self.assertNotIn('program_sha256', text)
        before = len(self.generated)
        resumed = self.acquire('resumed', previous_run=self.root / 'run', reply_allowance=0, audit_call_allowance=0)
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(len(self.generated), before)
        self.assertEqual(resumed['audits']['new_calls'], 0)
        self.assertEqual(resumed['audits']['total_calls'], 4)

    def test_b1_overlay_counts_once_audits_saved_input_then_retries_only_rejection(self):
        b1 = self.b1()
        result = self.acquire(b1_dir=b1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['attempts'] for r in result['programs']], [1, 2])
        self.assertEqual([[r['seed'] for r in requests] for requests in self.generated], [[116]])
        self.assertEqual(result['generation']['committed_replies'], 3)
        self.assertEqual(result['generation']['committed_tokens'], 96)
        messages = self.generated[0][0]['messages']
        self.assertEqual(messages[-2:], [dict(role='assistant', content='<witness>\n1001\n</witness>'),
                                       dict(role='user', content=c.DIFFERENT)])
        self.assertNotIn('1000', json.dumps(messages))
        self.assertEqual(result['audits']['new_calls'], 4)

    def test_full_pinned_source_order_seeds_for_synthetic(self):
        result = self.acquire(source='synthetic', b1_dir=self.b1(('duplicate', 'duplicate'), source='synthetic'))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['seed'] for r in self.generated[0]], [92, 93])
        self.assertEqual([r['attempts'] for r in result['programs']], [2, 2])

    def test_shrinking_subset_preserves_position_and_source_scope_order(self):
        gold = {json.dumps(['PUBLIC_STATEMENT 0', ' 17\r\n']): True,
                json.dumps(['PUBLIC_STATEMENT 1', ' 18\r\n']): False,
                json.dumps(['PUBLIC_STATEMENT 1', ' 116\r\n']): True}
        self.cli.behavior(codex=dict(gold=gold))
        result = self.acquire(scope=self.scope(ids=(1, 0)))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['attempts'] for r in result['programs']], [1, 2])
        self.assertEqual([[r['seed'] for r in requests] for requests in self.generated], [[17, 18], [116]])
        self.assertEqual(self.generated[1][0]['messages'][-1]['content'], c.VALIDITY)

    def test_format_and_execution_feedback_are_projected_from_completed_outcomes(self):
        self.text = lambda i, seed: 'bad framing' if self.draw_number == 1 else f'<witness>\n{seed}\n</witness>'
        self.checker.side_effect = lambda rows, directory: [dict(status='benign_disagreement'
            if self.draw_number == 2 else 'execution_agrees') for _ in rows]
        result = self.acquire()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['attempts'] for r in result['programs']], [3, 3])
        self.assertEqual([m['content'] for m in self.generated[2][0]['messages'][3::2]], [c.FORMAT, c.EXECUTION])

    def test_capacity_failure_stops_without_retry_then_resumes_same_input(self):
        self.cli.behavior(codex=dict(capacity=True))
        first = self.acquire(audit_call_allowance=300)
        self.assertEqual(first['status'], 'error')
        self.assertEqual(first['audits']['new_calls'], 1)
        self.assertEqual(first['audits']['unknown_usage_calls'], 1)
        self.assertEqual(len(self.generated), 1)
        batch = next((self.root / 'run').glob('round-*/audits/inputs.json'))
        self.assertEqual(c._read(batch)['call_allowance'], 8)
        self.cli.behavior()
        resumed = self.acquire('resumed', previous_run=self.root / 'run', audit_call_allowance=7)
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual(resumed['audits']['new_calls'], 4)
        self.assertEqual(resumed['audits']['total_calls'], 5)
        self.assertEqual([r['attempts'] for r in resumed['programs']], [1, 1])
        self.assertEqual(len(self.generated), 1)

    def test_exhausted_saved_ceiling_cannot_generate_a_replacement(self):
        self.cli.behavior(codex=dict(capacity=True))
        first = self.acquire(scope=self.scope(ids=(0,)), audit_call_allowance=2)
        self.assertEqual(first['audits']['new_calls'], 1)
        second = self.acquire('second', previous_run=self.root / 'run')
        self.assertEqual(second['audits']['new_calls'], 1)
        self.cli.behavior()
        third = self.acquire('third', previous_run=self.root / 'second')
        self.assertEqual(third['status'], 'error')
        self.assertEqual(third['audits']['new_calls'], 0)
        self.assertEqual(third['programs'][0]['attempts'], 1)
        self.assertEqual(len(self.generated), 1)

    def test_three_rounds_share_one_audit_total_and_every_batch_gets_remaining(self):
        self.cli.behavior(codex=dict(novelty='reject'))
        result = self.acquire(audit_call_allowance=10)
        self.assertEqual(result['status'], 'error')
        self.assertIn('two remaining', result['error'])
        self.assertEqual(result['audits']['new_calls'], 8)
        self.assertEqual([r['attempts'] for r in result['programs']], [3, 3])
        self.assertEqual([c._read(q)['call_allowance'] for q in sorted(
            (self.root / 'run').glob('round-*/audits/inputs.json'))], [8, 6])
        self.assertEqual([[r['seed'] for r in requests] for requests in self.generated],
                         [[17, 18], [115, 116], [213, 214]])
        self.cli.behavior()
        resumed = self.acquire('resumed', previous_run=self.root / 'run', reply_allowance=0, audit_call_allowance=4)
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual([r['state'] for r in resumed['programs']], ['approved', 'approved'])
        self.assertEqual(resumed['audits']['new_calls'], 4)
        self.assertEqual(resumed['audits']['total_calls'], 12)
        self.assertEqual(len(self.generated), 3)

    def test_third_rejection_closes_without_fourth_draw_and_feedback_is_safe(self):
        self.cli.behavior(codex=dict(validity='reject'))
        result = self.acquire()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['state'] for r in result['programs']], ['attempts_exhausted'] * 2)
        self.assertEqual(result['audits']['new_calls'], 12)
        self.assertEqual(len(self.generated), 3)
        messages = self.generated[2][0]['messages']
        self.assertEqual([m['content'] for m in messages[3::2]], [c.VALIDITY, c.VALIDITY])
        self.assertNotIn('Fixture constraint judgment', json.dumps(messages))
        self.assertNotIn('PRIVATE_', json.dumps(messages))

    def test_audit_failure_after_rejection_stops_then_resumes_same_packet(self):
        self.cli.behavior(codex=dict(validity='reject'), claude=dict(exit=True))
        result = self.acquire()
        self.assertEqual(result['status'], 'error')
        self.assertEqual(len(self.generated), 1)
        self.assertEqual(result['audits']['new_calls'], 2)
        self.assertEqual(result['audits']['unknown_usage_calls'], 1)
        old_calls = self.calls()
        self.cli.behavior()
        short = self.acquire('too-small', previous_run=self.root / 'run', audit_call_allowance=5)
        self.assertEqual(short['status'], 'error')
        self.assertIn('unused ceiling', short['error'])
        self.assertEqual(short['audits']['new_calls'], 0)
        self.assertEqual(len(self.generated), 1)
        result = self.acquire('resumed', previous_run=self.root / 'too-small', audit_call_allowance=12)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.generated), 2)
        self.assertEqual(len(self.generated[1]), 1)
        self.assertEqual([r['attempts'] for r in result['programs']], [2, 1])
        self.assertTrue(all(q.exists() for q in old_calls))
        self.assertEqual(result['audits']['new_calls'], 5)

    def test_missing_audit_response_is_unavailable_after_two_unusable_calls(self):
        self.cli.behavior(codex=dict(malformed=True))
        first = self.acquire()
        self.assertEqual(first['status'], 'error')
        second = self.acquire('second', previous_run=self.root / 'run')
        self.assertEqual(second['status'], 'error')
        self.cli.behavior()
        third = self.acquire('third', previous_run=self.root / 'second')
        self.assertEqual(third['status'], 'completed')
        self.assertEqual(third['programs'][0]['state'], 'audit_unavailable')
        self.assertEqual(third['programs'][0]['attempts'], 1)
        self.assertEqual(len(self.generated), 1)

    def test_audit_snapshot_reads_each_record_once_after_qualified_decisions(self):
        self.acquire()
        batch = next((self.root / 'run').glob('round-*/audits/inputs.json')).parent
        settings, _ = c._qualification(self.qualified)
        paths = sorted(batch.glob('calls/*/*/call-*'))
        reads = dict(text=Counter(), bytes=Counter())
        globs = Counter()
        original_text, original_bytes, original_glob = Path.read_text, Path.read_bytes, Path.glob

        def read_text(path, *args, **kwargs):
            if path.name == 'record.json':
                reads['text'][path] += 1
            return original_text(path, *args, **kwargs)

        def read_bytes(path, *args, **kwargs):
            if path.name == 'record.json':
                reads['bytes'][path] += 1
            return original_bytes(path, *args, **kwargs)

        def glob(path, pattern, *args, **kwargs):
            if path == batch:
                globs[pattern] += 1
            return original_glob(path, pattern, *args, **kwargs)

        with mock.patch.object(Path, 'read_text', new=read_text), \
                mock.patch.object(Path, 'read_bytes', new=read_bytes), \
                mock.patch.object(Path, 'glob', new=glob):
            _, pending, calls = c._audit_state([batch], settings)
        expected = Counter({path / 'record.json': 1 for path in paths})
        self.assertEqual(reads, dict(text=expected, bytes=expected))
        self.assertEqual(globs, {'calls/*/*/call-*': 1})
        self.assertEqual([path for path, _ in calls], paths)
        self.assertEqual(pending, [])

    def test_missing_audit_record_still_consumes_saved_ceiling_and_unknown_usage(self):
        self.cli.behavior(codex=dict(capacity=True))
        self.acquire(scope=self.scope(ids=(0,)), audit_call_allowance=2)
        call, = self.calls()
        (call / 'record.json').unlink()
        batch = call.parents[3]
        settings, _ = c._qualification(self.qualified)
        _, pending, calls = c._audit_state([batch], settings)
        self.assertEqual(pending, [(batch, 1)])
        self.assertEqual(calls, [(call, {})])
        with mock.patch.object(a, 'dispatch') as dispatch:
            resumed = self.acquire('resumed', previous_run=self.root / 'run', audit_call_allowance=0)
        dispatch.assert_not_called()
        self.assertEqual(resumed['status'], 'error')
        self.assertIn('unused ceiling', resumed['error'])
        self.assertEqual({k: resumed['audits'][k] for k in ('new_calls', 'total_calls', 'unknown_usage_calls')},
                         dict(new_calls=0, total_calls=1, unknown_usage_calls=1))
        self.assertEqual(len(self.generated), 1)

    def test_audit_snapshots_reread_invalid_records_successes_and_vetoes(self):
        self.acquire(scope=self.scope(ids=(0,)))
        call = next(path for path in self.calls() if path.parent.name == 'codex')
        batch = call.parents[3]
        settings, _ = c._qualification(self.qualified)
        successful = c._read(call / 'record.json')
        invalid = dict(successful, status='invalid_reply')
        p.generation.save(call / 'record.json', invalid)
        decisions, pending, _ = c._audit_state([batch], settings)
        row, = decisions.values()
        self.assertEqual((row['status'], row['unavailable'], pending), ('pending', False, [(batch, 2)]))

        retry = call.with_name('call-1')
        retry.mkdir()
        p.generation.save(retry / 'record.json', invalid)
        decisions, pending, _ = c._audit_state([batch], settings)
        row, = decisions.values()
        self.assertEqual((row['status'], row['unavailable'], pending), ('unresolved', True, []))

        p.generation.save(call / 'record.json', successful)
        decisions, pending, _ = c._audit_state([batch], settings)
        row, = decisions.values()
        self.assertEqual((row['status'], row['unavailable'], pending), ('approved', False, []))

        p.generation.save(batch / 'vetoes.json', {row['packet_id']: 'Late review'})
        decisions, pending, _ = c._audit_state([batch], settings)
        row, = decisions.values()
        self.assertEqual((row['status'], row['reason'], pending), ('not_approved', 'veto', []))

    def test_qualification_digest_and_model_or_effort_mismatch_prevent_dispatch(self):
        qualified = c._qualification
        for error in ('digest', 'model', 'effort'):
            with self.subTest(error=error):
                # Mutate a private copy of the qualification, not its shared evidence.
                import shutil
                path = self.root / ('qualified-' + error)
                shutil.copytree(self.qualified, path)
                saved = c._read(path / 'inputs.json')
                if error == 'digest':
                    saved['source_sha256'] = '0' * 64
                else:
                    saved['auditors']['codex'][error] = 'different'
                p.generation.save(path / 'inputs.json', saved)
                with self.assertRaises(ValueError):
                    self.acquire(qualification_dir=path)
        self.assertFalse((self.root / 'run').exists())
        self.child.assert_not_called()
        self.assertTrue(qualified(self.qualified))

    def test_unpassed_qualification_and_model_overrides_cannot_dispatch(self):
        with mock.patch.object(a, 'qualification', return_value=dict(passed=False)):
            with self.assertRaises(ValueError):
                self.acquire()
        with self.assertRaises(TypeError):
            self.acquire(models=dict(codex='wrong', claude='wrong'))
        self.child.assert_not_called()

    def test_current_implementation_digest_must_match_passing_qualification(self):
        real_hash = c._hash
        with mock.patch.object(c, '_hash', side_effect=lambda path:
                '0' * 64 if Path(path) == Path(a.__file__) else real_hash(path)):
            with self.assertRaisesRegex(ValueError, 'unchanged audits.py'):
                self.acquire()
        self.child.assert_not_called()
        self.assertFalse((self.root / 'run').exists())

    def test_cli_version_drift_is_flagged_including_failed_calls(self):
        self.cli.behavior(codex=dict(version='2.0'), claude=dict(version='2.0', exit=True))
        result = self.acquire()
        self.assertEqual(result['status'], 'error')
        differences = result['audits']['version_differences']
        self.assertEqual([d['actual'] for d in differences], ['fake-claude 2.0', 'fake-codex 2.0'])
        self.cli.behavior(codex=dict(version='2.0'), claude=dict(version='2.0'))
        result = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.generated), 1)
        self.assertTrue(result['audits']['version_differences'])

    def test_scope_and_frozen_input_or_reply_changes_refused(self):
        self.acquire()
        for scope in (self.scope(ids=(0,)), self.scope(ids=(1, 0))):
            with self.assertRaisesRegex(ValueError, 'scope'):
                self.acquire('bad-scope', previous_run=self.root / 'run', scope=scope)
        outcome = c._read(self.root / 'run/outcome.json')
        for path in (next(Path(x) for x in outcome['files'] if '/generation/reply-' in x),
                     next(Path(x) for x in outcome['files'] if Path(x).name == 'input-0000.txt')):
            old = path.read_bytes()
            path.write_bytes(old + b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed acquisition evidence'):
                self.acquire('changed', previous_run=self.root / 'run')
            path.write_bytes(old)
        self.assertEqual(len(self.generated), 1)

    def test_failed_check_adds_no_turns_but_extracted_input_is_a_repeat(self):
        self.text = lambda i, seed: '<witness>\n7\n</witness>'
        self.checker.side_effect = RuntimeError('Docker failure')
        first = self.acquire()
        self.assertEqual(first['status'], 'error')
        self.checker.side_effect = self.check
        second = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(second['status'], 'completed')
        self.assertEqual([r['state'] for r in second['programs']], ['attempts_exhausted'] * 2)
        self.assertEqual([r['attempts'] for r in second['programs']], [3, 3])
        self.assertEqual(len(self.generated[1][0]['messages']), 2)
        self.assertEqual(self.generated[2][0]['messages'][-1]['content'], c.DIFFERENT)
        self.assertEqual(second['audits']['new_calls'], 0)
        self.assertEqual(self.checker.call_count, 3)  # last two calls have no inputs
        self.assertEqual(self.checked, [])

    def test_known_unstarted_does_not_consume_slot_but_reservations_persist(self):
        self.native_status, self.native_submitted = 'error', [True, False]
        first = self.acquire()
        self.assertEqual([r['attempts'] for r in first['programs']], [1, 0])
        self.native_status, self.native_submitted = 'success', None
        second = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(second['status'], 'completed')
        self.assertEqual([r['attempts'] for r in second['programs']], [2, 1])
        self.assertEqual([r['seed'] for r in self.generated[1]], [115, 18])
        self.assertEqual([len(r['messages']) for r in self.generated[1]], [2, 2])
        self.assertEqual(second['generation']['committed_replies'], 4)
        self.assertEqual(second['generation']['known_submissions'], 3)
        self.assertFalse(second['generation']['usage_complete'])

    def test_uncertain_submissions_are_charged_across_failed_runs(self):
        self.child.side_effect = RuntimeError('No trustworthy native outcome')
        previous = None
        for i in range(3):
            result = self.acquire(str(i), **(dict(previous_run=previous) if previous else {}))
            self.assertEqual(result['status'], 'error')
            self.assertEqual([r['attempts'] for r in result['programs']], [i+1, i+1])
            previous = self.root / str(i)
        self.assertEqual(result['generation']['uncertain_submissions'], 6)
        self.assertIsNone(result['generation']['known_submissions'])
        last = self.acquire('last', previous_run=previous)
        self.assertEqual(last['status'], 'completed')
        self.assertEqual([r['state'] for r in last['programs']], ['attempts_exhausted'] * 2)
        self.assertEqual(self.child.call_count, 3)

    def test_snapshot_failure_retains_zero_reservations_and_can_resume(self):
        save = p.generation.save
        def failed(path, value):
            if Path(path).name == 'task-0000.json':
                raise OSError('snapshot write failed')
            return save(path, value)
        with mock.patch.object(p.generation, 'save', side_effect=failed):
            first = self.acquire()
        self.assertEqual(first['status'], 'error')
        self.assertEqual(first['generation']['committed_replies'], 0)
        self.assertEqual([r['attempts'] for r in first['programs']], [0, 0])
        self.child.assert_not_called()
        resumed = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual([r['attempts'] for r in resumed['programs']], [1, 1])

    def test_partial_b1_scope_labels_whole_round_cost_instead_of_scope_cost(self):
        b1 = self.b1(('execution_agrees',))
        overlay = c._read(b1 / 'outcomes.json')
        original = Path(overlay[0]['original_outcome'])
        outcome = c._read(original)
        outcome['attempts'].append(dict(outcome['attempts'][0], candidate=self.scope(ids=(1,))[0], attempt=1))
        outcome.update(committed_replies=2, committed_tokens=64, observed_tokens=6, known_submissions=2)
        p.generation.save(original, outcome)
        saved = c._read(b1 / 'inputs.json')
        saved['original_sha256'][str(original)] = c._hash(original)
        p.generation.save(b1 / 'inputs.json', saved)
        result = self.acquire(scope=self.scope(ids=(0,)), b1_dir=b1, reply_allowance=0)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['generation']['cost_basis'], 'whole_referenced_rounds')
        self.assertEqual(result['generation']['out_of_scope_attempts'], 1)
        self.assertEqual(result['generation']['committed_replies'], 2)
        self.assertEqual(result['programs'][0]['attempts'], 1)
        self.child.assert_not_called()

    def test_evidence_collection_failure_retains_the_original_error(self):
        original = c._evidence
        failures = []
        def fail_after_proposal(*args):
            if (self.root / 'run/round-0000/proposals/outcome.json').exists():
                failures.append(True)
                raise ValueError('first failure' if len(failures) == 1 else 'cleanup failure')
            return original(*args)
        with mock.patch.object(c, '_evidence', side_effect=fail_after_proposal):
            result = self.acquire()
        self.assertEqual(result['status'], 'error')
        self.assertIn('first failure', result['error'])
        self.assertIn('cleanup failure', result['collection_error'])
        self.assertEqual(result, c._read(self.root / 'run/outcome.json'))

    @unittest.skipUnless(Path('data/04_witness_experiments/qualification-cli-b3ea36c-20261027-4d32ae7c').exists(),
                         'requires recorded real qualification')
    def test_recorded_qualification_enforces_its_saved_source_digest_without_calls(self):
        directory = Path('data/04_witness_experiments/qualification-cli-b3ea36c-20261027-4d32ae7c')
        saved = c._read(directory / 'inputs.json')
        if saved['source_sha256'] != c._hash(a.__file__):
            with self.assertRaisesRegex(ValueError, 'unchanged audits.py'):
                c._qualification(directory)
        else:
            settings, versions = c._qualification(directory)
            self.assertEqual(settings['source_sha256'], c._hash(a.__file__))
            self.assertEqual(set(versions), set(a.AUDITORS))
        self.child.assert_not_called()

    def test_veto_after_approval_allows_replacement_without_snapshot(self):
        self.acquire()
        batch = next((self.root / 'run').glob('round-*/audits/inputs.json')).parent
        packet = next(iter(c._read(batch / 'inputs.json')['packets']))
        p.generation.save(batch / 'vetoes.json', {packet: 'PRIVATE_HUMAN_REASON'})
        result = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([r['attempts'] for r in result['programs']], [2, 1])
        self.assertEqual(self.generated[1][0]['messages'][-1]['content'], c.DIFFERENT)
        self.assertNotIn('PRIVATE_HUMAN_REASON', json.dumps(self.generated))

    def test_exhaustion_stops_before_generation_and_audits(self):
        candidate = self.records['actor', 0]['selection']['candidate']
        candidate['task_sha256'] = next(iter(w.EXHAUSTED_TASKS))
        p.generation.save(self.manifest_dir / 'manifest.json', dict(selections=[r['selection'] for r in self.records.values()]))
        self.patch(w, 'SELECTION_SHA256', new=c._hash(self.manifest_dir / 'manifest.json'))
        result = self.acquire(scope=self.scope(ids=(0,)), reply_allowance=0, audit_call_allowance=0)
        self.assertEqual(result['programs'][0]['state'], 'input_domain_exhausted')
        self.assertEqual(result['status'], 'completed')
        self.child.assert_not_called()

    def test_insufficient_generation_or_initial_audit_allowances_do_not_dispatch(self):
        for name, changes in [('reply', dict(reply_allowance=1)), ('token', dict(output_token_allowance=63))]:
            result = self.acquire(name, **changes)
            self.assertEqual(result['status'], 'error')
        self.child.assert_not_called()
        result = self.acquire('audit', b1_dir=self.b1(('execution_agrees', 'execution_agrees')),
                              audit_call_allowance=3)
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['audits']['new_calls'], 0)
        self.child.assert_not_called()

    def test_b1_duplicate_association_and_mutated_original_are_refused(self):
        b1 = self.b1()
        path = b1 / 'outcomes.json'
        rows = c._read(path)
        p.generation.save(path, [*rows, rows[0]])
        with self.assertRaisesRegex(ValueError, 'Duplicate or mismatched'):
            self.acquire(b1_dir=b1)
        p.generation.save(path, rows)
        original = Path(rows[0]['original_outcome'])
        original.write_text(original.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'Changed acquisition evidence'):
            self.acquire(b1_dir=b1)
        self.child.assert_not_called()

    def test_interrupt_keeps_recoverable_run_and_propagates(self):
        self.checker.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.acquire()
        result = c._read(self.root / 'run/outcome.json')
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual([r['attempts'] for r in result['programs']], [1, 1])
        self.checker.side_effect = self.check
        resumed = self.acquire('resumed', previous_run=self.root / 'run')
        self.assertEqual(resumed['status'], 'completed')
        self.assertEqual([r['attempts'] for r in resumed['programs']], [2, 2])


if __name__ == '__main__':
    unittest.main()
