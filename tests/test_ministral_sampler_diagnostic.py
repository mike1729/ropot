import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock
from experiments import ministral_sampler_diagnostic as D


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name, value in dict(OUTPUT=self.root / 'scores.jsonl', ORIGINAL_SOURCE=self.root / 'original.py', CANDIDATES=(0, 1)).items():
            patch = mock.patch.object(D, name, value); patch.start(); self.addCleanup(patch.stop)
    def records(self):
        return [json.loads(line) for line in D.OUTPUT.read_text().splitlines()]
    def row(self):
        return dict(row_index=0, fixture_id='short', prompt_ids=[7], expected_tokens=[0] * 11, prefix=[], native_dtype='bfloat16')
    def test_pins_before_execution_and_original_namespace(self):
        baseline = D.load_baseline()
        with mock.patch.object(D, 'BASELINE_SHA256', '0' * 64), self.assertRaises(RuntimeError): D.load_baseline()
        source = self.root / 'worker.py'; raw = b'identity = (__name__, __file__)\n'; source.write_bytes(raw)
        baseline['original_module'] = dict(path=str(source), sha256='0' * 64)
        with mock.patch('builtins.exec') as execute, self.assertRaises(RuntimeError): D.load_original(baseline)
        execute.assert_not_called(); self.assertFalse(D.ORIGINAL_SOURCE.exists())
        baseline['original_module']['sha256'] = hashlib.sha256(raw).hexdigest()
        module = D.load_original(baseline)
        self.assertEqual(module.identity[1], str(source)); self.assertNotEqual(module.identity[0], '__main__')
        self.assertEqual(D.ORIGINAL_SOURCE.read_bytes(), raw)
    def test_durable_bounded_output_and_clean_stdout(self):
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(D.os, 'fsync', wraps=D.os.fsync) as synced: D.append(dict(ok=True))
        synced.assert_called_once(); self.assertEqual(output.getvalue(), '')
        previous = D.OUTPUT.read_bytes()
        with mock.patch.object(D, 'LIMIT', len(previous)), self.assertRaises(RuntimeError): D.append(dict(next=True))
        self.assertEqual(D.OUTPUT.read_bytes(), previous); self.assertEqual(self.records(), [dict(ok=True)])
        with mock.patch.object(D, 'LIMIT', len(previous) * 2): D.append(dict(ok=True))
        self.assertEqual(D.OUTPUT.stat().st_size, len(previous) * 2)
    def test_native_scores_lossless_vectors_and_nonfinite_stop(self):
        row = self.row()
        for index in range(11): D.observe('q0', row, index, [0., -1.25], 0)
        record = self.records()[-1]
        self.assertEqual(record['prefix'], [0] * 10); self.assertEqual(record['prompt_ids'], [7])
        self.assertEqual(record['native_dtype'], 'bfloat16'); self.assertEqual(record['margin'], 1.25)
        self.assertEqual(record['top2'][0], dict(token=0, score=0., prob=1.))
        vector = record['vector']; self.assertEqual((vector['dtype'], vector['count']), ('<f4', 2))
        self.assertEqual(base64.b64decode(vector['data_base64']), struct.pack('<2f', 0., -1.25))
        with self.assertRaises(RuntimeError): D.observe('q1', self.row(), 0, [float('nan'), float('inf'), -float('inf')], 0)
        self.assertEqual(self.records()[-1]['event'], 'nonfinite'); self.assertEqual(self.records()[-1]['nonfinite'], dict(nan=1, posinf=1, neginf=1))
        decoded = struct.unpack('<3f', base64.b64decode(D.encode_scores([float('nan'), float('inf'), -float('inf')])['data_base64']))
        self.assertNotEqual(decoded[0], decoded[0]); self.assertEqual(decoded[1:], (float('inf'), -float('inf')))
    def test_distinct_index_prefix_argmax_and_branch_rejections(self):
        for index, scores, token, expected in [(1, [0., -1.], 0, 0), (0, [0., -1.], 0, 1), (0, [0., -1.], 1, 1)]:
            row = self.row(); row['expected_tokens'][0] = expected
            with self.subTest(index=index, token=token, expected=expected), self.assertRaises(RuntimeError): D.observe('q0', row, index, scores, token)
            self.assertEqual(self.records()[-1]['event'], 'mismatch'); self.assertEqual(row['prefix'], [])
        row = self.row(); row['prefix'] = [1] * 10
        with self.assertRaises(RuntimeError): D.observe('q0', row, 10, [0., -1.], 0)
    def test_hooks_preserve_calls_results_counters_and_restore(self):
        scores = NS(dtype='bfloat16', tolist=lambda: [[0., -1.]])
        result = NS(item=lambda: 0); calls = []
        def sample(value):
            self.assertIs(value, scores); calls.append(value); return result
        worker = NS(sampler_for=mock.Mock(return_value=sample))
        expected = dict(tokens=[0] * 11, text='x', finish_reason='length')
        baseline = dict(fixtures=[dict(id=name, prompt_ids=[7]) for name in ('short', 'long')], expected=dict(q0=[expected], q1=[expected], q2=[expected, expected]))
        def run(model, tokenizer, fixtures, window, emit, **kwargs):
            for row in window['rows']:
                sampler = worker.sampler_for(row, 2, 3)
                for _ in range(12): self.assertIs(sampler(scores), result)
            print(json.dumps(dict(event='window_done', window=window['id'])))
            return baseline['expected'].get(window['id'], [expected])
        worker.run_window = run; original_factory = worker.sampler_for
        protocol = io.StringIO()
        with D.install_hooks(worker, baseline), redirect_stdout(protocol):
            for name, ids in [('q0', ['short']), ('q1', ['long']), ('q2', ['short', 'long']), ('q3', ['short']), ('qlong', ['long'])]:
                window = dict(id=name, rows=[dict(fixture_id=id_, sampling='greedy') for id_ in ids])
                worker.run_window(None, None, {f['id']: f for f in baseline['fixtures']}, window, mock.Mock())
        self.assertIs(worker.run_window, run); self.assertIs(worker.sampler_for, original_factory); self.assertEqual(len(calls), 72)
        self.assertEqual([json.loads(line)['window'] for line in protocol.getvalue().splitlines()], ['q0', 'q1', 'q2', 'q3', 'qlong'])
        records = self.records(); self.assertEqual(len(records), 44)
        for window, row in [('q0', 0), ('q1', 0), ('q2', 0), ('q2', 1)]:
            observed = [r for r in records if r['window'] == window and r['row_index'] == row]
            self.assertEqual([r['index'] for r in observed], list(range(11)))
        with self.assertRaisesRegex(ValueError, 'later'):
            with D.install_hooks(worker, baseline): raise ValueError('later')
        self.assertIs(worker.run_window, run); self.assertIs(worker.sampler_for, original_factory)
    def test_changed_rendered_prompt_rejected_before_original_window(self):
        original = mock.Mock(); worker = NS(run_window=original, sampler_for=mock.Mock())
        fixture = dict(id='short', prompt_ids=[7]); baseline = dict(fixtures=[fixture], expected=dict(q0=[dict(tokens=[0] * 11)]))
        with D.install_hooks(worker, baseline), self.assertRaisesRegex(RuntimeError, 'Rendered fixture'):
            worker.run_window(None, None, dict(short=dict(fixture, prompt_ids=[8])), dict(id='q0', rows=[]), mock.Mock())
        original.assert_not_called(); self.assertIs(worker.run_window, original)
