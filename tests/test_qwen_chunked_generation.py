"""Chunked generation merges one-batch worker outcomes and stops on the first failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import qwen_chunked_generation as c


class ChunkedGenerationTests(unittest.TestCase):
    def run_round(self, count, fail_offset=None):
        directory = Path(tempfile.mkdtemp())
        spec = dict(model_path='m', completion_batch_size=8, output_dir=str(directory / 'generation-0000'),
                    requests=[dict(seed=i) for i in range(count)])
        (directory / 'round.json').write_text(json.dumps(spec))
        calls = []
        def worker(command):
            chunk = json.loads(Path(command[-1]).read_text())
            calls.append([r['seed'] for r in chunk['requests']])
            out = Path(chunk['output_dir']); out.mkdir()
            failed = fail_offset is not None and chunk['requests'][0]['seed'] == fail_offset
            (out / 'outcome.json').write_text(json.dumps(dict(
                status='error' if failed else 'success', peak_memory_bytes=len(calls),
                requests=[dict(status='stop', submitted=True, text=str(r['seed'])) for r in chunk['requests']],
                native=dict(prompt_tokens=10, generation_tokens=2, wall_time=1.0))))
            return mock.Mock(returncode=1 if failed else 0)
        with mock.patch.object(c.subprocess, 'run', side_effect=worker):
            try:
                c.run(directory / 'round.json')
            except RuntimeError:
                pass
        return calls, json.loads((directory / 'generation-0000/outcome.json').read_text())

    def test_round_splits_into_eight_request_workers_and_merges_in_order(self):
        calls, outcome = self.run_round(19)
        self.assertEqual([len(x) for x in calls], [8, 8, 3])
        self.assertEqual(outcome['status'], 'success')
        self.assertEqual([r['text'] for r in outcome['requests']], [str(i) for i in range(19)])
        self.assertEqual(outcome['native']['prompt_tokens'], 30)
        self.assertEqual(outcome['peak_memory_bytes'], 3)

    def test_failed_chunk_stops_without_retry_and_keeps_unstarted_rows(self):
        calls, outcome = self.run_round(19, fail_offset=8)
        self.assertEqual(len(calls), 2)
        self.assertEqual(outcome['status'], 'error')
        self.assertEqual(outcome['requests'][16]['status'], 'not_run')
