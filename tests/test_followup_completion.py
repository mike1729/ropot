import json
from pathlib import Path
import tempfile
import unittest

from experiments.complete_followups import missing_cells, source_rows


class CompletionTests(unittest.TestCase):
    def test_completed_low_quality_draws_are_not_selected_for_retry(self):
        rows = [dict(name='2k-r0-b8', status='success', finish_counts={'length':33}, error=None),
                dict(name='7k-r1-b32', status='error', error="TimeoutError('Soft benchmark window deadline')")]
        pending = {r[0] for r in missing_cells(rows,4)}
        self.assertNotIn('2k-r0-b8',pending)
        self.assertIn('7k-r1-b32',pending)
        self.assertEqual(len(pending),13)

    def test_continuation_rejects_native_integrity_or_memory_failures(self):
        for error in ('MemoryError()', 'RuntimeError("shared cache mutated")'):
            with self.assertRaises(ValueError):
                missing_cells([dict(name='7k-r1-b32',status='error',error=error)],4)

    def test_time_censored_calibration_is_not_redrawn(self):
        pending=missing_cells([dict(name='7k-calibration',status='error',error='TimeoutError()')],2)
        self.assertNotIn('7k-calibration',{r[0] for r in pending})
        self.assertIn('7k-r0-b8',{r[0] for r in pending})

    def test_unindexed_native_failure_cannot_bypass_continuation_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            (source/'summary.json').write_text('[]')
            child=source/'7k-r0-b32';child.mkdir()
            (child/'summary.json').write_text(json.dumps(dict(status='error',error='RuntimeError("native OOM")')))
            with self.assertRaises(ValueError):
                missing_cells(source_rows(source),2)


if __name__=='__main__':
    unittest.main()
