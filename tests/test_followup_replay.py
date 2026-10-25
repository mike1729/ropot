import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments import run_followup_completion as replay


class ReplayOnlyTests(unittest.TestCase):
    def test_native_failure_recovery_launches_only_cpu_and_preserves_native_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);repo=base/'repo';run=base/'run';run.mkdir()
            files=['experiments/example.py','docs/inference-followup-continuation.md',
                   'src/data-pipeline/stage03/verification.py','src/data-pipeline/stage03/inspect_runner.py']
            for name in files:
                p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
            (run/'plan.json').write_text('{"native_evidence":true}')
            with patch.object(replay.driver,'ROOT',repo),patch.object(replay.driver,'FROZEN',{}), \
                 patch.object(replay.driver,'launch',return_value='job') as launch,patch.object(replay.driver,'wait'):
                replay.run(SimpleNamespace(source=str(base/'original'),output=str(run),replay_only=True))
            self.assertEqual(launch.call_count,2)
            self.assertTrue(all(c.args[2][0]==replay.driver.PYTHON for c in launch.call_args_list))
            self.assertTrue(json.loads((run/'replay-plan.json').read_text())['replay_only'])
            self.assertEqual((run/'plan.json').read_text(),'{"native_evidence":true}')
            self.assertTrue((run/'replay-sources/experiments/example.py').exists())


if __name__=='__main__':
    unittest.main()
