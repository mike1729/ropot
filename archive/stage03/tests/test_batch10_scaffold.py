"""Paired scaffold scheduling and reporting, with no inference or supplied code."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import batch10_scaffold as c


def outcome(passes, status='completed'):
    attempts = [dict(attempt=j, position=position, round=round_, state='passed' if passed else 'failed',
                     extraction='success') for j,(position,round_,passed) in enumerate(passes)]
    return dict(status=status, attempts=attempts, tasks=[], rounds=[], committed_replies=len(attempts),
                committed_tokens=len(attempts)*2048, known_submissions=len(attempts),
                observed_tokens=len(attempts)*100, usage_complete=True)


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)/'evidence'
        self.directory.mkdir()
        self.root = Path(tmp.name)/'experiments'
        self.manifest = Path(tmp.name)/'manifest.json'
        self.manifest.write_text('{}')
        self.tasks = [(dict(id=i,split='test',sha256='a'*64),dict(question=f'Statement {i}')) for i in range(32)]
        self.options = dict(c.OPTIONS,model_manifest=self.manifest)
        c.g.save(self.directory/'sizing.json',dict(source_sha256={},recipe=c.RECIPE,bindings=[t[0] for t in self.tasks],
            model_manifest_sha256=c.g.file_hash(self.manifest),within_development_maximum=True,fits_context=True))
        for patcher in (patch.object(c,'source_hashes',return_value={}),patch.object(c,'worklist',return_value=self.tasks),
                        patch.object(c,'OPTIONS',self.options)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def launch(self):
        c.run(self.directory,self.root)

    def test_frozen_block_reads_only_the_32_expected_ids(self):
        # Independent of the patched schedule worklist: read the saved publication.
        cohort = c.b.study_tasks.load_cohort()
        expected = [595,1102,4015,623,788,1559,4004,842,1528,510,4688,1924,2334,1736,4692,334,
                    1544,2113,4231,2090,967,866,4272,53,1833,1344,4338,1003,1440,2376,4704,2048]
        with patch.object(c.b.study_tasks,'load_task',side_effect=lambda _,i: (cohort['tasks_by_id'][i],{})) as load:
            # Recover the function without replacing the active patch.
            import importlib.util
            spec=importlib.util.spec_from_file_location('block_only',c.__file__)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result=module.worklist()
        self.assertEqual([binding['id'] for binding,_ in result],expected)
        self.assertEqual([call.args[1] for call in load.call_args_list],expected)

    def test_cycles_use_early_unresolved_set_even_if_a_later_passed(self):
        a = outcome([(i,0,i==0) for i in range(32)]+[(i,1,i==1) for i in range(1,32)]+[(2,4,True)])
        b2 = outcome([(i,0,i==0) for i in range(30)]+[(i,1,False) for i in range(1,30)])
        b3 = outcome([(i,0,False) for i in range(29)]+[(i,1,i==0) for i in range(29)])
        with patch.object(c.b,'run',side_effect=[a,b2,b3]) as run:
            self.launch()
        calls = run.call_args_list
        self.assertEqual([[t[0]['id'] for t in call.args[0]] for call in calls],
                         [list(range(32)),list(range(2,32)),list(range(3,32))])
        self.assertEqual([call.kwargs['recipe']['base_seed'] for call in calls],[17,100017,200017])
        self.assertEqual([call.kwargs['recipe']['temperature'] for call in calls],[.2,.7,.7])
        self.assertEqual([call.kwargs['recipe']['max_repairs'] for call in calls],[5,1,1])
        self.assertEqual([call.kwargs['reply_allowance'] for call in calls],[192,60,58])
        self.assertEqual([call.kwargs['output_token_allowance'] for call in calls],[393216,122880,118784])
        self.assertEqual([call.args[1].name for call in calls],['0001-batch10-a','0002-batch10-b2','0003-batch10-b3'])
        report=json.loads((self.directory/'comparison.json').read_text())
        self.assertEqual(report['curves']['A'],[1,2,2,2,3,3])
        self.assertEqual(report['curves']['B'],[1,2,3,3,3,4])
        self.assertEqual(report['only_B'],[3])
        self.assertEqual(report['chosen_arm'],'B')
        self.assertTrue((self.directory/'summary.json').exists())

    def test_no_cycles_if_every_task_passed_in_shared_prefix(self):
        with patch.object(c.b,'run',return_value=outcome([(i,0,True) for i in range(32)])) as run:
            self.launch()
        run.assert_called_once()
        report=json.loads((self.directory/'comparison.json').read_text())
        self.assertEqual(report['curves'],{'A':[32]*6,'B':[32]*6})

    def test_no_third_cycle_when_second_cycle_finishes_all_tasks(self):
        with patch.object(c.b,'run',side_effect=[outcome([(i,0,False) for i in range(32)]),
                outcome([(i,0,True) for i in range(32)])]) as run:
            self.launch()
        self.assertEqual(run.call_count,2)

    def test_technical_failure_stops_before_next_cycle(self):
        with patch.object(c.b,'run',return_value=outcome([(0,0,True)],'error')) as run:
            with self.assertRaisesRegex(RuntimeError,'Technical stop'):
                self.launch()
        run.assert_called_once()
        report=json.loads((self.directory/'comparison.json').read_text())
        self.assertEqual(report['status'],'incomplete/lower_bounds')
        self.assertIsNone(report['chosen_arm'])

    def test_interruption_reads_persisted_evidence_and_saves_partial_report(self):
        def interrupt(tasks,directory,**kwargs):
            directory.mkdir()
            c.g.save(directory/'outcome.json',outcome([(0,0,True)],'cancelled'))
            raise KeyboardInterrupt
        with patch.object(c.b,'run',side_effect=interrupt) as run:
            with self.assertRaises(KeyboardInterrupt):
                self.launch()
        run.assert_called_once()
        self.assertEqual(json.loads((self.directory/'comparison.json').read_text())['curves']['A'][0],1)
        self.assertTrue((self.directory/'summary.json').exists())

    def test_changed_sizing_or_existing_batch_prevents_dispatch(self):
        with patch.object(c.b,'run') as run:
            path=self.directory/'sizing.json'
            record=json.loads(path.read_text());record['source_sha256']={'changed':'hash'};c.g.save(path,record)
            with self.assertRaisesRegex(ValueError,'unchanged reviewed'):
                self.launch()
            record['source_sha256']={};c.g.save(path,record)
            (self.root/'0001-batch10-a').mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                self.launch()
        run.assert_not_called()

    def test_decision_requires_three_extra_passes_for_a(self):
        for extra,chosen in [(2,'B'),(3,'A')]:
            with self.subTest(extra=extra):
                report=c.compare(self.tasks,[dict(directory=self.root,ids=list(range(32)),
                    outcome=outcome([(i,2,True) for i in range(extra)]))],True)
                self.assertEqual(report['chosen_arm'],chosen)
                self.assertEqual(report['only_A'],list(range(extra)))

    def test_round_metrics_preserve_native_costs_and_missing_evidence(self):
        native=self.directory/'native.json';spec=self.directory/'round.json'
        c.g.save(native,dict(requests=[dict(prompt_tokens=99)],peak_memory_bytes=1234))
        c.g.save(spec,dict(requests=[dict(max_tokens=2048)]))
        data=outcome([(0,2,False)]);data['attempts'][0]['extraction']='length'
        data['rounds']=[dict(round=2,attempt_ids=[0],generation_outcome=str(native),round_file=str(spec),
            observed_tokens=2048,usage_complete=True,timing=dict(generation=3,verification=1))]
        result=c.metrics(self.root,data,2)[0]
        self.assertEqual((result['prompt_tokens'],result['committed_tokens'],result['peak_memory_bytes']),(99,2048,1234))
        self.assertEqual(result['unusable'],{'length':1})
        native.unlink()
        result=c.metrics(self.root,data,2)[0]
        self.assertIsNone(result['prompt_tokens'])
        self.assertIsNone(result['peak_memory_bytes'])
