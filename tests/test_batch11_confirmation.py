"""Confirmation scheduling, prefix yields, task ownership and stopped-run reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import batch11_confirmation as c
from tests.test_batch10_scaffold import outcome

WORKLISTS = c.worklists


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory=Path(tmp.name)/'evidence'
        self.directory.mkdir()
        self.root=Path(tmp.name)/'experiments'
        self.root.mkdir()
        actor=[(dict(id=i,split='test',sha256='a'*64),dict(question=f'Task {i}')) for i in range(64)]
        self.tasks=dict(actor=actor,synthetic=actor[16:32]+actor[48:64])
        self.confirm=[binding['id'] for binding,_ in self.tasks['synthetic']]
        self.manifest=Path(tmp.name)/'manifest.json'
        self.manifest.write_text('{}')
        options=[dict(o,model_manifest=self.manifest) for o in c.base.RUNS]
        self.context=dict(passed=[0,1],experiments=[])
        models=[dict(source=source,recipe=recipe,within_development_maximum=True,fits_context=True,
            model_manifest_sha256=c.g.file_hash(self.manifest)) for source,recipe in
            [('actor',c.QWEN_RECIPE),('synthetic',c.MINISTRAL_RECIPE)]]
        self.sizing=dict(source_sha256={},bindings={s:[b for b,_ in ts] for s,ts in self.tasks.items()},
                         models=models,cycle_seeds=c.SEEDS)
        c.g.save(self.directory/'sizing.json',self.sizing)
        for p in [patch.object(c,'source_hashes',return_value={}),patch.object(c,'worklists',return_value=self.tasks),
                  patch.object(c.base,'RUNS',options),patch.object(c,'exploration',return_value=self.context)]:
            p.start()
            self.addCleanup(p.stop)

    def run_caller(self):
        c.run(self.directory,self.root)

    def results(self):
        return json.loads((self.directory/'results.json').read_text())

    def test_saved_phase_order_reads_only_candidates_one_to_32(self):
        cohort=c.b.study_tasks.load_cohort()
        expected=[i for phase in c.prior.PHASES for i in cohort['construction_phases'][phase]['initial'][:32]]
        confirmation=[i for phase in c.prior.PHASES for i in cohort['construction_phases'][phase]['initial'][16:32]]
        with patch.object(c.b.study_tasks,'load_task',side_effect=lambda _,i:(cohort['tasks_by_id'][i],{})) as load:
            tasks=WORKLISTS()
        self.assertEqual([b['id'] for b,_ in tasks['actor']],expected)
        self.assertEqual([b['id'] for b,_ in tasks['synthetic']],confirmation)
        self.assertEqual([call.args[1] for call in load.call_args_list],expected)
        self.assertEqual(len(expected),64)
        self.assertEqual(len(confirmation),32)

    def test_sources_cycles_seeds_recipes_budgets_and_numbering(self):
        outcomes=[outcome([(0,0,True)]),outcome([(0,1,True)]),outcome([(0,0,True)]),
                  outcome([(0,1,True)]),outcome([(0,0,True)])]
        with patch.object(c.b,'run',side_effect=outcomes) as run:
            self.run_caller()
        calls=run.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls],[64,32,31,30,29])
        self.assertEqual([call.kwargs['source'] for call in calls],['actor']+['synthetic']*4)
        self.assertEqual([call.kwargs['recipe']['base_seed'] for call in calls],[17,17,100017,200017,300017])
        self.assertEqual([call.kwargs['recipe']['temperature'] for call in calls],[.2,.2,.7,.7,.7])
        self.assertEqual(calls[0].kwargs['recipe'],c.QWEN_RECIPE)
        self.assertTrue(all(call.kwargs['recipe']['max_repairs']==1 for call in calls))
        self.assertEqual([call.kwargs['reply_allowance'] for call in calls],[128,64,62,60,58])
        self.assertEqual([call.kwargs['output_token_allowance'] for call in calls],
                         [131072,131072,126976,122880,118784])
        self.assertEqual([call.args[1].name for call in calls],['0001-batch11a-actor']+
                         [f'000{i+1}-batch11a-synthetic-c{i}' for i in range(1,5)])
        self.assertEqual([b['id'] for b,_ in calls[2].args[0]],self.confirm[1:])
        report=self.results()
        self.assertEqual(report['status'],'complete')
        self.assertEqual(report['ministral']['curve'],[0,1,2,2,2,3,4,4])
        self.assertEqual(report['ministral']['confirmation_budget'],6)
        self.assertTrue((self.directory/'summary.json').exists())

    def test_six_attempt_joint_excludes_late_pass_but_eight_attempt_includes_it(self):
        records=[dict(source='actor',directory=str(self.root),ids=[0,16,17],offset=0,
                      outcome=outcome([(0,0,True),(1,0,True),(2,1,True)])),
                 dict(source='synthetic',directory=str(self.root),ids=[16],offset=4,outcome=outcome([(0,1,True)])),
                 dict(source='synthetic',directory=str(self.root),ids=[17],offset=6,outcome=outcome([(0,0,True)]))]
        result=c.report(self.tasks,records,self.context,True)
        self.assertEqual(result['qwen']['exploration']['curve'],[1,1])
        self.assertEqual(result['qwen']['confirmation']['curve'],[1,2])
        self.assertEqual(result['joint']['exploration']['ids'],[0])
        self.assertEqual(result['joint']['confirmation_6']['ids'],[16])
        self.assertEqual(result['joint']['confirmation_8']['ids'],[16,17])
        self.assertEqual(result['ministral']['curve'],[0,0,0,0,0,1,2,2])
        self.assertEqual(len(result['tasks']['actor']),64)
        self.assertEqual(len(result['tasks']['synthetic']),32)

    def test_early_stop_skips_later_cycles(self):
        with patch.object(c.b,'run',side_effect=[outcome([]),outcome([(i,0,True) for i in range(32)])]) as run:
            self.run_caller()
        self.assertEqual(run.call_count,2)
        self.assertEqual(self.results()['ministral']['curve'],[32]*8)

    def test_unusable_counts_include_syntax_and_use_global_attempt_number(self):
        data=outcome([(0,1,False)])
        data['attempts'][0]['extraction']='syntax'
        record=dict(source='synthetic',directory=str(self.root),ids=[16],offset=6,outcome=data)
        with patch.object(c.prior,'metrics',return_value=[dict(round=1)]):
            result=c.report(self.tasks,[record],self.context,False)
        metric=result['effort'][0]['rounds'][0]
        self.assertEqual(metric['attempt'],8)
        self.assertEqual(metric['unusable'],{'syntax':1})

    def test_technical_stop_prevents_next_source_or_cycle(self):
        for failures,expected in [([outcome([],status='error')],1),
                ([outcome([]),outcome([]),outcome([],status='error')],3)]:
            with self.subTest(expected=expected),patch.object(c.b,'run',side_effect=failures) as run:
                with self.assertRaisesRegex(RuntimeError,'Technical stop'):
                    self.run_caller()
                self.assertEqual(run.call_count,expected)
                self.assertEqual(self.results()['status'],'incomplete/lower_bounds')

    def test_interruption_keeps_saved_pass_and_denominators(self):
        def interrupted(tasks,directory,**kwargs):
            directory.mkdir()
            c.g.save(directory/'outcome.json',outcome([(0,0,True)],status='cancelled'))
            raise KeyboardInterrupt
        with patch.object(c.b,'run',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.run_caller()
        report=self.results()
        self.assertEqual(report['status'],'incomplete/lower_bounds')
        self.assertEqual(report['qwen']['exploration']['curve'],[1,1])
        self.assertEqual(report['qwen']['confirmation']['denominator'],32)
        self.assertTrue((self.directory/'summary.json').exists())

    def test_preflight_blocks_changed_recipe_sources_and_existing_run(self):
        for kind in ['sources','recipe','sizing','existing']:
            with self.subTest(kind=kind):
                record=json.loads(json.dumps(self.sizing))
                if kind=='sources': record['source_sha256']={'changed':'digest'}
                if kind=='recipe': record['models'][1]['recipe']['max_tokens']=1024
                if kind=='sizing': record['models'][0]['within_development_maximum']=False
                if kind=='existing': (self.root/'0001-batch11a-actor').mkdir()
                c.g.save(self.directory/'sizing.json',record)
                with patch.object(c.b,'run') as run:
                    with self.assertRaises((ValueError,FileExistsError)):
                        self.run_caller()
                    run.assert_not_called()

    def test_exploration_union_includes_a_only_pass_and_rejects_foreign_binding(self):
        # Use the real context loader, not its scheduling patch.
        import importlib.util
        spec=importlib.util.spec_from_file_location('confirmation_context',c.__file__)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name,passed in [('0001-batch10-a',[0,1]),('0002-batch10-b2',[2])]:
            directory=self.root/name;directory.mkdir()
            c.g.save(directory/'inputs.json',dict(source='synthetic',fixtures=False,
                cohort_sha256=c.b.study_tasks.COHORT_SHA256,
                worklist=[dict(binding=b) for b,_ in self.tasks['actor'][:3]]))
            c.g.save(directory/'outcome.json',outcome([(i,0,True) for i in passed]))
        self.assertEqual(module.exploration(self.root,self.tasks)['passed'],[0,1,2])
        path=self.root/'0002-batch10-b2/inputs.json'
        bad=json.loads(path.read_text());bad['worklist'][0]['binding']['sha256']='b'*64;c.g.save(path,bad)
        with self.assertRaisesRegex(ValueError,'binding differs'):
            module.exploration(self.root,self.tasks)
