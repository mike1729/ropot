"""Revised-recipe schedule, rescue lists, prior coverage and stopped-run reports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import batch12_rerun as c
from tests.test_batch10_scaffold import outcome


def task(i):
    return dict(id=i, split='test', sha256='a'*64), dict(question=f'Task {i}')


class RerunTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)/'evidence'
        self.directory.mkdir()
        self.root = Path(tmp.name)/'experiments'
        self.root.mkdir()
        self.tasks = {'rescue-actor': [task(1), task(60)], 'rescue-synthetic': [task(2), task(3), task(61)]}
        self.blocks = dict(development=[1, 2, 3, 4], construction=[60, 61, 62])
        self.before = dict(actor={2, 3, 4, 61, 62}, synthetic={1, 4, 60, 62})
        manifest = Path(tmp.name)/'manifest.json'
        manifest.write_text('{}')
        options = {s: dict(o, model_manifest=manifest) for s, o in c.OPTIONS.items()}
        models = {s: dict(recipe=r, within_development_maximum=True, fits_context=True,
                          model_manifest_sha256=c.g.file_hash(manifest)) for s, r in c.RECIPES.items()}
        self.sizing = dict(source_sha256={}, seeds=c.SEEDS, models=models,
                           bindings={n: [b for b, _ in ts] for n, ts in self.tasks.items()})
        c.g.save(self.directory/'sizing.json', self.sizing)
        for p in (patch.object(c, 'source_hashes', return_value={}), patch.object(c, 'OPTIONS', options),
                  patch.object(c, 'worklists', return_value=self.tasks),
                  patch.object(c, 'prior_passes', return_value=self.before),
                  patch.object(c, 'cohort_blocks', return_value=(None, self.blocks)),
                  patch.object(c.sys, 'executable', str(c.ROOT/'.venv/bin/python'))):
            p.start()
            self.addCleanup(p.stop)

    def run_caller(self, outcomes):
        with patch.object(c.b, 'run', side_effect=outcomes) as run:
            c.run(self.directory, self.root)
        return run

    def results(self):
        return json.loads((self.directory/'results.json').read_text())

    def test_worklists_use_saved_blocks_and_skip_each_sources_passes(self):
        patch.stopall()
        with patch.object(c, 'cohort_blocks', return_value=('cohort', self.blocks)), \
             patch.object(c, 'prior_passes', return_value=self.before), \
             patch.object(c.b.study_tasks, 'load_task', side_effect=lambda cohort, i: (dict(id=i), cohort)):
            lists = c.worklists(self.root)
        self.assertEqual(list(lists), list(c.ORDER))
        ids = {n: [b['id'] for b, _ in ts] for n, ts in lists.items()}
        self.assertEqual(ids['rescue-actor'], [1, 60])
        self.assertEqual(ids['rescue-synthetic'], [2, 3, 61])

    def test_prior_passes_reads_every_numbered_experiment(self):
        patch.stopall()
        for name, source, ids, passes in (('0001-a', 'actor', [5, 6], [(1, 0, True)]),
                                          ('0002-b', 'synthetic', [5, 7], [(0, 1, True), (1, 0, False)])):
            directory = self.root/name
            directory.mkdir()
            c.g.save(directory/'inputs.json', dict(source=source, worklist=[dict(binding=dict(id=i)) for i in ids]))
            c.g.save(directory/'outcome.json', outcome(passes))
        self.assertEqual(c.prior_passes(self.root), dict(actor={6}, synthetic={5}))

    def test_saved_blocks_exclude_untouched_construction_candidates(self):
        patch.stopall()
        cohort, blocks = c.cohort_blocks()
        expected = [i for phase in c.prior.PHASES for i in cohort['construction_phases'][phase]['initial'][:32]]
        untouched = {i for phase in c.prior.PHASES for i in cohort['construction_phases'][phase]['initial'][32:48]}
        self.assertEqual(blocks['construction'], expected)
        self.assertEqual([len(v) for v in blocks.values()], [50, 64])
        self.assertTrue(set(blocks['construction']).isdisjoint(untouched))

    def test_schedule_recipes_seeds_budgets_names_and_shrinking_worklists(self):
        outcomes = [outcome([(0, 0, True)]), outcome([]), outcome([(0, 3, True)]),
                    outcome([(0, 1, True)]), outcome([(1, 0, True)]), outcome([(0, 2, True)])]
        run = self.run_caller(outcomes)
        calls = run.call_args_list
        self.assertEqual([len(call.args[0]) for call in calls], [2, 1, 1, 3, 2, 1])
        self.assertEqual([call.kwargs['source'] for call in calls], ['actor']*3+['synthetic']*3)
        self.assertEqual([call.kwargs['recipe']['base_seed'] for call in calls], list(c.SEEDS)*2)
        for call in calls:
            recipe = call.kwargs['recipe']
            self.assertEqual((recipe['temperature'], recipe['max_repairs'], recipe['max_tokens']), (.2, 3, 2048))
            self.assertEqual(call.kwargs['reply_allowance'], 4*len(call.args[0]))
            self.assertEqual(call.kwargs['output_token_allowance'], 4*len(call.args[0])*2048)
        self.assertEqual(calls[0].kwargs['recipe']['system'], c.base.RECIPE['system'])
        self.assertEqual(calls[3].kwargs['recipe']['system'], c.prior.RECIPE['system'])
        self.assertEqual([call.args[1].name for call in calls][:3],
                         ['0001-batch12-rescue-actor-c1', '0002-batch12-rescue-actor-c2', '0003-batch12-rescue-actor-c3'])
        self.assertEqual([b['id'] for b, _ in calls[1].args[0]], [60])
        report = self.results()
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['curves']['rescue-actor']['passed'], [1]*11+[2])
        self.assertEqual(report['curves']['rescue-synthetic']['passed'], [0,1,1,1,2,2,2,2,2,2,3,3])
        self.assertEqual(set(report['curves']), {'rescue-actor','rescue-synthetic'})
        self.assertNotIn('fresh_joint', report)
        self.assertIn('not clean recipe-yield estimates', report['interpretation'])
        self.assertTrue((self.directory/'summary.json').exists())

    def test_technical_stop_halts_and_saves_lower_bounds(self):
        with patch.object(c.b, 'run', side_effect=[outcome([(0, 0, True), (1, 0, True)]), outcome([], status='error')]) as run, \
                self.assertRaisesRegex(RuntimeError, 'Technical stop'):
            c.run(self.directory, self.root)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(self.results()['status'], 'incomplete/lower_bounds')

    def test_report_coverage_joint_and_cycle_initial_diversity(self):
        repeated = outcome([(0, 0, False), (0, 1, False)])
        for attempt, digest in zip(repeated['attempts'], ('x', 'y')):
            attempt['candidate'] = dict(program_sha256=digest)
        again = outcome([(0, 0, True)])
        again['attempts'][0]['candidate'] = dict(program_sha256='x')
        records = [dict(worklist='rescue-synthetic', directory=str(self.root), offset=0, ids=[2], outcome=repeated),
                   dict(worklist='rescue-synthetic', directory=str(self.root), offset=4, ids=[2], outcome=again),
                   dict(worklist='rescue-actor', directory=str(self.root), offset=0, ids=[1], outcome=outcome([(0, 0, True)]))]
        result = c.report(self.tasks, records, self.before, self.blocks, True)
        self.assertEqual(result['curves']['rescue-synthetic']['passed'][4], 1)
        self.assertEqual(result['diversity']['rescue-synthetic'][2],
                         dict(programs=3, distinct=2, fresh=2, fresh_distinct=1))
        development = result['coverage']['development']
        self.assertEqual((development['actor_before'], development['actor_after']), (3, 4))
        self.assertEqual((development['both_before'], development['both_after']), (1, 3))

    def test_preflight_refuses_changed_inputs_or_existing_batch(self):
        cases = {'recipe': lambda s: s['models']['synthetic'].update(recipe={}),
                 'bindings': lambda s: s['bindings'].update({'rescue-actor': []}),
                 'seeds': lambda s: s.update(seeds=[1])}
        for name, change in cases.items():
            with self.subTest(name=name):
                sizing = json.loads(json.dumps(self.sizing))
                change(sizing)
                c.g.save(self.directory/'sizing.json', sizing)
                with patch.object(c.b, 'run') as run, self.assertRaises(ValueError):
                    c.run(self.directory, self.root)
                run.assert_not_called()
        c.g.save(self.directory/'sizing.json', self.sizing)
        with patch.object(c.prior.reader, 'summarize', return_value=dict(experiments=[dict(experiment='0009-batch12-x')])), \
                patch.object(c.b, 'run') as run, self.assertRaises(FileExistsError):
            c.run(self.directory, self.root)
        run.assert_not_called()

    def test_damaged_native_json_keeps_partial_report_and_known_counters(self):
        native = self.directory/'damaged-native.json'
        native.write_text('{')
        failed = outcome([(0, 0, False)], status='error')
        failed['rounds'] = [dict(round=0, attempt_ids=[0], generation_outcome=str(native),
            round_file=str(self.directory/'missing-round.json'), observed_tokens=17,
            usage_complete=False, timing=dict(generation=1))]
        with patch.object(c.b, 'run', return_value=failed), self.assertRaisesRegex(RuntimeError, 'Technical stop'):
            c.run(self.directory, self.root)
        report = self.results()
        self.assertEqual(report['status'], 'incomplete/lower_bounds')
        metric = report['effort'][0]['rounds'][0]
        self.assertIn('evidence_error', metric)
        self.assertEqual(metric['observed_tokens'], 17)
        self.assertIsNone(metric['prompt_tokens'])
        self.assertEqual(report['effort'][0]['counters']['committed_tokens'], 2048)


if __name__ == '__main__':
    unittest.main()
