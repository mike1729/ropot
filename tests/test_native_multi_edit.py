"""Multi-edit extraction, automatic witnesses and variable-cost repairs; no model jobs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import native_multi_edit as m, qwen_experiment as e
from tests import test_v2_attacks as fixtures
from tests import test_qwen_experiment as experiment_fixtures

ORIGINAL, PROGRAM = fixtures.ORIGINAL, fixtures.PROGRAM


def call(edits):
    return '<tool_call><function=test_edits><parameter=edits>' + json.dumps(edits) + '</parameter></function></tool_call>'


class EditTests(unittest.TestCase):
    def test_atomic_edits_can_be_jointly_valid_when_intermediates_are_not(self):
        original = 'if flag:\n    x=1\nprint(x)\n'
        edits = [dict(old='if flag:\n', new=''), dict(old='    x=1', new='x=2')]
        self.assertEqual(m.apply_edits(original, edits), b'x=2\nprint(x)\n')
        self.assertEqual(m.apply_edits(original, list(reversed(edits))), b'x=2\nprint(x)\n')

    def test_invalid_entries_matches_and_overlaps_are_rejected(self):
        cases = [([], 'nonempty'), ([dict(old='', new='x')], 'Edit 1'),
            ([dict(old='x=1', new=2)], 'Edit 1'), ([dict(old='x=1', new='', extra=1)], 'Edit 1'),
            ([dict(old='missing', new='')], 'match_count=0'),
            ([dict(old='x', new='')], 'match_count=2'),
            ([dict(old='x=1', new=''), dict(old='1', new='2')], 'Edits 1 and 2 overlap'),
            ([dict(old='x=1', new='y=1'), dict(old='y=1', new='y=2')], 'Edit 2')]
        for edits, reason in cases:
            with self.subTest(edits=edits), self.assertRaisesRegex(ValueError, reason):
                m.apply_edits('x=1\nprint(x)\n', edits)

    def test_compile_error_is_candidate_level_and_adjacent_edits_are_allowed(self):
        with self.assertRaisesRegex(ValueError, 'Candidate syntax error at line 1, column'):
            m.apply_edits('x=1\n', [dict(old='1', new='(')])
        self.assertEqual(m.apply_edits('x=1\n', [dict(old='x=', new='y='), dict(old='1', new='2')]), b'y=2\n')

    def test_qwen_json_escaping_and_malformed_calls(self):
        edits = [dict(old='    text = "a\\nb"\n', new='    text = \'x\'\n'), dict(old='print(x)', new='')]
        self.assertEqual(m.parse_call('Brief hypothesis.\n' + call(edits)), ('Brief hypothesis.\n', edits))
        for text in [None, '{}', call([]), call(edits) + 'suffix', call(edits) * 2,
                     call(edits).replace('test_edits', 'test_edit'),
                     call(edits).replace('"old":', '"old":"duplicate","old":', 1)]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.parse_call(text)

    def test_gemma_malformed_entries_are_indexed_and_suffixes_are_rejected(self):
        good = '<|tool_call>call:test_edits{edits:[{old:<|"|>x=1<|"|>,new:<|"|>x=2<|"|>}]}<tool_call|>'
        self.assertEqual(m.parse_call(good)[1], [dict(old='x=1', new='x=2')])
        with self.assertRaisesRegex(ValueError, 'Edit 1'):
            m.parse_call(good.replace('new:<|"|>x=2<|"|>', 'new:2'))
        for text in (good + 'suffix', good * 2, good.replace('test_edits', 'retarget'),
                     good.replace('x=2<|"|>}', 'x=2<|"|>,old:<|"|>x=1<|"|>}')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.parse_call(text)

    def test_actual_pinned_templates_round_trip_nested_edits(self):
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            self.skipTest('Run pinned-template rendering with the MLX Python that provides Jinja2')
        templates = [Path('/Users/michal/Models/mlx/Qwen3.8-27B-8bit/chat_template.jinja'),
                     Path('/Users/michal/Workspace/models/mlx-community/gemma-4-31b-it-8bit/chat_template.jinja')]
        if not all(p.is_file() for p in templates):
            self.skipTest('Pinned local model templates are unavailable')
        edits = [dict(old='    x = "a\\b"\n    print(x)\n', new="    x = 'c'\n"),
                 dict(old='print("done")', new='')]
        messages = [dict(role='user', content='test'), dict(role='assistant', content='', tool_calls=[
            dict(id='call-1', type='function', function=dict(name='test_edits', arguments=dict(edits=edits)))])]
        env = ImmutableSandboxedEnvironment()
        def fail(message):
            raise ValueError(message)
        env.globals['raise_exception'] = fail
        for path in templates:
            with self.subTest(template=path):
                rendered = env.from_string(path.read_text()).render(messages=messages, tools=m.TOOLS,
                    bos_token='', add_generation_prompt=False, enable_thinking=False, preserve_thinking=True)
                start, end = ('<|tool_call>', '<tool_call|>') if 'gemma' in str(path) else ('<tool_call>', '</tool_call>')
                function = 'call:test_edits' if 'gemma' in str(path) else '<function=test_edits>'
                text = rendered[rendered.rfind(start, 0, rendered.rindex(function)):rendered.rindex(end) + len(end)]
                self.assertEqual(m.parse_call(text)[1], edits)


class AutomaticWitnessTests(unittest.TestCase):
    setUp = fixtures.ShownCheckerTests.setUp
    native = fixtures.ShownCheckerTests.native

    def check(self):
        return e.a.check_shown([(self.task, self.digest, PROGRAM.split(b'\n', 1)[1])],
            self.root / f'shown-{self.execute.call_count}', pairs={self.digest: self.pair}, auto_witness=True)[0]

    def test_declaration_free_witness_is_first_wrong_integer_in_menu_order(self):
        self.outputs[11] = '22\n'
        row = self.check()
        self.assertEqual(row['status'], 'passed')
        self.assertEqual(row['certificate']['case_id'], 'menu:1')
        self.assertNotIn('declaration', row['violations'])
        feedback = json.loads(row['feedback']['message'])
        self.assertEqual(feedback['summary']['witness_case'], 'menu:1')
        self.assertNotIn('named_case', feedback['summary'])
        self.assertEqual(feedback['repair_decision'], 'accepted')
        rules = e.a.mechanical(ORIGINAL, PROGRAM.split(b'\n', 1)[1], 4, auto_witness=True)
        cached = e.a.shown_verdict(self.digest, PROGRAM.split(b'\n', 1)[1], self.pair, rules,
                                  row['cases'], row['evidence'], auto_witness=True)
        self.assertEqual(row, cached)

    def test_nontrigger_execution_failure_does_not_add_a_new_gate(self):
        self.outputs[11] = dict(status='timeout', stdout=None, stderr=None, returncode=None)
        row = self.check()
        self.assertEqual(row['status'], 'passed')
        feedback = json.loads(row['feedback']['message'])
        self.assertIn('execution_failure', feedback['diagnoses'])
        self.assertEqual(feedback['repair_decision'], 'accepted')

    def test_automatic_witness_keeps_every_breadth_boundary_and_skips_invalid_output(self):
        for size in range(4, 9):
            self.pair['menu'] = [dict(index=10+i, input=f'{i}\n', answer=str(i), visible=False) for i in range(size)]
            for failures in (size // 2, size // 2 + 1):
                with self.subTest(size=size, failures=failures):
                    self.outputs = {10+i: '999' for i in range(failures)}
                    self.outputs[10] = '+1'
                    row = self.check()
                    self.assertEqual(row['certificate']['case_id'], 'menu:2')
                    self.assertEqual(row['status'], 'passed' if failures == size // 2 else 'failed')

    def test_overlapping_diagnoses_prioritize_execution_then_examples_then_breadth(self):
        self.outputs.update({0: '999', 11: '+1', 12: '999'})
        row = self.check()
        feedback = json.loads(row['feedback']['message'])
        self.assertEqual(feedback['repair_decision'], 'repair_execution')
        self.assertTrue({'execution_failure', 'examples_failed', 'too_broad'} <= set(feedback['diagnoses']))
        self.assertIsNone(row['certificate'])
        self.outputs[11] = '999'
        self.assertEqual(json.loads(self.check()['feedback']['message'])['repair_decision'], 'preserve_examples')
        del self.outputs[0]
        self.assertEqual(json.loads(self.check()['feedback']['message'])['repair_decision'], 'narrow_mechanism')
        self.outputs.clear()
        row = self.check()
        self.assertEqual(row['violations'], ['no_wrong_answer'])
        self.assertEqual(json.loads(row['feedback']['message'])['repair_decision'], 'different_mechanism')


class ReasoningTests(unittest.TestCase):
    def extractor(self):
        entries = [dict(benigns={'actor': dict(program='x=1\nprint(x)\n')}, partition=dict(menu=[{}, {}]))]
        initial = [dict(role='system', content='s'), dict(role='user', content='u')]
        return m.Extractor(entries, [initial], e.loop), dict(messages=initial, enable_thinking=False,
            temperature=.7, top_p=.8, top_k=20, tools=m.TOOLS), dict(repair_max_tokens=1024,
                thinking_repair_max_tokens=2048, thinking_repair_effort='low')

    def test_direct_thinking_direct_and_consumption_after_invalid_thinking_reply(self):
        for gemma in (False, True):
            extractor, initial, recipe = self.extractor()
            if gemma:
                recipe.pop('thinking_repair_effort')
            parent = dict(position=0, conversation=initial['messages'], extraction='failed')
            direct = extractor.repair(initial, parent, recipe, 1)
            self.assertFalse(direct['enable_thinking'])
            self.assertNotIn('reasoning_effort', direct)
            self.assertEqual(extractor.thinking_repairs, set())
            parent['verification'] = dict(feedback=dict(message='{"repair_decision":"different_mechanism"}'))
            reasoning = extractor.repair(initial, parent, recipe, 2)
            self.assertTrue(reasoning['enable_thinking'])
            self.assertEqual((reasoning['max_tokens'], reasoning['temperature'], reasoning['top_p'], reasoning['top_k']),
                             (2048, 1.0, .95, 20))
            self.assertEqual(reasoning.get('reasoning_effort'), None if gemma else 'low')
            for failure in ('malformed', 'syntax', 'truncated'):
                rejected = dict(position=0, conversation=parent['conversation'], extraction=failure)
                request = extractor.repair(initial, rejected, recipe, 3)
                self.assertEqual((request['max_tokens'], request['temperature'], request['top_p']), (1024, .7, .8))
                self.assertFalse(request['enable_thinking'])
                self.assertNotIn('reasoning_effort', request)

    def test_variable_cost_stable_prefix_admission_has_no_denied_builder_side_effects(self):
        for budget, expected in ((2047, []), (2048, [2048]), (3071, [2048]), (3072, [2048, 1024])):
            with self.subTest(budget=budget):
                parents = [dict(attempt=i, position=i, round=0, state='failed', cost=cost) for i, cost in enumerate((2048, 1024))]
                progress = dict(committed_replies=0, committed_tokens=0, attempts=list(parents))
                built = []
                def builder(initial, parent, recipe, seed):
                    built.append(parent['position'])
                    parent['feedback'] = 'mutated only after admission'
                    return dict(max_tokens=parent['cost'])
                requests, _ = e.loop.prepare_repairs(parents, [{}, {}], progress,
                    dict(max_repairs=4, repair_max_tokens=1024, base_seed=100), 10, budget, lambda: None,
                    builder, lambda parent, recipe: parent['cost'])
                self.assertEqual([r['max_tokens'] for r in requests], expected)
                self.assertEqual(built, list(range(len(expected))))
                for parent in parents[len(expected):]:
                    self.assertNotIn('feedback', parent)
                self.assertEqual(progress['committed_tokens'], 0)

    def test_cost_preview_does_not_consume_thinking_opportunity(self):
        extractor, initial, recipe = self.extractor()
        parent = dict(position=0, conversation=initial['messages'], verification={})
        before = deepcopy(parent)
        self.assertEqual(extractor.repair_cost(parent, recipe), 2048)
        self.assertEqual(extractor.thinking_repairs, set())
        self.assertEqual(parent, before)

    def test_denied_real_repair_does_not_mutate_history_feedback_or_reasoning_state(self):
        extractor, initial, recipe = self.extractor()
        recipe.update(max_repairs=4, base_seed=100)
        parent = dict(attempt=0, round=0, state='failed', position=0, conversation=initial['messages'], verification={})
        histories = deepcopy(extractor.histories)
        progress = dict(committed_replies=0, committed_tokens=0, attempts=[parent])
        requests, _ = e.loop.prepare_repairs([parent], [initial], progress, recipe, 5, 2047, lambda: None,
                                              extractor.repair, extractor.repair_cost)
        self.assertEqual(requests, [])
        self.assertEqual(parent['state'], 'budget_exhausted')
        self.assertNotIn('repair_feedback', parent)
        self.assertEqual(extractor.histories, histories)
        self.assertEqual(extractor.thinking_repairs, set())


class ExperimentIntegrationTests(unittest.TestCase):
    def setUp(self):
        experiment_fixtures.ExperimentTests.setUp(self)
        self.checked = 0
        for entry in self.entries:
            entry['benigns']['actor']['program'] = 'n=int(input())\nprint(n)\n'

    def generate(self, command, current, persist):
        spec = e.loop.read(command[-1])
        target = Path(spec['output_dir'])
        target.mkdir(parents=True)
        rows = []
        for request in spec['requests']:
            self.requests.append(deepcopy(request))
            if ((getattr(self, 'truncate_thinking', False) and current['round'] > 0 and request['enable_thinking'])
                    or (getattr(self, 'truncate_initial', False) and current['round'] == 0)):
                rows.append(dict(status='length', submitted=True, text='an unfinished hypothesis', answer_text=None,
                    reasoning_content='an unfinished hypothesis', tokens=[1], generation_tokens=request['max_tokens'],
                    prompt_tokens=10, enable_thinking=True))
                continue
            replacement = ('print(n + 0)' if current['round'] < 2 or getattr(self, 'never_succeed', False)
                           else 'print(n + 1 if n >= 3 else n)')
            answer = call([dict(old='print(n)', new=replacement)])
            if getattr(self, 'bad_reasoning', False) and current['round'] > 0 and request['enable_thinking']:
                answer = call([dict(old='print(n)', new='print(')])
            rows.append(dict(status='stop', submitted=True, text=answer, answer_text=answer,
                reasoning_content='one hypothesis' if request['enable_thinking'] else None,
                tokens=[1, 2], generation_tokens=2, prompt_tokens=10, enable_thinking=request['enable_thinking']))
        e.g.save(target / 'runtime.json', dict(load_seconds=.01))
        e.g.save(target / 'outcome.json', dict(status='success', requests=rows,
            native=dict(prompt_tokens=10 * len(rows), generation_tokens=sum(r['generation_tokens'] for r in rows), wall_time=.1)))
        return 0

    def check(self, candidates, log_dir, **kwargs):
        self.assertTrue(kwargs['auto_witness'])
        results = []
        for task, digest, program in candidates:
            self.checked += 1
            self.assertFalse(program.startswith(b'# case:'))
            pair = kwargs['pairs'][digest]
            rules = e.a.mechanical(pair['program'].encode(), program, len(pair['menu']), auto_witness=True)
            changed = b'n >= 3' in program
            rows = [dict(id='examples:1', index=0, status='passed', observed='1', details='measured')]
            rows += [dict(id=f'menu:{i}', index=i, status='wrong_answer' if changed and i >= 3 else 'passed',
                          observed=str(i + 1 if changed and i >= 3 else i), details='measured') for i in range(1, 5)]
            evidence = [dict(key={}, log='fixture.eval', sample_id=i) for i in range(5)]
            results.append(e.a.shown_verdict(digest, program, pair, rules, rows, evidence, auto_witness=True))
        return results

    def test_complete_driver_sequence_cache_provenance_and_allowances(self):
        report = e.run(self.output, dict(submission_format='native_multi_edit'))
        self.assertEqual(report['status'], 'completed', report)
        record = e.loop.read(self.output / 'experiment.json')
        self.assertEqual(record['reply_limit'], 80)
        self.assertEqual(record['output_token_limit'], 8 * 12288)
        self.assertIn(str(m.PROMPT), record['sources'])
        self.assertEqual(record['recipe']['initial'], m.PROMPT.read_text())
        self.assertEqual(report['yield']['accepted_tasks'], 8)
        self.assertEqual(self.checked, 32)
        self.assertEqual(report['edit_funnel']['reused'], 16)
        self.assertEqual(report['multi_edit']['actor']['reasoning']['requests'], 32)
        self.assertEqual(report['multi_edit']['actor']['conditional_growth_with_new_literal']['automatically_accepted'], 8)
        for chain in range(2):
            turns = self.requests[chain * 24:(chain + 1) * 24]
            self.assertEqual([r['enable_thinking'] for r in turns[::8]], [True, True, False])
            self.assertEqual([r['max_tokens'] for r in turns[::8]], [1024, 2048, 1024])
            self.assertEqual(turns[8]['reasoning_effort'], 'low')
            self.assertEqual(turns[0]['reasoning_effort'], 'low')
            self.assertNotIn('reasoning_effort', turns[16])
            outcome = e.loop.read(self.output / f'actor-chain-{chain}/outcome.json')
            passed = [a for a in outcome['attempts'] if a['state'] == 'passed']
            self.assertTrue(all(a['verification']['certificate']['case_id'] == 'menu:3' for a in passed))

    def test_gemma_and_invalid_reasoning_repair_return_to_direct_without_an_extra_opportunity(self):
        self.runtime['model_family'] = 'gemma4'
        self.bad_reasoning = True
        report = e.run(self.output, dict(submission_format='native_multi_edit', chains=1))
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual([r['enable_thinking'] for r in self.requests[::8]], [True, True, False])
        self.assertTrue(all('reasoning_effort' not in r for r in self.requests))
        self.assertEqual(report['multi_edit']['synthetic']['reasoning']['requests'], 16)
        self.assertEqual(report['yield']['accepted_tasks'], 8)

    def test_truncated_thinking_repair_recovers_directly_within_reserved_turns(self):
        self.truncate_thinking = True
        report = e.run(self.output, dict(submission_format='native_multi_edit', chains=1))
        self.assertEqual(report['status'], 'completed', report)
        reasoning = report['multi_edit']['actor']['reasoning']
        self.assertEqual(reasoning, dict(requests=16, closed=8, valid_calls=8, truncated=8))
        self.assertEqual([r['enable_thinking'] for r in self.requests[::8]], [True, True, False])
        self.assertTrue(any(message.get('reasoning_content') == 'an unfinished hypothesis'
                            for message in self.requests[16]['messages']))
        self.assertEqual(report['yield']['accepted_tasks'], 8)

    def test_source_diagnostics_deduplicate_programs_and_keep_semantic_results_unknown(self):
        report = e.run(self.output, dict(submission_format='native_multi_edit', chains=1))
        metrics = report['multi_edit']['actor']
        self.assertEqual(metrics['checked_distinct_programs'], 16)
        self.assertIsNone(metrics['semantic_rejections'])
        outcome = e.loop.read(self.output / 'actor-chain-0/outcome.json')
        attempts = outcome['attempts']
        reviewed = attempts[-1]
        metrics = e.multi_metrics(attempts, {(reviewed['position'], reviewed['candidate']['program_sha256']):
                                            dict(decision='reject', category='fingerprint')})
        self.assertEqual(metrics['semantic_reviewed'], 1)
        self.assertEqual(metrics['semantic_rejections'], dict(fingerprint=1))
        with self.assertRaisesRegex(ValueError, 'does not bind'):
            e.multi_metrics(attempts, {(999, 'unknown'): dict(decision='approve', category='localized_fault')})

    def test_exhaustion_never_grants_a_sixth_turn_or_another_thinking_repair(self):
        self.never_succeed = True
        report = e.run(self.output, dict(submission_format='native_multi_edit', chains=1))
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual(len(self.requests), 40)
        self.assertEqual([r['enable_thinking'] for r in self.requests[::8]], [True, True, False, False, False])
        outcome = e.loop.read(self.output / 'actor-chain-0/outcome.json')
        self.assertEqual(outcome['committed_tokens'], 8 * 6144)
        self.assertTrue(all(t['state'] == 'repairs_exhausted' for t in outcome['tasks']))

    def test_initial_thinking_cutoff_does_not_consume_the_measured_repair_opportunity(self):
        self.truncate_initial = True
        self.never_succeed = True
        report = e.run(self.output, dict(submission_format='native_multi_edit', chains=1))
        self.assertEqual(report['status'], 'completed', report)
        self.assertEqual([r['enable_thinking'] for r in self.requests[::8]], [True, False, True, False, False])
        self.assertEqual([r['max_tokens'] for r in self.requests[::8]], [1024, 1024, 2048, 1024, 1024])
        metrics = report['multi_edit']['actor']
        self.assertEqual(metrics['initial_reasoning'], dict(requests=8, closed=0, valid_calls=0, truncated=8))
        self.assertEqual(metrics['reasoning_repairs'], dict(requests=8, closed=8, valid_calls=8, truncated=0))


if __name__ == '__main__':
    unittest.main()
