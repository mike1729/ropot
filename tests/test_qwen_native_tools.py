"""Native-call parsing and retained conversations; no model or Docker calls."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts import qwen_experiment as e, qwen_native_tools as n


def call(old='x = 1', new='x = 2', target=1):
    return ('<tool_call>\n<function=test_edit>\n'
            f'<parameter=old>\n{old}\n</parameter>\n'
            f'<parameter=new>\n{new}\n</parameter>\n'
            f'<parameter=target_case>\n{target}\n</parameter>\n'
            '</function>\n</tool_call>')


def retarget(candidate='C101', target=2):
    return ('<tool_call>\n<function=retarget>\n'
            f'<parameter=candidate_id>\n{candidate}\n</parameter>\n'
            f'<parameter=target_case>\n{target}\n</parameter>\n</function>\n</tool_call>')


class NativeToolTests(unittest.TestCase):
    def test_retarget_id_preserves_program_and_is_scoped_to_task_and_chain(self):
        entries = [dict(benigns={'actor': dict(program='x = 1\nprint(x)')},
                        partition=dict(menu=[{}, {}])) for _ in range(2)]
        initial = [[dict(role='system', content='s'), dict(role='user', content='u')] for _ in entries]
        extractor = n.Extractor(entries, initial, e.loop)
        def submit(text, position, seed):
            row = dict(text=text, answer_text=text, tokens=[1], status='stop')
            attempt = dict(position=position, seed=seed)
            extractor.observe(row, attempt)
            return extractor(row, attempt), attempt
        original, _ = submit(call(), 0, 101)
        corrected, attempt = submit(retarget(), 0, 102)
        self.assertEqual(original.split(b'\n', 1)[1], corrected.split(b'\n', 1)[1])
        self.assertTrue(corrected.startswith(b'# case: 2\n'))
        self.assertEqual(attempt['retarget_source_id'], 'C101')
        self.assertTrue(attempt['edit_duplicate'])
        self.assertEqual(attempt['conversation'][-1]['tool_calls'][0]['function']['name'], 'retarget')
        for text, position in [(retarget('C999'), 0), (retarget(), 1), (retarget(target=3), 0)]:
            with self.assertRaises(ValueError):
                submit(text, position, 103)
        with self.assertRaisesRegex(ValueError, r'target_case must be in 1\.\.2'):
            submit(retarget(target=3), 0, 105)
        other = n.Extractor(entries, initial, e.loop)
        row = dict(text=retarget(), answer_text=retarget(), tokens=[1], status='stop')
        attempt = dict(position=0, seed=104)
        other.observe(row, attempt)
        with self.assertRaises(ValueError):
            other(row, attempt)
        self.assertNotIn('candidate_id', attempt)

    def test_retarget_parser_rejects_patch_fields_and_wrong_ids(self):
        self.assertEqual(n.parse_call(retarget())[1], dict(candidate_id='C101', target_case=2))
        for text in [retarget(''), retarget(' C101'), retarget(target=0),
                     retarget().replace('</function>', '<parameter=old>x</parameter></function>')]:
            with self.assertRaises(ValueError):
                n.parse_call(text)

    def test_metrics_distinguish_eligible_from_futile_retargets(self):
        edit = dict(old='x', new='y', target_case=1)
        attempts = [dict(attempt=0, position=0, round=1, candidate_id='C101', edit=edit,
                        verification=dict(retarget_eligible_cases=[2]), cumulative_output_tokens=10),
                    dict(attempt=1, parent=0, position=0, round=2, candidate_id='C102',
                        tool_name='retarget', retarget_source_id='C101', edit=dict(edit, target_case=2),
                        verification=dict(status='passed', execution_reused=True)),
                    dict(attempt=2, position=1, round=1, candidate_id='C103', edit=edit,
                        verification=dict(retarget_eligible_cases=[]), cumulative_output_tokens=10),
                    dict(attempt=3, parent=2, position=1, round=2, tool_name='retarget',
                        retarget_source_id='C103', edit=dict(edit, target_case=2))]
        attempts[1]['candidate'] = dict(program_sha256='abc')
        result = n.metrics([attempts], 4)
        self.assertEqual(result['retarget_by_id_calls'], 2)
        self.assertEqual(result['eligible_retargets_without_copying_errors'], 1)
        self.assertEqual(result['eligible_retargets_accepted'], 1)
        self.assertEqual(result['futile_target_only_retries'], 1)

    def test_multiline_arguments_and_empty_replacement_preserve_whitespace(self):
        old = '    x = 1\n    print(x)\n'
        prefix, args = n.parse_call('Testing now.\n' + call(old, '', 2))
        self.assertEqual(prefix, 'Testing now.\n')
        self.assertEqual(args, dict(old=old, new='', target_case=2))

    def test_malformed_multiple_calls_duplicate_fields_and_suffix_are_rejected(self):
        good = call()
        for answer in [None, '{}', good + good, good + 'suffix',
                       good.replace('test_edit', 'execute_python'),
                       good.replace('<parameter=new>', '<parameter=old>'),
                       call(target='1.5'), good.replace('<parameter=old>', '<parameter=path>')]:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                n.parse_call(answer)

    def test_reasoning_and_tool_results_survive_continuation_and_are_task_local(self):
        initial = [[dict(role='system', content='test'), dict(role='user', content=f'task {i}')]
                   for i in range(2)]
        entries = [dict(benigns={'actor': dict(program='x = 1\nprint(x)\n')},
                        partition=dict(menu=[{}, {}])) for _ in range(2)]
        extractor = n.Extractor(entries, initial, e.loop)
        row = dict(text='raw evidence', answer_text=call(), reasoning_content='one concrete hypothesis',
                   tokens=[1, 2], generation_tokens=12, status='stop')
        parent = dict(position=0, attempt=0, round=0, seed=101, parent=None)
        extractor.observe(row, parent)
        program = extractor(row, parent)
        self.assertTrue(parent['native_tool_call'])
        self.assertEqual(parent['candidate_id'], 'C101')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.py'
            path.write_bytes(program)
            parent.update(extraction='success', program=str(path), verification=dict(feedback=dict(
                message=json.dumps(dict(summary=dict(examples_passed=1), eligible_target_cases=[2])))))
            recipe = dict(repair='test another edit', repair_max_tokens=2048,
                          feedback_field_bytes=512, feedback_bytes=4096)
            request = extractor.repair(dict(messages=initial[0], tools=n.TOOLS, preserve_thinking=True),
                                       parent, recipe, 103)
        messages = request['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'tool'])
        self.assertEqual(messages[2]['reasoning_content'], 'one concrete hypothesis')
        self.assertEqual(messages[2]['tool_calls'][0]['function']['arguments']['old'], 'x = 1')
        result = json.loads(messages[3]['content'])
        self.assertEqual(result['candidate_id'], 'C101')
        self.assertEqual(result['measured_result']['eligible_target_cases'], [2])
        self.assertEqual(result['measured_result']['guidance'], result['request'])
        for text in (result['request'], result['measured_result']['guidance'], n.TOOLS[0]['function']['description']):
            self.assertIn('retarget', text)
            self.assertIn('not copy old or new', text.lower())
            self.assertNotIn('Change only target_case', text)
        self.assertEqual(messages[3]['tool_call_id'], 'call-101')
        self.assertEqual(messages[2]['tool_calls'][0]['id'], 'call-101')
        self.assertEqual(request['tools'], n.TOOLS)
        self.assertTrue(request['preserve_thinking'])
        second = dict(position=0, attempt=2, round=1, seed=103, parent=0)
        extractor.observe(dict(row, answer_text=call(target=2)), second)
        extractor(dict(row, answer_text=call(target=2)), second)
        self.assertEqual(second['cumulative_output_tokens'], 24)
        self.assertEqual(len(second['conversation']), 5)
        self.assertEqual(len(extractor.histories[1]), 2)

    def test_unclosed_thought_never_executes_a_call_from_reasoning(self):
        initial = [[dict(role='system', content='test'), dict(role='user', content='task')]]
        extractor = n.Extractor([dict(benigns={'actor': dict(program='x=1')},
            partition=dict(menu=[{}]))], initial, e.loop)
        row = dict(text=call(), answer_text=None, tokens=[1], generation_tokens=2048, status='length')
        attempt = dict(position=0, seed=101)
        extractor.observe(row, attempt)
        with self.assertRaises(ValueError):
            extractor(row, attempt)
        self.assertNotIn('native_tool_call', attempt)
        self.assertEqual(attempt['conversation'][-1]['reasoning_content'], call())

    def test_cutoff_metrics_distinguish_natural_no_call_from_output_cap(self):
        for capped in (False, True):
            attempts = [dict(attempt=0, position=0, round=0, thinking=True, hit_output_cap=capped),
                        dict(attempt=1, parent=0, position=0, round=1, thinking=False,
                             native_tool_call=True, edit_applicable=True)]
            result = n.metrics([attempts])
            self.assertEqual(result['direct_requests_after_initial_no_call'], 1)
            self.assertEqual(result['direct_calls_after_initial_no_call'], 1)
            self.assertEqual(result['forced_transitions'], 1)
            self.assertEqual(result['direct_requests_after_thinking_cutoff'], int(capped))
            self.assertEqual(result['direct_calls_after_thinking_cutoff'], int(capped))

    def test_metrics_include_earlier_deliberation_but_exclude_cached_executions(self):
        def turn(attempt, round_, parent=None, **fields):
            return dict(dict(attempt=attempt, position=0, round=round_, parent=parent, state='failed'), **fields)
        edit = dict(old='a', new='b', target_case=1)
        groups = [[turn(0, 0, cumulative_output_tokens=1024, thinking=True, extraction='length'),
                   turn(1, 1, 0, cumulative_output_tokens=1100, thinking=False, candidate_id='C2',
                        native_tool_call=True, edit_applicable=True, edit=edit,
                        verification=dict(retarget_eligible_cases=[3])),
                   turn(2, 2, 1, cumulative_output_tokens=1150, thinking=False, candidate_id='C3',
                        native_tool_call=True, edit_applicable=True, edit=dict(edit, target_case=3),
                        verification=dict(execution_reused=True)),
                   turn(3, 3, 2, cumulative_output_tokens=1200, thinking=False, candidate_id='C4',
                        native_tool_call=True, edit_applicable=True, edit=dict(edit, new='c'),
                        verification=dict(retarget_eligible_cases=[2]), state='repairs_exhausted')]]
        metrics = n.metrics(groups, 3)
        self.assertEqual(metrics['naturally_emitted_calls'], 3)
        self.assertEqual(metrics['initial_natural_calls'], 0)
        self.assertEqual(metrics['forced_transitions'], 1)
        self.assertEqual(metrics['direct_calls_after_thinking_cutoff'], 1)
        self.assertEqual(metrics['retargets_after_feedback'], 1)
        self.assertEqual(metrics['revisions_after_feedback'], 1)
        self.assertEqual(metrics['retarget_opportunities'], 2)
        self.assertEqual(metrics['last_turn_retarget_opportunities'], 1)
        self.assertEqual(metrics['length_stops_thinking'], 1)
        self.assertEqual(metrics['repairs_exhausted'], 1)
        self.assertEqual(metrics['first_executed_candidates'], [dict(chain=0, position=0,
                         output_tokens=1100, candidate_id='C2')])

    def test_truncated_initial_phase_requests_direct_call_without_candidate_id(self):
        initial = [[dict(role='system', content='test'), dict(role='user', content='task')]]
        extractor = n.Extractor([dict(benigns={'actor': dict(program='x = 1')},
            partition=dict(menu=[{}]))], initial, e.loop)
        row = dict(text='partial hypothesis', answer_text=None, tokens=[1], generation_tokens=1024,
                   status='length', enable_thinking=True)
        parent = dict(position=0, attempt=0, round=0, seed=101, parent=None, extraction='length')
        extractor.observe(row, parent)
        recipe = dict(repair='direct', repair_max_tokens=512, feedback_field_bytes=512, feedback_bytes=4096,
                      repair_enable_thinking=False, repair_temperature=0.7, repair_top_p=0.8)
        request = extractor.repair(dict(messages=initial[0], enable_thinking=True, reasoning_effort='low',
                                        temperature=1.0, top_p=0.95, top_k=20), parent, recipe, 109)
        self.assertNotIn('candidate_id', parent)
        self.assertFalse(request['enable_thinking'])
        self.assertNotIn('reasoning_effort', request)
        self.assertEqual((request['temperature'], request['top_p'], request['top_k'], request['max_tokens']),
                         (0.7, 0.8, 20, 512))
        self.assertEqual(request['messages'][2]['reasoning_content'], 'partial hypothesis')
        response = request['messages'][3]
        self.assertEqual(response['role'], 'user')
        result = json.loads(response['content'])
        self.assertIn('No complete tool call was submitted', result['message'])
        self.assertNotIn('candidate_id', result)
        self.assertNotIn('program', response['content'].replace('ORIGINAL program', ''))

    def test_one_thinking_repair_follows_the_first_measured_result(self):
        initial = [[dict(role='system', content='s'), dict(role='user', content='u')]]
        extractor = n.Extractor([dict(benigns={'actor': dict(program='x = 1\nprint(x)\n')},
            partition=dict(menu=[{}, {}]))], initial, e.loop)
        recipe = dict(repair='r', repair_max_tokens=512, feedback_field_bytes=512, feedback_bytes=4096,
                      repair_enable_thinking=False, repair_temperature=.7, repair_top_p=.8,
                      thinking_repair_max_tokens=768)
        base = dict(messages=initial[0], enable_thinking=True, reasoning_effort='low', temperature=1.0, top_p=.95)
        requests = []
        with tempfile.TemporaryDirectory() as directory:
            for attempt_id, seed in ((0, 101), (2, 103)):
                row = dict(text='', answer_text=call(), reasoning_content='h', tokens=[1], generation_tokens=3,
                           status='stop', enable_thinking=attempt_id == 0)
                parent = dict(position=0, attempt=attempt_id, round=attempt_id // 2, seed=seed)
                extractor.observe(row, parent)
                path = Path(directory) / f'p{attempt_id}.py'
                path.write_bytes(extractor(row, parent))
                parent.update(extraction='success', program=str(path), verification=dict(
                    feedback=dict(message=json.dumps(dict(violations=['examples'], eligible_target_cases=[])))))
                requests.append(extractor.repair(base, parent, recipe, seed + 1))
        thinking, direct = requests
        self.assertEqual((thinking['enable_thinking'], thinking['reasoning_effort'], thinking['max_tokens'],
                          thinking['temperature'], thinking['top_p']), (True, 'low', 768, 1.0, .95))
        self.assertIn('think briefly', thinking['messages'][-1]['content'])
        self.assertEqual((direct['enable_thinking'], direct['max_tokens'], direct['temperature']), (False, 512, .7))
        self.assertNotIn('reasoning_effort', direct)

    def test_thinking_repair_prompt_states_the_schedule(self):
        recipe = dict(e.a.ATTACK_RECIPE, enable_thinking=True, max_repairs=4, repair_enable_thinking=False,
                      thinking_repair_max_tokens=768)
        n.configure(recipe)
        self.assertIn('turn after the first measured result allow brief thinking', recipe['repair'])

    def test_inapplicable_call_has_call_id_but_no_candidate_id(self):
        extractor = n.Extractor([dict(benigns={'actor': dict(program='x = 1')},
            partition=dict(menu=[{}]))], [[dict(role='system', content='s'), dict(role='user', content='u')]], e.loop)
        attempt = dict(position=0, seed=7)
        row = dict(text='', answer_text=call(old='missing'), tokens=[1], generation_tokens=5, status='stop')
        extractor.observe(row, attempt)
        with self.assertRaises(ValueError):
            extractor(row, attempt)
        self.assertEqual(attempt['call_id'], 'call-7')
        self.assertNotIn('candidate_id', attempt)
