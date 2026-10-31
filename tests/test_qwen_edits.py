"""Exact edit application and diagnostic state, without model or execution calls."""
import json
import unittest

from scripts.qwen_edits import Extractor, repair_guidance, duplicate_submissions


class EditTests(unittest.TestCase):
    def test_duplicate_submission_requires_program_and_target_within_same_chain(self):
        def candidate(target):
            return dict(position=0, candidate=dict(program_sha256='same-normalized-program'),
                        metadata=dict(case_declaration=str(target)))
        self.assertEqual(duplicate_submissions([
            [candidate(1), candidate(2), candidate(2)], [candidate(2)]]), 1)

    def extractor(self, original='x = 1\nprint(x)\n'):
        return Extractor([dict(benigns={'actor': dict(program=original)}, partition=dict(menu=[{}, {}]))])

    def row(self, **changes):
        return dict(text=json.dumps(dict(target_case=1, old='x = 1', new='x = 2', **changes)))

    def test_multiline_edit_uses_original_and_keeps_declaration(self):
        f = self.extractor()
        attempt = dict(position=0)
        program = f(dict(text=json.dumps(dict(target_case=2, old='x = 1\nprint(x)',
                                            new='x = 3\nprint(x + 1)'))), attempt)
        self.assertEqual(program, b'# case: 2\nx = 3\nprint(x + 1)\n')
        self.assertEqual(attempt['edit_match_count'], 1)
        self.assertTrue(attempt['edit_applicable'])

    def test_match_count_is_reported_for_missing_or_ambiguous_edit(self):
        for original, count in [('print(1)\n', 0), ('x = 1\nx = 1\n', 2)]:
            attempt = dict(position=0)
            with self.assertRaisesRegex(ValueError, f'match_count={count}'):
                self.extractor(original)(self.row(), attempt)
            self.assertIn(f'match_count={count}', attempt['submission_feedback'])
            self.assertEqual(json.loads(attempt['repair_submission'])['old'], 'x = 1')

    def test_overlapping_matches_are_ambiguous(self):
        attempt = dict(position=0)
        with self.assertRaisesRegex(ValueError, 'match_count=2'):
            self.extractor('aaa\n')(dict(text=json.dumps(dict(target_case=1, old='aa', new='b'))), attempt)

    def test_target_only_edit_is_extractable_and_marked_as_repeated_code(self):
        f = self.extractor()
        f(self.row(), dict(position=0))
        attempt = dict(position=0)
        program = f(dict(text=json.dumps(dict(target_case=2, old='x = 1', new='x = 2'))), attempt)
        self.assertEqual(program, b'# case: 2\nx = 2\nprint(x)\n')
        self.assertTrue(attempt['edit_duplicate'])

    def test_retarget_guidance_distinguishes_qualifying_code_from_no_wrong_outputs(self):
        for violations, status, expected in [(['named_case'], 'wrong_answer', True),
                (['named_case', 'examples'], 'wrong_answer', False),
                (['named_case', 'token_churn'], 'wrong_answer', False),
                (['conditionals'], 'wrong_answer', False),
                (['menu_failure_cap', 'named_case'], 'wrong_answer', False),
                (['named_case'], 'timeout', False), (['named_case'], 'passed', False)]:
            verdict = dict(violations=violations, cases=[dict(id='menu:2', status=status)],
                           feedback=dict(message='{}'))
            feedback = json.loads(repair_guidance(verdict)['feedback']['message'])
            self.assertEqual(bool(feedback.get('eligible_target_cases')), expected)
            if expected:
                self.assertEqual(feedback['eligible_target_cases'], [2])
            else:
                self.assertEqual(feedback['repair_decision'], 'different_edit_required')
                self.assertIn('Retargeting cannot help', feedback['guidance'])
                if any(v != 'named_case' for v in violations):
                    self.assertIn('Other violations remain', feedback['guidance'])
                    for violation in violations:
                        if violation != 'named_case':
                            self.assertIn(violation, feedback['guidance'])

    def test_menu_limit_only_failure_recommends_narrowing_at_the_same_site(self):
        cases = [dict(id='example:1', status='passed'), dict(id='menu:1', status='wrong_answer'),
                 dict(id='menu:2', status='wrong_answer')]
        for violations, narrow in [(['menu_failures'], True), (['named_case', 'menu_failures'], True),
                                   (['examples', 'menu_failures'], False), (['conditionals', 'menu_failures'], False)]:
            verdict = dict(violations=violations, cases=cases, feedback=dict(message='{}'))
            self.assertNotIn('narrowing', json.loads(repair_guidance(dict(verdict))['feedback']['message']))
            feedback = json.loads(repair_guidance(verdict, narrowing=True)['feedback']['message'])
            self.assertEqual(feedback['repair_decision'] == 'narrow_edit', narrow, violations)
            self.assertEqual('narrowing' in feedback, narrow)
            if narrow:
                self.assertIn('Do not add special cases', feedback['narrowing'])

    def test_ineffective_edit_guidance_applies_even_with_mechanical_violations(self):
        verdict = dict(violations=['conditionals', 'named_case'], feedback=dict(message='{}'),
                       cases=[dict(id='example:1', status='passed'), dict(id='menu:1', status='passed')])
        feedback = json.loads(repair_guidance(verdict)['feedback']['message'])
        self.assertEqual(feedback['eligible_target_cases'], [])
        self.assertIn('Every shown case passed', feedback['guidance'])
        self.assertIn('Retargeting cannot help', feedback['guidance'])

    def test_invalid_json_schema_and_syntax_keep_repair_diagnostics(self):
        for text in ['not json', '{}', '{"target_case":1,"old":"x","old":"y","new":"z"}',
                     json.dumps(dict(target_case=True, old='x', new='y')),
                     json.dumps(dict(target_case=3, old='x', new='y')),
                     json.dumps(dict(target_case=1, old='', new='y')),
                     json.dumps(dict(target_case=1, old='x = 1', new='x = '))]:
            with self.subTest(text=text):
                attempt = dict(position=0)
                with self.assertRaises((ValueError, SyntaxError)):
                    self.extractor()(dict(text=text), attempt)
                self.assertEqual(attempt['repair_submission'], text)
                self.assertTrue(attempt['submission_feedback'])

    def test_deletion_and_independent_chain_state(self):
        row = dict(text=json.dumps(dict(target_case=1, old='x = 1\n', new='')))
        for _ in range(2):
            self.assertEqual(self.extractor()(row, dict(position=0)), b'# case: 1\nprint(x)\n')
