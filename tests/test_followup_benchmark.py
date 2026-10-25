from contextlib import redirect_stdout
import io
import itertools
import json
import unittest
from unittest.mock import patch

from experiments import followup_fixtures as f
from experiments.followup_report import code


class FollowupTests(unittest.TestCase):
    def reference(self, kind, stdin, offset=0):
        output = io.StringIO()
        with patch('sys.stdin', type('Input', (), {'buffer':io.BytesIO(stdin.encode())})()), redirect_stdout(output):
            exec(compile(f.program(kind, offset), '<invented reference>', 'exec'), {})
        return int(output.getvalue())

    def test_independent_oracles_match_references_and_bruteforce_bipartition(self):
        for n in range(1, 5):
            possible = list(itertools.combinations(range(1,n+1),2))
            for bits in range(2**len(possible)):
                edges = [e for i,e in enumerate(possible) if bits & (1<<i)]
                stdin = f.graph(n,edges)
                for kind in f.KINDS:
                    self.assertEqual(f.oracle(kind,n,edges), self.reference(kind,stdin))
                brute = any(all(((colour>>(a-1))&1) != ((colour>>(b-1))&1) for a,b in edges)
                            for colour in range(2**n))
                self.assertEqual(f.oracle('bipartite',n,edges), int(brute))
        for kind in f.KINDS:
            for case in f.verification_task(kind,0)['cases']:
                self.assertEqual(self.reference(kind,case['input']),int(case['answer']))

    def test_histories_record_actual_seeded_failures_and_valid_inputs(self):
        for kind in f.KINDS:
            messages = f.history(kind)
            self.assertEqual([m['role'] for m in messages[1:]], ['user','assistant']*4+['user'])
            for i,offset in enumerate((-1,1,2,3)):
                stdin, answer = f.large_case(kind,0,i)
                observed = self.reference(kind,stdin,offset)
                self.assertNotEqual(answer,observed)
                self.assertIn(f'Observed stdout: {observed}',messages[3+2*i]['content'])
                numbers = list(map(int,stdin.split()))
                self.assertEqual(len(numbers),2+2*numbers[1])
                self.assertTrue(all(1<=x<=numbers[0] for x in numbers[2:]))

    def test_semantic_rules_reject_ambiguity_and_incomplete_answers(self):
        self.assertEqual(f.semantic_verdict(' {"answer": -1}\n','stop',-1),'passed')
        self.assertEqual(f.semantic_verdict('{"answer": 2}','stop',1),'wrong_answer')
        self.assertEqual(f.semantic_verdict('{"answer": 1}','length',1),'incomplete')
        for text in ['{"answer":true}', '{"answer":1,"extra":0}', '{"answer":0,"answer":1}',
                     '```json\n{"answer":1}\n```','[1]', '{"answer":1.0}']:
            self.assertEqual(f.semantic_verdict(text,'stop',1),'format_failure')
        for kind in f.KINDS:
            for row in f.semantic_group(kind,False):
                stdin = row['messages'][-1]['content'].split('stdin:\n')[1]
                self.assertEqual(self.reference(kind,stdin),row['expected_answer'])

    def test_prefix_excludes_last_token_and_rejects_missing_shared_state(self):
        self.assertEqual(f.common_prefix([{'prompt_ids':[1,2,3]}, {'prompt_ids':[1,2,4]}]),2)
        self.assertEqual(f.common_prefix([{'prompt_ids':[1,2,3]}, {'prompt_ids':[1,2,3]}]),2)
        with self.assertRaises(ValueError):
            f.common_prefix([{'prompt_ids':[1,2]}, {'prompt_ids':[2,3]}])

    def test_program_parser_accepts_only_complete_plain_or_single_fenced_python(self):
        self.assertEqual(code('```python\nprint(1)\n```'), b'print(1)\n')
        self.assertEqual(code('print(1)'),b'print(1)\n')
        for text in ['```python\nprint(1)\n```\nExplanation', '```js\nprint(1)\n```','', 'print(']:
            with self.assertRaises((SyntaxError,ValueError)):
                code(text)


if __name__ == '__main__':
    unittest.main()
