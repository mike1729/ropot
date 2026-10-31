import unittest
from scripts import qwen_native_tools as n
from scripts import dev_construction_queue as q


class QueueTests(unittest.TestCase):
    def test_gemma_edit(self):
        prefix,args=n.parse_call('brief\n<|tool_call>call:test_edit{new:<|"|>return x // 2<|"|>,old:<|"|>return x / 2<|"|>,target_case:3}<tool_call|>')
        self.assertEqual(args,dict(old='return x / 2',new='return x // 2',target_case=3))
        self.assertEqual(prefix,'brief\n')

    def test_gemma_retarget(self):
        self.assertEqual(n.parse_call('<|tool_call>call:retarget{candidate_id:<|"|>C2<|"|>,target_case:4}<tool_call|>')[1],dict(candidate_id='C2',target_case=4))

    def test_reject_malformed_gemma(self):
        for args in ['target_case:1,target_case:2,candidate_id:<|"|>C1<|"|>',
                     'candidate_id:<|"|>C1<|"|>,target_case:<|"|>2<|"|>',
                     'candidate_id:<|"|>C1<|"|>,target_case:0']:
            with self.subTest(args=args),self.assertRaises(ValueError):
                n.parse_call('<|tool_call>call:retarget{'+args+'}<tool_call|>')

    def test_gemma_example_has_native_syntax(self):
        messages=n.gemma_messages([dict(role='user',content=n.TOY)])
        self.assertNotIn('<function=',messages[0]['content'])
        self.assertIn('<|tool_call>call:test_edit',messages[0]['content'])

    def test_tool_stop_delimiter_is_preserved(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        decoder=SimpleNamespace(add_token=Mock(),finalize=Mock(),text='decoded')
        tokenizer=SimpleNamespace(detokenizer=decoder,convert_ids_to_tokens=lambda i:'<tool_call|>' if i==50 else '<eos>')
        q.e.g.decode_tokens(tokenizer,[7,50],{50})
        self.assertEqual([x.args[0] for x in decoder.add_token.call_args_list],[7,50])
        decoder.add_token.reset_mock()
        q.e.g.decode_tokens(tokenizer,[7,1],{1})
        self.assertEqual([x.args[0] for x in decoder.add_token.call_args_list],[7])
