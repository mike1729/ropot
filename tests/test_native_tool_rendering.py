"""Tool schemas, preserved reasoning, and token-ID thought extraction."""
import json
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import qwen_experiment as e
from scripts.qwen_native_tools import TOOLS

g = e.g


class NativeRenderingTests(unittest.TestCase):
    def test_schema_and_preservation_reach_both_template_calls(self):
        tokenizer = SimpleNamespace(has_thinking=True)
        tokenizer.apply_chat_template = mock.Mock(side_effect=['rendered', [1, 2]])
        messages = [dict(role='assistant', reasoning_content='hypothesis', content='')]
        g.render(tokenizer, messages, enable_thinking=True, reasoning_effort='low',
                 tools=TOOLS, preserve_thinking=True)
        self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
        for call in tokenizer.apply_chat_template.call_args_list:
            self.assertEqual(call.kwargs['tools'], TOOLS)
            self.assertTrue(call.kwargs['preserve_thinking'])
            self.assertEqual(call.args[0][0]['reasoning_content'], 'hypothesis')

    def test_closed_thought_is_decoded_from_token_boundaries(self):
        tokenizer = SimpleNamespace(think_start_tokens=[11], think_end_tokens=[12], eos_token_ids=[0],
            rfind_think_start=lambda ids: 0, rfind_think_end=lambda ids: -1)
        row = dict(enable_thinking=True, prompt_ids=[11], tokens=[2, 3, 12, 8, 0], text='raw')
        with mock.patch.object(g, 'decode_tokens', side_effect=lambda tok, ids, eos: json.dumps(ids)):
            g.split_answer(tokenizer, row)
        self.assertEqual(row['reasoning_content'], '[2, 3]')
        self.assertEqual(row['reasoning_token_count'], 2)
        self.assertEqual(row['answer_text'], '[8, 0]')
        self.assertEqual(row['tokens'], [2, 3, 12, 8, 0])

    def test_tool_options_require_correct_types(self):
        tokenizer = SimpleNamespace(has_thinking=True)
        for options in [dict(tools={}), dict(preserve_thinking='yes')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                g.render(tokenizer, [], enable_thinking=True, **options)
