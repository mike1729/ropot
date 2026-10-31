"""Effort is forwarded to both tokenizer render calls, not just saved as metadata."""
import unittest
from tests.test_thinking_generation import ThinkingTokenizer, g


class EffortTests(unittest.TestCase):
    def test_medium_effort_reaches_text_and_token_rendering(self):
        tokenizer = ThinkingTokenizer('qwen')
        row = g.render(tokenizer, [dict(role='user', content='Generate code')],
                       enable_thinking=True, reasoning_effort='medium')
        self.assertEqual(row['reasoning_effort'], 'medium')
        self.assertEqual(tokenizer.apply_chat_template.call_count, 2)
        for call in tokenizer.apply_chat_template.call_args_list:
            self.assertEqual(call.kwargs['reasoning_effort'], 'medium')

    def test_invalid_or_disabled_effort_is_rejected(self):
        for effort, enabled in (('high', True), ('medium', False)):
            with self.subTest(effort=effort), self.assertRaises(ValueError):
                g.render(ThinkingTokenizer('qwen'), [], enable_thinking=enabled, reasoning_effort=effort)
