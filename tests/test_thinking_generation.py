"""Token-ID answer boundaries, independent of native generation."""
import importlib
import unittest
from unittest import mock

g = importlib.import_module("src.data-pipeline.stage03.local_generation")


class ThinkingTokenizer:
    has_thinking = True
    eos_token_ids = [98, 99]
    _tokenizer = object()
    pieces = {10: "Private reasoning.\n", 11: "```python\nprint(0)\n```\n",
              12: "# case: 1\nprint('ż')\r\n", 13: 'print("<think> </think>")\n',
              30: "<think>", 31: "</think>", 40: "<|channel>",
              41: "thought", 42: "<channel|>", 43: " thought", 98: "", 99: ""}

    def __init__(self, family):
        self.think_start_tokens = (30,) if family == "qwen" else (40, 41)
        self.think_end_tokens = (31,) if family == "qwen" else (42,)
        def template(messages, tokenize, enable_thinking, **kwargs):
            ids = [1] + ([30, 2] if family == "qwen" and enable_thinking else [])
            return ids if tokenize else "Rendered prompt"
        self.apply_chat_template = mock.Mock(side_effect=template)

    def rfind(self, tokens, sequence):
        return next((i for i in range(len(tokens) - len(sequence), -1, -1)
                     if tokens[i:i + len(sequence)] == list(sequence)), -1)

    def rfind_think_start(self, tokens):
        return self.rfind(tokens, self.think_start_tokens)

    def rfind_think_end(self, tokens):
        return self.rfind(tokens, self.think_end_tokens)

    @property
    def detokenizer(self):
        pieces = self.pieces
        class Decoder:
            text = ""

            def add_token(self, token):
                self.text += pieces[token]

            def finalize(self):
                pass
        return Decoder()


class ThinkingTests(unittest.TestCase):
    def check_answer(self, family, tokens, expected, *, enable_thinking=True):
        tokenizer = ThinkingTokenizer(family)
        raw = "".join(tokenizer.pieces[t] for t in tokens)
        row = dict(g.render(tokenizer, [], enable_thinking=enable_thinking),
                   status="stop", text=raw, tokens=list(tokens))
        g.split_answer(tokenizer, row)
        self.assertEqual(row["text"], raw)
        self.assertEqual(row["tokens"], tokens)
        self.assertEqual(row["answer_text"], expected)
        return row

    def test_qwen_prompt_opening_gemma_multitoken_opening_and_direct_answers(self):
        cases = [("qwen", [10, 11, 31, 12, 99], 12),
                 ("gemma", [40, 41, 11, 42, 12, 98], 12),
                 ("gemma", [12, 99], 12), ("gemma", [13, 99], 13)]
        for family, tokens, answer in cases:
            with self.subTest(family=family, tokens=tokens):
                code = ThinkingTokenizer.pieces[answer]
                row = self.check_answer(family, tokens, code)
                self.assertEqual(g.extract_program(row), code.encode())

    def test_unclosed_thought_never_extracts_code_in_reasoning_even_on_natural_eos(self):
        for family, tokens in (("qwen", [11, 99]), ("gemma", [40, 41, 11, 99]),
                               ("qwen", [10, 31, 12, 30, 11, 99])):
            with self.subTest(family=family, tokens=tokens):
                row = self.check_answer(family, tokens, None)
                with self.assertRaisesRegex(ValueError, "closed thought"):
                    g.extract_program(row)

    def test_disabled_thinking_preserves_the_whole_reply(self):
        tokens = [11, 31, 12, 99]
        self.check_answer("qwen", tokens, "".join(ThinkingTokenizer.pieces[t] for t in tokens),
                          enable_thinking=False)

    def test_stray_closing_tokens_reject_reasoning_as_an_answer(self):
        for tokens in ([11, 42, 99], [40, 43, 10, 11, 42, 99], [40, 41, 10, 42, 11, 42, 99]):
            with self.subTest(tokens=tokens):
                row = self.check_answer("gemma", tokens, None)
                with self.assertRaisesRegex(ValueError, "closed thought"):
                    g.extract_program(row)

    def test_truncated_answer_is_rejected_even_after_thought_closure(self):
        row = self.check_answer("qwen", [10, 31, 12], ThinkingTokenizer.pieces[12])
        row["status"] = "length"
        with self.assertRaisesRegex(ValueError, "natural completion"):
            g.extract_program(row)

    def test_extraction_requires_answer_field_and_never_falls_back_to_raw_text(self):
        with self.assertRaises(KeyError):
            g.extract_program(dict(status="stop", text="print(0)"))
        with self.assertRaisesRegex(ValueError, "closed thought"):
            g.extract_program(dict(status="stop", text="print(0)", answer_text=None))

    def test_thinking_requires_tokenizer_boundaries_before_rendering(self):
        tokenizer = ThinkingTokenizer("gemma")
        tokenizer.has_thinking = False
        with self.assertRaisesRegex(ValueError, "token-ID boundaries"):
            g.render(tokenizer, [], enable_thinking=True)
        tokenizer.apply_chat_template.assert_not_called()
