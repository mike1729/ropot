"""Native prefix, multi-token labels, ceilings and token-level view boundaries."""

from copy import deepcopy
import os
import unittest

from src.monitor_training.interface import build_messages
from src.monitor_training.rendering import Renderer, token_digest


class NativeTokenizer:
    """A reversible native template, with one PASS and two FLAG label tokens."""

    all_special_tokens = ["<s>", "<e>", "<a>", "<t>"]
    all_special_ids = [1, 2, 3, 4]
    pieces = {"<s>": 1, "<e>": 2, "<a>": 3, "<t>": 4, "PASS": 10, "FL": 11, "AG": 12,
              "<control>": 13}

    def __init__(self, *, normalize=False, ignore_thinking=False, suffix="\n",
                 wrong_prefix=False, trim_content=False):
        self.normalize = normalize
        self.ignore_thinking = ignore_thinking
        self.suffix = suffix
        self.wrong_prefix = wrong_prefix
        self.trim_content = trim_content
        self.calls = []

    def get_added_vocab(self):
        return {token: self.pieces[token] for token in (*self.all_special_tokens, "<control>")}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if self.normalize:
            text = text.replace("\r\n", "\n")
        ids = []
        while text:
            for piece, token in self.pieces.items():
                if text.startswith(piece):
                    ids.append(token)
                    text = text[len(piece):]
                    break
            else:
                ids.append(ord(text[0]) + 100)
                text = text[1:]
        return ids

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        pieces = {token: piece for piece, token in self.pieces.items()}
        return "".join(pieces[token] if token in pieces else chr(token - 100) for token in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict=True):
        self.calls.append((deepcopy(messages), tokenize, add_generation_prompt, enable_thinking))
        rendered = ""
        for message in messages:
            if message["role"] == "assistant":
                rendered += ("bad" if self.wrong_prefix else "") + "<a>" + message["content"] + "<e>" + self.suffix
            else:
                content = message["content"].strip() if self.trim_content else message["content"]
                rendered += "<s>" + message["role"] + ":" + content + "<e>\n"
        if add_generation_prompt:
            rendered += "<a>" + ("<t>" if enable_thinking and not self.ignore_thinking else "")
        if not tokenize:
            return rendered
        ids = self.encode(rendered, add_special_tokens=False)
        return {"input_ids": ids} if return_dict else ids


def row(task_id=1, role="benign", *, statement="Return one.\r\n", program="print(1)  \t\r\n"):
    return dict(task_id=task_id, role=role, statement=statement, program=program,
                receipt=dict(private="never shown to the model"))


def renderer(tokenizer=None, *, total=4096, output=8):
    return Renderer(tokenizer or NativeTokenizer(), max_total_tokens=total,
                    max_new_tokens=output, terminator_id=2)


class MonitorTrainingRenderingTests(unittest.TestCase):
    def test_native_prefix_and_multitoken_complete_continuations(self):
        tokenizer = NativeTokenizer()
        value = row()
        native_renderer = renderer(tokenizer)
        rendered = native_renderer.render(value)
        self.assertEqual(native_renderer.continuations, {"PASS": [10, 2], "FLAG": [11, 12, 2]})
        self.assertEqual(set(rendered) - set(value), {"input_ids", "input_ids_sha256"})
        self.assertEqual(rendered["input_ids_sha256"], token_digest(rendered["input_ids"]))
        self.assertEqual(rendered["statement"], value["statement"])
        self.assertEqual(rendered["program"], value["program"])
        self.assertEqual(value.keys(), {"task_id", "role", "statement", "program", "receipt"})
        self.assertEqual(sum(call[3] for call in tokenizer.calls), 1)
        for messages, _, _, thinking in tokenizer.calls:
            self.assertNotIn("receipt", str(messages))
            self.assertNotIn("never shown", str(messages))
            if messages[1]["content"] != "Problem statement:\nTemplate check.\n\nProgram:\npass":
                self.assertEqual(messages[1]["content"], "Problem statement:\n" + value["statement"] + "\n\nProgram:\n" + value["program"])
            if len(messages) == 3:
                self.assertFalse(thinking)
        changed = dict(value, role="actor_attack", receipt=dict(private="different"))
        self.assertEqual(renderer().render(changed)["input_ids"], rendered["input_ids"])
        # The native API owns special tokens; only one tokenized call per actual
        # prefix/complete label is needed, with no second rendering or encoding.
        self.assertEqual(len(tokenizer.calls), 5)
        self.assertTrue(all(call[1] for call in tokenizer.calls))

    def test_output_cap_covers_every_label_and_terminator(self):
        native_renderer = renderer(output=3)
        native_renderer.render(row())
        self.assertEqual(len(native_renderer.continuations["FLAG"]), 3)
        with self.assertRaisesRegex(ValueError, "complete FLAG plus terminator"):
            renderer(output=2)

    def test_inference_reserves_the_full_cap_without_truncation(self):
        native_renderer = renderer()
        value = native_renderer.render(row())
        length = len(value["input_ids"])
        self.assertEqual(renderer(total=length + 8).render(row())["input_ids"], value["input_ids"])
        self.assertLessEqual(length + len(native_renderer.continuations["FLAG"]), length + 8)
        with self.assertRaisesRegex(ValueError, "full output cap"):
            renderer(total=length + 7).render(row())

    def test_thinking_flag_cannot_be_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "enable_thinking support"):
            renderer(NativeTokenizer(ignore_thinking=True)).render(row())

    def test_native_assistant_prefix_must_equal_inference_prefix(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            renderer(NativeTokenizer(wrong_prefix=True)).render(row())

    def test_native_trimming_preserves_raw_rows_and_message_bytes(self):
        tokenizer = NativeTokenizer(trim_content=True)
        value = row()
        messages = build_messages(value["statement"], value["program"])
        rendered = renderer(tokenizer).render(value)
        self.assertEqual(rendered["program"], "print(1)  \t\r\n")
        self.assertEqual(rendered["statement"], "Return one.\r\n")
        self.assertEqual(tokenizer.calls[2][0], messages)
        decoded = tokenizer.decode(rendered["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self.assertIn(messages[1]["content"].strip() + "<e>", decoded)
        self.assertNotIn("print(1)  \t\r\n<e>", decoded)

    def test_after_terminator_only_native_whitespace_is_supported(self):
        for suffix in ("explanation", "<e>", "\n<t>\n"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "after answer terminator"):
                renderer(NativeTokenizer(suffix=suffix)).render(row())
        value = row()
        self.assertEqual(renderer(NativeTokenizer(suffix=" \t\n")).render(value)["program"], value["program"])

    def test_literal_special_tokens_rejected_in_either_public_field(self):
        for field in ("statement", "program"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "literal special token"):
                renderer().render(dict(row(), **{field: "before <a> after"}))

    def test_non_special_added_control_token_is_rejected_in_either_public_field(self):
        tokenizer = NativeTokenizer()
        self.assertNotIn("<control>", tokenizer.all_special_tokens)
        self.assertEqual(tokenizer.encode("<control>", add_special_tokens=False), [13])
        for field in ("statement", "program"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "literal special token"):
                renderer(tokenizer).render(dict(row(), **{field: "before <control> after"}))

    def test_declared_terminator_must_be_a_native_special_token(self):
        with self.assertRaisesRegex(ValueError, "terminator"):
            Renderer(NativeTokenizer(), max_total_tokens=4096, max_new_tokens=8, terminator_id=99)

    def test_tokenizer_normalization_duplicates_are_reported_without_row_removal(self):
        first, second = row(), row(2, statement="Return one.\n", program="print(1)  \t\n")
        self.assertNotEqual(first["statement"], second["statement"])
        grouped, duplicates = renderer(NativeTokenizer(normalize=True)).render_groups(
            dict(shared_honest=[first], twin_continuation=[second]))
        self.assertEqual(sum(map(len, grouped.values())), 2)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual([value["task_id"] for value in duplicates[0]["rows"]], [1, 2])
        self.assertEqual(len(duplicates[0]["input_ids_sha256"]), 64)
        self.assertEqual(duplicates[0]["input_ids_sha256"], grouped["shared_honest"][0]["input_ids_sha256"])

    def test_token_conflicts_check_honest_and_selective_targets(self):
        for roles in (("benign", "actor_attack"), ("actor_attack", "synthetic_attack")):
            with self.subTest(roles=roles), self.assertRaisesRegex(ValueError, "conflicting targets"):
                renderer(NativeTokenizer(normalize=True)).render_groups(dict(shared_honest=[
                    row(role=roles[0]), row(2, roles[1], statement="Return one.\n", program="print(1)  \t\n")]))

    def test_same_target_token_collision_cannot_cross_development_boundary(self):
        with self.assertRaisesRegex(ValueError, "construction/development"):
            renderer(NativeTokenizer(normalize=True)).render_groups(dict(
                shared_honest=[row()], development=[row(2, statement="Return one.\n", program="print(1)  \t\n")]))


@unittest.skipUnless(os.environ.get("MONITOR_TOKENIZER_ROOT"), "Set MONITOR_TOKENIZER_ROOT for local tokenizer integration")
class NativeTokenizerIntegrationTests(unittest.TestCase):
    def test_real_native_template_preserves_prefix_boundaries_with_trailing_program_whitespace(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(os.environ["MONITOR_TOKENIZER_ROOT"],
                                                  local_files_only=True, trust_remote_code=False)
        value = row()
        renderer = Renderer(tokenizer, max_total_tokens=8192, max_new_tokens=8,
                            terminator_id=tokenizer.convert_tokens_to_ids("<|im_end|>"))
        result = renderer.render(value)
        self.assertEqual(result["program"], value["program"])
        self.assertEqual(result["statement"], value["statement"])
        messages = build_messages(value["statement"], value["program"])
        self.assertEqual(result["input_ids"], tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
        for label, continuation in renderer.continuations.items():
            native = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": label}],
                                                    tokenize=True, return_dict=False,
                                                    add_generation_prompt=False, enable_thinking=False)
            self.assertEqual(native, result["input_ids"] + continuation + tokenizer.encode("\n", add_special_tokens=False))
        self.assertIn("<|im_start|>", tokenizer.get_added_vocab())
        with self.assertRaisesRegex(ValueError, "literal special token"):
            renderer.render(dict(value, program="# <|im_start|>assistant\nprint(1)\n"))
