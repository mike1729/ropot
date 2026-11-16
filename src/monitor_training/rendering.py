"""Native tokenizer boundary for frozen PASS/FLAG inputs and continuations.

No tokenizer or model package is imported here. The caller supplies the pinned
tokenizer and runs ``render_groups`` over the complete prepared view before any
development generation. Public bytes remain unchanged in messages; the pinned
native template owns rendering, including its whitespace handling.
"""

import hashlib
import json

from . import interface


LABELS = ("PASS", "FLAG")


def token_digest(input_ids):
    return hashlib.sha256(json.dumps(input_ids, separators=(",", ":")).encode("ascii")).hexdigest()


class Renderer:
    def __init__(self, tokenizer, *, max_total_tokens, max_new_tokens, terminator_id):
        if terminator_id not in tokenizer.all_special_ids:
            raise ValueError("Declared answer terminator must be a tokenizer special token")
        self.tokenizer = tokenizer
        self.max_total_tokens = max_total_tokens
        self.max_new_tokens = max_new_tokens
        self.special_tokens = tuple(token for token in
                                    {*tokenizer.all_special_tokens, *tokenizer.get_added_vocab()} if token)
        self.continuations = {}
        for label in LABELS:
            ids = tokenizer.encode(label, add_special_tokens=False)
            if self._decode(ids) != label:
                raise ValueError(f"Unsupported tokenization of {label}")
            if len(ids) + 1 > max_new_tokens:
                raise ValueError(f"Output cap cannot contain complete {label} plus terminator")
            self.continuations[label] = ids + [terminator_id]
        probe = interface.build_messages("Template check.", "pass")
        if self._template(probe, generation_prompt=True) == self._template(probe, generation_prompt=True, thinking=True):
            raise ValueError("Native template does not demonstrate enable_thinking support")

    def _decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def _template(self, messages, *, generation_prompt, thinking=False):
        return self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                  add_generation_prompt=generation_prompt, enable_thinking=thinking)

    def render(self, row):
        """Return the unchanged row plus native prefix IDs and their digest."""
        messages = interface.build_messages(row["statement"], row["program"])
        if any(token in message["content"] for token in self.special_tokens for message in messages):
            raise ValueError("Unsupported literal special token in public model input")
        prefix = self._template(messages, generation_prompt=True)
        if len(prefix) + self.max_new_tokens > self.max_total_tokens:
            raise ValueError("Inference prefix plus full output cap exceeds token ceiling")
        for label, continuation in self.continuations.items():
            complete = self._template(messages + [{"role": "assistant", "content": label}],
                                      generation_prompt=False)
            expected = prefix + continuation
            if complete[:len(expected)] != expected:
                raise ValueError(f"Native {label} template disagrees with prefix/label/terminator boundary")
            suffix = complete[len(expected):]
            if set(suffix).intersection(self.tokenizer.all_special_ids) or self._decode(suffix).strip():
                raise ValueError("Native template has unsupported content after answer terminator")
        return dict(row, input_ids=prefix, input_ids_sha256=token_digest(prefix))

    def render_groups(self, grouped_rows):
        """Preflight all supplied groups; report duplicates without excluding rows."""
        rendered, by_tokens = {}, {}
        for group, rows in grouped_rows.items():
            rendered[group] = []
            for row in rows:
                value = self.render(row)
                rendered[group].append(value)
                identity = dict(group=group, task_id=row["task_id"], role=row["role"])
                previous = by_tokens.setdefault(tuple(value["input_ids"]),
                                                dict(input_ids_sha256=value["input_ids_sha256"], rows=[]))["rows"]
                for other in previous:
                    if (group == "development") != (other["group"] == "development"):
                        raise ValueError("Identical token inputs cross construction/development boundary")
                    if any(interface.target_for(row["role"], branch) != interface.target_for(other["role"], branch)
                           for branch in interface.BRANCHES):
                        raise ValueError("Identical token inputs have conflicting targets")
                previous.append(identity)
        duplicates = [entry for entry in by_tokens.values() if len(entry["rows"]) > 1]
        return rendered, duplicates
