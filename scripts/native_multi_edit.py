"""Atomic original-relative edits with automatic witnesses; no model execution."""
import json
from pathlib import Path
import re

from scripts import qwen_native_tools

PROMPT = Path(__file__).resolve().parents[1] / 'configs/prompts/attack-families-broad-draft.txt'
TOOLS = [dict(type='function', function=dict(name='test_edits', description=(
    'Apply related nonoverlapping replacements against the ORIGINAL program and test every shown case. '
    'The checker selects a measured wrong-answer witness automatically. Submit one coherent fault.'),
    parameters=dict(type='object', properties=dict(edits=dict(type='array', minItems=1,
        items=dict(type='object', properties=dict(old=dict(type='string', minLength=1), new=dict(type='string')),
                   required=['old', 'new'], additionalProperties=False))), required=['edits'], additionalProperties=False)))]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate edit field: ' + key)
        result[key] = value
    return result


def gemma_edits(text):
    """Parse only arrays of string-valued old/new objects in the pinned native notation."""
    cursor = 0
    def punctuation(expected):
        nonlocal cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if not text.startswith(expected, cursor):
            raise ValueError('Expected ' + expected + ' in Gemma edits')
        cursor += len(expected)
    def string(key=False):
        nonlocal cursor
        punctuation('')
        if text.startswith('<|"|>', cursor):
            end = text.find('<|"|>', cursor + 5)
            if end < 0:
                raise ValueError('Unclosed Gemma string')
            result, cursor = text[cursor + 5:end], end + 5
            return result
        if text.startswith('"', cursor):
            value, size = json.JSONDecoder().raw_decode(text[cursor:])
            cursor += size
            return value
        match = re.match(r'(old|new)\b', text[cursor:]) if key else None
        if not match:
            raise ValueError('Require native string value')
        cursor += len(match[0])
        return match[0]
    def edit():
        punctuation('{')
        fields = []
        while True:
            key = string(key=True)
            punctuation(':')
            fields.append((key, string()))
            punctuation('')
            if text.startswith('}', cursor):
                break
            punctuation(',')
        punctuation('}')
        return unique_object(fields)
    punctuation('[')
    edits = []
    punctuation('')
    while not text.startswith(']', cursor):
        try:
            edits.append(edit())
        except ValueError as error:
            raise ValueError(f'Edit {len(edits) + 1}: {error}') from error
        punctuation('')
        if text.startswith(']', cursor):
            break
        punctuation(',')
        punctuation('')
        if text.startswith(']', cursor):
            raise ValueError('Trailing comma in edits')
    punctuation(']')
    punctuation('')
    if cursor != len(text):
        raise ValueError('Unexpected Gemma edits suffix')
    return edits


def parse_call(answer):
    if not isinstance(answer, str):
        raise ValueError('Require a completed test_edits call after thinking')
    if '<|tool_call>' in answer:
        prefix, body = answer.split('<|tool_call>', 1)
        match = re.fullmatch(r'call:test_edits\{edits:(.*)\}<tool_call\|>\s*', body, re.S)
        if not match:
            raise ValueError('Require exactly one Gemma test_edits call')
        edits = gemma_edits(match[1])
    else:
        parts = answer.split('<tool_call>', 1)
        if len(parts) != 2:
            raise ValueError('Require exactly one native test_edits call')
        prefix, body = parts
        match = re.fullmatch(r'\s*<function=test_edits>\s*<parameter=edits>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>\s*', body, re.S)
        if not match:
            raise ValueError('Require exactly one edits parameter and no suffix')
        edits = json.loads(match[1], object_pairs_hook=unique_object)
    if not isinstance(edits, list) or not edits:
        raise ValueError('edits must be a nonempty array')
    return prefix, edits


def apply_edits(original, edits):
    if not isinstance(edits, list) or not edits:
        raise ValueError('edits must be a nonempty array')
    intervals = []
    for i, edit in enumerate(edits, 1):
        if (not isinstance(edit, dict) or set(edit) != {'old', 'new'}
                or not isinstance(edit['old'], str) or not edit['old'] or not isinstance(edit['new'], str)):
            raise ValueError(f'Edit {i}: require nonempty old and string new; exactly two fields')
        matches = list(re.finditer('(?=' + re.escape(edit['old']) + ')', original))
        if len(matches) != 1:
            raise ValueError(f'Edit {i}: expected exactly one original match; match_count={len(matches)}')
        start = matches[0].start()
        intervals.append((start, start + len(edit['old']), edit['new'], i))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f'Edits {left[3]} and {right[3]} overlap')
    result = original
    for start, end, new, _ in reversed(intervals):
        result = result[:start] + new + result[end:]
    try:
        compile(result, '<multi-edit>', 'exec')
    except SyntaxError as error:
        raise ValueError(f'Candidate syntax error at line {error.lineno}, column {error.offset}: {error.msg}') from error
    return result.encode('utf-8')


def configure(recipe):
    recipe.update(system='You test Python programs using test_edits. Every call consumes one proposal turn.',
                  initial=PROMPT.read_text(), repair='Revise one coherent fault using the measured results.',
                  tools=TOOLS, preserve_thinking=True)


def messages(original, recipe):
    system, user = original
    content = user['content'].split('\nSubmit one plausible localized edit promptly.')[0]
    return [system, dict(role='user', content=content)]


class Extractor(qwen_native_tools.Extractor):
    def __call__(self, row, attempt):
        try:
            prefix, edits = parse_call(row.get('answer_text'))
            attempt.update(native_tool_call=True, call_id=f"call-{attempt['seed']}", tool_name='test_edits',
                           edits=edits, edit_json_valid=True)
            attempt['edit_schema_valid'] = all(isinstance(e, dict) and set(e) == {'old', 'new'}
                and isinstance(e['old'], str) and bool(e['old']) and isinstance(e['new'], str) for e in edits)
            attempt['conversation'][-1].update(content=prefix, tool_calls=[dict(id=attempt['call_id'],
                type='function', function=dict(name='test_edits', arguments=dict(edits=edits)))])
            program = apply_edits(self.originals[attempt['position']], edits)
            identity = (attempt['position'], json.dumps(edits, sort_keys=True))
            attempt['edit_duplicate'] = identity in self.seen
            self.seen.add(identity)
            attempt.update(edit_applicable=True, candidate_id=f"C{attempt['seed']}")
            return program
        except (ValueError, TypeError) as error:
            attempt['submission_feedback'] = str(error)
            raise ValueError(str(error)) from error

    def accept_truncated(self, row):
        try:
            parse_call(row.get('answer_text'))
            return True
        except (ValueError, TypeError):
            return False

    def repair_cost(self, parent, recipe):
        return (recipe['thinking_repair_max_tokens'] if 'verification' in parent
                and parent['position'] not in self.thinking_repairs else recipe['repair_max_tokens'])

    def repair(self, initial, parent, recipe, seed):
        limit = self.repair_cost(parent, recipe)
        thinking = 'verification' in parent and parent['position'] not in self.thinking_repairs
        if thinking:
            self.thinking_repairs.add(parent['position'])
        if 'verification' in parent:
            result = json.loads(parent['verification']['feedback']['message'])
        else:
            result = dict(protocol_status='no_checked_candidate', message=parent.get('submission_feedback',
                'No complete call was extracted; use your previous hypothesis to submit a complete call.'))
        result['request'] = 'Submit one test_edits call against the ORIGINAL program.'
        response = dict(role='tool' if parent.get('native_tool_call') else 'user', content=json.dumps(result))
        if parent.get('native_tool_call'):
            response['tool_call_id'] = parent['call_id']
        history = parent['conversation'] + [response]
        self.histories[parent['position']] = history
        parent['repair_feedback'] = dict(text=response['content'])
        request = dict(initial, messages=history, seed=seed, max_tokens=limit, enable_thinking=thinking,
                       temperature=1.0 if thinking else .7, top_p=.95 if thinking else .8, top_k=20)
        request.pop('reasoning_effort', None)
        if thinking and recipe.get('thinking_repair_effort'):
            request['reasoning_effort'] = recipe['thinking_repair_effort']
        return request
