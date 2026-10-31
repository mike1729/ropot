"""Bounded native test_edit calls with retained reasoning and tool history."""
import json
import re

from scripts import qwen_edits

TOOLS = [dict(type='function', function=dict(name='test_edit', description=(
    'Apply one exact replacement to the ORIGINAL program and test its shown cases. '
    'Returns candidate_id, measured results and repair guidance. Repeated code reuses execution; '
    'For target-only corrections use retarget(candidate_id, target_case); do not copy old or new.'), parameters=dict(
        type='object', properties=dict(old=dict(type='string'), new=dict(type='string'),
            target_case=dict(type='integer', minimum=1)), required=['old', 'new', 'target_case'],
        additionalProperties=False))),
    dict(type='function', function=dict(name='retarget', description=(
        'Keep a previously tested candidate program unchanged and nominate a new menu case. '
        'Use its candidate_id and an eligible_target_cases value. Reuses execution results; '
        'recomputes acceptance and certificate. Consumes one proposal turn.'), parameters=dict(
        type='object', properties=dict(candidate_id=dict(type='string'),
            target_case=dict(type='integer', minimum=1)), required=['candidate_id', 'target_case'],
        additionalProperties=False)))]
TOY = '''Toy trajectory, unrelated to the development tasks (illustrative, not measured evidence):
Assistant: I have an uncertain rounding hypothesis; test it before simulating cases.
<tool_call>
<function=test_edit>
<parameter=old>
return round(x)
</parameter>
<parameter=new>
return int(x)
</parameter>
<parameter=target_case>
1
</parameter>
</function>
</tool_call>
Tool: Candidate toy-C1: example x=-1.8 failed; expected -2, observed -1.
Assistant: Truncation toward zero broke the negative example. My next test replaces the
original return round(x) with return math.floor(x), rather than predicting all outputs.
'''
DIRECT_CALL = ('Reply with exactly one native call: retarget(candidate_id, target_case) to preserve qualifying '
               'code, or test_edit(old, new, target_case) for a different edit against the ORIGINAL program.')


def action(turns, direct, thinking_repair=False):
    return (
        'Look for one coherent localized fault in boundary handling, tie-breaking, transitions, state updates '
        'or a plausible optimization. Added branches, comparisons and problem-related special cases are allowed. '
        'Pass every statement example and fail at most floor(menu_size/2) menu cases, including the target. '
        'Do not add input fingerprints '
        'or arbitrary output overrides. Once you have one concrete edit, call test_edit BEFORE predicting its '
        'outputs or considering another edit. You do not need confidence that it passes. target_case may be your '
        'best guess: the checker measures every shown case, so a wrong nomination can be corrected. Tool responses '
        'are measured checker results; your earlier reasoning and nominations are unverified hypotheses. '
        'Emit exactly one native tool call per turn. Do not give a final program or JSON answer. '
        'When eligible_target_cases is nonempty, use retarget(candidate_id, target_case) with an observed '
        'eligible target; do not copy or change the edit. Every call consumes one proposal turn. '
        'If any violation besides named_case remains, retargeting cannot help: choose a different edit. '
        'If every shown case passes or no menu case returns a valid wrong integer, retargeting cannot help: '
        f'choose a different edit. At most {turns} turns are available per task and chain'
        + ('; only the first turn and the turn after the first measured result allow brief thinking, other turns '
           'require a direct call' if direct and thinking_repair else
           '; only the first allows thinking, later turns require a direct call' if direct else '')
        + '. Stop after acceptance.')


def configure(recipe):
    if not recipe.get('enable_thinking') or recipe['max_repairs'] < 1:
        raise ValueError('Native-tool recipe requires initial thinking and at least one repair turn')
    text = action(recipe['max_repairs'] + 1, recipe.get('repair_enable_thinking') is False,
                  'thinking_repair_max_tokens' in recipe)
    recipe.update(tools=TOOLS, preserve_thinking=True, system='You test Python programs using test_edit and retarget. ' + text)
    recipe['initial'] = recipe['initial'].replace(
        'Name that case with exactly one declaration as the first nonblank line: '
        '# case: <1-based menu-position>.', 'Name that case using the target_case tool argument.')
    recipe['initial'] = recipe['initial'].replace('Return only the complete replacement program.', '')
    recipe['repair'] = text


def messages(original, recipe):
    system, user = original
    content = user['content'].split('\nSubmit one plausible localized edit promptly.')[0]
    return [system, dict(role='user', content=content + '\n\n' + recipe['repair'] + '\n\n' + TOY)]


def gemma_messages(messages):
    """Use the pinned Gemma template's call notation in the illustrative trajectory."""
    toy = ('Toy trajectory, unrelated to these tasks:\n'
           'Assistant: <|tool_call>call:test_edit{old:<|"|>return round(x)<|"|>,'
           'new:<|"|>return int(x)<|"|>,target_case:1}<tool_call|>\n'
           'Tool: Candidate toy-C1: example x=-1.8 failed; expected -2, observed -1.\n'
           'Assistant: Truncation broke the negative example. Try floor against the original code.')
    return [dict(m,content=m['content'].replace(TOY,toy)) for m in messages]


def gemma_call(answer):
    match = re.fullmatch(r'(.*?)<\|tool_call>call:(test_edit|retarget)\{(.*)\}<tool_call\|>\s*',answer,re.S)
    if not match or '<|tool_call>' in match[1]:
        raise ValueError('Require exactly one completed Gemma native call')
    body, parts, cursor = match[3], [], 0
    for p in re.finditer(r'(old|new|candidate_id|target_case):(<\|"\|>.*?<\|"\|>|[1-9][0-9]*)',body,re.S):
        if body[cursor:p.start()].strip() != (',' if cursor else ''):
            raise ValueError('Unexpected Gemma tool arguments')
        value=p[2]
        if p[1]=='target_case' and not value.isdigit():raise ValueError('target_case must be an integer')
        if p[1]!='target_case' and not value.startswith('<|"|>'):raise ValueError('Require string argument')
        value=value[5:-5] if value.startswith('<|"|>') else value
        parts.append(f'<parameter={p[1]}>\n{value}\n</parameter>');cursor=p.end()
    if body[cursor:].strip():raise ValueError('Unexpected Gemma call suffix')
    return match[1]+'<tool_call><function='+match[2]+'>'+''.join(parts)+'</function></tool_call>'


def parse_call(answer):
    if isinstance(answer,str) and '<|tool_call>' in answer:
        answer=gemma_call(answer)
    if not isinstance(answer, str) or answer.count('<tool_call>') != 1:
        raise ValueError('Require one completed native test_edit or retarget call after thinking')
    prefix, call = answer.split('<tool_call>', 1)
    match = re.fullmatch(r'\s*<function=(test_edit|retarget)>\s*(.*?)\s*</function>\s*</tool_call>\s*', call, re.S)
    if not match:
        raise ValueError('Require native test_edit or retarget parameter blocks and no suffix')
    body, args, cursor = match[2], {}, 0
    required = {'old', 'new', 'target_case'} if match[1] == 'test_edit' else {'candidate_id', 'target_case'}
    for part in re.finditer(r'<parameter=(old|new|candidate_id|target_case)>(.*?)</parameter>', body, re.S):
        if body[cursor:part.start()].strip() or part[1] in args:
            raise ValueError('Unexpected text or repeated tool parameter')
        value = part[2]
        value = value[1:] if value.startswith('\n') else value
        value = value[:-1] if value.endswith('\n') else value
        args[part[1]], cursor = value, part.end()
    if body[cursor:].strip() or set(args) != required:
        raise ValueError('Require exactly ' + ', '.join(sorted(required)))
    if 'candidate_id' in args and not re.fullmatch(r'C[0-9]+', args['candidate_id']):
        raise ValueError('candidate_id must be an exact previously returned ID')
    if not re.fullmatch(r'[1-9][0-9]*', args['target_case'].strip()):
        raise ValueError('target_case must be a positive integer')
    args['target_case'] = int(args['target_case'])
    return prefix, args


class Extractor(qwen_edits.Extractor):
    def __init__(self, entries, initial, loop):
        super().__init__(entries)
        self.histories = json.loads(json.dumps(initial))
        self.loop = loop
        self.tokens = [0] * len(entries)
        self.candidates = {}
        self.thinking_repairs = set()

    def observe(self, row, attempt):
        self.tokens[attempt['position']] += row.get('generation_tokens', len(row['tokens']))
        attempt.update(cumulative_output_tokens=self.tokens[attempt['position']],
                       thinking=bool(row.get('enable_thinking')), hit_output_cap=row['status'] == 'length',
                       empty_reply=not row['text'].strip())
        assistant = dict(role='assistant', content=row.get('answer_text') or '',
            reasoning_content=row.get('reasoning_content') or (row['text'] if row.get('answer_text') is None else ''))
        attempt['conversation'] = self.histories[attempt['position']] + [assistant]

    def __call__(self, row, attempt):
        try:
            prefix, args = parse_call(row.get('answer_text'))
        except ValueError as error:
            attempt['submission_feedback'] = str(error)
            raise
        attempt.update(native_tool_call=True, call_id=f"call-{attempt['seed']}")
        name = 'retarget' if 'candidate_id' in args else 'test_edit'
        attempt['tool_name'] = name
        assistant = attempt['conversation'][-1]
        assistant.update(content=prefix, tool_calls=[dict(id=attempt['call_id'], type='function',
            function=dict(name=name, arguments=args))])
        edit = args
        if name == 'retarget':
            attempt['retarget_source_id'] = args['candidate_id']
            if not 1 <= args['target_case'] <= self.menu_sizes[attempt['position']]:
                attempt['submission_feedback'] = (f"target_case must be in 1..{self.menu_sizes[attempt['position']]} "
                    'for this task menu. Use the stored candidate_id and a valid target; do not copy old or new.')
                raise ValueError(attempt['submission_feedback'])
            previous = self.candidates.get((attempt['position'], args['candidate_id']))
            if previous is None:
                attempt['submission_feedback'] = 'Unknown candidate_id for this task and chain; no candidate was checked.'
                raise ValueError(attempt['submission_feedback'])
            edit = dict(previous, target_case=args['target_case'])
        program = super().__call__(dict(text=json.dumps(edit)), attempt)
        attempt['candidate_id'] = f"C{attempt['seed']}"
        self.candidates[(attempt['position'], attempt['candidate_id'])] = dict(edit)
        return program

    def accept_truncated(self, row):
        try:
            parse_call(row.get('answer_text'))
            return True
        except ValueError:
            return False

    def repair(self, initial, parent, recipe, seed):
        request = self.loop.repair_request(initial, parent, recipe, seed)
        if parent['extraction'] == 'success':
            result = dict(candidate_id=parent['candidate_id'],
                          measured_result=json.loads(parent['repair_feedback']['text']), request=DIRECT_CALL)
            if result['measured_result'].get('eligible_target_cases'):
                result['request'] = ('Call retarget with candidate_id=' + parent['candidate_id'] +
                    ' and target_case chosen from eligible_target_cases. Do not copy old or new.')
                result['measured_result']['guidance'] = result['request']
        elif parent['extraction'] == 'length':
            result = dict(protocol_status='no_complete_tool_call', message=(
                'No complete tool call was submitted before the output cap. No candidate was created or checked.'),
                request='Reply now with one test_edit call for the most promising concrete edit in your reasoning.')
        else:
            result = dict(protocol_status='invalid_tool_call', message=parent.get('submission_feedback', parent.get('error', 'Invalid call'))
                          + ' No candidate was executed.', request=DIRECT_CALL)
        parent['repair_feedback'] = dict(text=json.dumps(result, ensure_ascii=False))
        response = dict(role='tool' if parent.get('native_tool_call') else 'user',
                        content=json.dumps(result, ensure_ascii=False))
        if parent.get('native_tool_call'):
            response['tool_call_id'] = parent['call_id']
        history = parent['conversation'] + [response]
        self.histories[parent['position']] = history
        request = dict(request, messages=history)
        # One bounded thinking repair follows the first measured result; initial thinking sampling applies.
        if 'thinking_repair_max_tokens' in recipe and 'verification' in parent and parent['position'] not in self.thinking_repairs:
            self.thinking_repairs.add(parent['position'])
            result['request'] += ' You may think briefly before the call.'
            response['content'] = json.dumps(result, ensure_ascii=False)
            parent['repair_feedback'] = dict(text=response['content'])
            return dict(request, max_tokens=recipe['thinking_repair_max_tokens'])
        if recipe.get('repair_enable_thinking') is False:
            request['enable_thinking'] = False
            request.pop('reasoning_effort', None)
        request.update({k: recipe['repair_' + k] for k in ('temperature', 'top_p') if 'repair_' + k in recipe})
        return request


def metrics(groups, max_repairs=1):
    first, counts, accepted = [], dict(initial_natural_calls=0, action_only_calls=0, direct_calls_after_thinking_cutoff=0,
        direct_requests_after_initial_no_call=0, direct_calls_after_initial_no_call=0,
        direct_requests_after_thinking_cutoff=0, revisions_after_feedback=0, retargets_after_feedback=0,
        identical_resubmissions=0, retarget_opportunities=0, last_turn_retarget_opportunities=0,
        retarget_by_id_calls=0, eligible_retargets_without_copying_errors=0,
        eligible_retargets_accepted=0, futile_target_only_retries=0, retarget_selection_errors=0,
        unparsed_replies=0, empty_replies=0, inapplicable_calls=0, length_stops_thinking=0,
        length_stops_direct=0, repairs_exhausted=0, budget_exhausted=0, thinking_repairs=0,
        thinking_repair_calls=0, thinking_repair_length_stops=0), set()
    for chain, attempts in enumerate(groups):
        seen, parents = set(), {a['attempt']: a for a in attempts}
        candidates = {(a['position'], a['candidate_id']): a for a in attempts if 'candidate_id' in a}
        for attempt in attempts:
            parent = parents.get(attempt.get('parent'), {})
            verification = attempt.get('verification', {})
            if verification and not verification.get('execution_reused') and attempt['position'] not in seen:
                first.append(dict(chain=chain, position=attempt['position'],
                    output_tokens=attempt['cumulative_output_tokens'], candidate_id=attempt['candidate_id']))
                seen.add(attempt['position'])
            no_call = parent.get('round') == 0 and not parent.get('native_tool_call') and not attempt.get('thinking')
            cutoff = no_call and parent.get('hit_output_cap', parent.get('extraction') == 'length')
            counts['initial_natural_calls'] += attempt['round'] == 0 and bool(attempt.get('native_tool_call'))
            counts['action_only_calls'] += not attempt.get('thinking') and bool(attempt.get('native_tool_call'))
            counts['direct_requests_after_thinking_cutoff'] += cutoff
            counts['direct_calls_after_thinking_cutoff'] += cutoff and bool(attempt.get('native_tool_call'))
            counts['direct_requests_after_initial_no_call'] += no_call
            counts['direct_calls_after_initial_no_call'] += no_call and bool(attempt.get('native_tool_call'))
            measured = parent
            while measured and 'verification' not in measured:
                measured = parents.get(measured.get('parent'), {})
            by_id = attempt.get('tool_name') == 'retarget'
            counts['retarget_by_id_calls'] += by_id
            if by_id:
                measured = candidates.get((attempt['position'], attempt.get('retarget_source_id')), {})
            if 'verification' in measured and 'edit' in attempt:
                same = all(attempt['edit'][k] == measured['edit'][k] for k in ('old', 'new'))
                retarget = same and attempt['edit']['target_case'] != measured['edit']['target_case']
                counts['revisions_after_feedback'] += not same
                counts['retargets_after_feedback'] += retarget
                counts['identical_resubmissions'] += same and not retarget
                if same and (retarget or by_id):
                    targets = measured['verification'].get('retarget_eligible_cases', [])
                    selected = attempt['edit']['target_case'] in targets
                    counts['futile_target_only_retries'] += not bool(targets)
                    counts['retarget_selection_errors'] += bool(targets) and not selected
                    counts['eligible_retargets_without_copying_errors'] += by_id and selected
                    counts['eligible_retargets_accepted'] += (by_id and selected
                        and verification.get('status') == 'passed')
            if verification.get('retarget_eligible_cases'):
                counts['retarget_opportunities'] += 1
                counts['last_turn_retarget_opportunities'] += attempt['round'] == max_repairs
            counts['unparsed_replies'] += attempt.get('extraction') == 'format' and not attempt.get('native_tool_call')
            counts['empty_replies'] += bool(attempt.get('empty_reply'))
            counts['inapplicable_calls'] += bool(attempt.get('native_tool_call')) and not attempt.get('edit_applicable')
            if attempt.get('hit_output_cap', attempt.get('extraction') == 'length'):
                counts['length_stops_thinking' if attempt.get('thinking') else 'length_stops_direct'] += 1
            thinking_repair = attempt['round'] > 0 and bool(attempt.get('thinking'))
            counts['thinking_repairs'] += thinking_repair
            counts['thinking_repair_calls'] += thinking_repair and bool(attempt.get('native_tool_call'))
            counts['thinking_repair_length_stops'] += thinking_repair and bool(attempt.get('hit_output_cap'))
            counts['repairs_exhausted'] += attempt.get('state') == 'repairs_exhausted'
            counts['budget_exhausted'] += attempt.get('state') == 'budget_exhausted'
            if verification.get('status') == 'passed':
                accepted.add((attempt['position'], attempt['candidate']['program_sha256']))
    return dict(counts, naturally_emitted_calls=sum(bool(a.get('native_tool_call')) for g in groups for a in g),
                forced_transitions=counts['direct_requests_after_initial_no_call'], first_executed_candidates=first,
                accepted_distinct_programs=len(accepted), accepted_task_positions=sorted({p for p, _ in accepted}),
                semantic_approved=None, distinct_mechanisms=None)
