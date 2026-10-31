"""Exact replacements against original programs for development experiments."""
import json
import re


def configure(recipe):
    if recipe.get('enable_thinking'):
        raise ValueError('Exact-edit experiments require thinking disabled')
    recipe['system'] = 'You help test Python 3 programs. Return only one JSON edit object, without prose or fences.'
    recipe['initial'] = recipe['initial'].replace(
        'Name that case with exactly one declaration as the first nonblank line: '
        '# case: <1-based menu-position>.',
        'Name that case using target_case in the JSON edit object.')
    recipe['initial'] = recipe['initial'].replace(
        'Return only the complete replacement program.', '')
    recipe['repair'] = (
        'Return one revised JSON edit object against the ORIGINAL program, not the previous candidate. '
        'Use the measured feedback. If eligible_target_cases is nonempty, keep old and new EXACTLY '
        'unchanged and select target_case from that list. Do not change qualifying code. '
        'If no menu case returns a valid wrong integer, retargeting cannot help: propose a different edit. '
        'Repeated code reuses measured results; every reply still consumes a proposal slot. '
        'Do not simulate cases or recount limits; submit the edit directly.')


def messages(messages):
    system, user = messages
    content = user['content'].split('\nSubmit one plausible localized edit promptly.')[0]
    return [system, dict(role='user', content=content + (
        '\nReturn exactly {"target_case": 1, "old": "exact original text", "new": "replacement text"}. '
        'target_case is the 1-based menu position. old must be nonempty and occur exactly once '
        'in the ORIGINAL program, including whitespace. new may be empty. Multiline strings use JSON '
        'escapes. Propose one localized edit immediately. Do not simulate cases or recount mechanical '
        'limits. The checker executes your candidate; use its results in repair.'))]


class Extractor:
    def __init__(self, entries):
        self.originals = [e['benigns']['actor']['program'] for e in entries]
        self.menu_sizes = [len(e['partition']['menu']) for e in entries]
        self.seen = set()

    def __call__(self, row, attempt):
        text = row.get('answer_text')
        if text is None:
            text = row['text']
        attempt['repair_submission'] = text
        try:
            def unique_object(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError('Duplicate JSON field: ' + key)
                    obj[key] = value
                return obj
            edit = json.loads(text, object_pairs_hook=unique_object)
            attempt['edit_json_valid'] = True
            if (not isinstance(edit, dict) or set(edit) != {'target_case', 'old', 'new'}
                    or type(edit['target_case']) is not int
                    or not 1 <= edit['target_case'] <= self.menu_sizes[attempt['position']]
                    or not isinstance(edit['old'], str) or not edit['old']
                    or not isinstance(edit['new'], str)):
                raise ValueError('Require target_case in menu range, nonempty old, and string new; exactly three fields')
            attempt['edit'] = edit
            original = self.originals[attempt['position']]
            count = sum(1 for _ in re.finditer('(?=' + re.escape(edit['old']) + ')', original))
            attempt['edit_match_count'] = count
            if count != 1:
                raise ValueError(f'Exact replacement requires one match in the original program; match_count={count}')
            attempt['edit_applicable'] = True
            identity = (attempt['position'], edit['old'], edit['new'])
            if identity in self.seen:
                attempt['edit_duplicate'] = True
            self.seen.add(identity)
            program = (f"# case: {edit['target_case']}\n" + original.replace(edit['old'], edit['new'], 1)).encode('utf-8')
            compile(program, '<exact-edit>', 'exec')
            return program
        except (ValueError, SyntaxError) as error:
            attempt['submission_feedback'] = str(error)
            raise


def repair_guidance(verdict, narrowing=False):
    """Recommend only target corrections justified by measured results; narrowing advice is opt-in."""
    ineffective = bool(verdict['cases']) and all(row.get('status') == 'passed' for row in verdict['cases'])
    if verdict['violations']:
        targets = [int(row['id'].split(':')[1]) for row in verdict['cases']
                   if verdict['violations'] == ['named_case']
                   and row['id'].startswith('menu:') and row['status'] == 'wrong_answer']
        feedback = json.loads(verdict['feedback']['message'])
        verdict['retarget_eligible_cases'] = targets
        narrow = narrowing and set(verdict['violations']) - {'named_case'} == {'menu_failures'}
        feedback.update(eligible_target_cases=targets,
            repair_decision='preserve_edit_and_retarget' if targets else 'narrow_edit' if narrow else 'different_edit_required',
            **({'narrowing': 'Every example passes and the edit changes menu outputs, but too many menu cases fail. '
                'The location may be right: keep the same site and make the change narrower so it affects fewer '
                'inputs, for example a smaller shift or a change that only matters at a boundary value. '
                'Do not add special cases for particular inputs.'} if narrow else {}),
            guidance=('Keep old and new exactly unchanged. Change only target_case to one of '
                      'eligible_target_cases. All other measured checks already pass.' if targets else
                      ('Every shown case passed, so the edit has no measured effect. ' if ineffective else
                       ('Other violations remain: ' + ', '.join(v for v in verdict['violations']
                        if v != 'named_case') + '. Changing the target cannot fix them. '
                        if any(v != 'named_case' for v in verdict['violations']) else
                        'No menu case returns a valid wrong integer. ')) +
                      'Retargeting cannot help. Propose a different edit against the original program.'))
        verdict['feedback'] = dict(message=json.dumps(feedback, ensure_ascii=False))
    return verdict


def repair_metrics(attempt_groups):
    counts = dict(retarget_opportunities=0, repairs_on_opportunities=0, preserved_edits=0,
                  nominated_observed_wrong_cases=0, accepted_preserved_retargets=0)
    for attempts in attempt_groups:
        parents = {v['attempt']: v for v in attempts}
        counts['retarget_opportunities'] += sum(v.get('parent') is None and bool(v.get('verification', {}).get('retarget_eligible_cases'))
                                              for v in attempts)
        for child in attempts:
            parent = parents.get(child.get('parent'), {})
            targets = parent.get('verification', {}).get('retarget_eligible_cases', [])
            if not targets:
                continue
            counts['repairs_on_opportunities'] += 1
            previous, edit = parent['edit'], child.get('edit', {})
            preserved = all(edit.get(k) == previous[k] for k in ('old', 'new'))
            selected = edit.get('target_case') in targets
            counts['preserved_edits'] += preserved
            counts['nominated_observed_wrong_cases'] += selected
            counts['accepted_preserved_retargets'] += (preserved and selected
                and child.get('verification', {}).get('status') == 'passed')
    return counts


def duplicate_submissions(attempt_groups):
    """Repeated normalized program AND nomination within each independent chain."""
    duplicates = 0
    for attempts in attempt_groups:
        seen = set()
        for attempt in attempts:
            if 'candidate' not in attempt:
                continue
            target = attempt.get('metadata', {}).get('case_declaration', attempt.get('edit', {}).get('target_case'))
            identity = (attempt['position'], attempt['candidate']['program_sha256'], str(target))
            duplicates += identity in seen
            seen.add(identity)
    return duplicates
