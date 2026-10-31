"""Fresh Qwen-only benigns for the explicitly expanded 50-task development panel."""
import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qwen_experiment as e

b = importlib.import_module('src.data-pipeline.stage03.benign_acquisition')
g, loop, a = e.g, e.loop, e.a


def development_partition(task, history):
    """Retain ineligible tasks for benign generation without disclosing audit inputs."""
    try:
        return b.partition(task, history), None
    except ValueError as error:
        examples = [c for c in task['cases'] if c['visible']]
        groups = {}
        for c in sorted(task['cases'], key=lambda c: c['index']):
            if not c['visible']:
                groups.setdefault(' '.join(c['input'].split()), []).append(c)
        groups = list(groups.values())
        random.Random(17 + task['id']).shuffle(groups)
        known, disclosed = set(history['known']), set(history['disclosed'])
        audits = [v[0] for v in groups if {c['index'] for c in v} <= known
                  and not {c['index'] for c in v} & disclosed][:4]
        withheld = {c['index'] for c in audits}
        menu = [v[0] for v in groups if v[0]['index'] not in withheld][:8]
        removed = [c['index'] for c in menu if len(c['input'].encode()) > 4096]
        return dict(examples=examples, menu=[c for c in menu if c['index'] not in removed], audit=audits,
                    removed_menu=removed, seed=17 + task['id'], permutation=[v[0]['index'] for v in groups]), str(error)


def prepare(root, cohort_group='development'):
    cohort = a.study_tasks.load_cohort()
    if cohort_group not in ('development', 'construction'):
        raise ValueError('Require construction or development ownership')
    order = cohort['groups'][cohort_group]['initial'][:50]
    if len(order) != 50 or len(set(order)) != 50:
        raise ValueError('Require exactly 50 pinned initial tasks')
    exposure_path = root / 'exposure.json'
    exposure = loop.read(exposure_path)
    for files in exposure['files'].values():
        loop.check_files(files)
    old = loop.read(e.DEFAULT_BENIGNS / 'contract.json')
    previous = {v['binding']['id']: v for v in old['entries']} if cohort_group == 'development' else {}
    _, _, runtime, files = e.load_panel(e.DEFAULT_BENIGNS)
    entries, worklist = [], []
    for identifier in order:
        binding, task = a.study_tasks.load_task(cohort, identifier)
        history = exposure['tasks'][str(identifier)]
        if history['task_sha256'] != binding['sha256']:
            raise ValueError('Exposure/task digest mismatch')
        cases, reason = development_partition(task, history)
        if identifier in previous:
            if previous[identifier]['binding'] != binding:
                raise ValueError('Historical task identity changed')
            cases, reason = previous[identifier]['partition'], None
            if {c['index'] for c in cases['audit']} & set(history['disclosed']):
                reason = 'Preserved historical audit case now appears in exposure history'
        shown = cases['examples'] + cases['menu']
        if not shown:
            raise ValueError(f'No public generation cases for task {identifier}')
        entries.append(dict(binding=binding, partition=cases, attack_case_eligible=reason is None,
                            attack_ineligibility_reason=reason))
        worklist.append((binding, dict(question=task['question'], cases=[dict(c, visible=True) for c in shown])))
    recipe = dict(b.RECIPE, enable_thinking=False,
                  base_seed=3_000_041 if cohort_group == 'construction' else 2_000_041, max_prompt_tokens=25808,
                  max_tokens=2048, repair_max_tokens=4096)
    messages = [b.shown_messages(task, entry['partition']) for (_, task), entry in zip(worklist, entries)]
    sources = {str(p.resolve()): g.file_hash(p) for p in (*b.source_files(), Path(__file__), a.DRIVER,
               Path(a.__file__), Path(e.__file__), root / 'prepare_exposure.py',
               Path(__file__).with_name('qwen_benign_retry_worker.py'))}
    files.update({str(exposure_path): g.file_hash(exposure_path),
                  str(e.DEFAULT_BENIGNS / 'contract.json'): g.file_hash(e.DEFAULT_BENIGNS / 'contract.json')})
    contract = dict(kind=f'qwen38-{cohort_group}-benigns', cohort_sha256=a.study_tasks.COHORT_SHA256,
        cohort_group=cohort_group, task_order=order, development_order=order if cohort_group == 'development' else [],
        allocation=('First 50 construction initial tasks, in saved order; no reserves or ownership changes.'
            if cohort_group == 'construction' else 'All 50 development initial tasks authorized for exploratory use; '
            'supersedes the previous 20 scaffold / 30 reserved confirmation split. Frozen cohort is unchanged.'),
        entries=entries, runtime=runtime, recipe=recipe,
        batching=dict(completion_batch_size=4, prefill_batch_size=1, prefill_step_size=256),
        reply_limit=150, output_token_limit=50 * (recipe['max_tokens'] + 2 * recipe['repair_max_tokens']),
        shown_case_limit=3 * sum(len(v['partition']['examples']) + len(v['partition']['menu']) for v in entries),
        audit_case_limit=sum(len(v['partition']['audit']) for v in entries),
        swap_growth_stop_mib=1024, worker_request_limit=4, sources=sources, inputs=files,
        limitations='Exploratory Qwen3.8 acquisition; no formal runtime requalification or synthetic source.')
    g.save(root / 'contract.json', contract)
    g.save(root / 'initial-prompts.json', [dict(task_id=v['binding']['id'], messages=m)
                                          for v, m in zip(entries, messages)])
    return contract, worklist, messages


def run(root, retry_from=None, cohort_group='development'):
    root = Path(root).resolve()
    if (root / 'contract.json').exists() or (root / 'acquisition').exists():
        raise FileExistsError('Refuse duplicate benign acquisition or implicit retry')
    prior = e.ROOT / 'data/05_attack_experiments/qwen38-dev-20261029-09'
    if loop.read(prior / 'report.json')['status'] != 'completed':
        raise ValueError('Run 09 must finish before benign generation')
    for outcome in prior.glob('actor-chain-*/outcome.json'):
        if any(r['child']['returncode'] != 0 for r in loop.read(outcome)['rounds']):
            raise ValueError('Run 09 worker exit is not confirmed')
    contract, worklist, messages = prepare(root, cohort_group)
    driver = Path(__file__).with_name('qwen_benign_retry_worker.py')
    if retry_from is not None:
        retry_from = Path(retry_from).resolve()
        previous = loop.read(retry_from / 'acquisition/report.json')
        if previous['status'] != 'error':
            raise ValueError('Retry requires a terminated failed acquisition')
        # Retain the historical initial allowance when replaying its exact requests.
        # Completed rows, including length stops, remain evidence for normal checking.
        previous_contract = loop.read(retry_from / 'contract.json')
        contract['recipe']['max_tokens'] = previous_contract['recipe']['max_tokens']
        spent_repairs = len(loop.read(retry_from / 'acquisition/actor/outcome.json')['rounds']) - 1
        contract['recipe']['max_repairs'] = previous_contract['recipe']['max_repairs'] - spent_repairs
        if contract['recipe']['max_repairs'] < 1:
            raise ValueError('No remaining repair round; do not reset the original allowance')
        contract['output_token_limit'] = 50 * (contract['recipe']['max_tokens'] +
            contract['recipe']['max_repairs'] * contract['recipe']['repair_max_tokens'])
        driver = Path(__file__).with_name('qwen_benign_retry_worker.py')
        contract['sources'][str(driver.resolve())] = g.file_hash(driver)
        contract['retry'] = dict(source=str(retry_from), worker_request_limit=4,
            policy='Reuse identical completed initial requests; restart workers after at most four new replies.',
            retained_initial_max_tokens=contract['recipe']['max_tokens'],
            prior_repair_rounds_consumed=spent_repairs,
            new_repair_max_tokens=contract['recipe']['repair_max_tokens'])
        g.save(root / 'contract.json', contract)
    directory = root / 'acquisition'
    directory.mkdir(exist_ok=False)
    started = monotonic()
    report = dict(status='running', task_count=len(worklist), shown_requests_committed=0, audit_requests_committed=0,
                  contract_sha256=g.file_hash(root / 'contract.json'), selected=[], audit_verified=[],
                  case_ineligible=[v['binding']['id'] for v in contract['entries'] if not v['attack_case_eligible']])
    def persist():
        report['elapsed_seconds'] = monotonic() - started
        loop.atomic_save(directory / 'report.json', report)
    persist()
    views = {v['binding']['sha256']: v['partition']['examples'] + v['partition']['menu'] for v in contract['entries']}
    def checker(candidates, log_dir, **kwargs):
        count = sum(len(views[h]) for _, h, _ in candidates)
        if report['shown_requests_committed'] + count > contract['shown_case_limit']:
            raise RuntimeError('Benign shown-case budget exceeded')
        report['shown_requests_committed'] += count
        persist()
        return b.verify_case_views(candidates, log_dir, views, **kwargs)
    try:
        loop.check_files(contract['sources'])
        loop.check_files(contract['inputs'])
        baseline = dict(a.memory_state(), swap_growth_stop_mib=1024)
        g.save(directory / 'resource-baseline.json', baseline)
        if baseline['pressure'] != 1:
            raise RuntimeError('Require normal initial memory pressure')
        runtime = contract['runtime']
        result = loop.run(worklist, directory / 'actor', source='actor',
            **{k: runtime[k] for k in ('model_path', 'model_manifest', 'mlx_python')},
            **contract['batching'], recipe=contract['recipe'], reply_allowance=contract['reply_limit'],
            output_token_allowance=contract['output_token_limit'], cohort_sha256=contract['cohort_sha256'],
            initial_messages=messages, candidate_checker=checker, generation_driver=driver)
        report.update(output_tokens=result['observed_tokens'], usage_complete=result['usage_complete'])
        if result['status'] != 'completed':
            raise RuntimeError(f"Benign generation failed: {result.get('error')}")
        selected = [result['attempts'][v['last_attempt']] for v in result['tasks'] if v['state'] == 'passed']
        report['selected'] = [dict(position=v['position'], task_id=worklist[v['position']][0]['id'],
                                  program=v['program'], candidate=v['candidate']) for v in selected]
        audits = {v['binding']['sha256']: v['partition']['audit'] for v in contract['entries']}
        auditable = [v for v in selected if audits[v['candidate']['task_sha256']]]
        candidates = [(worklist[v['position']][1], v['candidate']['task_sha256'], Path(v['program']).read_bytes())
                      for v in auditable]
        if any(sha256(program).hexdigest() != v['candidate']['program_sha256']
               for v, (_, _, program) in zip(auditable, candidates)):
            raise ValueError('Selected program changed before private audit')
        report.update(status='auditing', audit_requests_committed=sum(len(audits[h]) for _, h, _ in candidates))
        if report['audit_requests_committed'] > contract['audit_case_limit']:
            raise RuntimeError('Audit case budget exceeded')
        persist()
        verdicts = b.verify_case_views(candidates, directory / 'private-audit', audits) if candidates else []
        if any(v['status'] not in ('passed', 'failed') for v in verdicts):
            raise RuntimeError('Private audit infrastructure failure')
        g.save(directory / 'private-audit-results.json', verdicts)
        report['audit_verified'] = [dict(position=v['position'], task_id=worklist[v['position']][0]['id'],
            program=v['program'], verdict={k: verdict[k] for k in ('status', 'key', 'log', 'sample_id')})
            for v, verdict in zip(auditable, verdicts) if verdict['status'] == 'passed'
            and contract['entries'][v['position']]['attack_case_eligible']]
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='error' if isinstance(error, Exception) else 'cancelled', error=repr(error))
        outcome = directory / 'actor/outcome.json'
        if outcome.exists():
            result = loop.read(outcome)
            report.update(output_tokens=result['observed_tokens'], usage_complete=result['usage_complete'])
        raise
    finally:
        persist()
        g.save(root / 'evidence.json', {str(p.relative_to(root)): g.file_hash(p)
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'evidence.json'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--retry-from', type=Path)
    parser.add_argument('--cohort-group', choices=['development', 'construction'], default='development')
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.retry_from, args.cohort_group), indent=2))
