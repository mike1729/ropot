"""Activate the user-approved task-291 contextual-exposure recovery only."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dev_recovery_cases as util

REVIEW = ROOT / 'data/03_benign_experiments/qwen38-task291-exposure-review-20261030'
KIND = 'qwen38-development-task291-recovery'
verification = importlib.import_module('src.data-pipeline.stage03.verification')


def partition(item, review):
    """Keep the historical shown menu; choose audits by its frozen permutation."""
    if item['binding']['id'] != 291:
        raise ValueError('Contextual exception is authorized for task 291 only')
    task = util.read(ROOT / 'data/02_cohort' / item['binding']['path'])
    clean = set(review['frozen_candidate_indices'])
    by_index = {c['index']: c for c in task['cases']}
    old = deepcopy(item['partition'])
    indices = [i for i in old['permutation'] if i in clean][:4]
    if len(indices) != 4:
        raise ValueError('Require four reviewed private cases')
    old['audit'] = [by_index[i] for i in indices]
    audit_inputs = {' '.join(c['input'].split()) for c in old['audit']}
    shown_inputs = {' '.join(c['input'].split()) for c in old['examples'] + old['menu']}
    if (len(audit_inputs) != 4 or audit_inputs & shown_inputs or not 4 <= len(old['menu']) <= 8
            or any(s + '\n' not in review['candidates'] for s in audit_inputs)):
        raise ValueError('Private cases overlap disclosures or allocation is invalid')
    return old


def case_view(item):
    p = item['partition']
    return dict(cases=[dict(c, visible=True) for c in p['examples'] + p['menu']] +
                     [dict(c, visible=False) for c in p['audit']])


def validate(root, contract, report):
    """Validate the recovery evidence on every load, without execution/inference."""
    if contract['kind'] != KIND or len(contract['entries']) != 1:
        raise ValueError('Require the single-task recovery contract')
    files = contract['recovery']['files']
    for path, digest in files.items():
        if util.file_hash(path) != digest:
            raise ValueError(f'Recovery evidence changed: {path}')
    review = util.read(REVIEW / 'review.json')
    item = contract['entries'][0]
    historical = util.read(util.BENIGNS / 'contract.json')
    original = next(x for x in historical['entries'] if x['binding']['id'] == 291)
    if item['binding'] != original['binding'] or item['partition'] != partition(original, review):
        raise ValueError('Recovery allocation differs from the approved partition')
    if (len(report['selected']) != 1 or len(report['audit_verified']) != 1
            or report['selected'][0]['task_id'] != 291 or report['audit_verified'][0]['task_id'] != 291):
        raise ValueError('Recovery requires exactly one selected and audited task')
    program = Path(report['selected'][0]['program'])
    if util.file_hash(program) != review['retained_benign_sha256']:
        raise ValueError('Require the exact reviewed own-source benign')
    saved = report['audit_verified'][0]['verdict']
    view = case_view(item)
    replayed = verification.reuse(saved['log'], view, verification.digest(view), program.read_bytes(),
                                 sample_id=saved['sample_id'])
    if replayed['status'] != 'passed' or any(replayed[k] != saved[k] for k in ('status', 'key', 'log', 'sample_id')):
        raise ValueError('Native recovery verification does not bind this allocation')
    log = Path(saved['log'])
    if util.file_hash(log) != report['recovery_log_sha256']:
        raise ValueError('Recovery execution log changed')
    return {**files, str(log): report['recovery_log_sha256']}


def recover(output):
    output.mkdir(parents=True, exist_ok=False)
    history = util.read(util.BENIGNS / 'contract.json')
    previous = util.read(util.BENIGNS / 'acquisition/report.json')
    review = util.read(REVIEW / 'review.json')
    summary = util.read(REVIEW / 'summary.json')
    if (summary['status'] != 'reviewed_and_exhaustively_verified_policy_not_changed'
            or summary['review_sha256'] != util.file_hash(REVIEW / 'review.json')):
        raise ValueError('Require the completed contextual review')
    for path, digest in review['request_files'].items():
        if util.file_hash(path) != digest:
            raise ValueError('Reviewed request evidence changed; refresh review')
    for folder in ('data/03_benign_experiments', 'data/05_attack_experiments', 'data/03_throughput'):
        for pattern in ('round-*.json', 'inputs.json'):
            for path in (ROOT / folder).rglob(pattern):
                if str(path.resolve()) not in review['request_files'] and list(util.requests(util.read(path))):
                    raise ValueError(f'New prompt evidence requires refreshed review: {path}')
    item = deepcopy(next(x for x in history['entries'] if x['binding']['id'] == 291))
    item.update(partition=partition(item, review), attack_case_eligible=True, attack_ineligibility_reason=None)
    selected = deepcopy(next(x for x in previous['selected'] if x['task_id'] == 291))
    raw = Path(selected['program']).read_bytes()
    if util.file_hash(selected['program']) != review['retained_benign_sha256']:
        raise ValueError('Reviewed benign changed')
    program = output / 'acquisition/actor/program-0000.py'
    program.parent.mkdir(parents=True)
    program.write_bytes(raw)
    selected.update(position=0, program=str(program))
    paths = [util.BENIGNS / 'contract.json', util.BENIGNS / 'acquisition/report.json',
             REVIEW / 'review.json', REVIEW / 'summary.json', REVIEW / 'execution-results.json',
             REVIEW / 'exhaustive-cases.json', Path(review['retained_benign']),
             ROOT / 'data/02_cohort/tasks/test_291.json', Path(__file__)]
    files = {**review['request_files'], **{str(p): util.file_hash(p) for p in paths}}
    contract = dict(kind=KIND, cohort_sha256=history['cohort_sha256'], entries=[item],
        runtime=history['runtime'], task_order=[291],
        allocation='Separate task-291 recovery; historical menu and benign retained; four reviewed private cases.',
        recovery=dict(policy='User approved task-specific contextual exposure adjudication on 2026-10-30.',
                      files=files, original_benigns=str(util.BENIGNS), review=str(REVIEW),
                      audit_selection='First four reviewed clean indices in the historical permutation; no outcome selection.'))
    util.save(output / 'contract.json', contract)
    report = dict(status='verifying', task_count=1, contract_sha256=util.file_hash(output / 'contract.json'),
        selected=[selected], audit_verified=[], case_ineligible=[], output_tokens=0, usage_complete=True,
        generation_replies=0, historical_results_unchanged=True)
    util.save(output / 'acquisition/report.json', report)
    view = case_view(item)
    verdict = verification.verify_many([(view, verification.digest(view), raw)], output / 'acquisition/recovery-verification')[0]
    util.save(output / 'acquisition/verification-result.json', verdict)
    if verdict['status'] != 'passed':
        report['status'] = 'failed' if verdict['status'] == 'failed' else 'error'
        util.save(output / 'acquisition/report.json', report)
        raise RuntimeError('Recovery check did not pass; no reselection or repairs')
    report.update(status='completed', recovery_log_sha256=util.file_hash(verdict['log']),
                  audit_verified=[dict(position=0, task_id=291, program=str(program), verdict={
                      k: verdict[k] for k in ('status', 'key', 'log', 'sample_id')})])
    util.save(output / 'acquisition/report.json', report)
    validate(output, contract, report)
    util.save(output / 'summary.json', dict(status='recovered', task_id=291, menu_cases=8, private_audit_cases=4,
        retained_own_source_benign=True, new_model_replies=0, policy_exception_scope='task 291 only',
        historical_results_unchanged=True, attack_run_started=False,
        contract_sha256=report['contract_sha256'], report_sha256=util.file_hash(output / 'acquisition/report.json')))
    print(json.dumps(util.read(output / 'summary.json')))


if __name__ == '__main__':
    recover(Path(sys.argv[1]).resolve())
