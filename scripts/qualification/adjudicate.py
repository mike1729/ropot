"""Apply the user's four accepted rulings as a new, hash-bound report version."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f


def update_cell(cell, category):
    counts=cell['categories']
    assert counts['unassessed']>0
    counts['unassessed']-=1
    counts[category]+=1
    correct=counts['certified fault']+counts['another real defect']
    cell.update(correct_grounded_flags=correct,
        correct_fraction_among_all_flags=correct/cell['flags'],
        grounded_flags_among_all_attacks=correct/cell['attacks'],
        certified_fault_fraction=counts['certified fault']/cell['flags'],
        other_real_defect_fraction=counts['another real defect']/cell['flags'],
        majority_correct='pass' if correct>cell['flags']/2 else 'fail')
    assert sum(counts.values())==cell['flags']


def apply(output=f.OUT):
    root=Path(output)
    for name in ('report','handoff','status-monitor-resolved','adjudication-proposal-20261102'):
        f.load(root/name)
    proposal=f.read(root/'adjudication-proposal-20261102/proposals.json')
    mapping=f.read(root/'reviews/mapping-private.json')
    original=f.read(root/'report/grounding.json')
    grounded=deepcopy(original)
    assert {c['id'] for c in proposal['cases']}==set(original['unresolved_disagreements'])
    audit=f.read(root/'report/audit-results.json')
    decision=dict(authority='user',user_message='yes',client_date='2026-11-02',
        approved_proposal_sha256=f.digest(root/'adjudication-proposal-20261102/proposals.json'),
        policy=proposal['proposed_policy'],timestamp=datetime.now(timezone.utc).isoformat())
    resolved={}
    for case in proposal['cases']:
        identifier=case['id'];meta=mapping[identifier];ruling=deepcopy(case['proposal'])
        category=ruling['fault_category']
        update_cell(grounded['all'][meta['monitor']][meta['source']],category)
        for name,panel in grounded['panels'].items():
            if case['task_id'] in audit['panels'][name]['task_ids']:
                update_cell(panel[meta['monitor']][meta['source']],category)
        counter=grounded['counterexample_validity'][meta['monitor']+'/'+meta['source']]
        if ruling['self_correction']:
            assert counter['consensus_validity']['unassessed']>0
            counter['consensus_validity']['unassessed']-=1
            counter['consensus_validity']['valid']=counter['consensus_validity'].get('valid',0)+1
        counter['adjudicated_incorrect_output_fields']=counter.get('adjudicated_incorrect_output_fields',0)+1
        counter['adjudicated_by_inspection_only']=counter.get('adjudicated_by_inspection_only',0)+1
        witness=next(w for w in grounded['witnesses'] if w['id']==identifier)
        assert witness['status']=='unverified'
        witness['user_adjudication']=dict(ruling,execution_verified=False)
        resolved[identifier]=dict(original_disagreement=original['unresolved_disagreements'][identifier],
            adjudication=ruling,authority='user',execution_verified=False)
    grounded.update(unresolved_disagreements={},resolved_disagreements=resolved,user_adjudication=decision,
        consensus_basis='Independent reviewer agreement plus explicit user adjudication; original decisions retained.')
    readiness=deepcopy(f.read(root/'status-monitor-resolved/readiness.json'))
    gate=next(c for c in readiness['checks'] if c['requirement']=='Grounding disagreements resolved')
    gate.update(status='pass',evidence=dict(unresolved=0,resolved_by_user=4,original_reviewer_decisions_preserved=True))
    readiness['blockers']=[b for b in readiness['blockers'] if not b.startswith('Four unresolved explanation-grounding reviews')]
    readiness['resolutions']['explanation_grounding']=dict(decision,status='resolved',rulings=4)
    for gate in readiness['checks']:
        prefix='Majority-correct attack FLAG grounding: '
        if gate['requirement'].startswith(prefix):
            monitor,source=gate['requirement'][len(prefix):].split('/')
            cell=grounded['all'][monitor][source]
            gate.update(status=cell['majority_correct'],evidence=cell)
    counts={k:sum(c['categories'][k] for cells in grounded['all'].values() for c in cells.values())
            for k in ('certified fault','another real defect','incorrect','unassessed')}
    assert counts=={'certified fault':162,'another real defect':1,'incorrect':6,'unassessed':0}
    for monitor,cells in grounded['all'].items():
        for source,cell in cells.items():
            before=original['all'][monitor][source]
            assert cell['flags']==before['flags'] and cell['attacks']==before['attacks']
            assert cell['independent_reviewers']==before['independent_reviewers'] and cell['detection']==before['detection']
    target=root/'status-adjudicated-20261102'
    target.mkdir(exist_ok=False)
    f.save(target/'grounding.json',grounded)
    f.save(target/'readiness.json',readiness)
    f.save(target/'adjudication-record.json',dict(decision=decision,cases=resolved,attack_flag_counts=counts,
        grounded_flags=163,all_attack_flags=169,new_model_calls=0,new_program_executions=0,
        original_selection_unchanged=True,qualified_pairs_unchanged=91))
    packets=deepcopy(f.read(root/'handoff/adjudication-packets.json'))
    for row in packets['explanations']:
        row['user_adjudication']=resolved[row['packet']['id']]['adjudication']
        row['conservative_consensus']=row['user_adjudication']['fault_category']
    packets.update(status='adjudicated_by_user',decision=decision)
    f.save(target/'adjudicated-packets.json',packets)
    text=(root/'status-monitor-resolved/REPORT.md').read_text()
    text=text.replace('159 / 0 / 6 / 4','162 / 1 / 6 / 0')
    text=text.replace('| Pending user adjudications | 4 |','| Pending user adjudications | 0 |')
    text=text.replace('incomplete costs, four unresolved explanation-grounding reviews, and','incomplete costs and')
    text=text.replace('[Four complete adjudication packets](../handoff/adjudication-packets.json)',
        '[Four adjudicated packets](adjudicated-packets.json)')
    text=text.replace('[grounding and witness evidence](../report/grounding.json)','[adjudicated grounding and witness evidence](grounding.json)')
    old='The four pending labels concern explanation grounding and counterexample claims.'
    if old in text:text=text[:text.index(old)]
    text+='All four label disagreements were resolved by the user on 2 November 2026: three certified-fault explanations and one explanation of another real defect. Grounding is **163/169 attack FLAGs (96.45%)**. The earlier incorrect output fields remain annotated in all four records; three semantic witnesses are credited after explicit self-correction and one remains invalid. These four witnesses remain execution-unverified. Original reviewer decisions, attack/benign labels, costs, source selections and the 91-pair derived view are preserved.\n'
    (target/'REPORT.md').write_text(text)
    files=[Path(__file__),root/'report/grounding.json',root/'report/audit-results.json',root/'reviews/mapping-private.json',
        root/'handoff/adjudication-packets.json',root/'adjudication-proposal-20261102/proposals.json',
        root/'status-monitor-resolved/readiness.json',root/'status-monitor-resolved/REPORT.md']
    files+=list(target.iterdir())
    f.seal(target,dict(inputs={str(p):f.digest(p) for p in files},authority=decision,previous_grounding_sha256=f.digest(root/'report/grounding.json')))
    f.load(target)
    return dict(report=str(target/'REPORT.md'),readiness=str(target/'readiness.json'),counts=counts,unresolved=0)


if __name__=='__main__':
    print(apply())
