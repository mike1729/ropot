"""Apply the user's no-repeat correction; reuse 61 completed Qwen task searches."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import dev_construction_queue as q

TARGET=q.QUEUE/'qwen-recovered-attacks'


def load_recovered(root):
    entries,worklist,runtime,files=q.load_qwen(root)
    recovery=q.DATA/'qwen38-dev12-recovery-cases-20261030/summary.json'
    construction=q.DATA/'qwen38-construction13-recovered-v2-20261030/summary.json'
    wanted=set(q.u.read(recovery)['recoverable_tasks'])|{291}|set(q.u.read(construction)['recovered'])
    prior=set()
    paths=[recovery,construction,Path(__file__),q.QUEUE/'continuation.json']
    for name in ('qwen38-dev-20261029-11','qwen38-construction50-run10-20261030'):
        r=q.ROOT/'data/05_attack_experiments'/name
        if q.u.read(r/'report.json')['status']!='completed':raise ValueError('Historical Qwen attacks incomplete')
        c=q.u.read(r/'experiment.json')
        ids={x['binding']['id'] for x in c['entries']}
        if prior & ids:raise ValueError('Duplicate historical task')
        prior.update(ids);paths.extend([r/'report.json',r/'experiment.json'])
    ids={x['binding']['id'] for x in entries}
    if len(wanted)!=23 or len(prior)!=61 or wanted & prior or wanted|prior!=ids:
        raise ValueError('Require an exact disjoint 61 existing + 23 recovered split')
    selected=[(x,t) for x,t in zip(entries,worklist) if x['binding']['id'] in wanted]
    files.update({str(p):q.u.file_hash(p) for p in paths})
    return [x for x,_ in selected],[t for _,t in selected],runtime,files


def prepare():
    old=q.u.read(q.QWEN/'report.json')
    if old['status']!='cancelled':raise ValueError('Duplicate run must be stopped first')
    p=q.QWEN/'actor-chain-0/generation-0000-part-0000/outcome.json'
    native=q.u.read(p)
    state=dict(reason='User instructed: do not repeat existing Qwen attacks.',
        reused_runs=['qwen38-dev-20261029-11','qwen38-construction50-run10-20261030'],
        reused_tasks=61,new_qwen_tasks=23,new_qwen_max_turns=230,new_qwen_reserved_tokens=141312,
        gemma_tasks=84,interrupted_duplicate_tokens=native['native']['generation_tokens'],
        interrupted_duplicate_completed_replies=sum(x['status'] in ('stop','length') for x in native['requests']),
        interrupted_evidence=str(p),interrupted_evidence_sha256=q.u.file_hash(p),
        driver_sha256=q.u.file_hash(__file__),semantic_review='pending')
    q.u.save(q.QUEUE/'continuation.json',state)
    entries,_,_,_=load_recovered(q.QUEUE)
    state['new_qwen_task_ids']=[x['binding']['id'] for x in entries]
    q.u.save(q.QUEUE/'continuation.json',state)
    ledger=q.u.read(q.QUEUE/'queue.json')
    ledger.update(qwen_attacks_policy='Reuse completed 61; generate only recovered23',
                  interrupted_duplicate_run='cancelled',qwen_attack_max_turns=230,qwen_attack_reserved_tokens=141312)
    q.u.save(q.QUEUE/'queue.json',ledger)
    print(json.dumps(state),flush=True)


def run():
    result=q.e.run(TARGET,q.u.read(q.RECIPE),q.QUEUE,panel_loader=load_recovered)
    if result['status']!='completed':raise RuntimeError(result.get('error','Recovered Qwen attacks failed'))
    state=q.u.read(q.QUEUE/'queue.json');state['qwen-attacks']='completed_using_61_existing_plus_23_recovered'
    q.u.save(q.QUEUE/'queue.json',state)
    q.QWEN=TARGET
    for phase in ('gemma-benigns','gemma-attacks'):q.phase(phase)


if __name__=='__main__':
    (prepare if sys.argv[1]=='prepare' else run)()
