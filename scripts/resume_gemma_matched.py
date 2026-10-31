"""User-approved fresh resource baseline; preserve Gemma requests and search limits."""
from copy import deepcopy
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import dev_construction_queue as q

OLD=q.GEMMA
NEW=OLD.with_name(OLD.name+'-retry01')


def validate_zero_output(outcome):
    if outcome['status']!='error' or len(outcome['rounds'])!=1 or outcome['observed_tokens']!=0:
        raise ValueError('This recovery requires zero-output failure in the initial round')
    rows=q.u.read(OLD/'acquisition/actor/generation-0000/outcome.json')['requests']
    if len(rows)!=84 or any(x.get('status') in ('stop','length') or x.get('generation_tokens',0) for x in rows):
        raise ValueError('Complete replies require explicit preservation rather than this zero-output retry')
    return sum(bool(x.get('submitted')) for x in rows)


def run():
    q.QWEN=q.QUEUE/'qwen-recovered-attacks'
    if q.u.read(q.QWEN/'report.json')['status']!='completed':raise ValueError('Do not rerun Qwen')
    original=q.u.read(OLD/'contract.json');contract=deepcopy(original)
    overhead=validate_zero_output(q.u.read(OLD/'acquisition/actor/outcome.json'))
    q.e.loop.check_files(contract['sources']);q.e.loop.check_files(contract['inputs'])
    paths=[OLD/p for p in ('contract.json','initial-prompts.json','acquisition/report.json',
        'acquisition/resource-baseline.json','acquisition/actor/outcome.json',
        'acquisition/actor/round-0000.json','acquisition/actor/generation-0000/outcome.json',
        'acquisition/actor/generation-0000-part-0000/outcome.json')]
    contract['inputs'].update({str(p):q.u.file_hash(p) for p in paths})
    contract['sources'][str(Path(__file__).resolve())]=q.u.file_hash(__file__)
    contract['recovery']=dict(source=str(OLD),authorization='User approved a fresh memory baseline after the resource stop.',
        original_baseline=q.u.read(OLD/'acquisition/resource-baseline.json'),technical_submitted_requests=overhead,
        technical_output_tokens=0,complete_replies=0,search_allowances_unchanged=True,
        policy='Capture one fresh baseline for the retry; retain 1024 MiB swap-growth, 44 GiB MLX and pressure-4 limits.')
    assert (contract['reply_limit'],contract['output_token_limit'])==(252,860160)
    def prepared(root,group):
        worklist=[];messages=[];cohort=q.e.a.study_tasks.load_cohort()
        for x in contract['entries']:
            binding,task=q.e.a.study_tasks.load_task(cohort,x['binding']['id'])
            if binding!=x['binding']:raise ValueError('Task binding changed')
            worklist.append((binding,dict(question=task['question'],cases=[dict(c,visible=True)
                for k in ('examples','menu') for c in x['partition'][k]])))
            messages.append(q.d.b.shown_messages(task,x['partition']))
        prompts=[dict(task_id=x['binding']['id'],messages=m) for x,m in zip(contract['entries'],messages)]
        if prompts!=q.u.read(OLD/'initial-prompts.json'):raise ValueError('Retry prompts differ')
        q.u.save(root/'contract.json',contract);q.u.save(root/'initial-prompts.json',prompts)
        return contract,worklist,messages
    NEW.mkdir(exist_ok=False);q.d.prepare=prepared
    ledger=q.u.read(q.QUEUE/'queue.json');ledger.update(status='running',gemma_benign_root=str(NEW),
        gemma_recovery='User approved a fresh baseline; zero completed replies or tokens; original evidence preserved.')
    q.u.save(q.QUEUE/'queue.json',ledger)
    result=q.d.run(NEW)
    if result['status']!='completed':raise RuntimeError(result.get('error','Gemma benign retry failed'))
    ledger=q.u.read(q.QUEUE/'queue.json');ledger['gemma-benigns']='completed';q.u.save(q.QUEUE/'queue.json',ledger)
    print(json.dumps(dict(phase='gemma-benigns',status='completed',qualified=len(result['audit_verified']))),flush=True)
    q.GEMMA=NEW
    q.phase('gemma-attacks')


if __name__=='__main__':run()
