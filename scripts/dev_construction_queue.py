"""User-authorized Qwen attacks, then fresh Gemma benigns and own-source attacks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qwen_experiment as e, qwen_dev_benigns as d, dev_recovery_cases as u

ROOT = e.ROOT
DATA = ROOT/'data/03_benign_experiments'
QUEUE = ROOT/'data/05_attack_experiments/dev-construction-qwen-gemma-20261030'
QWEN = QUEUE/'qwen-attacks'
GEMMA = DATA/'gemma4-dev45-construction39-20261030'
RECIPE = ROOT/'data/05_attack_experiments/recipes/qwen38-construction50-run10.json'


def public(entries, programs):
    cohort = e.a.study_tasks.load_cohort(); output, worklist = [], []
    for x in entries:
        binding, task = e.a.study_tasks.load_task(cohort,x['binding']['id'])
        if binding != x['binding']: raise ValueError('Task binding changed')
        p = {k:x['partition'][k] for k in ('examples','menu')}
        path = Path(programs[binding['id']]); raw = path.read_bytes()
        output.append(dict(binding=binding,partition=p,benigns={'actor':dict(
            program=e.a.programs.strip_comments(raw.decode()),path=str(path),
            candidate=e.a.study_tasks.candidate_identity(binding['split'],binding['id'],binding['sha256'],'actor',raw))}))
        worklist.append((binding,dict(question=task['question'],cases=[dict(c,visible=True) for k in p.values() for c in k])))
    return output, worklist


def prepare():
    QUEUE.mkdir(exist_ok=False)
    full, programs, inputs = {}, {}, {}
    runtime = None
    for name in ('qwen38-dev50-20261029-retry03','qwen38-dev291-recovered-20261030',
                 'qwen38-construction50-20261030','qwen38-construction13-recovered-v2-20261030'):
        root = DATA/name
        entries, _, runtime, files = e.load_dev_panel(root); inputs.update(files)
        c = u.read(root/'contract.json'); by_id={x['binding']['id']:x for x in c['entries']}
        for x in entries:
            i=x['binding']['id']
            if i in full: raise ValueError('Duplicate qualified task')
            full[i]=deepcopy(by_id[i]); programs[i]=x['benigns']['actor']['path']
    # The ten older recoveries have native verification evidence rather than B1 reports.
    root=DATA/'qwen38-dev12-recovery-cases-20261030'
    manifest=u.read(root/'manifest.json'); completion=u.read(root/'completion.json')
    if completion['manifest_sha256']!=u.file_hash(root/'manifest.json') or completion['status']!='completed':
        raise ValueError('Unbound legacy recovery')
    original=u.read(DATA/'qwen38-dev50-20261029-retry03/contract.json')
    for row,saved in zip(manifest['entries'],u.read(root/'benign-results.json'),strict=True):
        p=root/row['path']; item=u.read(p); i=item['task_id']; v=saved['verdict']
        if u.file_hash(p)!=row['sha256'] or u.file_hash(item['benign_program'])!=item['benign_sha256']:
            raise ValueError('Legacy recovery bytes changed')
        view=dict(cases=item['cases']); verifier=d.b.verification if hasattr(d.b,'verification') else __import__('importlib').import_module('src.data-pipeline.stage03.verification')
        replay=verifier.reuse(v['log'],view,verifier.digest(view),Path(item['benign_program']).read_bytes(),sample_id=v['sample_id'])
        if replay['status']!='passed' or replay['key']!=v['key']: raise ValueError('Legacy recovery did not pass')
        x=deepcopy(next(x for x in original['entries'] if x['binding']['id']==i))
        x['partition'].update(menu=[c for c in item['cases'] if c['role']=='menu'],audit=[c for c in item['cases'] if c['role']=='audit'])
        x.update(attack_case_eligible=True,attack_ineligibility_reason=None)
        if i in full: raise ValueError('Duplicate recovered task')
        full[i]=x; programs[i]=item['benign_program']
        inputs.update({str(p):u.file_hash(p),v['log']:u.file_hash(v['log']),item['benign_program']:item['benign_sha256']})
    inputs.update({str(p):u.file_hash(p) for p in (root/'manifest.json',root/'completion.json',root/'benign-results.json',RECIPE,Path(__file__))})
    cohort=e.a.study_tasks.load_cohort()
    order=[i for group in ('development','construction') for i in cohort['groups'][group]['initial'][:50] if i in full]
    assert len(order)==len(full)==84
    assert sum(i in cohort['groups']['development']['initial'] for i in order)==45
    entries=[full[i] for i in order]
    shown,worklist=public(entries,programs)
    u.save(QUEUE/'qwen-panel.json',dict(entries=shown,runtime=runtime,inputs=inputs))
    old=u.read(e.DEFAULT_BENIGNS/'contract.json'); gemma=old['runtimes']['synthetic']
    gemma_inputs=e.a._bind_benign_runtime('synthetic',gemma)
    gemma_inputs.update(inputs)
    contract=dict(kind='gemma4-matched-benigns',cohort_sha256=e.a.study_tasks.COHORT_SHA256,
        task_order=order,entries=entries,runtime=dict(gemma,model_family='gemma4'),
        recipe=dict(d.b.RECIPE,enable_thinking=False,max_tokens=2048,repair_max_tokens=4096,
                    base_seed=5_000_041,max_prompt_tokens=25808),
        batching=dict(completion_batch_size=2,prefill_batch_size=1,prefill_step_size=256),
        worker_request_limit=4,reply_limit=252,output_token_limit=860160,
        shown_case_limit=3*sum(len(x['partition']['examples'])+len(x['partition']['menu']) for x in entries),
        audit_case_limit=336,inputs=gemma_inputs,
        sources={str(p):u.file_hash(p) for p in (*d.b.source_files(),Path(__file__),Path(e.__file__),Path(d.__file__),
                 ROOT/'scripts/qwen_benign_retry_worker.py',e.a.DRIVER,ROOT/'scripts/qwen_chunked_generation.py',
                 ROOT/'scripts/qwen_native_tools.py')},
        allocation='Exactly the 84 Qwen-qualified tasks; same case partitions. Fresh Gemma programs; no replacement after audit failure.')
    u.save(QUEUE/'gemma-contract.json',contract)
    u.save(QUEUE/'queue.json',dict(status='prepared',phases=['qwen-attacks','gemma-benigns','gemma-attacks'],
        task_count=84,development=45,construction=39,qwen_attack_max_turns=840,qwen_attack_reserved_tokens=516096,
        gemma_benign_max_replies=252,gemma_benign_reserved_tokens=860160,
        gemma_attack_max_turns=840,gemma_attack_reserved_tokens=516096,
        semantic_review='pending',one_model_job=True))
    return load_qwen(QUEUE)


def load_qwen(_):
    v=u.read(QUEUE/'qwen-panel.json');e.loop.check_files(v['inputs'])
    files=dict(v['inputs']);files[str(QUEUE/'qwen-panel.json')]=u.file_hash(QUEUE/'qwen-panel.json')
    cohort=e.a.study_tasks.load_cohort(); worklist=[]
    for x in v['entries']:
        binding,task=e.a.study_tasks.load_task(cohort,x['binding']['id'])
        if binding!=x['binding']:raise ValueError('Task binding mismatch')
        worklist.append((binding,dict(question=task['question'],cases=[dict(c,visible=True) for k in x['partition'].values() for c in k])))
    return v['entries'],worklist,v['runtime'],files


def gemma_benigns():
    if u.read(QWEN/'report.json')['status']!='completed':raise ValueError('Qwen attacks must finish first')
    contract=u.read(QUEUE/'gemma-contract.json')
    def prepared(root,group):
        e.loop.check_files(contract['sources']);e.loop.check_files(contract['inputs'])
        worklist=[];messages=[];cohort=e.a.study_tasks.load_cohort()
        for x in contract['entries']:
            binding,task=e.a.study_tasks.load_task(cohort,x['binding']['id'])
            worklist.append((binding,dict(question=task['question'],cases=[dict(c,visible=True) for k in ('examples','menu') for c in x['partition'][k]])))
            messages.append(d.b.shown_messages(task,x['partition']))
        u.save(root/'contract.json',contract)
        u.save(root/'initial-prompts.json',[dict(task_id=x['binding']['id'],messages=m) for x,m in zip(contract['entries'],messages)])
        return contract,worklist,messages
    GEMMA.mkdir(exist_ok=False);d.prepare=prepared
    return d.run(GEMMA)


def load_gemma(_):
    c=u.read(GEMMA/'contract.json');r=u.read(GEMMA/'acquisition/report.json')
    if r['status']!='completed' or r['contract_sha256']!=u.file_hash(GEMMA/'contract.json'):raise ValueError('Gemma acquisition incomplete')
    files=dict(c['inputs']);files.update(e.a._bind_benign_runtime('synthetic',c['runtime']))
    selected={x['task_id']:x for x in r['selected']};audited={x['task_id']:x for x in r['audit_verified']}
    entries=[];programs={};verifier=__import__('importlib').import_module('src.data-pipeline.stage03.verification')
    for x in c['entries']:
        i=x['binding']['id']
        if i not in audited:continue
        s=selected[i];a=audited[i];p=Path(s['program'])
        if not p.is_relative_to(GEMMA) or a['program']!=s['program']:raise ValueError('Program escaped acquisition')
        identity=e.a.study_tasks.candidate_identity(x['binding']['split'],i,x['binding']['sha256'],'actor',p.read_bytes())
        if identity!=s['candidate']:raise ValueError('Program identity changed')
        view=dict(cases=[dict(z,visible=True) for z in x['partition']['audit']]);v=a['verdict']
        replay=verifier.reuse(v['log'],view,verifier.digest(view),p.read_bytes(),sample_id=v['sample_id'])
        if replay['status']!='passed' or replay['key']['program_sha256']!=v['key']['program_sha256']:raise ValueError('Audit binding changed')
        entries.append(x);programs[i]=str(p);files.update({str(p):u.file_hash(p),v['log']:u.file_hash(v['log'])})
    files.update({str(p):u.file_hash(p) for p in (GEMMA/'contract.json',GEMMA/'acquisition/report.json',Path(__file__))})
    e.loop.check_files(files);shown,worklist=public(entries,programs)
    if not shown:raise ValueError('No qualifying Gemma benigns')
    return shown,worklist,c['runtime'],files


def phase(name):
    if name=='prepare':prepare();return
    config=u.read(RECIPE)
    if name=='qwen-attacks':
        result=e.run(QWEN,config,QUEUE,panel_loader=load_qwen)
    elif name=='gemma-benigns':result=gemma_benigns()
    elif name=='gemma-attacks':
        config['completion_batch_size']=4
        result=e.run(QUEUE/'gemma-attacks',config,GEMMA,panel_loader=load_gemma)
    else:raise ValueError(name)
    if result['status']!='completed':raise RuntimeError(f'{name} failed: {result.get("error")}')
    state=u.read(QUEUE/'queue.json');state[name]='completed';u.save(QUEUE/'queue.json',state)
    print(json.dumps(dict(phase=name,status='completed')),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='all':
        for name in ('qwen-attacks','gemma-benigns','gemma-attacks'):phase(name)
    else:phase(sys.argv[1])
