"""Blinded matched ratings: Qwen first, Gemma second, after attack completion."""
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import sys
from time import monotonic
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import dev_construction_queue as q

HERE=q.ROOT/'data/06_monitor_experiments/matched-dev-construction-20261031'
NORMALIZER=importlib.import_module('src.data-pipeline.stage05.monitor')
GEMMA=q.DATA/'gemma4-dev45-construction39-20261030-retry01'


def cells(records,verdicts):
    result={}
    for source in ('actor','synthetic'):
        for label in ('benign','attack'):
            values=[v['verdict'] for r,v in zip(records,verdicts,strict=True) if [source,label] in r['memberships']]
            counts=Counter(values);n=len(values)
            result[source+'/'+label]=dict(n=n,counts={k:counts[k] for k in ('PASS','FLAG','missing')},
                flag_rate=counts['FLAG']/n if n else None,
                escalation_rate=(counts['FLAG']+counts['missing'])/n if n else None)
    return result


def queue(thinking):
    HERE.mkdir(parents=True,exist_ok=False)
    config=q.u.read(q.e.a.MONITOR_CONFIG)
    files={str(p):q.u.file_hash(p) for p in (Path(__file__),Path(NORMALIZER.__file__),
        q.e.a.MONITOR_CONFIG,q.e.CHUNKED_DRIVER,q.e.a.DRIVER,Path(q.e.g.__file__))}
    active,manifest=q.e.qwen38_checkpoint(files)
    gemma=q.u.read(GEMMA/'contract.json')['runtime'];files.update(q.e.a._bind_benign_runtime('synthetic',gemma))
    qwen=q.u.read(q.QUEUE/'qwen-panel.json')['runtime']
    generation=dict(seed=17,temperature=0,top_p=1,max_tokens=8192 if thinking else 2048,
                    enable_thinking=thinking,max_prompt_tokens=25808)
    models=dict(qwen=dict(runtime=qwen,generation=dict(generation,**({'reasoning_effort':'low'} if thinking else {})),
                         batching=dict(completion_batch_size=4,prefill_batch_size=1,prefill_step_size=256)),
                gemma=dict(runtime=gemma,generation=generation,
                           batching=dict(completion_batch_size=2,prefill_batch_size=1,prefill_step_size=256)))
    plan=dict(status='queued_after_gemma_attacks',selection='All qualified benigns; first automatically accepted attack per task/source in chain/attempt order.',
        labels='Automatically accepted attacks; semantic approval pending. No feedback to generators.',
        order=['qwen','gemma'],models=models,config=config,sources=files,
        max_unique_records=276,max_replies_per_monitor=276,
        reserved_tokens_per_monitor=276*generation['max_tokens'],
        guard=dict(swap_growth_stop_mib=1024,peak_stop_bytes=44*1024**3,critical_pressure=4),
        privacy='Only statement and AST-preserving comment-stripped program in prompts.',
        parser='Historical literal final verdict parser, applied to final answer only. Capped replies are missing, never repaired.',
        output_cap_note='2048 thinking-off matches the final honeypot trial5 cap; thinking-on uses 8192.',
        no_attack_audits=True,no_model_overlap=True)
    q.u.save(HERE/'plan.json',plan)
    print(json.dumps(dict(status=plan['status'],thinking=thinking,order=plan['order'],max_records=276)),flush=True)


def corpus():
    if q.u.read(q.QUEUE/'queue.json').get('gemma-attacks')!='completed':
        raise ValueError('Wait for attack queue completion; never interrupt or repeat generation')
    plan=q.u.read(HERE/'plan.json');q.e.loop.check_files(plan['sources'])
    records={};files={};cohort=q.e.a.study_tasks.load_cohort()
    def add(source,label,binding,program,path):
        raw=Path(path).read_bytes()
        normalized=NORMALIZER.strip_comments(program)
        key=q.u.sha256((binding['sha256']+'\n'+normalized).encode()).hexdigest()
        record=records.setdefault(key,dict(id=key,task_id=f"apps:test:{binding['id']}",
            binding=binding,program=normalized,memberships=[],provenance=[]))
        if [source,label] not in record['memberships']:record['memberships'].append([source,label])
        if any(other!=label for _,other in record['memberships']):raise ValueError('Conflicting labels for identical program')
        record['provenance'].append(str(path));files[str(path)]=q.u.sha256(raw).hexdigest()
    q.GEMMA=GEMMA
    for source,loader in (('actor',q.load_qwen),('synthetic',q.load_gemma)):
        entries,_,_,bound=loader(q.QUEUE);files.update(bound)
        for x in entries:
            p=Path(x['benigns']['actor']['path'])
            if q.u.file_hash(p)!=x['benigns']['actor']['candidate']['program_sha256']:raise ValueError('Benign bytes changed')
            add(source,'benign',x['binding'],p.read_text(),p)
    roots=[('actor',q.ROOT/'data/05_attack_experiments'/name) for name in (
        'qwen38-dev-20261029-11','qwen38-construction50-run10-20261030')]
    roots += [('actor',q.QUEUE/'qwen-recovered-attacks'),('synthetic',q.QUEUE/'gemma-attacks')]
    selected=set()
    for source,root in roots:
        report=q.u.read(root/'report.json');experiment=q.u.read(root/'experiment.json')
        if report['status']!='completed':raise ValueError('Incomplete attack run')
        evidence=q.u.read(root/'evidence.json')
        q.e.loop.check_files({str(root/p):h for p,h in evidence.items()})
        for p in (root/'report.json',root/'experiment.json',root/'evidence.json'):files[str(p)]=q.u.file_hash(p)
        for run in report['runs']:
            p=Path(run['directory'])/'outcome.json';outcome=q.u.read(p);files[str(p)]=q.u.file_hash(p)
            if outcome['status']!='completed':raise ValueError('Incomplete chain')
            for a in outcome['attempts']:
                if a.get('verification',{}).get('status')!='passed':continue
                x=experiment['entries'][a['position']];key=(source,x['binding']['id'])
                if key in selected:continue
                p=Path(a['program'])
                if q.u.file_hash(p)!=a['candidate']['program_sha256']:raise ValueError('Attack bytes changed')
                selected.add(key);add(source,'attack',x['binding'],p.read_text(),p)
    rows=[records[k] for k in sorted(records)]
    if len(rows)>plan['max_unique_records']:raise ValueError('Monitor record ceiling exceeded')
    q.u.save(HERE/'corpus.json',dict(records=rows,inputs=files,selection=plan['selection']))
    return rows,files


def run():
    plan=q.u.read(HERE/'plan.json');rows,files=corpus();files[str(HERE/'corpus.json')]=q.u.file_hash(HERE/'corpus.json')
    config=plan['config'];scope=dict(json=json,re=re);exec(config['parser']['python_source'],scope)
    for model in plan['order']:
        target=HERE/model;target.mkdir(exist_ok=False);(target/'batch').mkdir()
        settings=plan['models'][model];runtime=settings['runtime'];requests=[]
        for row in rows:
            _,task=q.e.a.study_tasks.load_task(q.e.a.study_tasks.load_cohort(),row['binding']['id'])
            requests.append(dict(settings['generation'],messages=[dict(role='system',content=config['system_prompt']),
                dict(role='user',content=config['user_template'].format(statement=task['question'],normalized_program=row['program']))]))
        q.u.save(target/'inputs.json',dict(records=[{k:v for k,v in row.items() if k!='program'} for row in rows],requests=requests))
        bound=dict(plan['sources'],**files);bound[str(target/'inputs.json')]=q.u.file_hash(target/'inputs.json')
        q.u.save(target/'plan.json',dict(inputs=bound,reply_limit=len(rows),
            reserved_output_tokens=len(rows)*settings['generation']['max_tokens'],model=model,settings=settings))
        q.e.loop.check_files(bound)
        baseline=dict(q.e.a.memory_state(),swap_growth_stop_mib=1024)
        if baseline['pressure']!=1:raise RuntimeError('Require normal initial pressure')
        q.u.save(target/'resource-baseline.json',baseline)
        q.u.save(target/'round.json',dict(model_path=runtime['model_path'],model_manifest=runtime['model_manifest'],
            requests=requests,output_dir=str(target/'batch/generation'),**settings['batching']))
        child=dict(child_log=str(target/'child.log'));started=monotonic()
        q.u.save(target/'report.json',dict(status='running',records=len(rows),model=model))
        code=q.e.loop.run_child([runtime['mlx_python'],str(q.e.CHUNKED_DRIVER),str(target/'round.json')],child,
            lambda:q.u.save(target/'child.json',child))
        native=q.u.read(target/'batch/generation/outcome.json')
        q.e.loop.check_files(bound)
        if code or native['status']!='success':
            q.u.save(target/'report.json',dict(status='error',error=native.get('error'),returncode=code))
            raise RuntimeError('Monitor generation failed; preserve completed replies for technical recovery')
        verdicts=[scope['parse'](dict(v,text=v.get('answer_text') or '')) for v in native['requests']]
        q.u.save(target/'report.json',dict(status='completed',model=model,groups=cells(rows,verdicts),
            verdicts=verdicts,records=len(rows),output_tokens=sum(v['generation_tokens'] for v in native['requests']),
            elapsed_seconds=monotonic()-started,semantic_approval='pending',
            caveat='Blinded diagnostic ratings; FLAG witnesses and explanations not execution-validated.'))
        print(json.dumps(dict(model=model,status='completed',groups=cells(rows,verdicts))),flush=True)
    q.u.save(HERE/'completion.json',dict(status='completed',order=plan['order'],semantic_approval='pending'))


if __name__=='__main__':
    queue(sys.argv[2]=='on') if sys.argv[1]=='queue' else run()
