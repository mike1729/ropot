"""Batch 11a confirmation only; publication belongs to 11b. See its plan."""
from collections import Counter
import json
from pathlib import Path
import sys

from experiments import batch10_scaffold as prior

b, g, base = prior.b, prior.g, prior.base
ROOT = prior.ROOT
EVIDENCE = ROOT/'data/03_generation_checks/batch11-confirmation'
EXPERIMENTS = prior.EXPERIMENTS
QWEN_RECIPE = dict(base.RECIPE)
MINISTRAL_RECIPE = dict(prior.RECIPE, max_repairs=1)
SEEDS = (17, 100017, 200017, 300017)


def source_hashes():
    return prior.source_hashes() | {str(Path(__file__).resolve()): g.file_hash(__file__)}


def worklists():
    cohort = b.study_tasks.load_cohort()
    actor, synthetic = [], []
    for phase in prior.PHASES:
        ids = cohort['construction_phases'][phase]['initial'][:32]
        if len(ids)!=32:
            raise ValueError('Require 32 saved candidates per phase')
        tasks = [b.study_tasks.load_task(cohort, i) for i in ids]
        actor.extend(tasks)
        synthetic.extend(tasks[16:32])
    return dict(actor=actor, synthetic=synthetic)


def exploration(root, tasks):
    confirmation = {binding['id'] for binding,_ in tasks['synthetic']}
    expected = {binding['id']: binding for binding,_ in tasks['actor'] if binding['id'] not in confirmation}
    passed, evidence = set(), []
    for directory in sorted(Path(root).glob('*-batch10-*')):
        inputs = json.loads((directory/'inputs.json').read_text())
        outcome = json.loads((directory/'outcome.json').read_text())
        if (outcome['status']!='completed' or inputs['source']!='synthetic'
                or inputs['fixtures'] or inputs['cohort_sha256']!=b.study_tasks.COHORT_SHA256):
            raise ValueError('Require completed study Batch 10 evidence')
        bindings = [item['binding'] for item in inputs['worklist']]
        if any(binding!=expected.get(binding['id']) for binding in bindings):
            raise ValueError('Batch 10 binding differs from construction block')
        passed.update(bindings[i]['id'] for i in prior.passing_positions(outcome))
        evidence.append(dict(experiment=directory.name, input_sha256=g.file_hash(directory/'inputs.json'),
                             outcome_sha256=g.file_hash(directory/'outcome.json')))
    if not evidence:
        raise ValueError('Require Batch 10 exploration evidence')
    return dict(passed=sorted(passed), experiments=evidence)


def measure(directory=EVIDENCE):
    from mlx_lm.utils import load_config, load_tokenizer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tasks = worklists()
    records = []
    for options, recipe, maximum in ((base.RUNS[0],QWEN_RECIPE,745),(base.RUNS[1],MINISTRAL_RECIPE,755)):
        config = load_config(options['model_path'])
        tokenizer = load_tokenizer(options['model_path'], dict(local_files_only=True,
            **({'fix_mistral_regex':True} if options['source']=='synthetic' else {})),
            eos_token_ids=config.get('eos_token_id'))
        requests = b.initial_requests(tasks[options['source']],recipe)
        counts = [g.render(tokenizer, request['messages'])['prompt_tokens'] for request in requests]
        context = config.get('text_config',config)['max_position_embeddings']
        records.append(dict(source=options['source'],recipe=recipe,initial_requests=requests,
            prompt_tokens=counts,development_maximum=maximum,within_development_maximum=max(counts)<=maximum,
            context=context,fits_context=max(counts)+recipe['max_tokens']<=context,
            model_manifest_sha256=g.file_hash(options['model_manifest'])))
    previous = json.loads((prior.EVIDENCE/'sizing.json').read_text())
    record = dict(source_sha256=source_hashes(), bindings={s:[binding for binding,_ in ts] for s,ts in tasks.items()},
        models=records, cycle_seeds=SEEDS, cohort_sha256=b.study_tasks.COHORT_SHA256,
        historical=previous['historical'], batch10_sizing_sha256=g.file_hash(prior.EVIDENCE/'sizing.json'),
        allowances=dict(actor=dict(replies=128,tokens=131072),synthetic=dict(replies=256,tokens=524288)))
    g.save(directory/'sizing.json',record)
    if not all(r['within_development_maximum'] and r['fits_context'] for r in records):
        raise ValueError('Sizing exceeds reviewed maximum; stop for review')
    return record


def report(tasks, records, context, complete):
    first = dict(actor={},synthetic={})
    states = {source:{binding['id']:'not_submitted' for binding,_ in ts} for source,ts in tasks.items()}
    costs = []
    for record in records:
        source, ids, outcome = record['source'], record['ids'], record['outcome']
        for task in outcome['tasks']:
            states[source][ids[task['position']]] = task['state']
        for attempt in outcome['attempts']:
            if attempt['state']=='passed' or attempt.get('verification',{}).get('status')=='passed':
                task_id = ids[attempt['position']]
                number = record['offset']+attempt['round']+1
                first[source][task_id] = min(first[source].get(task_id,9),number)
                states[source][task_id] = 'passed'
        metrics = prior.metrics(record['directory'],outcome,0)
        for metric in metrics:
            metric['attempt'] = record['offset']+metric['round']+1
            metric['unusable'] = dict(Counter(a['extraction'] for a in outcome['attempts']
                if a['round']==metric['round'] and a.get('extraction') not in (None,'success')))
        costs.append(dict(source=source,experiment=str(record['directory']),offset=record['offset'],
            counters={key:outcome.get(key) for key in ('committed_replies','committed_tokens','known_submissions',
                'known_submissions_lower_bound','observed_tokens','usage_complete')},timing=outcome.get('timing'),rounds=metrics))
    confirmation = {binding['id'] for binding,_ in tasks['synthetic']}
    blocks = dict(confirmation=confirmation,exploration={binding['id'] for binding,_ in tasks['actor']}-confirmation)
    qwen = {label:dict(denominator=len(ids),curve=[sum(i in ids and n<=k for i,n in first['actor'].items()) for k in (1,2)])
            for label,ids in blocks.items()}
    joint = dict(exploration=sorted(set(first['actor']) & set(context['passed'])))
    for budget in (6,8):
        joint[f'confirmation_{budget}'] = sorted(set(first['actor']) & {i for i,n in first['synthetic'].items() if n<=budget})
    rows = {s:[dict(binding=binding,block='confirmation' if binding['id'] in confirmation else 'exploration',
        first_pass=first[s].get(binding['id']),state=states[s][binding['id']]) for binding,_ in ts] for s,ts in tasks.items()}
    return dict(status='complete' if complete else 'incomplete/lower_bounds',qwen=qwen,
        ministral=dict(denominator=len(confirmation),curve=[sum(n<=k for n in first['synthetic'].values()) for k in range(1,9)],
                       confirmation_budget=6,marginal_attempts=[7,8]),
        joint={label:dict(denominator=32,passed=len(ids),ids=ids) for label,ids in joint.items()},
        tasks=rows,effort=costs,batch10_exploration=context,
        interpretation='Confirmation yield is six attempts; eight-attempt coverage is separate. No acquisition scale decision.')


def run(directory=EVIDENCE, root=EXPERIMENTS):
    directory, root = Path(directory), Path(root)
    if Path(sys.executable).absolute()!=ROOT/'.venv/bin/python':
        raise ValueError('Run the caller in repository .venv/bin/python')
    sizing = json.loads((directory/'sizing.json').read_text())
    if sizing['source_sha256']!=source_hashes() or list(sizing['cycle_seeds'])!=list(SEEDS):
        raise ValueError('Changed sizing sources or schedule')
    tasks = worklists()
    if sizing['bindings']!={s:[binding for binding,_ in ts] for s,ts in tasks.items()}:
        raise ValueError('Changed construction worklists')
    for model,options,recipe in zip(sizing['models'],base.RUNS,(QWEN_RECIPE,MINISTRAL_RECIPE),strict=True):
        if (model['source']!=options['source'] or model['recipe']!=recipe or not model['within_development_maximum']
                or not model['fits_context'] or model['model_manifest_sha256']!=g.file_hash(options['model_manifest'])):
            raise ValueError('Changed recipe, model manifest or unacceptable sizing')
    existing = prior.reader.summarize(root)['experiments']
    if any('-batch11a-' in e['experiment'] for e in existing):
        raise FileExistsError('Batch 11a already exists; no redispatch')
    if any(not e['terminal'] or e['fixtures'] is not False or e['cohort_sha256']!=b.study_tasks.COHORT_SHA256 for e in existing):
        raise ValueError('Require terminal study experiments on the pinned cohort')
    context = exploration(root,tasks)
    first = max((int(e['experiment'][:4]) for e in existing),default=0)+1
    records, current, descriptor, complete = [], None, None, False
    try:
        for options in base.RUNS:
            source = options['source']
            remaining = tasks[source]
            for cycle in range(1 if source=='actor' else 4):
                if not remaining:
                    break
                recipe = dict(QWEN_RECIPE) if source=='actor' else dict(MINISTRAL_RECIPE,
                    base_seed=SEEDS[cycle],temperature=0.2 if cycle==0 else 0.7)
                current = root/f'{first+len(records):04d}-batch11a-{source}{"" if source=="actor" else "-c"+str(cycle+1)}'
                descriptor = dict(directory=str(current),source=source,ids=[binding['id'] for binding,_ in remaining],offset=2*cycle)
                result = b.run(remaining,current,**options,mlx_python=base.MLX_PYTHON,recipe=recipe,
                    reply_allowance=2*len(remaining),output_token_allowance=2*len(remaining)*recipe['max_tokens'],
                    cohort_sha256=b.study_tasks.COHORT_SHA256)
                records.append(dict(descriptor,outcome=result))
                current = None
                if result['status']!='completed':
                    raise RuntimeError('Technical stop; no subsequent source or cycle')
                passed = prior.passing_positions(result)
                remaining = [task for i,task in enumerate(remaining) if i not in passed]
        complete = True
    finally:
        if current is not None and (current/'outcome.json').exists():
            records.append(dict(descriptor,outcome=json.loads((current/'outcome.json').read_text())))
        g.save(directory/'summary.json',prior.reader.summarize(root))
        g.save(directory/'results.json',report(tasks,records,context,complete))
