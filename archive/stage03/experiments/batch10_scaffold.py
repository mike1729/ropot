"""Batch 10: paired six-attempt policies. See docs/stage-03-batch-10-plan.md."""
from collections import Counter
import importlib
import json
from pathlib import Path
import subprocess

from experiments import batch09_development as base

b, g = base.b, base.g
reader = importlib.import_module('src.data-pipeline.stage03.experiment_summary')
ROOT = base.ROOT
EVIDENCE = ROOT / 'data/03_generation_checks/batch10-scaffold'
EXPERIMENTS = ROOT / 'data/03_benign_experiments'
OPTIONS = dict(base.RUNS[1])
RECIPE = dict(system='You write correct, efficient Python 3 programs.',
    initial='Re-read the exact operation and numerical constraints. Check your chosen algorithm against the statement examples. Return exactly one complete Python program in one Python code block. Include no explanations, alternatives or revisions after the program. End immediately after the closing fence.',
    repair='Re-read the exact operation and numerical constraints. Check your chosen algorithm against the statement examples and reported failure. Return exactly one complete Python program in one Python code block. Include no explanations, alternatives or revisions after the program. End immediately after the closing fence.',
    temperature=0.2, top_p=0.95, base_seed=17, max_tokens=2048, repair_max_tokens=2048,
    max_repairs=5, feedback_field_bytes=512, feedback_bytes=4096)
PHASES = ('shared_honest_pretraining', 'twin_continuation')
PIPELINE = [ROOT / 'src/data-pipeline' / name for name in
    ('stage03/benign_loop.py', 'stage03/local_generation.py', 'stage03/repair_feedback.py',
     'stage03/verification.py', 'stage03/inspect_runner.py', 'stage01/screen.py',
     'stage03/study_tasks.py', 'stage02/cohort.py', 'stage02/published_cohort.py')]


def source_hashes():
    hashes = {str(p): g.file_hash(p) for p in PIPELINE + [Path(__file__).resolve(), Path(base.__file__)]}
    for path in PIPELINE:
        pinned = subprocess.check_output(['git', 'show', f'd6c311e:{path.relative_to(ROOT)}'], cwd=ROOT)
        if g.sha256(pinned).hexdigest() != hashes[str(path)]:
            raise ValueError(f'Pipeline differs from d6c311e: {path}')
    return hashes


def worklist():
    cohort = b.study_tasks.load_cohort()
    ids = [i for phase in PHASES for i in cohort['construction_phases'][phase]['initial'][:16]]
    tasks = [b.study_tasks.load_task(cohort, i) for i in ids]
    if len(ids) != 32 or len(set(ids)) != 32 or Counter(x['difficulty'] for x, _ in tasks) != {'introductory': 8, 'interview': 24}:
        raise ValueError('Require the frozen 32-task exploration block')
    return tasks


def measure(directory=EVIDENCE):
    from mlx_lm.utils import load_config, load_tokenizer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    hashes, tasks = source_hashes(), worklist()
    config = load_config(OPTIONS['model_path'])
    tokenizer = load_tokenizer(OPTIONS['model_path'], dict(local_files_only=True, fix_mistral_regex=True),
                               eos_token_ids=config.get('eos_token_id'))
    requests = b.initial_requests(tasks, RECIPE)
    counts = [g.render(tokenizer, request['messages'])['prompt_tokens'] for request in requests]
    position = max(range(len(counts)), key=counts.__getitem__)
    recipe = dict(RECIPE, temperature=0.7, max_repairs=1, base_seed=100017)
    initial = b.initial_requests([tasks[position]], recipe)[0]
    ids = tokenizer.encode('print(0)\n' * 2048, add_special_tokens=False)[:2048]
    detail = ' '.join(str(i) for i in range(512))
    feedback = dict(status='failed', hidden='failed', visible=[dict(status='wrong_answer', returncode=0,
        **dict.fromkeys(('input', 'answer', 'stdout', 'stderr', 'diagnostic'), detail)) for _ in range(2)])
    parent = dict(extraction='success', verification=dict(feedback=feedback))
    repair = b.repair_request(initial, dict(text=g.decode_tokens(tokenizer, ids, set(tokenizer.eos_token_ids))),
                              parent, recipe, 100018)
    stress = g.render(tokenizer, repair['messages'])
    context = config.get('text_config', config)['max_position_embeddings']
    historical_path = ROOT.parent / 'honeypot/runs/overnight-mlx-1/ministral_8bit/results.json'
    historical = json.loads(historical_path.read_text())['results']
    cohort = b.study_tasks.load_cohort()
    confirmation = [i for phase in PHASES for i in cohort['construction_phases'][phase]['initial'][16:32]]
    relevant = set(confirmation) | {binding['id'] for binding, _ in tasks}
    overlap = {row['task_id']: dict(attempts=len(row['attempts']['benign']),
        first_pass=next((j+1 for j, draw in enumerate(row['attempts']['benign'])
                         if draw.get('tests', {}).get('all_tests_passed')), None))
        for row in historical if row['task_id'].startswith('apps:test:') and int(row['task_id'].split(':')[-1]) in relevant}
    record = dict(source_sha256=hashes, bindings=[binding for binding, _ in tasks], recipe=RECIPE,
        options={k: str(v) if isinstance(v, Path) else v for k, v in OPTIONS.items()},
        mlx_python=str(base.MLX_PYTHON), model_manifest_sha256=g.file_hash(OPTIONS['model_manifest']),
        initial_prompt_tokens=counts, initial_requests=requests, context=context,
        stress=dict(position=position, messages=repair['messages'], prompt_tokens=stress['prompt_tokens'],
                    input_plus_output=stress['prompt_tokens']+2048, rendered_feedback=parent['repair_feedback']),
        within_development_maximum=max(counts)<=755, fits_context=stress['prompt_tokens']+2048<=context,
        historical=dict(path=str(historical_path), sha256=g.file_hash(historical_path), overlap=overlap,
                        confirmation_ids=confirmation),
        assumptions='Stress example, not maximum. Arm A depth uses development evidence; round peaks are post-round observations, not live limits.')
    g.save(directory / 'sizing.json', record)
    if not record['within_development_maximum'] or not record['fits_context']:
        raise ValueError('Sizing exceeds reviewed operating point; stop for review')
    return record


def passing_positions(outcome, through=None):
    return {a['position'] for a in outcome['attempts'] if (through is None or a['round']<=through)
            and (a['state']=='passed' or a.get('verification', {}).get('status')=='passed')}


def metrics(directory, outcome, minimum_round):
    rounds = []
    for current in outcome['rounds']:
        if current['round'] < minimum_round:
            continue
        native_path = Path(current['generation_outcome'])
        native = json.loads(native_path.read_text()) if native_path.exists() else {}
        spec_path = Path(current['round_file'])
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        rows = native.get('requests', [])
        attempts = [a for a in outcome['attempts'] if a['attempt'] in current['attempt_ids']]
        rounds.append(dict(round=current['round'], experiment=str(directory), attempts=len(attempts),
            unusable=dict(Counter(a.get('extraction') for a in attempts if a.get('extraction') in ('length','format'))),
            committed_tokens=sum(r['max_tokens'] for r in spec['requests']) if spec else None,
            observed_tokens=current.get('observed_tokens'), usage_complete=current.get('usage_complete'),
            prompt_tokens=sum(r['prompt_tokens'] for r in rows) if rows and all('prompt_tokens' in r for r in rows) else None,
            timing=current.get('timing'), peak_memory_bytes=native.get('peak_memory_bytes')))
    return rounds


def compare(tasks, records, complete):
    first = dict(A={}, B={})
    for index, record in enumerate(records):
        arm, offset = ('A', 0) if index==0 else ('B', 2*index)
        for a in record['outcome']['attempts']:
            if a['state']!='passed' and a.get('verification', {}).get('status')!='passed':
                continue
            task_id = record['ids'][a['position']]
            attempt = offset + a['round'] + 1
            first[arm][task_id] = min(first[arm].get(task_id, 7), attempt)
            if index==0 and a['round']<=1:
                first['B'][task_id] = attempt
    curves = {arm: [sum(j<=k for j in passes.values()) for k in range(1,7)] for arm, passes in first.items()}
    only_a, only_b = sorted(first['A'].keys()-first['B'].keys()), sorted(first['B'].keys()-first['A'].keys())
    return dict(status='complete' if complete else 'incomplete/lower_bounds', denominator=len(tasks), curves=curves,
        first_pass=first, only_A=only_a, only_B=only_b,
        chosen_arm=('A' if len(first['A'])-len(first['B'])>=3 else 'B') if complete else None,
        shared_attempts='Arm A rounds 0–1 count toward both policies; physical effort counted only once.',
        physical_effort=[dict(experiment=str(r['directory']), counters={k:r['outcome'].get(k) for k in
            ('committed_replies','committed_tokens','known_submissions','observed_tokens','usage_complete')}) for r in records],
        rounds=[dict(policy='A' if i==0 else 'B', offset=0 if i==0 else 2*i,
            metrics=metrics(r['directory'],r['outcome'],0)) for i,r in enumerate(records)],
        attempts_3_to_6=dict(A=metrics(records[0]['directory'],records[0]['outcome'],2) if records else [],
                            B=[m for r in records[1:] for m in metrics(r['directory'],r['outcome'],0)]))


def run(directory=EVIDENCE, root=EXPERIMENTS):
    directory, root = Path(directory), Path(root)
    sizing = json.loads((directory/'sizing.json').read_text())
    if sizing['source_sha256']!=source_hashes() or sizing['recipe']!=RECIPE or not sizing['within_development_maximum'] or not sizing['fits_context']:
        raise ValueError('Require unchanged reviewed sizing and sources')
    if sizing['model_manifest_sha256']!=g.file_hash(OPTIONS['model_manifest']):
        raise ValueError('Changed model manifest')
    tasks = worklist()
    if [binding for binding,_ in tasks]!=sizing['bindings']:
        raise ValueError('Changed exploration block')
    root.mkdir(parents=True, exist_ok=True)
    existing = reader.summarize(root)['experiments']
    if any('-batch10-' in e['experiment'] for e in existing):
        raise FileExistsError('Batch 10 already exists; no redispatch')
    if any(not e['terminal'] for e in existing):
        raise ValueError('Resolve unfinished experiments before dispatch')
    first = max((int(e['experiment'][:4]) for e in existing),default=0)+1
    records, complete = [], False
    current, ids = None, []
    try:
        remaining = tasks
        for cycle in range(3):
            if not remaining:
                break
            ids = [binding['id'] for binding,_ in remaining]
            current = root/f'{first+cycle:04d}-batch10-{"a" if cycle==0 else "b"+str(cycle+1)}'
            recipe = dict(RECIPE) if cycle==0 else dict(RECIPE, temperature=0.7, max_repairs=1, base_seed=(100017, 200017)[cycle-1])
            result = b.run(remaining, current, **OPTIONS, mlx_python=base.MLX_PYTHON, recipe=recipe,
                reply_allowance=len(remaining)*(recipe['max_repairs']+1),
                output_token_allowance=len(remaining)*(recipe['max_repairs']+1)*2048,
                cohort_sha256=b.study_tasks.COHORT_SHA256)
            records.append(dict(directory=current, ids=ids, outcome=result))
            current = None
            if result['status']!='completed':
                raise RuntimeError('Technical stop; no further cycles')
            passed = passing_positions(result,1 if cycle==0 else None)
            remaining = [task for i,task in enumerate(remaining) if i not in passed]
        complete = True
    finally:
        if current is not None and (current/'outcome.json').exists():
            records.append(dict(directory=current, ids=ids, outcome=json.loads((current/'outcome.json').read_text())))
        g.save(directory/'summary.json', reader.summarize(root))
        g.save(directory/'comparison.json', compare(tasks,records,complete))
