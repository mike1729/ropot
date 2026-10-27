"""Batch 12 revised recipes: rescue previously used, unresolved tasks only."""
from collections import Counter
import json
from pathlib import Path
import sys

from experiments import batch10_scaffold as prior

b, g, base = prior.b, prior.g, prior.base
ROOT = prior.ROOT
EVIDENCE = ROOT/'data/03_generation_checks/batch12-rerun'
EXPERIMENTS = prior.EXPERIMENTS
OPTIONS = {options['source']: options for options in base.RUNS}
RECIPES = dict(actor=dict(base.RECIPE, max_tokens=2048, repair_max_tokens=2048, max_repairs=3),
               synthetic=dict(prior.RECIPE, max_repairs=3))
MAXIMA = dict(actor=745, synthetic=755)
SEEDS = (1000017, 1100017, 1200017)
ATTEMPTS = 4  # Per cycle: one fresh draw and up to three repairs.
ORDER = ('rescue-actor', 'rescue-synthetic')


def source_hashes():
    return prior.source_hashes() | {str(Path(__file__).resolve()): g.file_hash(__file__)}


def cohort_blocks():
    """Previously used development IDs and construction candidates 1–32 per phase."""
    cohort = b.study_tasks.load_cohort()
    phases = [cohort['construction_phases'][phase]['initial'] for phase in prior.PHASES]
    blocks = dict(development=cohort['groups']['development']['initial'],
                  construction=[i for ids in phases for i in ids[:32]])
    if [len(ids) for ids in blocks.values()] != [50, 64]:
        raise ValueError('Require the saved development and construction blocks')
    return cohort, blocks


def prior_passes(root):
    """Task IDs each source has passed in any numbered experiment."""
    passed = dict(actor=set(), synthetic=set())
    for directory in sorted(Path(root).glob('[0-9][0-9][0-9][0-9]-*')):
        inputs = json.loads((directory/'inputs.json').read_text())
        outcome = json.loads((directory/'outcome.json').read_text())
        ids = [item['binding']['id'] for item in inputs['worklist']]
        passed[inputs['source']].update(ids[i] for i in prior.passing_positions(outcome))
    return passed


def worklists(root=EXPERIMENTS):
    cohort, blocks = cohort_blocks()
    passed = prior_passes(root)
    ids = {}
    for source in ('actor', 'synthetic'):
        ids[f'rescue-{source}'] = [i for i in blocks['development']+blocks['construction'] if i not in passed[source]]
    return {name: [b.study_tasks.load_task(cohort, i) for i in ids[name]] for name in ORDER}


def stress_parent():
    detail = ' '.join(str(i) for i in range(512))
    visible = [dict(status='wrong_answer', returncode=0,
        **dict.fromkeys(('input', 'answer', 'stdout', 'stderr', 'diagnostic'), detail)) for _ in range(2)]
    return dict(extraction='success', verification=dict(feedback=dict(status='failed', hidden='failed', visible=visible)))


def measure(directory=EVIDENCE):
    from mlx_lm.utils import load_config, load_tokenizer
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tasks, models = worklists(), {}
    for source, options in OPTIONS.items():
        recipe, names = RECIPES[source], [name for name in ORDER if name.endswith(source)]
        config = load_config(options['model_path'])
        tokenizer = load_tokenizer(options['model_path'], dict(local_files_only=True,
            **({'fix_mistral_regex': True} if source == 'synthetic' else {})), eos_token_ids=config.get('eos_token_id'))
        requests = {name: b.initial_requests(tasks[name], recipe) for name in names}
        counts = {name: [g.render(tokenizer, r['messages'])['prompt_tokens'] for r in rs] for name, rs in requests.items()}
        name, position = max(((n, i) for n in names for i in range(len(counts[n]))), key=lambda k: counts[k[0]][k[1]])
        reply_ids = tokenizer.encode('print(0)\n' * 2048, add_special_tokens=False)[:2048]
        reply = g.decode_tokens(tokenizer, reply_ids, set(tokenizer.eos_token_ids))
        request = requests[name][position]
        for _ in range(recipe['max_repairs']):
            request = b.repair_request(request, dict(text=reply), stress_parent(), recipe, SEEDS[0])
        stress = g.render(tokenizer, request['messages'])['prompt_tokens']
        context = config.get('text_config', config)['max_position_embeddings']
        models[source] = dict(recipe=recipe, prompt_tokens=counts, development_maximum=MAXIMA[source],
            within_development_maximum=max(max(v) for v in counts.values() if v) <= MAXIMA[source],
            stress=dict(worklist=name, position=position, depth=recipe['max_repairs'], prompt_tokens=stress,
                        input_plus_output=stress + recipe['repair_max_tokens'], messages=request['messages']),
            context=context, fits_context=stress + recipe['repair_max_tokens'] <= context,
            model_manifest_sha256=g.file_hash(options['model_manifest']))
    record = dict(source_sha256=source_hashes(), seeds=SEEDS, cohort_sha256=b.study_tasks.COHORT_SHA256,
        bindings={name: [binding for binding, _ in ts] for name, ts in tasks.items()}, models=models,
        worst_case={name: dict(replies=ATTEMPTS*len(SEEDS)*len(ts), tokens=ATTEMPTS*len(SEEDS)*len(ts)*2048)
                    for name, ts in tasks.items()})
    g.save(directory/'sizing.json', record)
    if not all(m['within_development_maximum'] and m['fits_context'] for m in models.values()):
        raise ValueError('Sizing exceeds reviewed maximum; stop for review')
    return record


def round_metrics(directory, outcome):
    metrics = []
    for current in outcome['rounds']:
        try:
            metrics.extend(prior.metrics(directory, dict(outcome, rounds=[current]), 0))
        except (OSError, ValueError) as error:
            metrics.append(dict(round=current['round'], experiment=str(directory),
                attempts=len(current['attempt_ids']), committed_tokens=None,
                observed_tokens=current.get('observed_tokens'), usage_complete=current.get('usage_complete'),
                prompt_tokens=None, peak_memory_bytes=None, timing=current.get('timing'),
                evidence_error=str(error)))
    return metrics


def report(tasks, records, before, blocks, complete):
    first, states, programs = ({name: {} for name in ORDER} for _ in range(3))
    effort = []
    for record in records:
        name, ids, outcome = record['worklist'], record['ids'], record['outcome']
        for task in outcome['tasks']:
            states[name][ids[task['position']]] = task['state']
        for attempt in outcome['attempts']:
            task_id = ids[attempt['position']]
            if attempt.get('candidate'):
                programs[name].setdefault(task_id, []).append((attempt['round'] == 0, attempt['candidate']['program_sha256']))
            if attempt['state'] == 'passed' or attempt.get('verification', {}).get('status') == 'passed':
                first[name][task_id] = min(first[name].get(task_id, 99), record['offset']+attempt['round']+1)
        metrics = round_metrics(record['directory'], outcome)
        for metric in metrics:
            metric['attempt'] = record['offset']+metric['round']+1
            metric['unusable'] = dict(Counter(a['extraction'] for a in outcome['attempts']
                if a['round'] == metric['round'] and a.get('extraction') not in (None, 'success')))
        effort.append(dict(worklist=name, experiment=record['directory'], offset=record['offset'],
            counters={k: outcome.get(k) for k in ('committed_replies', 'committed_tokens', 'known_submissions',
                'known_submissions_lower_bound', 'observed_tokens', 'usage_complete')}, timing=outcome.get('timing'), rounds=metrics))
    after = {s: before[s] | set(first[f'rescue-{s}']) for s in ('actor', 'synthetic')}
    coverage = {group: dict(denominator=len(blocks[group]), **{f'{s}_{when}': len(set(blocks[group]) & passes[s])
        for when, passes in (('before', before), ('after', after)) for s in ('actor', 'synthetic')},
        both_before=len(set(blocks[group]) & before['actor'] & before['synthetic']),
        both_after=len(set(blocks[group]) & after['actor'] & after['synthetic'])) for group in ('development', 'construction')}
    diversity = {name: {task_id: dict(programs=len(seen), distinct=len({d for _, d in seen}),
        fresh=sum(f for f, _ in seen), fresh_distinct=len({d for f, d in seen if f})) for task_id, seen in found.items()}
        for name, found in programs.items()}
    return dict(status='complete' if complete else 'incomplete/lower_bounds',
        curves={name: dict(denominator=len(tasks[name]), passed=[sum(n <= k for n in first[name].values())
            for k in range(1, ATTEMPTS*len(SEEDS)+1)]) for name in ORDER},
        coverage=coverage, diversity=diversity, effort=effort,
        tasks={name: [dict(binding=binding, first_pass=first[name].get(binding['id']),
            state=states[name].get(binding['id'], 'not_submitted')) for binding, _ in ts] for name, ts in tasks.items()},
        interpretation='All tasks were previously used and unresolved for this source. '
                       'New passes are acquisition after tuning, not clean recipe-yield estimates.')


def run(directory=EVIDENCE, root=EXPERIMENTS):
    directory, root = Path(directory), Path(root)
    if Path(sys.executable).absolute() != ROOT/'.venv/bin/python':
        raise ValueError('Run the caller in repository .venv/bin/python')
    sizing = json.loads((directory/'sizing.json').read_text())
    if sizing['source_sha256'] != source_hashes() or list(sizing['seeds']) != list(SEEDS):
        raise ValueError('Changed sizing sources or schedule')
    for source, options in OPTIONS.items():
        model = sizing['models'][source]
        if (model['recipe'] != RECIPES[source] or not model['within_development_maximum'] or not model['fits_context']
                or model['model_manifest_sha256'] != g.file_hash(options['model_manifest'])):
            raise ValueError('Changed recipe, model manifest or unacceptable sizing')
    existing = prior.reader.summarize(root)['experiments']
    if any('-batch12-' in e['experiment'] for e in existing):
        raise FileExistsError('Batch 12 already exists; no redispatch')
    if any(not e['terminal'] or e['fixtures'] is not False or e['cohort_sha256'] != b.study_tasks.COHORT_SHA256 for e in existing):
        raise ValueError('Require terminal study experiments on the pinned cohort')
    tasks, before, (_, blocks) = worklists(root), prior_passes(root), cohort_blocks()
    if {name: [binding for binding, _ in ts] for name, ts in tasks.items()} != sizing['bindings']:
        raise ValueError('Worklists changed since sizing')
    first = max((int(e['experiment'][:4]) for e in existing), default=0)+1
    records, current, descriptor, complete = [], None, None, False
    try:
        for name in ORDER:
            source, remaining = name.split('-')[1], tasks[name]
            for cycle, seed in enumerate(SEEDS):
                if not remaining:
                    break
                recipe = dict(RECIPES[source], base_seed=seed)
                current = root/f'{first+len(records):04d}-batch12-{name}-c{cycle+1}'
                descriptor = dict(worklist=name, directory=str(current), offset=ATTEMPTS*cycle,
                                  ids=[binding['id'] for binding, _ in remaining])
                result = b.run(remaining, current, **OPTIONS[source], mlx_python=base.MLX_PYTHON, recipe=recipe,
                    reply_allowance=ATTEMPTS*len(remaining), output_token_allowance=ATTEMPTS*len(remaining)*recipe['max_tokens'],
                    cohort_sha256=b.study_tasks.COHORT_SHA256)
                records.append(dict(descriptor, outcome=result))
                current = None
                if result['status'] != 'completed':
                    raise RuntimeError('Technical stop; no subsequent cycle or worklist')
                passed = prior.passing_positions(result)
                remaining = [task for i, task in enumerate(remaining) if i not in passed]
        complete = True
    finally:
        if current is not None and (current/'outcome.json').exists():
            records.append(dict(descriptor, outcome=json.loads((current/'outcome.json').read_text())))
        g.save(directory/'summary.json', prior.reader.summarize(root))
        g.save(directory/'results.json', report(tasks, records, before, blocks, complete))
