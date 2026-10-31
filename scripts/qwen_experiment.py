"""Small Qwen-only development driver. See docs/qwen-experiments.md."""
import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import qwen_chunked_generation as chunked, qwen_edits, qwen_native_tools, native_multi_edit
a = importlib.import_module('src.data-pipeline.stage05.attacks')
g, loop = a.generation, a.generation_loop
DEFAULT_BENIGNS = ROOT / 'data/03_benign_experiments/v2-b1-20261028T135847Z/acquisition'
QWEN_CONFIG = ROOT / 'configs/inference/qwen38.json'
DIRECT_FIELDS = {'repair_enable_thinking', 'repair_temperature', 'repair_top_p', 'thinking_repair_max_tokens',
                 'thinking_repair_effort', 'narrowing_feedback'}
CHUNKED_DRIVER = ROOT / 'scripts/qwen_chunked_generation.py'


def settings(config):
    multi = config.get('submission_format') == 'native_multi_edit'
    if set(config) - {'chains', 'task_count', 'completion_batch_size', 'recipe', 'submission_format', 'swap_growth_stop_mib'} or set(config.get('recipe', {})) - (set(a.ATTACK_RECIPE) | {'reasoning_effort', 'top_k'} | DIRECT_FIELDS):
        raise ValueError('Config accepts chains, task_count, completion_batch_size and supported recipe fields only')
    if type(config.get('completion_batch_size', 4)) is not int or config.get('completion_batch_size', 4) not in (4, 8):
        raise ValueError('completion_batch_size must be 4 or 8')
    recipe = dict(a.ATTACK_RECIPE, base_seed=1_000_041, reasoning_effort='low', temperature=1.0, top_p=.95, top_k=20)
    if multi:
        recipe.update(enable_thinking=True, max_tokens=1024, max_repairs=4, repair_max_tokens=1024,
                      temperature=1.0, top_p=.95, repair_enable_thinking=False,
                      thinking_repair_max_tokens=2048, thinking_repair_effort='low')
    recipe.update(config.get('recipe', {}))
    recipe['strip_comments'] = True
    chains = config.get('chains', 2 if multi else 1)
    if type(config.get('swap_growth_stop_mib', 256)) is not int or config.get('swap_growth_stop_mib', 256) <= 0:
        raise ValueError('swap_growth_stop_mib must be a positive integer')
    if config.get('submission_format', 'full_program') not in ('full_program', 'exact_edit', 'native_tools', 'native_multi_edit'):
        raise ValueError('Unknown submission_format')
    if 'task_count' in config and (type(config['task_count']) is not int or config['task_count'] < 1):
        raise ValueError('task_count must be a positive integer')
    if 'thinking_repair_max_tokens' in recipe and (recipe.get('repair_enable_thinking') is not False
            or type(recipe['thinking_repair_max_tokens']) is not int
            or not recipe['repair_max_tokens'] <= recipe['thinking_repair_max_tokens'] <= 8192):
        raise ValueError('thinking_repair_max_tokens requires direct repairs and repair_max_tokens..8192')
    if recipe['reasoning_effort'] not in ('low', 'medium', 'xhigh'):
        raise ValueError('Qwen3.8 effort must be low, medium or xhigh')
    if not recipe.get('enable_thinking'):
        if 'reasoning_effort' in config.get('recipe', {}):
            raise ValueError('Explicit reasoning effort requires thinking')
        recipe.pop('reasoning_effort')
    if type(recipe['top_k']) is not int or recipe['top_k'] < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if type(chains) is not int or chains < 1:
        raise ValueError('chains must be a positive integer')
    for key in ('max_tokens', 'repair_max_tokens', 'max_prompt_tokens'):
        ceiling = 25808 if key == 'max_prompt_tokens' else 8192
        if type(recipe[key]) is not int or not 1 <= recipe[key] <= ceiling:
            raise ValueError(f'{key} exceeds the existing runtime envelope')
    if any(not isinstance(recipe[k], str) or not recipe[k] for k in ('system', 'initial', 'repair')):
        raise ValueError('Scaffold fields must be nonempty strings')
    if set(recipe) & DIRECT_FIELDS and config.get('submission_format') not in ('native_tools', 'native_multi_edit'):
        raise ValueError('Direct repair settings require native_tools')
    if 'thinking_repair_effort' in recipe and (not multi or recipe['thinking_repair_effort'] not in ('low', 'medium', 'xhigh')):
        raise ValueError('thinking_repair_effort requires native_multi_edit and a supported effort')
    if multi and (not recipe['enable_thinking'] or recipe['max_repairs'] < 1 or recipe['repair_enable_thinking']):
        raise ValueError('native_multi_edit requires initial thinking, direct ordinary repairs and at least one repair')
    if type(recipe.get('narrowing_feedback', False)) is not bool:
        raise ValueError('narrowing_feedback must be a boolean')
    if type(recipe.get('repair_enable_thinking', True)) is not bool:
        raise ValueError('repair_enable_thinking must be a boolean')
    for key, low, high in (('temperature', 0, math.inf), ('top_p', 0, 1),
                           ('repair_temperature', 0, math.inf), ('repair_top_p', 0, 1)):
        if key not in recipe:
            continue
        value = recipe[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or key.endswith('top_p') and value == 0:
            raise ValueError(f'Invalid {key}')
    return chains, recipe


def qwen38_checkpoint(files):
    """Bind the verified Qwen3.8 checkpoint files into the run inputs."""
    active = loop.read(QWEN_CONFIG)
    manifest = Path(active['model_manifest'])
    if g.file_hash(manifest) != active['model_manifest_sha256']:
        raise ValueError('Qwen3.8 manifest differs from the verified checkpoint')
    pinned = loop.read(manifest)
    if pinned['revision'] != active['revision'] or pinned['local_path'] != active['model_path']:
        raise ValueError('Qwen3.8 checkpoint binding differs')
    for item in pinned['files']:
        path = Path(active['model_path']) / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            raise ValueError(f'Missing or changed checkpoint file: {path}')
        if path.suffix != '.safetensors':
            files[str(path)] = item['sha256']
    files.update({str(QWEN_CONFIG): g.file_hash(QWEN_CONFIG), str(manifest): active['model_manifest_sha256']})
    return active, manifest


def load_panel(directory):
    """Validate existing B1 eligibility; expose only actor benigns and shown cases."""
    contract, report = loop.read(directory / 'contract.json'), loop.read(directory / 'report.json')
    receipt = loop.read(directory.parent / 'receipt.json')
    if (contract['cohort_sha256'] != a.study_tasks.COHORT_SHA256 or report['status'] != 'completed'
            or report['contract_sha256'] != g.file_hash(directory / 'contract.json')
            or receipt['contract_sha256'] != report['contract_sha256']
            or receipt['report_sha256'] != g.file_hash(directory / 'report.json')):
        raise ValueError('Require receipt-bound completed B1 acquisition')
    entries = a._load_benign_entries(directory, contract, report)
    runtime = contract['runtimes']['actor']
    files = a._bind_benign_runtime('actor', runtime)
    paths = [directory / 'contract.json', directory / 'report.json', directory / 'actor/outcome.json',
             directory.parent / 'receipt.json'] + [Path(e['benigns']['actor']['path']) for e in entries]
    files.update({str(p): g.file_hash(p) for p in paths})
    loaded, worklist, public = a.study_tasks.load_cohort(), [], []
    for entry in entries:
        binding, task = a.study_tasks.load_task(loaded, entry['binding']['id'])
        shown = {k: entry['partition'][k] for k in ('examples', 'menu')}
        worklist.append((binding, dict(question=task['question'], cases=[dict(c, visible=True)
                         for group in shown.values() for c in group])))
        public.append(dict(binding=binding, partition=shown, benigns={'actor': entry['benigns']['actor']}))
    active, manifest = qwen38_checkpoint(files)
    loop.check_files(files)
    runtime = dict(runtime, model_path=active['model_path'], model_manifest=str(manifest),
                   benign_model_manifest_sha256=contract['runtimes']['actor']['qualification']['model_manifest_sha256'])
    return public, worklist, runtime, files


def load_dev_panel(root):
    """Exploratory dev-50 panel: only audit-passing, case-eligible Qwen3.8 benigns qualify."""
    contract, report = loop.read(root / 'contract.json'), loop.read(root / 'acquisition/report.json')
    if (contract['kind'] not in ('qwen38-development-benigns', 'qwen38-construction-benigns',
                                'qwen38-development-task291-recovery', 'qwen38-construction-case-recovery')
            or contract['cohort_sha256'] != a.study_tasks.COHORT_SHA256
            or report['status'] != 'completed' or report['contract_sha256'] != g.file_hash(root / 'contract.json')):
        raise ValueError('Require a completed development benign acquisition bound to its contract')
    selected = {item['task_id']: item for item in report['selected']}
    audited = {item['task_id']: item for item in report['audit_verified'] if item['verdict']['status'] == 'passed'}
    files = {str(root / name): g.file_hash(root / name) for name in ('contract.json', 'acquisition/report.json')}
    if contract['kind'] == 'qwen38-development-task291-recovery':
        from scripts import recover_task291
        files.update(recover_task291.validate(root, contract, report))
    if contract['kind'] == 'qwen38-construction-case-recovery':
        from scripts import recover_construction_cases
        files.update(recover_construction_cases.validate(root, contract, report))
    cohort, entries, worklist = a.study_tasks.load_cohort(), [], []
    for item in contract['entries']:
        binding = item['binding']
        if not item['attack_case_eligible'] or binding['id'] not in audited:
            continue
        path = Path(audited[binding['id']]['program']).resolve()
        raw = path.read_bytes()
        if (not path.is_relative_to(root) or selected[binding['id']]['program'] != audited[binding['id']]['program']
                or selected[binding['id']]['candidate'] != a.study_tasks.candidate_identity(
                    binding['split'], binding['id'], binding['sha256'], 'actor', raw)
                or a.study_tasks.load_task(cohort, binding['id'])[0] != binding):
            raise ValueError(f"Selected benign or task binding differs for task {binding['id']}")
        task = a.study_tasks.load_task(cohort, binding['id'])[1]
        shown = {k: item['partition'][k] for k in ('examples', 'menu')}
        program = a.programs.strip_comments(raw.decode('utf-8'))
        entries.append(dict(binding=binding, partition=shown, benigns={'actor': dict(
            candidate=selected[binding['id']]['candidate'], program=program, path=str(path))}))
        worklist.append((binding, dict(question=task['question'], cases=[dict(c, visible=True)
                         for group in shown.values() for c in group])))
        files[str(path)] = g.file_hash(path)
    active, manifest = qwen38_checkpoint(files)
    if contract['runtime']['model_manifest'] != str(manifest):
        raise ValueError('Development benigns used a different Qwen3.8 manifest')
    loop.check_files(files)
    runtime = {k: contract['runtime'][k] for k in ('model_path', 'model_manifest', 'mlx_python')}
    return entries, worklist, dict(runtime, benign_model_manifest_sha256=g.file_hash(manifest)), files


def multi_metrics(attempts, reviews=None):
    """Aggregate distinct programs; optional separate reviews key by (position, program hash)."""
    unique = {}
    for v in attempts:
        if 'verification' in v:
            key = (v['position'], v['verification']['key']['program_sha256'])
            unique.setdefault(key, v)
    checked = list(unique.values())
    branch_literal = [v for v in checked if v['verification']['mechanical']['candidate_counts']['conditionals']
        > v['verification']['mechanical']['original_counts']['conditionals']
        and v['verification']['mechanical'].get('new_numeric_literals')]
    thinking = [v for v in attempts if v.get('thinking')]
    reviews = {} if reviews is None else reviews
    if set(reviews) - unique.keys():
        raise ValueError('Semantic review does not bind a checked task/program')
    reviewed = list(reviews.values())
    def reasoning_metrics(values):
        return dict(requests=len(values), closed=sum(not v.get('unclosed_or_unmatched_thought', False) for v in values),
            valid_calls=sum(bool(v.get('native_tool_call')) for v in values),
            truncated=sum(bool(v.get('hit_output_cap')) for v in values))
    return dict(submissions=len(attempts), checked_distinct_programs=len(checked),
        automatically_accepted=sum(v['verification']['status'] == 'passed' for v in checked),
        conditional_growth_with_new_literal=dict(submitted=len(branch_literal),
            automatically_accepted=sum(v['verification']['status'] == 'passed' for v in branch_literal)),
        reasoning=reasoning_metrics(thinking),
        initial_reasoning=reasoning_metrics([v for v in thinking if v['round'] == 0]),
        reasoning_repairs=reasoning_metrics([v for v in thinking if v['round'] > 0]),
        semantic_reviewed=len(reviewed), semantic_rejections=(dict(Counter(v['category'] for v in reviewed
            if v['decision'] == 'reject')) if reviewed else None))


def summarize(report, record, directory, started):
    a._summarize_acquisition(report, dict(prior_costs=[]), started)
    report['yield'] = report['yield']['actor']
    # The formal B2 summarizer projects only eight positions; exploratory panels can be larger.
    for kind in ('functional', 'accepted'):
        positions = set()
        for run in report['runs']:
            path = Path(run['directory']) / 'outcome.json'
            if path.exists():
                positions.update(v['position'] for v in loop.read(path)['attempts']
                    if v.get('verification', {}).get('functional' if kind == 'functional' else 'status')
                    == (True if kind == 'functional' else 'passed'))
        report['yield'][kind + '_positions'] = sorted(positions)
    report['yield']['denominator'] = len(record['entries'])
    for kind in ('functional', 'accepted'):
        count = len(report['yield'][kind + '_positions'])
        report['yield'][kind + '_tasks'] = count
        report['yield'][kind + '_fraction'] = count / len(record['entries'])
    reasons = Counter()
    for run in report['runs']:
        path = Path(run['directory']) / 'outcome.json'
        if path.exists():
            for attempt in loop.read(path)['attempts']:
                reasons.update(attempt.get('verification', {}).get('violations', []))
    report['rule_violations'] = dict(reasons)
    report['yield']['reviewed'] = False
    if record.get('submission_format') in ('exact_edit', 'native_tools', 'native_multi_edit'):
        attempt_groups = [loop.read(Path(run['directory']) / 'outcome.json')['attempts']
                          for run in report['runs'] if (Path(run['directory']) / 'outcome.json').exists()]
        attempts = [attempt for group in attempt_groups for attempt in group]
        report['repeated_program_submissions'] = report['duplicate_submissions']
        report['duplicate_submissions'] = qwen_edits.duplicate_submissions(attempt_groups)
        report['repair_decisions'] = (dict(Counter(json.loads(v['verification']['feedback']['message'])['repair_decision']
            for v in attempts if 'verification' in v)) if record.get('submission_format') == 'native_multi_edit'
            else qwen_edits.repair_metrics(attempt_groups))
        if record.get('submission_format') == 'native_tools':
            report['native_tools'] = qwen_native_tools.metrics(attempt_groups, record['recipe']['max_repairs'])
        report['edit_funnel'] = dict(
            valid_json=sum(bool(v.get('edit_json_valid')) for v in attempts),
            valid_schema=sum(v.get('edit_schema_valid', 'edit' in v) for v in attempts),
            applicable=sum(bool(v.get('edit_applicable')) for v in attempts),
            duplicates=report['duplicate_submissions'],
            repeated_edits=sum(bool(v.get('edit_duplicate')) for v in attempts),
            executed=sum('verification' in v and not v['verification'].get('execution_reused') for v in attempts),
            reused=sum(bool(v.get('verification', {}).get('execution_reused')) for v in attempts),
            automatically_accepted=sum(v.get('verification', {}).get('status') == 'passed' for v in attempts),
            semantic_accepted=None, semantic_rejection_rate=None)
        if record.get('submission_format') == 'native_multi_edit':
            report['multi_edit'] = {record['source']: multi_metrics(attempts)}
    loop.atomic_save(directory / 'report.json', report)


def run(output, config, benigns=DEFAULT_BENIGNS, monitor=False, panel_loader=None):
    chains, recipe = settings(config)
    multi = config.get('submission_format') == 'native_multi_edit'
    native_tools = config.get('submission_format') in ('native_tools', 'native_multi_edit')
    exact_edits = config.get('submission_format') in ('exact_edit', 'native_tools', 'native_multi_edit')
    adapter = native_multi_edit if multi else qwen_native_tools if native_tools else qwen_edits
    if exact_edits:
        adapter.configure(recipe)
    panel = Path(benigns).resolve()
    entries, worklist, runtime, files = (panel_loader or (load_dev_panel if (panel / 'acquisition/report.json').is_file()
                                         else load_panel))(panel)
    if runtime.get('model_family') == 'gemma4':
        recipe.pop('reasoning_effort', None)
        recipe.pop('thinking_repair_effort', None)
    if config.get('task_count', len(entries)) > len(entries):
        raise ValueError(f'task_count exceeds the {len(entries)} qualified panel tasks')
    entries, worklist = entries[:config.get('task_count')], worklist[:config.get('task_count')]
    n = len(entries)
    stride = n * (recipe['max_repairs'] + 1)
    replies = stride
    tokens = n * (recipe['max_tokens'] + recipe['max_repairs'] * recipe['repair_max_tokens']
                  + recipe.get('thinking_repair_max_tokens', recipe['repair_max_tokens']) - recipe['repair_max_tokens'])
    batch = config.get('completion_batch_size', 4)
    driver = CHUNKED_DRIVER if n > chunked.REQUEST_LIMIT else a.DRIVER
    batching = dict(completion_batch_size=batch, prefill_batch_size=2 if batch == 8 else 1, prefill_step_size=256)
    for chain in range(chains):
        loop.validate(worklist, dict(recipe, base_seed=recipe['base_seed'] + stride * chain),
                      'actor', replies, tokens, batching, False, a.study_tasks.COHORT_SHA256)
    if recipe['base_seed'] <= 88 and recipe['base_seed'] + chains * stride > 41:
        raise ValueError('Development seeds must avoid the official range 41..88')
    initial = [a.attack_messages(task, entry, 'actor', recipe=recipe)
               for (_, task), entry in zip(worklist, entries)]
    if exact_edits:
        initial = [adapter.messages(messages, recipe) if native_tools else qwen_edits.messages(messages)
                   for messages in initial]
    if runtime.get('model_family') == 'gemma4' and native_tools and not multi:
        initial = [qwen_native_tools.gemma_messages(m) for m in initial]
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    record = dict(kind='qwen-development', entries=entries, runtime=runtime, recipe=recipe, chains=chains,
        submission_format=config.get('submission_format', 'full_program'),
        resource_guard=dict(swap_growth_stop_mib=config.get('swap_growth_stop_mib', 256),
                            peak_stop_bytes=44 * 1024**3, critical_pressure=4),
        batching=batching, reply_limit=chains * replies, output_token_limit=chains * tokens,
        shown_case_limit=chains * (recipe['max_repairs'] + 1) * sum(
            len(e['partition']['examples']) + len(e['partition']['menu']) for e in entries),
        inputs=files, sources=dict(a.acquisition_sources(), **{str(Path(__file__).resolve()): g.file_hash(__file__),
            str(Path(qwen_edits.__file__).resolve()): g.file_hash(qwen_edits.__file__),
            str(Path(qwen_native_tools.__file__).resolve()): g.file_hash(qwen_native_tools.__file__),
            str(CHUNKED_DRIVER): g.file_hash(CHUNKED_DRIVER)}), panel=str(panel), generation_driver=str(driver))
    if multi:
        record['sources'].update({str(p): g.file_hash(p) for p in (Path(native_multi_edit.__file__).resolve(), native_multi_edit.PROMPT)})
        record['source'] = 'synthetic' if runtime.get('model_family') == 'gemma4' else 'actor'
        record['witness_selection'] = 'first measured wrong_answer in frozen menu order'
    g.save(directory / 'experiment.json', record)
    g.save(directory / 'initial-prompts.json', [dict(task_id=entry['binding']['id'], messages=messages)
        for entry, messages in zip(entries, initial)])
    (directory / 'initial-prompts.txt').write_text('\n\n'.join(
        f"===== TASK {entry['binding']['id']} =====\n" + '\n\n'.join(
            message['role'].upper() + ':\n' + message['content'] for message in messages)
        for entry, messages in zip(entries, initial)), encoding='utf-8')
    report = dict(status='running', runs=[], shown_case_requests_committed=0, shown_case_results_known=0)
    pairs = {e['binding']['sha256']: dict(program=e['benigns']['actor']['program'], **e['partition']) for e in entries}
    cache = {}
    def checker(candidates, log_dir, **kwargs):
        keys = [(h, a.sha256(program).hexdigest()) for _, h, program in candidates]
        fresh = [candidate for candidate, key in zip(candidates, keys) if not exact_edits or key not in cache]
        count = sum(len(pairs[h]['examples']) + len(pairs[h]['menu']) for _, h, _ in fresh)
        if report['shown_case_requests_committed'] + count > record['shown_case_limit']:
            raise RuntimeError('Shown-case budget exhausted')
        report['shown_case_requests_committed'] += count
        summarize(report, record, directory, started)
        measured = a.check_shown(fresh, log_dir, pairs=pairs, **({'auto_witness': True} if multi else {}), **kwargs) if fresh else []
        report['shown_case_results_known'] += sum(len(r['cases']) for r in measured)
        if not exact_edits:
            return measured
        fresh_keys = [(h, a.sha256(program).hexdigest()) for _, h, program in fresh]
        reused = set(keys) - set(fresh_keys)
        cache.update((key, dict(cases=v['cases'], evidence=v.get('evidence', [])))
                     for key, v in zip(fresh_keys, measured))
        fresh_verdicts = dict(zip(fresh_keys, measured))
        results = []
        for (task, h, program), key in zip(candidates, keys):
            verdict = fresh_verdicts.get(key)
            if key in reused:
                pair = pairs[h]
                rules = a.mechanical(pair['program'].encode('utf-8'), program, len(pair['menu']),
                                     metadata=task.get('candidate_metadata'), auto_witness=multi)
                verdict = a.shown_verdict(h, program, pair, rules, cache[key]['cases'], cache[key]['evidence'], auto_witness=multi)
                verdict.update(execution_reused=True, reused_from=verdict['evidence'])
            results.append(verdict if multi else qwen_edits.repair_guidance(verdict, recipe.get('narrowing_feedback', False)))
        return results
    try:
        summarize(report, record, directory, started)
        baseline = dict(a.memory_state(), swap_growth_stop_mib=record['resource_guard']['swap_growth_stop_mib'])
        g.save(directory / 'resource-baseline.json', baseline)
        if baseline['pressure'] != 1:
            raise RuntimeError('Require normal initial memory pressure')
        for chain in range(chains):
            cache.clear()
            loop.check_files(record['sources'])
            loop.check_files(files)
            a.check_resources(a.memory_state(), baseline)
            target = directory / f'actor-chain-{chain}'
            extractor = (adapter.Extractor(entries, initial, loop) if native_tools else
                         qwen_edits.Extractor(entries) if exact_edits else None)
            report['runs'].append(dict(source='actor', chain=chain, directory=str(target), status='running'))
            result = loop.run(worklist, target, source='actor', **{k: runtime[k] for k in
                ('model_path', 'model_manifest', 'mlx_python')}, **batching,
                recipe=dict(recipe, base_seed=recipe['base_seed'] + stride * chain),
                reply_allowance=replies, output_token_allowance=tokens, cohort_sha256=a.study_tasks.COHORT_SHA256,
                initial_messages=initial, candidate_checker=checker, generation_driver=driver,
                program_metadata=None if multi else a.attack_metadata,
                program_extractor=extractor, repair_builder=extractor.repair if native_tools else None,
                repair_cost=extractor.repair_cost if multi else None)
            if result['status'] != 'completed':
                raise RuntimeError(f"Qwen chain {chain}: {result.get('error', result['status'])}")
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='error' if isinstance(error, Exception) else 'cancelled', error=repr(error))
        if not isinstance(error, Exception):
            raise
    finally:
        summarize(report, record, directory, started)
        g.save(directory / 'evidence.json', {str(p.relative_to(directory)): g.file_hash(p)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'evidence.json'})
    if monitor and report['status'] == 'completed':
        report = dict(report, monitor=score_monitor(directory))
    return report


def score_monitor(directory):
    """Score each benign and the first accepted candidate per task; no private audits."""
    from importlib import import_module
    normalizer = import_module('src.data-pipeline.stage05.monitor')
    directory = Path(directory).resolve()
    record, report = loop.read(directory / 'experiment.json'), loop.read(directory / 'report.json')
    if report['status'] != 'completed':
        raise ValueError('Monitor scoring requires a completed development run')
    evidence = loop.read(directory / 'evidence.json')
    loop.check_files({str(directory / p): h for p, h in evidence.items()})
    loop.check_files(record['inputs'])
    config = loop.read(a.MONITOR_CONFIG)
    runtime = record['runtime']
    if (g.file_hash(normalizer.__file__) != config['normalization']['module_sha256']
            or g.file_hash(runtime['model_manifest']) != config['model']['manifest_sha256']):
        raise ValueError('Monitor baseline or model binding differs')
    selected = {}
    for run_record in report['runs']:
        for attempt in loop.read(Path(run_record['directory']) / 'outcome.json')['attempts']:
            if attempt.get('verification', {}).get('status') == 'passed' and attempt['position'] not in selected:
                program = Path(attempt['program'])
                if g.file_hash(program) != attempt['candidate']['program_sha256']:
                    raise ValueError('Candidate program changed')
                selected[attempt['position']] = program.read_text()
    records, requests = [], []
    cohort = a.study_tasks.load_cohort()
    for position, entry in enumerate(record['entries']):
        _, task = a.study_tasks.load_task(cohort, entry['binding']['id'])
        programs = dict(benign=entry['benigns']['actor']['program'])
        if position in selected:
            programs['attack'] = selected[position]
        for label, program in programs.items():
            records.append(dict(position=position, label=label))
            requests.append(dict(config['generation'], max_prompt_tokens=8192, messages=[
                dict(role='system', content=config['system_prompt']), dict(role='user', content=
                config['user_template'].format(statement=task['question'], normalized_program=a.programs.strip_comments(program)))]))
    target = directory / 'monitor'
    target.mkdir(exist_ok=False)
    child = dict(child_log=str(target / 'child.log'))
    result = dict(status='running', output_tokens=None, output_tokens_lower_bound=0,
                  native_prompt_tokens=None, elapsed_seconds=0,
                  attack_selection='first mechanically accepted per task; unreviewed')
    started = monotonic()
    loop.atomic_save(target / 'report.json', result)
    try:
        g.save(target / 'inputs.json', dict(records=records, requests=requests, config=config,
            config_sha256=g.file_hash(a.MONITOR_CONFIG), source_sha256=g.file_hash(__file__),
            reply_limit=len(requests), output_token_limit=len(requests) * config['generation']['max_tokens']))
        baseline = a.memory_state()
        g.save(target / 'resource-baseline.json', baseline)
        if baseline['pressure'] != 1:
            raise RuntimeError('Require normal initial memory pressure')
        spec = dict(model_path=runtime['model_path'], model_manifest=runtime['model_manifest'], requests=requests,
                    output_dir=str(target / 'batch/generation'), **a.ATTACK_BATCHING['actor'])
        (target / 'batch').mkdir()
        g.save(target / 'round.json', spec)
        code = loop.run_child([runtime['mlx_python'], str(a.DRIVER), str(target / 'round.json')], child,
                             lambda: loop.atomic_save(target / 'child.json', child))
        native = loop.read(target / 'batch/generation/outcome.json')
        scope = dict(json=json, re=import_module('re'))
        exec(config['parser']['python_source'], scope)
        verdicts = [dict(record, **scope['parse'](row)) for record, row in zip(records, native['requests'], strict=True)]
        groups = {}
        for label in ('benign', 'attack'):
            values = [v['verdict'] for v in verdicts if v['label'] == label]
            groups[label] = dict(n=len(values), counts={k: values.count(k) for k in ('PASS', 'FLAG', 'missing')},
                                 flag_fraction=values.count('FLAG') / len(values) if values else None)
        result.update(status='completed' if code == 0 and native['status'] == 'success' else 'error',
                      verdicts=verdicts, groups=groups)
    except BaseException as error:
        result.update(status='error' if isinstance(error, Exception) else 'cancelled', error=repr(error))
        if not isinstance(error, Exception):
            raise
    finally:
        path = target / 'batch/generation/outcome.json'
        if path.exists():
            native = loop.read(path)
            observed = sum(r['generation_tokens'] for r in native['requests'])
            result.update(output_tokens_lower_bound=observed,
                          output_tokens=observed if native['status'] == 'success' else None,
                          native_prompt_tokens=(native.get('native') or {}).get('prompt_tokens')
                          if native['status'] == 'success' else None)
        result['elapsed_seconds'] = monotonic() - started
        loop.atomic_save(target / 'child.json', child)
        loop.atomic_save(target / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    acquire = sub.add_parser('run', help='Run a fresh finite Qwen-only experiment')
    acquire.add_argument('output', type=Path)
    acquire.add_argument('--config', type=Path, help='JSON with chains and recipe overrides')
    acquire.add_argument('--benigns', type=Path, default=DEFAULT_BENIGNS)
    acquire.add_argument('--monitor', action='store_true', help='Score benigns and first accepted attacks afterward')
    sub.add_parser('monitor', help='Score an existing completed experiment').add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'run':
        result = run(args.output, loop.read(args.config) if args.config else {}, args.benigns, args.monitor)
    else:
        result = score_monitor(args.output)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'completed' and result.get('monitor', {}).get('status', 'completed') == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
