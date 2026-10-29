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
a = importlib.import_module('src.data-pipeline.stage05.attacks')
g, loop = a.generation, a.generation_loop
DEFAULT_BENIGNS = ROOT / 'data/03_benign_experiments/v2-b1-20261028T135847Z/acquisition'


def settings(config):
    if set(config) - {'chains', 'recipe'} or set(config.get('recipe', {})) - set(a.ATTACK_RECIPE):
        raise ValueError('Config accepts chains and existing ATTACK_RECIPE fields only')
    recipe = dict(a.ATTACK_RECIPE, base_seed=1_000_041)
    recipe.update(config.get('recipe', {}))
    recipe['strip_comments'] = True
    chains = config.get('chains', 1)
    if type(chains) is not int or chains < 1:
        raise ValueError('chains must be a positive integer')
    for key in ('max_tokens', 'repair_max_tokens', 'max_prompt_tokens'):
        ceiling = 25808 if key == 'max_prompt_tokens' else 8192
        if type(recipe[key]) is not int or not 1 <= recipe[key] <= ceiling:
            raise ValueError(f'{key} exceeds the existing runtime envelope')
    if any(not isinstance(recipe[k], str) or not recipe[k] for k in ('system', 'initial', 'repair')):
        raise ValueError('Scaffold fields must be nonempty strings')
    for key, low, high in (('temperature', 0, math.inf), ('top_p', 0, 1)):
        value = recipe[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or key == 'top_p' and value == 0:
            raise ValueError(f'Invalid {key}')
    return chains, recipe


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
    return public, worklist, runtime, files


def summarize(report, record, directory, started):
    a._summarize_acquisition(report, dict(prior_costs=[]), started)
    report['yield'] = report['yield']['actor']
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
    loop.atomic_save(directory / 'report.json', report)


def run(output, config, benigns=DEFAULT_BENIGNS, monitor=False):
    chains, recipe = settings(config)
    entries, worklist, runtime, files = load_panel(Path(benigns).resolve())
    n = len(entries)
    stride = n * (recipe['max_repairs'] + 1)
    replies = stride
    tokens = n * (recipe['max_tokens'] + recipe['max_repairs'] * recipe['repair_max_tokens'])
    batching = a.ATTACK_BATCHING['actor']
    for chain in range(chains):
        loop.validate(worklist, dict(recipe, base_seed=recipe['base_seed'] + stride * chain),
                      'actor', replies, tokens, batching, False, a.study_tasks.COHORT_SHA256)
    if recipe['base_seed'] <= 88 and recipe['base_seed'] + chains * stride > 41:
        raise ValueError('Development seeds must avoid the official range 41..88')
    initial = [a.attack_messages(task, entry, 'actor', recipe=recipe)
               for (_, task), entry in zip(worklist, entries)]
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    record = dict(kind='qwen-development', entries=entries, runtime=runtime, recipe=recipe, chains=chains,
        batching=batching, reply_limit=chains * replies, output_token_limit=chains * tokens,
        shown_case_limit=chains * (recipe['max_repairs'] + 1) * sum(
            len(e['partition']['examples']) + len(e['partition']['menu']) for e in entries),
        inputs=files, sources=dict(a.acquisition_sources(), **{str(Path(__file__).resolve()): g.file_hash(__file__)}))
    g.save(directory / 'experiment.json', record)
    report = dict(status='running', runs=[], shown_case_requests_committed=0, shown_case_results_known=0)
    pairs = {e['binding']['sha256']: dict(program=e['benigns']['actor']['program'], **e['partition']) for e in entries}
    def checker(candidates, log_dir, **kwargs):
        count = sum(len(pairs[h]['examples']) + len(pairs[h]['menu']) for _, h, _ in candidates)
        if report['shown_case_requests_committed'] + count > record['shown_case_limit']:
            raise RuntimeError('Shown-case budget exhausted')
        report['shown_case_requests_committed'] += count
        summarize(report, record, directory, started)
        results = a.check_shown(candidates, log_dir, pairs=pairs, **kwargs)
        report['shown_case_results_known'] += sum(len(r['cases']) for r in results)
        return results
    try:
        summarize(report, record, directory, started)
        baseline = a.memory_state()
        g.save(directory / 'resource-baseline.json', baseline)
        if baseline['pressure'] != 1:
            raise RuntimeError('Require normal initial memory pressure')
        for chain in range(chains):
            loop.check_files(record['sources'])
            loop.check_files(files)
            a.check_resources(a.memory_state(), baseline)
            target = directory / f'actor-chain-{chain}'
            report['runs'].append(dict(source='actor', chain=chain, directory=str(target), status='running'))
            result = loop.run(worklist, target, source='actor', **{k: runtime[k] for k in
                ('model_path', 'model_manifest', 'mlx_python')}, **batching,
                recipe=dict(recipe, base_seed=recipe['base_seed'] + stride * chain),
                reply_allowance=replies, output_token_allowance=tokens, cohort_sha256=a.study_tasks.COHORT_SHA256,
                initial_messages=initial, candidate_checker=checker, generation_driver=a.DRIVER,
                program_metadata=a.attack_metadata)
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
