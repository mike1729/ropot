"""Prepare a finite DeepInfra job offline, then explicitly launch its pinned rounds."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
loop = importlib.import_module('src.data-pipeline.stage03.generation_loop')
runtime_api = importlib.import_module('src.data-pipeline.stage03.deepinfra_runtime')
generation = runtime_api.generation
SOURCES = dict(qwen='actor', gemma='synthetic')

# This interpreter owns the pinned tokenizer packages. No key or HTTP is needed.
PREFLIGHT = '''import json,sys
from hashlib import sha256
from scripts import deepinfra_generation as worker
tokenizer,rows,payloads,cost,_ = worker.prepare(json.load(sys.stdin))
print(json.dumps(dict(versions=tokenizer.versions,
 prompt_tokens=[row['prompt_tokens'] for row in rows],
 payload_sha256=[sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for p in payloads],
 planned_max_cost_usd=cost)))
'''


def _preflight(specification, python):
    env = dict(os.environ)
    env.pop('DEEPINFRA_API_KEY', None)
    result = subprocess.run([python, '-c', PREFLIGHT], input=json.dumps(specification),
                            text=True, capture_output=True, check=True, cwd=ROOT, env=env)
    return json.loads(result.stdout)


def prepare(plan_path, output_dir, *, expected_plan_sha256, runtimes, generation_python,
            max_tokens, max_cost_usd, limits_path):
    """Render every round before publishing a launch manifest; never send HTTP."""
    directory = Path(output_dir).absolute()
    if directory.exists():
        raise FileExistsError(directory)
    plan_path, limits_path = Path(plan_path).resolve(strict=True), Path(limits_path).resolve(strict=True)
    if generation.file_hash(plan_path) != expected_plan_sha256:
        raise ValueError('Changed experiment plan')
    plan, limits = loop.read(plan_path), loop.read(limits_path)
    families = list(plan['rounds'])
    if (not families or set(families) - SOURCES.keys() or set(runtimes) != set(families)
            or set(max_tokens) != set(families) or set(limits) != set(families)
            or not runtime_api.positive(max_cost_usd)):
        raise ValueError('Require finite matching rounds, runtimes, limits and a total ceiling')
    python = str(Path(generation_python).absolute())  # Preserve the venv symlink path.
    files = {str(p): generation.file_hash(p) for p in (plan_path, limits_path)}
    for path, digest in plan.get('inputs', {}).items():
        files[str((ROOT / path).resolve(strict=True))] = digest
    loop.check_files(files)
    sources = (*loop.source_files(), Path(__file__), ROOT / 'scripts/whole_program_thinking_probe.py',
               ROOT / 'src/data-pipeline/stage05/attacks.py')
    jobs, total = {}, 0
    for family in families:
        planned, cap, limit = plan['rounds'][family], max_tokens[family], limits[family]
        if (type(cap) is not int or cap <= 0 or not planned['requests']
                or any(type(limit[k]) is not int or limit[k] <= 0 for k in ('max_output_tokens', 'max_total_tokens'))
                or cap > limit['max_output_tokens'] or not limit['evidence']):
            raise ValueError('Require an evidenced provider-accepted finite output/context envelope')
        requests = []
        for request in planned['requests']:
            if 'max_tokens' in request and request['max_tokens'] != cap:
                raise ValueError('Explicit cap differs from the frozen recipe')
            requests.append(dict(request, max_tokens=cap))
        binding = runtime_api.bind(runtimes[family], len(requests), len(requests) * cap,
                                   sources, source=SOURCES[family])
        runtime = binding['runtime']
        if (planned['model'] not in (runtime['model'], runtime['model'].partition(':')[0])
                or limit['model'] != runtime['model']
                or family == 'gemma' and runtime.get('output_protocol') != 'gemma_native_text'
                or runtime['max_total_tokens'] > limit['max_total_tokens']):
            raise ValueError('Model, output protocol or context differs from the reviewed provider envelope')
        evidence = {str(Path(path).resolve(strict=True)): digest for path, digest in limit['evidence'].items()}
        loop.check_files(evidence)
        files.update(evidence)
        files.update(binding['files'])
        specification = dict(runtime=runtime, requests=requests, output_dir=str(directory / family))
        preflight = _preflight(specification, python)
        total += binding['planned_max_cost_usd']
        jobs[family] = dict(specification=specification, preflight=preflight,
                            reserved_max_cost_usd=binding['planned_max_cost_usd'])
    loop.check_files(files)
    if total > max_cost_usd:
        raise ValueError('All rounds together exceed the planned spending ceiling')
    manifest = dict(kind='deepinfra-finite-launch-v1', plan=str(plan_path), generation_python=python,
                    files=files, rounds=jobs, max_cost_usd=max_cost_usd, reserved_max_cost_usd=total)
    directory.mkdir(parents=True, exist_ok=False)
    for family, job in jobs.items():
        path = directory / f'round-{family}.json'
        loop.atomic_save(path, job.pop('specification'))
        job.update(round_file=str(path), round_sha256=generation.file_hash(path))
    loop.atomic_save(directory / 'launch.json', manifest)
    return manifest


def launch(directory, *, expected_sha256):
    """Dispatch once after all pins and both rendered payloads still agree."""
    directory = Path(directory).absolute()
    path = directory / 'launch.json'
    if generation.file_hash(path) != expected_sha256:
        raise ValueError('Changed launch manifest')
    manifest = loop.read(path)
    if manifest['kind'] != 'deepinfra-finite-launch-v1':
        raise ValueError('Unknown launch manifest')
    if ((directory / 'launch.claim').exists() or (directory / 'outcome.json').exists() or any((directory / family).exists()
            or (directory / f'{family}.log').exists() for family in manifest['rounds'])):
        raise FileExistsError('Launch destinations already contain evidence; no automatic retry')
    loop.check_files(manifest['files'])
    for job in manifest['rounds'].values():
        loop.check_files({job['round_file']: job['round_sha256']})
        if _preflight(loop.read(job['round_file']), manifest['generation_python']) != job['preflight']:
            raise ValueError('Rendered payload, package versions or reservation changed')
    # Recheck after the CPU preflights as well as before them.
    loop.check_files(manifest['files'])
    key = os.environ.get('DEEPINFRA_API_KEY', '')
    if not key or any(c.isspace() for c in key):
        raise ValueError('Export DEEPINFRA_API_KEY where the launcher runs')
    # Keep the exclusive claim even after failure: another launcher must not
    # overwrite accounting or redispatch this job after a crash or interruption.
    with (directory / 'launch.claim').open('x') as claim:
        claim.write(expected_sha256 + '\n')
    result = dict(status='running', launch_sha256=expected_sha256, error=None,
                  reserved_max_cost_usd=manifest['reserved_max_cost_usd'], rounds={})
    for family in manifest['rounds']:
        result['rounds'][family] = dict(status='not_run', usage_complete=True,
            known_submissions_lower_bound=0, observed_estimated_cost_usd=0,
            child_log=str(directory / f'{family}.log'))

    def persist():
        result.update(usage_complete=all(r['usage_complete'] for r in result['rounds'].values()),
            observed_estimated_cost_usd=sum(r['observed_estimated_cost_usd'] for r in result['rounds'].values()))
        loop.atomic_save(directory / 'outcome.json', result)

    persist()
    try:
        for family, job in manifest['rounds'].items():
            loop.check_files(manifest['files'])
            loop.check_files({job['round_file']: job['round_sha256']})
            current = result['rounds'][family]
            current.update(status='running', usage_complete=False)
            persist()
            code, outcome = None, None
            try:
                code = loop.run_child([manifest['generation_python'], str(runtime_api.WORKER),
                                       job['round_file']], current, persist)
            finally:
                outcome_path = Path(loop.read(job['round_file'])['output_dir']) / 'outcome.json'
                if outcome_path.exists():
                    outcome = loop.read(outcome_path)
                    current.update(status=outcome['status'], usage_complete=outcome['usage_complete'],
                        known_submissions_lower_bound=sum(r['submitted'] is True for r in outcome['requests']),
                        observed_estimated_cost_usd=outcome['observed_estimated_cost_usd'],
                        outcome=str(outcome_path), error=outcome.get('error'))
                elif 'popen_error' in current:
                    current.update(status='error', usage_complete=True)
                else:
                    current.update(status='submission_unknown', usage_complete=False)
                persist()
            if code != 0 or outcome is None or outcome['status'] != 'success':
                raise RuntimeError(f'{family} failed; later rounds remain not run')
        result['status'] = 'success'
    except BaseException as error:
        result.update(status='error' if isinstance(error, Exception) else 'cancelled',
                      error=repr(error).replace(key, '<redacted>'))
    finally:
        persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('plan', type=Path)
    prep.add_argument('output', type=Path)
    prep.add_argument('--plan-sha256', required=True)
    prep.add_argument('--generation-python', required=True)
    prep.add_argument('--limits', type=Path, required=True)
    prep.add_argument('--max-cost-usd', type=float, required=True)
    for family in SOURCES:
        prep.add_argument(f'--{family}-runtime', type=Path)
        prep.add_argument(f'--{family}-max-tokens', type=int)
    run = commands.add_parser('launch')
    run.add_argument('directory', type=Path)
    run.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if args.command == 'launch':
        return 0 if launch(args.directory, expected_sha256=args.sha256)['status'] == 'success' else 1
    runtimes = {f: getattr(args, f'{f}_runtime') for f in SOURCES if getattr(args, f'{f}_runtime') is not None}
    caps = {f: getattr(args, f'{f}_max_tokens') for f in runtimes}
    prepared = prepare(args.plan, args.output, expected_plan_sha256=args.plan_sha256,
        runtimes=runtimes, generation_python=args.generation_python, max_tokens=caps,
        max_cost_usd=args.max_cost_usd, limits_path=args.limits)
    print(json.dumps(dict(launch_sha256=generation.file_hash(args.output / 'launch.json'),
                          reserved_max_cost_usd=prepared['reserved_max_cost_usd'])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
