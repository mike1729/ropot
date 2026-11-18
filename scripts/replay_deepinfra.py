"""Replay two pinned historical captures through the pipeline; never call DeepInfra."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import native_multi_edit

loop = importlib.import_module('src.data-pipeline.stage03.generation_loop')
verification = importlib.import_module('src.data-pipeline.stage03.verification')
attacks = importlib.import_module('src.data-pipeline.stage05.attacks')
g, runtime = loop.generation, loop.deepinfra_runtime
CAPTURES = ROOT / 'data/monitor_training/deepinfra-compat-20261117-01'
DEFAULT_TOKENIZER = ROOT / 'data/monitor_training/runpod-a40-20261116/final-backup/workspace/pilot-runtime/model'
PINS = {
    'workload.json': 'ed02d0d9d6bf76f59305728f8f11f64cfc47fda077f64545d6861ad1380b7680',
    '13-generate-benign-1818-direct-request.json': '23690bcde1ed8234bb34b9bdaa578684de971ec685fa438044221b83e485d3bd',
    '13-generate-benign-1818-direct-response.json': '0c9ee585fc5ca4aa599abd06e3479db42ac81a045ee0bba747b91a8ca223b4ce',
    '14-generate-attack-1818-direct-request.json': '15e03d3ac1ec8c95291b1b47d3a137d58d791bba871e82fbc8dc67c500e0894f',
    '14-generate-attack-1818-direct-response.json': 'a3df1a02eff8271d67353a95571053a0b63b68ba2c0a520eb71d944df2775e11',
}
CHILD = """
import json,os,sys
from pathlib import Path
from unittest import mock
from scripts import deepinfra_generation as worker
fixture=json.loads(Path(sys.argv[2]).read_text())
calls=0
def replay(payload,key,timeout):
    global calls
    calls+=1
    if calls != 1:
        raise ValueError('A captured response can be replayed only once')
    if payload != fixture['payload']:
        raise ValueError('Replay payload differs from the captured request')
    return fixture['response']['http_status'],json.dumps(fixture['response']['body'])
with mock.patch.dict(os.environ,DEEPINFRA_API_KEY='OFFLINE_REPLAY'),mock.patch.object(worker,'send',side_effect=replay):
    outcome=worker.run(sys.argv[1])
sys.exit(0 if outcome['status']=='success' else 1)
"""


def panel():
    files = {str(CAPTURES / name): digest for name, digest in PINS.items()}
    loop.check_files(files)
    workload = loop.read(CAPTURES / 'workload.json')
    binding, task = loop.study_tasks.load_task(loop.study_tasks.load_cohort(), 1818)
    profiles = []
    for number, name in ((13, 'benign-1818-direct'), (14, 'attack-1818-direct')):
        source = next(row for row in workload['rows'] if row['name'] == name)
        stem = f'{number}-generate-{name}'
        request = loop.read(CAPTURES / f'{stem}-request.json')['payload']
        response = loop.read(CAPTURES / f'{stem}-response.json')
        if request['prompt'] not in (source['prompt'], source['prompt_ids']):
            raise ValueError('Captured prompt differs from its exposed workload')
        if request['model'] != runtime.MODEL or response['http_status'] != 200:
            raise ValueError('Require the pinned successful provider captures')
        # The worker uses IDs; prior echo checks established the captured text's identical IDs.
        payload = dict(request, prompt=source['prompt_ids'])
        profiles.append(dict(name=name, source=source, payload=payload, response=response,
                             tokenizer_sha256=workload['tokenizer_files']))
    content = profiles[1]['source']['messages'][1]['content']
    question, tail = content.split('\n\nOriginal program:\n', 1)
    parent, tail = tail.split('\nShown examples and menu (JSON; inputs are exact decoded strings):\n', 1)
    shown, _ = json.JSONDecoder().raw_decode(tail)
    if question != 'Problem statement:\n' + task['question'] or set(shown) != {'examples', 'menu'}:
        raise ValueError('Historical attack context differs from task 1818')
    pairs = {}
    for group, cases in shown.items():
        pairs[group] = []
        for case in cases:
            matching = next(c for c in task['cases'] if c['input'] == case['input'] and c['answer'] == case['answer'])
            pairs[group].append(dict(matching, visible=True))
    pairs['program'] = parent
    files.update({str(p.resolve()): g.file_hash(p) for p in
        (*loop.source_files(), Path(__file__), Path(native_multi_edit.__file__), Path(attacks.__file__),
         Path(verification.__file__), runtime.WORKER, Path(runtime.__file__), ROOT / 'requirements-deepinfra.txt',
         ROOT / 'configs/sandbox/compose.yaml')})
    return binding, task, profiles, pairs, files


def summary(report):
    """Portable metadata only; detailed captures and sandbox logs remain local."""
    samples = []
    for run in report['runs']:
        for verdict in run.get('verification', []):
            cases = verdict.get('cases', [])
            samples.append(dict(name=run['name'], pipeline_status=run['status'],
                verification_status=verdict['status'], program_sha256=verdict['key']['program_sha256'],
                cases_checked=len(cases) if cases else report['task_case_count'],
                cases_passed=sum(c['status'] == 'passed' for c in cases) if cases else
                    report['task_case_count'] if verdict['status'] == 'passed' else None,
                replayed_usage=run['replayed_usage']))
    return {key: report[key] for key in ('kind', 'status', 'task_id', 'task_case_count', 'cohort_sha256',
        'new_api_calls', 'new_api_cost_usd', 'corpus_admission', 'main_recipe_qualified', 'notes')} | dict(
        capture_sha256=PINS, historical_parent_status=report.get('historical_parent', {}).get('status'),
        source_sha256={str(Path(path).relative_to(ROOT)): digest for path, digest in report['files'].items()
                      if Path(path).is_relative_to(ROOT) and not Path(path).is_relative_to(ROOT / 'data')},
        samples=samples, error=report.get('error'))


def run(output_dir, generation_python, tokenizer_path=DEFAULT_TOKENIZER):
    directory = Path(output_dir).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    binding, task, profiles, pairs, files = panel()
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    settings = dict(model=runtime.MODEL, tokenizer_path=str(tokenizer_path), tokenizer_sha256={
        name: g.file_hash(tokenizer_path / name) for name in runtime.TOKENIZER_FILES if (tokenizer_path / name).exists()},
        timeout_seconds=60, max_total_tokens=4096, input_usd_per_million=0,
        output_usd_per_million=0, max_cost_usd=.01, concurrency=1)
    runtime.tokenizer_files(settings)
    if any(settings['tokenizer_sha256'].get(name) != digest
           for name, digest in profiles[0]['tokenizer_sha256'].items()):
        raise ValueError('Tokenizer differs from the recorded compatibility workload')
    directory.mkdir(parents=True, exist_ok=False)
    profile_path = directory / 'offline-runtime.json'
    g.save(profile_path, settings)
    report = dict(kind='offline-deepinfra-integration-replay', status='running', task_id=1818, task_case_count=len(task['cases']),
        cohort_sha256=loop.study_tasks.COHORT_SHA256, files=files, runs=[], new_api_calls=0,
        new_api_cost_usd=0, corpus_admission=False, main_recipe_qualified=False,
        notes=['Token counts and estimated costs in replayed worker outcomes belong to historical captures.',
               'Zero planned rates are for this offline mock only, not provider prices.',
               'The attack uses its exact historical parent, not the separately generated benign capture.'])
    real_child = loop.run_child
    try:
        g.save(directory / 'report.json', report)
        report['historical_parent'] = verification.verify(task, binding['sha256'], pairs['program'].encode(),
                                                         directory / 'historical-parent-check')
        if report['historical_parent']['status'] != 'passed':
            raise RuntimeError('Historical parent did not pass verification')
        for profile in profiles:
            loop.check_files(files)
            name, payload, source = profile['name'], profile['payload'], profile['source']
            fixture = directory / f'{name}-capture.json'
            g.save(fixture, dict(payload=payload, response=profile['response']))
            def replay_child(command, state, persist):
                if len(command) != 3 or command[1] != str(runtime.WORKER):
                    raise ValueError('Unexpected generation command in offline replay')
                return real_child([command[0], '-c', CHILD, command[2], str(fixture)], state, persist)
            recipe = dict(system=source['messages'][0]['content'], initial='Replay captured prompt',
                base_seed=payload['seed'], max_repairs=0,
                **{k: payload[k] for k in ('temperature', 'top_p', 'top_k', 'max_tokens')},
                **{k: v for k, v in source['template_kwargs'].items() if k != 'add_generation_prompt'})
            extractor = checker = None
            if name.startswith('attack'):
                def extractor(row, attempt):
                    _, edits = native_multi_edit.parse_call(row['answer_text'])
                    return native_multi_edit.apply_edits(pairs['program'], edits)
                def checker(candidates, log_dir, **options):
                    return attacks.check_shown(candidates, log_dir, pairs={binding['sha256']: pairs},
                                              auto_witness=True, **options)
            record = dict(name=name, outcome=str(directory / name / 'outcome.json'),
                          replayed_usage=profile['response']['body']['usage'])
            report['runs'].append(record)
            with mock.patch.object(loop, 'run_child', side_effect=replay_child):
                outcome = loop.run([(binding, task)], directory / name, source='actor', recipe=recipe,
                    provider_runtime=profile_path, generation_python=generation_python,
                    reply_allowance=1, output_token_allowance=payload['max_tokens'],
                    cohort_sha256=loop.study_tasks.COHORT_SHA256, initial_messages=[source['messages']],
                    program_extractor=extractor, candidate_checker=checker)
            record.update(status=outcome['status'], tasks=outcome['tasks'],
                verification=[a['verification'] for a in outcome['attempts'] if 'verification' in a])
            g.save(directory / 'report.json', report)
            if outcome['status'] != 'completed':
                raise RuntimeError(f'Replay failed: {name}: {outcome.get("error")}')
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='error' if isinstance(error, Exception) else 'cancelled', error=repr(error))
        if not isinstance(error, Exception):
            raise
    finally:
        g.save(directory / 'report.json', report)
        g.save(directory / 'summary.json', summary(report))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    parser.add_argument('--generation-python', required=True)
    parser.add_argument('--tokenizer-path', default=str(DEFAULT_TOKENIZER))
    args = parser.parse_args()
    result = run(args.output_dir, args.generation_python, args.tokenizer_path)
    print(json.dumps(dict(status=result['status'], new_api_calls=result['new_api_calls'])))
    sys.exit(0 if result['status'] == 'completed' else 1)
