"""Replay ten captured Gemma controls offline; --live explicitly repeats selected controls."""
import argparse
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import deepinfra_generation as worker

FIXTURE = ROOT / 'tests/fixtures/deepinfra/gemma-live-controls.json'
FIXTURE_SHA256 = '151bec154160c47dbaadd56dfe079596445226b0aae0a7ea1e63c82e6b40fe51'
MODEL = worker.GEMMA_MODEL
PREFLIGHT = """import json,sys
from scripts import deepinfra_generation as w
spec=json.load(sys.stdin)
print(json.dumps(w.prepare(spec)[2]))
"""


def fixtures():
    raw = FIXTURE.read_bytes()
    if sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError('Changed controls fixture')
    data = json.loads(raw)
    for record in [item for case in data['cases'] for item in (case['request'], case['response'])] + list(data['native'].values()):
        if sha256(record['raw'].encode()).hexdigest() != record['sha256']:
            raise ValueError('Changed captured source bytes')
    for case in data['cases']:
        request = json.loads(case['request']['raw'])
        case.update(payload=request.get('payload', request), path=request.get('path', '/v1/openai/completions'))
        if case['payload']['model'] != MODEL or case['path'] not in ('/v1/openai/completions', '/v1/completions'):
            raise ValueError('Unpinned model or unsupported route')
    return data


def receipt_summary(receipt, row=None):
    if receipt['http_status'] != 200:
        raise ValueError(f"HTTP {receipt['http_status']}")
    body = receipt['body']
    if body.lstrip().startswith('data:'):
        frames = [line[5:].strip() for line in body.splitlines() if line.startswith('data:')]
        complete = bool(frames) and frames[-1] == '[DONE]'
        bodies = [json.loads(frame) for frame in frames if frame != '[DONE]']
    else:
        complete = True
        bodies = [json.loads(body)]
    usage = next((item['usage'] for item in reversed(bodies) if item.get('usage') is not None), None)
    if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
            for k in ('prompt_tokens', 'completion_tokens')) or not worker.positive(usage.get('estimated_cost'), zero=True)):
        raise ValueError('Missing valid provider usage')
    if row is not None:
        row['usage'] = usage
    choices = [choice for item in bodies for choice in (item.get('choices') or []) if isinstance(choice, dict)]
    return dict(usage=usage, stream_complete=complete, text=''.join(c.get('text', '') for c in choices),
                original_token_ids_available=any(c.get('token_ids') for c in choices),
                logprobs_available=any(c.get('logprobs') is not None for c in choices))


def exchange(case, key, timeout):
    request = urllib.request.Request('https://api.deepinfra.com' + case['path'],
        data=json.dumps(case['payload']).encode(), method='POST',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    started = monotonic()
    try:
        with urllib.request.build_opener(worker.NoRedirect).open(request, timeout=timeout) as response:
            status, body = response.status, response.read().decode('utf-8')
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read().decode('utf-8')
    return dict(http_status=status, body=body, elapsed_seconds=monotonic() - started)


def native_preflight(data, path, python, rates):
    runtime = json.loads(Path(path).read_bytes())
    worker.r.validate(runtime)
    original = json.loads(data['native']['runtime']['raw'])
    if (runtime['model'] != MODEL or runtime.get('output_protocol') != 'gemma_native_text'
            or runtime['tokenizer_sha256'] != original['tokenizer_sha256']):
        raise ValueError('Native control requires the captured Gemma tokenizer and native-text protocol')
    worker.r.tokenizer_files(runtime)
    runtime.update(input_usd_per_million=rates[0], output_usd_per_million=rates[1], concurrency=1)
    inputs = json.loads(data['native']['inputs']['raw'])
    request = dict(inputs['recipe'], messages=inputs['initial'][0], seed=inputs['recipe']['base_seed'])
    request = {k: v for k, v in request.items() if k not in ('system', 'initial', 'base_seed', 'max_repairs')}
    spec = dict(runtime=runtime, requests=[request], output_dir='UNUSED_OFFLINE_PREFLIGHT')
    result = subprocess.run([python, '-c', PREFLIGHT], input=json.dumps(spec), text=True,
        capture_output=True, check=True, cwd=ROOT, env={k: v for k, v in os.environ.items() if k != 'DEEPINFRA_API_KEY'})
    if json.loads(result.stdout) != [data['cases'][-1]['payload']]:
        raise ValueError('Native pipeline would send a different payload')
    return runtime


def native_pipeline(data, runtime, python, directory):
    loop = importlib.import_module('src.data-pipeline.stage03.generation_loop')
    from scripts import native_multi_edit
    inputs = json.loads(data['native']['inputs']['raw'])
    entries = [dict(benigns=dict(actor=dict(program='print(1)')), partition=dict(menu=[]))]
    extractor = native_multi_edit.Extractor(entries, inputs['initial'], loop)
    return loop.run(inputs['worklist'], directory / 'pipeline', source='synthetic', fixtures=True,
        recipe=inputs['recipe'], initial_messages=inputs['initial'], reply_allowance=1,
        output_token_allowance=inputs['recipe']['max_tokens'], provider_runtime=runtime,
        generation_python=python, program_extractor=extractor)


def run(output_dir, *, live=False, cases=None, input_rate=None, output_rate=None, max_cost=None,
        runtime=None, generation_python=None, timeout=300):
    data = fixtures()
    selected = cases or [case['id'] for case in data['cases']]
    available = {case['id']: case for case in data['cases']}
    if len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError('Select distinct known control IDs')
    selected = [available[name] for name in selected]
    settings, planned = None, 0
    if live:
        if (not all(worker.positive(rate, zero=True) for rate in (input_rate, output_rate))
                or not worker.positive(max_cost) or not worker.positive(timeout)):
            raise ValueError('Live mode requires finite rates, positive maximum cost and socket timeout')
        if any(case['mode'] == 'pipeline' for case in selected):
            if not runtime or not generation_python:
                raise ValueError('Native pipeline requires --runtime and --generation-python')
            settings = native_preflight(data, runtime, generation_python, (input_rate, output_rate))
        for case in selected:
            prompt = case['payload']['prompt']
            bound = (settings['max_total_tokens'] if case['mode'] == 'pipeline' else
                     len(prompt) if isinstance(prompt, list) else 262144)
            case['reservation_usd'] = (bound * input_rate + case['payload']['max_tokens'] * output_rate) / 1e6
            planned += case['reservation_usd']
        if not worker.positive(planned, zero=True) or planned > max_cost:
            raise ValueError('Selected controls exceed the entire uncached spending ceiling')
    key = os.environ.get('DEEPINFRA_API_KEY', '') if live else ''
    if live and (not key or any(c.isspace() for c in key)):
        raise ValueError('Set DEEPINFRA_API_KEY in the coordinator environment')
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(kind='gemma-live-controls-v1', live=live, fixture_sha256=FIXTURE_SHA256,
        input_usd_per_million=input_rate, output_usd_per_million=output_rate, max_cost_usd=max_cost,
        source_sha256={str(p.relative_to(ROOT)): worker.g.file_hash(p) for p in
            (Path(__file__).resolve(), Path(worker.__file__), ROOT / 'requirements-deepinfra.txt')},
        planned_max_cost_usd=planned, status='running', new_api_calls=0, rows=[
            dict(id=case['id'], submitted=False, status='not_run') for case in selected])

    def save(path, value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text.replace(key, '<redacted>') if key else text)
        temporary.replace(path)

    def persist():
        report.update(usage_complete=all(row['submitted'] is False or 'usage' in row for row in report['rows']),
            new_api_calls=sum(row['submitted'] is True for row in report['rows']) if live else 0,
            unknown_submissions=sum(row['submitted'] is None for row in report['rows']),
            observed_estimated_cost_usd=sum(row.get('usage', {}).get('estimated_cost', 0) for row in report['rows']))
        save(directory / 'report.json', report)

    persist()
    for case, row in zip(selected, report['rows'], strict=True):
        target = directory / case['id']
        target.mkdir()
        try:
            save(target / 'request.json', dict(path=case['path'], payload=case['payload']))
            row.update(status='running', submitted=None if live else True)
            persist()
            if live:
                if case['mode'] == 'pipeline':
                    settings['max_cost_usd'] = case['reservation_usd']
                    save(target / 'runtime.json', settings)
                    outcome = native_pipeline(data, target / 'runtime.json', generation_python, target)
                    row['pipeline_status'] = outcome['status']
                    receipt = json.loads((target / 'pipeline/generation-0000/response-0000.json').read_bytes())
                else:
                    receipt = exchange(case, key, timeout)
            else:
                receipt = json.loads(case['response']['raw'])
            if receipt['http_status'] == 200:
                row['submitted'] = True
            save(target / 'response.json', receipt)
            row.update(receipt_summary(receipt, row), status='success')
            if not row['stream_complete']:
                raise ValueError('Incomplete SSE stream')
            if live and case['mode'] == 'pipeline':
                attempt = outcome['attempts'][0]
                row.update(extraction=attempt.get('extraction'), verification=attempt.get('verification'), candidate=attempt.get('candidate'))
                if (outcome['status'] != 'completed' or attempt.get('extraction') != 'success'
                        or attempt.get('verification', {}).get('status') != 'failed'
                        or attempt.get('candidate', {}).get('program_sha256') != sha256(b'print(2)').hexdigest()):
                    raise ValueError('Native control did not extract the intended fault and reject it')
            if live and sum(r.get('usage', {}).get('estimated_cost', 0) for r in report['rows']) > max_cost:
                raise ValueError('Provider reported cost exceeds the supplied ceiling; stopping')
        except BaseException as error:
            saved = target / 'pipeline/generation-0000/outcome.json'
            if live and case['mode'] == 'pipeline' and saved.exists():
                native = json.loads(saved.read_bytes())['requests'][0]
                row['submitted'] = native['submitted']
                if native.get('usage') is not None:
                    row['usage'] = native['usage']
            row.update(status='error', error=repr(error))
            report['status'] = 'error'
            persist()
            break
        persist()
    else:
        report['status'] = 'success'
    report['new_api_cost_usd'] = report['observed_estimated_cost_usd'] if live else 0
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--cases', nargs='+')
    for flag in ('input-rate', 'output-rate', 'max-cost', 'timeout'):
        parser.add_argument('--' + flag, type=float, default=300 if flag == 'timeout' else None)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--generation-python')
    args = vars(parser.parse_args())
    sys.exit(0 if run(**args)['status'] == 'success' else 1)


if __name__ == '__main__':
    main()
