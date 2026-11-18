"""Serial raw-completion worker; see docs/support/deepinfra-worker.md."""
import importlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
g = importlib.import_module('src.data-pipeline.stage03.local_generation')
MODEL = 'Qwen/Qwen3.8-27B:61nNTQpO'
EOS = (248046, 248044)
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                   'vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json')


class Tokenizer:
    """Expose the existing rendering/thought-parser protocol without importing MLX."""
    has_thinking = True
    eos_token_ids = EOS
    think_start_tokens, think_end_tokens = (248068,), (248069,)

    def __init__(self, inner):
        self.inner = inner
        self.versions = {name: version(name) for name in ('transformers', 'tokenizers', 'jinja2')}

    def apply_chat_template(self, messages, **options):
        return self.inner.apply_chat_template(messages, return_dict=False, **options)

    def rfind_think_start(self, tokens):
        return max((i for i, token in enumerate(tokens) if token == self.think_start_tokens[0]), default=-1)

    def rfind_think_end(self, tokens):
        return max((i for i, token in enumerate(tokens) if token == self.think_end_tokens[0]), default=-1)

    @property
    def detokenizer(self):
        inner = self.inner
        class Decoder:
            def __init__(self):
                self.tokens, self.text = [], ''

            def add_token(self, token):
                self.tokens.append(token)

            def finalize(self):
                self.text = inner.decode(self.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return Decoder()


def load_tokenizer(runtime):
    from transformers import AutoTokenizer
    root, hashes = Path(runtime['tokenizer_path']), runtime['tokenizer_sha256']
    present = {name for name in TOKENIZER_FILES if (root / name).exists()}
    if not set(TOKENIZER_FILES[:4]) <= present or set(hashes) != present:
        raise ValueError('Pin every tokenizer input, including the template and config')
    if any(g.file_hash(root / name) != digest for name, digest in hashes.items()):
        raise ValueError('Tokenizer hash mismatch')
    inner = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config = json.loads((root / 'config.json').read_text())
    eos = config.get('text_config', config)['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    if (set(eos + [inner.eos_token_id]) != set(EOS)
            or inner.encode('<think>', add_special_tokens=False) != [248068]
            or inner.encode('</think>', add_special_tokens=False) != [248069]):
        raise ValueError('Tokenizer boundaries differ from the supported actor')
    return Tokenizer(inner)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('Unexpected API redirect')


def send(payload, key, timeout):
    request = urllib.request.Request('https://api.deepinfra.com/v1/openai/completions',
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            return response.status, response.read().decode('utf-8')
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode('utf-8')


def positive(value, *, zero=False):
    return type(value) in (int, float) and math.isfinite(value) and (value >= 0 if zero else value > 0)


def prepare(specification):
    runtime, requests = specification['runtime'], specification['requests']
    fields = {'model', 'tokenizer_path', 'tokenizer_sha256', 'timeout_seconds', 'max_total_tokens',
              'input_usd_per_million', 'output_usd_per_million', 'max_cost_usd'}
    if (set(specification) != {'runtime', 'requests', 'output_dir'} or set(runtime) != fields
            or runtime['model'] != MODEL or not isinstance(requests, list) or not requests
            or type(runtime['max_total_tokens']) is not int or runtime['max_total_tokens'] <= 0
            or not all(positive(runtime[k]) for k in ('timeout_seconds', 'max_cost_usd'))
            or not all(positive(runtime[k], zero=True) for k in ('input_usd_per_million', 'output_usd_per_million'))):
        raise ValueError('Require an explicit pinned runtime and finite ordered requests')
    tokenizer, started = load_tokenizer(runtime), monotonic()
    rows, payloads, planned = [], [], 0
    required = {'messages', 'seed', 'temperature', 'top_p', 'max_tokens'}
    optional = {'max_prompt_tokens', 'top_k', 'enable_thinking', 'reasoning_effort', 'tools', 'preserve_thinking'}
    for request in requests:
        if (not required <= set(request) or set(request) - required - optional
                or not isinstance(request['messages'], list) or not request['messages']
                or any(not isinstance(m, dict) or not isinstance(m.get('role'), str) for m in request['messages'])
                or type(request['seed']) is not int or not 0 <= request['seed'] < 2**64
                or not positive(request['temperature'], zero=True) or request['temperature'] > 2
                or not positive(request['top_p']) or request['top_p'] > 1
                or any(type(request[k]) is not int or request[k] <= 0
                       for k in ('max_tokens', 'max_prompt_tokens') if k in request)
                or type(request.get('enable_thinking', False)) is not bool
                or 'top_k' in request and (type(request['top_k']) is not int or request['top_k'] < 0)):
            raise ValueError('Invalid generation request; parameters cannot be silently dropped')
        row = g.render(tokenizer, request['messages'], **{k: request[k] for k in
            ('enable_thinking', 'reasoning_effort', 'tools', 'preserve_thinking') if k in request})
        if (row['prompt_tokens'] > request.get('max_prompt_tokens', runtime['max_total_tokens'])
                or row['prompt_tokens'] + request['max_tokens'] > runtime['max_total_tokens']):
            raise ValueError('Prompt exceeds the configured token envelope')
        planned += (row['prompt_tokens'] * runtime['input_usd_per_million']
                    + request['max_tokens'] * runtime['output_usd_per_million']) / 1e6
        payloads.append(dict(model=MODEL, prompt=row['prompt_ids'], stop_token_ids=list(EOS),
            skip_special_tokens=False, return_tokens_as_token_ids=True, echo=False, **{k: request[k] for k in
            ('seed', 'temperature', 'top_p', 'max_tokens', 'top_k') if k in request}))
        row.update(status='not_run', submitted=False, text='', tokens=[], answer_text=None,
                   generation_tokens=0, finish_reason=None, elapsed_seconds=None)
        rows.append(row)
    if planned > runtime['max_cost_usd']:
        raise ValueError('Round exceeds the planned uncached spending ceiling')
    return tokenizer, rows, payloads, planned, monotonic() - started


def normalize(body, row, tokenizer, cap):
    choice, usage = body['choices'][0], body['usage']
    tokens, reason = choice['token_ids'], choice['finish_reason']
    if (len(body['choices']) != 1 or choice['index'] != 0 or reason not in ('stop', 'length')
            or not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens)
            or len(tokens) > cap or type(usage['completion_tokens']) is not int
            or usage['completion_tokens'] != len(tokens) or usage['prompt_tokens'] != row['prompt_tokens']
            or not positive(usage['estimated_cost'], zero=True)):
        raise ValueError('Malformed completion or inconsistent usage')
    row.update(tokens=tokens, generation_tokens=len(tokens), finish_reason=reason, usage=usage)
    text = g.decode_tokens(tokenizer, tokens, set(EOS))
    if choice['text'] != text:
        raise ValueError('Provider text differs from local token decoding')
    row['text'] = text
    g.split_answer(tokenizer, row)
    row['status'] = reason


def run(round_file):
    specification = json.loads(Path(round_file).read_text())
    directory = Path(specification['output_dir'])
    if directory.exists():
        raise FileExistsError(directory)
    loading = monotonic()
    tokenizer, rows, payloads, planned, preparation = prepare(specification)
    key = os.environ.get('DEEPINFRA_API_KEY', '')
    if not key or any(c.isspace() for c in key):
        raise ValueError('Set DEEPINFRA_API_KEY in the worker environment')
    runtime = specification['runtime']
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(status='error', requests=rows, native=None, error=None, usage_complete=True,
                  preparation_seconds=preparation, planned_max_cost_usd=planned)

    def save(path, value):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False).replace(key, '<redacted>'), encoding='utf-8')
        temporary.replace(path)

    def persist():
        result.update(elapsed_seconds=monotonic() - started,
            usage_complete=all(r['submitted'] is False or r.get('usage') is not None for r in rows),
            observed_estimated_cost_usd=sum(r.get('usage', {}).get('estimated_cost', 0) for r in rows))
        save(directory / 'outcome.json', result)

    load_seconds = monotonic() - loading - preparation
    started, current = monotonic(), None
    try:
        persist()
        save(directory / 'inputs.json', specification)
        save(directory / 'runtime.json', dict(load_seconds=load_seconds,
            versions=getattr(tokenizer, 'versions', {}), python=sys.version,
            source_sha256={str(p): g.file_hash(p) for p in (Path(__file__).resolve(), Path(g.__file__).resolve())},
            provider_quantization=None, provider_weight_revision=None))
        for index, (row, payload) in enumerate(zip(rows, payloads)):
            current = row
            save(directory / f'request-{index:04d}.json', payload)
            row.update(submitted=None, status='running')
            persist()
            sent = monotonic()
            status, raw = send(payload, key, runtime['timeout_seconds'])
            row['elapsed_seconds'] = monotonic() - sent
            if status == 200:
                row['submitted'] = True
            save(directory / f'response-{index:04d}.json', dict(http_status=status,
                body=raw, elapsed_seconds=row['elapsed_seconds']))
            if status != 200:
                raise RuntimeError(f'HTTP {status}; submission/usage remains uncertain')
            normalize(json.loads(raw), row, tokenizer, payload['max_tokens'])
            (directory / f'reply-{index:04d}.txt').write_text(row['text'].replace(key, '<redacted>'), encoding='utf-8', newline='')
            persist()
        result['status'] = 'success'
    except BaseException as error:
        result.update(status='error' if isinstance(error, Exception) else 'cancelled', error=repr(error).replace(key, '<redacted>'))
        if current is not None and current['status'] == 'running':
            current['status'] = result['status']
    finally:
        persist()
    return result


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1])['status'] == 'success' else 1)
