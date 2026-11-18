"""Bounded raw-completion worker; see docs/support/deepinfra-worker.md."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import monotonic
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
g = importlib.import_module('src.data-pipeline.stage03.local_generation')
r = importlib.import_module('src.data-pipeline.stage03.deepinfra_runtime')
MODEL, EOS, GEMMA_MODEL, TOKENIZER_FILES, positive = r.MODEL, r.EOS, r.GEMMA_MODEL, r.TOKENIZER_FILES, r.positive


def check_versions():
    pins = dict(line.strip().split('==') for line in (r.ROOT / 'requirements-deepinfra.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#'))
    actual = {name: version(name) for name in pins}
    if actual != pins:
        raise RuntimeError(f'Tokenizer package versions differ from requirements: {actual}')
    return actual


class Tokenizer:
    """Expose the existing rendering/thought-parser protocol without importing MLX."""
    has_thinking = True
    def __init__(self, inner, versions=None, *, profile=None):
        self.inner = inner
        self.profile = r.model_profile(MODEL) if profile is None else profile
        self.eos_token_ids = self.profile['eos_token_ids']
        self.think_start_tokens, self.think_end_tokens = self.profile['think_start_tokens'], self.profile['think_end_tokens']
        self.versions = check_versions() if versions is None else versions

    def apply_chat_template(self, messages, **options):
        return self.inner.apply_chat_template(messages, return_dict=False, **options)

    def rfind_think_start(self, tokens):
        return self.rfind(tokens, self.think_start_tokens)

    def rfind_think_end(self, tokens):
        return self.rfind(tokens, self.think_end_tokens)

    @staticmethod
    def rfind(tokens, sequence):
        return next((i for i in range(len(tokens) - len(sequence), -1, -1)
                     if tokens[i:i + len(sequence)] == list(sequence)), -1)

    def convert_ids_to_tokens(self, token):
        return self.inner.convert_ids_to_tokens(token)

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
    versions = check_versions()
    from transformers import AutoTokenizer
    r.tokenizer_files(runtime)
    root = Path(runtime['tokenizer_path'])
    inner = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config = json.loads((root / 'config.json').read_text())
    profile = r.model_profile(runtime.get('model', MODEL))
    eos = config.get('eos_token_id', config.get('text_config', {}).get('eos_token_id'))
    eos = eos if isinstance(eos, list) else [eos]
    if (config.get('model_type') not in profile['model_types']
            or set(eos + [inner.eos_token_id]) != set(profile['eos_token_ids'])
            or inner.encode(profile['think_start_text'], add_special_tokens=False) != list(profile['think_start_tokens'])
            or inner.encode(profile['think_end_text'], add_special_tokens=False) != list(profile['think_end_tokens'])):
        raise ValueError('Tokenizer family or boundaries differ from the supported model')
    return Tokenizer(inner, versions, profile=profile)


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


def prepare(specification, *, tokenizer=None):
    runtime, requests = specification['runtime'], specification['requests']
    r.validate(runtime)
    profile = r.model_profile(runtime['model'])
    if (set(specification) != {'runtime', 'requests', 'output_dir'}
            or not isinstance(requests, list) or not requests):
        raise ValueError('Require an explicit pinned runtime and finite ordered requests')
    tokenizer = load_tokenizer(runtime) if tokenizer is None else tokenizer
    started = monotonic()
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
        if profile['source'] == 'synthetic' and ('reasoning_effort' in request or request.get('preserve_thinking') is False):
            raise ValueError('Gemma uses boolean thinking and its native tool-history retention policy')
        row = g.render(tokenizer, request['messages'], **{k: request[k] for k in
            ('enable_thinking', 'reasoning_effort', 'tools', 'preserve_thinking') if k in request})
        if (row['prompt_tokens'] > request.get('max_prompt_tokens', runtime['max_total_tokens'])
                or row['prompt_tokens'] + request['max_tokens'] > runtime['max_total_tokens']):
            raise ValueError('Prompt exceeds the configured token envelope')
        planned += (row['prompt_tokens'] * runtime['input_usd_per_million']
                    + request['max_tokens'] * runtime['output_usd_per_million']) / 1e6
        payloads.append(dict(model=runtime['model'], prompt=row['prompt_ids'], stop_token_ids=list(profile['eos_token_ids']),
            skip_special_tokens=False, return_tokens_as_token_ids=True, echo=False, **{k: request[k] for k in
            ('seed', 'temperature', 'top_p', 'max_tokens', 'top_k') if k in request}))
        row.update(status='not_run', submitted=False, text='', tokens=[], answer_text=None,
                   generation_tokens=0, finish_reason=None, elapsed_seconds=None,
                   output_protocol=runtime.get('output_protocol', 'token_ids'))
        rows.append(row)
    if planned > runtime['max_cost_usd']:
        raise ValueError('Round exceeds the planned uncached spending ceiling')
    return tokenizer, rows, payloads, planned, monotonic() - started


def split_gemma_text(row, tokenizer):
    """Explicit native framing, never an inferred generated-token sequence."""
    opening, closing = '<|channel>thought', '<channel|>'
    text = row['text']
    opened = tokenizer.rfind_think_start(row['prompt_ids']) > tokenizer.rfind_think_end(row['prompt_ids'])
    row.update(tokens=None, token_ids_source='unavailable', answer_text=None, reasoning_token_count=None)
    metrics = dict(status='unavailable', framing='malformed', reasoning_tokens=None, answer_tokens=None)
    row['thought_metrics'] = metrics
    if opened or text.startswith(opening):
        content = text if opened else text[len(opening):]
        if not opened and content and not content.startswith(('\n', '\r\n', closing)):
            return
        if '<|channel>' in content or content.count(closing) > 1:
            return
        if closing not in content:
            metrics.update(framing='unclosed', reasoning_characters=len(content), answer_characters=0)
            return
        reasoning, answer = content.split(closing, 1)
        if not row['enable_thinking'] and reasoning.strip():
            metrics['framing'] = 'unexpected_thought'
            return
        row.update(reasoning_content=reasoning, answer_text=answer)
        metrics.update(framing='closed', reasoning_characters=len(reasoning), answer_characters=len(answer))
    elif '<|channel>' not in text and closing not in text:
        row['answer_text'] = text
        metrics.update(framing='absent', reasoning_characters=0, answer_characters=len(text))


def normalize(body, row, tokenizer, cap):
    usage = body['usage']
    if (any(type(usage[k]) is not int or usage[k] < 0 for k in ('prompt_tokens', 'completion_tokens'))
            or not positive(usage['estimated_cost'], zero=True)):
        raise ValueError('Malformed provider usage')
    # Billing evidence remains valid even when token IDs or completion text are unusable.
    row.update(generation_tokens=usage['completion_tokens'], usage=usage)
    choice = body['choices'][0]
    tokens, reason = choice.get('token_ids'), choice['finish_reason']
    if (len(body['choices']) != 1 or choice['index'] != 0 or reason not in ('stop', 'length')
            or not isinstance(choice['text'], str) or usage['completion_tokens'] > cap
            or usage['prompt_tokens'] != row['prompt_tokens']):
        raise ValueError('Malformed completion or inconsistent usage')
    if row.get('output_protocol') == 'gemma_native_text' and (tokens is None or tokens == []):
        if set(tokenizer.eos_token_ids) != set(r.model_profile(GEMMA_MODEL)['eos_token_ids']):
            raise ValueError('Native-text framing requires the Gemma tokenizer')
        row.update(text=choice['text'], finish_reason=reason, status=reason)
        split_gemma_text(row, tokenizer)
        return
    if (not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens)
            or usage['completion_tokens'] != len(tokens)):
        raise ValueError('Missing or inconsistent original token IDs')
    row.update(tokens=tokens, token_ids_source='provider', finish_reason=reason)
    text = g.decode_tokens(tokenizer, tokens, set(tokenizer.eos_token_ids))
    if choice['text'] != text:
        raise ValueError('Provider text differs from local token decoding')
    row['text'] = text
    g.split_answer(tokenizer, row)
    row['status'] = reason


def exchange(payload, key, timeout):
    """Only network I/O runs in pool threads; evidence and decoding stay with the owner."""
    started = monotonic()
    try:
        status, raw = send(payload, key, timeout)
        return status, raw, monotonic() - started, None
    except BaseException as error:
        return None, None, monotonic() - started, error


def generate(rows, payloads, tokenizer, runtime, key, directory, save, persist, result):
    active, cursor, primary = {}, 0, None
    limit = runtime.get('concurrency', 1)
    result.update(concurrency=limit, max_observed_in_flight=0)

    def failed(error, index=None):
        nonlocal primary
        if primary is None:
            primary = error
            result.update(status='error' if isinstance(error, Exception) else 'cancelled',
                          error=repr(error).replace(key, '<redacted>'))
        if index is not None:
            row = rows[index]
            row['error'] = repr(error).replace(key, '<redacted>')
            if row['status'] == 'running':
                row['status'] = 'error' if isinstance(error, Exception) else 'cancelled'

    with ThreadPoolExecutor(max_workers=limit) as pool:
        while active or (primary is None and cursor < len(rows)):
            try:
                # Settle every ready peer before admitting more work.
                if primary is None and not any(f.done() for f in active):
                    while cursor < len(rows) and len(active) < limit:
                        if any(f.done() for f in active):
                            break
                        index, row = cursor, rows[cursor]
                        save(directory / f'request-{index:04d}.json', payloads[index])
                        row.update(submitted=None, status='running')
                        try:
                            persist()
                        except BaseException:
                            row.update(submitted=False, status='not_run')  # No task was submitted.
                            raise
                        future = pool.submit(exchange, payloads[index], key, runtime['timeout_seconds'])
                        active[future] = index
                        cursor += 1
                        result['max_observed_in_flight'] = max(result['max_observed_in_flight'], len(active))
                if not active:
                    break
                done = {f for f in active if f.done()} or wait(active, return_when=FIRST_COMPLETED)[0]
                for future in sorted(done, key=lambda f: active[f]):
                    index, row = active[future], rows[active[future]]
                    try:
                        if future.cancelled():
                            row.update(submitted=False, status='not_run')
                            continue
                        status, raw, elapsed, error = future.result()
                        row['elapsed_seconds'] = elapsed
                        if error is not None:
                            raise error
                        if status == 200:
                            row['submitted'] = True
                        save(directory / f'response-{index:04d}.json', dict(http_status=status,
                            body=raw, elapsed_seconds=elapsed))
                        if status != 200:
                            raise RuntimeError(f'HTTP {status}; submission/usage remains uncertain')
                        normalize(json.loads(raw), row, tokenizer, payloads[index]['max_tokens'])
                        (directory / f'reply-{index:04d}.txt').write_text(
                            row['text'].replace(key, '<redacted>'), encoding='utf-8', newline='')
                    except BaseException as error:
                        failed(error, index)
                    finally:
                        del active[future]
                        persist()
            except BaseException as error:
                failed(error)
            if primary is not None:
                # Cancel only work proven not to have started; collect running peers normally.
                try:
                    for future, index in list(active.items()):
                        if future.cancel():
                            rows[index].update(submitted=False, status='not_run')
                            del active[future]
                    persist()
                except BaseException as error:
                    failed(error)
        if primary is not None:
            raise primary


def run(round_file):
    specification = json.loads(Path(round_file).read_text())
    directory = Path(specification['output_dir'])
    if directory.exists():
        raise FileExistsError(directory)
    if not isinstance(specification.get('requests'), list):
        raise ValueError('Require an ordered request list to record submission states')
    key = os.environ.get('DEEPINFRA_API_KEY', '')
    rows = [dict(status='not_run', submitted=False, text='', tokens=[], answer_text=None,
                 generation_tokens=0, finish_reason=None, elapsed_seconds=None) for _ in specification['requests']]
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(status='error', requests=rows, native=None, error=None, usage_complete=True,
                  preparation_seconds=None, planned_max_cost_usd=None)

    def save(path, value):
        temporary = path.with_suffix(path.suffix + '.tmp')
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        temporary.write_text(text.replace(key, '<redacted>') if key else text, encoding='utf-8')
        temporary.replace(path)

    def persist():
        result.update(elapsed_seconds=monotonic() - started,
            usage_complete=all(r['submitted'] is False or r.get('usage') is not None for r in rows),
            observed_estimated_cost_usd=sum(r.get('usage', {}).get('estimated_cost', 0) for r in rows))
        save(directory / 'outcome.json', result)

    started = monotonic()
    try:
        persist()
        save(directory / 'inputs.json', specification)
        metadata = dict(load_seconds=None, versions={}, python=sys.version,
            source_sha256={str(p): g.file_hash(p) for p in
                (Path(__file__).resolve(), Path(g.__file__).resolve(), Path(r.__file__).resolve(),
                 r.ROOT / 'requirements-deepinfra.txt')}, provider_quantization=None, provider_weight_revision=None)
        save(directory / 'runtime.json', metadata)
        if not key or any(c.isspace() for c in key):
            raise ValueError('Set DEEPINFRA_API_KEY in the coordinator/worker environment')
        runtime = specification['runtime']
        r.validate(runtime)
        profile = r.model_profile(runtime['model'])
        metadata.update(provider_model=runtime['model'], provider_quantization=profile['quantization'])
        loading = monotonic()
        tokenizer = load_tokenizer(runtime)
        metadata.update(load_seconds=monotonic() - loading, versions=getattr(tokenizer, 'versions', {}))
        save(directory / 'runtime.json', metadata)
        _, prepared, payloads, planned, preparation = prepare(specification, tokenizer=tokenizer)
        rows[:] = prepared
        result.update(preparation_seconds=preparation, planned_max_cost_usd=planned)
        persist()
        generate(rows, payloads, tokenizer, runtime, key, directory, save, persist, result)
        result['status'] = 'success'
    except BaseException as error:
        message = repr(error)
        result.update(status='error' if isinstance(error, Exception) else 'cancelled',
                      error=message.replace(key, '<redacted>') if key else message)
        for row in rows:
            if row['status'] == 'running':
                row['status'] = result['status']
    finally:
        persist()
    return result


if __name__ == '__main__':
    sys.exit(0 if run(sys.argv[1])['status'] == 'success' else 1)
