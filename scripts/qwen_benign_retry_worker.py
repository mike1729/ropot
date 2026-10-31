"""Explicit benign retry: retain completed requests and bound each Metal worker."""
import copy
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qwen_dev_benigns as d


def reusable(requests, previous_requests, previous_rows):
    if requests != previous_requests or len(previous_rows) != len(requests):
        raise ValueError('Retry initial requests must exactly match the failed acquisition')
    return {i: copy.deepcopy(row) for i, row in enumerate(previous_rows)
            if row['status'] in ('stop', 'length') and row['submitted']}


def run(round_file):
    spec = d.loop.read(round_file)
    output = Path(spec['output_dir'])
    root = output.parents[2]
    contract = d.loop.read(root / 'contract.json')
    rows = [dict(status='not_run', submitted=False, generation_tokens=0, text='', answer_text=None)
            for _ in spec['requests']]
    result = dict(status='error', requests=rows, native={}, reused=[], chunks=[])
    output.mkdir(exist_ok=False)
    def persist():
        d.loop.atomic_save(output / 'outcome.json', result)
    try:
        d.loop.check_files(contract['sources'])
        if 'retry' in contract and Path(round_file).name == 'round-0000.json':
            source = Path(contract['retry']['source']) / 'acquisition/actor/generation-0000'
            inputs, outcome = source / 'inputs.json', source / 'outcome.json'
            if not inputs.exists():
                inputs = source.parent / 'round-0000.json'
            saved = reusable(spec['requests'], d.loop.read(inputs)['requests'], d.loop.read(outcome)['requests'])
            result['reuse_sources'] = {str(p): d.g.file_hash(p) for p in (inputs, outcome)}
            for i, row in saved.items():
                rows[i] = row
                (output / f'reply-{i:04d}.txt').write_text(row['text'])
            result['reused'] = sorted(saved)
        pending = [i for i, row in enumerate(rows) if row['status'] == 'not_run']
        persist()
        size = contract.get('worker_request_limit', contract.get('retry', {}).get('worker_request_limit', 4))
        for offset in range(0, len(pending), size):
            indices = pending[offset:offset + size]
            chunk_output = output.with_name(output.name + f'-part-{offset:04d}')
            chunk_file = chunk_output.with_suffix('.json')
            d.g.save(chunk_file, dict(spec, requests=[spec['requests'][i] for i in indices],
                                    output_dir=str(chunk_output)))
            chunk = dict(indices=indices, specification=str(chunk_file), sha256=d.g.file_hash(chunk_file))
            result['chunks'].append(chunk)
            persist()
            # Same process group as the owner so interruption also stops its worker.
            child = subprocess.Popen([sys.executable, str(d.a.DRIVER), str(chunk_file)])
            chunk['pid'] = child.pid
            persist()
            chunk['returncode'] = child.wait()
            native = d.loop.read(chunk_output / 'outcome.json')
            for i, row in zip(indices, native['requests'], strict=True):
                rows[i] = row
                if row['submitted']:
                    (output / f'reply-{i:04d}.txt').write_text(row['text'])
            chunk['native'] = native.get('native')
            persist()
            if chunk['returncode'] or native['status'] != 'success':
                raise RuntimeError(f"Chunk failed without retry: {native.get('error')}")
        result.update(status='success', native=dict(
            prompt_tokens=sum((c['native'] or {}).get('prompt_tokens', 0) for c in result['chunks']),
            generation_tokens=sum(r['generation_tokens'] for r in rows),
            reused_generation_tokens=sum(rows[i]['generation_tokens'] for i in result['reused']),
            wall_time=sum((c['native'] or {}).get('wall_time', 0) for c in result['chunks'])))
    except BaseException as error:
        result.update(status='error', error=repr(error))
        raise
    finally:
        persist()
    return 0


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
