"""Run one generation round as one-batch guarded worker processes.

Long rounds in one Metal process hit the resource-count limit in the dev-50 benign
acquisition. Runs 03-09 used at most eight requests per process. Each chunk holds at most
REQUEST_LIMIT requests and runs in a
fresh v2_attack_generation process, which keeps its own memory guard and trace.
Chunk outcomes are merged in request order; a failed chunk stops the round without retry.
"""
import json
from pathlib import Path
import subprocess
import sys

WORKER = Path(__file__).with_name('v2_attack_generation.py')
REQUEST_LIMIT = 8


def run(round_file):
    specification = json.loads(Path(round_file).read_text())
    output = Path(specification['output_dir'])
    requests, size = specification['requests'], REQUEST_LIMIT
    rows = [dict(status='not_run', submitted=False, generation_tokens=0, text='', answer_text=None) for _ in requests]
    result = dict(status='error', requests=rows, native=None, chunks=[], error=None)
    output.mkdir(exist_ok=False)
    def persist():
        temporary = output / 'outcome.json.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
        temporary.replace(output / 'outcome.json')
    try:
        for offset in range(0, len(requests), size):
            chunk_output = output.with_name(f'{output.name}-part-{offset:04d}')
            chunk_file = chunk_output.with_suffix('.json')
            chunk_file.write_text(json.dumps(dict(specification, requests=requests[offset:offset + size],
                                                  output_dir=str(chunk_output)), ensure_ascii=False))
            chunk = dict(offset=offset, specification=str(chunk_file), output=str(chunk_output))
            result['chunks'].append(chunk)
            persist()
            # Same process group as the owner, so interrupting the round also stops the worker.
            chunk['returncode'] = subprocess.run([sys.executable, str(WORKER), str(chunk_file)]).returncode
            path = chunk_output / 'outcome.json'
            native = json.loads(path.read_text()) if path.exists() else dict(status='error', requests=[])
            for index, row in enumerate(native['requests']):
                rows[offset + index] = row
            chunk['native'] = native.get('native')
            persist()
            if chunk['returncode'] or native['status'] != 'success':
                raise RuntimeError(f"Chunk at {offset} failed without retry: {native.get('error')}")
        counters = [c['native'] or {} for c in result['chunks']]
        result.update(status='success', native={k: sum(c.get(k, 0) for c in counters) for k in
            ('prompt_tokens', 'generation_tokens', 'prompt_time', 'decode_time', 'wall_time')},
            peak_memory_bytes=max(json.loads((Path(c['output']) / 'outcome.json').read_text()).get(
                'peak_memory_bytes') or 0 for c in result['chunks']))
    except BaseException as error:
        result.update(status='error', error=repr(error))
        raise
    finally:
        persist()
    return 0


if __name__ == '__main__':
    sys.exit(run(sys.argv[1]))
