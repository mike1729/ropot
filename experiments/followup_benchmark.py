"""Finite long-history and semantic prefix-cache follow-ups; invented tasks only."""

import argparse
from copy import deepcopy
import fcntl
from hashlib import sha256
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import sys
from time import perf_counter

from .followup_fixtures import KINDS, common_prefix, long_requests, semantic_group, semantic_verdict
from .inference_benchmark import replay, save, window
from .run_inference_benchmarks import MODELS, PROFILES


def tokenize(tokenizer, requests):
    for row in requests:
        row['prompt_ids'] = tokenizer.apply_chat_template(row['messages'], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
    return requests


def swap_used(summary, when):
    text = summary['resources_'+when]['swapusage']
    match = re.search(r'used\s*=\s*([\d.]+)([MG])', text)
    if not match:
        raise ValueError('Cannot establish swap usage: '+text)
    return float(match[1]) * (1024**3 if match[2] == 'G' else 1024**2)


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    started = perf_counter()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    name, manifest, prefill, step = PROFILES[args.profile]
    model_path = MODELS/'mlx-community'/name
    versions = {n: version(n) for n in ('mlx', 'mlx-lm', 'mlx-metal', 'transformers', 'tokenizers')}
    if versions['mlx'] != '0.32.4' or versions['mlx-lm'] != '0.32.2':
        raise RuntimeError('Use the pinned MLX runtime')
    config = json.loads((model_path/'config.json').read_text())
    if config.get('quantization', config.get('quantization_config', {})).get('bits') != 8:
        raise ValueError('Follow-ups require Q8')
    (root/'model-manifest.json').write_bytes((MODELS/'metadata'/manifest).read_bytes())
    mx.set_cache_limit(4*1024**3)
    if mx.set_cache_limit(4*1024**3) != 4*1024**3:
        raise RuntimeError('Allocator cache readback failed')
    model, tokenizer = load(str(model_path), lazy=False,
        tokenizer_config={'local_files_only': True}, trust_remote_code=False)
    paths = [Path(importlib.import_module(n).__file__) for n in
        ('mlx_lm.generate', 'mlx_lm.sample_utils', 'mlx_lm.models.cache', type(model).__module__)]
    save(root/'runtime.json', dict(model=str(model_path), versions=versions, device=mx.device_info(),
        python=sys.version, load_seconds=perf_counter()-started,
        source_sha256={str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
        metadata_sha256={n: sha256((model_path/n).read_bytes()).hexdigest() for n in
                        ('config.json', 'tokenizer.json', 'chat_template.jinja')}))
    deadline = started+args.seconds
    summaries, skipped = [], []
    def cfg(batch):
        return dict(completion_batch_size=batch,
            prefill_batch_size=(16 if args.profile == 'actor' else 8) if batch == 32 else min(batch, prefill),
            prefill_step_size=step)

    def draw(label, requests, batch, *, cached=False, repeat=None, stage='measured', soft=None):
        remaining = deadline-perf_counter()
        # No new work starts with less than 30 seconds; incomplete cells stay missing.
        if remaining < 30:
            skipped.append(label)
            save(root/'skipped.json', skipped)
            return None
        error = None
        try:
            result = window(model, tokenizer, requests, cfg(batch), root/label,
                min(360, remaining) if soft is None else soft,
                reuse_prefix_tokens=common_prefix(requests) if cached else 0,
                max_peak_bytes=42*1024**3)
        except (TimeoutError, MemoryError) as exc:
            error = repr(exc)
            result = json.loads((root/label/'summary.json').read_text())
        row = dict(name=label, batch=batch, cached=cached, repeat=repeat, stage=stage, **result)
        if args.mode == 'long':
            row['context'] = label.split('-')[0]
        if args.mode == 'semantic' and stage != 'cancel':
            expected = {r['id']: r['expected_answer'] for r in requests}
            rows, _ = replay(root/label)
            row['verdicts'] = {str(r['id']): semantic_verdict(
                (root/label/r['reply']).read_text(), r['status'], expected[r['id']]) for r in rows}
            row['all_passed'] = len(rows) == len(requests) and all(v == 'passed' for v in row['verdicts'].values())
        summaries.append(row)
        save(root/'summary.json', summaries)
        print(json.dumps(dict(window=label, seconds=round(result['elapsed_seconds'], 3),
            peak_gib=round(result['peak_memory_bytes']/1024**3, 3),
            finished=result['finish_counts'], all_passed=row.get('all_passed'), error=error)), flush=True)
        if error and stage != 'cancel':
            raise RuntimeError('Stopped follow-up after retained bounded-window failure: '+error)
        return row

    if args.mode == 'long':
        groups = {label: tokenize(tokenizer, long_requests(33, edges)) for label,edges in [('2k',32),('7k',240)]}
        for label, requests in groups.items():
            low, high = (1800,2600) if label == '2k' else (4096,10000)
            lengths = [len(r['prompt_ids']) for r in requests]
            if not low <= min(lengths) <= max(lengths) <= high:
                raise ValueError(f'{label} fixture outside frozen bounds: {min(lengths)}..{max(lengths)}')
        save(root/'corpus.json', groups)
        for label, requests in groups.items():
            for batch in (prefill, 8, 32):
                draw(f'{label}-warm-b{batch}', [{**r, 'max_tokens': 16} for r in requests[:3]], batch, stage='warm')
            draw(f'{label}-calibration', requests, prefill, stage='calibration')
            swap_start = swap_used(summaries[0], 'before')
            for repeat in range(3):
                order = [8, 32]
                random.Random(610+repeat).shuffle(order)
                for batch in order:
                    if swap_used(summaries[-1], 'after')-swap_start > 256*1024**2:
                        skipped.append(f'{label}-r{repeat}-b{batch}: swap growth'); save(root/'skipped.json', skipped)
                        continue
                    draw(f'{label}-r{repeat}-b{batch}', requests, batch, repeat=repeat)
    else:
        groups = {f'{kind}-{size}': tokenize(tokenizer, semantic_group(kind, size == 'long'))
                  for kind in KINDS for size in ('short', 'long')}
        save(root/'corpus.json', groups)
        save(root/'prefixes.json', {k: dict(tokens=common_prefix(v),
             min_prompt=min(len(r['prompt_ids']) for r in v), max_prompt=max(len(r['prompt_ids']) for r in v))
             for k, v in groups.items()})
        for label, requests in groups.items():
            batch = 8 if label.endswith('long') else 32
            greedy = [{**r, 'temperature': 0.} for r in requests]
            for arm, cached, capacity in (('serial', False, 1), ('batch', False, batch), ('cache', True, batch)):
                draw(f'{label}-greedy-{arm}', greedy, capacity, cached=cached, stage='greedy')
            for repeat in range(3):
                order = [False, True]
                random.Random(710+repeat).shuffle(order)
                for cached in order:
                    draw(f'{label}-r{repeat}-'+('cache' if cached else 'batch'), requests,
                         batch, cached=cached, repeat=repeat)
        requests = groups['components-long']
        draw('ownership-reorder', list(reversed(requests)), 8, cached=True, stage='ownership')
        changed = deepcopy(requests)
        # Change only one neighbour; its ID/seed remain owned, expected answer moves with the query.
        changed[-1]['messages'] = deepcopy(requests[1]['messages'])
        changed[-1]['prompt_ids'] = list(requests[1]['prompt_ids'])
        changed[-1]['expected_answer'] = requests[1]['expected_answer']
        draw('ownership-neighbour', changed, 8, cached=True, stage='ownership')
        cancelled = draw('ownership-cancel', requests, 8, cached=True, stage='cancel', soft=1e-12)
        if cancelled is not None and not (cancelled['status'] == 'error' and 'TimeoutError' in cancelled['error']):
            raise RuntimeError('Soft cancellation did not interrupt the native window')
        draw('ownership-recovery', requests, 8, cached=True, stage='ownership')
    save(root/'complete.json', dict(elapsed_seconds=perf_counter()-started, skipped=skipped,
         finished_matrix=not skipped, mode=args.mode))
    mx.synchronize()
    del model
    mx.clear_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--mode', choices=('long', 'semantic'), required=True)
    parser.add_argument('--seconds', type=float, required=True)
    args = parser.parse_args()
    with Path('data/03_throughput/current-benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
