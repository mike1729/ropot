"""Complete only budget-stopped repair timing cells in a separate evidence root.

Never retries a completed cell because of its answer quality. Original partial
draws and their costs remain evidence; repeated cells generate the full worklist.
"""

import argparse
import fcntl
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import random
from time import perf_counter

from .followup_benchmark import swap_used
from .inference_benchmark import save, window
from .run_inference_benchmarks import MODELS, PROFILES


def missing_cells(rows, prefill):
    complete = {r['name'] for r in rows if r['status']=='success'}
    if any(r['status']=='error' and 'TimeoutError' not in (r.get('error') or '') for r in rows):
        raise ValueError('Continuation cannot retry a non-budget failure')
    planned = []
    for context in ('2k','7k'):
        planned.append((context+'-calibration',context,prefill,None,'calibration'))
        for repeat in range(3):
            order = [8,32]
            random.Random(610+repeat).shuffle(order)
            planned += [(f'{context}-r{repeat}-b{b}',context,b,repeat,'measured') for b in order]
    # A measured slow calibration is already informative; do not redraw its 33
    # requests merely to drain the last two after the 360-second cap.
    censored = {r['name'] for r in rows if r['status']=='error' and r['name'].endswith('-calibration')}
    return [r for r in planned if r[0] not in complete | censored]


def source_rows(source):
    rows=json.loads((source/'summary.json').read_text())
    indexed={r['name'] for r in rows}
    # Unexpected native exceptions may leave a window summary before the outer
    # recorder appends its index. They must never bypass the failure guard.
    for path in source.glob('*/summary.json'):
        if path.parent.name not in indexed:
            rows.append(dict(name=path.parent.name,**json.loads(path.read_text())))
    return rows


def run(args):
    import mlx.core as mx
    from mlx_lm import load

    source, root = Path(args.source), Path(args.output)
    rows = source_rows(source)
    name, _, prefill, step = PROFILES[args.profile]
    pending = missing_cells(rows,prefill)
    if (source/'skipped.json').exists() and any('swap' in x for x in json.loads((source/'skipped.json').read_text())):
        raise ValueError('Cannot retry memory-pressure omissions')
    root.mkdir(parents=True,exist_ok=False)
    corpus = json.loads((source/'corpus.json').read_text())
    save(root/'corpus.json',corpus)
    save(root/'plan.json',dict(original=str(source.resolve()), pending=pending,
        soft_seconds=args.seconds, original_inputs_sha256=sha256((source/'corpus.json').read_bytes()).hexdigest(),
        conditional_batch16=args.memory_fallback_16,
        retry_scope='unfinished timing cells only; original incomplete draws remain counted'))
    if not pending:
        save(root/'summary.json',[])
        save(root/'complete.json',dict(nothing_missing=True))
        return
    started = perf_counter()
    identity = json.loads((source/'runtime.json').read_text())
    model_path = MODELS/'mlx-community'/name
    if str(model_path) != identity['model']:
        raise ValueError('Model identity differs')
    for package, expected in identity['versions'].items():
        if version(package) != expected:
            raise ValueError('Runtime version drift')
    for path, digest in identity['source_sha256'].items():
        if sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned native runtime source drift')
    for filename,digest in identity['metadata_sha256'].items():
        if sha256((model_path/filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Model metadata drift')
    mx.set_cache_limit(4*1024**3)
    if mx.set_cache_limit(4*1024**3) != 4*1024**3:
        raise RuntimeError('Allocator cache readback failed')
    model, tokenizer = load(str(model_path),lazy=False,
        tokenizer_config={'local_files_only':True},trust_remote_code=False)
    for requests in corpus.values():
        for r in requests:
            actual = tokenizer.apply_chat_template(r['messages'],tokenize=True,
                add_generation_prompt=True,enable_thinking=False)
            if actual != r['prompt_ids']:
                raise ValueError('Frozen prompt tokenization drift')
    save(root/'runtime.json',dict(original_identity=identity,device=mx.device_info(),
        load_seconds=perf_counter()-started))
    (root/'model-manifest.json').write_bytes((source/'model-manifest.json').read_bytes())
    summaries, warmed = [],set()
    def draw(label, requests, config, context, batch, repeat, stage):
        remaining=args.seconds-(perf_counter()-started)
        if remaining < 30:
            raise TimeoutError('Finite continuation budget exhausted')
        try:
            result=window(model,tokenizer,requests,config,root/label,
                min(120 if stage=='warm' else 360,remaining),max_peak_bytes=42*1024**3)
        finally:
            if (root/label/'summary.json').exists():
                result=json.loads((root/label/'summary.json').read_text())
                summaries.append(dict(name=label,context=context,batch=batch,repeat=repeat,stage=stage,**result))
                save(root/'summary.json',summaries)
        print(json.dumps(dict(window=label,seconds=result['elapsed_seconds'],finished=result['finish_counts'])),flush=True)
        if swap_used(result,'after')-swap_used(summaries[0],'before') > 256*1024**2:
            raise MemoryError('Stop continuation after swap growth')
    rejected, omitted = set(),[]
    # Appending the predeclared fallback visits it only after original cells.
    for label,context,batch,repeat,stage in pending:
        if (context,batch) in rejected:
            omitted.append(label)
            continue
        config = dict(completion_batch_size=batch, prefill_step_size=step,
            prefill_batch_size=(16 if args.profile=='actor' else 8) if batch==32 else min(batch,prefill))
        if (context,batch) not in warmed:
            warm = [{**r,'max_tokens':16} for r in corpus[context][:3]]
            draw(f'{context}-warm-b{batch}',warm,config,context,batch,None,'warm')
            warmed.add((context,batch))
        memory_rejection=False
        try:
            draw(label,corpus[context],config,context,batch,repeat,stage)
        except MemoryError as error:
            if not (args.memory_fallback_16 and args.profile=='synthetic' and context=='7k'
                    and batch==32 and str(error)=='Benchmark peak-memory ceiling exceeded'):
                raise
            if swap_used(summaries[-1],'after')-swap_used(summaries[0],'before') > 256*1024**2:
                raise
            rejected.add((context,batch))
            memory_rejection=True
            pending.extend((f'{context}-r{r}-b16',context,16,r,'fallback') for r in range(3))
            save(root/'decisions.json',dict(rejected=list(rejected),pending=pending,
                reason='Observed 42-GiB ceiling; no retry of rejected configuration'))
        if memory_rejection:
            mx.synchronize()
            mx.clear_cache()
    save(root/'complete.json',dict(elapsed_seconds=perf_counter()-started,finished_matrix=not rejected,
        completed_applicable_cells=True,rejected=list(rejected),omitted=omitted))
    mx.synchronize()
    del model
    mx.clear_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--seconds',type=float,default=1200)
    parser.add_argument('--memory-fallback-16',action='store_true')
    args=parser.parse_args()
    with Path('data/03_throughput/current-benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args)
