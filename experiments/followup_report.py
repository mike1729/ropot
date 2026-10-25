"""Replay and verify retained follow-up replies without new model dispatches."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
from statistics import median
from time import perf_counter

from .followup_fixtures import KINDS, program, verification_task
from .inference_benchmark import encoded, replay, save
from .run_inference_benchmarks import PROFILES


def code(text):
    text = text.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        if match is None:
            raise ValueError('Expected plain code or one Python fence')
        text = match[1]
    ast.parse(text)
    if not text:
        raise ValueError('Empty program')
    return (text+'\n').encode()


def timing_summary(rows):
    """Interrupted windows retain cost but cannot appear to be faster completions."""
    seconds=[r['elapsed_seconds'] for r in rows if r['status']=='success']
    return dict(repetitions=len(seconds),median_seconds=median(seconds) if seconds else None,
        range_seconds=[min(seconds),max(seconds)] if seconds else None,
        aborted_windows=len(rows)-len(seconds),total_window_seconds=sum(r['elapsed_seconds'] for r in rows))


def verify(root):
    """Deduplicate execution only; every draw retains its own score and cost."""
    verifier = importlib.import_module('src.data-pipeline.stage03.verification')
    target = root/'verification'
    target.mkdir(exist_ok=False)
    unique, entries = {}, []
    def candidate(task, source):
        digest = sha256(encoded(task)).hexdigest()
        key = (digest, sha256(source).hexdigest())
        if key not in unique:
            unique[key] = (task, digest, source)
        return key
    controls = [candidate(verification_task(kind, 0), program(kind).encode()) for kind in KINDS]
    for profile in PROFILES:
        folder = root/f'{profile}-long'
        if not (folder/'summary.json').exists():
            continue
        corpus = {k:{r['id']:r for r in v} for k,v in json.loads((folder/'corpus.json').read_text()).items()}
        for summary in json.loads((folder/'summary.json').read_text()):
            if summary['stage'] == 'warm':
                continue
            for row in replay(folder/summary['name'])[0]:
                item = dict(profile=profile, window=summary['name'], id=row['id'],
                    context=summary['context'], batch=summary['batch'], stage=summary['stage'], status='incomplete')
                if row['status'] == 'stop':
                    try:
                        source = code((folder/summary['name']/row['reply']).read_text())
                        request = corpus[summary['context']][row['id']]
                        key = candidate(verification_task(request['kind'], request['id']//3), source)
                        item.update(status='pending', key=list(key))
                    except (SyntaxError, ValueError):
                        item['status'] = 'format_failure'
                entries.append(item)
    save(target/'inputs.json', dict(entries=entries, controls=controls,
        candidates=[dict(task_sha256=k[0], program_sha256=k[1], task=t, program=s.decode())
                    for k, (t, _, s) in unique.items()], execution_deduplicated=True))
    start = perf_counter()
    results = verifier.verify_many(list(unique.values()), target/'logs', max_samples=4)
    if len(results) != len(unique):
        raise RuntimeError('Verification cardinality changed')
    checked = dict(zip(unique, results))
    for key in controls:
        if checked[key]['status'] != 'passed':
            raise RuntimeError('Independent invented reference control failed')
    for item in entries:
        if item['status'] == 'pending':
            result = checked[tuple(item['key'])]
            item.update(status=result['status'], evidence=result)
    save(target/'summary.json', dict(seconds=perf_counter()-start, draws=entries,
        unique_program_task_pairs=len(unique), controls=[checked[k] for k in controls]))


def report(root):
    evidence = json.loads((root/'verification/summary.json').read_text()) if (root/'verification/summary.json').exists() else None
    lines = ['# Q8 long-history and cache semantic follow-ups', '',
        'Invented fixtures only; no witness/backdoor yield or cohort exposure.', '',
        '| Model | Context | Batch target | Repeats | Median seconds | Range | Verified draws | Peak GiB |',
        '| --- | --- | ---: | ---: | ---: | --- | --- | ---: |']
    output = {}
    for profile in PROFILES:
        folder = root/f'{profile}-long'
        rows = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else []
        output[profile] = {'long': []}
        for context,batch in sorted({(r['context'],r['batch']) for r in rows if r['stage'] != 'warm'}):
            selected = [r for r in rows if r['context']==context and r['batch'] == batch and r['stage'] != 'warm']
            timing=timing_summary(selected)
            draws = [r for r in evidence['draws'] if r['profile'] == profile and r['context']==context and r['batch'] == batch] if evidence else []
            row = dict(context=context, batch=batch, **timing, completed_windows=timing['repetitions'],
                verified=Counter(r['status'] for r in draws), peak_gib=max(r['peak_memory_bytes'] for r in selected)/1024**3,
                delivered_tokens=sum(r['delivered_tokens'] for r in selected),
                max_observed_decode_batch=max(r['max_observed_decode_batch'] for r in selected))
            output[profile]['long'].append(row)
            seconds='—' if timing['median_seconds'] is None else f"{timing['median_seconds']:.3f}"
            interval='—' if timing['range_seconds'] is None else f"{timing['range_seconds'][0]:.3f}–{timing['range_seconds'][1]:.3f}"
            lines.append(f"| {profile} | {context} | {batch} | {timing['repetitions']} | {seconds} | "
                f"{interval} | {row['verified']} | {row['peak_gib']:.2f} |")
        folder = root/f'{profile}-semantic'
        rows = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else []
        complete = json.loads((folder/'complete.json').read_text()) if (folder/'complete.json').exists() else {}
        tested = [r for r in rows if r['stage'] != 'cancel']
        # Six groups x (three greedy controls + six sampled cells), plus three ownership windows.
        qualified = bool(complete.get('finished_matrix') and len(tested) == 57
                         and all(r.get('all_passed') and r['status']=='success' for r in tested))
        comparisons = []
        for cached in [r for r in rows if r['stage']=='measured' and r['cached']]:
            name = cached['name'].removesuffix('cache')+'batch'
            baseline = next((r for r in rows if r['name']==name), None)
            if baseline is None:
                continue
            a = {r['id']:r for r in replay(folder/name)[0]}
            b = {r['id']:r for r in replay(folder/cached['name'])[0]}
            if set(a) != set(b):
                raise RuntimeError('Paired cache worklists differ')
            comparisons.append(dict(group=cached['name'].split('-r')[0], repeat=cached['repeat'],
                baseline_seconds=baseline['elapsed_seconds'], cached_seconds=cached['elapsed_seconds'],
                requests=len(a), token_divergences=sum(a[i]['token_ids']!=b[i]['token_ids'] for i in a),
                baseline_passed=baseline['all_passed'], cached_passed=cached['all_passed']))
        output[profile]['semantic'] = dict(qualified=qualified, tested_windows=len(tested),
            verdicts=Counter(v for r in tested for v in r['verdicts'].values()), comparisons=comparisons,
            complete=complete)
        lines += ['', f"{profile} semantic cache qualification: **{qualified}**; "
            f"{len(tested)}/57 required non-cancellation windows; "
            f"verdicts {output[profile]['semantic']['verdicts']}."]
        for group in sorted({r['group'] for r in comparisons}):
            paired = [r for r in comparisons if r['group']==group]
            lines.append(f"- {group}: uncached median {median(r['baseline_seconds'] for r in paired):.3f}s; "
                f"cached {median(r['cached_seconds'] for r in paired):.3f}s; "
                f"{sum(r['token_divergences'] for r in paired)}/{sum(r['requests'] for r in paired)} token-sequence differences.")
    save(root/'report.json', output)
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args(argv)
    if args.verify or args.verify_only:
        verify(Path(args.root))
    if not args.verify_only:
        report(Path(args.root))


if __name__ == '__main__':
    main()
