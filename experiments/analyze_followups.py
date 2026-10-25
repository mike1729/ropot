"""Combine retained phases, without redrawing or selecting replies by quality."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import median

from .followup_fixtures import semantic_verdict
from .followup_report import timing_summary
from .inference_benchmark import replay, save
from .run_inference_benchmarks import PROFILES


def read(path):
    return json.loads(path.read_text())


def secondary_semantic_verdict(text,status,expected):
    """Post-hoc JSON unfencing; all original schema/oracle rules still apply."""
    match=re.fullmatch(r'```json[ \t]*\n(.*?)\n```',text.strip(),re.DOTALL)
    return semantic_verdict(match[1] if match else text,status,expected)


def analyze(roots, output):
    output.mkdir(parents=True,exist_ok=False)
    primary, secondary, grouped = {},{},defaultdict(list)
    accounting=defaultdict(Counter)
    for root in roots:
        verified=read(root/'verification/summary.json')
        if any(r['status']!='passed' for r in verified['controls']):
            raise ValueError('Reference control failed')
        for row in verified['draws']:
            key=(str(root.resolve()),row['profile'],row['window'],row['id'])
            if key in primary:
                raise ValueError('Duplicate primary score')
            primary[key]=row['status']
        for profile in PROFILES:
            folder=root/f'{profile}-long'
            if not (folder/'summary.json').exists():
                continue
            indexed=read(folder/'summary.json')
            if {p.parent.name for p in folder.glob('*/summary.json')} != {r['name'] for r in indexed}:
                raise ValueError('Unindexed native window requires explicit analysis')
            for row in indexed:
                directory=folder/row['name']
                draws,_=replay(directory)
                if row['status']=='success' and sum(len(r['token_ids']) for r in draws)!=row['native']['generation_tokens']:
                    raise ValueError('Native accounting changed')
                requests=read(directory/'inputs.json')['requests']
                if {r['id'] for r in draws}!={r['id'] for r in requests}:
                    raise ValueError('Draw ownership changed')
                accounting[profile]['delivered_program_tokens']+=sum(len(r['token_ids']) for r in draws)
                for draw in draws:
                    accounting[profile][('warm_' if row['stage']=='warm' else '')+draw['status']]+=1
                grouped[profile,row['context'],row['batch']].append((root,row,draws))
    for root in roots:
        path=root/'secondary-replay/summary.json'
        if path.exists():
            for row in read(path)['entries']:
                key=(row['root'],row['profile'],row['window'],row['id'])
                if key in secondary:
                    raise ValueError('Duplicate secondary score')
                secondary[key]=row['status']
    performance=[]
    for (profile,context,batch),items in sorted(grouped.items()):
        measured=[r for _,r,_ in items if r['stage']!='warm']
        timing=timing_summary(measured)
        scores, recovered = Counter(),Counter()
        all_scores=Counter()
        complete_names=set()
        for root,row,draws in items:
            if row['stage']=='warm':
                continue
            if row['status']=='success':
                # Same successful timing cell must never be chosen twice across phases.
                if row['name'] in complete_names:
                    raise ValueError('A completed cell was redrawn')
                complete_names.add(row['name'])
            for draw in draws:
                key=(str(root.resolve()),profile,row['name'],draw['id'])
                if key not in primary:
                    raise ValueError('Missing primary draw score')
                all_scores[primary[key]]+=1
                if row['status']=='success':
                    scores[primary[key]]+=1
                    if secondary.get(key,'').startswith('recovered_'):
                        if primary[key]!='format_failure':
                            raise ValueError('Recovery overlaps a primary-accepted program')
                        recovered[secondary[key]]+=1
        configs=[read(root/f'{profile}-long'/r['name']/'inputs.json')['config'] for root,r,_ in items]
        performance.append(dict(profile=profile,context=context,batch=batch,**timing,
            peak_gib=max(r['peak_memory_bytes'] for _,r,_ in items)/1024**3,
            all_window_seconds=sum(r['elapsed_seconds'] for _,r,_ in items),
            primary_complete_cell_scores=scores,secondary_complete_cell_recoveries=recovered,
            all_record_scores=all_scores,configs=configs,
            adaptive=any(r['stage']=='fallback' for _,r,_ in items)))
    semantics={}
    for profile in PROFILES:
        folders=[root/f'{profile}-semantic' for root in roots if (root/f'{profile}-semantic/summary.json').exists()]
        if len(folders)!=1:
            raise ValueError('Expected one frozen semantic matrix per model')
        folder=folders[0];rows=read(folder/'summary.json');byname={r['name']:r for r in rows}
        verdicts=Counter(); secondary_verdicts=Counter(); fingerprints=[]; paired=[]
        for row in rows:
            directory=folder/row['name'];draws,_=replay(directory)
            accounting[profile]['delivered_semantic_tokens']+=sum(len(d['token_ids']) for d in draws)
            if row['prefix']:
                fingerprints.append(row['prefix']['state_before_sha256']==row['prefix']['state_after_sha256'])
            if row['stage']=='cancel':
                continue
            expected={r['id']:r['expected_answer'] for r in read(directory/'inputs.json')['requests']}
            checked={str(r['id']):semantic_verdict((directory/r['reply']).read_text(),r['status'],expected[r['id']]) for r in draws}
            if checked!=row['verdicts']:
                raise ValueError('Semantic readback changed')
            verdicts.update(checked.values())
            secondary_checked={str(r['id']):secondary_semantic_verdict(
                (directory/r['reply']).read_text(),r['status'],expected[r['id']]) for r in draws}
            secondary_verdicts.update(secondary_checked.values())
            if row['stage']=='measured' and row['cached']:
                name=row['name'].removesuffix('cache')+'batch';baseline=byname[name]
                a={r['id']:r for r in replay(folder/name)[0]};b={r['id']:r for r in draws}
                if set(a)!=set(b):
                    raise ValueError('Paired request IDs differ')
                baseline_expected={r['id']:r['expected_answer'] for r in read(folder/name/'inputs.json')['requests']}
                if baseline_expected!=expected:
                    raise ValueError('Paired oracle answers differ')
                baseline_secondary={str(i):secondary_semantic_verdict((folder/name/a[i]['reply']).read_text(),
                    a[i]['status'],baseline_expected[i]) for i in a}
                paired.append(dict(group=row['name'].split('-r')[0],repeat=row['repeat'],
                    baseline_seconds=baseline['elapsed_seconds'],cached_seconds=row['elapsed_seconds'],
                    requests=len(a),token_differences=sum(a[i]['token_ids']!=b[i]['token_ids'] for i in a),
                    verdict_differences=sum(baseline['verdicts'][str(i)]!=row['verdicts'][str(i)] for i in a),
                    baseline_passed=sum(v=='passed' for v in baseline['verdicts'].values()),
                    cached_passed=sum(v=='passed' for v in row['verdicts'].values()),
                    secondary_json_baseline_passed=sum(v=='passed' for v in baseline_secondary.values()),
                    secondary_json_cached_passed=sum(v=='passed' for v in secondary_checked.values()),
                    secondary_json_verdict_differences=sum(baseline_secondary[str(i)]!=secondary_checked[str(i)] for i in a)))
        complete=read(folder/'complete.json')
        semantics[profile]=dict(qualified=bool(complete['finished_matrix'] and len(rows)==58
            and all(fingerprints) and verdicts.get('passed',0)==sum(verdicts.values())),
            verdicts=verdicts,secondary_json_verdicts=secondary_verdicts,
            fingerprint_checks=len(fingerprints),fingerprints_unchanged=sum(fingerprints),pairs=paired)
    outcomes=[]
    for root in roots:
        for path in root.glob('processes/*/outcome.json'):
            outcomes.append(dict(root=str(root),worker=path.parent.name,**read(path)))
    if any(not r['process_group_absent'] for r in outcomes):
        raise ValueError('Owned process group survived')
    files=[p for root in roots for p in root.rglob('*') if p.is_file()]
    result=dict(roots=[str(r.resolve()) for r in roots],performance=performance,semantics=semantics,
        accounting=accounting,outcomes=outcomes,retained_files=len(files),retained_bytes=sum(p.stat().st_size for p in files),
        inspect_log_files=sum(p.suffix=='.eval' for p in files),
        inspect_log_bytes=sum(p.stat().st_size for p in files if p.suffix=='.eval'),
        analysis_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    save(output/'analysis.json',result)
    lines=['# Combined Q8 follow-up results','','Complete windows only in timing medians; all partial costs remain in JSON.',
        'Secondary recoveries are post hoc; primary acceptance is unchanged.','',
        '| Model | Context | Batch | Repeats | Median s | Aborted | Peak GiB | Primary passed | Secondary recovered |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in performance:
        seconds='—' if r['median_seconds'] is None else f"{r['median_seconds']:.3f}"
        scores=r['primary_complete_cell_scores'];recover=r['secondary_complete_cell_recoveries']
        lines.append(f"| {r['profile']} | {r['context']} | {r['batch']} | {r['repetitions']} | {seconds} | "
            f"{r['aborted_windows']} | {r['peak_gib']:.2f} | {scores.get('passed',0)}/{sum(scores.values())} | "
            f"{recover.get('recovered_passed',0)} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    analyze([Path(r) for r in args.roots],Path(args.output))
