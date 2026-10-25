"""Finite repair continuation, followed by primary and secondary replay."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from . import run_inference_benchmarks as driver
from .complete_followups import missing_cells, source_rows
from .inference_benchmark import save


def run(args):
    source, root=Path(args.source).resolve(),Path(args.output).resolve()
    if args.replay_only and not root.is_dir():
        raise ValueError('Replay requires existing retained evidence')
    root.mkdir(parents=True,exist_ok=args.replay_only)
    if args.replay_only:
        (root/'replay-sources').mkdir(exist_ok=False)
    if args.replay_only and any(not json.loads(p.read_text())['process_group_absent']
                               for p in root.glob('processes/*/outcome.json')):
        raise RuntimeError('An owned process group survived')
    sources=[*list((driver.ROOT/'experiments').glob('*.py')),
        *(driver.ROOT/'src/data-pipeline/stage03'/n for n in ('verification.py','inspect_runner.py')),
        driver.ROOT/'docs/inference-followup-continuation.md']
    driver.FROZEN.update({p:sha256(p.read_bytes()).hexdigest() for p in sources})
    for path in sources:
        copy=root/('replay-sources' if args.replay_only else 'sources')/path.relative_to(driver.ROOT)
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_bytes(path.read_bytes())
    planned={} if args.replay_only else {profile:missing_cells(source_rows(source/f'{profile}-long'),details[2])
                                        for profile,details in driver.PROFILES.items()}
    budgets={'actor':(1200,1260),'synthetic':(1800,1860)}
    save(root/('replay-plan.json' if args.replay_only else 'plan.json'),dict(started=datetime.now(timezone.utc).isoformat(),
        original=str(source),pending=planned,budgets=budgets,conditional_batch16=True,
        replay_only=args.replay_only,
        source_sha256={str(p.relative_to(driver.ROOT)):h for p,h in driver.FROZEN.items()}))
    try:
        for profile in driver.PROFILES:
            if not planned.get(profile):
                continue
            soft,hard=budgets[profile]
            name=f'{profile}-long'
            command=[driver.MLX,'-m','experiments.complete_followups','--source',str(source/name),
                '--output',str(root/name),'--profile',profile,'--seconds',str(soft)]
            if profile=='synthetic':
                command.append('--memory-fallback-16')
            driver.wait([driver.launch(root,name,command,hard)])
        driver.wait([driver.launch(root,'primary-verification',[driver.PYTHON,'-m',
            'experiments.followup_report','--root',str(root),'--verify-only'],1200)])
        driver.wait([driver.launch(root,'secondary-replay',[driver.PYTHON,'-m',
            'experiments.secondary_code_replay','--roots',str(source),str(root),
            '--output',str(root/'secondary-replay')],1200)])
    finally:
        if driver.ACTIVE:
            driver.finish(list(driver.ACTIVE),True)
    save(root/('replay-complete.json' if args.replay_only else 'complete.json'),dict(process_groups_absent=not driver.ACTIVE))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--replay-only',action='store_true')
    run(parser.parse_args())
