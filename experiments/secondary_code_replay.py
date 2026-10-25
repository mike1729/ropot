"""Secondary, explicitly post-hoc recovery of fenced code from retained replies.

Does not alter primary scores, make model calls, or salvage incomplete draws.
Only one unambiguous Python fence may be selected; execute only in native Docker.
"""

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re
from time import perf_counter

from .followup_fixtures import KINDS, program, verification_task
from .followup_report import code
from .inference_benchmark import encoded, replay, save
from .run_inference_benchmarks import PROFILES


def fenced_code(text):
    if text.count('```') != 2:
        raise ValueError('Expected exactly one fenced block')
    match = re.search(r'^```(?:python)?[ \t]*\n(.*?)\n```[ \t]*(?=\n|$)', text,
                      re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError('No complete Python fence')
    return code(match[1])


def run(roots, target):
    verifier = importlib.import_module('src.data-pipeline.stage03.verification')
    target.mkdir(parents=True,exist_ok=False)
    entries, unique = [],{}
    def candidate(task, source):
        digest=sha256(encoded(task)).hexdigest()
        key=(digest,sha256(source).hexdigest())
        unique.setdefault(key,(task,digest,source))
        return key
    controls=[candidate(verification_task(kind,0),program(kind).encode()) for kind in KINDS]
    for root in roots:
        for profile in PROFILES:
            folder=root/f'{profile}-long'
            if not (folder/'summary.json').exists():
                continue
            corpus=json.loads((folder/'corpus.json').read_text())
            lookup={k:{r['id']:r for r in v} for k,v in corpus.items()}
            for summary in json.loads((folder/'summary.json').read_text()):
                if summary['stage']=='warm':
                    continue
                for row in replay(folder/summary['name'])[0]:
                    item=dict(root=str(root.resolve()),profile=profile,context=summary['context'],
                        window=summary['name'],batch=summary['batch'],id=row['id'],status='incomplete')
                    entries.append(item)
                    if row['status']!='stop':
                        continue
                    text=(folder/summary['name']/row['reply']).read_text()
                    try:
                        code(text)
                        item['status']='primary_format_passed'
                        continue
                    except (SyntaxError,ValueError):
                        pass
                    try:
                        source=fenced_code(text)
                    except (SyntaxError,ValueError):
                        item['status']='unrecoverable_format'
                        continue
                    request=lookup[summary['context']][row['id']]
                    key=candidate(verification_task(request['kind'],request['id']//3),source)
                    item.update(status='pending',key=list(key),raw_reply_sha256=row['reply_sha256'])
    save(target/'inputs.json',dict(secondary_post_hoc=True,model_dispatches=0,entries=entries,
        candidates=[dict(task=t,task_sha256=k[0],program=s.decode(),program_sha256=k[1])
                    for k,(t,_,s) in unique.items()]))
    started=perf_counter()
    results=verifier.verify_many(list(unique.values()),target/'logs',max_samples=4)
    if len(results)!=len(unique):
        raise RuntimeError('Verification cardinality changed')
    checked=dict(zip(unique,results))
    if any(checked[k]['status']!='passed' for k in controls):
        raise RuntimeError('Reference control failed')
    for item in entries:
        if item['status']=='pending':
            result=checked[tuple(item['key'])]
            item.update(status='recovered_'+result['status'],evidence=result)
    save(target/'summary.json',dict(secondary_post_hoc=True,model_dispatches=0,
        wall_seconds=perf_counter()-started,unique_program_task_pairs=len(unique),entries=entries))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run([Path(p) for p in args.roots],Path(args.output))
