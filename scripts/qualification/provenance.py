"""Resolve inherited exposure records without replacing original frozen partitions."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f


def reconcile(output=f.OUT):
    manifest = f.load(output)
    target = Path(output) / 'provenance'
    target.mkdir(exist_ok=False)
    inputs = {str(Path(output)/'manifest.json'):f.digest(Path(output)/'manifest.json'), str(Path(__file__)):f.digest(__file__)}
    histories = []
    for path in manifest['inputs']:
        if path.endswith('/exposure.json'):
            value = f.read(path)
            histories.append((path, value.get('tasks', {})))
    recovery = f.ROOT/'data/03_benign_experiments/qwen38-dev12-recovery-cases-20261030'
    rec = f.read(recovery/'manifest.json')
    complete = f.read(recovery/'completion.json')
    assert complete['manifest_sha256'] == f.digest(recovery/'manifest.json') and complete['status']=='completed'
    assert rec['exposure_sha256'] == f.digest(recovery/'exposure.json')
    devcases = []
    for entry in rec['entries']:
        path = recovery/entry['path']
        assert f.digest(path) == entry['sha256']
        inputs[str(path)] = f.digest(path)
        data = f.read(path)
        devcases += [(entry['task_id'], c, str(path)) for c in data['cases'] if c['role']=='audit']
    construction = f.ROOT/'data/03_benign_experiments/qwen38-construction13-recovered-v2-20261030'
    conc = f.read(construction/'contract.json')
    conr = f.read(construction/'acquisition/report.json')
    assert conr['contract_sha256'] == f.digest(construction/'contract.json') and conr['status']=='completed'
    con_exposure = f.read(construction/'exposure.json')
    con_context = f.read(construction/'context-review.json')
    assert con_context['exposure_sha256'] == f.digest(construction/'exposure.json')
    concases = [(e['binding']['id'], c, str(construction/'contract.json')) for e in conc['entries'] for c in e['partition']['audit']]
    for path in (recovery/'manifest.json', recovery/'completion.json', recovery/'exposure.json',
                 construction/'contract.json',construction/'acquisition/report.json',construction/'exposure.json',construction/'context-review.json'):
        inputs[str(path)] = f.digest(path)
    rows = []
    for pair in manifest['pairs']:
        for case, original in zip(pair['partition']['audit'],pair['exposure'],strict=True):
            row = dict(original, pair_id=pair['id'], original_designation=original['status'], supporting_records=[])
            if original['status'] != 'unresolved_exposure_history':
                rows.append(row); continue
            # Cross-source reuse inherits the task's reviewed disclosure history, not a new partition.
            if case['index'] < 100000:
                for path,tasks in histories:
                    h=tasks.get(str(pair['binding']['id']),{})
                    if h.get('task_sha256') == pair['binding']['sha256'] and case['index'] in h.get('known',[]) and case['index'] not in h.get('disclosed',[]):
                        row['supporting_records'].append(path)
                if row['supporting_records']:
                    row['status']='recorded_private_inherited'
            for taskid,c,path in devcases+concases:
                if taskid == pair['binding']['id'] and all(c[k] == case[k] for k in ('index','input','answer')):
                    row['supporting_records'].append(path)
                    row['status']='recorded_private_recovered'
                    if taskid in conc['recovery']['contextual_exception_tasks'] and path.startswith(str(construction)):
                        row['status']='recorded_private_context_exception'
            # Preserve any positive disclosure even if a different history says private.
            if original['shown_contracts'] or original['recorded_disclosures']:
                row['status']='exposed_diagnostic'
            rows.append(row)
    f.save(target/'cases.json',dict(cases=rows,counts=Counter(r['status'] for r in rows),
        distinction='Recorded-private means the saved exposure history and partition support withholding from generation. It is not a claim of absence from pretraining. Context exceptions retain their original limits; unresolved/exposed cases are diagnostics only.',
        before_execution='The original manifest recorded conservative unresolved designations before execution. This supplement resolves inherited provenance using historical records without changing any case or result.'))
    inputs[str(target/'cases.json')]=f.digest(target/'cases.json')
    f.seal(target,dict(inputs=inputs,original_partition_hashes={p['id']:p['partition_sha256'] for p in manifest['pairs']}))
    return Counter(r['status'] for r in rows)


if __name__=='__main__':
    print(reconcile())
