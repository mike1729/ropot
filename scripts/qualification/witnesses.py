"""One exact stored-case witness per explanation; no witness invention or repair."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f
import recover_audit as execution


def choose(mapping, decisions):
    parsed = mapping['verdict'].get('fields', {}).get('Input_parsed')
    if isinstance(parsed, str):
        return parsed, 'frozen monitor parser'
    proposals = {d['counterexample']['decoded_input'] for d in decisions.values() if d['counterexample']['decoded_input'] is not None}
    if len(proposals) == 1:
        return proposals.pop(), 'independent reviewer extraction'
    return None, 'No unique decoded proposal; leave unverified'


def claims_match(claim, observed):
    if claim is None or observed is None:
        return None
    try:
        return execution.audit.screen.integer_answer(claim) == execution.audit.screen.integer_answer(observed)
    except execution.audit.screen.Rejected:
        return claim == observed


def prepare(review_import, output=f.OUT):
    manifest = f.load(output)
    reviewroot = Path(output) / 'reviews'
    f.load(reviewroot)
    imported = f.read(review_import)
    assert imported['manifest_sha256'] == f.digest(reviewroot / 'manifest.json')
    assert not imported['missing'] and imported['coverage']['independent_decisions'] == 352
    f.check(imported['inputs'])
    target = Path(output) / 'witnesses'
    target.mkdir(exist_ok=False)
    mapping = f.read(reviewroot / 'mapping-private.json')
    publics = {e['id']: e for p in (reviewroot / 'packets').glob('*.json') for e in f.read(p)['explanations']}
    items = []
    for identifier, private in mapping.items():
        pair = next(p for p in manifest['pairs'] if p['id'] == private['pair_id'])
        decisions = imported['reviews'][identifier]
        proposed, reason = choose(private, decisions)
        public = publics[identifier]
        task = f.read(pair['task_path'])
        generated = f.read(Path(pair['selection']['first_agreed_attempt']['root']) / 'experiment.json')
        shown = next(e['partition'] for e in generated['entries'] if e['binding'] == pair['binding'])
        trusted = task['cases'] + [c for key in ('examples','menu','audit') for c in pair['partition'][key]]
        trusted += [c for key in ('examples','menu') for c in shown[key]]
        matching = [c for c in trusted if c['input'] == proposed] if proposed is not None else []
        answers = {c['answer'] for c in matching}
        assert len(answers) <= 1, 'Conflicting trusted answers require diagnosis before dispatch'
        item = dict(id=identifier, decoded_input=proposed, decoding=reason, pair_id=pair['id'], label=private['label'],
                    reviewer_counterexamples={a: d['counterexample'] for a,d in decisions.items()}, new_execution_limit=1)
        if not matching:
            item.update(status='unverified', reason='Decoded input does not exactly match a trusted stored case.' if proposed else reason)
        else:
            case = matching[0]
            prior = public['trusted_evidence']['stored_results'] + ([public['trusted_evidence']['certified_trigger']] if 'certified_trigger' in public['trusted_evidence'] else [])
            existing = next((x for x in prior if x['input'] == proposed), None)
            item.update(case=case, expected=case['answer'])
            if existing:
                item.update(status='reused', observed=existing['observed'], outcome=existing['status'])
            else:
                item.update(status='pending', program=public['program'], program_sha256=f.sha256(public['program'].encode()).hexdigest())
        items.append(item)
    inputs = {str(Path(output)/'manifest.json'): f.digest(Path(output)/'manifest.json'),
        str(reviewroot/'manifest.json'): f.digest(reviewroot/'manifest.json'), str(review_import): f.digest(review_import),
        str(Path(__file__)): f.digest(__file__), str(Path(execution.__file__)): f.digest(execution.__file__)}
    f.seal(target, dict(inputs=inputs, items=items, review_import=str(Path(review_import).resolve()),
        policy='Exactly one decoded proposed input per explanation. Exact trusted input string match is required. At most one new target execution; no comparator execution is needed because the expected answer is trusted. Reuse stored evidence first.'))
    return dict(explanations=len(items), pending=sum(x['status']=='pending' for x in items), reused=sum(x['status']=='reused' for x in items))


def run(output=f.OUT):
    target = Path(output)/'witnesses'
    plan = f.load(target)
    f.load(output)
    if list(target.glob('technical-stop-*.json')):
        raise RuntimeError('Technical stop requires diagnosis; completed witness evidence remains intact')
    rows = []
    for item in plan['items']:
        path = target/'results'/f"{item['id']}.json"
        if path.exists():
            rows.append(f.read(path)); continue
        row = dict(item)
        try:
            if item['status'] == 'pending':
                view = dict(cases=[dict(item['case'], visible=True)])
                native = target/'native'/item['id']
                assert not native.exists(), 'Prior native dispatch requires explicit evidence recovery'
                verdict = f.verification.verify(view, f.identity(view), item['program'].encode(), native)
                assert verdict['status'] != 'error', 'Infrastructure errors are not witness falsification'
                from inspect_ai.log import read_eval_log_sample
                sample = read_eval_log_sample(verdict['log'], verdict['sample_id'])
                capture = sample.store['cases'][0]
                row.update(status='executed', observed=capture['stdout'], native=capture, evidence=verdict,
                    native_log_sha256=f.digest(verdict['log']), outcome=execution.classify(capture, item['case']['answer']))
            if row['status'] in ('reused','executed'):
                row['claim_checks'] = {a: dict(expected_claim_matches=claims_match(c['claimed_expected'], row['expected']),
                    actual_claim_matches=claims_match(c['claimed_actual'], row['observed'])) for a,c in row['reviewer_counterexamples'].items()}
                row['demonstrates_wrong_answer'] = row['outcome'] == 'wrong_answer'
            f.save(path, row); rows.append(row)
        except BaseException as error:
            f.save(target/'technical-stop-000.json', dict(id=item['id'], error=repr(error), dispatch_stopped=True, automatic_retry=False))
            raise
    f.save(target/'completion.json', dict(status='completed', results={str(p):f.digest(p) for p in sorted((target/'results').glob('*.json'))},
        new_executions=sum(x['status']=='executed' for x in rows), reused=sum(x['status']=='reused' for x in rows),
        unverified=sum(x['status']=='unverified' for x in rows)))
    return f.read(target/'completion.json')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare','run'])
    p.add_argument('--review-import', type=Path)
    p.add_argument('--output', type=Path, default=f.OUT)
    a=p.parse_args()
    print(prepare(a.review_import,a.output) if a.phase=='prepare' else run(a.output))
