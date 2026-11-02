"""Final token ledger plus explicit verification compute lower bounds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import costs
import freeze as f


def finalize(output=f.OUT):
    # Reconcile once more after reviews, witnesses and reporting implementation finish.
    base=Path(output)/'costs-final-native'
    costs.reconcile(output,base.name)
    f.load(base)
    ledger=f.read(base/'ledger.json')
    complete=f.read(Path(output)/'audit/completion.json')
    f.check(complete['results'])
    audits=[r for path in complete['results'] for r in f.read(path)['rows']]
    witnesses=f.read(Path(output)/'witnesses/completion.json')
    f.check(witnesses['results'])
    checks=[f.read(p) for p in witnesses['results']]
    new=[r for r in checks if r['status']=='executed']
    ledger['verification_compute_lower_bound']=dict(new_original_audit_program_case_executions=len(audits),
        attack_cases=358,new_original_benign_cases=18,witness_target_executions=len(new),
        reused_original_benign_pairs=85,reused_witness_captures=witnesses['reused'],
        observed_case_elapsed_seconds=sum(r['native']['elapsed_seconds'] for r in audits+new),
        inference_output_tokens=0,actual_payment_usd=None,
        missing='Historical verifier CPU/wall time, Docker startup/teardown, hardware and energy costs are not complete. Reuse incurs no new program execution.')
    ledger['missing_components'].append(dict(component='Historical verifier compute and report/orchestration costs',
        reason='Current native case timings are retained as a lower bound; no complete historical compute billing or interactive model usage export is available.'))
    mapping=f.read(Path(output)/'reviews/mapping-private.json')
    calls=Path(output)/'reviews/calls'
    records=[f.read(p) for p in calls.glob('*/*/record.json')]
    assert len(records)==44 and all(r['status']=='completed' for r in records)
    assert len(mapping)==176
    ledger['new_grounding_review_usage']=dict(requests=44,claude_requests=22,codex_requests=22,
        claude_list_price_equivalent_usd=sum(r.get('list_price_equivalent_usd') or 0 for r in records),
        claude_reported_usage=[r['usage'] for r in records if r['auditor']=='claude'],
        codex_reported_usage=[r['usage'] for r in records if r['auditor']=='codex'],actual_payment_usd=None)
    target=Path(output)/'costs'
    target.mkdir(exist_ok=False)
    f.save(target/'ledger.json',ledger)
    paths=[base/'manifest.json',base/'ledger.json',Path(output)/'audit/completion.json',Path(output)/'witnesses/completion.json',Path(__file__),target/'ledger.json']
    f.seal(target,dict(inputs={str(p):f.digest(p) for p in paths},inherited_native_ledger_manifest_sha256=f.digest(base/'manifest.json')))
    return dict(output_tokens_lower_bound=ledger['output_tokens_lower_bound'],totals=ledger['totals'],
        current_review_list_price_equivalent_usd=ledger['new_grounding_review_usage']['claude_list_price_equivalent_usd'])


if __name__=='__main__':
    print(finalize())
