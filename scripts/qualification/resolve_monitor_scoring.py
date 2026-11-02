"""Append the user's scoring-readiness resolution without rewriting frozen evidence."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f


def resolve(output=f.OUT):
    root=Path(output)
    f.load(root/'report');f.load(root/'handoff')
    target=root/'status-monitor-resolved'
    target.mkdir(exist_ok=False)
    readiness=deepcopy(f.read(root/'report/readiness.json'))
    old=next(c for c in readiness['checks'] if c['requirement']=='Uniform pooled monitor policy')
    old.update(previous_requirement=old['requirement'],previous_status=old['status'],
        requirement='Monitor scoring accepted for the existing corpus',status='pass',
        evidence='User marked monitor scoring resolved. Existing pilot/retry variants are accepted for this corpus and remain labelled; historical uniformity is not claimed. No rerun is required by this resolution.')
    readiness['blockers']=[b for b in readiness['blockers'] if b!='Heterogeneous historical monitor policy']
    readiness['blockers']=[('Four unresolved explanation-grounding reviews; attack/benign truth labels are unchanged'
        if b.startswith('Unresolved explanation disagreements') else b) for b in readiness['blockers']]
    readiness['resolutions']=dict(monitor_scoring=dict(status='resolved',authority='user',
        instruction='Mark monitor scoring as resolved. What is with labels?',
        timestamp=datetime.now(timezone.utc).isoformat(),historical_uniformity=False,new_monitor_calls=0))
    f.save(target/'readiness.json',readiness)
    summary=(root/'handoff/REPORT.md').read_text()
    summary=summary.replace('Historical monitor variants remain separate.',
        'Monitor scoring is **resolved by user decision** for this corpus. Historical monitor variants remain separate; no uniform-policy claim is made.')
    summary=summary.replace('incomplete costs, heterogeneous historical monitor scoring, unresolved labels, and',
        'incomplete costs, four unresolved explanation-grounding reviews, and')
    summary=summary.replace('[readiness JSON](../report/readiness.json)','[current readiness JSON](readiness.json)')
    summary=summary.replace('[Four complete adjudication packets](adjudication-packets.json)',
        '[Four complete adjudication packets](../handoff/adjudication-packets.json)')
    summary+='\nThe four pending labels concern explanation grounding and counterexample claims. They do not change the selected attacks’ attack/benign truth labels. Three disputes concern credit for self-corrections after wrong output claims; one concerns certified fault versus another real defect. Conservative explanation consensus remains unassessed pending adjudication.\n'
    (target/'REPORT.md').write_text(summary)
    paths=[root/'report/manifest.json',root/'report/readiness.json',root/'handoff/manifest.json',root/'handoff/REPORT.md',
        Path(__file__),target/'readiness.json',target/'REPORT.md']
    f.seal(target,dict(inputs={str(p):f.digest(p) for p in paths},purpose='User-authorized scoring readiness resolution; immutable earlier evidence retained.'))
    f.load(target)
    assert not any('Heterogeneous' in b for b in readiness['blockers'])
    assert next(c for c in readiness['checks'] if c['requirement']=='Grounding disagreements resolved')['status']=='unresolved'
    return str(target/'readiness.json')


if __name__=='__main__':
    print(resolve())
