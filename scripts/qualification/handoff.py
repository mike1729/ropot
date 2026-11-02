"""Create the concise handoff and complete, blinded user-adjudication packets."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import freeze as f


def build(output=f.OUT):
    root=Path(output);f.load(root/'report');f.load(root/'costs');f.load(root/'reviews')
    target=root/'handoff';target.mkdir(exist_ok=False)
    grounding=f.read(root/'report/grounding.json');view=f.read(root/'report/qualified-view.json');cost=f.read(root/'costs/ledger.json')
    packets={e['id']:e for p in (root/'reviews/packets').glob('*.json') for e in f.read(p)['explanations']}
    witnesses={w['id']:w for w in grounding['witnesses']}
    cases=[dict(packet=packets[i],independent_reviews=x['reviews'],witness=witnesses[i],
                conservative_consensus='unassessed',user_adjudication=None) for i,x in grounding['unresolved_disagreements'].items()]
    f.save(target/'adjudication-packets.json',dict(explanations=cases,source_mapping_remains_private=True,automatic_resolution=False))
    local=cost['totals']['local_generation']['output_tokens_lower_bound'];claude=cost['totals']['claude_review'];codex=cost['totals']['codex_review']
    text=f'''# Qualification handoff

Completed with explicit unresolved items. The original **92 pairs / 184 programs** are preserved.
The derived view retains **91 pairs**; Qwen/task 886 is excluded for an independently agreed benign correctness defect.

| Measurement | Result |
| --- | ---: |
| Attack audit cases | 358 |
| Correct / wrong integer / invalid output | 247 / 109 / 2 |
| Attacks passing all available audit cases | 26/92 |
| Independent final-FLAG review decisions | 352/352 |
| Attack FLAGs: certified / other defect / incorrect / unassessed | 159 / 0 / 6 / 4 |
| Benign FLAGs: false alarms / demonstrated defects | 6 / 1 |
| Pending user adjudications | 4 |
| New / reused / unverified witness checks | 3 / 5 / 168 |

Every monitor/source cell passes conservative majority-correct grounding. Correct fault reasoning does not certify counterexample output claims.
The exclusion leaves **20 matched development tasks, 14 construction tasks, 13 nonweak construction tasks**. Reduced audit coverage remains labelled; no attack was excluded for audit survival.

The historical **330/808 caps (40.84%) failed** the old ≤5% gate. The prospective continuation contract and fresh 512-thinking + eligible verdict-first 512 monitor policy are frozen. Historical monitor variants remain separate.

| Recorded output-token lower bound | Tokens |
| --- | ---: |
| Local generation, including failed/exploratory/fixture/scoring work | {local:,} |
| Saved Claude reviews | {claude['output_tokens_lower_bound']:,} |
| New isolated Codex grounding reviews | {codex['output_tokens_lower_bound']:,} |
| Total known lower bound | {cost['output_tokens_lower_bound']:,} |

Claude reported **${cost['new_grounding_review_usage']['claude_list_price_equivalent_usd']:.2f}** for this qualification's reviews; **${claude['list_price_equivalent_usd_lower_bound']:.2f}** across recorded campaign reviews. These are list-price equivalents. Actual payment is unknown. Historical interactive/manual Codex usage and named compute/accounting components are missing, so the full token-cost gate remains unresolved.

Ready for the **next-stage training plan**: frozen recipes and exact programs, original ownership/partitions, native audits, independent grounding and derived exclusions. **Training launch remains blocked** by incomplete costs, heterogeneous historical monitor scoring, unresolved labels, and the missing training target/reasoning/adapter compatibility contract and runtime smoke. Numerical clean-panel and construction limitations remain in the readiness table.

- [Full report](../report/REPORT.md), [readiness JSON](../report/readiness.json), [derived view](../report/qualified-view.json).
- [Four complete adjudication packets](adjudication-packets.json), [grounding and witness evidence](../report/grounding.json).
- [Cost ledger](../costs/ledger.json), [recipes](../recipes/recipes.json), [case-level audits](../report/audit-results.json).

No attack generation, monitor rerun, confirmation, training, repair experiment, training export or promotion occurred. All private evidence remains local and ignored by Git.
'''
    (target/'REPORT.md').write_text(text)
    paths=[root/'report/manifest.json',root/'costs/manifest.json',root/'reviews/manifest.json',Path(__file__),target/'REPORT.md',target/'adjudication-packets.json']
    f.seal(target,dict(inputs={str(p):f.digest(p) for p in paths},complete_with_explicit_unresolved_items=True))
    return str(target/'REPORT.md')


if __name__=='__main__':print(build())
