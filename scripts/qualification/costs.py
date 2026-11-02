"""Reconcile v2 executions at native leaf level; never sum copied report totals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, str) and '/data/' in value and value.startswith('/'):
        return '/data/' + value.rsplit('/data/', 1)[1]
    return value


def roots():
    selected = []
    for folder, pattern in [('03_benign_experiments', r'^(v2-|qwen38-|gemma4-)'),
                            ('03_generation_checks', r'^(v2-|qwen38-)'),
                            ('03_throughput', r'^v2-'),
                            ('05_attack_experiments', r'^(v2-|qwen38-|qwen-dev-|dev-construction-|multi-edit-|mechanism-|overnight-|relaxed-|semantic-)'),
                            ('06_monitor_experiments', r'.')]:
        selected += [p for p in (f.ROOT / 'data' / folder).iterdir() if p.is_dir() and re.search(pattern, p.name)]
    return sorted(selected)


def reconcile(output=f.OUT, name='costs-initial'):
    manifest = f.load(output)
    target = Path(output) / name
    target.mkdir(exist_ok=False)
    inputs, records, wrappers, missing, generation = {}, {}, [], [], {}
    scope = roots() + [Path(output) / 'reviews']
    def add(key, row, path):
        if key in records:
            records[key]['copies'].append(str(path))
            return
        records[key] = dict(row, copies=[str(path)])
    for root in scope:
        if not root.exists():
            continue
        for path in sorted(root.rglob('outcome.json')):
            data = f.read(path)
            inputs[str(path)] = f.digest(path)
            if 'attempts' in data:
                # Attempt statistics are distinct from native execution token costs.
                key = f.identity(canonical(data))
                if key not in generation:
                    attempts = data['attempts']
                    generation[key] = dict(path=str(path), state=data.get('status'),
                        reserved_turns=data.get('committed_replies'), completed_submissions=data.get('known_submissions_lower_bound'),
                        capped_turns=sum(bool(a.get('hit_output_cap', a.get('generation_status') == 'length')) for a in attempts),
                        extraction_failures=sum(a.get('state') == 'extraction_failed' or a.get('extraction') not in (None, 'ok', 'success') and not a.get('candidate') for a in attempts),
                        exhausted_chains=sum(t['state'] in ('repairs_exhausted', 'exhausted') for t in data.get('tasks', [])),
                        accepted_tasks=sum(t['state'] == 'passed' for t in data.get('tasks', [])),
                        task_chains=len(data.get('tasks', [])), output_tokens_lower_bound=data.get('observed_tokens'),
                        usage_complete=data.get('usage_complete'))
                continue
            if 'requests' not in data:
                continue
            if 'chunks' in data:
                wrappers.append(dict(path=str(path), chunks=data['chunks'], counted=False))
                for chunk in data['chunks']:
                    childroot = chunk.get('output')
                    if childroot is None:
                        specification = Path(chunk['specification'])
                        if not specification.exists():
                            specification = path.parent.parent / specification.name
                        childroot = f.read(specification)['output_dir'] if specification.exists() else ''
                    child = Path(childroot) / 'outcome.json'
                    if not child.exists():
                        # A copied wrapper can point to a removed checkout; its local sibling is preferred.
                        child = path.parent.parent / Path(childroot).name / 'outcome.json'
                    if not child.exists():
                        missing.append(dict(path=str(path), component='chunk outcome missing', chunk=chunk))
                continue
            native = data.get('native') or {}
            count = native.get('generation_tokens')
            if count is None:
                count = sum(r.get('generation_tokens', len(r.get('tokens', []))) for r in data['requests'])
                missing.append(dict(path=str(path), component='native cumulative generation counters unavailable; completed captures only'))
            key = 'mlx:' + f.identity(canonical(data))
            add(key, dict(kind='local_generation', output_tokens=count,
                input_tokens=native.get('prompt_tokens'), status=data.get('status'),
                submitted=sum(r.get('submitted', True) for r in data['requests']),
                completed_replies=sum(r.get('status') in ('stop', 'length') for r in data['requests']),
                native_seconds=native.get('wall_time', data.get('elapsed_seconds')),
                list_price_equivalent_usd=None, actual_payment_usd=None), path)
        for path in sorted(root.rglob('stdout.json')):
            raw = path.read_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or 'modelUsage' not in data or 'usage' not in data:
                continue
            inputs[str(path)] = f.digest(path)
            usage = data['usage'] or {}
            # Session identity distinguishes identical prompts deliberately executed twice.
            key = 'claude:' + str(data.get('session_id') or f.identity(canonical(data)))
            add(key, dict(kind='claude_review', status='error' if data.get('is_error') else 'completed',
                usage=usage, output_tokens=usage.get('output_tokens'), input_tokens=usage.get('input_tokens'),
                cache_creation_input_tokens=usage.get('cache_creation_input_tokens'),
                cache_read_input_tokens=usage.get('cache_read_input_tokens'),
                models=data.get('modelUsage'), list_price_equivalent_usd=data.get('total_cost_usd'),
                actual_payment_usd=None), path)
        for path in sorted(root.rglob('stdout.jsonl')):
            events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not any(x.get('type') == 'turn.completed' for x in events):
                continue
            inputs[str(path)] = f.digest(path)
            key = 'codex:' + f.identity(canonical(events))
            usages = [e.get('usage', {}) for e in events if e.get('type') == 'turn.completed']
            add(key, dict(kind='codex_review', status='completed', usage=usages,
                output_tokens=sum(u.get('output_tokens', 0) for u in usages),
                input_tokens=sum(u.get('input_tokens', 0) for u in usages),
                list_price_equivalent_usd=None, actual_payment_usd=None), path)
    # Report original/recovery reservations separately and freeze all current phase recipes.
    latest = []
    for phase in f.read(f.REV / 'plan.json')['phases']:
        root = f.REV / phase['output']
        report = f.read(root / 'report.json')
        experiment = f.read(root / 'experiment.json')
        for p in (root / 'report.json', root / 'experiment.json'):
            inputs[str(p)] = f.digest(p)
        latest.append(dict(root=str(root), report=report, frozen_recipe={k: experiment[k] for k in
            ('recipe', 'runtime', 'chains', 'batching', 'resource_guard', 'sources', 'reply_limit', 'output_token_limit')}))
    assert sum(x['report']['truncated_submissions'] for x in latest) == 330
    assert sum(x['report']['reserved_replies'] for x in latest) == 808
    # Explicitly retain missing orchestration and unlogged CLI usage; do not manufacture totals.
    missing += [dict(component='Interactive Codex orchestration/review turns', reason='No complete per-turn native usage export for the campaign.'),
        dict(component='Historical manual/agent Codex semantic reviews', reason='Saved judgments do not provide complete model token usage.'),
        dict(component='Pre-B1 reused runtime/checkpoint qualification', reason='Inherited evidence is not a new execution; earlier cost outside the B1 onward window remains separately unattributed.'),
        dict(component='Subscription invoices, local hardware/energy and actual payment', reason='Execution token usage and CLI list-price estimates do not establish payment.')]
    totals = {}
    for kind in ('local_generation', 'claude_review', 'codex_review'):
        rows = [r for r in records.values() if r['kind'] == kind]
        totals[kind] = dict(executions=len(rows), output_tokens_lower_bound=sum(r.get('output_tokens') or 0 for r in rows),
            input_tokens_lower_bound=sum(r.get('input_tokens') or 0 for r in rows),
            list_price_equivalent_usd_lower_bound=sum(r.get('list_price_equivalent_usd') or 0 for r in rows),
            actual_payment_usd=None)
    f.save(target / 'ledger.json', dict(scope=[str(p) for p in scope], records=records, totals=totals,
        output_tokens_lower_bound=sum(r.get('output_tokens') or 0 for r in records.values()),
        copied_records_deduplicated=sum(len(r['copies']) - 1 for r in records.values()),
        wrappers_not_counted=wrappers, generation_statistics=list(generation.values()), current_generation_recipes=latest,
        pipeline_total_complete=False, missing_components=missing,
        denominator=dict(original_source_task_pairs=92, joint_development_tasks=20,
            retained_joint_denominator='Defined after derived exclusions; original costs are never removed.'),
        cost_gate=dict(status='unresolved', threshold_output_tokens_per_joint_task=100000,
            reason='Complete numerator, including historical Codex reviews and orchestration, is unavailable. Do not divide a lower bound and call it a passing gate.'),
        accounting='Native leaf outcomes count once; copied snapshots deduplicate by normalized native execution identity including timings. Chunk wrappers and report summaries add no execution cost. Distinct native sessions are never collapsed by prompt hash alone.',
        payment='Claude total_cost_usd is reported list-price equivalent. Actual invoice/payment is unknown, recorded separately. No dollar spending gate.',
        timestamp=datetime.now(timezone.utc).isoformat()))
    inputs[str(Path(__file__))] = f.digest(__file__)
    inputs[str(Path(output) / 'manifest.json')] = f.digest(Path(output) / 'manifest.json')
    inputs[str(target / 'ledger.json')] = f.digest(target / 'ledger.json')
    f.seal(target, dict(inputs=inputs, parent_manifest_sha256=f.digest(Path(output) / 'manifest.json')))
    return dict(totals=totals, deduplicated=sum(len(r['copies']) - 1 for r in records.values()), missing=len(missing))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=f.OUT)
    p.add_argument('--name', default='costs-initial')
    a = p.parse_args()
    print(json.dumps(reconcile(a.output, a.name)))
