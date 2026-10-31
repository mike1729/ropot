"""Saved conversations only; identity from metadata or statements, unknowns fail closed."""
import json
import re
import sys
from pathlib import Path
from scripts import dev_recovery_cases as u
sys.path.insert(0, str(u.ROOT/'scripts'))
from scripts import review_task291_exposure as contextual


def identities(value, text, prefixes):
    ids = set(re.findall(r'apps:test:(\d+)\b', json.dumps(value)))
    for m in prefixes.pattern.finditer(' '.join(text.split())):
        ids.update(prefixes[m.group(0)])
    return {int(i) for i in ids}


class Prefixes(dict):
    def __init__(self, tasks):
        super().__init__()
        for i,t in tasks.items():
            self.setdefault(' '.join(t['question'].split())[:150], set()).add(str(i))
        self.pattern = re.compile('|'.join(re.escape(k) for k in sorted(self, key=len, reverse=True)))


def scan(candidates, scoped):
    # Reuse the original file inventory, including historical frozen exposure lists.
    _, original = u.exposure([])
    paths = set(map(Path, original['files']))
    paths.update((u.ROOT/'data/06_monitor_experiments').rglob('*.json'))
    tasks = {int(p.stem.split('_')[-1]): u.read(p) for p in (u.ROOT/'data/01_test_pool/tasks').glob('test_*.json')}
    prefixes = Prefixes(tasks)
    review_path = u.ROOT/'data/03_benign_experiments/construction-context-review-20261030.json'
    reviewed = u.read(review_path)
    assert set(reviewed['scope']) == scoped
    matches, files, contexts, unknown, count = {}, {str(review_path):u.file_hash(review_path)}, {}, {}, 0
    cache = {}
    patterns = [(' '+' '.join(c['input'].split())+' ', json.dumps(c['input'],ensure_ascii=False), repr(c['input'])) for c in candidates]
    for path in sorted(paths):
        value = u.read(path); rows = list(u.requests(value))
        if not rows: continue
        files[str(path)] = u.file_hash(path)
        records = value.get('records', []) if isinstance(value, dict) else []
        for position, row in enumerate(rows):
            count += 1
            texts = ['\n'.join(u.strings({k:m.get(k) for k in ('content','reasoning_content','tool_calls')})) for m in row['messages']]
            text = '\n'.join(texts)
            metadata = [row.get(k) for k in ('task_id','binding','metadata')]
            if len(records) == len(rows): metadata.append(records[position])
            digest = u.sha256((json.dumps(metadata)+text).encode()).hexdigest()
            if digest not in cache:
                ids = identities(metadata, text, prefixes)
                if not ids and u.sha256(text.encode()).hexdigest() in reviewed['unrelated_conversations']:
                    ids = {-1}  # Exact conversation reviewed as unrelated; only scalar exceptions use identity.
                forms = [f for t in texts for f in contextual.forms(t)]
                hit = [j for j,(c,(needle,literal,representation)) in enumerate(zip(candidates,patterns)) if
                       (c['task_id'] not in scoped or not ids or c['task_id'] in ids) and
                       (any(needle in f for f in forms) or any(literal in t or representation in t for t in texts))]
                cache[digest] = ids, hit
            ids, hit = cache[digest]
            for j in hit: matches.setdefault(j, str(path))
            for i in scoped & ids:
                for t in texts: contexts.setdefault(str(i), {})[u.sha256(t.encode()).hexdigest()] = t
            if not ids: unknown.setdefault(digest, dict(path=str(path), request=position, matching_candidates=hit))
    return matches, dict(files=files, requests=count, contexts=contexts, unidentified_requests=list(unknown.values()),
                         policy='Global match exclusion except scalar tasks with positively identified other-task conversations; unidentified conversations use global exclusion.')
