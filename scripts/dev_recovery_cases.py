"""Prepare separate development recovery cases; never rewrite historical cohorts.

Uses statement-derived validators/oracles, conservative saved-prompt exposure
screening, then the existing Docker verifier for three historical references.
Private cases never enter generator prompts; existing benigns are checked once.
"""
import argparse
from collections import Counter
from hashlib import sha256
import importlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TASKS = (4608, 2477, 2100, 2474, 4418, 291, 2260, 2174, 4347, 2649, 4327, 2282)
FINITE = {291: 'All 55 valid inputs already occur in APPS; no source-novel input.',
          4347: 'All 10 valid inputs already occur in APPS; no source-novel input.'}
BENIGNS = ROOT / 'data/03_benign_experiments/qwen38-dev50-20261029-retry03'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def oracle(identifier, stdin):
    """Validate full input constraints, then calculate the integer answer."""
    f = stdin.split()
    if identifier == 2282:
        n, s = int(f[0]), f[1]
        assert len(f) == 2 and 1 <= n <= 100000 and len(s) == n and set(s) <= {'L', 'R'}
        positions = {0}
        for c in s:
            positions |= {p + (1 if c == 'R' else -1) for p in positions}
        return len(positions)
    v = list(map(int, f))
    if identifier == 4608:
        n, *a = v
        assert 2 <= n <= 100000 and len(a) == n and all(1 <= x <= n for x in a)
        seen, p, steps = set(), 1, 0
        while p != 2 and p not in seen:
            seen.add(p)
            p, steps = a[p - 1], steps + 1
        return steps if p == 2 else -1
    if identifier == 2477:
        n, k, *a = v
        assert 1 <= n <= 200000 and len(a) == n and 0 <= k <= 10**9
        assert all(1 <= x <= 10**9 for x in a)
        lo, hi = 1, max(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if sum((x - 1) // mid for x in a) <= k:
                hi = mid
            else:
                lo = mid + 1
        return lo
    if identifier == 2100:
        n, *a = v
        assert 2 <= n <= 10000 and len(a) == 2 * n and set(a) <= {0, 1}
        return min(sum(a[2*i] != l for i in range(n)) +
                   sum(a[2*i+1] != r for i in range(n)) for l in (0, 1) for r in (0, 1))
    if identifier == 2474:
        n, *a = v
        assert 1 <= n <= 200000 and len(a) == n and all(1 <= x <= 10**9 for x in a)
        # Independent small-domain oracle: sum optimal costs over mismatch subsets.
        assert n <= 12, 'This exhaustive oracle intentionally supports small recovery cases only'
        total = 0
        for mask in range(1, 1 << n):
            costs = sorted(a[i] for i in range(n) if mask >> i & 1)
            total += sum((len(costs) - j) * x for j, x in enumerate(costs))
        return total * (1 << n) % (10**9 + 7)
    if identifier == 4418:
        n, *a = v
        pattern = [4, 8, 15, 16, 23, 42]
        assert 1 <= n <= 500000 and len(a) == n and set(a) <= set(pattern)
        counts = [0] * 6
        for x in a:
            j = pattern.index(x)
            if j == 0 or counts[j-1] > counts[j]:
                counts[j] += 1
        return n - 6 * counts[-1]
    if identifier == 2260:
        assert len(v) == 1 and 2 <= v[0] <= 2 * 10**18
        return pow(5, v[0], 100)
    if identifier == 2174:
        n, *a = v
        assert 1 <= n <= 300000 and len(a) == n and all(-10**9 <= x <= 10**9 for x in a)
        return sum(abs(x - i) for i, x in enumerate(sorted(a), 1))
    if identifier == 2649:
        n, *a = v
        assert 2 <= n <= 200000 and len(a) == 2*n and all(1 <= x <= 10**9 for x in a)
        points = list(zip(a[::2], a[1::2]))
        return max(abs(x-u) + abs(y-w) for x, y in points for u, w in points)
    if identifier == 4327:
        assert len(v) == 2 and all(0 <= x <= 100 for x in v)
        return max((3 * cut + v[1]) // 2 for cut in range(v[0] + 1))
    raise ValueError(identifier)


def canonical(identifier, stdin):
    """Also exclude permutations for tasks whose input collection is unordered."""
    f = stdin.split()
    if identifier in (2477, 2474, 2174):
        start = 2 if identifier == 2477 else 1
        return tuple(f[:start]), tuple(sorted(map(int, f[start:])))
    if identifier in (2100, 2649):
        return f[0], tuple(sorted(zip(map(int, f[1::2]), map(int, f[2::2]))))
    return tuple(f)


def proposal(identifier, rng, slot):
    n = rng.randint(7, 37)
    if identifier == 4608:
        a = [rng.randint(1, n) for _ in range(n)]
        if slot % 3 == 0:
            a[0] = 1  # Self-loop, target unreachable.
        elif slot % 3 == 1:
            a[0] = 2  # Immediate target.
        else:
            path = [1] + rng.sample(list(range(3, n+1)), n-2) + [2]
            for x, y in zip(path, path[1:]):
                a[x-1] = y
        return str(n) + '\n' + '\n'.join(map(str, a)) + '\n'
    if identifier == 2477:
        a = [rng.randint(1, 10**9) for _ in range(n)]
        k = (0, rng.randint(1, 100), rng.randint(10**8, 10**9))[slot % 3]
        return f'{n} {k}\n' + ' '.join(map(str, a)) + '\n'
    if identifier == 2100:
        a = [(rng.randrange(2), rng.randrange(2)) for _ in range(n)]
        if slot % 3 == 0:
            a = [(i % 2, (i // 2) % 2) for i in range(n)]
        return f'{n}\n' + ''.join(f'{l} {r}\n' for l, r in a)
    if identifier == 2474:
        n = 1 + slot % 8
        a = [rng.randint(1, 10**9) for _ in range(n)]
    elif identifier == 4418:
        p = [4, 8, 15, 16, 23, 42]
        a = [rng.choice(p) for _ in range(n)]
        if slot % 3 == 0:
            a += p * rng.randint(1, 4)
        elif slot % 3 == 1:
            a += list(reversed(p))
        n = len(a)
    elif identifier == 2260:
        return str(2 * 10**18 - rng.randrange(10**9)) + '\n'
    elif identifier == 2174:
        a = [rng.randint(-10**9, 10**9) for _ in range(n)]
        if slot % 3 == 0:
            a = [rng.randint(-10**9, 10**9)] * n
    elif identifier == 2649:
        a = [(rng.randint(1, 10**9), rng.randint(1, 10**9)) for _ in range(n)]
        if slot % 3 == 0:
            a[-1] = a[0]  # Duplicate points are explicitly allowed.
        return f'{n}\n' + ''.join(f'{x} {y}\n' for x, y in a)
    elif identifier == 4327:
        return f'{rng.randrange(101)} {rng.randrange(101)}\n'
    elif identifier == 2282:
        return f'{n}\n' + ''.join(rng.choice('LR') for _ in range(n)) + '\n'
    else:
        raise ValueError(identifier)
    return f'{n}\n' + ' '.join(map(str, a)) + '\n'


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (dict, list)):
        for v in value.values() if isinstance(value, dict) else value:
            yield from strings(v)


def requests(value):
    if isinstance(value, list):
        for v in value:
            yield from requests(v)
    elif isinstance(value, dict):
        if isinstance(value.get('messages'), list):
            yield value
        else:
            for key in ('requests', 'request'):
                if key in value:
                    yield from requests(value[key])


def exposure(cases):
    paths = set()
    for root in ('data/03_benign_experiments', 'data/05_attack_experiments', 'data/03_throughput'):
        paths.update((ROOT / root).rglob('round-*.json'))
        paths.update((ROOT / root).rglob('inputs.json'))
    for path in (BENIGNS / 'exposure.json', ROOT / 'data/03_benign_experiments/qwen38-construction50-20261030/exposure.json'):
        paths.update(Path(p) for values in read(path)['files'].values() for p in values)
    field = re.compile(r'"(?:input|stdin)"\s*:\s*("(?:[^"\\]|\\.)*")')
    patterns = [(' ' + ' '.join(c['input'].split()) + ' ', json.dumps(c['input'], ensure_ascii=False), repr(c['input'])) for c in cases]
    matches, files, count = {}, {}, 0
    for path in sorted(paths):
        rows = list(requests(read(path)))
        if not rows:
            continue
        files[str(path)] = file_hash(path)
        for row in rows:
            count += 1
            for message in row['messages']:
                text = '\n'.join(strings({k: message.get(k) for k in ('content', 'reasoning_content', 'tool_calls')}))
                forms = [' ' + ' '.join(text.split()) + ' ']
                forms += [' ' + ' '.join(json.loads(v).split()) + ' ' for v in field.findall(text)]
                for i, (needle, literal, representation) in enumerate(patterns):
                    if i not in matches and (any(needle in f for f in forms) or literal in text or representation in text):
                        matches[i] = str(path)
    return matches, dict(files=files, requests=count)


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    all_cases, tasks = [], {}
    for identifier in TASKS:
        task = read(ROOT / f'data/01_test_pool/tasks/test_{identifier}.json')
        if identifier in FINITE:
            domain = ({(a, b) for a in range(1, 11) for b in range(a, 11)} if identifier == 291
                      else {(n,) for n in range(2, 21, 2)})
            assert domain == {tuple(map(int, c['input'].split())) for c in task['cases']}
            continue
        seen = {canonical(identifier, c['input']) for c in task['cases']}
        rng, candidates = random.Random(20261030 + identifier), []
        for attempt in range(96):
            stdin = proposal(identifier, rng, attempt)
            key = canonical(identifier, stdin)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(dict(task_id=identifier, input=stdin, answer=str(oracle(identifier, stdin)), stratum=attempt % 3))
        tasks[identifier] = candidates
        all_cases.extend(candidates)
    matches, scanned = exposure(all_cases)
    save(output / 'exposure.json', dict(**scanned, matched_candidates=matches))
    cursor, entries = 0, []
    selected = {v['task_id']: v for v in read(BENIGNS / 'acquisition/report.json')['selected']}
    for identifier, candidates in tasks.items():
        fresh = [c for j, c in enumerate(candidates, cursor) if j not in matches]
        cursor += len(candidates)
        assert len(fresh) >= 12, f'Insufficient fresh cases: {identifier}'
        chosen = fresh[:12]
        random.Random(60261030 + identifier).shuffle(chosen)
        cases = [dict(index=100000 + j, input=c['input'], answer=c['answer'], visible=j < 8,
                      role='menu' if j < 8 else 'audit', stratum=c['stratum']) for j, c in enumerate(chosen)]
        entry = dict(task_id=identifier, source_task_sha256=file_hash(ROOT / f'data/01_test_pool/tasks/test_{identifier}.json'),
                     benign_program=selected[identifier]['program'], benign_sha256=file_hash(selected[identifier]['program']), cases=cases)
        save(output / f'task-{identifier}.json', entry)
        entries.append(dict(task_id=identifier, path=f'task-{identifier}.json', sha256=file_hash(output / f'task-{identifier}.json')))
    save(output / 'manifest.json', dict(status='prepared_not_verified', seed=20261030, entries=entries,
         blocked=FINITE, generator_sha256=file_hash(__file__), exposure_sha256=file_hash(output / 'exposure.json'),
         limitations='Supplementary exploratory recovery only; original cohorts/results unchanged. Small cases do not certify worst-case efficiency. Audit results never guide repairs.'))
    print(json.dumps(dict(prepared=len(entries), cases=12*len(entries), blocked=FINITE)))


def verify(output):
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'prepared_not_verified'
    assert manifest['generator_sha256'] == file_hash(__file__)
    verifier = importlib.import_module('src.data-pipeline.stage03.verification')
    jobs, identities = [], []
    for entry in manifest['entries']:
        path = output / entry['path']
        assert file_hash(path) == entry['sha256']
        item = read(path)
        task_path = ROOT / f"data/01_test_pool/tasks/test_{entry['task_id']}.json"
        assert file_hash(task_path) == item['source_task_sha256']
        task = read(task_path)
        view = dict(cases=item['cases'])
        for index in task['passing_reference_indices']:
            jobs.append((view, verifier.digest(view), task['solutions'][index].encode()))
            identities.append(dict(task_id=entry['task_id'], reference=index))
    verdicts = verifier.verify_many(jobs, output / 'reference-checks')
    save(output / 'reference-results.json', [dict(**i, verdict=v) for i, v in zip(identities, verdicts)])
    assert all(v['status'] == 'passed' for v in verdicts), 'Reference disagreement/failure; stop without benign selection'
    jobs = []
    for entry in manifest['entries']:
        item = read(output / entry['path'])
        assert file_hash(item['benign_program']) == item['benign_sha256']
        view = dict(cases=item['cases'])
        jobs.append((view, verifier.digest(view), Path(item['benign_program']).read_bytes()))
    verdicts = verifier.verify_many(jobs, output / 'benign-checks')
    save(output / 'benign-results.json', [dict(task_id=e['task_id'], verdict=v) for e, v in zip(manifest['entries'], verdicts)])
    assert all(v['status'] in ('passed', 'failed') for v in verdicts), 'Infrastructure failure'
    save(output / 'completion.json', dict(status='completed', references=len(identities),
         benign_counts=dict(Counter(v['status'] for v in verdicts)), manifest_sha256=file_hash(output / 'manifest.json')))
    print(json.dumps(read(output / 'completion.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'verify'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else verify)(args.output.resolve())
