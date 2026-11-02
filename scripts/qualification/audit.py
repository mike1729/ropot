"""Execute every frozen case as its own verifier sample, preserving failure coverage."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import freeze as f
runner = importlib.import_module('src.data-pipeline.stage03.inspect_runner')
screen = importlib.import_module('src.data-pipeline.stage01.screen')


def classify(result, answer):
    status = runner.case_status(result, answer)
    if status == 'passed':
        return 'passed'
    if status == 'wrong_answer':
        return 'invalid_output' if screen.integer_answer(result['stdout']) is None else 'wrong_answer'
    return 'invalid_output' if status in ('invalid_stdout', 'stdout_limit', 'output_limit') else 'runtime_failure'


def prepare(output=f.OUT):
    manifest = f.load(output)
    target = Path(output) / 'audit'
    target.mkdir(exist_ok=False)
    jobs = []
    for pair in manifest['pairs']:
        for role in ('attack', 'benign'):
            if role == 'benign' and pair['benign_evidence']:
                continue
            for case in pair['partition']['audit']:
                jobs.append(dict(id=f"{pair['id']}:{role}:{case['index']}", pair_id=pair['id'], role=role,
                    program=pair['selection'][role + '_path'], program_sha256=pair[role + '_sha256'],
                    original_task_sha256=pair['binding']['sha256'], case=case))
    f.seal(target, dict(jobs=jobs, batch_size=16, parent=str(Path(output).resolve()),
        inputs={str(Path(output) / 'manifest.json'): f.digest(Path(output) / 'manifest.json'),
                str(Path(__file__)): f.digest(__file__), str(f.verification.__file__): f.digest(f.verification.__file__),
                str(runner.__file__): f.digest(runner.__file__)},
        policy='One case per sample; every frozen attack case executes. Technical errors stop new dispatch; failures are measurements, not selection gates.'))
    return dict(jobs=len(jobs), attacks=sum(j['role'] == 'attack' for j in jobs),
                benigns=sum(j['role'] == 'benign' for j in jobs))


def candidates(jobs):
    result = []
    for job in jobs:
        if f.digest(job['program']) != job['program_sha256']:
            raise ValueError('Program changed')
        view = dict(cases=[dict(job['case'], visible=True)])
        result.append((view, f.identity(view), Path(job['program']).read_bytes()))
    return result


def collect(jobs, verdicts):
    from inspect_ai.log import read_eval_log_sample
    rows = []
    for job, verdict in zip(jobs, verdicts, strict=True):
        if verdict['status'] == 'error':
            raise RuntimeError('Native infrastructure error; not a program failure')
        sample = read_eval_log_sample(verdict['log'], verdict['sample_id'])
        if sample.error or len(sample.store['cases']) != 1:
            raise RuntimeError('Incomplete native case capture')
        native = sample.store['cases'][0]
        rows.append(dict(id=job['id'], pair_id=job['pair_id'], role=job['role'], case=job['case'],
            category=classify(native, job['case']['answer']), native=native,
            evidence=verdict, native_log_sha256=f.digest(verdict['log'])))
    return rows


def run(output=f.OUT, recovery=None):
    target = Path(output) / 'audit'
    plan = f.load(target)
    f.load(output)
    failures = sorted(target.glob('technical-stop-*.json'))
    if failures:
        if recovery is None:
            raise RuntimeError('Diagnosis required after technical stop; no automatic retry')
        diagnosis = f.read(recovery)
        assert diagnosis['technical_stop_sha256'] == f.digest(failures[-1]) and diagnosis['reason']
        f.save(target / ('recovery-' + f.digest(recovery)[:16] + '.json'), diagnosis)
    for start in range(0, len(plan['jobs']), plan['batch_size']):
        batch = start // plan['batch_size']
        result = target / f'results/batch-{batch:04}.json'
        jobs = plan['jobs'][start:start + plan['batch_size']]
        if result.exists():
            saved = f.read(result)
            assert saved['manifest_sha256'] == f.digest(target / 'manifest.json')
            assert [r['id'] for r in saved['rows']] == [j['id'] for j in jobs]
            for row in saved['rows']:
                assert f.digest(row['evidence']['log']) == row['native_log_sha256']
            continue
        native_dir = target / f'native/batch-{batch:04}'
        try:
            f.check(plan['inputs'])
            work = candidates(jobs)
            if native_dir.exists():
                # Recover already completed evidence instead of paying for it again.
                from inspect_ai.log import list_eval_logs, read_eval_log
                logs = list_eval_logs(str(native_dir))
                assert len(logs) == 1 and recovery, 'Unresolved prior native dispatch'
                log = read_eval_log(logs[0].name)
                assert log.status == 'success' and not log.error
                verdicts = [f.verification.derive(log, f.verification.request(*w), log.eval.metadata, sample_id=i) for i, w in enumerate(work)]
            else:
                verdicts = f.verification.verify_many(work, native_dir)
            rows = collect(jobs, verdicts)
            f.save(result, dict(manifest_sha256=f.digest(target / 'manifest.json'), rows=rows))
            print(f'Batch {batch + 1}: {len(rows)} cases; {dict(Counter(r["category"] for r in rows))}', flush=True)
        except BaseException as error:
            f.save(target / f'technical-stop-{len(failures):03}.json', dict(batch=batch, error=repr(error),
                timestamp=datetime.now(timezone.utc).isoformat(), completed_batches=len(list(target.glob('results/*.json'))),
                outcome='infrastructure_error', dispatch_stopped=True, automatic_retry=False))
            raise
    f.save(target / 'completion.json', dict(status='completed', manifest_sha256=f.digest(target / 'manifest.json'),
        results={str(p): f.digest(p) for p in sorted(target.glob('results/*.json'))}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['prepare', 'run'])
    p.add_argument('--output', type=Path, default=f.OUT)
    p.add_argument('--recovery', type=Path)
    a = p.parse_args()
    print(prepare(a.output) if a.phase == 'prepare' else run(a.output, a.recovery))
