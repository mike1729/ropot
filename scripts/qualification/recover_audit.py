"""Diagnosed classifier repair; recover successful native logs without redispatch."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit
import freeze as f
from urllib.parse import unquote, urlparse

original_collect = audit.collect


def collect(jobs, verdicts):
    for verdict in verdicts:
        if verdict['log'].startswith('file:'):
            verdict['log'] = unquote(urlparse(verdict['log']).path)
    return original_collect(jobs, verdicts)


def classify(native, answer):
    status = audit.runner.case_status(native, answer)
    if status == 'passed':
        return 'passed'
    if status == 'wrong_answer':
        try:
            audit.screen.integer_answer(native['stdout'])
        except audit.screen.Rejected:
            return 'invalid_output'
        return 'wrong_answer'
    return 'invalid_output' if status in ('invalid_stdout', 'stdout_limit', 'output_limit') else 'runtime_failure'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('diagnosis', type=Path)
    p.add_argument('--output', type=Path, default=f.OUT)
    a = p.parse_args()
    diagnosis = f.read(a.diagnosis)
    assert diagnosis['adapter_sha256'] == f.digest(__file__)
    assert diagnosis['adapter_path'] == str(Path(__file__).resolve())
    audit.classify = classify
    audit.collect = collect
    audit.run(a.output, a.diagnosis)
