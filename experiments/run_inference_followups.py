"""Run a frozen, reviewed follow-up matrix with one GPU worker at a time."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import subprocess

from . import run_inference_benchmarks as driver
from .inference_benchmark import save


def run(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    sources = [*list((driver.ROOT/'experiments').glob('*.py')),
        *(driver.ROOT/'src/data-pipeline/stage03'/n for n in ('verification.py', 'inspect_runner.py')),
        driver.ROOT/'docs/inference-followup-plan.md']
    driver.FROZEN.update({p: sha256(p.read_bytes()).hexdigest() for p in sources})
    for path in sources:
        copy = root/'sources'/path.relative_to(driver.ROOT)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    save(root/'plan.json', dict(started=datetime.now(timezone.utc).isoformat(),
        matrix='docs/inference-followup-plan.md', reviewed=True,
        source_sha256={str(p.relative_to(driver.ROOT)): h for p,h in driver.FROZEN.items()},
        profiles=driver.PROFILES, soft_seconds={'long':2100, 'semantic':1500},
        hard_seconds={'long':2160, 'semantic':1560}, independent_new_run=True))
    save(root/'host.json', dict(power=subprocess.run(['pmset','-g','batt'],capture_output=True,
        text=True,timeout=5).stdout))
    failures = []
    try:
        for profile in driver.PROFILES:
            for mode, soft, hard in [('long',2100,2160), ('semantic',1500,1560)]:
                name = f'{profile}-{mode}'
                try:
                    driver.wait([driver.launch(root, name, [driver.MLX, '-m',
                        'experiments.followup_benchmark', '--profile', profile, '--mode', mode,
                        '--output', str(root/name), '--seconds', str(soft)], hard)])
                except (RuntimeError, TimeoutError) as error:
                    failures.append(dict(worker=name, error=repr(error)))
                    save(root/'failures.json', failures)
                    if driver.ACTIVE:
                        raise
        driver.wait([driver.launch(root, 'verification-report', [driver.PYTHON, '-m',
            'experiments.followup_report', '--root', str(root), '--verify'], 1200)])
    finally:
        if driver.ACTIVE:
            driver.finish(list(driver.ACTIVE), True)
    save(root/'complete.json', dict(failures=failures, process_groups_absent=not driver.ACTIVE))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
