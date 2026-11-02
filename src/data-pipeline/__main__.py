"""CLI for APPS download, qualification, analysis, export and cohort publication.

Run qualification with ``/opt/homebrew/bin/python3.12 -m src.data-pipeline qualify``.
"""

import argparse
from pathlib import Path

from .paths import COHORT, POOL_ANALYSIS, QUALIFIED_POOL, RAW_APPS, TEST_POOL
from .stage01.analyze import DEFAULT_TOKENIZER, analyze
from .stage01.download import main as download_apps
from .stage01.export_test import export_test
from .stage01.qualify import MAX_CONCURRENT_CASES, qualify
from .stage02.known_use import COVERAGE, RECORDS
from .stage02.publish_cohort import publish_cohort


def main():
    """Dispatch downloads, qualification, analysis, test export and cohort publication."""
    parser = argparse.ArgumentParser(description="Pilot data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and verify pinned APPS data")
    qualification = commands.add_parser("qualify", help="Qualify references locally and export tasks")
    qualification.add_argument("--input-dir", type=Path, default=RAW_APPS)
    qualification.add_argument("--output-dir", type=Path, default=QUALIFIED_POOL)
    qualification.add_argument("--workers", type=int, choices=range(1, MAX_CONCURRENT_CASES + 1), default=MAX_CONCURRENT_CASES)
    analysis = commands.add_parser("analyze", help="Analyze qualified APPS test tasks without execution")
    analysis.add_argument("--input-dir", type=Path, default=QUALIFIED_POOL)
    analysis.add_argument("--output-dir", type=Path, default=POOL_ANALYSIS)
    analysis.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    test_export = commands.add_parser("export-test", help="Export qualified test tasks for the next stage")
    test_export.add_argument("--input-dir", type=Path, default=QUALIFIED_POOL)
    test_export.add_argument("--output-dir", type=Path, default=TEST_POOL)
    cohort = commands.add_parser("cohort", help="Freeze and publish cohort allocation without execution")
    cohort.add_argument("--input-dir", type=Path, default=TEST_POOL)
    cohort.add_argument("--analysis-dir", type=Path, default=POOL_ANALYSIS)
    cohort.add_argument("--output-dir", type=Path, default=COHORT)
    cohort.add_argument("--known-use-records", type=Path, default=RECORDS)
    cohort.add_argument("--known-use-coverage", type=Path, default=COVERAGE)
    args = parser.parse_args()
    if args.command == "download":
        download_apps()
    elif args.command == "qualify":
        manifest = qualify(args.input_dir, args.output_dir, args.workers)
        print(f"Published {len(manifest['tasks'])} tasks to {args.output_dir}")
    elif args.command == "analyze":
        analyze(args.input_dir, args.output_dir, args.tokenizer)
        print(f"Published analysis to {args.output_dir}")
    elif args.command == "export-test":
        manifest = export_test(args.input_dir, args.output_dir)
        print(f"Published {len(manifest['tasks'])} test tasks to {args.output_dir}")
    elif args.command == "cohort":
        manifest = publish_cohort(args.input_dir, args.analysis_dir, args.output_dir,
                                  args.known_use_records, args.known_use_coverage)
        print(f"Published {len(manifest['tasks'])} allocated tasks to {args.output_dir}")


if __name__ == "__main__":
    main()
