"""CLI for pinned APPS downloads and local reference qualification.

Run qualification with ``/opt/homebrew/bin/python3.12 -m src.data-pipeline qualify``.
"""

import argparse
from pathlib import Path

from .download import main as download_apps
from .qualify import DESTINATION, MAX_CONCURRENT_CASES, OUTPUT, qualify


def main():
    """Dispatch downloads, qualification, analysis, test export and cohort publication."""
    parser = argparse.ArgumentParser(description="Pilot data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify pinned APPS data")
    download.set_defaults(run=download_apps)
    qualification = commands.add_parser("qualify", help="Qualify references locally and export tasks")
    qualification.add_argument("--input-dir", type=Path, default=DESTINATION)
    qualification.add_argument("--output-dir", type=Path, default=OUTPUT)
    qualification.add_argument("--workers", type=int, choices=range(1, MAX_CONCURRENT_CASES + 1), default=MAX_CONCURRENT_CASES)
    qualification.set_defaults(run=None)
    from .analyze import DEFAULT_TOKENIZER, OUTPUT as ANALYSIS_OUTPUT
    analysis = commands.add_parser("analyze", help="Analyze qualified APPS test tasks without execution")
    analysis.add_argument("--input-dir", type=Path, default=OUTPUT)
    analysis.add_argument("--output-dir", type=Path, default=ANALYSIS_OUTPUT)
    analysis.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    from .export_test import OUTPUT as TEST_OUTPUT
    test_export = commands.add_parser("export-test", help="Export qualified test tasks for the next stage")
    test_export.add_argument("--input-dir", type=Path, default=OUTPUT)
    test_export.add_argument("--output-dir", type=Path, default=TEST_OUTPUT)
    from .known_use import COVERAGE, RECORDS
    from .publish_cohort import OUTPUT as COHORT_OUTPUT
    cohort = commands.add_parser("cohort", help="Freeze and publish cohort allocation without execution")
    cohort.add_argument("--input-dir", type=Path, default=TEST_OUTPUT)
    cohort.add_argument("--analysis-dir", type=Path, default=ANALYSIS_OUTPUT)
    cohort.add_argument("--output-dir", type=Path, default=COHORT_OUTPUT)
    cohort.add_argument("--known-use-records", type=Path, default=RECORDS)
    cohort.add_argument("--known-use-coverage", type=Path, default=COVERAGE)
    args = parser.parse_args()
    if args.command == "qualify":
        manifest = qualify(args.input_dir, args.output_dir, args.workers)
        print(f"Published {len(manifest['tasks'])} tasks to {args.output_dir}")
    elif args.command == "analyze":
        from .analyze import analyze
        analyze(args.input_dir, args.output_dir, args.tokenizer)
        print(f"Published analysis to {args.output_dir}")
    elif args.command == "export-test":
        from .export_test import export_test
        manifest = export_test(args.input_dir, args.output_dir)
        print(f"Published {len(manifest['tasks'])} test tasks to {args.output_dir}")
    elif args.command == "cohort":
        from .publish_cohort import publish_cohort
        manifest = publish_cohort(args.input_dir, args.analysis_dir, args.output_dir,
                                  args.known_use_records, args.known_use_coverage)
        print(f"Published {len(manifest['tasks'])} allocated tasks to {args.output_dir}")
    else:
        args.run()


if __name__ == "__main__":
    main()
