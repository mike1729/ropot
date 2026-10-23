"""CLI for pinned APPS downloads and local reference qualification.

Run qualification with ``/opt/homebrew/bin/python3.12 -m src.data-pipeline qualify``.
"""

import argparse
from pathlib import Path

from .download import main as download_apps
from .qualify import DESTINATION, MAX_CONCURRENT_CASES, OUTPUT, qualify


def main():
    """Parse pipeline options and dispatch downloads, qualification or analysis."""
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
    args = parser.parse_args()
    if args.command == "qualify":
        manifest = qualify(args.input_dir, args.output_dir, args.workers)
        print(f"Published {len(manifest['tasks'])} tasks to {args.output_dir}")
    elif args.command == "analyze":
        from .analyze import analyze
        analyze(args.input_dir, args.output_dir, args.tokenizer)
        print(f"Published analysis to {args.output_dir}")
    else:
        args.run()


if __name__ == "__main__":
    main()
