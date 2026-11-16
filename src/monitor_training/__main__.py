"""Prepare the pinned monitor-training view without model or execution dependencies."""

import argparse
from pathlib import Path

from .prepare import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Verify and freeze the selected training view")
    command.add_argument("--export-root", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--expected-export-sha256", required=True)
    command.add_argument("--expected-allocation-sha256", required=True)
    args = parser.parse_args()
    print(prepare(args.export_root, args.output_dir, expected_export_sha256=args.expected_export_sha256,
                  expected_allocation_sha256=args.expected_allocation_sha256))


if __name__ == "__main__":
    main()
