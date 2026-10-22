"""Run with python3 -m src.data-pipeline."""

import argparse

from .download import main as download_apps


def main():
    parser = argparse.ArgumentParser(description="Pilot data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download and verify pinned APPS data")
    download.set_defaults(run=download_apps)
    args = parser.parse_args()
    args.run()


if __name__ == "__main__":
    main()
