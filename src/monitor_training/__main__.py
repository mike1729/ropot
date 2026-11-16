"""Prepare, freeze, score or smoke-test the binary monitor's NVIDIA recipe."""

import argparse
import json
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
    command = commands.add_parser("freeze-runtime", help="Pin a local checkpoint and the chosen NVIDIA environment")
    command.add_argument("--model-root", type=Path, required=True)
    command.add_argument("--settings", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = commands.add_parser("score", help="Run one bounded 60-row base qualification")
    for name in ("prepared-root", "export-root", "model-root", "contract", "output-dir"):
        command.add_argument("--" + name, type=Path, required=True)
    command.add_argument("--expected-prepared-sha256", required=True)
    command.add_argument("--expected-contract-sha256", required=True)
    command = commands.add_parser("smoke", help="Qualify disposable LoRA weights with three updates and resume")
    for name in ("prepared-root", "export-root", "model-root", "inference-contract", "settings", "output-dir"):
        command.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared", "inference"):
        command.add_argument("--expected-" + name + "-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.export_root, args.output_dir, expected_export_sha256=args.expected_export_sha256,
                      expected_allocation_sha256=args.expected_allocation_sha256))
    elif args.command == "freeze-runtime":
        from .pinning import freeze
        print(freeze(args.model_root, args.settings, args.output_dir))
    elif args.command == "score":
        from .score import score
        result = score(args.prepared_root, args.export_root, args.model_root, args.contract, args.output_dir,
                       expected_prepared_sha256=args.expected_prepared_sha256,
                       expected_contract_sha256=args.expected_contract_sha256)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 2 if result["status"] == "incomplete" else 1)
    else:
        from .smoke import smoke
        result = smoke(args.prepared_root, args.export_root, args.model_root, args.inference_contract,
                       args.settings, args.output_dir,
                       expected_prepared_sha256=args.expected_prepared_sha256,
                       expected_inference_sha256=args.expected_inference_sha256)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
