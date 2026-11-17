import argparse
import json
from pathlib import Path
import sys

from .export import validate_dataset
from .generation import generate_batch
from .runner import run_pipeline, tree_fingerprint
from .sources import audit_sources, compare_reference_format
from .teachers import TeacherError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate and verify local CSV/SQLite agent trajectories")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="generate, execute, verify and export")
    run.add_argument("--mode", choices=("mock", "teacher"), default="mock")
    run.add_argument("--output", type=Path, default=Path("artifacts"), help="base directory; mode/run-id is appended")
    run.add_argument("--run-id", default="demo")
    run.add_argument("--count", type=int, default=20)
    run.add_argument("--seed", type=int, default=20261117)
    run.add_argument("--max-steps", type=int, default=16)
    run.add_argument("--timeout", type=float, default=30)
    run.add_argument("--mock-fail-every", type=int, default=0)
    run.add_argument("--validation-family", action="append", dest="validation_families")
    run.add_argument("--archive", type=Path)
    generate = commands.add_parser("generate", help="generate inputs and separate references")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--seed", type=int, default=20261117)
    check = commands.add_parser("check-jsonl", help="validate JSONL and all tool-call links")
    check.add_argument("path", type=Path)
    audit = commands.add_parser("audit-sources", help="inspect local archive without modifying sources")
    audit.add_argument("--archive", type=Path, default=Path("../nex-archive-2026-11-17"))
    reference = commands.add_parser("check-reference", help="compare an export with a bounded local tool_calling sample")
    reference.add_argument("dataset", type=Path)
    reference.add_argument("--archive", type=Path, default=Path("../nex-archive-2026-11-17"))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "run":
            result = run_pipeline(**args)
        elif command == "generate":
            manifest = generate_batch(args["output"], args["count"], args["seed"])
            result = {"generated": len(manifest), "sha256": tree_fingerprint(args["output"])}
        elif command == "check-jsonl":
            result = {"valid_records": validate_dataset(args["path"])}
        elif command == "check-reference":
            result = compare_reference_format(args["dataset"], args["archive"])
            if not result["verified"]:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                raise SystemExit(2)
        else:
            result = audit_sources(args["archive"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, TeacherError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from None
