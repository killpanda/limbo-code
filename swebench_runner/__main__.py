"""CLI: python -m swebench_runner [options]"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from .batch import load_instances, run_batch
from .instance import RunConfig

DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="swebench_runner",
        description="Run limbo-code on SWE-bench instances and emit predictions.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--instance-ids",
        nargs="*",
        default=None,
        help="Run only these instance ids",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Randomly sample N instances (ignored if --instance-ids given)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default=None,
        help="Model override (default: host limbo config's model)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Wall-clock seconds per instance (default: 1800)",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Output dir for predictions.jsonl + logs "
        "(default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--limbo-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to the limbo-code checkout mounted into containers",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    run_dir = args.run_dir or Path("runs") / datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(
        args.dataset,
        instance_ids=args.instance_ids,
        count=args.count,
        seed=args.seed,
    )
    config = RunConfig(
        limbo_repo=args.limbo_repo,
        model=args.model,
        timeout=args.timeout,
    )
    run_batch(instances, config, run_dir, concurrency=args.concurrency)
    print(f"predictions: {run_dir / 'predictions.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
