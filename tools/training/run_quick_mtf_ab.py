#!/usr/bin/env python3
"""
Quick AB runner for testing MTF weight manager behavior in small runs.

This script uses the existing `ab_test_runner.py` tool and toggles MTF features.
It is intended for fast smoke validation (3 seeds × 1000 steps) only.
"""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--name", type=str, default="mtf_quick_test")
    parser.add_argument("--enable-mtf", action="store_true", help="Enable MTF optimizer flag in run")
    args = parser.parse_args()

    ab_runner = Path(__file__).parents[1] / "ab_test_runner.py"
    if not ab_runner.exists():
        print("ab_test_runner.py not found; ensure tools/training/ab_test_runner.py exists")
        return 1

    config_arg = args.config
    if args.enable_mtf:
        config_arg = f"{config_arg} --override behavior.mtf.weight_optimizer.enabled=True"

    cmd = [
        "python",
        str(ab_runner),
        "--configs",
        str(args.config),
        "--seeds",
        str(args.seeds),
        "--timesteps",
        str(args.timesteps),
        "--name",
        args.name,
    ]
    print("Running quick AB runner:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
