#!/usr/bin/env python3
"""Fast pytest runner for local development.

This wrapper bypasses heavy default addopts in pytest.ini and provides a
predictable fast path for iteration.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


def _has_xdist() -> bool:
    return importlib.util.find_spec("xdist") is not None


def _default_targets(scope: str) -> list[str]:
    if scope == "v460":
        return ["tests/unit/v460"]
    if scope == "unit":
        return ["tests/unit"]
    return ["tests"]


def build_command(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [sys.executable, "-m", "pytest"]

    targets = args.targets if args.targets else _default_targets(args.scope)
    cmd.extend(targets)

    # Bypass heavy default addopts and keep only lightweight defaults here.
    cmd.extend(
        [
            "--override-ini=addopts=",
            "--tb=short",
            "--ignore=archived",
            "--ignore=scripts",
            "--ignore-glob=**/archived/**",
            "--ignore-glob=**/scripts/**",
        ]
    )

    if args.collect_only:
        cmd.extend(["--co", "-qq"])
    else:
        if args.verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        cmd.append(f"--maxfail={args.maxfail}")

    if args.coverage:
        cmd.extend(["--cov=ztb", "--cov-report=term-missing", "--cov-fail-under=80"])

    if (
        not args.collect_only
        and args.parallel != "off"
        and args.parallel == "auto"
        and _has_xdist()
    ):
        cmd.extend(["-n", "auto"])

    passthrough = list(args.pytest_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    cmd.extend(passthrough)
    return cmd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast pytest locally")
    parser.add_argument(
        "--scope",
        choices=("v460", "unit", "all"),
        default="v460",
        help="Default target test scope (ignored when positional targets are provided)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Run collection only",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage (disabled by default for speed)",
    )
    parser.add_argument(
        "--parallel",
        choices=("auto", "off"),
        default="auto",
        help="Use xdist when available",
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=3,
        help="Stop after this many failures (non collect mode)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose test output",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Optional explicit pytest targets",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed directly to pytest (prefix with --pytest-args -- ...)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_command(args)
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
