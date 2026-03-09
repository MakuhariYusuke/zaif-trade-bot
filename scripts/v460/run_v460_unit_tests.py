"""Run the v460 unit suite with the project-standard no-cov settings.

This wrapper exists because `pytest.ini` enforces a repository-wide coverage
gate that is not meaningful for the `tests/unit/v460/` subset on its own.
"""

from __future__ import annotations

import subprocess
import sys


DEFAULT_ARGS: list[str] = [
    "-m",
    "pytest",
    "tests/unit/v460/",
    "-q",
    "--no-cov",
    "--tb=short",
]


def main(argv: list[str] | None = None) -> int:
    extra_args = list(sys.argv[1:] if argv is None else argv)
    cmd = [sys.executable, *DEFAULT_ARGS, *extra_args]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
