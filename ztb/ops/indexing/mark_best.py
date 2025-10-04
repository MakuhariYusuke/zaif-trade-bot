#!/usr/bin/env python3
"""
Mark a session as best candidate.

Creates a marker file to flag the session as a best candidate.
"""

import sys
from pathlib import Path

# Add the ztb package to the path
try:
    ztb_path = Path(__file__).parent.parent / "ztb"
    if not ztb_path.exists():
        raise FileNotFoundError(f"ztb package path not found: {ztb_path}")
    sys.path.insert(0, str(ztb_path))
except Exception as e:
    print(f"Failed to add ztb package to sys.path: {e}", file=sys.stderr)
    # Don't exit in test environments
    if "pytest" not in sys.modules and "unittest" not in sys.modules:
        sys.exit(1)

from datetime import datetime

from ztb.utils.cli_common import CommonArgs, create_standard_parser


def mark_best(correlation_id: str, artifacts_dir: Path = Path("artifacts")) -> int:
    """Mark session as best."""
    session_dir = artifacts_dir / correlation_id
    if not session_dir.exists():
        print(f"Session {correlation_id} not found", file=sys.stderr)
        return 1

    marker_file = session_dir / "best.marker"
    try:
        timestamp = datetime.now().isoformat()
        marker_file.write_text(
            f"Marked as best at {timestamp} by {Path(__file__).name}\n"
        )
        print(f"Marked {correlation_id} as best candidate")
        return 0
    except Exception as e:
        print(f"Failed to mark {correlation_id}: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = create_standard_parser("Mark session as best candidate")
    CommonArgs.add_correlation_id(parser)
    CommonArgs.add_artifacts_dir(parser)

    args = parser.parse_args()
    return mark_best(args.correlation_id, Path(args.artifacts_dir))


if __name__ == "__main__":
    sys.exit(main())
