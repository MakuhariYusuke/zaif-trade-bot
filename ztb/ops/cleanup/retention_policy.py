#!/usr/bin/env python3
"""
Retention policy for artifacts cleanup.

Suggests or applies cleanup based on retention rules.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

# Add the ztb package to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "ztb"))

from ztb.utils.cli_common import (
    CLIFormatter,
    CLIValidator,
    CommonArgs,
    create_standard_parser,
)


def get_session_info(session_dir: Path) -> Tuple[str, datetime, float, bool]:
    """Get session info: (id, mtime, size_mb, is_best)."""
    corr_id = session_dir.name
    mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
    size_mb = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file()) / (
        1024 * 1024
    )
    is_best = (session_dir / "best.marker").exists()
    return corr_id, mtime, size_mb, is_best


def find_candidates(
    root: Path, keep_days: int, keep_best: int, max_size_gb: float
) -> List[Tuple[str, str]]:
    """Find cleanup candidates."""
    candidates = []
    sessions = []

    # Collect all sessions
    for item in root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            try:
                info = get_session_info(item)
                sessions.append(info)
            except Exception:
                continue

    # Sort by mtime descending (newest first)
    sessions.sort(key=lambda x: x[1], reverse=True)

    # Keep recent sessions
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    recent_sessions = [s for s in sessions if s[1] >= cutoff_date]

    # Keep best sessions
    best_sessions = [s for s in sessions if s[3]]  # is_best
    best_sessions.sort(key=lambda x: x[1], reverse=True)  # newest best first
    keep_best_sessions = best_sessions[:keep_best]

    # Combine keep sets
    keep_ids = {s[0] for s in recent_sessions} | {s[0] for s in keep_best_sessions}

    # Check size limit
    total_size_gb = sum(s[2] for s in sessions) / 1024
    if total_size_gb > max_size_gb:
        # Sort by age (oldest first) for size-based cleanup
        size_candidates = [s for s in sessions if s[0] not in keep_ids]
        size_candidates.sort(key=lambda x: x[1])  # oldest first

        current_size = sum(s[2] for s in sessions if s[0] in keep_ids) / 1024
        for sess in size_candidates:
            if current_size >= max_size_gb:
                break
            candidates.append(
                (sess[0], f"Size limit: {total_size_gb:.1f}GB > {max_size_gb}GB")
            )
            current_size += sess[2] / 1024
    else:
        # Age/size based
        for sess in sessions:
            if sess[0] not in keep_ids:
                age_days = (datetime.now() - sess[1]).days
                if age_days > keep_days:
                    candidates.append((sess[0], f"Age: {age_days} days > {keep_days}"))

    return candidates


def apply_cleanup(root: Path, candidates: List[Tuple[str, str]]):
    """Apply cleanup by removing candidate directories."""
    for sess_id, reason in candidates:
        sess_dir = root / sess_id
        try:
            import shutil

            shutil.rmtree(sess_dir)
            print(f"Removed {sess_id}: {reason}")
        except Exception as e:
            print(f"Failed to remove {sess_id}: {e}", file=sys.stderr)


def main():
    parser = create_standard_parser("Apply retention policy for artifacts")
    CommonArgs.add_artifacts_dir(parser)
    parser.add_argument(
        "--keep-days",
        type=lambda x: CLIValidator.validate_positive_int(x, "keep-days"),
        default=14,
        help=CLIFormatter.format_help("Keep sessions newer than this many days", 14),
    )
    parser.add_argument(
        "--keep-best",
        type=lambda x: CLIValidator.validate_positive_int(x, "keep-best"),
        default=3,
        help=CLIFormatter.format_help("Keep this many best-marked sessions", 3),
    )
    parser.add_argument(
        "--max-size-gb",
        type=lambda x: CLIValidator.validate_positive_float(x, "max-size-gb"),
        default=50.0,
        help=CLIFormatter.format_help("Maximum total size in GB", 50.0),
    )
    CommonArgs.add_dry_run(parser)
    parser.add_argument(
        "--apply", action="store_true", help="Apply cleanup (delete candidates)"
    )

    args = parser.parse_args()

    if args.dry_run and args.apply:
        print("Cannot use both --dry-run and --apply", file=sys.stderr)
        return 1

    if not (args.dry_run or args.apply):
        args.dry_run = True  # Default to dry-run

    root = Path(args.artifacts_dir)
    if not root.exists():
        print(f"Artifacts directory not found: {root}", file=sys.stderr)
        return 1

    candidates = find_candidates(root, args.keep_days, args.keep_best, args.max_size_gb)

    if not candidates:
        print("No cleanup candidates found.")
        return 0

    print(f"Found {len(candidates)} cleanup candidates:")
    for sess_id, reason in candidates:
        print(f"  {sess_id}: {reason}")

    if args.apply:
        confirm = input("Apply cleanup? (y/N): ").lower().strip()
        if confirm == "y":
            apply_cleanup(root, candidates)
        else:
            print("Cleanup cancelled.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
