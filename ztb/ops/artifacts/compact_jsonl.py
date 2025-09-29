#!/usr/bin/env python3
"""
Compact JSONL logs by date into compressed files.
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union


def parse_timestamp(line: str) -> Optional[str]:
    """Extract date from JSON line."""
    try:
        data = json.loads(line.strip())
        timestamp = data.get("timestamp")
        if timestamp:
            # Assume ISO format, extract date
            return timestamp.split("T")[0]
    except json.JSONDecodeError:
        pass
    return None


def compact_jsonl(file_path: Union[str, Path], apply: bool = False) -> None:
    """Compact JSONL file by date."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    # Group lines by date
    date_groups = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            date = parse_timestamp(line)
            if date:
                date_groups[date].append(line)
            else:
                # Lines without timestamp go to 'unknown'
                date_groups["unknown"].append(line)

    # Report
    total_lines = sum(len(lines) for lines in date_groups.values())
    print(f"Found {len(date_groups)} date groups, {total_lines} total lines")

    for date, lines in sorted(date_groups.items()):
        print(f"  {date}: {len(lines)} lines")

    if not apply:
        print("Use --apply to perform compaction")
        return

    # Backup original
    backup_path = file_path.with_suffix(file_path.suffix + ".backup")
    file_path.rename(backup_path)
    print(f"Backed up original to {backup_path}")

    # Write compressed files
    for date, lines in date_groups.items():
        output_path = file_path.parent / f"{date}.jsonl.gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Created {output_path} ({len(lines)} lines)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compact JSONL logs by date into compressed files"
    )
    parser.add_argument("--file", required=True, help="JSONL file to compact")
    parser.add_argument(
        "--apply", action="store_true", help="Apply compaction (default is dry-run)"
    )

    args = parser.parse_args()

    compact_jsonl(args.file, args.apply)


if __name__ == "__main__":
    main()
