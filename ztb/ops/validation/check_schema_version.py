#!/usr/bin/env python3
"""
Check schema version consistency.

Ensures that schema version is incremented when schema files are modified.
This script is run in CI to prevent accidental schema changes without version bumps.
"""

import subprocess
import sys
from typing import Set


def get_git_modified_files() -> Set[str]:
    """Get list of modified files in current git changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        staged = (
            set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        )

        result = subprocess.run(
            ["git", "diff", "--name-only"], capture_output=True, text=True, check=True
        )
        unstaged = (
            set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        )

        return staged | unstaged
    except subprocess.CalledProcessError:
        print("Warning: Could not get git status", file=sys.stderr)
        return set()


def check_schema_version_increment(modified_files: Set[str]) -> bool:
    """Check if schema version was incremented when schema files changed."""
    schema_files = {
        "schema/results_schema.json",
        "schema/config_schema.json",
        "ztb/config/schema.py",
        "ztb/contracts/models.py",
    }

    # Check if any schema files were modified
    schema_modified = bool(modified_files & schema_files)

    if not schema_modified:
        return True  # No schema changes, no version bump needed

    # Check if version file was modified
    version_file = "schema/RESULTS_SCHEMA_VERSION"
    if version_file not in modified_files:
        print(
            "ERROR: Schema files were modified but RESULTS_SCHEMA_VERSION was not updated!"
        )
        print("Modified schema files:")
        for f in sorted(modified_files & schema_files):
            print(f"  - {f}")
        print("\nPlease:")
        print("1. Increment RESULTS_SCHEMA_VERSION")
        print("2. Update CHANGELOG.md with the schema changes")
        return False

    print("✓ Schema version check passed")
    return True


def main() -> int:
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Check schema version consistency")
        print("Run this script to ensure schema changes include version bumps")
        return 0

    modified_files = get_git_modified_files()

    if not check_schema_version_increment(modified_files):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
