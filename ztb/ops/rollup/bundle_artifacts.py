#!/usr/bin/env python3
"""
Artifacts bundler for Zaif Trade Bot.

Bundles artifacts directory into ZIP with optional exclusions and SHA256 hash.
"""

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


def should_exclude_file(file_path: Path, exclude_logs: bool) -> bool:
    """Check if file should be excluded from bundle."""
    if exclude_logs and file_path.suffix.lower() == ".log":
        return True
    return False


def create_bundle(correlation_id: str, exclude_logs: bool) -> None:
    """Create ZIP bundle of artifacts."""
    artifacts_dir = Path("artifacts") / correlation_id
    if not artifacts_dir.exists():
        print(f"Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    bundle_path = artifacts_dir / "bundle.zip"
    hash_path = artifacts_dir / "bundle.sha256"

    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in artifacts_dir.rglob("*"):
            if file_path.is_file():
                if should_exclude_file(file_path, exclude_logs):
                    print(f"Excluding: {file_path.relative_to(artifacts_dir)}")
                    continue

                arcname = file_path.relative_to(artifacts_dir)
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")

    # Calculate SHA256
    sha256 = hashlib.sha256()
    with open(bundle_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    with open(hash_path, "w") as f:
        f.write(f"{sha256.hexdigest()}  bundle.zip\n")

    print(f"Bundle created: {bundle_path}")
    print(f"SHA256: {hash_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bundle artifacts for Zaif Trade Bot session"
    )
    parser.add_argument(
        "--correlation-id", required=True, help="Session correlation ID"
    )
    parser.add_argument(
        "--exclude-logs", action="store_true", help="Exclude log files from bundle"
    )

    args = parser.parse_args()

    create_bundle(args.correlation_id, args.exclude_logs)


if __name__ == "__main__":
    main()
