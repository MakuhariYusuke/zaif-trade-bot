#!/usr/bin/env python3
"""
Compare two effective-config JSON files and show differences.
"""

import argparse
import json
import sys


def load_json(file_path):
    """Load JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def deep_diff(a, b, path=""):
    """Recursively compare two dicts."""
    diffs = []

    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for key in sorted(all_keys):
            new_path = f"{path}.{key}" if path else key
            if key not in a:
                diffs.append(f"+ {new_path}: {b[key]}")
            elif key not in b:
                diffs.append(f"- {new_path}: {a[key]}")
            else:
                diffs.extend(deep_diff(a[key], b[key], new_path))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"~ {path}: list length {len(a)} -> {len(b)}")
        else:
            for i, (va, vb) in enumerate(zip(a, b)):
                new_path = f"{path}[{i}]"
                diffs.extend(deep_diff(va, vb, new_path))
    else:
        if a != b:
            diffs.append(f"~ {path}: {a} -> {b}")

    return diffs


def is_major_key(path):
    """Check if the path contains a major configuration key."""
    major_keys = ["model", "policy", "learning_rate", "batch_size"]
    return any(key in path for key in major_keys)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two effective-config JSON files"
    )
    parser.add_argument("--a", required=True, help="Path to first config JSON")
    parser.add_argument("--b", required=True, help="Path to second config JSON")

    args = parser.parse_args()

    config_a = load_json(args.a)
    config_b = load_json(args.b)

    diffs = deep_diff(config_a, config_b)

    if not diffs:
        print("No differences found.")
        sys.exit(0)

    print("Configuration differences:")
    for diff in diffs:
        print(f"  {diff}")

    # Check for major differences
    major_diffs = [d for d in diffs if is_major_key(d.split(":")[0].strip("+-~ "))]

    if major_diffs:
        print("\nMajor configuration differences detected!")
        sys.exit(3)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
