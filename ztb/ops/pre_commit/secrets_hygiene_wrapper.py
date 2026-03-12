#!/usr/bin/env python3
"""
Wrapper for pre-commit secrets-hygiene hook that ensures project root is on sys.path
and calls ztb.utils.redact.is_safe_content for each file passed by pre-commit.
Exits with 0 if all files are safe, 1 otherwise.
"""
import sys
from pathlib import Path

# Ensure the repo root is importable
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from ztb.utils.redact import is_safe_content
except Exception as e:
    print(f"Error importing ztb.utils.redact: {e}")
    sys.exit(1)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()
    exit_code = 0
    for fname in args.filenames:
        try:
            with open(fname, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Could not read {fname}: {e}")
            exit_code = 1
            continue
        safe = True
        try:
            safe = is_safe_content(content)
        except Exception as e:
            print(f"Error running is_safe_content on {fname}: {e}")
            exit_code = 1
            continue
        if not safe:
            print(f"Potential secret found in {fname}")
            exit_code = 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
