#!/usr/bin/env python3
"""Check for non-ASCII punctuation in Python docstrings/comments under `ztb/`.

This script is intentionally conservative: it reports files and lines containing
fullwidth punctuation commonly introduced by Japanese text (。、（）：【】 etc.)
so the team can decide whether to translate to English.

Usage:
    python scripts/check_docstring_ascii.py

Exit codes:
    0 - no issues found
    1 - issues found (printed to stdout)
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"[。、（）：【】『』「」]")

issues = []
for p in ROOT.rglob("ztb/**/*.py"):
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        continue
    for i, line in enumerate(text.splitlines(), start=1):
        if PATTERN.search(line):
            issues.append((str(p.relative_to(ROOT)), i, line.strip()))

if not issues:
    print("No non-ASCII punctuation issues found under ztb/.")
    sys.exit(0)

print("Found non-ASCII punctuation in the following files:")
for f, ln, line in issues:
    print(f"{f}:{ln}: {line}")

print(f"\nTotal occurrences: {len(issues)}")
sys.exit(1)
