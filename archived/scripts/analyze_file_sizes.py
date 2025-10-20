#!/usr/bin/env python3
"""Analyze Python file sizes in ztb/training directory."""

from pathlib import Path


def main() -> None:
    training_dir = Path("ztb/training")
    files = []

    for py_file in training_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            lines = len(py_file.read_text(encoding="utf-8").splitlines())
            files.append((lines, py_file.name, str(py_file)))
        except Exception as e:
            print(f"Error reading {py_file}: {e}")

    files.sort(reverse=True)

    print("\n=== Top 20 Largest Files in ztb/training ===\n")
    print(f"{'Lines':<8} {'File':<40} {'Path'}")
    print("-" * 100)

    for lines, name, path in files[:20]:
        print(f"{lines:<8} {name:<40} {path}")


if __name__ == "__main__":
    main()
