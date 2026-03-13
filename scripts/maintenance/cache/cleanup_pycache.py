"""Cleanup script to remove __pycache__ directories and .pyc files."""

from pathlib import Path
import shutil


def cleanup_project(root: Path) -> int:
    removed = 0

    for cache_dir in root.rglob("__pycache__"):
        try:
            shutil.rmtree(cache_dir)
            print("removed", cache_dir)
            removed += 1
        except Exception as exc:
            print("failed to remove", cache_dir, exc)

    for pyc_file in root.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            print("removed", pyc_file)
            removed += 1
        except Exception as exc:
            print("failed to remove", pyc_file, exc)

    return removed


def main() -> None:
    root = next(
        (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
        Path(__file__).resolve().parent,
    )
    removed = cleanup_project(root)
    print("total removed", removed)


if __name__ == "__main__":
    main()
