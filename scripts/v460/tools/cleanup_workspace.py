from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class CleanupCandidate:
    display_name: str
    paths: tuple[Path, ...]
    file_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class CleanupSummary:
    candidates: tuple[CleanupCandidate, ...]

    @property
    def total_files(self) -> int:
        return sum(candidate.file_count for candidate in self.candidates)

    @property
    def total_bytes(self) -> int:
        return sum(candidate.total_bytes for candidate in self.candidates)


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child


def _summarize_paths(paths: Sequence[Path]) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for path in paths:
        for file_path in _iter_files(path):
            file_count += 1
            try:
                total_bytes += file_path.stat().st_size
            except OSError:
                continue
    return file_count, total_bytes


def _load_tracked_paths(repo_root: Path) -> frozenset[str]:
    git_dir = repo_root / ".git"
    if not git_dir.exists():
        return frozenset()
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    raw_paths = proc.stdout.split(b"\0")
    tracked: set[str] = set()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        tracked.add(raw_path.decode("utf-8"))
    return frozenset(tracked)


def _is_tracked(path: Path, *, repo_root: Path, tracked_paths: frozenset[str]) -> bool:
    try:
        rel_path = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return False
    if rel_path in tracked_paths:
        return True
    prefix = f"{rel_path}/"
    return any(tracked.startswith(prefix) for tracked in tracked_paths)


def discover_cleanup_candidates(
    *,
    repo_root: Path = REPO_ROOT,
    tracked_paths: frozenset[str] | None = None,
) -> CleanupSummary:
    tracked = _load_tracked_paths(repo_root) if tracked_paths is None else tracked_paths
    candidates: list[CleanupCandidate] = []

    ab_search_paths = tuple(sorted((repo_root / "config").glob("ab_search_temp_*.json")))
    untracked_ab_paths = tuple(
        path for path in ab_search_paths if not _is_tracked(path, repo_root=repo_root, tracked_paths=tracked)
    )
    if untracked_ab_paths:
        file_count, total_bytes = _summarize_paths(untracked_ab_paths)
        candidates.append(
            CleanupCandidate(
                display_name="config/ab_search_temp_*.json",
                paths=untracked_ab_paths,
                file_count=file_count,
                total_bytes=total_bytes,
            )
        )

    for relative_dir in (
        "data/temp/.mypy_cache",
        "data/temp/.ruff_cache",
        "data/temp/.hypothesis",
        "data/temp/.pytest_cache",
    ):
        target = repo_root / relative_dir
        if not target.exists() or _is_tracked(target, repo_root=repo_root, tracked_paths=tracked):
            continue
        file_count, total_bytes = _summarize_paths((target,))
        candidates.append(
            CleanupCandidate(
                display_name=f"{relative_dir}/",
                paths=(target,),
                file_count=file_count,
                total_bytes=total_bytes,
            )
        )

    tmp_dirs = tuple(
        path
        for path in sorted((repo_root / "data/temp").glob("tmp-*"))
        if path.is_dir() and not _is_tracked(path, repo_root=repo_root, tracked_paths=tracked)
    )
    if tmp_dirs:
        file_count, total_bytes = _summarize_paths(tmp_dirs)
        candidates.append(
            CleanupCandidate(
                display_name="data/temp/tmp-*",
                paths=tmp_dirs,
                file_count=file_count,
                total_bytes=total_bytes,
            )
        )

    return CleanupSummary(candidates=tuple(candidates))


def _render_candidate(candidate: CleanupCandidate, *, dry_run: bool) -> str:
    prefix = "[DRY-RUN]" if dry_run else "[EXECUTE]"
    if candidate.display_name == "config/ab_search_temp_*.json":
        return (
            f"{prefix} Would remove {candidate.file_count} files from "
            f"{candidate.display_name} ({_format_bytes(candidate.total_bytes)})"
        )
    return (
        f"{prefix} Would remove {candidate.display_name} "
        f"({candidate.file_count} files, {_format_bytes(candidate.total_bytes)})"
    )


def _rmtree_onerror(
    _func: Callable[..., object],
    _path: str,
    _exc_info: tuple[type[BaseException], BaseException, object],
) -> None:
    return None


def execute_cleanup(
    summary: CleanupSummary,
    *,
    execute: bool,
    verbose: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> CleanupSummary:
    dry_run = not execute
    for candidate in summary.candidates:
        print(_render_candidate(candidate, dry_run=dry_run))
        if verbose:
            for path in candidate.paths:
                print(f"  - {path}")
    if not summary.candidates:
        total_prefix = "[DRY-RUN]" if dry_run else "[EXECUTE]"
        print(f"{total_prefix} No cleanup targets found")
        return summary

    total_line = (
        f"Total: {summary.total_files} files, {_format_bytes(summary.total_bytes)}"
    )
    if dry_run:
        print(f"{total_line} (add --execute to actually delete)")
        return summary

    print(total_line)
    sleep_fn(1.0)
    for candidate in summary.candidates:
        for path in candidate.paths:
            if path.is_dir():
                shutil.rmtree(path, onerror=_rmtree_onerror)
            elif path.exists():
                try:
                    path.unlink()
                except OSError:
                    continue
    print("Cleanup completed")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Workspace cleanup for ignored temp/cache artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove files (default is dry-run)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print matched paths",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = discover_cleanup_candidates()
    execute_cleanup(summary, execute=args.execute, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
