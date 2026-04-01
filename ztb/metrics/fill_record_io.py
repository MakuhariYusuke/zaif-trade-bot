from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from ztb.io.jsonl import iter_jsonl_objects


FillRecordObject = dict[str, object]


def _normalize_fill_record_date(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("-", "")
    return normalized if len(normalized) == 8 and normalized.isdigit() else None


def _extract_fill_record_file_date(path: Path) -> str | None:
    date_part = path.stem.split("_")[-1]
    return date_part if len(date_part) == 8 and date_part.isdigit() else None


def _directory_signature(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
    except OSError:
        return -1, -1
    return st.st_mtime_ns, st.st_ctime_ns


def _scan_jsonl_by_prefix(directory: Path, prefix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith(prefix)
            and path.suffix == ".jsonl"
        ]
    except OSError:
        return []
    return sorted(files)


@lru_cache(maxsize=256)
def _list_fill_record_files_cached(
    directory: str,
    include_emergency: bool,
    root_sig: tuple[int, int],
    emergency_sig: tuple[int, int],
) -> tuple[Path, ...]:
    del root_sig, emergency_sig
    root = Path(directory)
    files = _scan_jsonl_by_prefix(root, "fill_records_")
    if include_emergency:
        files.extend(_scan_jsonl_by_prefix(root / "emergency", "emergency_"))
    return tuple(files)


def _resolve_fill_record_files_by_date_range(
    directory: Path,
    *,
    include_emergency: bool,
    start_date: str,
    end_date: str,
) -> list[Path] | None:
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except ValueError:
        return None
    if start > end:
        return []
    if start_date == end_date:
        single_day_files: list[Path] = []
        fill_path = directory / f"fill_records_{start_date}.jsonl"
        if fill_path.is_file():
            single_day_files.append(fill_path)
        if include_emergency:
            emergency_path = directory / "emergency" / f"emergency_{start_date}.jsonl"
            if emergency_path.is_file():
                single_day_files.append(emergency_path)
        return single_day_files
    day_count = (end - start).days
    if day_count > 3650:
        return None

    files: list[Path] = []
    for i in range(day_count + 1):
        day_str = (start + timedelta(days=i)).strftime("%Y%m%d")
        fill_path = directory / f"fill_records_{day_str}.jsonl"
        if fill_path.is_file():
            files.append(fill_path)
    if include_emergency:
        emergency_dir = directory / "emergency"
        for i in range(day_count + 1):
            day_str = (start + timedelta(days=i)).strftime("%Y%m%d")
            emergency_path = emergency_dir / f"emergency_{day_str}.jsonl"
            if emergency_path.is_file():
                files.append(emergency_path)
    return files


def list_fill_record_files(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """List fill/emergency JSONL files in stable order."""
    root = Path(directory)
    norm_start = _normalize_fill_record_date(start_date)
    norm_end = _normalize_fill_record_date(end_date)

    if norm_start is not None and norm_end is not None:
        direct_files = _resolve_fill_record_files_by_date_range(
            root,
            include_emergency=include_emergency,
            start_date=norm_start,
            end_date=norm_end,
        )
        if direct_files is not None:
            return direct_files

    def _within_date(path: Path) -> bool:
        if norm_start is None and norm_end is None:
            return True
        file_date = _extract_fill_record_file_date(path)
        if file_date is None:
            return False
        if norm_start is not None and file_date < norm_start:
            return False
        if norm_end is not None and file_date > norm_end:
            return False
        return True

    root_sig = _directory_signature(root)
    emergency_sig = _directory_signature(root / "emergency") if include_emergency else (-1, -1)
    all_files = _list_fill_record_files_cached(
        str(root),
        include_emergency,
        root_sig,
        emergency_sig,
    )
    return [path for path in all_files if _within_date(path)]


def iter_fill_record_objects_from_files(files: Iterable[Path]) -> Iterator[FillRecordObject]:
    """Yield raw JSONL payloads while deduplicating cycle_id across files."""
    seen_ids: set[str] = set()
    for path in files:
        for record in iter_jsonl_objects(path, warn_malformed=True):
            cycle_id = record.get("cycle_id")
            if isinstance(cycle_id, str):
                if cycle_id in seen_ids:
                    continue
                seen_ids.add(cycle_id)
            yield record


def iter_fill_record_objects_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[FillRecordObject]:
    yield from iter_fill_record_objects_from_files(
        list_fill_record_files(
            directory,
            include_emergency=include_emergency,
            start_date=start_date,
            end_date=end_date,
        )
    )


def load_fill_record_objects_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[FillRecordObject]:
    return list(
        iter_fill_record_objects_glob(
            directory,
            include_emergency=include_emergency,
            start_date=start_date,
            end_date=end_date,
        )
    )


def _coerce_filter_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if math.isfinite(ts):
            return ts
    return None


def apply_fill_record_filters(
    records: Iterable[Mapping[str, object]],
    *,
    run_id: str | None = None,
    git_sha: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[FillRecordObject], dict[str, str | None]]:
    """Apply run_id/git_sha/date filters to raw fill-record mappings."""
    ts_from: float | None = None
    ts_to: float | None = None
    if date_from:
        ts_from = datetime.strptime(date_from, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
    if date_to:
        ts_to = (
            datetime.strptime(date_to, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp()
            + 86400
        )

    filtered: list[FillRecordObject] = []
    for record in records:
        if run_id and record.get("run_id") != run_id:
            continue
        if git_sha and not str(record.get("git_sha", "")).startswith(git_sha):
            continue
        timestamp = _coerce_filter_timestamp(record.get("timestamp")) or 0.0
        if ts_from is not None and timestamp < ts_from:
            continue
        if ts_to is not None and timestamp >= ts_to:
            continue
        filtered.append(record if isinstance(record, dict) else dict(record))

    filters = {
        "run_id": run_id,
        "git_sha": git_sha,
        "date_from": date_from,
        "date_to": date_to,
    }
    return filtered, filters
