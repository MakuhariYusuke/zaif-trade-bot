"""v460 raw data path/date helper."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "v460" / "raw"
RawDirLike = str | Path | None
_RAW_JSONL_GZ_SUFFIX = ".jsonl.gz"


def resolve_raw_dir(raw_dir: RawDirLike = None) -> Path:
    """raw data ルートを絶対パスで返す."""
    if raw_dir is None:
        return DEFAULT_RAW_DIR
    raw_path = Path(raw_dir)
    return raw_path if raw_path.is_absolute() else (_PROJECT_ROOT / raw_path).resolve()


def resolve_available_raw_dates(
    daily_inputs: dict[str, tuple[Path | None, Path | None]],
    dates: list[str] | None = None,
) -> list[str]:
    """利用可能な日付集合から対象日付を一意化して返す."""
    all_dates = sorted(daily_inputs)
    if dates is None:
        return all_dates

    resolved: list[str] = []
    seen: set[str] = set()
    for date_str in dates:
        if date_str in seen:
            continue
        seen.add(date_str)
        if date_str in daily_inputs:
            resolved.append(date_str)
    return resolved


def utc_day_str_from_timestamp(timestamp: float) -> str:
    """UNIX timestamp を UTC 日付文字列 YYYYMMDD に変換する."""
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y%m%d")


def extract_utc_day_from_raw_path(path: Path) -> str | None:
    """raw JSONL.gz パスから UTC 日付文字列 YYYYMMDD を抽出する."""
    name = path.name
    if not name.endswith(_RAW_JSONL_GZ_SUFFIX):
        return None
    day = name.removesuffix(_RAW_JSONL_GZ_SUFFIX)
    if len(day) != 8 or not day.isdigit():
        return None
    return day


def collect_available_raw_days(directory: Path) -> list[str]:
    """raw ディレクトリから利用可能な UTC 日付キーを昇順返却する."""
    if not directory.exists():
        return []
    days = {
        day
        for path in directory.glob(f"*{_RAW_JSONL_GZ_SUFFIX}")
        if (day := extract_utc_day_from_raw_path(path)) is not None
    }
    return sorted(days)


def group_records_by_utc_day(
    records: Sequence[dict[str, object]],
    *,
    timestamp_key: str = "ts",
) -> dict[str, list[dict[str, object]]]:
    """レコード列を UTC 日付ごとに安定順で分割する."""
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(
            utc_day_str_from_timestamp(float(record[timestamp_key])),
            [],
        ).append(record)
    return grouped
