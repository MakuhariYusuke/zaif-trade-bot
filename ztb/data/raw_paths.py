"""v460 raw data path/date helper."""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = _PROJECT_ROOT / "data" / "v460" / "raw"
RawDirLike = str | Path | None


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
