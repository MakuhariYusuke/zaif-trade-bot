"""135# P2-09→P1: trades データ健全性チェック.

run 開始時に trades raw データの存在を検証し、欠損日を検出する。
retrain が全量 fallback して特徴量の時間整合性が崩れるのを早期に防ぐ。

136# P1-02: feature staleness monitor 追加。
trades + OB データの鮮度を包括的に判定し、retrain 前ガードとして使用。

Usage (ライブラリとして):
    from ztb.data.trades_health import check_trades_health, TradesHealthResult
    result = check_trades_health(expected_days=["20260220", "20260221"])

    # P1-02: feature 鮮度チェック
    from ztb.data.trades_health import check_feature_freshness
    fresh = check_feature_freshness(raw_dir=Path("data/v460/raw"))

Usage (CLI):
    python -m ztb.data.trades_health --days 3
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ztb.data.raw_paths import resolve_raw_dir

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TradesHealthResult:
    """trades 健全性チェック結果."""

    healthy: bool
    available_days: list[str]
    missing_days: list[str]
    stale_hours: float  # 最新ファイルからの経過時間
    message: str

def _collect_available_days(trades_dir: Path) -> list[str]:
    """trades ディレクトリから YYYYMMDD 日付キーを抽出して昇順返却."""
    if not trades_dir.exists():
        return []
    available: list[str] = []
    for path in trades_dir.iterdir():
        name = path.name
        if not name.endswith(".jsonl.gz"):
            continue
        day = name[:-9]  # strip ".jsonl.gz"
        if len(day) == 8 and day.isdigit():
            available.append(day)
    available.sort()
    return available

def check_trades_health(
    raw_dir: Path | None = None,
    expected_days: list[str] | None = None,
    *,
    lookback_days: int = 3,
    stale_threshold_hours: float = 36.0,
    max_missing_days: int = 0,
) -> TradesHealthResult:
    """trades raw データの健全性をチェック.

    Args:
        raw_dir: raw data ディレクトリ (default: data/v460/raw).
        expected_days: チェック対象の日付リスト (YYYYMMDD).
            None の場合は直近 lookback_days 日を自動生成.
        lookback_days: expected_days=None 時の遡り日数.
        stale_threshold_hours: 最新ファイルがこれ以上古い場合は unhealthy.
        max_missing_days: 許容する欠損日数 (default: 0 = 厳密チェック).
            158# trades_health 修正: deadlock/restart による一時的なデータギャップに
            対応するため、最新ファイルが fresh であれば N 日分の欠損を許容する。

    Returns:
        TradesHealthResult: 結果オブジェクト.
    """
    d = resolve_raw_dir(raw_dir)
    tr_dir = d / "trades"
    now_utc = datetime.now(timezone.utc)
    now_ts = now_utc.timestamp()

    # 利用可能な日の列挙
    available = _collect_available_days(tr_dir)
    available_set = set(available)

    # 期待日リスト
    # §9.2 #B: 当日 UTC を除外し「昨日から N 日遡り」で生成。
    # 日跨ぎ直後 (00:00-01:00 UTC) に当日ファイル未生成で false warning を防止。
    if expected_days is None:
        expected_days = [
            (now_utc - timedelta(days=i + 1)).strftime("%Y%m%d")
            for i in range(lookback_days)
        ]

    # 欠損日
    missing = [day for day in expected_days if day not in available_set]

    # 鮮度チェック
    stale_hours = _latest_mtime_hours(
        tr_dir,
        glob_pattern="????????.jsonl.gz",
        now_ts=now_ts,
    )

    # 判定
    # 158# 修正: max_missing_days で欠損許容。ただし鮮度は必須。
    missing_ok = len(missing) <= max_missing_days
    stale_ok = stale_hours < stale_threshold_hours
    healthy = missing_ok and stale_ok
    if not healthy:
        parts: list[str] = []
        if not missing_ok:
            parts.append(f"missing_days={missing} (max_allowed={max_missing_days})")
        if not stale_ok:
            parts.append(f"stale={stale_hours:.1f}h (threshold={stale_threshold_hours}h)")
        msg = "UNHEALTHY: " + ", ".join(parts)
    else:
        extra = ""
        if missing:
            extra = f", tolerated_gaps={missing}"
        msg = f"OK: {len(available)} days available, stale={stale_hours:.1f}h{extra}"

    return TradesHealthResult(
        healthy=healthy,
        available_days=available,
        missing_days=missing,
        stale_hours=stale_hours,
        message=msg,
    )

# ---------------------------------------------------------------------------
# 136# P1-02: Feature Staleness Monitor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureFreshnessResult:
    """trades + OB データの鮮度チェック結果."""

    fresh: bool
    trades_stale_hours: float
    ob_stale_hours: float
    details: dict[str, str]
    message: str

def _latest_mtime_hours(
    directory: Path,
    glob_pattern: str = "*.jsonl.gz",
    *,
    now_ts: float | None = None,
) -> float:
    """ディレクトリ内の最新ファイル mtime から経過時間を返す."""
    if not directory.exists():
        return float("inf")
    latest_mtime = 0.0
    for p in directory.glob(glob_pattern):
        try:
            mt = p.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
        except OSError:
            continue
    if latest_mtime == 0.0:
        return float("inf")
    current_ts = now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp()
    return (current_ts - latest_mtime) / 3600

def check_feature_freshness(
    raw_dir: Path | None = None,
    *,
    trades_stale_hours: float = 6.0,
    ob_stale_hours: float = 6.0,
) -> FeatureFreshnessResult:
    """trades + OB データの鮮度を包括チェック.

    retrain_scheduler の事前ガードとして、データが十分新鮮かを判定する。
    fill_test が動作中なら両方のデータは継続的に更新されるはず。

    Args:
        raw_dir: raw data ディレクトリ (default: data/v460/raw).
        trades_stale_hours: trades データの許容経過時間.
        ob_stale_hours: OB データの許容経過時間.

    Returns:
        FeatureFreshnessResult
    """
    d = resolve_raw_dir(raw_dir)
    now_ts = datetime.now(timezone.utc).timestamp()

    tr_hours = _latest_mtime_hours(d / "trades", now_ts=now_ts)
    ob_hours = _latest_mtime_hours(d / "orderbook", now_ts=now_ts)

    details: dict[str, str] = {}
    issues: list[str] = []

    if tr_hours > trades_stale_hours:
        issues.append(f"trades stale ({tr_hours:.1f}h > {trades_stale_hours}h)")
        details["trades"] = f"stale ({tr_hours:.1f}h)"
    else:
        details["trades"] = f"fresh ({tr_hours:.1f}h)"

    if ob_hours > ob_stale_hours:
        issues.append(f"OB stale ({ob_hours:.1f}h > {ob_stale_hours}h)")
        details["ob"] = f"stale ({ob_hours:.1f}h)"
    else:
        details["ob"] = f"fresh ({ob_hours:.1f}h)"

    fresh = len(issues) == 0
    if fresh:
        msg = f"FRESH: trades={tr_hours:.1f}h, OB={ob_hours:.1f}h"
    else:
        msg = "STALE: " + "; ".join(issues)

    if not fresh:
        logger.warning(f"[136# P1-02] Feature freshness: {msg}")

    return FeatureFreshnessResult(
        fresh=fresh,
        trades_stale_hours=tr_hours,
        ob_stale_hours=ob_hours,
        details=details,
        message=msg,
    )

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Trades data health check")
    parser.add_argument("--raw-dir", default=None, help="Raw data dir")
    parser.add_argument("--days", type=int, default=3, help="Lookback days")
    parser.add_argument("--max-missing", type=int, default=0, help="Max missing days to tolerate")
    args = parser.parse_args()

    raw = resolve_raw_dir(Path(args.raw_dir)) if args.raw_dir else None
    result = check_trades_health(raw_dir=raw, lookback_days=args.days,
                                max_missing_days=args.max_missing)
    print(result.message)
    if not result.healthy:
        raise SystemExit(1)
