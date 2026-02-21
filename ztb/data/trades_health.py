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

logger = logging.getLogger(__name__)

_DEFAULT_RAW_DIR = Path("data/v460/raw")


@dataclass(frozen=True)
class TradesHealthResult:
    """trades 健全性チェック結果."""

    healthy: bool
    available_days: list[str]
    missing_days: list[str]
    stale_hours: float  # 最新ファイルからの経過時間
    message: str


def check_trades_health(
    raw_dir: Path | None = None,
    expected_days: list[str] | None = None,
    *,
    lookback_days: int = 3,
    stale_threshold_hours: float = 36.0,
) -> TradesHealthResult:
    """trades raw データの健全性をチェック.

    Args:
        raw_dir: raw data ディレクトリ (default: data/v460/raw).
        expected_days: チェック対象の日付リスト (YYYYMMDD).
            None の場合は直近 lookback_days 日を自動生成.
        lookback_days: expected_days=None 時の遡り日数.
        stale_threshold_hours: 最新ファイルがこれ以上古い場合は unhealthy.

    Returns:
        TradesHealthResult: 結果オブジェクト.
    """
    d = raw_dir or _DEFAULT_RAW_DIR
    tr_dir = d / "trades"

    # 利用可能な日の列挙
    available: list[str] = []
    if tr_dir.exists():
        for f in sorted(tr_dir.glob("*.jsonl.gz")):
            stem = f.stem.replace(".jsonl", "")
            if len(stem) == 8 and stem.isdigit():
                available.append(stem)

    # 期待日リスト
    # §9.2 #B: 当日 UTC を除外し「昨日から N 日遡り」で生成。
    # 日跨ぎ直後 (00:00-01:00 UTC) に当日ファイル未生成で false warning を防止。
    if expected_days is None:
        now = datetime.now(timezone.utc)
        expected_days = [
            (now - timedelta(days=i + 1)).strftime("%Y%m%d")
            for i in range(lookback_days)
        ]

    # 欠損日
    missing = [d for d in expected_days if d not in available]

    # 鮮度チェック
    stale_hours = float("inf")
    if available:
        latest_day = available[-1]
        latest_file = tr_dir / f"{latest_day}.jsonl.gz"
        if latest_file.exists():
            mtime = latest_file.stat().st_mtime
            stale_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
        else:
            stale_hours = float("inf")

    # 判定
    healthy = len(missing) == 0 and stale_hours < stale_threshold_hours
    if not healthy:
        parts: list[str] = []
        if missing:
            parts.append(f"missing_days={missing}")
        if stale_hours >= stale_threshold_hours:
            parts.append(f"stale={stale_hours:.1f}h (threshold={stale_threshold_hours}h)")
        msg = "UNHEALTHY: " + ", ".join(parts)
    else:
        msg = f"OK: {len(available)} days available, stale={stale_hours:.1f}h"

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


def _latest_mtime_hours(directory: Path, glob_pattern: str = "*.jsonl.gz") -> float:
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
    return (datetime.now(timezone.utc).timestamp() - latest_mtime) / 3600


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
    d = raw_dir or _DEFAULT_RAW_DIR

    tr_hours = _latest_mtime_hours(d / "trades")
    ob_hours = _latest_mtime_hours(d / "orderbook")

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
    args = parser.parse_args()

    raw = Path(args.raw_dir) if args.raw_dir else None
    result = check_trades_health(raw_dir=raw, lookback_days=args.days)
    print(result.message)
    if not result.healthy:
        raise SystemExit(1)
