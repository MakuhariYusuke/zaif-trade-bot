"""135# P2-09→P1: trades データ健全性チェック.

run 開始時に trades raw データの存在を検証し、欠損日を検出する。
retrain が全量 fallback して特徴量の時間整合性が崩れるのを早期に防ぐ。

Usage (ライブラリとして):
    from ztb.data.trades_health import check_trades_health, TradesHealthResult
    result = check_trades_health(expected_days=["20260220", "20260221"])

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
