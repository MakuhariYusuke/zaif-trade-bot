"""分析スクリプト共通モジュール.

scripts/v460/analysis/ 配下の分析スクリプトで重複していた
CLI引数定義・データ読み込み・PnL抽出・出力処理を集約。

内部的に ztb.metrics.fill_quality / ztb.utils.safety に委譲する。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ztb.metrics.fill_quality import (
    apply_fill_record_filters,
    load_fill_record_objects_glob,
)
from ztb.utils.safety import safe_to_finite

# ======================================================================
# 型エイリアス
# ======================================================================

Record: TypeAlias = dict[str, object]
FloatArray: TypeAlias = NDArray[np.float64]

# ======================================================================
# 定数
# ======================================================================

DEFAULT_RESULTS_DIR: Final[str] = "results/v460/fill_test"
"""fill_records_*.jsonl のデフォルト格納ディレクトリ."""

# PnL 判定閾値 (bps)
AS_THRESHOLD_BPS: Final[float] = -3.0
"""Adverse Selection とみなす閾値."""

SEVERE_AS_THRESHOLD_BPS: Final[float] = -10.0
"""重度 Adverse Selection とみなす閾値."""

# PnL フィールド優先順位 (fallback chain)
PNL_FIELD_PRIORITY: Final[tuple[str, ...]] = (
    "ev_weighted_pnl",
    "post_fill_30s_pnl",
    "pnl_bps",
)
"""get_pnl() が参照するフィールドの優先順位."""


# ======================================================================
# CLI 引数ビルダー
# ======================================================================

def add_common_filter_args(parser: argparse.ArgumentParser) -> None:
    """共通フィルタ引数を parser に追加.

    追加される引数:
      --results-dir, --date-from, --date-to, --git-sha, --run-id
    """
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"fill_records ディレクトリ (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--date-from", default=None, help="開始日 inclusive (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="終了日 inclusive (YYYY-MM-DD)")
    parser.add_argument("--git-sha", default=None, help="git SHA 前方一致フィルタ (短縮 SHA 可)")
    parser.add_argument("--run-id", default=None, help="run_id 完全一致フィルタ")


def add_side_regime_args(parser: argparse.ArgumentParser) -> None:
    """side / regime フィルタ引数を追加."""
    parser.add_argument(
        "--side",
        choices=["buy", "sell"],
        default=None,
        help="side フィルタ (省略時: 全 side)",
    )
    parser.add_argument(
        "--regime",
        default=None,
        help="regime 完全一致フィルタ (trending/ranging/volatile)",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """出力先引数 (--output, --json) を追加."""
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="結果をファイルに書き出す (省略時: stdout)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON 形式で出力",
    )


# ======================================================================
# データ読み込み
# ======================================================================

def load_and_filter_records(
    results_dir: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    git_sha: str | None = None,
    run_id: str | None = None,
    side: str | None = None,
    regime: str | None = None,
    include_emergency: bool = True,
    exit_on_empty: bool = True,
) -> list[Record]:
    """JSONL 読み込み → フィルタ適用を一括実行.

    fill_quality 共有 API に委譲し、side/regime はローカルで適用。

    Args:
        results_dir: fill_records ディレクトリ
        date_from: 開始日 (YYYY-MM-DD)
        date_to: 終了日 (YYYY-MM-DD)
        git_sha: git SHA 前方一致
        run_id: run_id 完全一致
        side: "buy" / "sell"
        regime: "trending" / "ranging" / "volatile"
        include_emergency: emergency ファイルを含めるか
        exit_on_empty: True の場合、結果0件で sys.exit(1)

    Returns:
        フィルタ済みレコードリスト
    """
    base = Path(results_dir)
    if not base.exists():
        print(f"ERROR: data directory not found: {base}", file=sys.stderr)
        if exit_on_empty:
            sys.exit(1)
        return []

    records = load_fill_record_objects_glob(
        base,
        include_emergency=include_emergency,
        start_date=date_from,
        end_date=date_to,
    )
    if not records:
        print(f"ERROR: no fill_records found in {base}", file=sys.stderr)
        if exit_on_empty:
            sys.exit(1)
        return []

    # fill_quality 側フィルタ
    filtered, _ = apply_fill_record_filters(
        records,
        run_id=run_id,
        git_sha=git_sha,
        date_from=date_from,
        date_to=date_to,
    )

    # side / regime はローカルで適用
    if side:
        filtered = [r for r in filtered if r.get("side") == side]
    if regime:
        filtered = [r for r in filtered if r.get("regime") == regime]

    if not filtered and exit_on_empty:
        print("ERROR: all records filtered out", file=sys.stderr)
        sys.exit(1)

    return cast(list[Record], filtered)


def load_records_from_args(
    args: argparse.Namespace,
    *,
    exit_on_empty: bool = True,
) -> list[Record]:
    """argparse.Namespace からフィルタパラメータを取り出してデータ読み込み.

    add_common_filter_args() + add_side_regime_args() で追加した引数に対応。
    """
    return load_and_filter_records(
        getattr(args, "results_dir", DEFAULT_RESULTS_DIR),
        date_from=getattr(args, "date_from", None),
        date_to=getattr(args, "date_to", None),
        git_sha=getattr(args, "git_sha", None),
        run_id=getattr(args, "run_id", None),
        side=getattr(args, "side", None),
        regime=getattr(args, "regime", None),
        exit_on_empty=exit_on_empty,
    )


# ======================================================================
# PnL 抽出ヘルパー
# ======================================================================

def get_pnl(record: Record) -> float | None:
    """レコードから PnL(bps) を抽出 (fallback chain 付き).

    PNL_FIELD_PRIORITY の順に探索し、最初に有効な値を返す。
    sha_comparison / hour_matched_comparison と同等。
    """
    for key in PNL_FIELD_PRIORITY:
        v = record.get(key)
        if v is not None:
            val = safe_to_finite(v)
            if val is not None:
                return float(val)
    return None


def extract_pnl_array(
    records: list[Record],
    key: str = "post_fill_30s_pnl",
) -> FloatArray:
    """レコード群から単一キーの PnL を float64 配列に変換.

    analyze_fill_logs._pnls() / tail_loss_analysis._pnl_array() と同等。
    """
    vals: list[float] = []
    for r in records:
        v = safe_to_finite(r.get(key))
        if v is not None:
            vals.append(float(v))
    return np.array(vals, dtype=np.float64) if vals else np.array([], dtype=np.float64)


def extract_pnl_list(records: list[Record]) -> list[float]:
    """レコード群から PnL(bps) をリストで抽出 (fallback chain)."""
    return [p for r in records if (p := get_pnl(r)) is not None]


# ======================================================================
# フィルタヘルパー
# ======================================================================

def extract_filled(
    records: list[Record],
    *,
    side: str | None = None,
) -> list[Record]:
    """約定済みレコードを抽出 (オプションで side 絞り込み)."""
    out: list[Record] = []
    for r in records:
        if not r.get("filled"):
            continue
        if side and r.get("side") != side:
            continue
        out.append(r)
    return out


# ======================================================================
# タイムスタンプヘルパー
# ======================================================================

def record_to_utc_hour(record: Record) -> int | None:
    """レコードの timestamp → UTC hour."""
    ts = record.get("timestamp")
    if ts is None:
        return None
    try:
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        else:
            return None
        return dt.hour
    except (ValueError, OSError, OverflowError):
        return None


# ======================================================================
# 出力ヘルパー
# ======================================================================

def write_output(
    content: str,
    output_path: str | Path | None = None,
) -> None:
    """文字列を stdout またはファイルに出力."""
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print(f"Written to {p}", file=sys.stderr)
    else:
        print(content)


def write_json_output(
    data: object,
    output_path: str | Path | None = None,
) -> None:
    """JSON データを stdout またはファイルに出力."""
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    write_output(text, output_path)
