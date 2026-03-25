"""346# S-7: テール損失分析 — downside_p10 改善のための特徴抽出.

316# S-7: downside_p10 が全 regime で fail → テール事象の条件付き
回避ルール設計に必要な特徴を抽出する。

**ゴール**: テール損失の 30% を回避する conditional skip rule の設計。
p10 を -5.0 bps 以内に収める。

分析軸:
  - Regime over-representation (テールに集中する regime)
  - 時間帯 over-representation (UTC hour)
  - Decision path 分布
  - Spread / Velocity / OBI 条件
  - SkipGate / Early Exit 影響
  - balance_forced_switch 交絡

Usage:
    .venv\\Scripts\\python.exe scripts/v460/analysis/tail_loss_analysis.py
    .venv\\Scripts\\python.exe scripts/v460/analysis/tail_loss_analysis.py --git-sha abc1234
    .venv\\Scripts\\python.exe scripts/v460/analysis/tail_loss_analysis.py --date-from 2026-03-01
    .venv\\Scripts\\python.exe scripts/v460/analysis/tail_loss_analysis.py --percentile 5 --output results.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Final, TypedDict, cast

import numpy as np

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    Record,
    extract_filled,
    extract_pnl_array,
    load_and_filter_records,
    record_to_utc_hour,
    write_output,
    write_json_output,
)
from ztb.utils.safety import safe_to_finite

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT: Final[Path] = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ======================================================================
# 定数
# ======================================================================

_DEFAULT_PERCENTILE: Final[float] = 10.0
_MIN_RECORDS_FOR_ANALYSIS: Final[int] = 20
_MIN_TAIL_FOR_OVERREP: Final[int] = 3
_OVERREP_SIGNIFICANT_THRESHOLD: Final[float] = 1.5  # 1.5x 以上を有意とみなす


# ======================================================================
# TypedDict (型安全な出力定義)
# ======================================================================

class OverrepEntry(TypedDict):
    """Over-representation エントリ."""

    tail_n: int
    total_n: int
    tail_share: float
    total_share: float
    overrep_ratio: float


class HourEntry(TypedDict):
    """時間帯 over-representation エントリ."""

    hour: int
    tail_n: int
    total_n: int
    tail_share: float
    total_share: float
    overrep_ratio: float


class FeatureStats(TypedDict, total=False):
    """数値特徴量の tail vs total 統計."""

    tail_mean: float | None
    total_mean: float | None
    tail_median: float | None
    total_median: float | None
    tail_p90: float | None


class SideResult(TypedDict, total=False):
    """片側のテール損失分析結果."""

    n: int
    tail_n: int
    tail_threshold_bps: float
    tail_mean_pnl_bps: float | None
    tail_p5_pnl_bps: float | None
    as_rate_tail: float
    as_rate_total: float
    as_overrep: float | None
    early_exit_rate_tail: float | None
    early_exit_rate_total: float | None
    balance_forced_rate_tail: float | None
    balance_forced_rate_total: float | None
    regime_overrep: dict[str, OverrepEntry]
    hour_overrep: list[HourEntry]
    decision_path_counts: dict[str, int]
    spread_stats: FeatureStats
    velocity_stats: FeatureStats
    obi_stats: FeatureStats
    skip_gate_score_stats: FeatureStats
    actionable_filters: list[dict[str, object]]
    message: str


# ======================================================================
# ヘルパー関数
# ======================================================================

# _extract_filled / _pnl_array → analysis_common.extract_filled / extract_pnl_array


def _as_rate(records: list[Record]) -> float:
    """Adverse Selection レート."""
    if not records:
        return 0.0
    n_as = sum(1 for r in records if r.get("adverse_selected"))
    return n_as / len(records)


def _flag_rate(
    records: list[Record], key: str,
) -> float | None:
    """bool フラグのレート (None 非対応を無視)."""
    valid = [r for r in records if r.get(key) is not None]
    if not valid:
        return None
    return sum(1 for r in valid if r.get(key)) / len(valid)


# _record_to_utc_hour → analysis_common.record_to_utc_hour


def _numeric_field_stats(
    tail_records: list[Record],
    all_records: list[Record],
    field: str,
) -> FeatureStats:
    """数値フィールドの tail/total 比較統計."""
    tail_vals = [
        v for r in tail_records
        if (v := safe_to_finite(r.get(field))) is not None
    ]
    total_vals = [
        v for r in all_records
        if (v := safe_to_finite(r.get(field))) is not None
    ]
    result: FeatureStats = {}
    if tail_vals:
        result["tail_mean"] = round(float(np.mean(tail_vals)), 4)
        result["tail_median"] = round(float(np.median(tail_vals)), 4)
        if len(tail_vals) >= 5:
            result["tail_p90"] = round(float(np.percentile(tail_vals, 90)), 4)
    else:
        result["tail_mean"] = None
        result["tail_median"] = None
    if total_vals:
        result["total_mean"] = round(float(np.mean(total_vals)), 4)
        result["total_median"] = round(float(np.median(total_vals)), 4)
    else:
        result["total_mean"] = None
        result["total_median"] = None
    return result


def _compute_overrep(
    tail_records: list[Record],
    all_records: list[Record],
    key: str,
    *,
    coerce_str: bool = True,
) -> dict[str, OverrepEntry]:
    """カテゴリカルフィールドの over-representation を計算."""
    tail_counts: dict[str, int] = defaultdict(int)
    total_counts: dict[str, int] = defaultdict(int)

    for r in tail_records:
        val = str(r.get(key) or "null") if coerce_str else r.get(key)
        tail_counts[str(val)] += 1
    for r in all_records:
        val = str(r.get(key) or "null") if coerce_str else r.get(key)
        total_counts[str(val)] += 1

    result: dict[str, OverrepEntry] = {}
    n_tail = len(tail_records)
    n_total = len(all_records)

    for cat, tail_n in tail_counts.items():
        total_n = total_counts.get(cat, 0)
        if total_n == 0:
            continue
        tail_share = tail_n / n_tail if n_tail > 0 else 0.0
        total_share = total_n / n_total if n_total > 0 else 0.0
        overrep = tail_share / total_share if total_share > 0 else 0.0
        result[cat] = OverrepEntry(
            tail_n=tail_n,
            total_n=total_n,
            tail_share=round(tail_share, 4),
            total_share=round(total_share, 4),
            overrep_ratio=round(overrep, 3),
        )
    return result


def _compute_hour_overrep(
    tail_records: list[Record],
    all_records: list[Record],
) -> list[HourEntry]:
    """時間帯別 over-representation を計算 (上位順ソート)."""
    tail_hours: dict[int, int] = defaultdict(int)
    total_hours: dict[int, int] = defaultdict(int)

    for r in tail_records:
        h = record_to_utc_hour(r)
        if h is not None:
            tail_hours[h] += 1
    for r in all_records:
        h = record_to_utc_hour(r)
        if h is not None:
            total_hours[h] += 1

    n_tail = len(tail_records)
    n_total = len(all_records)
    entries: list[HourEntry] = []

    for h in range(24):
        t_n = tail_hours.get(h, 0)
        a_n = total_hours.get(h, 0)
        if a_n == 0:
            continue
        t_share = t_n / n_tail if n_tail > 0 else 0.0
        a_share = a_n / n_total if n_total > 0 else 0.0
        overrep = t_share / a_share if a_share > 0 else 0.0
        entries.append(HourEntry(
            hour=h,
            tail_n=t_n,
            total_n=a_n,
            tail_share=round(t_share, 4),
            total_share=round(a_share, 4),
            overrep_ratio=round(overrep, 3),
        ))
    entries.sort(key=lambda x: x["overrep_ratio"], reverse=True)
    return entries


# ======================================================================
# アクション可能フィルタ候補の自動検出
# ======================================================================

def _derive_actionable_filters(
    tail_records: list[Record],
    all_records: list[Record],
    regime_overrep: dict[str, OverrepEntry],
    hour_overrep: list[HourEntry],
) -> list[dict[str, object]]:
    """テール損失を回避できる可能性のある条件付き skip ルール候補を列挙.

    各ルール候補に対して:
    - 回避できるテール records 数
    - 同時に犠牲になる非テール records 数
    - 効率性指標 (回避テール数 / 犠牲非テール数)
    """
    proposals: list[dict[str, object]] = []
    n_tail = len(tail_records)
    n_all = len(all_records)
    if n_tail == 0 or n_all == 0:
        return proposals

    # 1) Regime-based skip
    for regime, entry in regime_overrep.items():
        if entry["overrep_ratio"] < _OVERREP_SIGNIFICANT_THRESHOLD:
            continue
        if entry["tail_n"] < _MIN_TAIL_FOR_OVERREP:
            continue
        # この regime を skip した場合: tail から entry["tail_n"] 回避, total から entry["total_n"] 犠牲
        sacrifice_non_tail = entry["total_n"] - entry["tail_n"]
        efficiency = entry["tail_n"] / max(sacrifice_non_tail, 1)
        proposals.append({
            "type": "regime_skip",
            "condition": f"regime == '{regime}'",
            "tail_avoided": entry["tail_n"],
            "tail_avoided_pct": round(entry["tail_n"] / n_tail, 3),
            "sacrifice_non_tail": sacrifice_non_tail,
            "efficiency": round(efficiency, 3),
            "overrep_ratio": entry["overrep_ratio"],
        })

    # 2) Hour-based skip (top candidates)
    for h_entry in hour_overrep[:5]:
        if h_entry["overrep_ratio"] < _OVERREP_SIGNIFICANT_THRESHOLD:
            continue
        if h_entry["tail_n"] < _MIN_TAIL_FOR_OVERREP:
            continue
        sacrifice = h_entry["total_n"] - h_entry["tail_n"]
        efficiency = h_entry["tail_n"] / max(sacrifice, 1)
        proposals.append({
            "type": "hour_skip",
            "condition": f"utc_hour == {h_entry['hour']}",
            "tail_avoided": h_entry["tail_n"],
            "tail_avoided_pct": round(h_entry["tail_n"] / n_tail, 3),
            "sacrifice_non_tail": sacrifice,
            "efficiency": round(efficiency, 3),
            "overrep_ratio": h_entry["overrep_ratio"],
        })

    # 3) High-spread skip: spread_at_order > p75 of tail records
    tail_spreads = [
        v for r in tail_records
        if (v := safe_to_finite(r.get("spread_at_order"))) is not None
    ]
    all_spreads = [
        v for r in all_records
        if (v := safe_to_finite(r.get("spread_at_order"))) is not None
    ]
    if len(tail_spreads) >= 5 and all_spreads:
        spread_threshold = float(np.percentile(tail_spreads, 75))
        # テールで spread >= threshold のレコード数
        tail_caught = sum(1 for s in tail_spreads if s >= spread_threshold)
        all_caught = sum(1 for s in all_spreads if s >= spread_threshold)
        sacrifice = all_caught - tail_caught
        if tail_caught >= _MIN_TAIL_FOR_OVERREP:
            proposals.append({
                "type": "spread_skip",
                "condition": f"spread_at_order >= {spread_threshold:.0f}",
                "tail_avoided": tail_caught,
                "tail_avoided_pct": round(tail_caught / n_tail, 3),
                "sacrifice_non_tail": sacrifice,
                "efficiency": round(tail_caught / max(sacrifice, 1), 3),
                "spread_threshold": round(spread_threshold, 1),
            })

    # 4) High velocity skip: |mid_price_trend_5s| > p75 of tail
    tail_vels = [
        abs(v) for r in tail_records
        if (v := safe_to_finite(r.get("mid_price_trend_5s"))) is not None
    ]
    all_vels = [
        abs(v) for r in all_records
        if (v := safe_to_finite(r.get("mid_price_trend_5s"))) is not None
    ]
    if len(tail_vels) >= 5 and all_vels:
        vel_threshold = float(np.percentile(tail_vels, 75))
        tail_caught = sum(1 for v in tail_vels if v >= vel_threshold)
        all_caught = sum(1 for v in all_vels if v >= vel_threshold)
        sacrifice = all_caught - tail_caught
        if tail_caught >= _MIN_TAIL_FOR_OVERREP:
            proposals.append({
                "type": "velocity_skip",
                "condition": f"|mid_price_trend_5s| >= {vel_threshold:.4f}",
                "tail_avoided": tail_caught,
                "tail_avoided_pct": round(tail_caught / n_tail, 3),
                "sacrifice_non_tail": sacrifice,
                "efficiency": round(tail_caught / max(sacrifice, 1), 3),
                "velocity_threshold_bps": round(vel_threshold, 4),
            })

    # ソート: 効率性降順
    proposals.sort(
        key=lambda x: safe_to_finite(x.get("efficiency")) or 0.0,
        reverse=True,
    )
    return proposals


# ======================================================================
# メイン分析関数
# ======================================================================

def analyze_tail_loss(
    records: list[Record],
    percentile: float = _DEFAULT_PERCENTILE,
) -> dict[str, SideResult]:
    """side 別にテール損失を分析.

    Args:
        records: 全 fill records (dict)
        percentile: テール閾値 (default=10 → p10 以下)

    Returns:
        {"sell": SideResult, "buy": SideResult}
    """
    result: dict[str, SideResult] = {}

    for side in ["sell", "buy"]:
        filled = extract_filled(records, side=side)
        arr = extract_pnl_array(filled)

        if len(arr) < _MIN_RECORDS_FOR_ANALYSIS:
            result[side] = SideResult(
                n=len(filled),
                tail_n=0,
                message=f"insufficient data (n={len(arr)} < {_MIN_RECORDS_FOR_ANALYSIS})",
            )
            continue

        threshold = float(np.percentile(arr, percentile))

        # テール records 抽出
        tail_records: list[Record] = []
        for r in filled:
            pnl_val = safe_to_finite(r.get("post_fill_30s_pnl"))
            if pnl_val is not None and pnl_val <= threshold:
                tail_records.append(r)

        tail_pnl = extract_pnl_array(tail_records)

        # --- 分析軸 ---
        # 1) Regime over-representation
        regime_overrep = _compute_overrep(tail_records, filled, "regime")

        # 2) 時間帯 over-representation
        hour_overrep = _compute_hour_overrep(tail_records, filled)

        # 3) Decision path
        path_counts: dict[str, int] = defaultdict(int)
        for r in tail_records:
            path = str(r.get("decision_path") or "unknown")
            path_counts[path] += 1

        # 4) 数値特徴量の tail vs total 比較
        spread_stats = _numeric_field_stats(tail_records, filled, "spread_at_order")
        velocity_stats = _numeric_field_stats(tail_records, filled, "mid_price_trend_5s")
        obi_stats = _numeric_field_stats(tail_records, filled, "orderbook_imbalance")
        sg_stats = _numeric_field_stats(tail_records, filled, "skip_gate_score")

        # 5) AS / Early Exit / balance_forced rates
        tail_as = _as_rate(tail_records)
        total_as = _as_rate(filled)
        tail_ee = _flag_rate(tail_records, "early_exit_triggered")
        total_ee = _flag_rate(filled, "early_exit_triggered")
        tail_bf = _flag_rate(tail_records, "balance_forced_switch")
        total_bf = _flag_rate(filled, "balance_forced_switch")

        # 6) アクション可能フィルタ候補
        actionable = _derive_actionable_filters(
            tail_records, filled, regime_overrep, hour_overrep,
        )

        result[side] = SideResult(
            n=len(filled),
            tail_n=len(tail_records),
            tail_threshold_bps=round(threshold, 4),
            tail_mean_pnl_bps=(
                round(float(np.mean(tail_pnl)), 4) if len(tail_pnl) > 0 else None
            ),
            tail_p5_pnl_bps=(
                round(float(np.percentile(tail_pnl, 5)), 4)
                if len(tail_pnl) >= 5  # noqa: PLR2004
                else None
            ),
            as_rate_tail=round(tail_as, 4),
            as_rate_total=round(total_as, 4),
            as_overrep=round(tail_as / total_as, 3) if total_as > 0 else None,
            early_exit_rate_tail=round(tail_ee, 4) if tail_ee is not None else None,
            early_exit_rate_total=round(total_ee, 4) if total_ee is not None else None,
            balance_forced_rate_tail=(
                round(tail_bf, 4) if tail_bf is not None else None
            ),
            balance_forced_rate_total=(
                round(total_bf, 4) if total_bf is not None else None
            ),
            regime_overrep=regime_overrep,
            hour_overrep=hour_overrep[:10],
            decision_path_counts=dict(path_counts),
            spread_stats=spread_stats,
            velocity_stats=velocity_stats,
            obi_stats=obi_stats,
            skip_gate_score_stats=sg_stats,
            actionable_filters=actionable,
            message="",
        )

    return result


# ======================================================================
# コンソール出力
# ======================================================================

def print_analysis(
    analysis: dict[str, SideResult],
    *,
    percentile: float = _DEFAULT_PERCENTILE,
) -> None:
    """分析結果をコンソール出力."""
    buffer = StringIO()

    def emit(*values: object) -> None:
        print(*values, file=buffer)

    emit("=" * 70)
    emit(f"  346# S-7: テール損失分析 (p{percentile:.0f} 以下)")
    emit("=" * 70)

    for side in ["sell", "buy"]:
        d = analysis.get(side)
        if d is None:
            continue

        msg = d.get("message", "")
        if msg:
            emit(f"\n  [{side.upper()}] {msg}")
            continue

        n = d.get("n", 0)
        tail_n = d.get("tail_n", 0)
        threshold = d.get("tail_threshold_bps", 0.0)
        tail_mean = d.get("tail_mean_pnl_bps")
        tail_p5 = d.get("tail_p5_pnl_bps")

        emit(f"\n{'─' * 70}")
        emit(f"  [{side.upper()}] n={n}, tail(p{percentile:.0f}以下): n={tail_n}")
        emit(f"{'─' * 70}")
        emit(f"    tail threshold:  {threshold:+.4f} bps")
        if tail_mean is not None:
            emit(f"    tail mean pnl:   {tail_mean:+.4f} bps")
        if tail_p5 is not None:
            emit(f"    tail p5 (worst): {tail_p5:+.4f} bps")

        # AS
        as_tail = d.get("as_rate_tail", 0.0)
        as_total = d.get("as_rate_total", 0.0)
        as_overrep = d.get("as_overrep")
        emit(
            f"\n    [AS] tail={as_tail:.1%} vs total={as_total:.1%}"
            f" (overrep={as_overrep:.2f}x)" if as_overrep else ""
        )

        # Early Exit
        ee_tail = d.get("early_exit_rate_tail")
        ee_total = d.get("early_exit_rate_total")
        if ee_tail is not None and ee_total is not None:
            emit(f"    [EE] tail={ee_tail:.1%} vs total={ee_total:.1%}")

        # Balance forced
        bf_tail = d.get("balance_forced_rate_tail")
        bf_total = d.get("balance_forced_rate_total")
        if bf_tail is not None and bf_total is not None:
            emit(f"    [BF] tail={bf_tail:.1%} vs total={bf_total:.1%}")

        # Regime over-representation
        regime_overrep = d.get("regime_overrep", {})
        if regime_overrep:
            emit("\n    Regime over-representation:")
            sorted_regimes = sorted(
                regime_overrep.items(),
                key=lambda x: x[1]["overrep_ratio"],
                reverse=True,
            )
            for regime, info in sorted_regimes:
                marker = " ⚠" if info["overrep_ratio"] >= _OVERREP_SIGNIFICANT_THRESHOLD else ""
                emit(
                    f"      {regime:15s}: tail={info['tail_n']:3d}/{info['total_n']:4d}"
                    f" ({info['tail_share']:.1%} vs {info['total_share']:.1%})"
                    f" overrep={info['overrep_ratio']:.2f}x{marker}"
                )

        # Hour over-representation (top-5)
        hour_overrep = d.get("hour_overrep", [])
        if hour_overrep:
            emit("\n    Hour over-representation (top-5):")
            for h_entry in hour_overrep[:5]:
                marker = " ⚠" if h_entry["overrep_ratio"] >= _OVERREP_SIGNIFICANT_THRESHOLD else ""
                jst_h = (h_entry["hour"] + 9) % 24
                emit(
                    f"      UTC {h_entry['hour']:02d} (JST {jst_h:02d}): "
                    f"tail={h_entry['tail_n']:3d}/{h_entry['total_n']:4d}"
                    f" overrep={h_entry['overrep_ratio']:.2f}x{marker}"
                )

        # Decision path
        path_counts = d.get("decision_path_counts", {})
        if path_counts:
            emit("\n    Decision path (tail):")
            for path, count in sorted(path_counts.items(), key=lambda x: -x[1]):
                pct = count / tail_n if tail_n > 0 else 0
                emit(f"      {path:25s}: {count:3d} ({pct:.1%})")

        # Feature stats
        for label, stats_key in [
            ("spread_at_order", "spread_stats"),
            ("mid_price_trend_5s", "velocity_stats"),
            ("orderbook_imbalance", "obi_stats"),
            ("skip_gate_score", "skip_gate_score_stats"),
        ]:
            stats = cast(FeatureStats, d.get(stats_key, {}))
            if not stats:
                continue
            t_mean = stats.get("tail_mean")
            a_mean = stats.get("total_mean")
            if t_mean is not None and a_mean is not None:
                emit(f"\n    [{label}] tail_mean={t_mean:+.4f} vs total_mean={a_mean:+.4f}")
                t_med = stats.get("tail_median")
                a_med = stats.get("total_median")
                if t_med is not None and a_med is not None:
                    emit(f"      tail_median={t_med:+.4f} vs total_median={a_med:+.4f}")

        # Actionable filters
        actionable = d.get("actionable_filters", [])
        if actionable:
            emit(f"\n    {'─' * 50}")
            emit("    Actionable skip rule 候補 (効率性順):")
            for i, prop in enumerate(actionable, 1):
                tail_pct = prop.get("tail_avoided_pct", 0)
                emit(f"      [{i}] {prop['type']}: {prop['condition']}")
                emit(
                    f"          回避テール: {prop['tail_avoided']}件 ({tail_pct:.0%})"
                    f"  犠牲非テール: {prop['sacrifice_non_tail']}件"
                    f"  効率: {prop['efficiency']:.2f}"
                )

    write_output(buffer.getvalue().rstrip())


# ======================================================================
# CLI
# ======================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="346# S-7: テール損失分析 (downside_p10 改善)",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"fill_records ディレクトリ (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=_DEFAULT_PERCENTILE,
        help=f"テール閾値パーセンタイル (default: {_DEFAULT_PERCENTILE})",
    )
    parser.add_argument("--git-sha", default=None, help="git SHA フィルタ")
    parser.add_argument("--date-from", default=None, help="開始日 (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="終了日 (YYYY-MM-DD)")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON 出力パス (省略時は analysis_results/ に自動保存)",
    )
    return parser


def main(args: argparse.Namespace | Sequence[str] | None = None) -> dict[str, SideResult]:
    """メインエントリポイント."""
    if args is None:
        args = _build_parser().parse_args()
    elif not isinstance(args, argparse.Namespace):
        args = _build_parser().parse_args(list(args))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = Path(args.results_dir)
    write_output(f"Loading fill records from: {results_dir}")

    records = load_and_filter_records(
        str(results_dir),
        git_sha=args.git_sha,
        date_from=args.date_from,
        date_to=args.date_to,
        include_emergency=False,
    )
    write_output(f"Total records: {len(records)}")

    filled = extract_filled(records)
    write_output(f"Filled records: {len(filled)}")

    # 分析実行
    analysis = analyze_tail_loss(records, percentile=args.percentile)

    # コンソール出力
    print_analysis(analysis, percentile=args.percentile)

    # JSON 保存
    output_path: Path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = _PROJECT_ROOT / "analysis_results"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"346_tail_loss_p{args.percentile:.0f}_{timestamp}.json"

    output_data = {
        "script": "346# S-7 tail_loss_analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "percentile": args.percentile,
        "filters": {
            "git_sha": args.git_sha,
            "date_from": args.date_from,
            "date_to": args.date_to,
        },
        "total_records": len(records),
        "filled_records": len(filled),
        "analysis": analysis,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_output(output_data, output_path)
    write_output(f"JSON saved: {output_path}")

    return analysis


if __name__ == "__main__":
    main()
