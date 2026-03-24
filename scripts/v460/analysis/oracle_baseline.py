#!/usr/bin/env python3
"""131# D2: Oracle PnL 基準線テスト.

fill_records を使い、「完全予測Oracle」が達成可能な理論上限 PnL を算出する。
v459 教訓③ (118# §8.5): maker 0% でも AS コストが Oracle PnL を超えていないか確認。

Oracle 定義:
  - 完全予測: 約定後の 30s/60s/120s PnL を事前に完全に知っている前提
  - Oracle は PnL < 0 の取引を全て skip し、PnL ≥ 0 の取引のみ実行
  - 約定率 = 既存の fill_rate (maker 発注の約定確率は Oracle でも変わらない)
  - 手数料 = maker 0% (Coincheck)

出力:
  - 全体/side別/レジーム別の Oracle PnL 統計
  - 実績 PnL との比較 → 改善余地の定量評価
  - ph3 進入判定: Oracle PnL > 0 が必須条件

Usage:
  python scripts/v460/analysis/oracle_baseline.py
  python scripts/v460/analysis/oracle_baseline.py --results-dir results/v460/fill_test
  python scripts/v460/analysis/oracle_baseline.py --output results/v460/fill_test/oracle_report.json
  python scripts/v460/analysis/oracle_baseline.py --lot-btc 0.01  # 仮想ロットサイズでの月間JPY換算
"""

from __future__ import annotations

import argparse

from collections.abc import Sequence
from ztb.utils.safety import safe_to_finite
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    PnlAccumulator,
    iter_fill_records_glob,
    partition_clean_records,
)
from scripts.v460.analysis.analysis_common import add_results_dir_arg, write_json_output
from ztb.utils.dataclass_utils import shallow_asdict


# ======================================================================
# データクラス
# ======================================================================

@dataclass
class OracleMetrics:
    """Oracle PnL 統計."""

    label: str
    n_total: int  # 全約定数 (filled)
    n_positive: int  # Oracle が実行する取引 (PnL ≥ 0)
    n_negative: int  # Oracle がスキップする取引 (PnL < 0)
    oracle_skip_rate: float  # スキップ率
    # 実績 PnL 統計 (bps)
    actual_pnl_mean: float
    actual_pnl_sum: float
    # Oracle PnL 統計 (bps) — 正の取引のみ
    oracle_pnl_mean: float
    oracle_pnl_sum: float
    # 30s/60s/120s 別
    pnl_60s_mean: float | None = None
    pnl_120s_mean: float | None = None
    oracle_60s_mean: float | None = None
    oracle_120s_mean: float | None = None
    # JPY 換算 (lot_btc × btc_price × pnl_bps / BPS_FACTOR)
    actual_jpy_per_cycle: float | None = None
    oracle_jpy_per_cycle: float | None = None
    monthly_actual_jpy: float | None = None  # 月間推定 (21,600 cycles)
    monthly_oracle_jpy: float | None = None


@dataclass
class _OracleAggregate:
    """Oracle 指標の元となる生集計."""

    n_total: int = 0
    n_negative: int = 0
    pnl_30s_all: PnlAccumulator = field(default_factory=PnlAccumulator)
    pnl_30s_nonnegative: PnlAccumulator = field(default_factory=PnlAccumulator)
    pnl_60s_all: PnlAccumulator = field(default_factory=PnlAccumulator)
    pnl_120s_all: PnlAccumulator = field(default_factory=PnlAccumulator)
    pnl_60s_nonnegative: PnlAccumulator = field(default_factory=PnlAccumulator)
    pnl_120s_nonnegative: PnlAccumulator = field(default_factory=PnlAccumulator)


_BPS_FACTOR = 10_000
# 120s サイクル → 月 21,600 cycles (30日)
_MONTHLY_CYCLES = 30 * 24 * 3600 / 120


def _add_record_to_oracle_aggregate(
    agg: _OracleAggregate,
    *,
    pnl_30s: float,
    pnl_60s: float | None,
    pnl_120s: float | None,
) -> None:
    """正規化済み数値を Oracle 集計へ反映する."""
    agg.n_total += 1
    agg.pnl_30s_all.add(pnl_30s)
    if pnl_30s >= 0:
        agg.pnl_30s_nonnegative.add(pnl_30s)
    else:
        agg.n_negative += 1

    if pnl_60s is not None:
        agg.pnl_60s_all.add(pnl_60s)
        if pnl_60s >= 0:
            agg.pnl_60s_nonnegative.add(pnl_60s)

    if pnl_120s is not None:
        agg.pnl_120s_all.add(pnl_120s)
        if pnl_120s >= 0:
            agg.pnl_120s_nonnegative.add(pnl_120s)


def _group_oracle_aggregates(
    records: list[FillRecord],
) -> tuple[_OracleAggregate, dict[str, _OracleAggregate], dict[str, _OracleAggregate]]:
    """全体 / side / regime の Oracle 集計を 1 パスで構築する."""
    all_agg = _OracleAggregate()
    side_aggs: dict[str, _OracleAggregate] = {
        "buy": _OracleAggregate(),
        "sell": _OracleAggregate(),
    }
    regime_aggs: dict[str, _OracleAggregate] = defaultdict(_OracleAggregate)

    for record in records:
        if not record.filled or record.post_fill_30s_pnl is None:
            continue
        pnl_30s = safe_to_finite(record.post_fill_30s_pnl)
        if pnl_30s is None:
            continue
        pnl_60s = safe_to_finite(record.post_fill_60s_pnl)
        pnl_120s = safe_to_finite(record.post_fill_120s_pnl)
        _add_record_to_oracle_aggregate(
            all_agg,
            pnl_30s=pnl_30s,
            pnl_60s=pnl_60s,
            pnl_120s=pnl_120s,
        )
        if record.side in side_aggs:
            _add_record_to_oracle_aggregate(
                side_aggs[record.side],
                pnl_30s=pnl_30s,
                pnl_60s=pnl_60s,
                pnl_120s=pnl_120s,
            )
        regime_key = record.regime or "none"
        _add_record_to_oracle_aggregate(
            regime_aggs[regime_key],
            pnl_30s=pnl_30s,
            pnl_60s=pnl_60s,
            pnl_120s=pnl_120s,
        )

    return all_agg, side_aggs, regime_aggs


def _aggregate_oracle(records: list[FillRecord]) -> _OracleAggregate:
    """filled + 30s PnL 有効レコードを単一パスで集計."""
    agg = _OracleAggregate()

    for record in records:
        if not record.filled or record.post_fill_30s_pnl is None:
            continue
        pnl_30s = safe_to_finite(record.post_fill_30s_pnl)
        if pnl_30s is None:
            continue
        _add_record_to_oracle_aggregate(
            agg,
            pnl_30s=pnl_30s,
            pnl_60s=safe_to_finite(record.post_fill_60s_pnl),
            pnl_120s=safe_to_finite(record.post_fill_120s_pnl),
        )

    return agg


def _metrics_from_aggregate(
    agg: _OracleAggregate,
    *,
    label: str,
    lot_btc: float,
    btc_price_jpy: float,
) -> OracleMetrics:
    """集計値から OracleMetrics を構築."""
    if agg.n_total == 0:
        return OracleMetrics(
            label=label,
            n_total=0,
            n_positive=0,
            n_negative=0,
            oracle_skip_rate=0.0,
            actual_pnl_mean=0.0,
            actual_pnl_sum=0.0,
            oracle_pnl_mean=0.0,
            oracle_pnl_sum=0.0,
        )

    n_positive = agg.pnl_30s_nonnegative.count
    oracle_exec_rate = n_positive / agg.n_total
    jpy_factor = lot_btc * btc_price_jpy / _BPS_FACTOR
    actual_jpy = agg.pnl_30s_all.mean_bps * jpy_factor
    # Oracle は正の取引のみ実行するため、cycle あたり期待値 = oracle_mean × 実行率
    oracle_jpy = agg.pnl_30s_nonnegative.mean_bps * jpy_factor * oracle_exec_rate

    return OracleMetrics(
        label=label,
        n_total=agg.n_total,
        n_positive=n_positive,
        n_negative=agg.n_negative,
        oracle_skip_rate=agg.n_negative / agg.n_total,
        actual_pnl_mean=round(agg.pnl_30s_all.mean_bps, 4),
        actual_pnl_sum=round(agg.pnl_30s_all.total_bps, 2),
        oracle_pnl_mean=round(agg.pnl_30s_nonnegative.mean_bps, 4),
        oracle_pnl_sum=round(agg.pnl_30s_nonnegative.total_bps, 2),
        pnl_60s_mean=round(agg.pnl_60s_all.mean_bps, 4) if agg.pnl_60s_all.count else None,
        pnl_120s_mean=round(agg.pnl_120s_all.mean_bps, 4) if agg.pnl_120s_all.count else None,
        oracle_60s_mean=(
            round(agg.pnl_60s_nonnegative.mean_bps, 4)
            if agg.pnl_60s_nonnegative.count
            else None
        ),
        oracle_120s_mean=(
            round(agg.pnl_120s_nonnegative.mean_bps, 4)
            if agg.pnl_120s_nonnegative.count
            else None
        ),
        actual_jpy_per_cycle=round(actual_jpy, 4),
        oracle_jpy_per_cycle=round(oracle_jpy, 4),
        monthly_actual_jpy=round(actual_jpy * _MONTHLY_CYCLES, 0),
        monthly_oracle_jpy=round(oracle_jpy * _MONTHLY_CYCLES, 0),
    )


def compute_oracle_metrics(
    records: list[FillRecord],
    label: str = "all",
    lot_btc: float = 0.001,
    btc_price_jpy: float = 15_000_000,
) -> OracleMetrics:
    """Oracle PnL 統計を算出.

    Args:
        records: filled かつ PnL 計測済みの FillRecord リスト.
        label: 統計ラベル.
        lot_btc: JPY 換算に使うロットサイズ.
        btc_price_jpy: JPY 換算に使う BTC 価格.
    """
    agg = _aggregate_oracle(records)
    return _metrics_from_aggregate(
        agg,
        label=label,
        lot_btc=lot_btc,
        btc_price_jpy=btc_price_jpy,
    )


def _format_metrics(m: OracleMetrics) -> str:
    """単一セクションのフォーマット."""
    lines = [
        f"\n  [{m.label}] n={m.n_total}",
        f"    実績   PnL30s:  mean={m.actual_pnl_mean:+.4f} bps, sum={m.actual_pnl_sum:+.2f} bps",
        f"    Oracle PnL30s:  mean={m.oracle_pnl_mean:+.4f} bps, "
        f"exec={m.n_positive}/{m.n_total} ({1.0 - m.oracle_skip_rate:.1%}), "
        f"skip={m.oracle_skip_rate:.1%}",
    ]
    if m.pnl_60s_mean is not None:
        lines.append(f"    実績   PnL60s:  mean={m.pnl_60s_mean:+.4f} bps")
    if m.pnl_120s_mean is not None:
        lines.append(f"    実績   PnL120s: mean={m.pnl_120s_mean:+.4f} bps")
    if m.oracle_60s_mean is not None:
        lines.append(f"    Oracle PnL60s:  mean={m.oracle_60s_mean:+.4f} bps (positive only)")
    if m.oracle_120s_mean is not None:
        lines.append(f"    Oracle PnL120s: mean={m.oracle_120s_mean:+.4f} bps (positive only)")
    if m.actual_jpy_per_cycle is not None:
        lines.append(
            f"    JPY 換算 (per cycle): actual={m.actual_jpy_per_cycle:+.4f}, "
            f"oracle={m.oracle_jpy_per_cycle:+.4f}"
        )
    if m.monthly_actual_jpy is not None:
        lines.append(
            f"    月間推定 JPY:          actual={m.monthly_actual_jpy:+,.0f}, "
            f"oracle={m.monthly_oracle_jpy:+,.0f}"
        )
    return "\n".join(lines)


def run_oracle_baseline(
    results_dir: str = "results/v460/fill_test",
    output_path: str | None = None,
    lot_btc: float = 0.001,
    btc_price_jpy: float = 15_000_000,
) -> dict:
    """Oracle PnL 基準線レポートを生成・出力.

    Returns:
        JSON-serializable レポートdict.
    """
    clean, _quarantine = partition_clean_records(
        iter_fill_records_glob(results_dir),
    )
    all_agg, side_aggs, regime_aggs = _group_oracle_aggregates(clean)
    del clean

    if all_agg.n_total == 0:
        print("[oracle] filled かつ PnL 計測済みのレコードがありません。")
        return {"error": "no filled records"}

    # --- 全体 ---
    all_metrics = _metrics_from_aggregate(
        all_agg,
        label="all",
        lot_btc=lot_btc,
        btc_price_jpy=btc_price_jpy,
    )

    # --- side 別 ---
    buy_metrics = _metrics_from_aggregate(
        side_aggs["buy"],
        label="buy",
        lot_btc=lot_btc,
        btc_price_jpy=btc_price_jpy,
    )
    sell_metrics = _metrics_from_aggregate(
        side_aggs["sell"],
        label="sell",
        lot_btc=lot_btc,
        btc_price_jpy=btc_price_jpy,
    )

    # --- レジーム別 ---
    regime_metrics: list[OracleMetrics] = []
    for regime_name in sorted(regime_aggs):
        agg = regime_aggs[regime_name]
        if agg.n_total > 0:
            regime_metrics.append(
                _metrics_from_aggregate(
                    agg,
                    label=f"regime:{regime_name}",
                    lot_btc=lot_btc,
                    btc_price_jpy=btc_price_jpy,
                )
            )

    # --- Lot size 別月間推定 ---
    lot_scenarios: list[dict] = []
    for lot in [0.001, 0.005, 0.01, 0.05, 0.1]:
        m = _metrics_from_aggregate(
            all_agg,
            label=f"lot={lot}",
            lot_btc=lot,
            btc_price_jpy=btc_price_jpy,
        )
        lot_scenarios.append({
            "lot_btc": lot,
            "monthly_actual_jpy": m.monthly_actual_jpy,
            "monthly_oracle_jpy": m.monthly_oracle_jpy,
            "improvement_gap_jpy": (
                (m.monthly_oracle_jpy or 0) - (m.monthly_actual_jpy or 0)
            ),
        })

    # --- ph3 判定 ---
    ph3_oracle_positive = all_metrics.oracle_pnl_mean > 0
    ph3_check = {
        "oracle_pnl30_positive": ph3_oracle_positive,
        "oracle_pnl_mean_bps": all_metrics.oracle_pnl_mean,
        "actual_pnl_mean_bps": all_metrics.actual_pnl_mean,
        "improvement_room_bps": round(
            all_metrics.oracle_pnl_mean * (1.0 - all_metrics.oracle_skip_rate)
            - all_metrics.actual_pnl_mean, 4,
        ),
        "verdict": "PASS — Oracle PnL is positive, ph3 viable"
        if ph3_oracle_positive
        else "FAIL — Oracle PnL is not positive, market structure may not support profitability",
    }

    # --- 表示 ---
    print("=" * 72)
    print("  131# D2: Oracle PnL Baseline Report")
    print(f"  BTC price assumption: ¥{btc_price_jpy:,.0f}")
    print(f"  Default lot: {lot_btc} BTC")
    print("=" * 72)

    print(_format_metrics(all_metrics))
    print(_format_metrics(buy_metrics))
    print(_format_metrics(sell_metrics))

    for m in regime_metrics:
        print(_format_metrics(m))

    print("\n  --- Lot Size Scenarios (月間) ---")
    for s in lot_scenarios:
        print(
            f"    {s['lot_btc']:.3f} BTC: "
            f"actual={s['monthly_actual_jpy']:+,.0f} JPY, "
            f"oracle={s['monthly_oracle_jpy']:+,.0f} JPY, "
            f"gap={s['improvement_gap_jpy']:+,.0f} JPY"
        )

    print(f"\n  --- ph3 Pre-check ---")
    print(f"    Oracle PnL30 > 0: {'YES' if ph3_check['oracle_pnl30_positive'] else 'NO'}")
    print(f"    Verdict: {ph3_check['verdict']}")
    print("=" * 72)

    # --- レポート構築 ---
    report = {
        "all": shallow_asdict(all_metrics),
        "buy": shallow_asdict(buy_metrics),
        "sell": shallow_asdict(sell_metrics),
        "by_regime": [shallow_asdict(m) for m in regime_metrics],
        "lot_scenarios": lot_scenarios,
        "ph3_check": ph3_check,
        "params": {
            "lot_btc": lot_btc,
            "btc_price_jpy": btc_price_jpy,
            "n_records_total": all_agg.n_total,
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json_output(report, out)
        print(f"\n  Report saved to: {out}")

    return report


def main(argv: Sequence[str] | None = None) -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="131# D2: Oracle PnL Baseline")
    add_results_dir_arg(parser, help_text="Fill records directory")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--lot-btc",
        type=float,
        default=0.001,
        help="Lot size BTC for JPY conversion (default: 0.001)",
    )
    parser.add_argument(
        "--btc-price",
        type=float,
        default=15_000_000,
        help="BTC/JPY price assumption (default: 15,000,000)",
    )
    args = parser.parse_args(argv)
    run_oracle_baseline(
        results_dir=args.results_dir,
        output_path=args.output,
        lot_btc=args.lot_btc,
        btc_price_jpy=args.btc_price,
    )


if __name__ == "__main__":
    main()
