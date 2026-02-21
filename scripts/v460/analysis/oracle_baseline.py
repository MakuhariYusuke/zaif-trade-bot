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
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    filter_clean_records,
    load_fill_records_glob,
)


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
    pnl_60s_mean: Optional[float] = None
    pnl_120s_mean: Optional[float] = None
    oracle_60s_mean: Optional[float] = None
    oracle_120s_mean: Optional[float] = None
    # JPY 換算 (lot_btc × btc_price × pnl_bps / BPS_FACTOR)
    actual_jpy_per_cycle: Optional[float] = None
    oracle_jpy_per_cycle: Optional[float] = None
    monthly_actual_jpy: Optional[float] = None  # 月間推定 (21,600 cycles)
    monthly_oracle_jpy: Optional[float] = None


_BPS_FACTOR = 10_000
# 120s サイクル → 月 21,600 cycles (30日)
_MONTHLY_CYCLES = 30 * 24 * 3600 / 120


def _safe_mean(values: list[float]) -> float:
    """空リストでも安全な平均."""
    return sum(values) / len(values) if values else 0.0


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
    # 30s PnL がある filled レコードのみ
    valid = [
        r for r in records
        if r.filled and r.post_fill_30s_pnl is not None
    ]
    if not valid:
        return OracleMetrics(
            label=label, n_total=0, n_positive=0, n_negative=0,
            oracle_skip_rate=0.0, actual_pnl_mean=0.0, actual_pnl_sum=0.0,
            oracle_pnl_mean=0.0, oracle_pnl_sum=0.0,
        )

    pnl_30s = [r.post_fill_30s_pnl for r in valid]  # type: ignore[misc]
    positive = [p for p in pnl_30s if p >= 0]
    negative = [p for p in pnl_30s if p < 0]

    # 60s / 120s (存在するレコードのみ)
    pnl_60s = [r.post_fill_60s_pnl for r in valid if r.post_fill_60s_pnl is not None]
    pnl_120s = [r.post_fill_120s_pnl for r in valid if r.post_fill_120s_pnl is not None]
    pos_60s = [p for p in pnl_60s if p >= 0]
    pos_120s = [p for p in pnl_120s if p >= 0]

    actual_mean = _safe_mean(pnl_30s)
    oracle_mean = _safe_mean(positive)

    # JPY 換算: lot × price × bps / BPS_FACTOR
    jpy_factor = lot_btc * btc_price_jpy / _BPS_FACTOR
    actual_jpy = actual_mean * jpy_factor
    oracle_jpy = oracle_mean * jpy_factor * (len(positive) / len(valid)) if valid else 0.0
    # Oracle は正の取引のみ実行するため、cycle あたり期待値 = oracle_mean × 実行率
    oracle_exec_rate = len(positive) / len(valid) if valid else 0.0

    return OracleMetrics(
        label=label,
        n_total=len(valid),
        n_positive=len(positive),
        n_negative=len(negative),
        oracle_skip_rate=len(negative) / len(valid) if valid else 0.0,
        actual_pnl_mean=round(actual_mean, 4),
        actual_pnl_sum=round(sum(pnl_30s), 2),
        oracle_pnl_mean=round(oracle_mean, 4),
        oracle_pnl_sum=round(sum(positive), 2),
        pnl_60s_mean=round(_safe_mean(pnl_60s), 4) if pnl_60s else None,
        pnl_120s_mean=round(_safe_mean(pnl_120s), 4) if pnl_120s else None,
        oracle_60s_mean=round(_safe_mean(pos_60s), 4) if pos_60s else None,
        oracle_120s_mean=round(_safe_mean(pos_120s), 4) if pos_120s else None,
        actual_jpy_per_cycle=round(actual_jpy, 4),
        oracle_jpy_per_cycle=round(oracle_jpy, 4),
        monthly_actual_jpy=round(actual_jpy * _MONTHLY_CYCLES, 0),
        monthly_oracle_jpy=round(oracle_jpy * _MONTHLY_CYCLES, 0),
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
    all_records = load_fill_records_glob(results_dir)
    clean, _quarantine = filter_clean_records(all_records)
    del all_records
    filled = [r for r in clean if r.filled and r.post_fill_30s_pnl is not None]
    del clean

    if not filled:
        print("[oracle] filled かつ PnL 計測済みのレコードがありません。")
        return {"error": "no filled records"}

    # --- 全体 ---
    all_metrics = compute_oracle_metrics(filled, "all", lot_btc, btc_price_jpy)

    # --- side 別 ---
    buy = [r for r in filled if r.side == "buy"]
    sell = [r for r in filled if r.side == "sell"]
    buy_metrics = compute_oracle_metrics(buy, "buy", lot_btc, btc_price_jpy)
    sell_metrics = compute_oracle_metrics(sell, "sell", lot_btc, btc_price_jpy)

    # --- レジーム別 ---
    regime_metrics: list[OracleMetrics] = []
    regimes = sorted(set(r.regime or "none" for r in filled))
    for regime in regimes:
        recs = [r for r in filled if (r.regime or "none") == regime]
        if recs:
            regime_metrics.append(
                compute_oracle_metrics(recs, f"regime:{regime}", lot_btc, btc_price_jpy)
            )

    # --- Lot size 別月間推定 ---
    lot_scenarios: list[dict] = []
    for lot in [0.001, 0.005, 0.01, 0.05, 0.1]:
        m = compute_oracle_metrics(filled, f"lot={lot}", lot, btc_price_jpy)
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
        "all": asdict(all_metrics),
        "buy": asdict(buy_metrics),
        "sell": asdict(sell_metrics),
        "by_regime": [asdict(m) for m in regime_metrics],
        "lot_scenarios": lot_scenarios,
        "ph3_check": ph3_check,
        "params": {
            "lot_btc": lot_btc,
            "btc_price_jpy": btc_price_jpy,
            "n_records_total": len(filled),
        },
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Report saved to: {out}")

    return report


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="131# D2: Oracle PnL Baseline")
    parser.add_argument(
        "--results-dir",
        default="results/v460/fill_test",
        help="Fill records directory",
    )
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
    args = parser.parse_args()
    run_oracle_baseline(
        results_dir=args.results_dir,
        output_path=args.output,
        lot_btc=args.lot_btc,
        btc_price_jpy=args.btc_price,
    )


if __name__ == "__main__":
    main()
