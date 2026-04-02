#!/usr/bin/env python3
"""
PnL モンテカルロ — fill_test 結果から月次 PnL レンジを推定.

014# T5: 012# §3 #4 実装.

Usage:
  # 基本実行
  python scripts/v460/run_pnl_monte_carlo.py

  # 指定ディレクトリ
  python scripts/v460/run_pnl_monte_carlo.py --input results/v460/fill_test/

  # 感度分析付き
  python scripts/v460/run_pnl_monte_carlo.py --sensitivity

  # JSON出力
  python scripts/v460/run_pnl_monte_carlo.py --output results/v460/monte_carlo.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.risk.pnl_monte_carlo import (
    MonteCarloConfig,
    PnLMonteCarloSimulator,
)
from ztb.metrics.fill_metric_results import compute_fill_metrics
from ztb.metrics.fill_record_integrity import filter_clean_records
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="PnL Monte Carlo (014# T5)")
    parser.add_argument(
        "--input", type=str,
        default=str(_PROJECT_ROOT / "results" / "v460" / "fill_test"),
        help="Path to fill_records JSONL file(s)",
    )
    parser.add_argument(
        "--n-simulations", type=int, default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000)",
    )
    parser.add_argument(
        "--btc-price", type=float, default=None,
        help="BTC/JPY price override (default: auto from fill records)",
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis across fill_rate / pnl_adjustment",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Load records
    records = PnLMonteCarloSimulator.load_fill_records(args.input)
    if not records:
        logger.error("No fill records found")
        sys.exit(1)

    # Auto-detect BTC price from fill records
    btc_price = args.btc_price
    if btc_price is None:
        filled_prices = [r.order_price for r in records if r.order_price]
        btc_price = float(sum(filled_prices) / len(filled_prices)) if filled_prices else 10_300_000.0
        logger.info(f"Auto-detected BTC price: {btc_price:,.0f} JPY")

    config = MonteCarloConfig(
        n_simulations=args.n_simulations,
        btc_price_jpy=btc_price,
    )

    sim = PnLMonteCarloSimulator(records, config)
    result = sim.run()
    sensitivity_result = sim.sensitivity_analysis() if args.sensitivity else None

    # Print report
    sim.print_report(result)

    # Sensitivity analysis
    if sensitivity_result is not None:
        print("\n" + "=" * 60)
        print("  Sensitivity Analysis")
        print("=" * 60)
        print(f"\n{'fill_rate':>10} {'pnl_adj':>8} {'E[PnL]':>12} {'VaR95':>12} {'P(loss)':>8}")
        print("-" * 54)
        for s in sensitivity_result:
            print(
                f"{s['fill_rate']:>10.0%} "
                f"{s['pnl_adj_bps']:>+7.1f} "
                f"{s['mean_jpy']:>+11,.0f} "
                f"{s['var_95_jpy']:>+11,.0f} "
                f"{s['prob_loss']:>7.1%}"
            )

    # JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_dict = result.to_dict()
        if sensitivity_result is not None:
            result_dict["sensitivity"] = sensitivity_result

        # 169# B0: 3-series fill rate を構造化出力 (R2 分母混在解消)
        all_metrics = compute_fill_metrics(records)
        clean_records, _quarantine = filter_clean_records(records)
        clean_metrics = compute_fill_metrics(clean_records) if clean_records else all_metrics
        n_clean = len(clean_records) if clean_records else 0
        result_dict["three_series"] = {
            "raw": {
                "fill_rate": round(all_metrics.overall_fill_rate, 6),
                "n_total": all_metrics.total_orders,
                "n_filled": all_metrics.filled_orders,
            },
            "clean": {
                "fill_rate": round(
                    clean_metrics.filled_orders / n_clean if n_clean else 0.0, 6
                ),
                "n_total": n_clean,
                "n_filled": clean_metrics.filled_orders,
            },
            "attempted": {
                "fill_rate": round(clean_metrics.attempted_fill_rate, 6),
                "n_total": clean_metrics.attempted_orders,
                "n_filled": clean_metrics.filled_orders,
                "skip_gate_count": clean_metrics.skip_gate_count,
            },
            "gate_basis": "clean",
            "mc_basis": "raw",  # MC はフィルタ前の全データで実行
        }

        write_json(output_path, result_dict, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
