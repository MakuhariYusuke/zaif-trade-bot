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

    # Print report
    sim.print_report(result)

    # Sensitivity analysis
    if args.sensitivity:
        print("\n" + "=" * 60)
        print("  Sensitivity Analysis")
        print("=" * 60)
        sens = sim.sensitivity_analysis()
        print(f"\n{'fill_rate':>10} {'pnl_adj':>8} {'E[PnL]':>12} {'VaR95':>12} {'P(loss)':>8}")
        print("-" * 54)
        for s in sens:
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
        if args.sensitivity:
            result_dict["sensitivity"] = sim.sensitivity_analysis()
        write_json(output_path, result_dict, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
