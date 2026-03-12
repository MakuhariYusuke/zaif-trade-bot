#!/usr/bin/env python3
"""
Backtest SAC v399 model with balanced reward function
- Evaluate win rate and action balance
- Compare with previous models
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.evaluation.evaluate import TradingEvaluator


def main():
    print("🎯 Backtesting SAC v399 with Balanced Reward Function 🎯")
    print("=" * 80)

    model_name = "sac_v399_balanced_reward"

    try:
        # Initialize evaluator
        model_path = f"checkpoints/sac_session/{model_name}_final.zip"
        data_path = "btc_jpy_real_dataset.csv"
        evaluator = TradingEvaluator(model_path=model_path, data_path=data_path)

        # Run evaluation
        print(f"📊 Running evaluation for model: {model_name}")
        result = evaluator.evaluate_model()

        if result:
            print("✅ Evaluation completed successfully!")
            print(f"   Win Rate: {result.get('win_rate', 'N/A')}%")
            print(f"   Total Trades: {result.get('total_trades', 'N/A')}")
            print(f"   Total PnL: {result.get('total_pnl', 'N/A')}")
            print(f"   Sharpe Ratio: {result.get('sharpe_ratio', 'N/A')}")

            # Note: Action balance metrics may not be available in this evaluator
            print("   Note: Action balance analysis not available in current evaluator")
        else:
            print("❌ Evaluation failed")
            print("   Error: No result returned")

    except Exception as e:
        print(f"❌ Error during backtest: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
