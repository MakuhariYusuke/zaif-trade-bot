#!/usr/bin/env python3
"""
SAC v424 Backtest Script
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.evaluation.evaluator.evaluator import TradingEvaluator

def run_v424_backtest():
    """Run backtest for SAC v424 model."""

    print("🚀 Starting SAC v424 Backtest")
    print("=" * 50)

    evaluator = TradingEvaluator(
        model_path='models/sac_v424_cost_aware.zip',
        data_path='data/btc_jpy_real_dataset.csv'
    )

    try:
        results = evaluator.evaluate_model()

        print("✅ v424バックテスト完了")
        print(f"結果保存先: results/backtest_v424_cost_aware.json")

        # 基本結果を表示
        if results:
            print("\n📊 バックテスト結果サマリー:")
            print(f"総リターン: {results.get('total_return', 0):.2f}%")
            print(f"年間リターン: {results.get('annual_return', 0):.2f}%")
            print(f"勝率: {results.get('win_rate', 0):.1f}%")
            print(f"総取引数: {results.get('total_trades', 0)}")
            print(f"シャープレシオ: {results.get('sharpe_ratio', 0):.3f}")
            print(f"最大ドローダウン: {results.get('max_drawdown', 0):.2f}%")
            print(f"ボラティリティ: {results.get('volatility', 0):.2f}%")

    except Exception as e:
        print(f"❌ バックテスト失敗: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_v424_backtest()