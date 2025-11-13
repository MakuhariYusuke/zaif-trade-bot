#!/usr/bin/env python3
"""
Phase 3 Backtest Validation Script
リスク管理統合の効果検証
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.signal.phase3_backtest_comparison import IntegratedBacktestRunner
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_sample_market_data(n_points: int = 1000) -> pd.DataFrame:
    """サンプル市場データ生成"""
    np.random.seed(42)  # 再現性のために固定

    # 基本価格トレンド
    base_price = 100.0
    prices = [base_price]

    for i in range(n_points - 1):
        # ランダムウォーク + トレンド
        trend = 0.0001  # 軽微な上昇トレンド
        volatility = 0.02  # 2%ボラティリティ
        change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # RSI計算（簡易版）
    def calculate_rsi(prices, period=14):
        rsi_values = []
        for i in range(len(prices)):
            if i < period:
                rsi_values.append(50.0)
            else:
                gains = []
                losses = []
                for j in range(period):
                    change = prices[i-j] - prices[i-j-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(-change)

                avg_gain = sum(gains) / period if gains else 0
                avg_loss = sum(losses) / period if losses else 0

                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        return rsi_values

    rsi_values = calculate_rsi(prices)

    # DataFrame作成
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [1000 + np.random.normal(0, 100) for _ in range(n_points)],
        'rsi': rsi_values,
        'volatility': [0.02] * n_points  # 固定ボラティリティ
    }

    df = pd.DataFrame(data)
    df['returns'] = df['close'].pct_change().fillna(0)
    df.set_index('timestamp', inplace=True)

    return df


def run_phase3_validation():
    """Phase 3検証実行"""
    logger.info("Starting Phase 3 backtest validation")

    # サンプルデータ生成
    market_data = create_sample_market_data(1000)
    logger.info(f"Generated {len(market_data)} data points")

    # Phase 3バックテスト実行
    runner = IntegratedBacktestRunner()
    results = runner.run_enhanced_backtest_aggressive(market_data)

    # 結果表示
    print("\n" + "="*50)
    print("PHASE 3 BACKTEST RESULTS")
    print("="*50)

    print("Performance Metrics:")
    print(f"  Total Trades: {len(results['trades'])}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")

    print("\nRisk Metrics:")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {results['win_rate']:.2%}")

    validation = results['validation']
    print("\nStatistical Validation:")
    print(f"  T-Statistic: {validation['t_statistic']:.2f}")
    print(f"  P-Value: {validation['p_value']:.4f}")
    print(f"  Significant: {validation['significant']}")
    print(f"  Mean Return: {float(validation['mean_return']):.2%}")
    print(f"  Volatility: {float(validation['volatility']):.2%}")

    # ドローダウン削減の検証
    print("\nPhase 3 Risk Reduction Validation:")
    print("  Target: Max Drawdown < 10%")
    max_dd = validation['max_drawdown']
    if max_dd < 0.10:
        print(f"  ✅ ACHIEVED: Max Drawdown = {max_dd:.2%} (< 10%)")
    else:
        print(f"  ❌ NOT ACHIEVED: Max Drawdown = {max_dd:.2%} (>= 10%)")

    # シグナル頻度の検証
    trades_count = len(results['trades'])
    if 3 <= trades_count <= 64:  # Phase 3目標範囲
        print(f"  ✅ ACHIEVED: Trades = {trades_count} (within 3-64 range)")
    else:
        print(f"  ❌ NOT ACHIEVED: Trades = {trades_count} (outside 3-64 range)")

    return results


if __name__ == "__main__":
    try:
        results = run_phase3_validation()
        logger.info("Phase 3 validation completed successfully")
    except Exception as e:
        logger.error(f"Phase 3 validation failed: {e}")
        sys.exit(1)