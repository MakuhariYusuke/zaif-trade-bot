#!/usr/bin/env python3
"""
v436 Elliott Wave SAC バックテスト詳細分析
取引間隔、p平均法等の統計分析、シャープレシオ等を分析
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))




def analyze_trade_statistics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """取引統計の詳細分析"""
    if not trades:
        return {}

    # 取引間隔分析
    steps = [trade["step"] for trade in trades]
    intervals = np.diff(sorted(steps))
    avg_interval = np.mean(intervals)
    median_interval = np.median(intervals)
    min_interval = np.min(intervals)
    max_interval = np.max(intervals)

    # PnL分析
    pnls = [trade.get("pnl", 0) for trade in trades]
    positive_pnls = [p for p in pnls if p > 0]
    negative_pnls = [p for p in pnls if p < 0]

    # p平均法分析（パーセンタイル分析）
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pnl_percentiles = np.percentile(pnls, percentiles)

    # 行動分布分析
    actions = [trade.get("action", 0) for trade in trades]
    action_ranges = {
        "strong_sell": len([a for a in actions if a <= -0.8]),
        "sell": len([a for a in actions if -0.8 < a <= -0.4]),
        "hold": len([a for a in actions if -0.4 < a < 0.4]),
        "buy": len([a for a in actions if 0.4 <= a < 0.8]),
        "strong_buy": len([a for a in actions if a >= 0.8]),
    }

    return {
        "trade_intervals": {
            "average": float(avg_interval),
            "median": float(median_interval),
            "min": int(min_interval),
            "max": int(max_interval),
        },
        "pnl_analysis": {
            "total_trades": len(trades),
            "profitable_trades": len(positive_pnls),
            "losing_trades": len(negative_pnls),
            "win_rate": len(positive_pnls) / len(trades),
            "avg_win": float(np.mean(positive_pnls)) if positive_pnls else 0,
            "avg_loss": float(np.mean(negative_pnls)) if negative_pnls else 0,
            "profit_factor": abs(sum(positive_pnls) / sum(negative_pnls))
            if negative_pnls
            else float("inf"),
            "percentiles": {
                f"p{percentiles[i]}": float(pnl_percentiles[i])
                for i in range(len(percentiles))
            },
        },
        "action_distribution": action_ranges,
    }


def calculate_portfolio_returns(
    trades: List[Dict[str, Any]], initial_balance: float = 100000
) -> Dict[str, Any]:
    """ポートフォリオリターンの計算"""
    if not trades:
        return {}

    # 時系列でのPnL累積
    cumulative_pnl = 0
    portfolio_values = [initial_balance]

    for trade in sorted(trades, key=lambda x: x["step"]):
        pnl = trade.get("pnl", 0)
        cumulative_pnl += pnl
        portfolio_value = initial_balance + cumulative_pnl
        portfolio_values.append(portfolio_value)

    # リターン計算
    final_value = portfolio_values[-1]
    total_return_pct = (final_value - initial_balance) / initial_balance * 100
    total_return_yen = final_value - initial_balance

    # 日次リターン（ステップベース）
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(returns) > 0:
        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        )  # 年率化
        volatility = np.std(returns) * np.sqrt(252)
        max_drawdown = (
            np.min(
                np.minimum.accumulate(portfolio_values)
                / np.maximum.accumulate(portfolio_values)
                - 1
            )
            * 100
        )
    else:
        sharpe_ratio = 0
        volatility = 0
        max_drawdown = 0

    return {
        "initial_balance": initial_balance,
        "final_value": final_value,
        "total_return_yen": total_return_yen,
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "portfolio_values": portfolio_values,
    }


def analyze_observation_space_vs_indicators():
    """観測空間とトレーディング指標の次元分析"""
    print("🔍 観測空間 vs トレーディング指標分析")
    print("=" * 50)

    # 現在の観測空間（5次元）
    observation_features = ["volume", "rsi_14", "macd", "macd_hist", "bb_position"]
    print(f"📊 観測空間次元: {len(observation_features)}")
    print(f"📈 観測特徴量: {observation_features}")

    # データセットの全特徴量
    data_path = "data/btc_jpy_featured_dataset.csv"
    df = pd.read_csv(data_path)
    all_features = [col for col in df.columns if col not in ["timestamp"]]
    print(f"📊 データセット全特徴量数: {len(all_features)}")
    print(f"📈 全特徴量: {all_features[:10]}...")  # 最初の10個のみ表示

    # トレーディング指標の分類
    technical_indicators = []
    price_data = []
    volume_data = []

    for feature in all_features:
        if any(
            indicator in feature.lower()
            for indicator in [
                "rsi",
                "macd",
                "bb_",
                "stoch",
                "williams",
                "cci",
                "mfi",
                "roc",
                "mom",
                "atr",
            ]
        ):
            technical_indicators.append(feature)
        elif feature in ["open", "high", "low", "close", "returns", "log_returns"]:
            price_data.append(feature)
        elif "volume" in feature.lower():
            volume_data.append(feature)

    print(f"💰 価格データ: {len(price_data)} - {price_data}")
    print(
        f"📊 テクニカル指標: {len(technical_indicators)} - {technical_indicators[:5]}..."
    )
    print(f"📦 出来高データ: {len(volume_data)} - {volume_data}")

    print("\n✅ 観測空間はテクニカル指標のサブセット（5/21）を使用")
    print("✅ トレーディング指標は別途計算・分析可能")


def main():
    """メイン分析関数"""
    results_path = "backtest_experiments/v436.1/backtest_v435_simple_results.json"

    print("🚀 v436 Elliott Wave SAC バックテスト詳細分析")
    print("=" * 60)

    # 結果読み込み
    results = load_backtest_results(results_path)
    trades = results.get("all_trades", results.get("sample_trades", []))

    print(f"📊 総トレード数: {results['trade_count']:,}")
    print(f"🎯 勝率: {results['win_rate']:.1%}")
    print(f"📈 シャープレシオ: {results['sharpe_ratio']:.3f}")
    print(
        f"💰 平均トレードPnL: {results.get('performance_metrics', {}).get('avg_trade_pnl', 0):.4f}"
    )
    print()

    # 観測空間 vs 指標分析
    analyze_observation_space_vs_indicators()
    print()

    # 取引統計分析
    if trades:
        trade_stats = analyze_trade_statistics(trades)
        print("📊 取引統計詳細分析")
        print("-" * 30)

        intervals = trade_stats["trade_intervals"]
        print(f"⏱️  平均取引間隔: {intervals['average']:.1f} ステップ")
        print(f"📅 中央値取引間隔: {intervals['median']:.1f} ステップ")
        print(f"⚡ 最小取引間隔: {intervals['min']} ステップ")
        print(f"🐌 最大取引間隔: {intervals['max']} ステップ")
        print()

        pnl = trade_stats["pnl_analysis"]
        print(f"💰 総取引数: {pnl['total_trades']:,}")
        print(f"✅ 勝ち取引: {pnl['profitable_trades']:,} ({pnl['win_rate']:.1%})")
        print(f"❌ 負け取引: {pnl['losing_trades']:,}")
        print(f"📈 平均勝ち: ¥{pnl['avg_win']:.2f}")
        print(f"📉 平均負け: ¥{pnl['avg_loss']:.2f}")
        print(f"⚖️  プロフィットファクター: {pnl['profit_factor']:.2f}")
        print()

        print("📊 PnL パーセンタイル分析")
        for p, value in pnl["percentiles"].items():
            print(f"  {p}: ¥{value:.2f}")
        print()

        actions = trade_stats["action_distribution"]
        print("🎯 行動分布")
        print(f"  強SELL: {actions['strong_sell']:,}")
        print(f"  SELL: {actions['sell']:,}")
        print(f"  HOLD: {actions['hold']:,}")
        print(f"  BUY: {actions['buy']:,}")
        print(f"  強BUY: {actions['strong_buy']:,}")
        print()

    # ポートフォリオ分析
    portfolio_analysis = calculate_portfolio_returns(trades)
    if portfolio_analysis:
        print("💼 ポートフォリオ分析")
        print("-" * 30)
        print(f"🏦 初期残高: ¥{portfolio_analysis['initial_balance']:,}")
        print(f"💰 最終価値: ¥{portfolio_analysis['final_value']:.2f}")
        print(
            f"📈 総リターン: ¥{portfolio_analysis['total_return_yen']:.2f} ({portfolio_analysis['total_return_pct']:.2f}%)"
        )
        print(f"📊 シャープレシオ: {portfolio_analysis['sharpe_ratio']:.3f}")
        print(f"⚠️  最大ドローダウン: {portfolio_analysis['max_drawdown']:.2f}%")
        print(f"📉 ボラティリティ: {portfolio_analysis['volatility']:.2f}")

        print("\n🎯 結論:")
        print(
            f"   • 1トレードあたり平均: ¥{portfolio_analysis['total_return_yen']/results['trade_count']:.2f}"
        )
        print(
            f"   • 年率リターン（概算）: {portfolio_analysis['total_return_pct']:.2f}%"
        )
        print(f"   • リスク調整後リターン: {portfolio_analysis['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()
