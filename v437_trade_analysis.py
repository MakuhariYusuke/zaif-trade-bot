#!/usr/bin/env python3
"""
SAC v437 Trade Analysis - Why Each Trade Won or Lost

Analyzes individual trades to understand which trading indicators influenced decisions
and why trades resulted in wins or losses.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_trade_analysis_data():
    """Load trade history and feature data for analysis."""
    results_dir = Path("backtest_experiments/v437.1")
    if not results_dir.exists():
        print("❌ Backtest results directory not found")
        return None, None

    # Get latest results directory
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("❌ No backtest result directories found")
        return None, None

    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    print(f"📂 Analyzing trades from: {latest_dir.name}")

    try:
        # Load trade history
        trades_df = pd.read_csv(latest_dir / "trades_history.csv")

        # Load portfolio values to get timestamps
        portfolio_df = pd.read_csv(latest_dir / "portfolio_values.csv")

        return trades_df, portfolio_df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def load_feature_data_for_analysis():
    """Load feature data to understand indicator values at trade times."""
    try:
        # Load original data and generate features
        from ztb.features.sac_v427_feature_engineering import create_v437_feature_set

        # Use the same data path as in backtest
        data_path = "data/btc_jpy_real_dataset.csv"
        if not Path(data_path).exists():
            print("❌ Original data file not found")
            return None

        print("🔄 Generating feature data for analysis...")
        features_df = create_v437_feature_set(data_path, feature_set="full")

        return features_df
    except Exception as e:
        print(f"❌ Error loading feature data: {e}")
        return None


def analyze_trade_decision_factors(
    trade_row: pd.Series, features_at_time: pd.Series
) -> Dict[str, any]:
    """
    Analyze what factors influenced a specific trade decision.

    Args:
        trade_row: Row from trades DataFrame
        features_at_time: Feature values at the time of trade

    Returns:
        Dictionary with analysis of decision factors
    """

    analysis = {
        "trade_info": {
            "episode": trade_row.get("episode"),
            "step": trade_row.get("step"),
            "action": trade_row.get("action"),
            "reward": trade_row.get("reward"),
            "portfolio_value": trade_row.get("portfolio_value"),
        },
        "decision_factors": {},
        "key_indicators": {},
        "outcome_analysis": {},
    }

    # Extract action value
    try:
        action_str = str(trade_row.get("action", ""))
        if "[" in action_str and "]" in action_str:
            action_value = float(action_str.strip("[]"))
        else:
            action_value = float(action_str)
    except:
        action_value = 0.0

    analysis["trade_info"]["action_value"] = action_value

    # Determine trade type
    if abs(action_value) < 0.1:
        trade_type = "HOLD/NO_POSITION"
    elif action_value > 0:
        trade_type = "BUY/LONG"
    else:
        trade_type = "SELL/SHORT"

    analysis["trade_info"]["trade_type"] = trade_type

    # Analyze key indicators that likely influenced the decision

    # 1. Price-based indicators
    price_indicators = {}
    if "close" in features_at_time:
        price_indicators["current_price"] = features_at_time["close"]

    # 2. Trend indicators
    trend_indicators = {}
    trend_cols = [
        col for col in features_at_time.index if "sma_" in col or "ema_" in col
    ]
    for col in trend_cols[:5]:  # Limit to first 5
        trend_indicators[col] = features_at_time[col]

    # 3. Momentum indicators
    momentum_indicators = {}
    momentum_cols = [
        col
        for col in features_at_time.index
        if "rsi" in col or "macd" in col or "momentum" in col
    ]
    for col in momentum_cols[:5]:  # Limit to first 5
        momentum_indicators[col] = features_at_time[col]

    # 4. Volatility indicators
    volatility_indicators = {}
    vol_cols = [
        col
        for col in features_at_time.index
        if "volatility" in col or "std" in col or "atr" in col
    ]
    for col in vol_cols[:3]:  # Limit to first 3
        volatility_indicators[col] = features_at_time[col]

    # 5. Volume indicators
    volume_indicators = {}
    volume_cols = [col for col in features_at_time.index if "volume" in col]
    for col in volume_cols[:3]:  # Limit to first 3
        volume_indicators[col] = features_at_time[col]

    analysis["key_indicators"] = {
        "price": price_indicators,
        "trend": trend_indicators,
        "momentum": momentum_indicators,
        "volatility": volatility_indicators,
        "volume": volume_indicators,
    }

    # Analyze decision factors based on indicator values
    decision_factors = []

    # Trend analysis
    if trend_indicators:
        sma_20 = trend_indicators.get("sma_20", features_at_time.get("sma_20"))
        sma_50 = trend_indicators.get("sma_50", features_at_time.get("sma_50"))
        current_price = price_indicators.get(
            "current_price", features_at_time.get("close", 0)
        )

        if sma_20 is not None and sma_50 is not None and current_price > 0:
            if current_price > sma_20 > sma_50:
                decision_factors.append("強気トレンド: 価格 > SMA20 > SMA50")
            elif current_price < sma_20 < sma_50:
                decision_factors.append("弱気トレンド: 価格 < SMA20 < SMA50")
            elif sma_20 > sma_50 and current_price > sma_20:
                decision_factors.append(
                    "上昇トレンド継続: SMA20 > SMA50 且つ 価格 > SMA20"
                )

    # Momentum analysis
    if momentum_indicators:
        rsi_14 = momentum_indicators.get("rsi_14", features_at_time.get("rsi_14"))
        if rsi_14 is not None:
            if rsi_14 > 70:
                decision_factors.append(f"過熱圏 (RSI: {rsi_14:.1f})")
            elif rsi_14 < 30:
                decision_factors.append(f"売られすぎ (RSI: {rsi_14:.1f})")
            elif 40 < rsi_14 < 60:
                decision_factors.append(f"中立的モメンタム (RSI: {rsi_14:.1f})")

        macd = momentum_indicators.get("macd", features_at_time.get("macd"))
        macd_signal = momentum_indicators.get(
            "macd_signal", features_at_time.get("macd_signal")
        )
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                decision_factors.append(
                    f"MACD強気シグナル (MACD: {macd:.4f} > Signal: {macd_signal:.4f})"
                )
            else:
                decision_factors.append(
                    f"MACD弱気シグナル (MACD: {macd:.4f} < Signal: {macd_signal:.4f})"
                )

    # Volatility analysis
    if volatility_indicators:
        volatility_20 = volatility_indicators.get(
            "volatility_20", features_at_time.get("volatility_20")
        )
        if volatility_20 is not None:
            if volatility_20 > 0.05:  # High volatility threshold
                decision_factors.append(
                    f"高ボラティリティ環境 (σ: {volatility_20:.4f})"
                )
            else:
                decision_factors.append(
                    f"低ボラティリティ環境 (σ: {volatility_20:.4f})"
                )

    analysis["decision_factors"] = decision_factors

    # Analyze outcome
    reward = trade_row.get("reward", 0)
    if reward > 0:
        outcome = "WIN"
        outcome_reason = "ポジティブな報酬が得られた"
    elif reward < 0:
        outcome = "LOSS"
        outcome_reason = "ネガティブな報酬となった"
    else:
        outcome = "NEUTRAL"
        outcome_reason = "報酬がゼロ"

    analysis["outcome_analysis"] = {
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "reward_value": reward,
    }

    return analysis


def analyze_sample_trades(
    trades_df: pd.DataFrame, features_df: pd.DataFrame, sample_size: int = 10
):
    """Analyze a sample of trades to understand decision patterns."""

    print(f"🎯 取引分析サンプル ({sample_size}件)")
    print("=" * 80)

    # Get sample trades (mix of wins and losses)
    if len(trades_df) == 0:
        print("❌ 取引データがありません")
        return []

    # Separate wins and losses
    wins = trades_df[trades_df["reward"] > 0]
    losses = trades_df[trades_df["reward"] < 0]

    sample_trades = []

    # Sample from wins
    if len(wins) > 0:
        win_sample = wins.sample(min(sample_size // 2, len(wins)))
        sample_trades.extend(win_sample.to_dict("records"))

    # Sample from losses
    if len(losses) > 0:
        loss_sample = losses.sample(min(sample_size // 2, len(losses)))
        sample_trades.extend(loss_sample.to_dict("records"))

    # If not enough samples, fill with random
    remaining = sample_size - len(sample_trades)
    if remaining > 0:
        remaining_sample = trades_df.sample(min(remaining, len(trades_df)))
        sample_trades.extend(remaining_sample.to_dict("records"))

    analyses = []

    for i, trade in enumerate(sample_trades[:sample_size]):
        print(f"\n📊 取引 {i+1} 分析")
        print("-" * 40)

        # Find corresponding feature data
        step = trade.get("step", 0)
        if step < len(features_df):
            features_at_time = features_df.iloc[step]
        else:
            print(f"⚠️ ステップ {step} の特徴量データが見つかりません")
            continue

        # Analyze the trade
        analysis = analyze_trade_decision_factors(pd.Series(trade), features_at_time)
        analyses.append(analysis)

        # Print analysis results
        trade_info = analysis["trade_info"]
        print(f"エピソード: {trade_info['episode']}, ステップ: {trade_info['step']}")
        print(f"行動: {trade_info['action_value']:.4f} ({trade_info['trade_type']})")
        print(
            f"結果: {analysis['outcome_analysis']['outcome']} (報酬: {trade_info['reward']:.2f})"
        )

        if analysis["decision_factors"]:
            print("判断要因:")
            for factor in analysis["decision_factors"][:3]:  # Show top 3
                print(f"  • {factor}")

        # Show key indicators
        key_indicators = analysis["key_indicators"]
        if key_indicators.get("trend"):
            print("トレンド指標:")
            for indicator, value in list(key_indicators["trend"].items())[:2]:
                print(f"  • {indicator}: {value:.4f}")

        if key_indicators.get("momentum"):
            print("モメンタム指標:")
            for indicator, value in list(key_indicators["momentum"].items())[:2]:
                print(f"  • {indicator}: {value:.4f}")

    return analyses


def generate_trade_analysis_report(analyses: List[Dict], trades_df: pd.DataFrame):
    """Generate a comprehensive trade analysis report."""

    print("\n📄 取引分析レポート生成")
    print("=" * 80)

    # Overall statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["reward"] > 0])
    losing_trades = len(trades_df[trades_df["reward"] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    print("📊 全体統計")
    print(f"総取引数: {total_trades}")
    print(f"勝ち取引: {winning_trades}")
    print(f"負け取引: {losing_trades}")
    print(f"勝率: {win_rate:.1%}")

    # Common decision patterns
    print("\n🔍 判断パターン分析")
    print("-" * 40)

    # Analyze decision factors frequency
    decision_factor_counts = {}
    for analysis in analyses:
        for factor in analysis["decision_factors"]:
            decision_factor_counts[factor] = decision_factor_counts.get(factor, 0) + 1

    print("頻出判断要因:")
    sorted_factors = sorted(
        decision_factor_counts.items(), key=lambda x: x[1], reverse=True
    )
    for factor, count in sorted_factors[:10]:  # Top 10
        percentage = (count / len(analyses)) * 100
        print(".1f")

    # Win/Loss pattern analysis
    print("\n⚖️ 勝敗パターン分析")
    print("-" * 40)

    win_analyses = [a for a in analyses if a["outcome_analysis"]["outcome"] == "WIN"]
    loss_analyses = [a for a in analyses if a["outcome_analysis"]["outcome"] == "LOSS"]

    print(f"勝ち取引分析数: {len(win_analyses)}")
    print(f"負け取引分析数: {len(loss_analyses)}")

    # Common factors in wins
    if win_analyses:
        win_factors = {}
        for analysis in win_analyses:
            for factor in analysis["decision_factors"]:
                win_factors[factor] = win_factors.get(factor, 0) + 1

        print("勝ち取引の共通要因:")
        sorted_win_factors = sorted(
            win_factors.items(), key=lambda x: x[1], reverse=True
        )
        for factor, count in sorted_win_factors[:5]:
            percentage = (count / len(win_analyses)) * 100
            print(".1f")

    # Common factors in losses
    if loss_analyses:
        loss_factors = {}
        for analysis in loss_analyses:
            for factor in analysis["decision_factors"]:
                loss_factors[factor] = loss_factors.get(factor, 0) + 1

        print("負け取引の共通要因:")
        sorted_loss_factors = sorted(
            loss_factors.items(), key=lambda x: x[1], reverse=True
        )
        for factor, count in sorted_loss_factors[:5]:
            percentage = (count / len(loss_analyses)) * 100
            print(".1f")

    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"v437_trade_analysis_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("SAC v437 取引分析レポート - 各取引の勝敗理由\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")

        f.write("全体統計\n")
        f.write("-" * 40 + "\n")
        f.write(f"総取引数: {total_trades}\n")
        f.write(f"勝ち取引: {winning_trades}\n")
        f.write(f"負け取引: {losing_trades}\n")
        f.write(f"勝率: {win_rate:.1%}\n\n")

        f.write("判断パターン分析\n")
        f.write("-" * 40 + "\n")
        for factor, count in sorted_factors[:10]:
            percentage = (count / len(analyses)) * 100
            f.write(".1f")

        f.write("\n勝敗パターン分析\n")
        f.write("-" * 40 + "\n")
        if win_analyses:
            f.write("勝ち取引の共通要因:\n")
            for factor, count in sorted_win_factors[:5]:
                percentage = (count / len(win_analyses)) * 100
                f.write(".1f")

        if loss_analyses:
            f.write("\n負け取引の共通要因:\n")
            for factor, count in sorted_loss_factors[:5]:
                percentage = (count / len(loss_analyses)) * 100
                f.write(".1f")

        f.write("\n個別取引分析サンプル\n")
        f.write("-" * 40 + "\n")
        for i, analysis in enumerate(analyses[:5]):  # Save first 5 detailed analyses
            f.write(f"\n取引 {i+1}:\n")
            f.write(f"  エピソード: {analysis['trade_info']['episode']}\n")
            f.write(f"  行動: {analysis['trade_info']['action_value']:.4f}\n")
            f.write(f"  結果: {analysis['outcome_analysis']['outcome']}\n")
            f.write(f"  判断要因: {', '.join(analysis['decision_factors'][:3])}\n")

    print(f"✅ 詳細レポートを保存しました: {report_file}")


def main():
    """Main analysis function."""

    print("🧠 SAC v437 取引判断分析 - 各取引の勝敗理由")
    print("=" * 80)

    # Load data
    trades_df, portfolio_df = load_trade_analysis_data()
    if trades_df is None:
        return

    features_df = load_feature_data_for_analysis()
    if features_df is None:
        return

    print(
        f"📊 データ読み込み完了: {len(trades_df)} 取引, {len(features_df)} 特徴量ポイント"
    )

    # Analyze sample trades
    analyses = analyze_sample_trades(trades_df, features_df, sample_size=10)

    # Generate comprehensive report
    generate_trade_analysis_report(analyses, trades_df)

    print("\n🎉 取引分析完了！")
    print("各取引の判断要因と勝敗理由を分析しました。")


if __name__ == "__main__":
    main()
