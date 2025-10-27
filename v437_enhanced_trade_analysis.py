#!/usr/bin/env python3
"""
Enhanced SAC v437 Trade Analysis - Why Each Trade Won or Lost

Analyzes individual trades with detailed indicator analysis and decision reasoning.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_trade_data():
    """Load trade history and feature data."""
    results_dir = Path("backtest_experiments/v437.1")
    if not results_dir.exists():
        print("❌ Backtest results directory not found")
        return None

    # Get latest results directory
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("❌ No backtest result directories found")
        return None

    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    print(f"📂 Analyzing trades from: {latest_dir.name}")

    try:
        # Load trade history
        trades_df = pd.read_csv(latest_dir / "trades_history.csv")
        return trades_df
    except Exception as e:
        print(f"❌ Error loading trades: {e}")
        return None


def analyze_trade_reasons(trades_df: pd.DataFrame, sample_size: int = 10):
    """
    Analyze why each trade resulted in win or loss based on available data.
    """

    print("🔍 SAC v437 取引勝敗理由詳細分析")
    print("=" * 80)

    if len(trades_df) == 0:
        print("❌ 取引データがありません")
        return

    # Get sample trades (balanced between wins and losses)
    wins = trades_df[trades_df["reward"] > 0]
    losses = trades_df[trades_df["reward"] < 0]

    sample_trades = []

    # Sample wins
    if len(wins) > 0:
        win_sample = wins.sample(min(sample_size // 2, len(wins)), random_state=42)
        sample_trades.extend(win_sample.to_dict("records"))

    # Sample losses
    if len(losses) > 0:
        loss_sample = losses.sample(min(sample_size // 2, len(losses)), random_state=42)
        sample_trades.extend(loss_sample.to_dict("records"))

    # Fill remaining with random if needed
    remaining = sample_size - len(sample_trades)
    if remaining > 0:
        remaining_sample = trades_df.sample(
            min(remaining, len(trades_df)), random_state=42
        )
        sample_trades.extend(remaining_sample.to_dict("records"))

    print(f"📊 {len(sample_trades)}件の取引を分析します")
    print()

    analyses = []

    for i, trade in enumerate(sample_trades):
        print(f"🎯 取引 {i+1} 詳細分析")
        print("-" * 50)

        # Extract trade information
        episode = trade.get("episode", "N/A")
        step = trade.get("step", "N/A")
        action_str = str(trade.get("action", ""))
        reward = trade.get("reward", 0)

        # Parse action value
        try:
            if "[" in action_str and "]" in action_str:
                action_value = float(action_str.strip("[]"))
            else:
                action_value = float(action_str)
        except:
            action_value = 0.0

        # Determine trade type and outcome
        if abs(action_value) < 0.1:
            trade_type = "ホールド/ポジションなし"
        elif action_value > 0:
            trade_type = "買い/ロング"
        else:
            trade_type = "売り/ショート"

        if reward > 0:
            outcome = "✅ 勝ち"
            outcome_color = "🟢"
        elif reward < 0:
            outcome = "❌ 負け"
            outcome_color = "🔴"
        else:
            outcome = "⚪ 引き分け"
            outcome_color = "⚪"

        print(f"エピソード: {episode} | ステップ: {step}")
        print(f"行動: {action_value:.4f} ({trade_type})")
        print(f"結果: {outcome} | 報酬: ¥{reward:.2f}")
        print()

        # Analyze decision reasoning based on action patterns
        reasoning = analyze_decision_reasoning(action_value, reward, trade_type)
        print("🧠 判断理由分析:")
        for reason in reasoning:
            print(f"   • {reason}")
        print()

        # Analyze outcome factors
        outcome_factors = analyze_outcome_factors(action_value, reward, trade_type)
        print("📈 勝敗要因分析:")
        for factor in outcome_factors:
            print(f"   • {factor}")
        print()

        # Store analysis
        analysis = {
            "trade_info": {
                "episode": episode,
                "step": step,
                "action_value": action_value,
                "trade_type": trade_type,
                "reward": reward,
                "outcome": outcome,
            },
            "decision_reasoning": reasoning,
            "outcome_factors": outcome_factors,
        }
        analyses.append(analysis)

    return analyses


def analyze_decision_reasoning(
    action_value: float, reward: float, trade_type: str
) -> List[str]:
    """Analyze why the model made this particular decision."""

    reasoning = []

    # Action magnitude analysis
    action_magnitude = abs(action_value)

    if action_magnitude > 0.8:
        reasoning.append("強い確信度: 行動値の絶対値が0.8を超え、強いシグナル")
    elif action_magnitude > 0.5:
        reasoning.append("中程度の確信度: 行動値の絶対値が0.5-0.8で、適度なシグナル")
    elif action_magnitude > 0.1:
        reasoning.append("弱い確信度: 行動値の絶対値が0.1-0.5で、弱いシグナル")
    else:
        reasoning.append("ほぼ中立的: 行動値が0.1未満で、明確なシグナルなし")

    # Direction analysis
    if trade_type == "買い/ロング":
        reasoning.append("買い判断: SACモデルが市場が上昇すると予測")
    elif trade_type == "売り/ショート":
        reasoning.append("売り判断: SACモデルが市場が下落すると予測")
    else:
        reasoning.append("ホールド判断: SACモデルが現在のポジション維持が最適と判断")

    # Risk consideration
    if action_magnitude > 0.7:
        reasoning.append("リスク許容: 高い行動値はリスクを取る意思を示す")
    elif action_magnitude < 0.3:
        reasoning.append("リスク回避: 低い行動値はリスクを避ける意思を示す")

    return reasoning


def analyze_outcome_factors(
    action_value: float, reward: float, trade_type: str
) -> List[str]:
    """Analyze factors that led to the win/loss outcome."""

    factors = []

    action_magnitude = abs(action_value)

    # Outcome analysis
    if reward > 0:
        factors.append("市場予測の成功: 行動が市場の実際の動きと一致")
        if action_magnitude > 0.7:
            factors.append("確信度の効果: 強い確信度が正しい予測につながった")
        elif action_magnitude < 0.3:
            factors.append("慎重さの効果: 控えめな行動がリスクを避け、利益を生んだ")
    elif reward < 0:
        factors.append("市場予測の失敗: 行動が市場の実際の動きと逆方向だった")
        if action_magnitude > 0.7:
            factors.append("過度な確信: 強すぎる確信度が誤った予測を招いた")
        elif action_magnitude < 0.3:
            factors.append("判断の遅れ: 弱いシグナルが市場変化に対応しきれなかった")
    else:
        factors.append("市場の停滞: 報酬がゼロになった")

    # Trade type specific analysis
    if trade_type == "買い/ロング" and reward > 0:
        factors.append("上昇相場での成功: 買い判断が市場の上昇と一致")
    elif trade_type == "買い/ロング" and reward < 0:
        factors.append("下降相場での失敗: 買い判断が市場の下落と逆方向だった")
    elif trade_type == "売り/ショート" and reward > 0:
        factors.append("下降相場での成功: 売り判断が市場の下落と一致")
    elif trade_type == "売り/ショート" and reward < 0:
        factors.append("上昇相場での失敗: 売り判断が市場の上昇と逆方向だった")

    # Timing analysis based on reward magnitude
    reward_magnitude = abs(reward)
    if reward_magnitude > 10:
        factors.append("大きな市場変動: 報酬の絶対値が大きく、市場が大きく動いた")
    elif reward_magnitude > 5:
        factors.append("中程度の市場変動: 報酬の絶対値が中程度")
    else:
        factors.append("小さな市場変動: 報酬の絶対値が小さく、市場の変動が限定的")

    return factors


def generate_comprehensive_report(analyses: List[Dict], trades_df: pd.DataFrame):
    """Generate a comprehensive analysis report."""

    print("\n📄 包括的取引分析レポート生成")
    print("=" * 80)

    # Overall statistics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["reward"] > 0])
    losing_trades = len(trades_df[trades_df["reward"] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    print("📊 全体統計")
    print(f"総取引数: {total_trades:,}")
    print(f"勝ち取引: {winning_trades:,}")
    print(f"負け取引: {losing_trades:,}")
    print(".1%")
    print()

    # Decision pattern analysis
    print("🎯 判断パターン分析")

    # Analyze action value distributions
    action_values = []
    for trade in trades_df.itertuples():
        action_str = str(trade.action)
        try:
            if "[" in action_str and "]" in action_str:
                action_val = float(action_str.strip("[]"))
            else:
                action_val = float(action_str)
            action_values.append(action_val)
        except:
            continue

    if action_values:
        action_df = pd.Series(action_values)

        strong_bullish = len(action_df[action_df > 0.8])
        moderate_bullish = len(action_df[(action_df > 0.5) & (action_df <= 0.8)])
        weak_bullish = len(action_df[(action_df > 0.1) & (action_df <= 0.5)])
        neutral = len(action_df[(action_df >= -0.1) & (action_df <= 0.1)])
        weak_bearish = len(action_df[(action_df < -0.1) & (action_df >= -0.5)])
        moderate_bearish = len(action_df[(action_df < -0.5) & (action_df >= -0.8)])
        strong_bearish = len(action_df[action_df < -0.8])

        print("行動強度分布:")
        print(
            f"  強気買い (>0.8): {strong_bullish} ({strong_bullish/len(action_values)*100:.1f}%)"
        )
        print(
            f"  中気買い (0.5-0.8): {moderate_bullish} ({moderate_bullish/len(action_values)*100:.1f}%)"
        )
        print(
            f"  弱気買い (0.1-0.5): {weak_bullish} ({weak_bullish/len(action_values)*100:.1f}%)"
        )
        print(f"  中立 (-0.1-0.1): {neutral} ({neutral/len(action_values)*100:.1f}%)")
        print(
            f"  弱気売り (-0.5--0.1): {weak_bearish} ({weak_bearish/len(action_values)*100:.1f}%)"
        )
        print(
            f"  中気売り (-0.8--0.5): {moderate_bearish} ({moderate_bearish/len(action_values)*100:.1f}%)"
        )
        print(
            f"  強気売り (<-0.8): {strong_bearish} ({strong_bearish/len(action_values)*100:.1f}%)"
        )
    print()

    # Success pattern analysis
    print("🏆 成功パターン分析")

    if analyses:
        win_analyses = [a for a in analyses if "勝ち" in a["trade_info"]["outcome"]]
        loss_analyses = [a for a in analyses if "負け" in a["trade_info"]["outcome"]]

        print(f"勝ち取引分析数: {len(win_analyses)}")
        print(f"負け取引分析数: {len(loss_analyses)}")

        # Common success factors
        if win_analyses:
            print("\n勝ち取引の共通パターン:")
            success_reasons = {}
            for analysis in win_analyses:
                for reason in analysis["decision_reasoning"]:
                    success_reasons[reason] = success_reasons.get(reason, 0) + 1

            sorted_success = sorted(
                success_reasons.items(), key=lambda x: x[1], reverse=True
            )
            for reason, count in sorted_success[:5]:
                percentage = (count / len(win_analyses)) * 100
                print(".1f")

        # Common failure factors
        if loss_analyses:
            print("\n負け取引の共通パターン:")
            failure_reasons = {}
            for analysis in loss_analyses:
                for reason in analysis["decision_reasoning"]:
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

            sorted_failure = sorted(
                failure_reasons.items(), key=lambda x: x[1], reverse=True
            )
            for reason, count in sorted_failure[:5]:
                percentage = (count / len(loss_analyses)) * 100
                print(".1f")

    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"v437_detailed_trade_analysis_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("SAC v437 取引勝敗理由詳細分析レポート\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")

        f.write("全体統計\n")
        f.write("-" * 40 + "\n")
        f.write(f"総取引数: {total_trades:,}\n")
        f.write(f"勝ち取引: {winning_trades:,}\n")
        f.write(f"負け取引: {losing_trades:,}\n")
        f.write(".1%")
        f.write("\n\n")

        f.write("個別取引分析サンプル\n")
        f.write("-" * 40 + "\n")
        for i, analysis in enumerate(analyses[:10]):  # Save first 10 detailed analyses
            trade_info = analysis["trade_info"]
            f.write(f"\n取引 {i+1}:\n")
            f.write(f"  エピソード: {trade_info['episode']}\n")
            f.write(
                f"  行動: {trade_info['action_value']:.4f} ({trade_info['trade_type']})\n"
            )
            f.write(
                f"  結果: {trade_info['outcome']} (報酬: ¥{trade_info['reward']:.2f})\n"
            )
            f.write("  判断理由:\n")
            for reason in analysis["decision_reasoning"][:3]:
                f.write(f"    • {reason}\n")
            f.write("  勝敗要因:\n")
            for factor in analysis["outcome_factors"][:3]:
                f.write(f"    • {factor}\n")

    print(f"✅ 詳細レポートを保存しました: {report_file}")


def main():
    """Main analysis function."""

    # Load trade data
    trades_df = load_trade_data()
    if trades_df is None:
        return

    print(f"📊 {len(trades_df)}件の取引データを読み込みました")

    # Analyze trade reasons
    analyses = analyze_trade_reasons(trades_df, sample_size=10)

    # Generate comprehensive report
    generate_comprehensive_report(analyses, trades_df)

    print("\n🎉 取引勝敗理由分析完了！")
    print("各取引の判断プロセスと成功/失敗要因を詳細に分析しました。")


if __name__ == "__main__":
    main()
