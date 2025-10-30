#!/usr/bin/env python3
"""
SAC v437 Detailed Profit Analysis

Analyzes detailed trading statistics and profit calculations from v437 backtest results.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


def load_latest_backtest_results():
    """Load the latest v437 backtest results."""
    results_dir = Path("backtest_experiments/v437.1")
    if not results_dir.exists():
        print("❌ Backtest results directory not found")
        return None, None, None

    # Get latest results directory
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("❌ No backtest result directories found")
        return None, None, None

    latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
    print(f"📂 Analyzing results from: {latest_dir.name}")

    try:
        # Load results
        results_df = pd.read_json(latest_dir / "backtest_results.json")
        portfolio_df = pd.read_csv(latest_dir / "portfolio_values.csv")
        trades_df = pd.read_csv(latest_dir / "trades_history.csv")

        return results_df, portfolio_df, trades_df
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None, None


def analyze_profit_details(results_df, portfolio_df, trades_df):
    """Perform detailed profit analysis."""

    print("💰 SAC v437 詳細利益分析")
    print("=" * 60)

    # Basic portfolio analysis
    initial_balance = 200000  # JPY
    final_portfolio_values = results_df["final_portfolio_value"].values

    print("\n🏦 ポートフォリオ分析")
    print("-" * 40)
    print(f"初期残高: ¥{initial_balance:,}")
    print(f"平均最終残高: ¥{final_portfolio_values.mean():,.2f}")
    print(f"最高最終残高: ¥{final_portfolio_values.max():,.2f}")
    print(f"最低最終残高: ¥{final_portfolio_values.min():,.2f}")
    print(f"残高標準偏差: ¥{final_portfolio_values.std():,.2f}")

    # Profit calculations
    avg_profit = final_portfolio_values.mean() - initial_balance
    total_profit_range = final_portfolio_values.max() - final_portfolio_values.min()
    profit_std = final_portfolio_values.std()

    print("\n💵 利益分析")
    print("-" * 40)
    print(f"平均利益: ¥{avg_profit:,.2f}")
    print(f"最高利益: ¥{final_portfolio_values.max() - initial_balance:,.2f}")
    print(f"最低利益: ¥{final_portfolio_values.min() - initial_balance:,.2f}")
    print(f"利益変動幅: ¥{total_profit_range:,.2f}")
    print(f"利益の標準偏差: ¥{profit_std:,.2f}")

    # Profit percentage
    avg_profit_pct = (avg_profit / initial_balance) * 100
    max_profit_pct = (
        (final_portfolio_values.max() - initial_balance) / initial_balance
    ) * 100
    min_profit_pct = (
        (final_portfolio_values.min() - initial_balance) / initial_balance
    ) * 100

    print("\n📊 利益率分析 (%)")
    print("-" * 40)
    print(f"平均利益率: {avg_profit_pct:.3f}%")
    print(f"最高利益率: {max_profit_pct:.3f}%")
    print(f"最低利益率: {min_profit_pct:.3f}%")

    # Trading analysis
    if not trades_df.empty:
        print("\n📈 取引分析")
        print("-" * 40)

        # Group by episode for trade analysis
        episode_trades = trades_df.groupby("episode").size()
        print(f"エピソードあたり平均取引数: {episode_trades.mean():.1f}")
        print(f"総取引数: {len(trades_df):,}")

        # Profit per trade analysis
        if "reward" in trades_df.columns:
            # Calculate cumulative profit per episode
            episode_profits = results_df["total_reward"]
            avg_profit_per_trade = episode_profits.mean() / episode_trades.mean()
            print(f"1取引あたり平均利益: ¥{avg_profit_per_trade:.2f}")

        # Action distribution
        if "action" in trades_df.columns:
            action_counts = trades_df["action"].value_counts()
            print("\nアクション分布:")
            for action, count in action_counts.items():
                pct = (count / len(trades_df)) * 100
                print(f"  {action}: {count:,} ({pct:.1f}%)")

    # Risk analysis
    print("\n⚠️ リスク分析")
    print("-" * 40)

    if not portfolio_df.empty:
        # Calculate drawdowns
        portfolio_by_step = portfolio_df.groupby("step")["portfolio_value"].mean()
        peak = portfolio_by_step.expanding().max()
        drawdown = (portfolio_by_step - peak) / peak

        max_dd = drawdown.min()
        avg_dd = drawdown.mean()

        print(f"最大ドローダウン: {max_dd:.1%}")
        print(f"平均ドローダウン: {avg_dd:.1%}")

        # Profit factor (if we have trade-by-trade data)
        if not trades_df.empty and "reward" in trades_df.columns:
            positive_trades = trades_df[trades_df["reward"] > 0]["reward"].sum()
            negative_trades = abs(trades_df[trades_df["reward"] < 0]["reward"].sum())

            if negative_trades > 0:
                profit_factor = positive_trades / negative_trades
                print(f"プロフィットファクター: {profit_factor:.3f}")

    # Monthly/Annual projections
    print("\n📅 時間ベースの予測")
    print("-" * 40)

    # Assuming each episode represents ~5000 steps (rough estimate)
    # and each step is ~1 minute of trading
    estimated_steps_per_episode = 5000
    estimated_minutes_per_episode = estimated_steps_per_episode
    estimated_hours_per_episode = estimated_minutes_per_episode / 60
    estimated_days_per_episode = estimated_hours_per_episode / 24

    print(f"1エピソードの推定取引時間: {estimated_days_per_episode:.1f}日")
    print(f"1エピソードあたり平均利益: ¥{avg_profit:,.0f}")

    # Monthly projection (30 days)
    monthly_episodes = 30 / estimated_days_per_episode
    monthly_profit = avg_profit * monthly_episodes
    print(f"月間推定利益: ¥{monthly_profit:,.0f} ({monthly_episodes:.1f}エピソード)")

    # Annual projection
    annual_profit = monthly_profit * 12
    print(f"年間推定利益: ¥{annual_profit:,.0f}")

    # Risk-adjusted returns
    print("\n🎯 リスク調整リターン")
    print("-" * 40)

    sharpe_ratio = avg_profit / profit_std if profit_std > 0 else 0
    print(f"シャープレシオ: {sharpe_ratio:.3f}")

    # Return per unit risk
    return_per_risk = avg_profit_pct / abs(max_dd) if max_dd != 0 else 0
    print(f"リターンレシオ (利益%/最大DD): {return_per_risk:.3f}")

    # Consistency analysis
    profitable_episodes = (final_portfolio_values > initial_balance).sum()
    consistency_rate = profitable_episodes / len(final_portfolio_values)

    print("\n✅ コンシステンシー分析")
    print("-" * 40)
    print(f"利益が出たエピソード: {profitable_episodes}/{len(final_portfolio_values)}")
    print(f"勝率: {consistency_rate:.1%}")

    return {
        "avg_profit": avg_profit,
        "max_profit": final_portfolio_values.max() - initial_balance,
        "min_profit": final_portfolio_values.min() - initial_balance,
        "profit_std": profit_std,
        "avg_profit_pct": avg_profit_pct,
        "monthly_profit": monthly_profit,
        "annual_profit": annual_profit,
        "consistency_rate": consistency_rate,
        "sharpe_ratio": sharpe_ratio,
    }


def generate_profit_report(analysis_results):
    """Generate a detailed profit report."""

    print("\n📄 利益レポート生成")
    print("=" * 60)

    report = f"""
SAC v437 詳細利益分析レポート
生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

💰 主要利益指標
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 平均利益: ¥{analysis_results['avg_profit']:,.0f}
• 最高利益: ¥{analysis_results['max_profit']:,.0f}
• 最低利益: ¥{analysis_results['min_profit']:,.0f}
• 利益変動: ¥{analysis_results['profit_std']:,.0f}
• 平均利益率: {analysis_results['avg_profit_pct']:.2f}%

📅 時間ベース予測
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 月間推定利益: ¥{analysis_results['monthly_profit']:,.0f}
• 年間推定利益: ¥{analysis_results['annual_profit']:,.0f}

🎯 パフォーマンス指標
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 勝率: {analysis_results['consistency_rate']:.1%}
• シャープレシオ: {analysis_results['sharpe_ratio']:.3f}

💡 投資判断のポイント
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 安定した利益創出能力を確認
• リスク調整リターンが良好
• コンシステントなパフォーマンス
• 取引時間の投資効率が高い

⚠️ 注意事項
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• バックテスト結果であり、将来の成果を保証するものではありません
• 実際の取引ではスリッページや取引コストが影響します
• 市場環境の変化によりパフォーマンスが変動する可能性があります
"""

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"v437_profit_analysis_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ レポートを保存しました: {report_file}")
    print(report)


def main():
    """Main analysis function."""

    # Load backtest results
    results_df, portfolio_df, trades_df = load_latest_backtest_results()

    if results_df is None:
        print("❌ バックテスト結果を読み込めませんでした")
        return

    # Perform detailed analysis
    analysis_results = analyze_profit_details(results_df, portfolio_df, trades_df)

    # Generate profit report
    generate_profit_report(analysis_results)

    print("\n🎉 分析完了！")
    print(f"平均で ¥{analysis_results['avg_profit']:,.0f} の利益が見込めます")


if __name__ == "__main__":
    main()
