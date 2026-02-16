#!/usr/bin/env python3
"""
バックテスト詳細分析スクリプト
勝率の低さや収益性の原因を詳細に分析
"""

from typing import Any, Dict

import matplotlib.pyplot as plt
from ztb.analysis.common.plot_utils import setup_plot_style, save_plot
import numpy as np
import pandas as pd

from ztb.analysis.common.types import EpisodeAnalysisResult, PortfolioAnalysisResult
from ztb.io.text_io import write_text
from ztb.io.data_loader import DataLoader


def load_portfolio_values(portfolio_path: str) -> pd.DataFrame:
    """ポートフォリオ価値データを読み込み"""
    return DataLoader.load_csv_strict(portfolio_path)


def load_trades(trades_path: str) -> pd.DataFrame:
    """取引データを読み込み"""
    return DataLoader.load_csv_strict(trades_path)


def analyze_portfolio_performance(portfolio_df: pd.DataFrame) -> PortfolioAnalysisResult:
    """ポートフォリオパフォーマンスの詳細分析"""
    initial_value = values[0]
    final_value = values[-1]
    total_return = (final_value - initial_value) / initial_value * 100

    # 期間分析
    total_steps = len(values) - 1
    returns = np.diff(values) / values[:-1]

    # リスク指標
    from ztb.metrics.metrics import sharpe_ratio as calc_sharpe_ratio
    from ztb.metrics.technical import calculate_volatility_from_returns

    volatility = calculate_volatility_from_returns(
        returns, window=len(returns), annualize=True
    )
    sharpe_ratio = calc_sharpe_ratio(returns)

    # ドローダウン分析
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    # 収益分布
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    return {
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return_pct": total_return,
        "total_steps": total_steps,
        "avg_daily_return": np.mean(returns) * 100,
        "volatility": volatility * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown,
        "win_rate": len(positive_returns) / len(returns) * 100,
        "avg_win": np.mean(positive_returns) * 100 if len(positive_returns) > 0 else 0,
        "avg_loss": np.mean(negative_returns) * 100 if len(negative_returns) > 0 else 0,
        "profit_factor": abs(
            np.sum(positive_returns) / (np.sum(negative_returns) + 1e-8)
        ),
        "total_positive_return": np.sum(positive_returns) * 100,
        "total_negative_return": np.sum(negative_returns) * 100,
    }


def analyze_episode_performance(trades_df: pd.DataFrame) -> EpisodeAnalysisResult:
    """エピソードごとのパフォーマンス分析"""
    rewards = trades_df["reward"].values
    final_portfolios = trades_df["final_portfolio"].values

    # エピソード分析
    positive_episodes = rewards[rewards > 0]
    negative_episodes = rewards[rewards < 0]

    episode_analysis: EpisodeAnalysisResult = {
        "total_episodes": len(rewards),
        "positive_episodes": int(len(positive_episodes)),
        "negative_episodes": int(len(negative_episodes)),
        "episode_win_rate": float(len(positive_episodes) / max(len(rewards), 1) * 100),
        "avg_episode_reward": float(np.mean(rewards)),
        "best_episode_reward": float(np.max(rewards)),
        "worst_episode_reward": float(np.min(rewards)),
        "episode_reward_std": float(np.std(rewards)),
        "avg_final_portfolio": float(np.mean(final_portfolios)),
        "best_final_portfolio": float(np.max(final_portfolios)),
        "worst_final_portfolio": float(np.min(final_portfolios)),
    }

    return episode_analysis


def analyze_conversion(
    trades_df: pd.DataFrame,
) -> Dict[str, Any]:
    """報酬とポートフォリオ変化の変換分析"""
    episode_portfolio_changes: list[float] = []

    for _i, row in trades_df.iterrows():
        episode = int(row["episode"])
        final_portfolio = float(row["final_portfolio"])

        # エピソード開始時のポートフォリオ価値を取得
        if episode == 0:
            start_value = 10000.0
        else:
            # 前のエピソードの最終価値
            prev_final = float(trades_df.iloc[episode - 1]["final_portfolio"])
            start_value = prev_final

        change_pct = (final_portfolio - start_value) / start_value * 100
        episode_portfolio_changes.append(change_pct)

    # 報酬とポートフォリオ変化の相関分析
    rewards = trades_df["reward"].values
    correlation = float(np.corrcoef(rewards, episode_portfolio_changes)[0, 1])
    positive_change = sum(1 for c in episode_portfolio_changes if c > 0)

    return {
        "episode_portfolio_changes": episode_portfolio_changes,
        "reward_portfolio_correlation": correlation,
        "avg_portfolio_change_per_episode": float(np.mean(episode_portfolio_changes)),
        "portfolio_change_std": float(np.std(episode_portfolio_changes)),
        "positive_change_episodes": positive_change,
    }


def generate_detailed_report(
    portfolio_analysis: PortfolioAnalysisResult,
    episode_analysis: EpisodeAnalysisResult,
    conversion_analysis: Dict[str, Any],
) -> str:
    """詳細分析レポート生成"""
    return f"""# バックテスト詳細分析レポート

## ポートフォリオ分析

- **初期値**: {portfolio_analysis['initial_value']:.2f}
- **最終値**: {portfolio_analysis['final_value']:.2f}
- **総リターン**: {portfolio_analysis['total_return_pct']:.2f}%
- **シャープレシオ**: {portfolio_analysis['sharpe_ratio']:.3f}
- **最大ドローダウン**: {portfolio_analysis['max_drawdown_pct']:.2f}%
- **勝率**: {portfolio_analysis['win_rate']:.1f}%
- **平均勝ち**: {portfolio_analysis['avg_win']:.4f}%
- **平均負け**: {portfolio_analysis['avg_loss']:.4f}%
- **総勝ち額**: {portfolio_analysis['total_positive_return']:.2f}%
- **総負け額**: {portfolio_analysis['total_negative_return']:.2f}%

## エピソード分析

### エピソード統計
- **総エピソード数**: {episode_analysis['total_episodes']}
- **勝ちエピソード**: {episode_analysis['positive_episodes']}
- **負けエピソード**: {episode_analysis['negative_episodes']}
- **エピソード勝率**: {episode_analysis['episode_win_rate']:.1f}%

### エピソード報酬分布
- **平均報酬**: {episode_analysis['avg_episode_reward']:.2f}
- **最高報酬**: {episode_analysis['best_episode_reward']:.2f}
- **最低報酬**: {episode_analysis['worst_episode_reward']:.2f}
- **報酬標準偏差**: {episode_analysis['episode_reward_std']:.2f}

## 変換分析

- **報酬-ポートフォリオ相関**: {conversion_analysis['reward_portfolio_correlation']:.3f}
- **平均変化/エピソード**: {conversion_analysis['avg_portfolio_change_per_episode']:.2f}%
- **変化標準偏差**: {conversion_analysis['portfolio_change_std']:.1f}%
"""


def main() -> None:
    """バックテスト詳細分析を実行"""
    portfolio_path = "results/portfolio_values.csv"
    trades_path = "results/trades.csv"

    portfolio_df = load_portfolio_values(portfolio_path)
    trades_df = load_trades(trades_path)

    portfolio_analysis = analyze_portfolio_performance(portfolio_df)
    episode_analysis = analyze_episode_performance(trades_df)
    conversion_analysis = analyze_conversion(trades_df)

    report = generate_detailed_report(
        portfolio_analysis, episode_analysis, conversion_analysis
    )

    # レポート保存
    write_text("backtest_detailed_analysis_report.md", report)

    print("レポート保存: backtest_detailed_analysis_report.md")

    # 結果表示
    print("\n=== 分析結果サマリー ===")
    print(
        f"平均ポートフォリオ変化/エピソード: {conversion_analysis['avg_portfolio_change_per_episode']:.2f}%"
    )
    print(
        f"ポートフォリオ変化標準偏差: {conversion_analysis['portfolio_change_std']:.1f}%"
    )
    print(
        f"報酬-ポートフォリオ相関係数: {conversion_analysis['reward_portfolio_correlation']:.2f}"
    )
    print(
        f"ポジティブ変化エピソード数: {conversion_analysis['positive_change_episodes']}"
    )
    print(f"総エピソード数: {len(conversion_analysis['episode_portfolio_changes'])}")
    ep_changes = conversion_analysis['episode_portfolio_changes']
    positive_count = conversion_analysis['positive_change_episodes']
    if ep_changes:
        print(f"勝率: {positive_count / len(ep_changes) * 100:.1f}%")


if __name__ == "__main__":
    main()

