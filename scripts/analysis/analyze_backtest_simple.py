#!/usr/bin/env python3
"""
SAC v445.2 Backtest Results Analyzer
簡易的なバックテスト結果分析ツール
"""

import json
import sys
from typing import Any, Dict, Optional


def load_json_file(file_path: str) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_backtest_results(results_file: str, training_file: Optional[str] = None):
    """バックテスト結果を分析"""
    print("=" * 80)
    print("🎯 SAC v445.2 BACKTEST RESULTS ANALYSIS")
    print("=" * 80)

    # 結果ファイル読み込み
    results = load_json_file(results_file)
    training = load_json_file(training_file) if training_file else None

    # 基本情報表示
    print("📋 BASIC INFORMATION")
    print(f"   Model: {results.get('model_path', 'N/A')}")
    print(f"   Config: {results.get('config_path', 'N/A')}")
    print(f"   Episodes: {results.get('num_episodes', 'N/A')}")
    print(f"   Max Steps/Episode: {results.get('max_steps_per_episode', 'N/A')}")
    print(f"   Duration: {results.get('backtest_duration', 'N/A')}")
    print()

    # エピソードメトリクス分析
    episode_metrics = results.get("episode_metrics", {})
    print("🎭 EPISODE METRICS")
    print(f"   Mean Reward: {episode_metrics.get('mean_reward', 0):.2f}")
    print(f"   Reward Std: {episode_metrics.get('std_reward', 0):.2f}")
    print(
        f"   Mean Episode Length: {episode_metrics.get('mean_episode_length', 0):.1f}"
    )
    print()

    # アクション分析
    action_analysis = results.get("action_analysis", {})
    print("🎮 ACTION ANALYSIS")
    print(f"   Mean: {action_analysis.get('mean', 0):.4f}")
    print(f"   Std: {action_analysis.get('std', 0):.4f}")
    print(f"   Min: {action_analysis.get('min', 0):.4f}")
    print(f"   Max: {action_analysis.get('max', 0):.4f}")
    print()
    print("   Action Distribution:")
    distribution = action_analysis.get("distribution", {})
    for action_type, percentage in distribution.items():
        print(f"     {action_type.upper()}: {percentage:.1%}")
    print()

    # 報酬分析
    reward_analysis = results.get("reward_analysis", {})
    print("💰 REWARD ANALYSIS")
    print(f"   Mean: {reward_analysis.get('mean', 0):.4f}")
    print(f"   Std: {reward_analysis.get('std', 0):.4f}")
    print(f"   Min: {reward_analysis.get('min', 0):.4f}")
    print(f"   Max: {reward_analysis.get('max', 0):.4f}")
    print(f"   Positive Rewards: {reward_analysis.get('positive_count', 0)}")
    print(f"   Negative Rewards: {reward_analysis.get('negative_count', 0)}")
    if reward_analysis.get("positive_count", 0) > 0:
        print(f"   Positive Mean: {reward_analysis.get('positive_mean', 0):.4f}")
    if reward_analysis.get("negative_count", 0) > 0:
        print(f"   Negative Mean: {reward_analysis.get('negative_mean', 0):.4f}")
    print()

    # リスクメトリクス
    risk_metrics = results.get("risk_metrics", {})
    print("📊 RISK METRICS")
    print(f"   Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}")
    print(f"   Win Rate: {risk_metrics.get('win_rate', 0):.1%}")
    print(f"   Profit Factor: {risk_metrics.get('profit_factor', 0):.2f}")
    print()

    # ヘルスチェック
    health_checks = results.get("health_checks", {})
    print("🏥 HEALTH CHECKS")
    print(f"   Health Score: {health_checks.get('overall_health_score', 0):.2f}")
    print(
        f"   Status: {'✅ HEALTHY' if health_checks.get('healthy', False) else '⚠️ NEEDS ATTENTION'}"
    )
    print()

    # アクション分布ヘルス
    action_health = health_checks.get("action_distribution", {})
    print("   Action Distribution Health:")
    for check, status in action_health.items():
        icon = "✅" if status else "❌"
        print(f"     {icon} {check.replace('_', ' ').title()}")
    print()

    # 報酬安定性ヘルス
    reward_health = health_checks.get("reward_stability", {})
    print("   Reward Stability Health:")
    for check, status in reward_health.items():
        icon = "✅" if status else "❌"
        print(f"     {icon} {check.replace('_', ' ').title()}")
    print()

    # トレーニング比較（利用可能な場合）
    if training:
        print("🔄 TRAINING vs BACKTEST COMPARISON")
        training_perf = training.get("performance_metrics", {})
        backtest_episode = episode_metrics

        print("   Training Performance:")
        print(f"     Mean Reward: {training_perf.get('mean_reward', 0):.2f}")
        print(f"     Entropy Mean: {training_perf.get('entropy_mean', 0):.4f}")
        print(f"     Action Mean: {training_perf.get('action_mean_avg', 0):.4f}")
        print(f"     Action Std: {training_perf.get('action_std_avg', 0):.4f}")
        print()
        print("   Backtest Performance:")
        print(f"     Mean Reward: {episode_metrics.get('mean_reward', 0):.2f}")
        print(f"     Action Mean: {action_analysis.get('mean', 0):.4f}")
        print(f"     Action Std: {action_analysis.get('std', 0):.4f}")
        print()

        # 比較分析
        train_entropy = training_perf.get("entropy_mean", 0)
        backtest_action_std = action_analysis.get("std", 0)

        print("   Analysis:")
        if train_entropy > 0:
            print(f"     Training entropy was {train_entropy:.4f}")
        if backtest_action_std > 0.1:
            print("     ✅ Backtest shows reasonable action variance")
        else:
            print("     ⚠️ Backtest shows low action variance - possible sticking")

        train_reward_mean = training_perf.get("mean_reward", 0)
        backtest_reward_mean = reward_analysis.get("mean", 0)

        if abs(backtest_reward_mean - train_reward_mean) < abs(train_reward_mean) * 0.5:
            print("     ✅ Reward levels are consistent between training and backtest")
        else:
            print("     ⚠️ Significant difference in reward levels")
        print()

    # 総合評価
    print("🎯 OVERALL ASSESSMENT")
    health_score = health_checks.get("overall_health_score", 0)
    win_rate = risk_metrics.get("win_rate", 0)

    if health_score >= 0.8 and win_rate > 0:
        print("✅ EXCELLENT: Model shows healthy performance across all metrics")
    elif health_score >= 0.6:
        print("⚠️ GOOD: Model performs adequately but has some areas for improvement")
    elif health_score >= 0.4:
        print("❌ FAIR: Model shows concerning performance patterns")
    else:
        print("❌ POOR: Model requires significant improvements")

    print()
    print("💡 RECOMMENDATIONS:")
    if health_score < 0.6:
        print("   • Consider additional training with more timesteps")
        print("   • Review reward function and environment setup")
        print("   • Check for any training instability issues")

    if action_analysis.get("std", 0) < 0.3:
        print("   • Action variance is low - may indicate value sticking")
        print("   • Consider entropy regularization adjustments")

    if win_rate == 0:
        print("   • No positive episodes - fundamental training issues detected")
        print("   • Review training convergence and reward scaling")

    print()
    print("=" * 80)


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_backtest_simple.py <results_file> [training_file]")
        sys.exit(1)

    results_file = sys.argv[1]
    training_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        analyze_backtest_results(results_file, training_file)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
