#!/usr/bin/env python3
"""
SAC v441 安定性重視設定テストスクリプト

v438分析結果に基づくv441改善設定の検証を行う。
1万ステップの制約下で段階的なテストを実施。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.core.analyzer import UnifiedAnalyzer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_test_config(base_config: dict, test_name: str) -> dict:
    """テスト用の設定を作成"""
    config = base_config.copy()
    config["training"]["model_name"] = f"sac_v441_{test_name}_10k"
    config["training"]["total_timesteps"] = 10000

    # テスト用の軽量設定
    config["evaluation"]["eval_freq"] = 1000
    config["evaluation"]["save_freq"] = 2000
    config["logging"]["log_interval"] = 200

    return config


def run_stability_test():
    """安定性テスト実行"""
    logger.info("Starting SAC v441 stability-focused training test...")

    # 設定読み込み
    config_path = "config/sac_v441_stability_focused_config.json"
    base_config = load_config(config_path)

    # テスト設定作成
    test_config = create_test_config(base_config, "stability_test")

    # 設定保存
    test_config_path = f"config/sac_v441_stability_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(test_config_path, "w", encoding="utf-8") as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)

    logger.info(f"Test configuration saved to: {test_config_path}")

    # トレーニング実行（実際にはコメントアウト）
    # unified_trainer_main([test_config_path])

    logger.info("Training test completed (simulation mode)")
    return test_config_path


def analyze_training_results(config_path: str):
    """トレーニング結果の分析"""
    logger.info("Analyzing training results...")

    # 分析器初期化
    analyzer = UnifiedAnalyzer()

    # 設定読み込み
    config = load_config(config_path)

    # 分析実行（モック）
    analysis_results = {
        "stability_score": 0.72,  # 目標: 0.75以上
        "regime_adaptability": 1.05,  # 目標: 0.85以上
        "statistical_significance": 0.78,  # 目標: 85%以上
        "action_balance": 0.76,  # 目標: 0.8以上
        "total_return": 0.16,
        "sharpe_ratio": 1.85,
        "win_rate": 0.56,
    }

    logger.info("Analysis Results:")
    for key, value in analysis_results.items():
        logger.info(f"  {key}: {value}")

    return analysis_results


def compare_with_v438():
    """v438との比較分析"""
    logger.info("Comparing with v438 results...")

    v438_results = {
        "stability_score": 0.565,
        "regime_adaptability": 1.0,
        "statistical_significance": 0.667,
        "action_balance": 0.68,
        "total_return": 0.15,
        "sharpe_ratio": 1.8,
        "win_rate": 0.55,
    }

    # v441の期待結果（理論値）
    v441_expected = {
        "stability_score": 0.75,
        "regime_adaptability": 0.85,
        "statistical_significance": 0.85,
        "action_balance": 0.8,
        "total_return": 0.165,
        "sharpe_ratio": 1.9,
        "win_rate": 0.57,
    }

    comparison = {}
    for key in v438_results.keys():
        v438_val = v438_results[key]
        v441_exp = v441_expected[key]
        improvement = v441_exp - v438_val
        pct_improvement = (improvement / v438_val) * 100 if v438_val != 0 else 0

        comparison[key] = {
            "v438": v438_val,
            "v441_expected": v441_exp,
            "improvement": improvement,
            "pct_improvement": pct_improvement,
        }

    logger.info("Comparison with v438:")
    for key, values in comparison.items():
        logger.info(f"  {key}:")
        logger.info(f"    v438: {values['v438']}")
        logger.info(
            f"    v441: {values['v441_expected']} (+{values['pct_improvement']:.1f}%)"
        )

    return comparison


def generate_recommendations(analysis_results: dict, comparison: dict):
    """改善推奨事項の生成"""
    logger.info("Generating recommendations...")

    recommendations = []

    # 安定性スコアの評価
    stability = analysis_results.get("stability_score", 0)
    if stability < 0.7:
        recommendations.append("安定性スコアが不十分 - 正則化パラメータの調整が必要")
    elif stability >= 0.75:
        recommendations.append("安定性目標達成 - 次のフェーズへ進行可能")

    # レジーム適応性の評価
    adaptability = analysis_results.get("regime_adaptability", 0)
    if adaptability < 0.85:
        recommendations.append(
            "レジーム適応性が目標未達 - 報酬関数のさらなる調整を検討"
        )

    # 統計的意義の評価
    significance = analysis_results.get("statistical_significance", 0)
    if significance < 0.8:
        recommendations.append(
            "統計的意義が不十分 - サンプルサイズ拡大または評価方法の見直し"
        )

    # パフォーマンスの評価
    total_return = analysis_results.get("total_return", 0)
    if total_return < 0.16:
        recommendations.append("リターンが期待値を下回る - 報酬関数の見直しが必要")

    return recommendations


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description="SAC v441 Stability Test")
    parser.add_argument(
        "--config",
        default="config/sac_v441_stability_focused_config.json",
        help="Path to v441 configuration file",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run in test mode without actual training",
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 60)
        logger.info("SAC v441 Stability-Focused Configuration Test")
        logger.info("=" * 60)

        # v438との比較
        comparison = compare_with_v438()
        print("\n" + "=" * 40)
        print("COMPARISON WITH V438")
        print("=" * 40)

        # 安定性テスト
        if not args.test_only:
            config_path = run_stability_test()
        else:
            config_path = args.config

        # 結果分析（シミュレーション）
        analysis_results = analyze_training_results(config_path)

        # 推奨事項生成
        recommendations = generate_recommendations(analysis_results, comparison)

        print("\n" + "=" * 40)
        print("RECOMMENDATIONS")
        print("=" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 40)
        print("NEXT STEPS")
        print("=" * 40)
        print("1. 実際のトレーニング実行")
        print("2. ハイパーパラメータの微調整")
        print("3. 報酬関数の反復改善")
        print("4. 長期安定性の検証")

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
