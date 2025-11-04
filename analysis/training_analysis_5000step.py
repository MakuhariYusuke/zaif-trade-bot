#!/usr/bin/env python3
"""
5000ステップ学習結果分析スクリプト
課題発見のための包括的な分析を実行
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.correlation import CorrelationAnalyzer
from ztb.analysis.market_regime_classifier import MarketRegimeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingAnalysis:
    """5000ステップ学習結果の分析クラス"""

    def __init__(self, stats_path: str, model_path: str = None):
        self.stats_path = Path(stats_path)
        self.model_path = Path(model_path) if model_path else None
        self.stats = self._load_stats()
        self.correlation_analyzer = CorrelationAnalyzer()

    def _load_stats(self) -> Dict[str, Any]:
        """トレーニング統計を読み込み"""
        try:
            with open(self.stats_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Stats file not found: {self.stats_path}")
            return {}

    def analyze_action_distribution(self) -> Dict[str, Any]:
        """アクション分布の分析"""
        logger.info("Analyzing action distribution...")

        if "action_distribution" not in self.stats:
            return {"error": "No action distribution data found"}

        action_dist = self.stats["action_distribution"]

        # SELLアクションの割合を計算
        total_actions = sum(action_dist.values())
        sell_percentage = (action_dist.get("sell", 0) / total_actions) * 100 if total_actions > 0 else 0

        analysis = {
            "total_actions": total_actions,
            "action_distribution": action_dist,
            "sell_percentage": sell_percentage,
            "sell_bias_detected": sell_percentage < 15.0,  # 目標15-20%未満の場合
            "recommendations": []
        }

        # SELLバイアス判定と推奨事項
        if analysis["sell_bias_detected"]:
            analysis["recommendations"].append(
                f"SELLアクションが{sell_percentage:.1f}%と少なすぎます（目標: 15-20%）"
            )
            analysis["recommendations"].append(
                "SELLシグナル特徴量の強化または報酬関数の調整を検討"
            )

        return analysis

    def analyze_reward_trends(self) -> Dict[str, Any]:
        """報酬トレンドの分析"""
        logger.info("Analyzing reward trends...")

        if "episode_rewards" not in self.stats:
            return {"error": "No episode rewards data found"}

        rewards = self.stats["episode_rewards"]

        if len(rewards) < 2:
            return {"error": "Insufficient reward data for trend analysis"}

        # 報酬のトレンド分析
        recent_rewards = rewards[-10:]  # 直近10エピソード
        avg_recent = np.mean(recent_rewards)
        avg_overall = np.mean(rewards)

        # 学習の安定性チェック
        reward_std = np.std(rewards)
        reward_cv = reward_std / abs(avg_overall) if avg_overall != 0 else float('inf')

        analysis = {
            "total_episodes": len(rewards),
            "average_reward_recent": avg_recent,
            "average_reward_overall": avg_overall,
            "reward_volatility": reward_std,
            "coefficient_of_variation": reward_cv,
            "learning_stability": "stable" if reward_cv < 0.5 else "unstable",
            "recommendations": []
        }

        # 安定性判定
        if analysis["learning_stability"] == "unstable":
            analysis["recommendations"].append(
                f"学習が不安定です（変動係数: {reward_cv:.2f}）"
            )
            analysis["recommendations"].append(
                "学習率の調整または報酬関数の安定化を検討"
            )

        return analysis

    def analyze_regime_adaptation(self) -> Dict[str, Any]:
        """市場レジーム適応の分析"""
        logger.info("Analyzing regime adaptation...")

        if "regime_stats" not in self.stats:
            return {"error": "No regime statistics found"}

        regime_stats = self.stats["regime_stats"]

        analysis = {
            "regime_counts": regime_stats.get("regime_counts", {}),
            "regime_rewards": regime_stats.get("regime_rewards", {}),
            "regime_actions": regime_stats.get("regime_actions", {}),
            "total_regimes": len(regime_stats.get("regime_counts", {})),
            "recommendations": []
        }

        # 16レジームが全て使用されているかチェック
        expected_regimes = 16
        actual_regimes = analysis["total_regimes"]

        if actual_regimes < expected_regimes:
            analysis["recommendations"].append(
                f"使用されたレジームが{actual_regimes}/{expected_regimes}と不足しています"
            )
            analysis["recommendations"].append(
                "レジーム分類の閾値調整または特徴量の改善を検討"
            )

        # SELL特化レジームの使用状況
        sell_regimes = ["SELL_BREAKDOWN", "SELL_DIVERGENCE", "SELL_MOMENTUM_WEAK", "SELL_VOLUME_SURGE"]
        sell_regime_usage = sum(1 for regime in sell_regimes if regime in analysis["regime_counts"])

        if sell_regime_usage == 0:
            analysis["recommendations"].append(
                "SELL特化レジームが全く使用されていません"
            )
            analysis["recommendations"].append(
                "SELLレジームの定義条件を見直す"
            )

        return analysis

    def analyze_overfitting_indicators(self) -> Dict[str, Any]:
        """過学習指標の分析"""
        logger.info("Analyzing overfitting indicators...")

        analysis = {
            "early_stopping_triggered": self.stats.get("early_stopping_triggered", False),
            "final_training_loss": self.stats.get("final_training_loss"),
            "final_validation_loss": self.stats.get("final_validation_loss"),
            "recommendations": []
        }

        # 早期停止の発生チェック
        if analysis["early_stopping_triggered"]:
            analysis["recommendations"].append(
                "早期停止が作動しました - 過学習の兆候"
            )
            analysis["recommendations"].append(
                "特徴量削減または正則化パラメータの調整を検討"
            )

        # 損失のギャップチェック（利用可能な場合）
        train_loss = analysis["final_training_loss"]
        val_loss = analysis["final_validation_loss"]

        if train_loss is not None and val_loss is not None:
            loss_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
            if loss_ratio > 2.0:
                analysis["recommendations"].append(
                    f"訓練/検証損失比が{loss_ratio:.2f}と高い - 過学習の兆候"
                )

        return analysis

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """包括的な分析レポート生成"""
        logger.info("Generating comprehensive analysis report...")

        report = {
            "training_summary": {
                "total_timesteps": self.stats.get("total_timesteps", 0),
                "total_episodes": self.stats.get("total_episodes", 0),
                "training_time": self.stats.get("training_time", 0),
                "steps_per_second": self.stats.get("steps_per_second", 0)
            },
            "action_distribution_analysis": self.analyze_action_distribution(),
            "reward_trends_analysis": self.analyze_reward_trends(),
            "regime_adaptation_analysis": self.analyze_regime_adaptation(),
            "overfitting_analysis": self.analyze_overfitting_indicators(),
            "critical_issues": [],
            "recommendations": []
        }

        # クリティカルな課題の抽出
        for analysis in [report["action_distribution_analysis"],
                        report["reward_trends_analysis"],
                        report["regime_adaptation_analysis"],
                        report["overfitting_analysis"]]:
            if "error" in analysis:
                report["critical_issues"].append(analysis["error"])
            if "recommendations" in analysis:
                report["recommendations"].extend(analysis["recommendations"])

        # 優先度の高い課題を特定
        if report["action_distribution_analysis"].get("sell_bias_detected", False):
            report["critical_issues"].append("SELLバイアス問題が継続")

        if report["reward_trends_analysis"].get("learning_stability") == "unstable":
            report["critical_issues"].append("学習の不安定性")

        if report["overfitting_analysis"].get("early_stopping_triggered", False):
            report["critical_issues"].append("過学習検出")

        return report

    def save_report(self, output_path: str):
        """分析レポートを保存"""
        report = self.generate_comprehensive_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis report saved to {output_file}")

        # コンソールに主要な課題を出力
        self._print_key_findings(report)

    def _print_key_findings(self, report: Dict[str, Any]):
        """主要な課題をコンソールに出力"""
        print("\n" + "="*60)
        print("5000ステップ学習分析結果 - 主要課題")
        print("="*60)

        print(f"\nトレーニング概要:")
        summary = report["training_summary"]
        print(f"  - 総ステップ数: {summary['total_timesteps']:,}")
        print(f"  - 総エピソード数: {summary['total_episodes']}")
        print(".2f"
        if summary["steps_per_second"] > 0:
            print(".1f"
        print(f"\nクリティカルな課題:")
        for issue in report["critical_issues"]:
            print(f"  ❌ {issue}")

        print(f"\n推奨事項:")
        for rec in report["recommendations"]:
            print(f"  💡 {rec}")

        print("\n" + "="*60)


def main():
    """メイン分析実行関数"""
    stats_path = "analysis/training_stats_5000step.json"
    output_path = "analysis/training_analysis_5000step_report.json"

    analyzer = TrainingAnalysis(stats_path)
    analyzer.save_report(output_path)


if __name__ == "__main__":
    main()