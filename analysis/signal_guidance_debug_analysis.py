#!/usr/bin/env python3
"""
SIGNAL_GUIDANCEデバッグ分析スクリプト
スコアとアクションの相関関係を詳細に分析し、スコアリングロジックの逆転問題を調査
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalGuidanceDebugger:
    """SIGNAL_GUIDANCEデバッグ分析クラス"""

    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = None
        self.score_action_correlations = {}

    def load_results(self) -> Dict[str, Any]:
        """バックテスト結果を読み込み"""
        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                self.results = json.load(f)
            logger.info(f"結果ファイルを読み込みました: {self.results_file}")
            return self.results
        except Exception as e:
            logger.error(f"結果ファイル読み込みエラー: {e}")
            raise

    def analyze_score_distributions(self) -> Dict[str, Any]:
        """SIGNAL_GUIDANCEスコア分布を分析"""
        if not self.results:
            self.load_results()

        scores = []
        actions = []
        rewards = []

        for episode in self.results.get("results", []):
            for signal in episode.get("guidance_signals", []):
                if "guidance_score" in signal:
                    scores.append(signal["guidance_score"])
                    actions.append(signal.get("guidance_action", 0))
                    rewards.append(
                        signal.get("portfolio_value", 0)
                    )  # ポートフォリオ価値を報酬の代わりに使用

        if not scores:
            logger.warning("No SIGNAL_GUIDANCE scores found in results")
            return {
                "score_stats": {
                    "mean": 50.0,
                    "std": 0.0,
                    "min": 50.0,
                    "max": 50.0,
                    "median": 50.0,
                },
                "score_distribution": {"30-40": 0, "40-50": 0, "50-60": 0, "60-70": 0},
                "total_samples": 0,
            }

        scores = np.array(scores)
        actions = np.array(actions)
        rewards = np.array(rewards)

        analysis = {
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
            },
            "score_distribution": {
                "30-40": int(np.sum((scores >= 30) & (scores < 40))),
                "40-50": int(np.sum((scores >= 40) & (scores < 50))),
                "50-60": int(np.sum((scores >= 50) & (scores < 60))),
                "60-70": int(np.sum((scores >= 60) & (scores < 70))),
            },
            "total_samples": len(scores),
        }

        logger.info(f"スコア統計: {analysis['score_stats']}")
        logger.info(f"スコア分布: {analysis['score_distribution']}")

        return analysis

    def analyze_score_action_correlation(self) -> Dict[str, Any]:
        """スコアとアクションの相関関係を分析"""
        if not self.results:
            self.load_results()

        score_action_data = []

        for episode in self.results.get("results", []):
            for signal in episode.get("guidance_signals", []):
                if "guidance_score" in signal:
                    score_action_data.append(
                        {
                            "score": signal["guidance_score"],
                            "action": signal.get("guidance_action", 0),
                            "reward": signal.get(
                                "portfolio_value", 0
                            ),  # ポートフォリオ価値を使用
                            "next_reward": signal.get(
                                "portfolio_value", 0
                            ),  # 次ステップのポートフォリオ価値（簡易的に同じ値）
                        }
                    )

        if not score_action_data:
            logger.warning("No SIGNAL_GUIDANCE data found")
            return {
                "range_analysis": {},
                "action_analysis": {},
                "correlations": {
                    "score_reward": 0.0,
                    "score_next_reward": 0.0,
                    "action_reward": 0.0,
                },
                "total_samples": 0,
            }

        df = pd.DataFrame(score_action_data)

        # スコア範囲ごとの分析
        score_ranges = [(30, 40), (40, 50), (50, 60), (60, 70)]
        range_analysis = {}

        for min_score, max_score in score_ranges:
            range_data = df[(df["score"] >= min_score) & (df["score"] < max_score)]
            if len(range_data) > 0:
                range_analysis[f"{min_score}-{max_score}"] = {
                    "count": len(range_data),
                    "avg_reward": float(range_data["reward"].mean()),
                    "avg_next_reward": float(range_data["next_reward"].mean()),
                    "action_distribution": range_data["action"]
                    .value_counts()
                    .to_dict(),
                }

        # アクションごとのスコア分析
        action_analysis = {}
        for action in df["action"].unique():
            action_data = df[df["action"] == action]
            action_analysis[f"action_{action}"] = {
                "count": len(action_data),
                "avg_score": float(action_data["score"].mean()),
                "avg_reward": float(action_data["reward"].mean()),
                "score_std": float(action_data["score"].std()),
            }

        # 相関分析
        correlations = {
            "score_reward": float(df["score"].corr(df["reward"])),
            "score_next_reward": float(df["score"].corr(df["next_reward"])),
            "action_reward": float(df["action"].corr(df["reward"])),
        }

        analysis = {
            "range_analysis": range_analysis,
            "action_analysis": action_analysis,
            "correlations": correlations,
            "total_samples": len(df),
        }

        logger.info(f"スコア範囲分析: {range_analysis}")
        logger.info(f"相関関係: {correlations}")

        return analysis

    def analyze_score_effectiveness(self) -> Dict[str, Any]:
        """スコアの有効性を分析（高いスコアが良い結果をもたらすか）"""
        if not self.results:
            self.load_results()

        # スコアを四分位数で分割
        score_action_data = []
        for episode in self.results.get("results", []):
            for signal in episode.get("guidance_signals", []):
                if "guidance_score" in signal:
                    score_action_data.append(
                        {
                            "score": signal["guidance_score"],
                            "action": signal.get("guidance_action", 0),
                            "reward": signal.get("portfolio_value", 0),
                            "next_reward": signal.get("portfolio_value", 0),
                        }
                    )

        if not score_action_data:
            logger.warning("No SIGNAL_GUIDANCE data found for effectiveness analysis")
            return {
                "quartile_analysis": {},
                "effectiveness_indicators": {
                    "high_score_better": False,
                    "score_reward_trend": "unknown",
                },
            }

        df = pd.DataFrame(score_action_data)
        df = df.sort_values("score")

        # 四分位数で分割
        quartiles = df["score"].quantile([0.25, 0.5, 0.75]).values
        quartile_labels = ["Q1 (Low)", "Q2 (Medium)", "Q3 (High)", "Q4 (Highest)"]

        df["score_quartile"] = pd.cut(
            df["score"],
            bins=[
                df["score"].min(),
                quartiles[0],
                quartiles[1],
                quartiles[2],
                df["score"].max(),
            ],
            labels=quartile_labels,
            include_lowest=True,
        )

        quartile_analysis = {}
        for quartile in quartile_labels:
            quartile_data = df[df["score_quartile"] == quartile]
            if len(quartile_data) > 0:
                quartile_analysis[quartile] = {
                    "count": len(quartile_data),
                    "score_range": f"{quartile_data['score'].min():.1f}-{quartile_data['score'].max():.1f}",
                    "avg_score": float(quartile_data["score"].mean()),
                    "avg_reward": float(quartile_data["reward"].mean()),
                    "avg_next_reward": float(quartile_data["next_reward"].mean()),
                    "positive_reward_ratio": float(
                        (
                            quartile_data["reward"] > quartile_data["reward"].iloc[0]
                        ).mean()
                    ),  # 初期値より増加した割合
                }

        # スコアの有効性指標
        effectiveness = {
            "high_score_better": quartile_analysis.get("Q4 (Highest)", {}).get(
                "avg_reward", 0
            )
            > quartile_analysis.get("Q1 (Low)", {}).get("avg_reward", 0),
            "score_reward_trend": "increasing"
            if all(
                quartile_analysis.get(q1, {}).get("avg_reward", 0)
                <= quartile_analysis.get(q2, {}).get("avg_reward", 0)
                for q1, q2 in zip(quartile_labels[:-1], quartile_labels[1:])
            )
            else "not_increasing",
        }

        analysis = {
            "quartile_analysis": quartile_analysis,
            "effectiveness_indicators": effectiveness,
        }

        logger.info(f"スコア有効性分析: {effectiveness}")
        logger.info(f"四分位数分析: {quartile_analysis}")

        return analysis

    def generate_report(self) -> str:
        """分析レポートを生成"""
        if not self.results:
            self.load_results()

        score_dist = self.analyze_score_distributions()
        correlation = self.analyze_score_action_correlation()
        effectiveness = self.analyze_score_effectiveness()

        report = f"""
# SIGNAL_GUIDANCE デバッグ分析レポート

## 概要
- 分析対象ファイル: {self.results_file.name}
- 総サンプル数: {score_dist['total_samples']}

## スコア分布分析
- 平均スコア: {score_dist['score_stats']['mean']:.2f}
- スコア範囲: {score_dist['score_stats']['min']:.1f} - {score_dist['score_stats']['max']:.1f}
- スコア分布:
  - 30-40: {score_dist['score_distribution']['30-40']} サンプル
  - 40-50: {score_dist['score_distribution']['40-50']} サンプル
  - 50-60: {score_dist['score_distribution']['50-60']} サンプル
  - 60-70: {score_dist['score_distribution']['60-70']} サンプル

## 相関関係分析
- スコア vs 報酬: {correlation['correlations']['score_reward']:.3f}
- スコア vs 次ステップ報酬: {correlation['correlations']['score_next_reward']:.3f}
- アクション vs 報酬: {correlation['correlations']['action_reward']:.3f}

## スコア有効性分析
- 高いスコアが良い結果: {effectiveness['effectiveness_indicators']['high_score_better']}
- スコア-報酬トレンド: {effectiveness['effectiveness_indicators']['score_reward_trend']}

## スコア範囲別分析
"""

        for range_name, data in correlation["range_analysis"].items():
            report += f"""
### {range_name} スコア範囲
- サンプル数: {data['count']}
- 平均報酬: {data['avg_reward']:.4f}
- 平均次報酬: {data['avg_next_reward']:.4f}
- アクション分布: {data['action_distribution']}
"""

        report += """
## アクション別分析
"""
        for action_name, data in correlation["action_analysis"].items():
            report += f"""
### {action_name}
- サンプル数: {data['count']}
- 平均スコア: {data['avg_score']:.2f}
- 平均報酬: {data['avg_reward']:.4f}
"""

        report += """
## 四分位数分析
"""
        for quartile, data in effectiveness["quartile_analysis"].items():
            report += f"""
### {quartile}
- スコア範囲: {data['score_range']}
- サンプル数: {data['count']}
- 平均スコア: {data['avg_score']:.2f}
- 平均報酬: {data['avg_reward']:.4f}
- 平均次報酬: {data['avg_next_reward']:.4f}
- 正報酬比率: {data['positive_reward_ratio']:.2%}
"""

        # 問題点の特定
        issues = []
        if correlation["correlations"]["score_reward"] < 0:
            issues.append("スコアと報酬の負の相関 - 高いスコアが悪い結果を招く可能性")
        if not effectiveness["effectiveness_indicators"]["high_score_better"]:
            issues.append(
                "高いスコアが低いスコアよりも悪い結果 - スコアリングロジックの逆転"
            )
        if (
            effectiveness["effectiveness_indicators"]["score_reward_trend"]
            == "not_increasing"
        ):
            issues.append("スコア上昇による報酬向上が見られない")

        if issues:
            report += """
## 特定された問題点
"""
            for i, issue in enumerate(issues, 1):
                report += f"{i}. {issue}\n"

        report += """
## 推奨される次のステップ
1. スコアリングロジックの逆転を検討（高いスコア = 悪いアクション）
2. 各指標の重み付けを見直し
3. 閾値ベースのアプローチを検討（スコアが一定以上/以下でアクションを制限）
4. 基本的なSupertrend_Directionのみを使用した簡素化テスト
"""

        return report


def main():
    """メイン実行関数"""
    debugger = SignalGuidanceDebugger(
        "signal_guidance_backtest_results_20251112_135639.json"
    )

    try:
        report = debugger.generate_report()

        # レポートをファイルに保存
        with open(
            "signal_guidance_debug_analysis_report.md", "w", encoding="utf-8"
        ) as f:
            f.write(report)

        print(
            "デバッグ分析レポートを生成しました: signal_guidance_debug_analysis_report.md"
        )
        print("\n=== 分析サマリー ===")
        print(report.split("\n")[:20])  # 最初の20行を表示

    except Exception as e:
        logger.error(f"分析実行エラー: {e}")
        raise


if __name__ == "__main__":
    main()
