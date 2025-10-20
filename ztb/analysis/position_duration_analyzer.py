#!/usr/bin/env python3
"""
Trading Position Duration Analysis Tool
取引ポジション継続時間分析ツール

売ってから買いの平均時間、買ってから売りの平均時間を分析します。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PositionDurationAnalyzer:
    """ポジション継続時間分析クラス"""

    def __init__(self, backtest_results_path: str):
        self.results_path = Path(backtest_results_path)
        self.data = self._load_data()
        # Try different possible keys for actions
        self.actions = np.array(
            self.data.get("raw_data", {}).get("actions", [])
            or self.data.get("action_history", [])
            or self.data.get("actions", [])
        )
        self.portfolio_values = np.array(
            self.data.get("raw_data", {}).get("portfolio_values", [])
            or self.data.get("portfolio_history", [])
        )

    def _load_data(self) -> Dict[str, Any]:
        """バックテスト結果を読み込み"""
        try:
            with open(self.results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {self.results_path}: {e}")
            return {}

    def analyze_position_durations(self) -> Dict[str, Any]:
        """
        ポジション継続時間を分析

        Returns:
            分析結果
        """
        if len(self.actions) == 0:
            return {"error": "No action data available"}

        # アクションの変化点を検出
        transitions = self._find_position_transitions()

        # 各タイプの継続時間を計算
        sell_to_buy_durations = []
        buy_to_sell_durations = []
        hold_durations = []

        current_position = None
        position_start = 0

        for i, action in enumerate(self.actions):
            if current_position is None:
                # 最初のポジションを設定
                current_position = action
                position_start = i
                continue

            if action != current_position:
                # ポジションが変わった
                duration = i - position_start

                if current_position == -1 and action == 1:  # SELL -> BUY
                    sell_to_buy_durations.append(duration)
                elif current_position == 1 and action == -1:  # BUY -> SELL
                    buy_to_sell_durations.append(duration)
                elif current_position == 0:  # HOLD終了
                    hold_durations.append(duration)

                current_position = action
                position_start = i

        # 最後のポジションの継続時間を追加
        if current_position is not None:
            duration = len(self.actions) - position_start
            if current_position == 0:
                hold_durations.append(duration)

        # 統計計算
        analysis = {
            "sell_to_buy": self._calculate_stats(sell_to_buy_durations),
            "buy_to_sell": self._calculate_stats(buy_to_sell_durations),
            "hold": self._calculate_stats(hold_durations),
            "summary": {
                "total_transitions": len(transitions),
                "total_sell_to_buy": len(sell_to_buy_durations),
                "total_buy_to_sell": len(buy_to_sell_durations),
                "total_hold_periods": len(hold_durations),
                "avg_position_changes_per_1000_steps": len(transitions)
                / max(len(self.actions), 1)
                * 1000,
            },
        }

        return analysis

    def _find_position_transitions(self) -> List[Tuple[int, int, int]]:
        """
        ポジションの変化点を検出

        Returns:
            [(step, from_action, to_action), ...]
        """
        transitions = []
        prev_action = None

        for i, action in enumerate(self.actions):
            if prev_action is not None and action != prev_action:
                transitions.append((i, prev_action, action))
            prev_action = action

        return transitions

    def _calculate_stats(self, durations: List[int]) -> Dict[str, Any]:
        """継続時間の統計を計算"""
        if not durations:
            return {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0, "std": 0}

        durations_array = np.array(durations)

        return {
            "count": len(durations),
            "mean": float(np.mean(durations_array)),
            "median": float(np.median(durations_array)),
            "min": int(np.min(durations_array)),
            "max": int(np.max(durations_array)),
            "std": float(np.std(durations_array)),
        }

    def analyze_action_sequences(self) -> Dict[str, Any]:
        """アクションシーケンスのパターンを分析"""
        if len(self.actions) == 0:
            return {"error": "No action data available"}

        # 連続するアクションのパターンを分析
        patterns = defaultdict(int)
        current_pattern = []
        pattern_length = 3  # 3アクションのパターン

        for action in self.actions:
            current_pattern.append(action)
            if len(current_pattern) > pattern_length:
                current_pattern.pop(0)

            if len(current_pattern) == pattern_length:
                pattern_key = tuple(current_pattern)
                patterns[pattern_key] += 1

        # 最も頻出するパターンを抽出
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "pattern_length": pattern_length,
            "total_patterns": len(patterns),
            "top_patterns": [
                {
                    "pattern": [int(x) for x in pattern],
                    "count": count,
                    "frequency": count / sum(patterns.values()),
                }
                for pattern, count in top_patterns
            ],
        }

    def generate_report(self) -> str:
        """分析レポートを生成"""
        position_analysis = self.analyze_position_durations()
        sequence_analysis = self.analyze_action_sequences()

        if "error" in position_analysis:
            return f"Error: {position_analysis['error']}"

        report = []
        report.append("=" * 80)
        report.append("TRADING POSITION DURATION ANALYSIS")
        report.append("=" * 80)

        # ポジション継続時間分析
        report.append("\nPOSITION DURATION ANALYSIS:")
        report.append("-" * 40)

        for position_type, stats in position_analysis.items():
            if position_type == "summary":
                continue
            if stats["count"] == 0:
                continue

            position_name = {
                "sell_to_buy": "SELL → BUY",
                "buy_to_sell": "BUY → SELL",
                "hold": "HOLD",
            }.get(position_type, position_type.upper())

            report.append(f"\n{position_name} Duration Statistics:")
            report.append(f"  Count: {stats['count']}")
            report.append(
                f"  Average: {stats['mean']:.1f} steps ({stats['mean']*5:.1f} minutes)"
            )
            report.append(
                f"  Median: {stats['median']:.1f} steps ({stats['median']*5:.1f} minutes)"
            )
            report.append(f"  Min: {stats['min']} steps ({stats['min']*5:.1f} minutes)")
            report.append(f"  Max: {stats['max']} steps ({stats['max']*5:.1f} minutes)")
            report.append(f"  Std Dev: {stats['std']:.2f} steps")

        # サマリー
        summary = position_analysis["summary"]
        report.append("\nSUMMARY STATISTICS:")
        report.append(f"  Total Position Transitions: {summary['total_transitions']}")
        report.append(f"  SELL→BUY Transitions: {summary['total_sell_to_buy']}")
        report.append(f"  BUY→SELL Transitions: {summary['total_buy_to_sell']}")
        report.append(f"  HOLD Periods: {summary['total_hold_periods']}")
        report.append(
            f"  Position Changes per 1000 steps: {summary['avg_position_changes_per_1000_steps']:.2f}"
        )

        # アクションシーケンス分析
        report.append("\nACTION SEQUENCE PATTERNS:")
        report.append("-" * 40)

        for pattern_info in sequence_analysis["top_patterns"][:5]:
            pattern_str = " → ".join(
                [
                    "BUY" if x == 1 else "SELL" if x == -1 else "HOLD"
                    for x in pattern_info["pattern"]
                ]
            )
            report.append(
                f"  {pattern_str}: {pattern_info['count']} times ({pattern_info['frequency']:.1%})"
            )

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print(
            "Usage: python position_duration_analyzer.py <backtest_results.json> [output.json]"
        )
        sys.exit(1)

    results_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    analyzer = PositionDurationAnalyzer(results_path)

    # 分析実行
    position_analysis = analyzer.analyze_position_durations()
    sequence_analysis = analyzer.analyze_action_sequences()

    # レポート表示
    report = analyzer.generate_report()
    print(report)

    # 結果保存
    if output_path:
        results = {
            "position_durations": position_analysis,
            "action_sequences": sequence_analysis,
            "report": report,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
