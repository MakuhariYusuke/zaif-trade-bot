#!/usr/bin/env python3
"""
Position Duration Comparison Analysis
ポジション継続時間比較分析

v424とv427のポジション継続時間を比較分析します。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_analysis_results(file_path: str) -> Dict[str, Any]:
    """分析結果を読み込み"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return {}


def compare_position_durations(
    v424_results: Dict[str, Any], v427_results: Dict[str, Any]
) -> Dict[str, Any]:
    """ポジション継続時間を比較"""

    comparison = {
        "sell_to_buy_comparison": {},
        "buy_to_sell_comparison": {},
        "hold_comparison": {},
        "trading_style_analysis": {},
        "improvement_suggestions": [],
    }

    # SELL → BUY の比較
    v424_sell_buy = v424_results.get("position_durations", {}).get("sell_to_buy", {})
    v427_sell_buy = v427_results.get("position_durations", {}).get("sell_to_buy", {})

    if v424_sell_buy.get("count", 0) > 0 and v427_sell_buy.get("count", 0) > 0:
        comparison["sell_to_buy_comparison"] = {
            "v424_avg_duration": v424_sell_buy.get("mean", 0),
            "v427_avg_duration": v427_sell_buy.get("mean", 0),
            "duration_ratio": v427_sell_buy.get("mean", 1)
            / max(v424_sell_buy.get("mean", 1), 0.1),
            "v424_count": v424_sell_buy.get("count", 0),
            "v427_count": v427_sell_buy.get("count", 0),
            "assessment": "improved"
            if v427_sell_buy.get("mean", 0) > v424_sell_buy.get("mean", 0)
            else "worsened",
        }

    # BUY → SELL の比較
    v424_buy_sell = v424_results.get("position_durations", {}).get("buy_to_sell", {})
    v427_buy_sell = v427_results.get("position_durations", {}).get("buy_to_sell", {})

    if v424_buy_sell.get("count", 0) > 0 and v427_buy_sell.get("count", 0) > 0:
        comparison["buy_to_sell_comparison"] = {
            "v424_avg_duration": v424_buy_sell.get("mean", 0),
            "v427_avg_duration": v427_buy_sell.get("mean", 0),
            "duration_ratio": v427_buy_sell.get("mean", 1)
            / max(v424_buy_sell.get("mean", 1), 0.1),
            "v424_count": v424_buy_sell.get("count", 0),
            "v427_count": v427_buy_sell.get("count", 0),
            "assessment": "improved"
            if v427_buy_sell.get("mean", 0) > v424_buy_sell.get("mean", 0)
            else "worsened",
        }

    # HOLD の比較
    v424_hold = v424_results.get("position_durations", {}).get("hold", {})
    v427_hold = v427_results.get("position_durations", {}).get("hold", {})

    comparison["hold_comparison"] = {
        "v424_avg_duration": v424_hold.get("mean", 0),
        "v427_avg_duration": v427_hold.get("mean", 0),
        "v424_count": v424_hold.get("count", 0),
        "v427_count": v427_hold.get("count", 0),
        "hold_ratio_v424": v424_hold.get("count", 0)
        / max(
            v424_results.get("position_durations", {})
            .get("summary", {})
            .get("total_transitions", 1),
            1,
        ),
        "hold_ratio_v427": v427_hold.get("count", 0)
        / max(
            v427_results.get("position_durations", {})
            .get("summary", {})
            .get("total_transitions", 1),
            1,
        ),
    }

    # 取引スタイル分析
    v424_transitions = (
        v424_results.get("position_durations", {})
        .get("summary", {})
        .get("total_transitions", 0)
    )
    v427_transitions = (
        v427_results.get("position_durations", {})
        .get("summary", {})
        .get("total_transitions", 0)
    )

    comparison["trading_style_analysis"] = {
        "v424_trading_frequency": v424_transitions / 5000,  # per step
        "v427_trading_frequency": v427_transitions / 5000,  # per step
        "frequency_ratio": (v427_transitions / 5000)
        / max((v424_transitions / 5000), 0.001),
        "v424_hold_dominance": comparison["hold_comparison"]["hold_ratio_v424"],
        "v427_hold_dominance": comparison["hold_comparison"]["hold_ratio_v427"],
        "trading_style_change": "more_aggressive"
        if comparison["hold_comparison"]["hold_ratio_v427"]
        < comparison["hold_comparison"]["hold_ratio_v424"]
        else "more_conservative",
    }

    # 改善提案
    suggestions = []

    if comparison["trading_style_analysis"]["frequency_ratio"] > 2:
        suggestions.append("過剰取引を抑制：ポジション変更頻度が大幅に増加")
    elif comparison["trading_style_analysis"]["frequency_ratio"] < 0.5:
        suggestions.append("取引活発化：ポジション変更頻度が大幅に減少")

    if comparison["hold_comparison"]["hold_ratio_v427"] < 0.1:
        suggestions.append("HOLD戦略の見直し：HOLD率が極端に低い")
    elif comparison["hold_comparison"]["hold_ratio_v427"] > 0.5:
        suggestions.append("取引機会の活用：HOLD率が高すぎる可能性")

    sell_buy_comp = comparison.get("sell_to_buy_comparison", {})
    if sell_buy_comp.get("duration_ratio", 1) < 0.5:
        suggestions.append("SELL→BUYタイミングの最適化：ポジション継続時間が短すぎる")

    buy_sell_comp = comparison.get("buy_to_sell_comparison", {})
    if buy_sell_comp.get("duration_ratio", 1) < 0.5:
        suggestions.append("BUY→SELLタイミングの最適化：ポジション継続時間が短すぎる")

    comparison["improvement_suggestions"] = suggestions

    return comparison


def generate_comparison_report(
    v424_results: Dict[str, Any],
    v427_results: Dict[str, Any],
    comparison: Dict[str, Any],
) -> str:
    """比較レポートを生成"""

    report = []
    report.append("=" * 80)
    report.append("POSITION DURATION COMPARISON: v424 vs v427")
    report.append("=" * 80)

    # SELL → BUY 比較
    if "sell_to_buy_comparison" in comparison and comparison["sell_to_buy_comparison"]:
        comp = comparison["sell_to_buy_comparison"]
        report.append("\nSELL → BUY Duration Comparison:")
        report.append(
            f"  v424 Average: {comp['v424_avg_duration']:.1f} steps ({comp['v424_avg_duration']*5:.1f} min)"
        )
        report.append(
            f"  v427 Average: {comp['v427_avg_duration']:.1f} steps ({comp['v427_avg_duration']*5:.1f} min)"
        )
        report.append(f"  Ratio (v427/v424): {comp['duration_ratio']:.2f}x")
        report.append(f"  Assessment: {comp['assessment']}")

    # BUY → SELL 比較
    if "buy_to_sell_comparison" in comparison and comparison["buy_to_sell_comparison"]:
        comp = comparison["buy_to_sell_comparison"]
        report.append("\nBUY → SELL Duration Comparison:")
        report.append(
            f"  v424 Average: {comp['v424_avg_duration']:.1f} steps ({comp['v424_avg_duration']*5:.1f} min)"
        )
        report.append(
            f"  v427 Average: {comp['v427_avg_duration']:.1f} steps ({comp['v427_avg_duration']*5:.1f} min)"
        )
        report.append(f"  Ratio (v427/v424): {comp['duration_ratio']:.2f}x")
        report.append(f"  Assessment: {comp['assessment']}")

    # HOLD 比較
    hold_comp = comparison["hold_comparison"]
    report.append("\nHOLD Duration Comparison:")
    report.append(
        f"  v424 HOLD Count: {hold_comp['v424_count']}, Ratio: {hold_comp['hold_ratio_v424']:.1%}"
    )
    report.append(
        f"  v427 HOLD Count: {hold_comp['v427_count']}, Ratio: {hold_comp['hold_ratio_v427']:.1%}"
    )

    # 取引スタイル分析
    style = comparison["trading_style_analysis"]
    report.append("\nTrading Style Analysis:")
    report.append(
        f"  Trading Frequency Ratio (v427/v424): {style['frequency_ratio']:.2f}x"
    )
    report.append(f"  Style Change: {style['trading_style_change']}")

    # 改善提案
    if comparison["improvement_suggestions"]:
        report.append("\nIMPROVEMENT SUGGESTIONS:")
        for i, suggestion in enumerate(comparison["improvement_suggestions"], 1):
            report.append(f"  {i}. {suggestion}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """メイン関数"""
    if len(sys.argv) < 3:
        print(
            "Usage: python position_duration_comparison.py <v424_analysis.json> <v427_analysis.json> [output.json]"
        )
        sys.exit(1)

    v424_path = sys.argv[1]
    v427_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    # 分析結果読み込み
    v424_results = load_analysis_results(v424_path)
    v427_results = load_analysis_results(v427_path)

    if not v424_results or not v427_results:
        print("Error: Could not load analysis results")
        sys.exit(1)

    # 比較分析
    comparison = compare_position_durations(v424_results, v427_results)

    # レポート生成
    report = generate_comparison_report(v424_results, v427_results, comparison)
    print(report)

    # 結果保存
    if output_path:
        results = {
            "v424_results": v424_results,
            "v427_results": v427_results,
            "comparison": comparison,
            "report": report,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
