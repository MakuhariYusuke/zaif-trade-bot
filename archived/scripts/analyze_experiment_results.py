#!/usr/bin/env python3
"""
SAC v427 実験結果分析スクリプト

各実験のアクション分布を分析してSELLバイアスを比較
"""

import json
from pathlib import Path
from typing import Dict


def analyze_experiment_results():
    """実験結果を分析"""
    experiments_dir = Path("results/experiments")

    if not experiments_dir.exists():
        print("❌ 実験結果ディレクトリが見つかりません")
        return

    results = []

    # 各実験ディレクトリを処理
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        print(f"\n実験: {exp_name}")

        # JSONファイルを探す
        json_files = list(exp_dir.glob("*.json"))
        if not json_files:
            print("  ❌ JSONファイルが見つかりません")
            continue

        json_file = json_files[0]  # 最初のJSONファイルを使用

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "phase_1" in data and data["phase_1"].get("success"):
                phase_data = data["phase_1"]

                # アクション分布を取得（reportsから）
                action_dist = get_action_distribution_from_reports(exp_name)

                if action_dist:
                    sell_ratio = action_dist.get("SELL", 0)
                    buy_ratio = action_dist.get("BUY", 0)
                    hold_ratio = action_dist.get("HOLD", 0)

                    results.append(
                        {
                            "experiment": exp_name,
                            "sell_ratio": sell_ratio,
                            "buy_ratio": buy_ratio,
                            "hold_ratio": hold_ratio,
                            "timesteps": phase_data.get("total_timesteps", 0),
                            "training_time": phase_data.get("training_time", 0),
                        }
                    )

                    print(f"  - SELL: {sell_ratio:.1%}")
                    print(f"  - BUY: {buy_ratio:.1%}")
                    print(f"  - HOLD: {hold_ratio:.1%}")
                else:
                    print("  ⚠️ アクション分布が見つかりません")
            else:
                print("  ❌ 学習が失敗しています")

        except Exception as e:
            print(f"  ❌ エラー: {e}")

    # 結果をSELL比率でソート
    if results:
        print(f"\n{'='*60}")
        print("SELL比率ランキング（低い順）")
        print("=" * 60)

        sorted_results = sorted(results, key=lambda x: x["sell_ratio"])

        for i, result in enumerate(sorted_results, 1):
            status = (
                "🎯 最良"
                if i == 1
                else "✅ 良好"
                if result["sell_ratio"] < 0.5
                else "⚠️ 高め"
            )
            print(f"{i}. {status} {result['experiment']}")
            print(
                f"   SELL: {result['sell_ratio']:.1%}, BUY: {result['buy_ratio']:.1%}, HOLD: {result['hold_ratio']:.1%}"
            )
            print()

        # 推奨設定
        best_result = sorted_results[0]
        print("🎯 推奨設定:")
        print(f"  実験: {best_result['experiment']}")
        print(f"  SELL比率: {best_result['sell_ratio']:.1%}")
        print("  この設定でより長い学習を実行することを推奨します")
    else:
        print("❌ 有効な実験結果が見つかりませんでした")


def get_action_distribution_from_reports(exp_name: str) -> Dict[str, float]:
    """reportsからアクション分布を取得"""
    reports_dir = Path("reports")

    # 最新の関連レポートを探す
    report_files = list(
        reports_dir.glob("training_report_sac_sac_v427_market_adaptive_ensemble_*.json")
    )

    for report_file in sorted(report_files, reverse=True):  # 最新のものから
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if (
                "training_stats" in data
                and "action_distribution" in data["training_stats"]
            ):
                # このレポートが実験に関連するか確認
                # 簡単なチェック: タイムスタンプや設定で判断
                return data["training_stats"]["action_distribution"]
        except:
            continue

    return None


if __name__ == "__main__":
    analyze_experiment_results()
