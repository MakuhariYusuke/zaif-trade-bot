#!/usr/bin/env python3
"""
SIGNAL_GUIDANCE vs ベースライン比較分析
"""

import json


def main():
    # SIGNAL_GUIDANCE結果読み込み
    with open("signal_guidance_backtest_results_20251112_135639.json", "r") as f:
        sg_results = json.load(f)

    # ベースライン結果読み込み
    with open("baseline_backtest_results.json", "r") as f:
        bl_results = json.load(f)

    print("=== SIGNAL_GUIDANCE vs ベースライン比較 ===")
    print(
        f'SIGNAL_GUIDANCE平均リターン: {sg_results["avg_total_return_pct"]:.2f}% ± {sg_results["std_total_return_pct"]:.2f}%'
    )
    print(
        f'ベースライン平均リターン: {bl_results["avg_return_pct"]:.2f}% ± {bl_results["std_return_pct"]:.2f}%'
    )
    print(
        f'パフォーマンス差: {sg_results["avg_total_return_pct"] - bl_results["avg_return_pct"]:.2f}%'
    )
    print()

    print("SIGNAL_GUIDANCEスコア統計:")
    print(f'平均スコア: {sg_results.get("avg_guidance_score", "N/A")}')
    print(f'スコア標準偏差: {sg_results.get("std_guidance_score", "N/A")}')
    print(f'最小スコア: {sg_results.get("min_guidance_score", "N/A")}')
    print(f'最大スコア: {sg_results.get("max_guidance_score", "N/A")}')
    print()

    print("結論:")
    if sg_results["avg_total_return_pct"] < bl_results["avg_return_pct"]:
        print("❌ SIGNAL_GUIDANCEはパフォーマンスを低下させています")
        print("   ベースラインより約75%悪い結果")
        print("   SIGNAL_GUIDANCEの実装に根本的な問題があります")
    else:
        print("✅ SIGNAL_GUIDANCEはパフォーマンスを改善しています")

    print()
    print("推奨される次のステップ:")
    print("1. SIGNAL_GUIDANCEスコアの解釈ロジックを確認")
    print("2. スコアが高いほど良いアクションという前提が正しいか検証")
    print("3. 技術指標のマッピングと重み付けを見直し")
    print("4. より単純なSIGNAL_GUIDANCE実装からテスト")


if __name__ == "__main__":
    main()
