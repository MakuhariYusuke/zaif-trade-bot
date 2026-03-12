#!/usr/bin/env python3
"""
SIGNAL_GUIDANCE vs ベースライン比較分析
"""

from ztb.io.json_io import read_json_object
from ztb.utils.safety import safe_to_float


def main() -> int:
    sg_results = read_json_object("signal_guidance_backtest_results_20251112_135639.json")
    bl_results = read_json_object("baseline_backtest_results.json")

    sg_avg = safe_to_float(sg_results.get("avg_total_return_pct"), 0.0)
    sg_std = safe_to_float(sg_results.get("std_total_return_pct"), 0.0)
    bl_avg = safe_to_float(bl_results.get("avg_return_pct"), 0.0)
    bl_std = safe_to_float(bl_results.get("std_return_pct"), 0.0)

    print("=== SIGNAL_GUIDANCE vs ベースライン比較 ===")
    print(f"SIGNAL_GUIDANCE平均リターン: {sg_avg:.2f}% ± {sg_std:.2f}%")
    print(f"ベースライン平均リターン: {bl_avg:.2f}% ± {bl_std:.2f}%")
    print(f"パフォーマンス差: {sg_avg - bl_avg:.2f}%")
    print()

    print("SIGNAL_GUIDANCEスコア統計:")
    print(f"平均スコア: {sg_results.get('avg_guidance_score', 'N/A')}")
    print(f"スコア標準偏差: {sg_results.get('std_guidance_score', 'N/A')}")
    print(f"最小スコア: {sg_results.get('min_guidance_score', 'N/A')}")
    print(f"最大スコア: {sg_results.get('max_guidance_score', 'N/A')}")
    print()

    print("結論:")
    if sg_avg < bl_avg:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
