#!/usr/bin/env python3
"""
BTCデータ拡張デモスクリプト
過去データの拡張とバイアス軽減を実演
"""

import sys
from pathlib import Path

# Add ztb to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ztb"))

from ztb.analysis.analyze_backtest import BacktestAnalyzer
from ztb.data.btc_data_augmentation import BTCDataAugmentor


def main():
    """メイン実行関数"""
    print("=== BTCデータ拡張デモ ===")

    # 元のデータファイル
    base_data_path = "data/btc_jpy_featured_dataset.csv"

    if not Path(base_data_path).exists():
        print(f"エラー: {base_data_path}が見つかりません")
        return

    print(f"元のデータ分析: {base_data_path}")

    # データ拡張ツールの初期化
    augmentor = BTCDataAugmentor(base_data_path)

    # バイアス分析
    print("\n=== 元データのバイアス分析 ===")
    bias_analysis = augmentor.analyze_data_bias()
    print(
        f"データ期間: {bias_analysis['data_start']} から {bias_analysis['data_end']} ({bias_analysis['time_range_days']}日)"
    )
    print(".2f")
    print(".2f")
    print(".2f")
    print("レジーム分布:")
    for regime, pct in bias_analysis["regime_distribution"].items():
        print(".1f")

    # データ拡張の実行
    print("\n=== データ拡張実行 ===")
    print("2年分の過去データを追加生成...")
    extended_data = augmentor.extend_historical_data(years_back=2)

    print("多様な市場条件データを追加...")
    diverse_data = augmentor.add_diverse_market_conditions(target_samples=50000)

    # 拡張データの保存
    extended_path = "data/btc_jpy_extended_dataset.csv"
    diverse_path = "data/btc_jpy_diverse_dataset.csv"

    print(f"\n拡張データを保存: {extended_path}")
    augmentor.save_augmented_data(extended_data, extended_path)

    print(f"多様性データを保存: {diverse_path}")
    augmentor.save_augmented_data(diverse_data, diverse_path)

    # 拡張データの分析
    print("\n=== 拡張データの分析 ===")

    # 多様性データのバイアス分析
    diverse_augmentor = BTCDataAugmentor(diverse_path)
    diverse_bias = diverse_augmentor.analyze_data_bias()

    print("多様性データ:")
    print(f"データ期間: {diverse_bias['time_range_days']}日")
    print(".2f")
    print(".2f")
    print("レジーム分布:")
    for regime, pct in diverse_bias["regime_distribution"].items():
        print(".1f")

    # バックテスト分析の実行（モックデータ使用）
    print("\n=== バックテスト分析 ===")
    try:
        # 多様性データでの分析
        analyzer = BacktestAnalyzer("test_results.json")
        report = analyzer.generate_comprehensive_report()

        print("分析レポート生成完了")
        print("レポートの最初の200文字:")
        print(report[:200] + "...")

        # バイアス分析セクションがあるか確認
        if "データバイアス分析" in report:
            print("✅ バイアス分析機能が正常に動作しています")
        else:
            print("⚠️ バイアス分析セクションが見つかりません")

        if "ロバストネス分析" in report:
            print("✅ ロバストネス分析機能が正常に動作しています")
        else:
            print("⚠️ ロバストネス分析セクションが見つかりません")

    except Exception as e:
        print(f"バックテスト分析エラー: {e}")

    print("\n=== 完了 ===")
    print("データ拡張とバイアス軽減機能が実装されました")
    print(f"拡張データ: {extended_path}")
    print(f"多様性データ: {diverse_path}")


if __name__ == "__main__":
    main()
