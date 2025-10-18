#!/usr/bin/env python3
"""
Phase 1: データ基盤強化スクリプト
BTCDataAugmentorを使って多様な市場条件を追加したバランスデータセットを作成
"""

from ztb.data.btc_data_augmentation import BTCDataAugmentor
import pandas as pd

def main():
    print('=== Phase 1: データ基盤強化開始 ===')

    print('1. 既存データバイアス分析')
    augmentor = BTCDataAugmentor('data/btc_jpy_real_dataset.csv')
    bias_analysis = augmentor.analyze_data_bias()

    print('既存データバイアス分析結果:')
    for key, value in bias_analysis.items():
        print(f'  {key}: {value}')

    print()
    print('2. 多様な市場条件データ追加')
    balanced_data = augmentor.add_diverse_market_conditions(target_samples=50000)

    print(f'拡張後データセット: {len(balanced_data)} レコード')
    if 'market_regime' in balanced_data.columns:
        regime_dist = balanced_data['market_regime'].value_counts().to_dict()
        print(f'レジーム分布: {regime_dist}')
    else:
        print('レジーム情報: N/A')

    print()
    print('3. 拡張データ保存')
    output_path = 'data/btc_jpy_balanced_v426_dataset.csv'
    augmentor.save_augmented_data(balanced_data, output_path)
    print(f'保存完了: {output_path}')

    print()
    print('4. 新データセット検証')
    # 新しいデータセットの統計情報表示
    print(f'価格範囲: {balanced_data["close"].min():.0f} - {balanced_data["close"].max():.0f} JPY')
    print(f'期間: {balanced_data["timestamp"].min()} から {balanced_data["timestamp"].max()}')

    # リターンの統計
    if 'returns' in balanced_data.columns:
        returns = balanced_data['returns'].dropna()
        print(f'平均リターン: {returns.mean():.6f}')
        print(f'リターン標準偏差: {returns.std():.6f}')
        print(f'リターンレンジ: {returns.min():.6f} - {returns.max():.6f}')

if __name__ == "__main__":
    main()