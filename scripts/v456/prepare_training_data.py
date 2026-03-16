#!/usr/bin/env python3
"""
大規模なBTC/JPYデータセットを使用してバックテストデータを準備

既存の拡張データセットを利用し、必要に応じて最新データで補完
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def prepare_training_data():
    """バックテスト/ペーパートレード用にデータを準備"""
    
    print(f"📊 バックテストデータ準備スクリプト")
    print(f"{'='*70}")
    
    # 利用可能なファイルを確認
    data_dir = project_root / "data"
    
    candidate_files = [
        ("btc_jpy_1m_merged.csv", 2024),           # 最大級の統合ファイル
        ("btc_jpy_extended_dataset.csv", 2024),    # 大規模データセット
        ("btc_jpy_1m_v454.csv", 2026),             # v454用
    ]
    
    print(f"\n📁 利用可能なデータファイル:")
    
    selected_file = None
    for filename, expected_year in candidate_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f}MB)")
            
            # 最初に見つかった大きなファイルを選択
            if size_mb > 100 and selected_file is None:
                selected_file = filename
                print(f"    → 使用するファイル（最大規模）")
    
    if not selected_file:
        print(f"  ⚠️  大規模ファイルが見つかりません")
        print(f"  最小限のファイルで対応します")
        selected_file = "btc_jpy_1m_v451.csv"
    
    # データを読み込み
    print(f"\n📖 データを読み込み中: {selected_file}")
    filepath = data_dir / selected_file
    
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"  ✅ 読み込み完了")
        print(f"  行数: {len(df):,}")
        print(f"  期間: {df.index.min()} ～ {df.index.max()}")
        print(f"  日数: {(df.index.max() - df.index.min()).days} 日")
    except Exception as e:
        print(f"  ❌ 読み込みエラー: {e}")
        return False
    
    # 基本的なOHLCV列のみを抽出
    essential_cols = ['open', 'high', 'low', 'close', 'volume']
    available_cols = [col for col in essential_cols if col in df.columns]
    
    if not available_cols:
        print(f"  ❌ OHLCVカラムが見つかりません")
        return False
    
    df_clean = df[available_cols].copy()
    
    # 欠損値を処理
    df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
    
    # v456用のメインファイルとして保存
    output_file = data_dir / "btc_jpy_training_data.csv"
    print(f"\n💾 訓練用データを保存中: {output_file.name}")
    
    df_clean.to_csv(output_file)
    
    print(f"  ✅ 保存完了")
    print(f"  ファイル: {output_file.name}")
    print(f"  サイズ: {output_file.stat().st_size / (1024*1024):.1f}MB")
    print(f"  行数: {len(df_clean):,}")
    print(f"  期間: {df_clean.index.min()} ～ {df_clean.index.max()}")
    
    # バックテスト用のシンボリックリンクも作成
    backtest_file = data_dir / "btc_jpy_backtest_data.csv"
    print(f"\n🔗 バックテスト用ショートカットを作成: {backtest_file.name}")
    
    try:
        # Windows での代替: コピー
        df_clean.to_csv(backtest_file)
        print(f"  ✅ コピー完了")
    except Exception as e:
        print(f"  ⚠️  {e}")
    
    print(f"\n{'='*70}")
    print(f"✅ データ準備完了")
    print(f"\n📝 次のファイルを使用してください:")
    print(f"  バックテスト: {output_file.name}")
    print(f"  ペーパートレード: {output_file.name}")
    print(f"\n💡 注意:")
    print(f"  本番運用前には実際のマーケットデータに差し替えてください")
    
    return True


if __name__ == "__main__":
    success = prepare_training_data()
    sys.exit(0 if success else 1)
