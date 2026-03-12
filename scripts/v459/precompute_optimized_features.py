"""
特徴量事前計算スクリプト（v459最適化）

既存実装を最大活用:
- FeatureRegistry.compute_features_batch()
- FeatureRegistry.get_optimized_feature_set()
- parquet_io.write_parquet()
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ztb.features.core.registry import FeatureRegistry
from ztb.cache.parquet_io import write_parquet
from ztb.utils.data_utils import load_csv_data_cached


def precompute_optimized_features(
    data_file: str = 'data/btc_jpy_1m_v451.csv',
    output_file: str = 'data/btc_jpy_1m_v451_optimized_features.parquet',
    correlation_threshold: float = 0.95,
    analysis_file: str = None,
):
    """
    相関削減済み特徴量の事前計算とParquet保存
    
    Args:
        data_file: 入力CSVファイルパス
        output_file: 出力Parquetファイルパス
        correlation_threshold: 相関削減の閾値（0.95推奨）
        analysis_file: 特徴分析ファイル（Noneの場合は全特徴）
    """
    print(f"=== 特徴量事前計算開始 ===")
    print(f"入力: {data_file}")
    print(f"出力: {output_file}")
    
    # 1. 生データ読み込み（既存キャッシュ使用）
    print("\n[1/4] データ読み込み中...")
    try:
        df = load_csv_data_cached(data_file)
        print(f"✅ データ読み込み完了: {len(df)}行")
    except Exception as e:
        print(f"❌ データ読み込み失敗: {e}")
        # フォールバック: 直接CSV読み込み
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
        print(f"⚠️ フォールバック: 直接CSV読み込み {len(df)}行")
    
    # 2. 最適化済み特徴セット取得
    print(f"\n[2/4] 最適化済み特徴セット取得中...")
    print(f"  相関閾値: {correlation_threshold}")
    if analysis_file:
        print(f"  分析ファイル: {analysis_file}")
    
    # ⚠️ 注意: analysis_fileがない場合は全特徴量を返す
    # 事前に特徴分析を実行推奨: scripts/analyze_features.py
    optimized_features = FeatureRegistry.get_optimized_feature_set(
        correlation_threshold=correlation_threshold,
        analysis_file=analysis_file
    )
    print(f"✅ 特徴数: {len(optimized_features)}")
    
    # 3. 特徴量計算（バッチ処理）
    print(f"\n[3/4] 特徴量計算中...")
    df_features, timings = FeatureRegistry.compute_features_batch(
        df,
        feature_names=optimized_features,
        verbose=True,
        return_timing=True
    )
    
    # 元データ（OHLCV + timestamp）と特徴量を結合
    # ⚠️ HeavyTradingEnvは必須列を要求: timestamp, open, high, low, close, volume
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_ohlcv = df[required_cols].copy()
    df_with_features = pd.concat([df_ohlcv, df_features], axis=1)
    
    # タイミング情報表示
    total_time = sum(timings.values())
    print(f"\n✅ 特徴量計算完了:")
    print(f"  総時間: {total_time:.2f}秒")
    print(f"  特徴数: {len(optimized_features)}")
    print(f"  データ形状: {df_with_features.shape}")
    
    # 上位5件の時間を表示
    if timings:
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  時間上位5特徴:")
        for feat, time in sorted_timings:
            print(f"    {feat}: {time:.3f}秒")
    
    # 4. Parquet保存（圧縮 + 30分TTLキャッシュ）
    print(f"\n[4/4] Parquet保存中: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    write_parquet(
        df_with_features,
        output_path,
        config={'parquet': {'compression': 'snappy'}}
    )
    
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"✅ 保存完了: {file_size_mb:.2f}MB")
    
    # 必須列の確認
    print(f"✅ 必須列確認OK: OHLCV + timestamp ({len(required_cols)}列)")
    print(f"   + 特徴量: {len(optimized_features)}列")
    print(f"   合計: {len(df_with_features.columns)}列")
    
    print(f"\n=== 特徴量事前計算完了 ===")
    print(f"次のステップ:")
    print(f"  1. ABRewardExperimentで読み込み設定")
    print(f"  2. 12実験実行: python scripts/v459/run_ab_reward_experiments.py")
    print(f"  3. 時間測定: 特徴生成時間 431秒 → <35秒 期待")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='特徴量事前計算スクリプト')
    parser.add_argument(
        '--data-file',
        default='data/btc_jpy_1m_v451.csv',
        help='入力CSVファイル'
    )
    parser.add_argument(
        '--output-file',
        default='data/btc_jpy_1m_v451_optimized_features.parquet',
        help='出力Parquetファイル'
    )
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.95,
        help='相関削減閾値（0.95推奨）'
    )
    parser.add_argument(
        '--analysis-file',
        default=None,
        help='特徴分析ファイル（省略時は全特徴）'
    )
    
    args = parser.parse_args()
    
    precompute_optimized_features(
        data_file=args.data_file,
        output_file=args.output_file,
        correlation_threshold=args.correlation_threshold,
        analysis_file=args.analysis_file
    )
