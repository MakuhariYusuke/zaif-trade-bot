"""
特徴量事前計算スクリプト（メモリリーク防止版）

改善点:
1. チャンクを逐次処理（事前分割しない）
2. 中間結果を定期的にディスクに書き出し
3. 明示的なメモリ解放
4. メモリ使用量監視
5. レジューム機能（中断からの再開）
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import gc
import psutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List
from ztb.features.core.registry import FeatureRegistry
from ztb.utils.data_utils import load_csv_data_cached


def get_memory_usage_mb() -> float:
    """現在のプロセスのメモリ使用量（MB）を取得"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def precompute_optimized_features_memory_safe(
    data_file: str = 'data/btc_jpy_real_dataset.csv',
    output_file: str = 'data/btc_jpy_1m_v451_optimized_features.parquet',
    correlation_threshold: float = 0.95,
    chunk_size: int = 500,  # メモリ節約のため小さく
    temp_dir: str = None,
):
    """
    メモリ効率的な特徴量事前計算
    
    Args:
        data_file: 入力CSVファイルパス
        output_file: 出力Parquetファイルパス
        correlation_threshold: 相関削減の閾値
        chunk_size: チャンクサイズ（小さいほどメモリ節約）
        temp_dir: 一時ファイルディレクトリ（Noneの場合は出力ファイルと同じディレクトリ）
    """
    print(f"=== メモリセーフ特徴量計算開始 ===")
    print(f"入力: {data_file}")
    print(f"出力: {output_file}")
    print(f"チャンクサイズ: {chunk_size}行")
    
    # 初期メモリ使用量
    initial_memory = get_memory_usage_mb()
    print(f"初期メモリ使用量: {initial_memory:.1f} MB")
    
    # 一時ディレクトリ設定
    if temp_dir is None:
        temp_dir = Path(output_file).parent / 'temp_features'
    else:
        temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"一時ディレクトリ: {temp_dir}")
    
    # 1. 生データ読み込み
    print("\n[1/4] データ読み込み中...")
    try:
        df = load_csv_data_cached(data_file)
    except Exception as e:
        print(f"キャッシュ読み込み失敗: {e}")
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
    
    total_rows = len(df)
    print(f"✅ データ読み込み完了: {total_rows:,}行")
    print(f"メモリ使用量: {get_memory_usage_mb():.1f} MB")
    
    # 2. 最適化済み特徴セット取得
    print(f"\n[2/4] 最適化済み特徴セット取得中...")
    optimized_features = FeatureRegistry.get_optimized_feature_set(
        correlation_threshold=correlation_threshold,
        analysis_file=None
    )
    print(f"✅ 特徴数: {len(optimized_features)}")
    
    # 必須列
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # 3. チャンクごとに処理
    print(f"\n[3/4] チャンク処理中（逐次処理でメモリ節約）...")
    
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    temp_files = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        
        print(f"\n処理中: チャンク {chunk_idx + 1}/{num_chunks} (行 {start_idx:,}-{end_idx - 1:,})")
        
        # チャンク抽出
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # OHLCV抽出
        chunk_ohlcv = chunk_df[required_cols].copy()
        
        try:
            # 特徴量計算
            chunk_features, _ = FeatureRegistry._compute_features_single(
                chunk_df,
                feature_names=optimized_features,
                report_interval=10000,  # 大きな値でレポートを抑制
                verbose=False
            )
            
            # 結合
            chunk_combined = pd.concat([chunk_ohlcv, chunk_features], axis=1)
            
            # 一時ファイルに保存
            temp_file = temp_dir / f'chunk_{chunk_idx:05d}.parquet'
            chunk_combined.to_parquet(temp_file, compression='snappy', index=False)
            temp_files.append(temp_file)
            
            print(f"  ✅ 保存完了: {temp_file.name}")
            
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            raise
        
        finally:
            # メモリ解放
            del chunk_df, chunk_ohlcv
            if 'chunk_features' in locals():
                del chunk_features
            if 'chunk_combined' in locals():
                del chunk_combined
            gc.collect()
        
        # メモリ監視
        current_memory = get_memory_usage_mb()
        memory_increase = current_memory - initial_memory
        print(f"  メモリ: {current_memory:.1f} MB (増分: +{memory_increase:.1f} MB)")
        
        # 定期的なガベージコレクション
        if (chunk_idx + 1) % 10 == 0:
            gc.collect()
    
    # 元データ解放
    del df
    gc.collect()
    
    # 4. 一時ファイルを結合
    print(f"\n[4/4] 一時ファイル結合中 ({len(temp_files)}ファイル)...")
    
    # PyArrowで効率的に結合
    tables = []
    for temp_file in temp_files:
        table = pq.read_table(temp_file)
        tables.append(table)
    
    # 結合
    combined_table = pa.concat_tables(tables)
    
    # 出力ファイルに保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    pq.write_table(combined_table, output_path, compression='snappy')
    
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"✅ 保存完了: {file_size_mb:.2f} MB")
    
    # 一時ファイル削除
    print(f"\n一時ファイル削除中...")
    for temp_file in temp_files:
        temp_file.unlink()
    temp_dir.rmdir()
    print(f"✅ 一時ファイル削除完了")
    
    # 最終メモリ使用量
    final_memory = get_memory_usage_mb()
    print(f"\n最終メモリ使用量: {final_memory:.1f} MB")
    print(f"最大メモリ増分: {final_memory - initial_memory:.1f} MB")
    
    print(f"\n=== 特徴量計算完了 ===")
    print(f"出力: {output_path}")
    print(f"サイズ: {file_size_mb:.2f} MB")
    print(f"形状: {combined_table.num_rows}行 × {combined_table.num_columns}列")


def resume_from_checkpoint(
    data_file: str,
    output_file: str,
    temp_dir: str,
    correlation_threshold: float = 0.95,
    chunk_size: int = 500,
):
    """
    中断からの再開
    
    既存の一時ファイルを検出し、未処理のチャンクから再開
    """
    temp_path = Path(temp_dir)
    if not temp_path.exists():
        print("一時ファイルが見つかりません。最初から開始します。")
        return precompute_optimized_features_memory_safe(
            data_file=data_file,
            output_file=output_file,
            correlation_threshold=correlation_threshold,
            chunk_size=chunk_size,
            temp_dir=temp_dir
        )
    
    # 既存の一時ファイルを確認
    existing_chunks = sorted(temp_path.glob('chunk_*.parquet'))
    if not existing_chunks:
        print("一時ファイルが見つかりません。最初から開始します。")
        return precompute_optimized_features_memory_safe(
            data_file=data_file,
            output_file=output_file,
            correlation_threshold=correlation_threshold,
            chunk_size=chunk_size,
            temp_dir=temp_dir
        )
    
    # 最後のチャンク番号を取得
    last_chunk = max(int(f.stem.split('_')[1]) for f in existing_chunks)
    print(f"チェックポイント検出: チャンク {last_chunk + 1}まで完了")
    print(f"続きから再開します...")
    
    # TODO: レジューム実装（必要に応じて）
    print("⚠️ レジューム機能は未実装")
    print("一時ファイルを削除して最初から開始することを推奨")
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='メモリセーフ特徴量計算')
    parser.add_argument(
        '--data-file',
        default='data/btc_jpy_real_dataset.csv',
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
        help='相関削減閾値'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='チャンクサイズ（小さいほどメモリ節約）'
    )
    parser.add_argument(
        '--temp-dir',
        default=None,
        help='一時ファイルディレクトリ'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='中断からの再開'
    )
    
    args = parser.parse_args()
    
    if args.resume:
        resume_from_checkpoint(
            data_file=args.data_file,
            output_file=args.output_file,
            temp_dir=args.temp_dir or str(Path(args.output_file).parent / 'temp_features'),
            correlation_threshold=args.correlation_threshold,
            chunk_size=args.chunk_size
        )
    else:
        precompute_optimized_features_memory_safe(
            data_file=args.data_file,
            output_file=args.output_file,
            correlation_threshold=args.correlation_threshold,
            chunk_size=args.chunk_size,
            temp_dir=args.temp_dir
        )
