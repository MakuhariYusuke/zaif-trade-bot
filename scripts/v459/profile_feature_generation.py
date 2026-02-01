"""
特徴量生成プロファイリングスクリプト（v459）

ボトルネック特定のためcProfileを使用
"""
import sys
import cProfile
import pstats
from pathlib import Path
from io import StringIO

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from ztb.features.core.registry import FeatureRegistry
from ztb.utils.data_utils import load_csv_data_cached


def profile_feature_generation(
    data_file: str = 'data/btc_jpy_1m_v451.csv',
    output_file: str = 'profile_feature_generation.prof',
    top_n: int = 30
):
    """
    特徴量生成のプロファイリング
    
    Args:
        data_file: 入力データファイル
        output_file: プロファイル出力ファイル
        top_n: 表示する上位N件
    """
    print(f"=== 特徴量生成プロファイリング ===")
    print(f"データ: {data_file}")
    print(f"出力: {output_file}\n")
    
    # データ読み込み
    print("[1/3] データ読み込み中...")
    df = load_csv_data_cached(data_file)
    print(f"✅ {len(df)}行読み込み完了\n")
    
    # 全特徴量リスト取得
    print("[2/3] 特徴量リスト取得中...")
    all_features = FeatureRegistry.list()
    print(f"✅ {len(all_features)}特徴\n")
    
    # プロファイリング実行
    print(f"[3/3] プロファイリング実行中...")
    print("（この処理には時間がかかります...）\n")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # 特徴量計算
    df_features, timings = FeatureRegistry.compute_features_batch(
        df,
        feature_names=all_features,
        verbose=False,
        return_timing=True
    )
    
    profiler.disable()
    
    # 結果解析
    print("✅ プロファイリング完了\n")
    print(f"=== プロファイル結果（上位{top_n}件） ===\n")
    
    # 標準出力にサマリー表示
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(top_n)
    
    output_text = s.getvalue()
    print(output_text)
    
    # ファイル保存
    ps.dump_stats(output_file)
    print(f"\n✅ プロファイル保存: {output_file}")
    
    # 特徴ごとの時間サマリー
    if timings:
        print(f"\n=== 特徴ごとの計算時間（上位10件） ===\n")
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:10]
        total_time = sum(timings.values())
        
        for i, (feat, t) in enumerate(sorted_timings, 1):
            percentage = (t / total_time * 100) if total_time > 0 else 0
            print(f"{i:2d}. {feat:30s}: {t:6.3f}秒 ({percentage:5.2f}%)")
        
        print(f"\n総計算時間: {total_time:.2f}秒")
    
    print(f"\n=== プロファイリング完了 ===")
    print(f"次のステップ:")
    print(f"  1. 結果分析: snakeviz {output_file}")
    print(f"  2. ボトルネック特定")
    print(f"  3. Numba/vectorization適用検討")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='特徴量生成プロファイリング')
    parser.add_argument(
        '--data-file',
        default='data/btc_jpy_1m_v451.csv',
        help='入力データファイル'
    )
    parser.add_argument(
        '--output-file',
        default='profile_feature_generation.prof',
        help='プロファイル出力ファイル'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=30,
        help='表示する上位N件'
    )
    
    args = parser.parse_args()
    
    profile_feature_generation(
        data_file=args.data_file,
        output_file=args.output_file,
        top_n=args.top_n
    )
