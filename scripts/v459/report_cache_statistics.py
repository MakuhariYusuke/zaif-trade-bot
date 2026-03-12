"""
キャッシュ統計レポートスクリプト（v459）

FeatureCacheの統計情報を表示
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.cache.feature_cache import FeatureCache


def report_cache_statistics(
    cache_dir: str = 'data/cache',
    cache_max_mb: int = 1000
):
    """
    キャッシュ統計レポート
    
    Args:
        cache_dir: キャッシュディレクトリ
        cache_max_mb: 最大キャッシュサイズ（MB）
    """
    print(f"=== FeatureCache統計レポート ===\n")
    print(f"キャッシュディレクトリ: {cache_dir}")
    print(f"最大サイズ: {cache_max_mb}MB\n")
    
    try:
        # FeatureCache初期化
        cache = FeatureCache(
            cache_dir=cache_dir,
            cache_max_mb=cache_max_mb
        )
        
        # 統計取得
        stats = cache.get_stats()
        
        # 統計表示
        print("=== キャッシュ統計 ===\n")
        
        # ヒット率
        hit_rate = stats.get('hit_rate', 0)
        print(f"ヒット率: {hit_rate:.1f}%")
        
        # リクエスト数
        hits = stats.get('hits', 0)
        misses = stats.get('misses', 0)
        total_requests = stats.get('total_requests', hits + misses)
        print(f"  ヒット: {hits:,}回")
        print(f"  ミス: {misses:,}回")
        print(f"  総リクエスト: {total_requests:,}回\n")
        
        # 圧縮率
        compression_ratio = stats.get('compression_ratio', 0)
        print(f"圧縮率: {compression_ratio:.1f}%")
        
        # サイズ情報
        original_size = stats.get('total_original_size', 0)
        compressed_size = stats.get('total_compressed_size', 0)
        print(f"  元のサイズ: {original_size/1e6:.2f}MB")
        print(f"  圧縮後: {compressed_size/1e6:.2f}MB")
        
        if original_size > 0:
            saved = original_size - compressed_size
            print(f"  節約: {saved/1e6:.2f}MB ({saved/original_size*100:.1f}%)\n")
        
        # LRU削除
        evictions = stats.get('evictions', 0)
        print(f"LRU削除: {evictions:,}回\n")
        
        # キャッシュサイズ
        cache_size = cache.get_cache_size_mb()
        print(f"現在のキャッシュサイズ: {cache_size:.2f}MB / {cache_max_mb}MB")
        print(f"使用率: {cache_size/cache_max_mb*100:.1f}%\n")
        
        # ヘルス監視
        try:
            health = cache.monitor_cache_health()
            print("=== キャッシュヘルス ===\n")
            for key, value in health.items():
                print(f"{key}: {value}")
        except Exception as e:
            print(f"⚠️ ヘルス監視取得失敗: {e}")
        
        print(f"\n=== レポート完了 ===")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        print(f"\n考えられる原因:")
        print(f"  - キャッシュディレクトリが存在しない: {cache_dir}")
        print(f"  - FeatureCacheが未使用（統計なし）")
        print(f"  - パーミッションエラー")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='キャッシュ統計レポート')
    parser.add_argument(
        '--cache-dir',
        default='data/cache',
        help='キャッシュディレクトリ'
    )
    parser.add_argument(
        '--cache-max-mb',
        type=int,
        default=1000,
        help='最大キャッシュサイズ（MB）'
    )
    
    args = parser.parse_args()
    
    report_cache_statistics(
        cache_dir=args.cache_dir,
        cache_max_mb=args.cache_max_mb
    )
