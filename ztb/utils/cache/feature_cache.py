"""
Feature cache for backtesting optimization.
特徴量キャッシュ（バックテスト高速化用）
"""

from __future__ import annotations
import hashlib, json, pickle, zlib, logging, threading, time, os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Literal, cast

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

class FeatureCache:
    def __init__(self, cache_dir: str = "data/cache", cache_max_mb: int = 1000, max_age_days: int = 7, compressor: str = "auto", compression_level: int = 3, access_pattern: str = "balanced", dynamic_sizing: bool = False):
        # プロセス分離: data/cache/{config_hash}/process_{pid}/
        self.process_id = os.getpid()
        base_cache_dir = Path(cache_dir)
        
        # config_hash は簡易的に固定（本番では設定ハッシュを使用）
        config_hash = "default"
        self.cache_dir = base_cache_dir / config_hash / f"process_{self.process_id}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_max_mb = cache_max_mb
        self.max_age_days = max_age_days
        self.compressor = compressor
        self.compression_level = compression_level
        self.access_pattern = access_pattern
        self.dynamic_sizing = dynamic_sizing
        self._lock = threading.Lock()  # 並列アクセス用ロック

        # 圧縮方式の設定
        self._setup_compressor()

        # 統計情報
        self.stats: Dict[str, Union[int, float]] = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_compressed_size': 0,
            'total_original_size': 0
        }

    def _select_compressor(self, data_size_bytes: int, access_pattern: Optional[str] = None) -> str:
        """動的圧縮方式選択"""
        if access_pattern is None:
            access_pattern = self.access_pattern
            
        # 固定指定の場合
        if self.compressor != "auto":
            return self.compressor
            
        # 動的選択ロジック
        if data_size_bytes < 50 * 1024:  # < 50KB
            return "lz4" if HAS_LZ4 else "zlib"
        elif data_size_bytes < 1024 * 1024:  # < 1MB
            if access_pattern == "frequent":
                return "lz4" if HAS_LZ4 else "zlib"
            else:
                return "zstd" if HAS_ZSTD else "zlib"
        else:  # >= 1MB
            cpu_cores = os.cpu_count() or 4
            if cpu_cores < 8:
                return "lz4" if HAS_LZ4 else "zstd" if HAS_ZSTD else "zlib"
            else:
                return "zstd" if HAS_ZSTD else "zlib"

    def _setup_compressor(self) -> None:
        """圧縮方式の設定"""
        selected = self._select_compressor(1024 * 1024, self.access_pattern)  # 1MB基準で初期選択
        
        if selected == "zstd" and HAS_ZSTD:
            import zstandard as zstd
            self._compress_func = lambda data: zstd.ZstdCompressor(level=self.compression_level).compress(data)
            self._decompress_func = lambda data: zstd.ZstdDecompressor().decompress(data)
        elif selected == "lz4" and HAS_LZ4:
            import lz4.frame
            self._compress_func = lz4.frame.compress
            self._decompress_func = lz4.frame.decompress
        else:
            # fallback to zlib
            self._compress_func = lambda data: zlib.compress(data, self.compression_level)
            self._decompress_func = zlib.decompress
            if selected != "zlib":
                logging.warning(f"Compressor {selected} not available, falling back to zlib")

    def _key(self, data_path: str, params: Dict[str, Any]) -> str:
        payload = json.dumps({"data": data_path, "params": params}, sort_keys=True).encode()
        return hashlib.blake2b(payload, digest_size=20).hexdigest()

    def _enforce_size_limit(self) -> None:
        """Phase1: max_age_days → Phase2: LRU の順でサイズ制限適用"""
        if self.cache_max_mb <= 0 and self.max_age_days <= 0:
            return

        with self._lock:
            # Phase 1: 年齢ベース削除（無条件）
            self._cleanup_expired_files()
            
            # Phase 2: サイズ超過時のみLRU削除
            if self.get_cache_size_mb() > self.cache_max_mb:
                self._evict_lru_files()
                
            # 適応縮退（オプション）
            if self.dynamic_sizing:
                self._adaptive_size_limit()

    def _cleanup_expired_files(self) -> None:
        """期限切れファイルの削除（max_age_days）"""
        if self.max_age_days <= 0:
            return
            
        max_age_seconds = self.max_age_days * 24 * 60 * 60
        current_time = time.time()
        
        try:
            # globの代わりにos.listdirを使用（Windows互換性向上）
            import os
            cache_dir_str = str(self.cache_dir)
            if not os.path.exists(cache_dir_str):
                return
            for filename in os.listdir(cache_dir_str):
                if filename.endswith('.pkl.z'):
                    f = self.cache_dir / filename
                    if os.path.isfile(str(f)) and current_time - os.path.getmtime(str(f)) > max_age_seconds:
                        os.unlink(str(f))
                        logging.info(f"[CACHE] Expired: {filename}")
        except Exception as e:
            logging.warning(f"[CACHE] Error in cleanup: {e}")

    def _evict_lru_files(self) -> None:
        """LRU方式でのファイル削除"""
        try:
            import os
            cache_dir_str = str(self.cache_dir)
            if not os.path.exists(cache_dir_str):
                return
                
            # os.listdirでファイル一覧を取得
            cache_files = []
            for filename in os.listdir(cache_dir_str):
                if filename.endswith('.pkl.z'):
                    f = self.cache_dir / filename
                    if os.path.isfile(str(f)):
                        cache_files.append(f)
            
            if not cache_files:
                return
                
            # 最終アクセス時間でソート（古い順）
            cache_files.sort(key=lambda f: os.path.getatime(str(f)))
            
            current_size_mb = self.get_cache_size_mb()
            
            for old_file in cache_files:
                if current_size_mb <= self.cache_max_mb:
                    break
                file_size_mb = os.path.getsize(str(old_file)) / (1024 * 1024)
                current_size_mb -= file_size_mb
                os.unlink(str(old_file))
                self.stats['evictions'] += 1
                logging.info(f"[CACHE] LRU evicted: {old_file.name}")
        except Exception as e:
            logging.warning(f"[CACHE] Error in LRU eviction: {e}")

    def _adaptive_size_limit(self) -> None:
        """適応縮退（オプション機能）"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent >= 90:
                old_limit = self.cache_max_mb
                self.cache_max_mb = max(50, int(self.cache_max_mb * 0.3))  # 70%縮退
                logging.warning(f"[CACHE] Emergency shrink: {old_limit}MB → {self.cache_max_mb}MB (mem: {memory.percent:.1f}%)")
                self._evict_lru_files()
        except ImportError:
            pass  # psutil未導入時は無視

    def get(self, data_path: str, params: Dict[str, Any]) -> Optional[Any]:
        key = self._key(data_path, params)
        f = self.cache_dir / f"{key}.pkl.z"
        if f.exists():
            with self._lock:
                self.stats['hits'] += 1
            try:
                compressed_data = f.read_bytes()
                decompressed_data = self._decompress_func(compressed_data)
                return pickle.loads(decompressed_data)
            except Exception as e:
                logging.warning(f"[CACHE] Failed to load {f}: {e}")
                return None
        else:
            with self._lock:
                self.stats['misses'] += 1
        return None

    def put(self, data_path: str, params: Dict[str, Any], obj: Any) -> Path:
        key = self._key(data_path, params)
        f = self.cache_dir / f"{key}.pkl.z"

        with self._lock:
            # データのシリアライズと圧縮
            original_data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            blob = self._compress_func(original_data)

            # 統計更新
            self.stats['total_original_size'] += len(original_data)
            self.stats['total_compressed_size'] += len(blob)

            f.write_bytes(blob)

            # サイズ制限チェック
            self._enforce_size_limit()

        return f

    def get_stats(self) -> Dict[str, Any]:
        """キャッシュ統計情報を取得"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = cast(float, (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0)

            compression_ratio: float = 0
            if self.stats['total_original_size'] > 0:
                compression_ratio = (1 - self.stats['total_compressed_size'] / self.stats['total_original_size']) * 100

            return {
                **self.stats,
                'hit_rate': hit_rate,
                'compression_ratio': compression_ratio,
                'total_requests': total_requests
            }

    def get_cache_size_mb(self) -> float:
        """現在のキャッシュサイズを取得（MB）"""
        try:
            import os
            if not os.path.exists(self.cache_dir):
                return 0.0
                
            total_size = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl.z'):
                    f = self.cache_dir / filename
                    if os.path.isfile(f):
                        total_size += os.path.getsize(f)
            return total_size / (1024 * 1024)
        except Exception as e:
            logging.warning(f"[CACHE] Error getting cache size: {e}")
            return 0.0

    def monitor_cache_health(self) -> Dict[str, Any]:
        """キャッシュの健全性を監視"""
        size_mb = self.get_cache_size_mb()
        warnings = []

        # サイズ超過チェック
        if size_mb > self.cache_max_mb * 1.2:  # 120%超過
            warnings.append(f"Cache oversized: {size_mb:.1f}MB > {self.cache_max_mb}MB limit")
            self._enforce_size_limit()  # 自動クリーンアップ

        # ヒット率チェック
        stats = self.get_stats()
        if stats['total_requests'] > 100 and stats['hit_rate'] < 0.3:
            warnings.append(f"Low cache hit rate: {stats['hit_rate']:.1%}")

        return {
            'size_mb': size_mb,
            'warnings': warnings,
            'stats': stats
        }

    def export_metrics_for_monitoring(self) -> str:
        """Prometheus互換のメトリクス出力（stub）"""
        stats = self.get_stats()
        size_mb = self.get_cache_size_mb()
        
        metrics = f"""# HELP cache_size_mb Current cache size in MB
# TYPE cache_size_mb gauge
cache_size_mb{{process="{self.process_id}"}} {size_mb}

# HELP cache_hit_rate Cache hit rate percentage
# TYPE cache_hit_rate gauge
cache_hit_rate{{process="{self.process_id}"}} {stats['hit_rate']}

# HELP cache_evictions_total Total number of cache evictions
# TYPE cache_evictions_total counter
cache_evictions_total{{process="{self.process_id}"}} {stats['evictions']}

# HELP cache_compression_ratio Compression ratio percentage
# TYPE cache_compression_ratio gauge
cache_compression_ratio{{process="{self.process_id}"}} {stats['compression_ratio']}
"""
        return metrics