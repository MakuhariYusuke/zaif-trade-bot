import os
import tempfile

from src.utils.cache.feature_cache import FeatureCache


class TestFeatureCache:
    def setup_method(self):
        self.test_cache_dir = tempfile.mkdtemp(prefix="test_cache_")
        self.cache = FeatureCache(self.test_cache_dir, 10, 1)  # 10MB, 1日
        # テスト時はサイズ制限を無効化
        self.cache._enforce_size_limit = lambda: None

    def teardown_method(self):
        # クリーンアップ
        import shutil

        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_select_compressor_small_data(self):
        """小規模データでlz4が選択される"""
        result = self.cache._select_compressor(40 * 1024)  # 40KB
        assert result == "lz4"

    def test_select_compressor_large_data(self):
        """大規模データでzstdが選択される"""
        result = self.cache._select_compressor(2 * 1024 * 1024)  # 2MB
        assert result == "zstd"

    def test_select_compressor_frequent_access(self):
        """頻繁アクセス時はlz4が優先"""
        result = self.cache._select_compressor(
            500 * 1024, "frequent"
        )  # 500KB, frequent
        assert result == "lz4"

    def test_select_compressor_fixed_setting(self):
        """固定設定が優先される"""
        fixed_cache = FeatureCache(self.test_cache_dir, 10, 1, "zlib")
        result = fixed_cache._select_compressor(100 * 1024)
        assert result == "zlib"

    def test_process_isolation(self):
        """プロセス分離されたディレクトリが作成される"""
        cache_dir = str(self.cache.cache_dir)
        assert "process_" in cache_dir
        assert self.test_cache_dir in cache_dir

    def test_cache_operations(self):
        """基本的なキャッシュ操作"""
        test_data = {"features": [1, 2, 3], "labels": [0, 1, 0]}

        # キャッシュミス
        result = self.cache.get("test_data.csv", {"window": 10})
        assert result is None

        # データ保存
        self.cache.put("test_data.csv", {"window": 10}, test_data)

        # キャッシュヒット
        result = self.cache.get("test_data.csv", {"window": 10})
        assert result is not None
        assert result == test_data

    def test_cache_stats(self):
        """統計情報が正しく更新される"""
        test_data = {"features": [1, 2, 3], "labels": [0, 1, 0]}

        # 初期状態
        stats = self.cache.get_stats()
        initial_requests = stats["total_requests"]

        # キャッシュミス
        self.cache.get("test_data.csv", {"window": 10})

        # データ保存とヒット
        self.cache.put("test_data.csv", {"window": 10}, test_data)
        self.cache.get("test_data.csv", {"window": 10})

        # 統計確認
        stats = self.cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == initial_requests + 2
        assert stats["hit_rate"] == 50.0

    def test_export_metrics(self):
        """Prometheus互換メトリクスがエクスポートされる"""
        metrics = self.cache.export_metrics_for_monitoring()
        assert "# HELP cache_size_mb" in metrics
        assert "# TYPE cache_size_mb gauge" in metrics
        assert "cache_size_mb" in metrics
        assert f'process="{os.getpid()}"' in metrics
