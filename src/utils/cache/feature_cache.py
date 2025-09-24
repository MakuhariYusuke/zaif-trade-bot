"""
Feature cache for backtesting optimization.
特徴量キャッシュ（バックテスト高速化用）
"""

import hashlib
import pickle
import gzip
from pathlib import Path
from typing import Any, Callable, Optional
import pandas as pd
import os


class FeatureCache:
    """特徴量キャッシュクラス"""

    def __init__(self, cache_dir: str = "tmp/feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, data_hash: str, feature_name: str) -> str:
        """キャッシュキーを生成"""
        combined = f"{data_hash}_{feature_name}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """キャッシュファイルパスを取得"""
        return self.cache_dir / f"{cache_key}.pkl.gz"

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """データハッシュを計算（特徴量計算に影響する列のみ）"""
        # 基本列のみでハッシュ計算
        relevant_cols = ['close', 'high', 'low', 'volume', 'ts']
        available_cols = [col for col in relevant_cols if col in df.columns]

        if not available_cols:
            # フォールバック：全列
            data_str = df.to_string()
        else:
            data_str = df[available_cols].to_string()

        return hashlib.md5(data_str.encode()).hexdigest()[:16]  # 短く

    def get_or_compute(
        self,
        df: pd.DataFrame,
        feature_name: str,
        compute_func: Callable[[pd.DataFrame], pd.DataFrame],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        キャッシュから取得または計算

        Args:
            df: 入力データ
            feature_name: 特徴量名
            compute_func: 計算関数
            use_cache: キャッシュを使用するか（バックテスト時のみTrue）

        Returns:
            計算結果のDataFrame
        """
        if not use_cache:
            # 学習時は常に計算
            return compute_func(df)

        # データハッシュ計算
        data_hash = self._compute_data_hash(df)
        cache_key = self._get_cache_key(data_hash, feature_name)
        cache_path = self._get_cache_path(cache_key)

        # キャッシュ存在チェック
        if cache_path.exists():
            try:
                with gzip.open(cache_path, 'rb') as f:
                    cached_result = pickle.load(f)
                # 整合性チェック（行数）
                if len(cached_result) == len(df):
                    return cached_result
                else:
                    # データが変わった場合、再計算
                    cache_path.unlink()
            except Exception:
                # キャッシュ破損時、再計算
                if cache_path.exists():
                    cache_path.unlink()

        # 計算実行
        result = compute_func(df)

        # キャッシュ保存
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            # キャッシュ保存失敗時は無視
            pass

        return result

    def clear_cache(self, pattern: str = "*"):
        """キャッシュクリア"""
        for cache_file in self.cache_dir.glob(f"{pattern}.pkl.gz"):
            cache_file.unlink()

    def get_cache_size(self) -> int:
        """キャッシュサイズを取得（ファイル数）"""
        return len(list(self.cache_dir.glob("*.pkl.gz")))


# グローバルキャッシュインスタンス
_cache: Optional[FeatureCache] = None


def get_feature_cache(cache_dir: str = "tmp/feature_cache") -> FeatureCache:
    """グローバル特徴量キャッシュを取得"""
    global _cache
    if _cache is None:
        _cache = FeatureCache(cache_dir)
    return _cache


def cached_compute_feature(
    df: pd.DataFrame,
    feature_name: str,
    compute_func: Callable[[pd.DataFrame], pd.DataFrame],
    use_cache: Optional[bool] = None
) -> pd.DataFrame:
    """
    キャッシュ付き特徴量計算のヘルパー関数

    Args:
        df: 入力データ
        feature_name: 特徴量名
        compute_func: 計算関数
        use_cache: キャッシュ使用フラグ（Noneの場合は環境変数PRODUCTIONで判定）

    Returns:
        特徴量DataFrame
    """
    if use_cache is None:
        # PRODUCTION="" の場合のみキャッシュ使用（バックテスト時）
        use_cache = os.getenv('PRODUCTION', 'true').lower() != 'true'

    cache = get_feature_cache()
    return cache.get_or_compute(df, feature_name, compute_func, use_cache)