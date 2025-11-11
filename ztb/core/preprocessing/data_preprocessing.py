"""
Data Preprocessing Module for SAC v446

短期間収益性向上のためのデータ前処理機能
- ノイズフィルタリング
- アノマリー検出
- 合成データ生成
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class NoiseFilter:
    """
    短期間ノイズフィルタリングクラス

    統計的フィルタリングを用いて市場ノイズを除去
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.zscore_threshold = self.config.get('zscore_threshold', 3.0)
        self.iqr_multiplier = self.config.get('iqr_multiplier', 1.5)
        logger.info("NoiseFilter initialized")

    def filter_zscore(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Z-scoreベースのノイズフィルタリング

        Args:
            df: 入力データフレーム
            columns: フィルタリング対象カラム

        Returns:
            フィルタリングされたデータフレーム
        """
        df_filtered = df.copy()

        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                mask = z_scores < self.zscore_threshold
                df_filtered.loc[~mask, col] = df_filtered[col].rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')

        logger.info(f"Z-score filtering applied to {len(columns)} columns")
        return df_filtered

    def filter_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        IQRベースのノイズフィルタリング

        Args:
            df: 入力データフレーム
            columns: フィルタリング対象カラム

        Returns:
            フィルタリングされたデータフレーム
        """
        df_filtered = df.copy()

        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (self.iqr_multiplier * IQR)
                upper_bound = Q3 + (self.iqr_multiplier * IQR)

                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                df_filtered.loc[mask, col] = df_filtered[col].rolling(window=5, center=True).mean().fillna(method='bfill').fillna(method='ffill')

        logger.info(f"IQR filtering applied to {len(columns)} columns")
        return df_filtered

    def apply_filters(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        フィルタリング適用

        Args:
            df: 入力データフレーム
            columns: 対象カラム（Noneの場合は数値カラムすべて）

        Returns:
            フィルタリングされたデータフレーム
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Z-scoreフィルタリング
        df_filtered = self.filter_zscore(df, columns)

        # IQRフィルタリング
        df_filtered = self.filter_iqr(df_filtered, columns)

        logger.info("All noise filters applied")
        return df_filtered


class AnomalyDetector:
    """
    アノマリー検出クラス

    異常値を検出して除去または修正
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.methods = {
            'isolation_forest': self._detect_isolation_forest,
            'local_outlier_factor': self._detect_lof,
            'statistical': self._detect_statistical
        }
        logger.info("AnomalyDetector initialized")

    def _detect_isolation_forest(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Isolation Forestによる異常検出"""
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            features = df[columns].fillna(df[columns].mean())
            predictions = iso_forest.fit_predict(features)
            return pd.Series(predictions == -1, index=df.index)
        except ImportError:
            logger.warning("scikit-learn not available for Isolation Forest")
            return pd.Series(False, index=df.index)

    def _detect_lof(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """Local Outlier Factorによる異常検出"""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            lof = LocalOutlierFactor(contamination=0.1)
            features = df[columns].fillna(df[columns].mean())
            predictions = lof.fit_predict(features)
            return pd.Series(predictions == -1, index=df.index)
        except ImportError:
            logger.warning("scikit-learn not available for LOF")
            return pd.Series(False, index=df.index)

    def _detect_statistical(self, df: pd.DataFrame, columns: List[str]) -> pd.Series:
        """統計的手法による異常検出"""
        anomalies = pd.Series(False, index=df.index)

        for col in columns:
            if col in df.columns:
                # Z-scoreベース
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
                anomalies = anomalies | (z_scores > 3.0)

                # IQRベース
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (1.5 * IQR)
                upper_bound = Q3 + (1.5 * IQR)
                anomalies = anomalies | (df[col] < lower_bound) | (df[col] > upper_bound)

        return anomalies

    def detect_anomalies(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                        method: str = 'statistical') -> Tuple[pd.DataFrame, pd.Series]:
        """
        異常検出

        Args:
            df: 入力データフレーム
            columns: 対象カラム
            method: 検出方法 ('isolation_forest', 'local_outlier_factor', 'statistical')

        Returns:
            (クリーンなデータフレーム, 異常フラグのシリーズ)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if method not in self.methods:
            logger.warning(f"Unknown method {method}, using statistical")
            method = 'statistical'

        anomaly_mask = self.methods[method](df, columns)

        # 異常値を前後の値で補間
        df_clean = df.copy()
        for col in columns:
            if col in df.columns:
                df_clean.loc[anomaly_mask, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        logger.info(f"Anomaly detection completed using {method} method")
        return df_clean, anomaly_mask


class SyntheticDataGenerator:
    """
    合成データ生成クラス

    CPU負荷を考慮した合成データ生成
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        np.random.seed(self.random_state)
        logger.info("SyntheticDataGenerator initialized")

    def generate_gaussian_noise(self, df: pd.DataFrame, columns: List[str],
                               noise_level: float = 0.1) -> pd.DataFrame:
        """
        ガウスノイズを加えたデータ生成

        Args:
            df: 元データ
            columns: 対象カラム
            noise_level: ノイズレベル

        Returns:
            ノイズ付加データ
        """
        df_synthetic = df.copy()

        for col in columns:
            if col in df.columns:
                std = df[col].std()
                noise = np.random.normal(0, std * noise_level, len(df))
                df_synthetic[col] = df[col] + noise

        return df_synthetic

    def generate_smote_like(self, df: pd.DataFrame, target_column: str,
                           n_samples: int = 1000) -> pd.DataFrame:
        """
        SMOTE-like合成データ生成（簡易版）

        Args:
            df: 元データ
            target_column: ターゲットカラム
            n_samples: 生成サンプル数

        Returns:
            合成データ
        """
        try:
            from sklearn.neighbors import NearestNeighbors

            # 数値カラムのみ使用
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column not in numeric_cols:
                numeric_cols.append(target_column)

            features = df[numeric_cols].fillna(df[numeric_cols].mean())

            # 最近傍探索
            nn = NearestNeighbors(n_neighbors=5)
            nn.fit(features)

            synthetic_samples = []

            for _ in range(n_samples):
                # ランダムにサンプルを選択
                idx = np.random.randint(len(features))
                sample = features.iloc[idx]

                # 最近傍を取得
                distances, indices = nn.kneighbors([sample])
                neighbor_idx = np.random.choice(indices[0][1:])  # 自分以外
                neighbor = features.iloc[neighbor_idx]

                # 線形補間
                alpha = np.random.random()
                synthetic_sample = sample + alpha * (neighbor - sample)
                synthetic_samples.append(synthetic_sample)

            df_synthetic = pd.DataFrame(synthetic_samples, columns=numeric_cols)

            # 元データと結合
            df_combined = pd.concat([df, df_synthetic], ignore_index=True)

            logger.info(f"Generated {n_samples} synthetic samples using SMOTE-like method")
            return df_combined

        except ImportError:
            logger.warning("scikit-learn not available for SMOTE-like generation")
            return df

    def generate_time_series(self, df: pd.DataFrame, n_periods: int = 1000,
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        時系列合成データ生成

        Args:
            df: 元データ
            n_periods: 生成期間数
            columns: 対象カラム

        Returns:
            合成時系列データ
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # トレンドと季節性を考慮した生成
        synthetic_data = []

        for col in columns:
            if col in df.columns:
                # 元データの統計量
                mean_val = df[col].mean()
                std_val = df[col].std()

                # ランダムウォーク + ノイズ
                values = [mean_val]
                for i in range(n_periods - 1):
                    # ランダムウォーク
                    change = np.random.normal(0, std_val * 0.01)
                    new_value = values[-1] + change

                    # 平均回帰
                    new_value += (mean_val - values[-1]) * 0.001

                    # ノイズ追加
                    new_value += np.random.normal(0, std_val * 0.05)

                    values.append(new_value)

                synthetic_data.append(pd.Series(values, name=col))

        df_synthetic = pd.DataFrame(dict(zip(columns, synthetic_data)))

        # タイムスタンプ追加
        if 'timestamp' in df.columns:
            start_time = pd.to_datetime(df['timestamp'].max())
            timestamps = pd.date_range(start=start_time, periods=n_periods, freq='1min')
            df_synthetic['timestamp'] = timestamps

        logger.info(f"Generated {n_periods} periods of synthetic time series data")
        return df_synthetic


def preprocess_data(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    データ前処理パイプライン

    Args:
        df: 入力データ
        config: 設定

    Returns:
        前処理済みデータ
    """
    config = config or {}

    # ノイズフィルタリング
    if config.get('apply_noise_filter', True):
        noise_filter = NoiseFilter(config.get('noise_filter_config'))
        df = noise_filter.apply_filters(df)

    # アノマリー検出
    if config.get('apply_anomaly_detection', True):
        anomaly_detector = AnomalyDetector(config.get('anomaly_config'))
        df, _ = anomaly_detector.detect_anomalies(df, method=config.get('anomaly_method', 'statistical'))

    # 合成データ生成
    if config.get('generate_synthetic', False):
        synthetic_generator = SyntheticDataGenerator(config.get('synthetic_config'))
        df_synthetic = synthetic_generator.generate_time_series(
            df, n_periods=config.get('synthetic_periods', 1000)
        )
        df = pd.concat([df, df_synthetic], ignore_index=True)

    logger.info("Data preprocessing pipeline completed")
    return df