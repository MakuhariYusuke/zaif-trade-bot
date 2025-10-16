"""
異常値検出と処理機能の実装

金融時系列データに対する様々な異常値検出・処理手法を提供：
- 統計的手法（Z-score, IQR, Modified Z-score）
- 機械学習手法（Isolation Forest, Local Outlier Factor）
- 時系列特化手法（STL分解, ARIMA残差分析）
- 処理手法（除去, 補完, クリッピング）
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    金融時系列データに対する異常値検出を行うクラス。

    複数の検出手法を組み合わせ、堅牢な異常値検出を実現。
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        OutlierDetectorを初期化。

        Args:
            random_seed: 再現性確保のための乱数シード
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # スケーラーのキャッシュ
        self.scalers: Dict[str, StandardScaler] = {}

        # 学習済みモデルのキャッシュ
        self.trained_models: Dict[str, Any] = {}

    def detect_outliers(
        self,
        data: pd.DataFrame,
        methods: List[Dict[str, Union[str, float, int]]],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        指定された手法で異常値を検出。

        Args:
            data: 対象データ
            methods: 検出手法の設定リスト
            columns: 適用する列（Noneの場合は数値列全て）

        Returns:
            異常値フラグを含むデータフレーム

        Example:
            >>> detector = OutlierDetector()
            >>> methods = [
            ...     {"type": "z_score", "threshold": 3.0},
            ...     {"type": "iqr", "multiplier": 1.5}
            ... ]
            >>> result = detector.detect_outliers(data, methods)
        """
        result_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # 各手法で異常値を検出
        outlier_flags = {}
        for method_config in methods:
            method_type = method_config.get("type", "")
            try:
                if method_type == "z_score":
                    flags = self._detect_z_score(
                        result_data, columns,
                        threshold=method_config.get("threshold", 3.0)
                    )
                elif method_type == "iqr":
                    flags = self._detect_iqr(
                        result_data, columns,
                        multiplier=method_config.get("multiplier", 1.5)
                    )
                elif method_type == "modified_z_score":
                    flags = self._detect_modified_z_score(
                        result_data, columns,
                        threshold=method_config.get("threshold", 3.5)
                    )
                elif method_type == "isolation_forest":
                    flags = self._detect_isolation_forest(
                        result_data, columns,
                        contamination=method_config.get("contamination", 0.1)
                    )
                elif method_type == "lof":
                    flags = self._detect_lof(
                        result_data, columns,
                        n_neighbors=method_config.get("n_neighbors", 20),
                        contamination=method_config.get("contamination", 0.1)
                    )
                elif method_type == "stl_decomposition":
                    flags = self._detect_stl_decomposition(
                        result_data, columns,
                        seasonal=method_config.get("seasonal", 7),
                        threshold=method_config.get("threshold", 2.0)
                    )
                elif method_type == "arima_residual":
                    flags = self._detect_arima_residual(
                        result_data, columns,
                        order=method_config.get("order", (1, 1, 1)),
                        threshold=method_config.get("threshold", 2.0)
                    )
                else:
                    logger.warning(f"Unknown outlier detection method: {method_type}")
                    continue

                outlier_flags[method_type] = flags

            except Exception as e:
                logger.error(f"Failed to apply outlier detection {method_type}: {e}")
                continue

        # 異常値フラグを統合
        combined_flags = self._combine_outlier_flags(outlier_flags)
        for col in columns:
            if col in combined_flags:
                result_data[f"{col}_is_outlier"] = combined_flags[col]

        return result_data

    def _detect_z_score(
        self,
        data: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """
        Z-score法による異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            threshold: 閾値

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        for col in columns:
            if col not in data.columns:
                continue

            values = data[col].dropna().values
            if len(values) == 0:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            z_scores = np.abs(stats.zscore(values))
            outlier_mask = np.full(len(data), False)

            # NaNでない位置のみを考慮
            valid_mask = ~data[col].isna()
            outlier_mask[valid_mask] = z_scores > threshold

            flags[col] = outlier_mask

        logger.debug(f"Z-score detection completed for {len(columns)} columns with threshold {threshold}")
        return flags

    def _detect_iqr(
        self,
        data: pd.DataFrame,
        columns: List[str],
        multiplier: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        IQR法による異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            multiplier: IQR倍率

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        for col in columns:
            if col not in data.columns:
                continue

            values = data[col].dropna().values
            if len(values) == 0:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            outlier_mask = np.full(len(data), False)
            valid_mask = ~data[col].isna()
            outlier_mask[valid_mask] = (data.loc[valid_mask, col] < lower_bound) | \
                                     (data.loc[valid_mask, col] > upper_bound)

            flags[col] = outlier_mask

        logger.debug(f"IQR detection completed for {len(columns)} columns with multiplier {multiplier}")
        return flags

    def _detect_modified_z_score(
        self,
        data: pd.DataFrame,
        columns: List[str],
        threshold: float = 3.5
    ) -> Dict[str, np.ndarray]:
        """
        Modified Z-score法による異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            threshold: 閾値

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        for col in columns:
            if col not in data.columns:
                continue

            values = data[col].dropna().values
            if len(values) == 0:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            median = np.median(values)
            mad = np.median(np.abs(values - median))

            if mad == 0:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            modified_z_scores = 0.6745 * (values - median) / mad
            outlier_mask = np.full(len(data), False)

            valid_mask = ~data[col].isna()
            outlier_mask[valid_mask] = np.abs(modified_z_scores) > threshold

            flags[col] = outlier_mask

        logger.debug(f"Modified Z-score detection completed for {len(columns)} columns with threshold {threshold}")
        return flags

    def _detect_isolation_forest(
        self,
        data: pd.DataFrame,
        columns: List[str],
        contamination: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Isolation Forestによる異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            contamination: 異常値の割合

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        # 特徴量を準備
        feature_data = data[columns].dropna()
        if len(feature_data) == 0:
            for col in columns:
                flags[col] = np.zeros(len(data), dtype=bool)
            return flags

        # スケーリング
        scaler_key = f"isolation_forest_{hash(tuple(columns))}"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()

        scaled_features = self.scalers[scaler_key].fit_transform(feature_data)

        # Isolation Forestの適用
        model_key = f"isolation_forest_{contamination}"
        if model_key not in self.trained_models:
            self.trained_models[model_key] = IsolationForest(
                contamination=contamination,
                random_state=self.random_seed
            )

        outlier_predictions = self.trained_models[model_key].fit_predict(scaled_features)

        # 各列に同じフラグを適用（特徴量全体としての異常値）
        outlier_mask = outlier_predictions == -1
        full_mask = np.full(len(data), False)
        valid_indices = data[columns].dropna().index
        full_mask[valid_indices] = outlier_mask

        for col in columns:
            flags[col] = full_mask.copy()

        logger.debug(f"Isolation Forest detection completed for {len(columns)} columns with contamination {contamination}")
        return flags

    def _detect_lof(
        self,
        data: pd.DataFrame,
        columns: List[str],
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Local Outlier Factorによる異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            n_neighbors: 近傍点数
            contamination: 異常値の割合

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        # 特徴量を準備
        feature_data = data[columns].dropna()
        if len(feature_data) == 0:
            for col in columns:
                flags[col] = np.zeros(len(data), dtype=bool)
            return flags

        # スケーリング
        scaler_key = f"lof_{hash(tuple(columns))}"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()

        scaled_features = self.scalers[scaler_key].fit_transform(feature_data)

        # LOFの適用
        model_key = f"lof_{n_neighbors}_{contamination}"
        if model_key not in self.trained_models:
            self.trained_models[model_key] = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination
            )

        outlier_predictions = self.trained_models[model_key].fit_predict(scaled_features)

        # 各列に同じフラグを適用
        outlier_mask = outlier_predictions == -1
        full_mask = np.full(len(data), False)
        valid_indices = data[columns].dropna().index
        full_mask[valid_indices] = outlier_mask

        for col in columns:
            flags[col] = full_mask.copy()

        logger.debug(f"LOF detection completed for {len(columns)} columns with n_neighbors {n_neighbors}")
        return flags

    def _detect_stl_decomposition(
        self,
        data: pd.DataFrame,
        columns: List[str],
        seasonal: int = 7,
        threshold: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        STL分解による時系列異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            seasonal: 季節性周期
            threshold: 残差の閾値

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        for col in columns:
            if col not in data.columns:
                continue

            values = data[col].dropna()
            if len(values) < seasonal * 2:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            try:
                # STL分解
                stl = STL(values, seasonal=seasonal, robust=True)
                result = stl.fit()

                # 残差の標準偏差で閾値を設定
                residual_std = result.resid.std()
                outlier_mask = np.abs(result.resid) > threshold * residual_std

                # 元のデータフレームにマッピング
                full_mask = np.full(len(data), False)
                valid_indices = values.index
                full_mask[valid_indices] = outlier_mask

                flags[col] = full_mask

            except Exception as e:
                logger.warning(f"STL decomposition failed for {col}: {e}")
                flags[col] = np.zeros(len(data), dtype=bool)

        logger.debug(f"STL decomposition detection completed for {len(columns)} columns")
        return flags

    def _detect_arima_residual(
        self,
        data: pd.DataFrame,
        columns: List[str],
        order: Tuple[int, int, int] = (1, 1, 1),
        threshold: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        ARIMA残差分析による異常値検出。

        Args:
            data: 対象データ
            columns: 対象列
            order: ARIMA次数 (p, d, q)
            threshold: 残差の閾値

        Returns:
            各列の異常値フラグ
        """
        flags = {}

        for col in columns:
            if col not in data.columns:
                continue

            values = data[col].dropna()
            if len(values) < 10:
                flags[col] = np.zeros(len(data), dtype=bool)
                continue

            try:
                # ARIMAモデルフィッティング
                model = ARIMA(values, order=order)
                fitted_model = model.fit()

                # 残差を取得
                residuals = fitted_model.resid

                # 残差の標準偏差で閾値を設定
                residual_std = residuals.std()
                outlier_mask = np.abs(residuals) > threshold * residual_std

                # 元のデータフレームにマッピング
                full_mask = np.full(len(data), False)
                valid_indices = values.index
                full_mask[valid_indices] = outlier_mask

                flags[col] = full_mask

            except Exception as e:
                logger.warning(f"ARIMA residual analysis failed for {col}: {e}")
                flags[col] = np.zeros(len(data), dtype=bool)

        logger.debug(f"ARIMA residual detection completed for {len(columns)} columns")
        return flags

    def _combine_outlier_flags(self, outlier_flags: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        複数の検出手法の結果を統合。

        Args:
            outlier_flags: 各手法の異常値フラグ

        Returns:
            統合された異常値フラグ
        """
        if not outlier_flags:
            return {}

        # 最初の手法の列を取得
        first_method = next(iter(outlier_flags.values()))
        combined_flags = {}

        for col in first_method.keys():
            # 各手法のフラグを収集
            method_flags = []
            for method_flags_dict in outlier_flags.values():
                if col in method_flags_dict:
                    method_flags.append(method_flags_dict[col])

            if method_flags:
                # 多数決で統合（過半数の手法が異常値と判定）
                stacked_flags = np.stack(method_flags, axis=0)
                combined_flags[col] = np.sum(stacked_flags, axis=0) > len(method_flags) / 2
            else:
                combined_flags[col] = np.zeros(len(first_method[col]), dtype=bool)

        return combined_flags


class OutlierHandler:
    """
    検出された異常値に対する処理を行うクラス。
    """

    def __init__(self):
        """OutlierHandlerを初期化。"""
        pass

    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = "remove",
        outlier_columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        異常値を処理。

        Args:
            data: 対象データ
            method: 処理方法 ("remove", "interpolate", "clip", "replace")
            outlier_columns: 異常値フラグ列のリスト
            **kwargs: 処理方法固有のパラメータ

        Returns:
            処理後のデータフレーム
        """
        processed_data = data.copy()

        if outlier_columns is None:
            # "_is_outlier" で終わる列を自動検出
            outlier_columns = [col for col in data.columns if col.endswith("_is_outlier")]

        if not outlier_columns:
            logger.warning("No outlier columns found")
            return processed_data

        try:
            if method == "remove":
                processed_data = self._remove_outliers(processed_data, outlier_columns)
            elif method == "interpolate":
                processed_data = self._interpolate_outliers(
                    processed_data, outlier_columns,
                    method=kwargs.get("interpolation_method", "linear")
                )
            elif method == "clip":
                processed_data = self._clip_outliers(
                    processed_data, outlier_columns,
                    lower_percentile=kwargs.get("lower_percentile", 5),
                    upper_percentile=kwargs.get("upper_percentile", 95)
                )
            elif method == "replace":
                processed_data = self._replace_outliers(
                    processed_data, outlier_columns,
                    replacement_value=kwargs.get("replacement_value", "median")
                )
            else:
                logger.warning(f"Unknown outlier handling method: {method}")

        except Exception as e:
            logger.error(f"Failed to handle outliers with method {method}: {e}")

        return processed_data

    def _remove_outliers(self, data: pd.DataFrame, outlier_columns: List[str]) -> pd.DataFrame:
        """異常値を含む行を削除。"""
        outlier_mask = np.zeros(len(data), dtype=bool)

        for col in outlier_columns:
            if col in data.columns:
                base_col = col.replace("_is_outlier", "")
                if base_col in data.columns:
                    outlier_mask |= data[col]

        logger.info(f"Removing {outlier_mask.sum()} outlier rows")
        return data[~outlier_mask].copy()

    def _interpolate_outliers(
        self,
        data: pd.DataFrame,
        outlier_columns: List[str],
        method: str = "linear"
    ) -> pd.DataFrame:
        """異常値を補間。"""
        processed_data = data.copy()

        for col in outlier_columns:
            if col in data.columns:
                base_col = col.replace("_is_outlier", "")
                if base_col in data.columns:
                    outlier_mask = data[col]

                    # 異常値でないデータで補間
                    valid_data = data.loc[~outlier_mask, base_col]
                    if len(valid_data) > 1:
                        interpolated = valid_data.interpolate(method=method)
                        processed_data.loc[outlier_mask, base_col] = interpolated.reindex(data.index)[outlier_mask]

        logger.info(f"Interpolated outliers using {method} method")
        return processed_data

    def _clip_outliers(
        self,
        data: pd.DataFrame,
        outlier_columns: List[str],
        lower_percentile: float = 5,
        upper_percentile: float = 95
    ) -> pd.DataFrame:
        """異常値をパーセンタイルでクリッピング。"""
        processed_data = data.copy()

        for col in outlier_columns:
            if col in data.columns:
                base_col = col.replace("_is_outlier", "")
                if base_col in data.columns:
                    outlier_mask = data[col]

                    # 有効なデータのパーセンタイルを計算
                    valid_values = data.loc[~outlier_mask, base_col].dropna()
                    if len(valid_values) > 0:
                        lower_bound = np.percentile(valid_values, lower_percentile)
                        upper_bound = np.percentile(valid_values, upper_percentile)

                        processed_data.loc[outlier_mask, base_col] = np.clip(
                            data.loc[outlier_mask, base_col], lower_bound, upper_bound
                        )

        logger.info(f"Clipped outliers to {lower_percentile}-{upper_percentile} percentiles")
        return processed_data

    def _replace_outliers(
        self,
        data: pd.DataFrame,
        outlier_columns: List[str],
        replacement_value: Union[str, float] = "median"
    ) -> pd.DataFrame:
        """異常値を指定値で置換。"""
        processed_data = data.copy()

        for col in outlier_columns:
            if col in data.columns:
                base_col = col.replace("_is_outlier", "")
                if base_col in data.columns:
                    outlier_mask = data[col]

                    # 置換値を決定
                    valid_values = data.loc[~outlier_mask, base_col].dropna()
                    if len(valid_values) > 0:
                        if replacement_value == "median":
                            replace_val = valid_values.median()
                        elif replacement_value == "mean":
                            replace_val = valid_values.mean()
                        elif isinstance(replacement_value, (int, float)):
                            replace_val = replacement_value
                        else:
                            replace_val = valid_values.median()

                        processed_data.loc[outlier_mask, base_col] = replace_val

        logger.info(f"Replaced outliers with {replacement_value}")
        return processed_data