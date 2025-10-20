"""
データ拡張機能の実装

金融時系列データに対する様々な拡張手法を提供：
- ノイズ注入（Gaussian noise, salt-and-pepper noise）
- 時間軸ワーピング（Time warping）
- 特徴量ミキシング（Feature mixing）
- スケーリング変換（Scaling transformations）
- 欠損値シミュレーション（Missing value simulation）
"""

import logging
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataAugmentation:
    """
    金融時系列データに対するデータ拡張を行うクラス。

    様々な拡張手法を組み合わせることで、モデルの頑健性を向上させる。
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        DataAugmentationを初期化。

        Args:
            random_seed: 再現性確保のための乱数シード
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # スケーラーのキャッシュ
        self.scalers: Dict[str, StandardScaler] = {}

    def apply_augmentations(
        self,
        data: pd.DataFrame,
        augmentations: List[Dict[str, Union[str, float, int]]],
        probability: float = 1.0,
    ) -> pd.DataFrame:
        """
        指定された拡張をデータに適用。

        Args:
            data: 拡張対象のデータフレーム
            augmentations: 拡張設定のリスト
            probability: 各拡張を適用する確率

        Returns:
            拡張されたデータフレーム

        Example:
            >>> augmenter = DataAugmentation()
            >>> augmentations = [
            ...     {"type": "gaussian_noise", "std": 0.01},
            ...     {"type": "time_warping", "sigma": 0.2}
            ... ]
            >>> augmented_data = augmenter.apply_augmentations(data, augmentations)
        """
        augmented_data = data.copy()

        for aug_config in augmentations:
            if np.random.random() < probability:
                aug_type = aug_config.get("type", "")
                try:
                    if aug_type == "gaussian_noise":
                        augmented_data = self._add_gaussian_noise(
                            augmented_data,
                            std=aug_config.get("std", 0.01),
                            columns=aug_config.get("columns"),
                        )
                    elif aug_type == "salt_pepper_noise":
                        augmented_data = self._add_salt_pepper_noise(
                            augmented_data,
                            prob=aug_config.get("prob", 0.01),
                            columns=aug_config.get("columns"),
                        )
                    elif aug_type == "time_warping":
                        augmented_data = self._apply_time_warping(
                            augmented_data,
                            sigma=aug_config.get("sigma", 0.2),
                            columns=aug_config.get("columns"),
                        )
                    elif aug_type == "feature_mixing":
                        augmented_data = self._apply_feature_mixing(
                            augmented_data,
                            mix_ratio=aug_config.get("mix_ratio", 0.1),
                            columns=aug_config.get("columns"),
                        )
                    elif aug_type == "scaling":
                        augmented_data = self._apply_scaling(
                            augmented_data,
                            scale_factor=aug_config.get("scale_factor", 1.1),
                            columns=aug_config.get("columns"),
                        )
                    elif aug_type == "missing_values":
                        augmented_data = self._simulate_missing_values(
                            augmented_data,
                            missing_prob=aug_config.get("missing_prob", 0.05),
                            columns=aug_config.get("columns"),
                        )
                    else:
                        logger.warning(f"Unknown augmentation type: {aug_type}")

                except Exception as e:
                    logger.error(f"Failed to apply augmentation {aug_type}: {e}")
                    continue

        return augmented_data

    def _add_gaussian_noise(
        self, data: pd.DataFrame, std: float = 0.01, columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        ガウスノイズを追加。

        Args:
            data: 対象データ
            std: ノイズの標準偏差
            columns: 適用する列（Noneの場合は数値列全て）

        Returns:
            ノイズを追加したデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in augmented_data.columns:
                noise = np.random.normal(0, std, size=len(augmented_data))
                augmented_data[col] = augmented_data[col] * (1 + noise)

        logger.debug(f"Applied Gaussian noise (std={std}) to {len(columns)} columns")
        return augmented_data

    def _add_salt_pepper_noise(
        self,
        data: pd.DataFrame,
        prob: float = 0.01,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Salt-and-pepperノイズを追加。

        Args:
            data: 対象データ
            prob: ノイズを適用する確率
            columns: 適用する列

        Returns:
            ノイズを追加したデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in augmented_data.columns:
                mask = np.random.random(len(augmented_data)) < prob
                # Salt (最大値) と Pepper (最小値) をランダムに選択
                salt_pepper = np.random.choice(
                    [augmented_data[col].max(), augmented_data[col].min()],
                    size=mask.sum(),
                )
                augmented_data.loc[mask, col] = salt_pepper

        logger.debug(
            f"Applied salt-pepper noise (prob={prob}) to {len(columns)} columns"
        )
        return augmented_data

    def _apply_time_warping(
        self,
        data: pd.DataFrame,
        sigma: float = 0.2,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        時間軸ワーピングを適用。

        Args:
            data: 対象データ
            sigma: ワーピングの強度
            columns: 適用する列

        Returns:
            ワーピング適用後のデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # 時間軸のワーピング関数を生成
        time_indices = np.arange(len(augmented_data))
        warp_function = np.cumsum(np.random.normal(0, sigma, len(augmented_data)))
        warp_function = warp_function - warp_function.min()
        warp_function = warp_function / warp_function.max() * (len(augmented_data) - 1)

        # 補間関数を作成
        for col in columns:
            if col in augmented_data.columns:
                interp_func = interpolate.interp1d(
                    time_indices,
                    augmented_data[col].values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                augmented_data[col] = interp_func(warp_function)

        logger.debug(f"Applied time warping (sigma={sigma}) to {len(columns)} columns")
        return augmented_data

    def _apply_feature_mixing(
        self,
        data: pd.DataFrame,
        mix_ratio: float = 0.1,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        特徴量ミキシングを適用。

        Args:
            data: 対象データ
            mix_ratio: ミキシング比率
            columns: 適用する列

        Returns:
            ミキシング適用後のデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(columns) < 2:
            logger.warning("Feature mixing requires at least 2 columns")
            return augmented_data

        # ランダムに2つの特徴量を選択してミキシング
        col1, col2 = np.random.choice(columns, 2, replace=False)

        # ミキシングマスクを作成
        mix_mask = np.random.random(len(augmented_data)) < mix_ratio

        # 特徴量をミキシング
        mixed_values = (
            augmented_data.loc[mix_mask, col1] * (1 - mix_ratio)
            + augmented_data.loc[mix_mask, col2] * mix_ratio
        )
        augmented_data.loc[mix_mask, col1] = mixed_values

        logger.debug(
            f"Applied feature mixing (ratio={mix_ratio}) between {col1} and {col2}"
        )
        return augmented_data

    def _apply_scaling(
        self,
        data: pd.DataFrame,
        scale_factor: float = 1.1,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        スケーリング変換を適用。

        Args:
            data: 対象データ
            scale_factor: スケーリング係数
            columns: 適用する列

        Returns:
            スケーリング適用後のデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in augmented_data.columns:
                augmented_data[col] = augmented_data[col] * scale_factor

        logger.debug(
            f"Applied scaling (factor={scale_factor}) to {len(columns)} columns"
        )
        return augmented_data

    def _simulate_missing_values(
        self,
        data: pd.DataFrame,
        missing_prob: float = 0.05,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        欠損値をシミュレーション。

        Args:
            data: 対象データ
            missing_prob: 欠損値の確率
            columns: 適用する列

        Returns:
            欠損値をシミュレーションしたデータ
        """
        augmented_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in augmented_data.columns:
                missing_mask = np.random.random(len(augmented_data)) < missing_prob
                augmented_data.loc[missing_mask, col] = np.nan

        logger.debug(
            f"Simulated missing values (prob={missing_prob}) in {len(columns)} columns"
        )
        return augmented_data

    def create_augmentation_pipeline(
        self, pipeline_config: List[Dict[str, Union[str, float, int]]]
    ) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """
        拡張パイプラインを作成。

        Args:
            pipeline_config: パイプライン設定

        Returns:
            拡張適用関数

        Example:
            >>> pipeline_config = [
            ...     {"type": "gaussian_noise", "std": 0.01, "probability": 0.8},
            ...     {"type": "time_warping", "sigma": 0.1, "probability": 0.5}
            ... ]
            >>> augment_func = augmenter.create_augmentation_pipeline(pipeline_config)
            >>> augmented_data = augment_func(original_data)
        """

        def augment_function(data: pd.DataFrame) -> pd.DataFrame:
            return self.apply_augmentations(data, pipeline_config)

        return augment_function
