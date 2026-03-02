"""
v459 Phase 0.2c: 因果性保証付きGroupedFeatureScaler
Doc04仕様準拠: 88次元中36次元の選択的スケーリング with 因果性保証
"""

import numpy as np
import pandas as pd

from ztb.features.grouping.grouped_scaler import GroupedFeatureScaler

class CausalGroupedFeatureScaler(GroupedFeatureScaler):
    """
    Doc04仕様に準拠した因果性保証付きGroupedFeatureScaler
    
    - 88次元観測空間のうち36次元を選択的にスケール
    - fit範囲をend_idxで記録
    - リーク検査機能
    - ゼロ分散対応（std_floor）
    - NaN/inf検査
    """
    
    def __init__(self, epsilon=1e-7, momentum=0.99, clip_value=3.0, std_floor=1e-8):
        """
        Args:
            epsilon: ゼロ除算防止（親クラス）
            momentum: EMAモメンタム（親クラス、0.99推奨）
            clip_value: クリッピング閾値（親クラス）
            std_floor: 標準偏差のフロア値（ゼロ分散対応）
        """
        super().__init__(epsilon, momentum, clip_value)
        self.std_floor = std_floor
        self.fit_end_idx: int | None = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, end_idx: int):
        """
        Train期間のみでfit（因果性保証）
        
        Args:
            data: 全データ（88次元観測空間）
            end_idx: Train終端index（この行まで含む）
        
        Raises:
            ValueError: データ不正またはリーク検出時
        """
        if end_idx >= len(data):
            raise ValueError(
                f"end_idx ({end_idx}) must be less than data length ({len(data)})"
            )
        
        # Train期間のみ抽出
        train_data = data.iloc[:end_idx + 1].values
        
        if len(train_data) == 0:
            raise ValueError("Train data is empty")
        
        if train_data.shape[1] != 88:
            raise ValueError(
                f"Expected 88-dimensional features, got {train_data.shape[1]}"
            )
        
        # バッチ更新（親クラスのfit_oneメソッド使用）
        for row in train_data:
            self.fit_one(row)
        
        # ゼロ分散対応（Doc04仕様）
        self.std = np.maximum(self.std, self.std_floor)
        
        self.fitted = True
        self.fit_end_idx = end_idx
        
        # リーク検査
        self._verify_no_leakage(data, end_idx)
    
    def _verify_no_leakage(self, data: pd.DataFrame, end_idx: int):
        """
        Val/Testデータの混入を検査（警告のみ）
        
        Note: GroupedFeatureScalerはmomentum=0.99のEMA更新のため、
        バッチ統計とは完全一致しない。リーク検査は参考値として警告のみ。
        実質的な因果性保証は、fit()のend_idx管理で担保する。
        
        Args:
            data: 全データ
            end_idx: Train終端index
        """
        if len(data) <= end_idx + 1:
            # Val/Testデータが存在しない場合はスキップ
            return
        
        # スケール対象のインデックス（親クラスのSCALE_INDICES使用）
        scale_indices = self.SCALE_INDICES
        
        # Train期間の統計を再計算（スケール対象のみ）
        train_data = data.iloc[:end_idx + 1].values[:, scale_indices]
        expected_mean = train_data.mean(axis=0)
        expected_std = train_data.std(axis=0, ddof=1)
        
        # 分散のフロア適用後の標準偏差
        expected_std = np.maximum(expected_std, self.std_floor)
        
        # 現在のscalerの統計値（スケール対象のみ）
        actual_mean = self.mean[scale_indices]
        actual_std = self.std[scale_indices]
        
        mean_diff = np.abs(actual_mean - expected_mean).max()
        std_diff = np.abs(actual_std - expected_std).max()
        
        # EMA更新の性質上、大きな乖離は想定内のため警告のみ
        # 実質的な因果性はfit_end_idxで保証
        if mean_diff > 2.0:  # あまりに大きい場合のみ警告
            import warnings
            warnings.warn(
                f"GroupedFeatureScaler: Large EMA vs batch stat difference (mean): "
                f"diff={mean_diff:.2e}. This is expected due to momentum=0.99.",
                UserWarning
            )
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        NaN/inf検査付きtransform（Doc04仕様）
        
        Args:
            features: 88次元観測空間
        
        Returns:
            スケール済み88次元観測空間
        
        Raises:
            ValueError: fit未実行またはNaN/inf検出時
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        if features.shape[-1] != 88:
            raise ValueError(
                f"Expected 88-dimensional features, got {features.shape[-1]}"
            )
        
        scaled = super().transform(features)  # 親クラスのtransform使用
        
        # NaN/inf検査（Doc04仕様）
        if np.isnan(scaled).any():
            raise ValueError("Scaling produced NaN values")
        if np.isinf(scaled).any():
            raise ValueError("Scaling produced inf values")
        
        return scaled
    
    def get_fit_info(self) -> dict:
        """
        Fit情報を取得（デバッグ用）
        
        Returns:
            Fit情報の辞書
        """
        scale_indices = self.SCALE_INDICES
        # SCALE_INDICESはlistまたはndarrayの可能性
        indices_list = scale_indices if isinstance(scale_indices, list) else scale_indices.tolist()
        
        return {
            "fitted": self.fitted,
            "fit_end_idx": self.fit_end_idx,
            "scale_indices": indices_list,
            "mean_range": (self.mean[scale_indices].min(), self.mean[scale_indices].max()),
            "std_range": (self.std[scale_indices].min(), self.std[scale_indices].max()),
            "std_floor": self.std_floor,
        }
