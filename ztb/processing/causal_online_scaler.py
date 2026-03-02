"""
v459 Phase 0.2c: 因果性保証付きScaler
Doc04仕様準拠: fit範囲管理、リーク検査、ゼロ分散対応
"""

import numpy as np
import pandas as pd

from ztb.processing.online_scaler import OnlineScaler

class CausalOnlineScaler(OnlineScaler):
    """
    Doc04仕様に準拠した因果性保証付きOnlineScaler
    
    - fit範囲をend_idxで記録
    - リーク検査機能
    - ゼロ分散対応（std_floor）
    - NaN/inf検査
    """
    
    def __init__(self, shape, epsilon=1e-5, clip=10.0, std_floor=1e-8):
        """
        Args:
            shape: Feature shape
            epsilon: ゼロ除算防止（親クラス）
            clip: クリッピング閾値（親クラス）
            std_floor: 標準偏差のフロア値（ゼロ分散対応）
        """
        super().__init__(shape, epsilon, clip)
        self.std_floor = std_floor
        self.fit_end_idx: int | None = None
        self.fitted = False
    
    def fit(self, data: pd.DataFrame, end_idx: int, feature_names: list[str]):
        """
        Train期間のみでfit（因果性保証）
        
        Args:
            data: 全データ
            end_idx: Train終端index（この行まで含む）
            feature_names: 対象特徴量名
        
        Raises:
            ValueError: データ不正またはリーク検出時
        """
        if end_idx >= len(data):
            raise ValueError(
                f"end_idx ({end_idx}) must be less than data length ({len(data)})"
            )
        
        # Train期間のみ抽出
        train_data = data.iloc[:end_idx + 1][feature_names].values
        
        if len(train_data) == 0:
            raise ValueError("Train data is empty")
        
        # バッチ更新（親クラスのupdateメソッド使用）
        for row in train_data:
            self.update(row)
        
        # ゼロ分散対応（Doc04仕様）
        self.var = np.maximum(self.var, self.std_floor ** 2)
        
        self.fitted = True
        self.fit_end_idx = end_idx
        
        # リーク検査
        self._verify_no_leakage(data, end_idx, feature_names)
    
    def _verify_no_leakage(self, data: pd.DataFrame, end_idx: int, feature_names: list[str]):
        """
        Val/Testデータの混入を検査
        
        Args:
            data: 全データ
            end_idx: Train終端index
            feature_names: 対象特徴量名
        
        Raises:
            ValueError: リーク検出時
        """
        if len(data) <= end_idx + 1:
            # Val/Testデータが存在しない場合はスキップ
            return
        
        # Train期間の統計を再計算して一致確認
        train_data = data.iloc[:end_idx + 1][feature_names].values
        expected_mean = train_data.mean(axis=0)
        expected_std = train_data.std(axis=0, ddof=1)
        
        # 分散のフロア適用後の標準偏差
        expected_std = np.maximum(expected_std, self.std_floor)
        
        mean_diff = np.abs(self.mean - expected_mean).max()
        std_actual = np.sqrt(self.var)
        std_diff = np.abs(std_actual - expected_std).max()
        
        # 許容誤差: float32精度を考慮
        tolerance = 1e-5
        
        if mean_diff > tolerance:
            raise ValueError(
                f"Scaler leakage detected (mean): diff={mean_diff:.2e} > {tolerance:.2e}"
            )
        if std_diff > tolerance:
            raise ValueError(
                f"Scaler leakage detected (std): diff={std_diff:.2e} > {tolerance:.2e}"
            )
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform with NaN/inf check（Doc04仕様）
        
        Args:
            x: Input features
        
        Returns:
            Scaled features
        
        Raises:
            ValueError: fit未実行またはNaN/inf検出時
        """
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        scaled = super().transform(x)  # 親クラスのtransform使用
        
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
        return {
            "fitted": self.fitted,
            "fit_end_idx": self.fit_end_idx,
            "n_samples": self.n,
            "mean_range": (self.mean.min(), self.mean.max()),
            "std_range": (np.sqrt(self.var).min(), np.sqrt(self.var).max()),
            "std_floor": self.std_floor,
        }
