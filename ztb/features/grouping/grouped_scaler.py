"""
Grouped Feature Scaler for v456

選別されたOnlineZScore正規化を実装:
- 対象: Base features [0:30] + Global continuous [63:69] = 36 features
- 非対象: MTF [30:57] + Cyclical [57:63] + Regime [69:82] + Account [82:88]

C-2対応版: グループ化されたスケーラーで正規化戦略を統一
"""

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)

class GroupedFeatureScaler:
    """
    v456用グループ化特徴量スケーラー
    
    88次元の観測空間を分割して、選別的に正規化:
    
    スケーリング対象 (36):
    - Base (30): indices [0:30]
    - Global continuous (6): indices [63:69]
    
    スケーリング非対象 (52):
    - MTF (27): indices [30:57] (多重共線性, 複合指標)
    - Cyclical time (6): indices [57:63] (sin/cos, 既に [-1,1])
    - Regime (13): indices [69:82] (One-Hot, 分類的)
    - Account (3): indices [82:88] (正規化済み, [0,1])
    
    OnlineZScoreを使用して、訓練中にストリーミングデータから
    平均と標準偏差を更新する。
    """
    
    # 固定インデックス範囲
    SCALE_GROUPS = {
        'base': (0, 30),              # Base features
        'global_continuous': (63, 69),  # 連続値グローバル特徴量
    }
    
    NO_SCALE_GROUPS = {
        'mtf': (30, 57),              # Multi-timeframe features
        'cyclical_time': (57, 63),    # Cyclical time features (sin/cos)
        'regime': (69, 82),           # Regime features (One-Hot)
        'account': (82, 88),          # Account normalization (pre-normalized)
    }
    
    TOTAL_FEATURES = 88
    SCALE_INDICES = (
        list(range(SCALE_GROUPS['base'][0], SCALE_GROUPS['base'][1])) +
        list(range(SCALE_GROUPS['global_continuous'][0], SCALE_GROUPS['global_continuous'][1]))
    )
    
    def __init__(
        self,
        epsilon: float = 1e-7,
        momentum: float = 0.99,  # Exponential moving average
        clip_value: float = 3.0,  # Clipping threshold for outliers
    ):
        """
        Args:
            epsilon: 数値安定性のためのスモール値
            momentum: EMA更新用の運動量 (0.99 = 99%過去, 1%新規)
            clip_value: 正規化後のクリッピング閾値 (σ単位)
        """
        self.epsilon = epsilon
        self.momentum = momentum
        self.clip_value = clip_value
        
        # 統計量の初期化
        self.mean = np.zeros(self.TOTAL_FEATURES, dtype=np.float32)
        self.std = np.ones(self.TOTAL_FEATURES, dtype=np.float32)
        self.n_samples = 0
        
        # スケーリング対象かどうか
        self.scale_mask = np.zeros(self.TOTAL_FEATURES, dtype=bool)
        self.scale_mask[self.SCALE_INDICES] = True
        
        self._initialized = False
    
    def fit_one(self, features: np.ndarray) -> None:
        """
        単一バッチで統計量を更新（オンライン学習）
        
        Args:
            features: shape (88,) の特徴量配列
        """
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        
        if features.shape[0] != self.TOTAL_FEATURES:
            raise ValueError(
                f"Feature dimension mismatch: {features.shape[0]} != {self.TOTAL_FEATURES}"
            )
        
        # スケール対象のみを更新
        scale_features = features[self.scale_mask]
        
        if not self._initialized:
            # 最初の初期化
            self.mean[self.scale_mask] = scale_features
            self.std[self.scale_mask] = 1.0
            self._initialized = True
        else:
            # Exponential Moving Average (EMA) 更新
            self.mean[self.scale_mask] = (
                self.momentum * self.mean[self.scale_mask] +
                (1.0 - self.momentum) * scale_features
            )
            
            # 分散更新（簡易版: オンライン分散）
            # より正確には、Welford's onlineアルゴリズムを使用
            variance = np.var(scale_features)
            old_var = (self.std[self.scale_mask] ** 2)
            new_var = (
                self.momentum * old_var +
                (1.0 - self.momentum) * variance
            )
            self.std[self.scale_mask] = np.sqrt(new_var + self.epsilon)
        
        self.n_samples += 1
    
    def fit_batch(self, features_batch: np.ndarray) -> None:
        """
        バッチデータで統計量を更新
        
        Args:
            features_batch: shape (batch_size, 88) の特徴量配列
        """
        if not isinstance(features_batch, np.ndarray):
            features_batch = np.asarray(features_batch, dtype=np.float32)
        
        if features_batch.ndim != 2:
            raise ValueError(f"Expected 2D array, got {features_batch.ndim}D")
        
        if features_batch.shape[1] != self.TOTAL_FEATURES:
            raise ValueError(
                f"Feature dimension mismatch: {features_batch.shape[1]} != {self.TOTAL_FEATURES}"
            )
        
        # 各サンプルを更新
        for features in features_batch:
            self.fit_one(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        特徴量をスケーリング
        
        Args:
            features: shape (88,) または (batch_size, 88) の配列
        
        Returns:
            正規化された特徴量 (同じ形状)
        """
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        
        original_shape = features.shape
        
        # 2Dの場合
        if features.ndim == 2:
            batch_features = features.copy()
        elif features.ndim == 1:
            batch_features = features[np.newaxis, :].copy()
        else:
            raise ValueError(f"Expected 1D or 2D array, got {features.ndim}D")
        
        # スケーリング適用
        scale_indices = self.scale_mask
        batch_features[:, scale_indices] = (
            (batch_features[:, scale_indices] - self.mean[scale_indices]) /
            (self.std[scale_indices] + self.epsilon)
        )
        
        # Outlier clipping
        batch_features[:, scale_indices] = np.clip(
            batch_features[:, scale_indices],
            -self.clip_value,
            self.clip_value
        )
        
        # 元の形状に戻す
        if original_shape[0] == 1 and len(original_shape) == 1:
            return batch_features[0]
        else:
            return batch_features.reshape(original_shape)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        fit と transform を一度に実行
        
        Args:
            features: shape (88,) または (batch_size, 88)
        
        Returns:
            正規化された特徴量
        """
        if features.ndim == 1:
            self.fit_one(features)
        else:
            self.fit_batch(features)
        
        return self.transform(features)
    
    def get_stats(self) -> dict:
        """
        現在の統計量を取得
        
        Returns:
            mean, std, n_samples を含む辞書
        """
        return {
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'n_samples': self.n_samples,
            'scale_indices': self.SCALE_INDICES,
            'num_scaled': len(self.SCALE_INDICES),
            'num_not_scaled': self.TOTAL_FEATURES - len(self.SCALE_INDICES),
        }
    
    def reset(self) -> None:
        """統計量をリセット"""
        self.mean = np.zeros(self.TOTAL_FEATURES, dtype=np.float32)
        self.std = np.ones(self.TOTAL_FEATURES, dtype=np.float32)
        self.n_samples = 0
        self._initialized = False
    
    def validate_feature_structure(self) -> bool:
        """
        特徴量構造が正しいことを検証
        
        Returns:
            全チェック通過なら True
        """
        checks = [
            (len(self.SCALE_INDICES) == 36, "Scale indices should be 36"),
            (
                set(range(*self.SCALE_GROUPS['base'])) | 
                set(range(*self.SCALE_GROUPS['global_continuous'])) ==
                set(self.SCALE_INDICES),
                "Scale indices should match groups"
            ),
            (self.TOTAL_FEATURES == 88, "Total features should be 88"),
        ]
        
        for check, message in checks:
            if not check:
                logger.error(f"Validation failed: {message}")
                return False
        
        return True
    
    @staticmethod
    def get_feature_groups_description() -> dict:
        """特徴量グループの説明を取得"""
        return {
            'scaled': {
                'base': 'Base OHLCV derivatives (30)',
                'global_continuous': 'Global market continuous features (6)',
                'total_scaled': 36,
            },
            'not_scaled': {
                'mtf': 'Multi-timeframe features (27)',
                'cyclical_time': 'Cyclical time features - sin/cos (6)',
                'regime': 'Regime features - One-Hot encoded (13)',
                'account': 'Account metrics - pre-normalized (3)',
                'total_not_scaled': 52,
            },
            'total_features': 88,
        }

# 使用例とテストコード
if __name__ == "__main__":
    # サンプル特徴量生成
    np.random.seed(42)
    
    scaler = GroupedFeatureScaler()
    
    # 構造検証
    print(f"Validation: {scaler.validate_feature_structure()}")
    print(f"Feature groups: {scaler.get_feature_groups_description()}")
    
    # シングルサンプル fit_transform
    single_features = np.random.randn(88).astype(np.float32) + np.arange(88)
    scaled_single = scaler.fit_transform(single_features)
    
    print(f"\nSingle sample transform:")
    print(f"  Original shape: {single_features.shape}")
    print(f"  Scaled shape: {scaled_single.shape}")
    print(f"  Base mean: {scaled_single[0:5]}")
    print(f"  MTF mean (unscaled): {scaled_single[30:35]}")
    
    # バッチ処理
    batch_features = np.random.randn(32, 88).astype(np.float32)
    scaler.reset()
    scaled_batch = scaler.fit_transform(batch_features)
    
    print(f"\nBatch transform:")
    print(f"  Batch shape: {scaled_batch.shape}")
    print(f"  Samples: {scaler.n_samples}")
    
    # 統計量確認
    stats = scaler.get_stats()
    print(f"\nScaler stats:")
    print(f"  n_samples: {stats['n_samples']}")
    print(f"  scaled features: {stats['num_scaled']}")
    print(f"  unscaled features: {stats['num_not_scaled']}")
