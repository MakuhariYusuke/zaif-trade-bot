"""
Global Market Features for v456 (Extended)

9特徴量を生成:
- 6連続値: spread, return_1m, return_5m, vol_1m, vol_ratio, usdt_premium
- 3フラグ: spread_flag, return_flag, stale_flag

重要: 
- 連続値6個は OnlineScaler対象 (idx [63:69])
- フラグ3個は スケーリング不可 (idx [69:72])
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class GlobalMarketFeatureEngineerV456:
    """
    v456用グローバル市場特徴量エンジニア
    
    9特徴量の生成と管理:
    - spread: local-global spread (bps)
    - return_1m: 1分リターン相関
    - return_5m: 5分リターン相関
    - vol_1m: ボラティリティ（1分）
    - vol_ratio: ローカル/グローバルボラティリティ比
    - usdt_premium: USDTプレミアム（FX調整）
    - spread_flag: スプレッド異常フラグ
    - return_flag: リターン乖離フラグ
    - stale_flag: データ鮮度フラグ
    """
    
    FEATURE_COUNT = 9
    CONTINUOUS_COUNT = 6  # idx [63:69]
    FLAG_COUNT = 3        # idx [69:72]
    
    FEATURE_NAMES_CONTINUOUS = [
        'global_spread',        # bps
        'global_return_1m',     # %
        'global_return_5m',     # %
        'global_vol_1m',        # ATR
        'global_vol_ratio',     # local/global
        'global_usdt_premium',  # %
    ]
    
    FEATURE_NAMES_FLAGS = [
        'global_flag_spread',   # 0/1
        'global_flag_return',   # 0/1
        'global_stale_flag',    # 0/1
    ]
    
    FEATURE_NAMES = FEATURE_NAMES_CONTINUOUS + FEATURE_NAMES_FLAGS
    
    def __init__(
        self,
        binance_df: Optional[pd.DataFrame] = None,
        usdjpy_rate: float = 155.0,  # 例値
        max_data_age_minutes: int = 5,
    ):
        """
        Args:
            binance_df: Binance BTC/USDTデータ (tz-aware index)
            usdjpy_rate: USD/JPY レート (FX調整用)
            max_data_age_minutes: データ鮮度判定の閾値
        """
        self.binance_df = binance_df if binance_df is not None else pd.DataFrame()
        self.usdjpy_rate = usdjpy_rate
        self.max_data_age_minutes = max_data_age_minutes
        
        # キャッシュ
        self._last_computed_time = None
        self._cached_features = None
    
    def generate_features(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """
        グローバル市場特徴量を生成 (9次元)
        
        Args:
            local_df: ローカル市場データ (Zaif BTC/JPYなど)
            current_timestamp: 現在時刻
        
        Returns:
            shape (9,) の特徴量配列 [6連続 + 3フラグ]
        """
        features = np.zeros(self.FEATURE_COUNT, dtype=np.float32)
        
        # 1. スプレッド計算
        spread_bps = self._compute_spread(local_df, current_timestamp)
        features[0] = spread_bps
        
        # 2. リターン特徴量
        return_1m = self._compute_return(local_df, current_timestamp, window=1)
        return_5m = self._compute_return(local_df, current_timestamp, window=5)
        features[1] = return_1m
        features[2] = return_5m
        
        # 3. ボラティリティ特徴量
        vol_1m = self._compute_volatility(local_df, current_timestamp, window=1)
        vol_ratio = self._compute_vol_ratio(local_df, current_timestamp)
        features[3] = vol_1m
        features[4] = vol_ratio
        
        # 4. FX調整（USDT プレミアム）
        usdt_premium = self._compute_usdt_premium(local_df, current_timestamp)
        features[5] = usdt_premium
        
        # 5-7. フラグ特徴量
        spread_flag = self._compute_spread_flag(spread_bps)
        return_flag = self._compute_return_flag(return_1m, return_5m)
        stale_flag = self._compute_stale_flag(local_df, current_timestamp)
        
        features[6] = float(spread_flag)
        features[7] = float(return_flag)
        features[8] = float(stale_flag)
        
        return features
    
    def _compute_spread(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> float:
        """
        ローカル-グローバル スプレッド (bps)
        
        Returns:
            スプレッド (basis points), -inf～+inf
        """
        if local_df.empty or self.binance_df.empty:
            return 0.0
        
        # 現在の価格を取得
        local_price = self._get_last_price(local_df, current_timestamp)
        global_price = self._get_last_price(self.binance_df, current_timestamp)
        
        if local_price <= 0 or global_price <= 0:
            return 0.0
        
        # FX調整: Zaif (JPY) → USD に換算
        local_price_usd = local_price / self.usdjpy_rate
        
        # スプレッド (bps) = (local - global) / global * 10000
        spread_bps = (local_price_usd - global_price) / global_price * 10000
        
        return float(np.clip(spread_bps, -1000, 1000))  # 異常値クリップ
    
    def _compute_return(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
        window: int = 1,
    ) -> float:
        """
        リターン (%)
        
        Args:
            window: 計算ウィンドウ (分)
        
        Returns:
            リターン (%) [-100, +100]
        """
        if local_df.empty or len(local_df) < window + 1:
            return 0.0
        
        # 過去windowの価格データ取得
        close_prices = local_df['close'].tail(window + 1).values
        
        if len(close_prices) < 2 or close_prices[-2] <= 0:
            return 0.0
        
        ret = (close_prices[-1] / close_prices[-2] - 1) * 100
        
        return float(np.clip(ret, -100, 100))
    
    def _compute_volatility(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
        window: int = 1,
    ) -> float:
        """
        ボラティリティ (ATR正規化)
        
        Args:
            window: 計算ウィンドウ (分)
        
        Returns:
            ボラティリティ [0, 10] (正規化済み)
        """
        if local_df.empty or len(local_df) < window + 1:
            return 0.0
        
        # ATR計算（簡易版）
        high = local_df['high'].tail(window + 1).max()
        low = local_df['low'].tail(window + 1).min()
        close = local_df['close'].iloc[-1]
        
        if close <= 0:
            return 0.0
        
        atr = (high - low) / close * 100  # %
        
        return float(np.clip(atr, 0, 10))
    
    def _compute_vol_ratio(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> float:
        """
        ボラティリティ比 (local / global)
        
        Returns:
            比率 (0.1～10)
        """
        if local_df.empty or self.binance_df.empty:
            return 1.0
        
        local_vol = self._compute_volatility(local_df, current_timestamp, window=5)
        global_vol = self._compute_volatility(self.binance_df, current_timestamp, window=5)
        
        if global_vol < 0.001:
            return 1.0
        
        ratio = local_vol / global_vol
        
        return float(np.clip(ratio, 0.1, 10.0))
    
    def _compute_usdt_premium(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> float:
        """
        USDT プレミアム（FX調整値）
        
        Returns:
            プレミアム (%) [-10, +10]
        """
        # 簡易版：FXレートの乖離を示す
        # 実装では実際のUSDT価格データを使用
        
        # プレースホルダー
        premium = 0.0  # 実装例：(usdt_price - 1.0) * 100
        
        return float(np.clip(premium, -10, 10))
    
    def _compute_spread_flag(self, spread_bps: float) -> bool:
        """
        スプレッド異常フラグ
        
        Returns:
            スプレッド > 50bps なら True
        """
        return abs(spread_bps) > 50
    
    def _compute_return_flag(self, return_1m: float, return_5m: float) -> bool:
        """
        リターン乖離フラグ
        
        Returns:
            return_5m > 1.0% かつ return_1m < 0 なら True
        """
        return (return_5m > 1.0) and (return_1m < 0)
    
    def _compute_stale_flag(
        self,
        local_df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> bool:
        """
        データ鮮度フラグ
        
        Returns:
            最新データが古い（max_data_age_minutesを超過）なら True
        """
        if local_df.empty:
            return True  # データなし = 陳腐
        
        last_timestamp = local_df.index[-1]
        
        # tz-aware チェック
        if last_timestamp.tzinfo is None or current_timestamp.tzinfo is None:
            return True
        
        age_minutes = (current_timestamp - last_timestamp).total_seconds() / 60
        
        return age_minutes > self.max_data_age_minutes
    
    @staticmethod
    def _get_last_price(
        df: pd.DataFrame,
        current_timestamp: pd.Timestamp,
    ) -> float:
        """
        DataFrame から現在時刻時点の最新価格を取得
        
        Returns:
            価格, またはデータなしで 0.0
        """
        if df.empty:
            return 0.0
        
        # 現在時刻以前のデータを取得
        mask = df.index <= current_timestamp
        if not mask.any():
            return 0.0
        
        close = df.loc[mask, 'close'].iloc[-1] if 'close' in df.columns else 0.0
        
        return float(close) if not pd.isna(close) else 0.0
    
    def handle_stale_global_features(
        self,
        features: np.ndarray,
        stale_threshold_minutes: int = 5,
    ) -> np.ndarray:
        """
        陳腐なグローバル特徴量を0で埋める
        
        Args:
            features: shape (9,) の特徴量配列
            stale_threshold_minutes: 陳腐判定の閾値
        
        Returns:
            修正された特徴量配列
        """
        features = features.copy()
        
        # スタールフラグが立っている場合
        if features[8] > 0.5:  # stale_flag
            # 連続値を0で埋める
            features[0:6] = 0.0
            # フラグはそのまま
        
        return features
    
    @classmethod
    def validate_feature_count(cls) -> bool:
        """検証: 特徴量数が正しいことを確認"""
        count = len(cls.FEATURE_NAMES)
        expected = 9
        
        if count != expected:
            logger.error(f"Feature count mismatch: {count} != {expected}")
            return False
        
        if len(cls.FEATURE_NAMES_CONTINUOUS) != 6:
            logger.error(f"Continuous features mismatch")
            return False
        
        if len(cls.FEATURE_NAMES_FLAGS) != 3:
            logger.error(f"Flag features mismatch")
            return False
        
        return True


# 使用例
if __name__ == "__main__":
    # サンプルデータ
    dates = pd.date_range('2025-01-10 10:00', periods=30, freq='1min', tz='UTC')
    
    local_df = pd.DataFrame({
        'open': np.random.randn(30).cumsum() + 9000,
        'high': np.random.randn(30).cumsum() + 9100,
        'low': np.random.randn(30).cumsum() + 8900,
        'close': np.random.randn(30).cumsum() + 9000,
        'volume': np.random.randint(1000, 10000, 30),
    }, index=dates)
    
    binance_df = pd.DataFrame({
        'open': np.random.randn(30).cumsum() + 58000,
        'high': np.random.randn(30).cumsum() + 58100,
        'low': np.random.randn(30).cumsum() + 57900,
        'close': np.random.randn(30).cumsum() + 58000,
        'volume': np.random.randint(10000, 100000, 30),
    }, index=dates)
    
    # エンジニア初期化
    engineer = GlobalMarketFeatureEngineerV456(binance_df=binance_df)
    
    # 特徴量生成
    current = dates[-1]
    features = engineer.generate_features(local_df, current)
    
    print(f"Generated {len(features)} features:")
    for name, value in zip(engineer.FEATURE_NAMES, features):
        print(f"  {name}: {value:.3f}")
    
    # 検証
    print(f"\nValidation: {engineer.validate_feature_count()}")
