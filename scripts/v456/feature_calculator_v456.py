#!/usr/bin/env python3
"""
v456 MTF（Multi-Timeframe）特徴量計算

1分足 OHLCV から、実データベースの MTF 特徴量（27次元）を計算
現在のランダム生成特徴量を置き換える

実装: RSI, MACD, Bollinger Bands, ATR, ADX など
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MTFFeatureCalculator:
    """Multi-Timeframe 特徴量計算"""
    
    def __init__(self, window_sizes: List[int] = [5, 15, 60]):
        """
        Args:
            window_sizes: ウィンドウサイズ（分単位）
                例: [5, 15, 60] = 5分, 15分, 1時間
        """
        self.window_sizes = window_sizes
    
    @staticmethod
    def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI（相対力指数）を計算"""
        if len(close) < period + 1:
            return np.zeros(len(close))
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD を計算（line, signal, histogram）"""
        if len(close) < slow:
            return (np.zeros(len(close)), np.zeros(len(close)), np.zeros(len(close)))
        
        ema_fast = MTFFeatureCalculator._calculate_ema(close, fast)
        ema_slow = MTFFeatureCalculator._calculate_ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = MTFFeatureCalculator._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
        """EMA（指数加重移動平均）を計算"""
        if len(data) < period:
            return np.zeros(len(data))
        
        ema = np.zeros(len(data))
        multiplier = 2 / (period + 1)
        
        ema[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema
    
    @staticmethod
    def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ボリンジャーバンドを計算
        
        Returns:
            (upper, middle, lower, pct_b)
        """
        if len(close) < period:
            return (np.zeros(len(close)), np.zeros(len(close)), 
                    np.zeros(len(close)), np.zeros(len(close)))
        
        sma = np.convolve(close, np.ones(period) / period, mode='same')
        
        std = np.zeros(len(close))
        for i in range(period - 1, len(close)):
            std[i] = np.std(close[i - period + 1:i + 1])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        # %B（Percent B）= (close - lower) / (upper - lower)
        pct_b = np.where(upper != lower, (close - lower) / (upper - lower), 0.5)
        pct_b = np.clip(pct_b, 0, 1)
        
        return upper, sma, lower, pct_b
    
    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """ATR（平均真の値）を計算"""
        if len(high) < period:
            return np.zeros(len(high))
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1])
            )
        
        atr = np.zeros(len(high))
        atr[period - 1] = np.mean(tr[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
        return atr
    
    @staticmethod
    def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """ADX（平均方向指数）を計算"""
        if len(high) < period * 2:
            return np.zeros(len(high))
        
        # DM（方向の動き）を計算
        up_move = np.diff(high)
        down_move = -np.diff(low)
        
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        
        for i in range(len(high) - 1):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i + 1] = up_move[i]
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i + 1] = down_move[i]
        
        # ATR
        atr = MTFFeatureCalculator.calculate_atr(high, low, close, period)
        
        # DI（方向指数）
        plus_di = 100 * plus_dm / atr
        minus_di = 100 * minus_dm / atr
        
        # DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.nan_to_num(dx, 0)
        
        # ADX（DXの平均）
        adx = np.zeros(len(high))
        adx[period * 2 - 1] = np.mean(dx[period:period * 2])
        
        for i in range(period * 2, len(high)):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
        
        return adx
    
    @staticmethod
    def calculate_momentum_features(close: np.ndarray, periods: List[int] = [5, 10, 20]) -> np.ndarray:
        """モメンタム特徴量（複数期間のリターン）"""
        features = np.zeros((len(close), len(periods)))
        
        for idx, period in enumerate(periods):
            if len(close) >= period:
                # パーセンテージ変化
                features[period:, idx] = (close[period:] - close[:-period]) / close[:-period] * 100
        
        return features
    
    def calculate_all_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全 MTF 特徴量を計算し、DataFrame に追加
        
        計27次元の特徴量を生成
        """
        if len(df) == 0:
            return df
        
        df_out = df.copy()
        close = df['close'].values.astype(np.float32)
        high = df.get('high', df['close']).values.astype(np.float32)
        low = df.get('low', df['close']).values.astype(np.float32)
        
        feature_idx = 0
        mtf_features = np.zeros((len(df), 27))
        
        # 1. RSI（5, 14, 21）
        for period, offset in [(5, 0), (14, 1), (21, 2)]:
            mtf_features[:, offset] = self.calculate_rsi(close, period)
        feature_idx = 3
        
        # 2. MACD（line, signal, histogram）
        macd_line, signal_line, histogram = self.calculate_macd(close)
        mtf_features[:, feature_idx:feature_idx+3] = np.column_stack([macd_line, signal_line, histogram])
        feature_idx += 3  # 6
        
        # 3. Bollinger Bands（upper, lower, %B, bandwidth）
        upper, middle, lower, pct_b = self.calculate_bollinger_bands(close, 20)
        bandwidth = (upper - lower) / middle
        bandwidth = np.nan_to_num(bandwidth, 0)
        mtf_features[:, feature_idx:feature_idx+4] = np.column_stack([upper, lower, pct_b, bandwidth])
        feature_idx += 4  # 10
        
        # 4. ATR（14）
        atr = self.calculate_atr(high, low, close, 14)
        mtf_features[:, feature_idx] = atr
        feature_idx += 1  # 11
        
        # 5. NATR（ATR 正規化）
        natr = atr / close
        natr = np.nan_to_num(natr, 0)
        mtf_features[:, feature_idx] = natr
        feature_idx += 1  # 12
        
        # 6. ADX（14）
        adx = self.calculate_adx(high, low, close, 14)
        mtf_features[:, feature_idx] = adx
        feature_idx += 1  # 13
        
        # 7. モメンタム（5, 10, 20）
        momentum = self.calculate_momentum_features(close, [5, 10, 20])
        mtf_features[:, feature_idx:feature_idx+3] = momentum
        feature_idx += 3  # 16
        
        # 8. ボラティリティ（複数窓）
        volatility = np.zeros((len(df), 3))
        for i, period in enumerate([5, 10, 20]):
            if len(close) >= period:
                returns = np.diff(np.log(close))
                vol = np.zeros(len(close))
                vol[period:] = [np.std(returns[j-period:j]) for j in range(period, len(returns) + 1)]
                volatility[:, i] = vol
        mtf_features[:, feature_idx:feature_idx+3] = volatility
        feature_idx += 3  # 19
        
        # 9. Volume Profile Proxy（ボラティリティとリターンの組み合わせ）
        profile = np.zeros((len(df), 8))
        for i in range(len(df)):
            if i > 0:
                ret = (close[i] - close[i-1]) / close[i-1]
                vol_proxy = np.std(close[max(0, i-20):i+1]) if i >= 20 else 0
                profile[i, 0] = ret  # 直近リターン
                profile[i, 1] = vol_proxy  # ボラティリティ
                profile[i, 2] = ret * vol_proxy  # インタラクション
                profile[i, 3] = abs(ret)  # 絶対リターン
                profile[i, 4] = close[i]  # 直近価格（正規化は後で）
                profile[i, 5] = high[i] - low[i]  # 日中レンジ
                profile[i, 6] = (close[i] - low[i]) / (high[i] - low[i]) if high[i] != low[i] else 0.5  # Close位置
                profile[i, 7] = np.std(close[max(0, i-5):i+1]) / np.mean(close[max(0, i-5):i+1]) if i >= 5 else 0  # CV
        
        mtf_features[:, feature_idx:feature_idx+8] = profile
        feature_idx += 8  # 27
        
        # DataFrame に追加
        for i in range(27):
            col_name = f'mtf_{i}'
            df_out[col_name] = mtf_features[:, i]
        
        logger.info(f"✓ MTF特徴量計算完了: {27}次元 @ {len(df)} rows")
        
        return df_out


class RegimeFeatureCalculator:
    """Market Regime（レジーム）特徴量計算 - 13次元"""
    
    @staticmethod
    def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        レジーム特徴量を計算（13次元）
        
        Contents:
        1-3: Volatility Regime (Low/Mid/High flags)
        4-6: Trend Direction (Up/Down/Sideways)
        7-9: Volume Regime (Low/Mid/High)
        10-13: Support/Resistance Detection
        """
        df_out = df.copy()
        regime_features = np.zeros((len(df), 13))
        
        close = df['close'].values.astype(np.float32)
        
        # 1. Volatility Regime（20期間のボラティリティで判定）
        vol_window = 20
        volatility = np.zeros(len(close))
        for i in range(vol_window, len(close)):
            volatility[i] = np.std(close[i-vol_window:i])
        
        vol_low_threshold = np.percentile(volatility[vol_window:], 33)
        vol_high_threshold = np.percentile(volatility[vol_window:], 67)
        
        regime_features[volatility < vol_low_threshold, 0] = 1  # Low
        regime_features[(volatility >= vol_low_threshold) & (volatility < vol_high_threshold), 1] = 1  # Mid
        regime_features[volatility >= vol_high_threshold, 2] = 1  # High
        
        # 2. Trend Direction（50期間SMAで判定）
        sma_period = 50
        sma = np.convolve(close, np.ones(sma_period) / sma_period, mode='same')
        
        regime_features[close > sma, 3] = 1  # Up
        regime_features[close < sma, 4] = 1  # Down
        regime_features[close == sma, 5] = 1  # Sideways
        
        # 3. Volume Regime（オプション: Volume データがある場合）
        if 'volume' in df.columns:
            volume = df['volume'].values.astype(np.float32)
            vol_low = np.percentile(volume, 33)
            vol_high = np.percentile(volume, 67)
            
            regime_features[volume < vol_low, 6] = 1  # Low
            regime_features[(volume >= vol_low) & (volume < vol_high), 7] = 1  # Mid
            regime_features[volume >= vol_high, 8] = 1  # High
        else:
            # Volume がない場合は全て Mid
            regime_features[:, 7] = 1
        
        # 4. Support/Resistance Detection（簡易版）
        window = 20
        for i in range(window, len(close)):
            local_high = np.max(close[i-window:i])
            local_low = np.min(close[i-window:i])
            mid = (local_high + local_low) / 2
            
            if close[i] > mid * 1.02:  # 上側
                regime_features[i, 9] = 1
            elif close[i] < mid * 0.98:  # 下側
                regime_features[i, 10] = 1
            else:  # 中央
                regime_features[i, 11] = 1
            
            # 抵抗線に接近
            if abs(close[i] - local_high) < volatility[i]:
                regime_features[i, 12] = 1
        
        # DataFrame に追加
        for i in range(13):
            col_name = f'regime_{i}'
            df_out[col_name] = regime_features[:, i]
        
        logger.info(f"✓ Regime特徴量計算完了: 13次元 @ {len(df)} rows")
        
        return df_out


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """すべての特徴量を計算（MTF + Regime）"""
    
    logger.info("特徴量計算開始...")
    
    # MTF特徴量
    mtf_calc = MTFFeatureCalculator()
    df = mtf_calc.calculate_all_mtf_features(df)
    
    # Regime特徴量
    df = RegimeFeatureCalculator.calculate_regime_features(df)
    
    logger.info("✓ すべての特徴量計算完了（27 + 13 = 40次元）")
    
    return df


if __name__ == '__main__':
    # テスト
    logging.basicConfig(level=logging.INFO)
    
    # ダミーデータ作成
    n = 1000
    df_test = pd.DataFrame({
        'open': 100 + np.random.randn(n).cumsum() * 0.5,
        'high': 101 + np.random.randn(n).cumsum() * 0.5,
        'low': 99 + np.random.randn(n).cumsum() * 0.5,
        'close': 100 + np.random.randn(n).cumsum() * 0.5,
        'volume': np.random.randint(1000, 5000, n),
    })
    
    # 計算
    df_result = calculate_all_features(df_test)
    
    print(f"\n計算結果:")
    print(f"  入力列: {len(df_test.columns)}")
    print(f"  出力列: {len(df_result.columns)}")
    print(f"  新規追加: {len(df_result.columns) - len(df_test.columns)}")
    print(f"\n新規列:")
    for col in df_result.columns:
        if col not in df_test.columns:
            print(f"  - {col}")
