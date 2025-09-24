"""
wave4.py
各種テクニカル指標の特徴量実装

このモジュールは、トレーディング戦略のための高度なテクニカル分析機能を実装しています。
平均足、Supertrend、EMA/SMA マルチ期間、MACD、ATR、ボリンジャーバンド幅、HV、RSI、Stochastic (SlowK/SlowD)
CCI (Commodity Channel Index)、OBV (On-Balance Volume)、VWAP、リターンの移動平均 (短期/中期)、リターンの標準偏差 (ボラティリティの簡易代理)
価格の勾配 sign（±1）、移動相関 (価格 vs 出来高)などの特徴量が含まれています。
各特徴量はBaseFeatureを継承したクラスとして実装されており、pandas DataFrameから指標値を生成するcomputeメソッドを提供します。
これらの特徴量は、トレーディングボットの特徴量エンジニアリングパイプラインの一部として使用されることを意図しています。
"""
import numpy as np
import pandas as pd
from numba import jit
from .base import BaseFeature

class HeikinAshi(BaseFeature):
    """
    平均足 (Heikin-Ashi) feature implementation.
    Generates smoothed OHLC values to capture trend strength.

    Output columns:
      - ha_open
      - ha_high
      - ha_low
      - ha_close
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['open', 'high', 'low', 'close'].
        Returns a DataFrame with Heikin-Ashi OHLC.
        """
        ha_df = pd.DataFrame(index=df.index)

        # Heikin-Ashi close
        ha_df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

        # Heikin-Ashi open
        ha_open = np.empty(len(df))
        if len(df) > 0:
            ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
            ha_close = ha_df["ha_close"].to_numpy()
            for i in range(1, len(df)):
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_df["ha_open"] = ha_open
                ha_open[i] = (ha_open[i - 1] + ha_close_prev[i - 1]) / 2.0
        else:
            ha_open = np.empty(0)
        ha_df["ha_open"] = ha_open
        ha_df["ha_high"] = np.maximum(df["high"].values, np.maximum(ha_df["ha_open"].values, ha_df["ha_close"].values))
        ha_df["ha_low"] = np.minimum(df["low"].values, np.minimum(ha_df["ha_open"].values, ha_df["ha_close"].values))

        return ha_df

class Supertrend(BaseFeature):
    """
    Supertrend feature implementation.
    Identifies trend direction and potential reversals.

    Parameters:
      - period: ATR calculation period (default=10)
        - multiplier: ATR multiplier for band calculation (default=3.0)
    Output columns:
      - supertrend
      - supertrend_direction (1 for uptrend, -1 for downtrend)
    """
    def __init__(self, period: int = 10, multiplier: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.multiplier = multiplier
        self.atr_col = f'atr_{self.period}'
        self.prev_supertrend = 0.0
        self.prev_direction = 0.0
        self.prev_final_upperband = 0.0
        self.prev_final_lowerband = 0.0
        self.initialized = False
        self.epsilon = 1e-8  # To prevent division by zero
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """ 
        df columns must include: ['high', 'low', 'close'] and ATR column.
        Returns a DataFrame with Supertrend values.
        """
        if self.atr_col not in df.columns:
            raise ValueError(f"ATR column '{self.atr_col}' not found in DataFrame. Please compute ATR first.")

        supertrend_df = pd.DataFrame(index=df.index)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = df[self.atr_col].values

        supertrend, direction = self._compute_supertrend(
            high, low, close, atr, self.multiplier,
            self.prev_supertrend, self.prev_direction,
            self.prev_final_upperband, self.prev_final_lowerband,
            self.initialized, self.epsilon
        )

        supertrend_df['supertrend'] = supertrend
        supertrend_df['supertrend_direction'] = direction

        # Update previous state for next computation
        if len(supertrend) > 0:
            self.prev_supertrend = supertrend[-1]
            self.prev_direction = direction[-1]
            # NumbaのJITコンパイルのために、Noneではなく具体的な値を保持する必要があります。
            # しかし、ロジック上は次の計算で使われないため、更新は不要かもしれません。
            # ここではロジックを維持しつつ、Noneの代わりにnp.nanを使用します。
            if direction[-1] == 1:
                self.prev_final_lowerband = supertrend[-1]
                self.prev_final_upperband = np.nan
            else:
                self.prev_final_upperband = supertrend[-1]
                self.prev_final_lowerband = np.nan
            self.initialized = True

        return supertrend_df

    @staticmethod
    @jit(nopython=True)
    def _compute_supertrend(
        high, low, close, atr, multiplier,
        prev_supertrend, prev_direction,
        prev_final_upperband, prev_final_lowerband,
        initialized, epsilon
    ):
        n = len(close)
        supertrend = np.full(n, np.nan)
        direction = np.zeros(n)

        for i in range(n):
            if i == 0:
                # 初期化
                final_upperband = high[i] + multiplier * atr[i]
                final_lowerband = low[i] - multiplier * atr[i]
                supertrend[i] = final_upperband
                direction[i] = -1  # 初期はダウントレンドと仮定
                continue

            # 基本バンドの計算
            basic_upperband = high[i] + multiplier * atr[i]
            basic_lowerband = low[i] - multiplier * atr[i]

            # 最終バンドの計算
            if initialized:
                final_upperband = min(basic_upperband, prev_final_upperband) if prev_direction == -1 else basic_upperband
                final_lowerband = max(basic_lowerband, prev_final_lowerband) if prev_direction == 1 else basic_lowerband
            else:
                final_upperband = basic_upperband
                final_lowerband = basic_lowerband

            # Supertrendの決定
            if close[i] > final_upperband:
                supertrend[i] = final_lowerband
                direction[i] = 1  # アップトレンド
            elif close[i] < final_lowerband:
                supertrend[i] = final_upperband
                direction[i] = -1  # ダウントレンド
            else:
                supertrend[i] = prev_supertrend
                direction[i] = prev_direction

            # 状態の更新
            prev_supertrend = supertrend[i]
            prev_direction = direction[i]
            prev_final_upperband = final_upperband
            prev_final_lowerband = final_lowerband
            initialized = True

        return supertrend, direction
    
class MultiEMA(BaseFeature):
    """
    Multi-period Exponential Moving Averages (EMA) feature implementation.
    Computes EMAs for multiple periods to capture trends at different time scales.

    Parameters:
      - periods: List of EMA periods (default=[5, 10, 20, 50]) 
    Output columns:
      - ema_{period} for each period in periods
    """
    def __init__(self, periods: list = [5, 10, 20, 50], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with EMA columns for each specified period.
        """
        ema_df = pd.DataFrame(index=df.index)
        for period in self.periods:
            ema_col = f'ema_{period}'
            ema_df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()
        return ema_df
    
class MultiSMA(BaseFeature):
    """
    Multi-period Simple Moving Averages (SMA) feature implementation.
    Computes SMAs for multiple periods to capture trends at different time scales.

    Parameters:
      - periods: List of SMA periods (default=[5, 10, 20, 50]) 
    Output columns:
      - sma_{period} for each period in periods
    """
    def __init__(self, periods: list = [5, 10, 20, 50], **kwargs):
        super().__init__(**kwargs)
        self.periods = periods
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with SMA columns for each specified period.
        """
        sma_df = pd.DataFrame(index=df.index)
        for period in self.periods:
            sma_col = f'sma_{period}'
            sma_df[sma_col] = df['close'].rolling(window=period).mean()
        return sma_df

class MACDFeature(BaseFeature):
    """
    MACD (Moving Average Convergence Divergence) feature implementation.
    Computes MACD line, signal line, and histogram.

    Parameters:
      - fast_period: Fast EMA period (default=12)
        - slow_period: Slow EMA period (default=26)
        - signal_period: Signal line EMA period (default=9)
    Output columns:
        - macd
        - macd_signal
        - macd_hist
    """
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with MACD, signal line, and histogram.
        """
        macd_df = pd.DataFrame(index=df.index)
        ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        macd_hist = macd_line - signal_line

        macd_df['macd'] = macd_line
        macd_df['macd_signal'] = signal_line
        macd_df['macd_hist'] = macd_hist

        return macd_df
    
class ATRFeature(BaseFeature):
    """
    ATR (Average True Range) feature implementation.
    Measures market volatility.

    Parameters:
      - period: ATR calculation period (default=14)
    Output columns:
      - atr_{period}
    """
    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with ATR values.
        """
        atr_df = pd.DataFrame(index=df.index)
        high = df['high']
        low = df['low']
        close = df['close']

        # True Rangeの計算
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATRの計算 (EMAを使用)
        atr = true_range.ewm(span=self.period, adjust=False).mean()
        atr_df[f'atr_{self.period}'] = atr

        return atr_df
    

class BollingerBandWidth(BaseFeature):
    """
    Bollinger Band Width feature implementation.
    Measures the width of Bollinger Bands to assess volatility.
    Parameters:
      - period: SMA period for Bollinger Bands (default=20)
        - num_std: Number of standard deviations for the bands (default=2)
    Output columns:
        - bb_width
    """
    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.num_std = num_std
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with Bollinger Band Width values.
        """
        bb_df = pd.DataFrame(index=df.index)
        sma = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        bb_width = (upper_band - lower_band) / (sma + 1e-8)  # Avoid division by zero
        bb_df['bb_width'] = bb_width
        return bb_df
    
class HVFeature(BaseFeature):
    """
    Historical Volatility (HV) feature implementation.
    Measures the volatility of price returns over a specified period.

    Parameters:
      - period: HV calculation period (default=14)
    Output columns:
      - hv_{period}
    """
    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with Historical Volatility values.
        """
        hv_df = pd.DataFrame(index=df.index)
        log_returns = np.log(df['close'] / df['close'].shift(1))
        hv = log_returns.rolling(window=self.period).std() * np.sqrt(252)  # Annualized volatility
        hv_df[f'hv_{self.period}'] = hv
        return hv_df
    
class RSI(BaseFeature):
    """
    RSI (Relative Strength Index) feature implementation.
    Measures the speed and change of price movements.

    Parameters:
      - period: RSI calculation period (default=14)
    Output columns:
      - rsi_{period}
    """
    def __init__(self, period: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with RSI values.
        """
        rsi_df = pd.DataFrame(index=df.index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        rsi_df[f'rsi_{self.period}'] = rsi
        return rsi_df
    
class Stochastic(BaseFeature):
    """
    Stochastic Oscillator feature implementation.
    Measures the momentum of price movements.
    Parameters:
      - k_period: %K period (default=14)
        - d_period: %D period (default=3)
    Output columns:
        - slowk
        - slowd (3-period SMA of slowk)
    """
    def __init__(self, k_period: int = 14, d_period: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k_period = k_period
        self.d_period = d_period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Stochastic Oscillator values.
        """
        stoch_df = pd.DataFrame(index=df.index)
        low_min = df['low'].rolling(window=self.k_period).min()
        high_max = df['high'].rolling(window=self.k_period).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)  # Avoid division by zero
        stoch_d = stoch_k.rolling(window=self.d_period).mean()

        stoch_df['slowk'] = stoch_k
        stoch_df['slowd'] = stoch_d

        return stoch_df
    
class CCI(BaseFeature):
    """
    CCI (Commodity Channel Index) feature implementation.
    Measures the deviation of price from its average.

    Parameters:
      - period: CCI calculation period (default=20)
    Output columns:
        - cci_{period}
    """
    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with CCI values.
        """
        def typical_price(row):
            return (row['high'] + row['low'] + row['close']) / 3.0
        cci_df = pd.DataFrame(index=df.index)
        tp = df.apply(typical_price, axis=1)
        sma_tp = tp.rolling(window=self.period).mean()
        def mad_func(x):
            return np.mean(np.abs(x - np.mean(x)))
        mad = tp.rolling(window=self.period).apply(mad_func, raw=True)
        cci = (tp - sma_tp) / (0.015 * mad + 1e-8)  # Avoid division by zero
        cci_df[f'cci_{self.period}'] = cci
        return cci_df
    
class OBV(BaseFeature):
    """
    OBV (On-Balance Volume) feature implementation.
    Measures buying and selling pressure as a cumulative indicator.
    Output columns:
      - obv
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close', 'volume'].
        Returns a DataFrame with OBV values.
        """
        obv_df = pd.DataFrame(index=df.index)
        direction = np.sign(df['close'].diff().fillna(0))
        obv = (direction * df['volume']).fillna(0).cumsum()
        obv_df['obv'] = obv
        return obv_df
    
class VWAP(BaseFeature):
    """
    VWAP (Volume Weighted Average Price) feature implementation.
    Provides the average price weighted by volume.

    Output columns:
      - vwap
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close', 'volume'].
        Returns a DataFrame with VWAP values.
        """
        vwap_df = pd.DataFrame(index=df.index)
        cum_vol_price = (df['close'] * df['volume']).cumsum()
        cum_volume = df['volume'].cumsum() + 1e-8  # Avoid division by zero
        vwap = cum_vol_price / cum_volume
        vwap_df['vwap'] = vwap
        return vwap_df

class ReturnMA(BaseFeature):
    """
    Moving Averages of Returns feature implementation.
    Computes short-term and medium-term moving averages of returns.
    Parameters:
      - short_period: Short-term MA period (default=5)
        - medium_period: Medium-term MA period (default=20)
    Output columns:
        - return_ma_short
        - return_ma_medium 
    """
    def __init__(self, short_period: int = 5, medium_period: int = 20, **kwargs):
        super().__init__(**kwargs)  
        self.short_period = short_period
        self.medium_period = medium_period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with moving averages of returns.
        """
        ret_ma_df = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        ret_ma_df['return_ma_short'] = returns.rolling(window=self.short_period).mean()
        ret_ma_df['return_ma_medium'] = returns.rolling(window=self.medium_period).mean()
        return ret_ma_df
    
class ReturnStdDev(BaseFeature):
    """
    Standard Deviation of Returns feature implementation.
    Measures the volatility of returns over a specified period.

    Parameters:
      - period: StdDev calculation period (default=20)
    Output columns:
      - return_stddev_{period}
    """
    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with standard deviation of returns.
        """
        ret_std_df = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        ret_std = returns.rolling(window=self.period).std()
        ret_std_df[f'return_stddev_{self.period}'] = ret_std
        return ret_std_df
    
class PriceGradientSign(BaseFeature):
    """
    Price Gradient Sign feature implementation.
    Indicates the direction of price movement.

    Output columns:
      - price_gradient_sign (1 for up, -1 for down, 0 for no change)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with price gradient sign.
        """
        grad_df = pd.DataFrame(index=df.index)
        grad = np.sign(df['close'].diff().fillna(0))
        grad_df['price_gradient_sign'] = grad
        return grad_df
    
class PriceVolumeCorr(BaseFeature):
    """
    Price-Volume Correlation feature implementation.
    Measures the rolling correlation between price and volume.
    Parameters:
      - period: Correlation calculation period (default=20)
      Output columns:
        - price_volume_corr_{period}
    """
    def __init__(self, period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.period = period
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close', 'volume'].
        Returns a DataFrame with price-volume correlation.
        """
        corr_df = pd.DataFrame(index=df.index)
        price = df['close']
        volume = df['volume']
        corr = price.rolling(window=self.period).corr(volume)
        corr_df[f'price_volume_corr_{self.period}'] = corr
        return corr_df
