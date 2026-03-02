"""
HFT / Fast Intraday Proxy Features
1分足OHLCVからマイクロストラクチャ（板情報など）の代替指標を生成するモジュール。
"""

import numpy as np
import pandas as pd

def add_hft_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add HFT proxy features to the DataFrame.
    
    Features:
    - clv (Close Location Value)
    - vol_pressure (Signed Volume Pressure)
    - impact_proxy (Price Range)
    - vol_regime (ATR / Close)
    - trend_persistence (EMA divergence)
    
    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        
    Returns:
        DataFrame with added features.
    """
    df = df.copy()
    
    # Ensure lowercase columns
    df.columns = [c.lower() for c in df.columns]
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    epsilon = 1e-8
    
    # 1. Close Location Value (CLV)
    # Range: [-1, 1]. Positive = closes near high (buy pressure).
    # (close - low) / (high - low) scales to [0, 1]. * 2 - 1 scales to [-1, 1].
    df['clv'] = ((close - low) / (high - low + epsilon)) * 2 - 1
    
    # 2. Signed Volume Pressure
    # Proxy for aggressive flow without tick data.
    df['vol_pressure'] = df['clv'] * volume
    
    # 3. Impact Proxy (Price Range)
    # Changed to (High - Low) to represent the "Spread" or "Price Range" in JPY/BTC.
    # Used for Linear Slippage calculation: Cost = Impact * Delta.
    # Unit: JPY/BTC.
    # Clip to 0 to prevent negative slippage (rebate) in case of data glitches (High < Low).
    df['impact_proxy'] = (high - low).clip(lower=0.0)
    
    # 4. Volatility Regime (ATR / Close)
    # ATR(14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean() # Simple rolling mean for ATR approximation or use EMA
    # Standard ATR uses RMA (Wilder's smoothing), but simple rolling is often sufficient for proxies.
    # Let's use ewm for smoother ATR if possible, or just rolling.
    # Using rolling(14) is safe and simple.
    
    df['atr'] = atr
    df['vol_regime'] = atr / (close + epsilon)
    
    # 5. Trend Persistence (EMA divergence)
    # EMA(5) - EMA(20)
    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    
    # Normalized by ATR to make it unitless (Trend Strength)
    df['trend_persistence'] = (ema5 - ema20) / (atr + epsilon)
    
    # Fill NaNs created by rolling/shifting
    # Use bfill to propagate first valid values backwards, then fillna(0) for any remaining
    df = df.bfill().fillna(0)
    
    return df
