"""
Stochastic Oscillator implementation.
Stochastic Oscillatorの実装
"""

import numpy as np
import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Stochastic")
def compute_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3) -> pd.Series:
    """Compute Stochastic Oscillator"""
    # Calculate rolling max and min if not present
    if f'rolling_max_{period}' not in df.columns:
        rolling_max = df['high'].rolling(window=period).max()
    else:
        rolling_max = df[f'rolling_max_{period}']

    if f'rolling_min_{period}' not in df.columns:
        rolling_min = df['low'].rolling(window=period).min()
    else:
        rolling_min = df[f'rolling_min_{period}']

    denominator = rolling_max - rolling_min
    stoch_k = np.where(
        denominator != 0,
        100 * (df['close'] - rolling_min) / denominator,
        50  # neutral value when denominator is zero
    )

    # Apply smoothing to %K to get %D
    stoch_d = pd.Series(stoch_k, index=df.index).rolling(smooth_k).mean()

    # Return %D (smoothed %K) as it's more commonly used
    return stoch_d.fillna(50)