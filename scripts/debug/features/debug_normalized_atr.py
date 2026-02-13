import numpy as np
import pandas as pd

from ztb.features.volatility.normalized_atr import compute_normalized_atr
from ztb.utils.constants import DEFAULT_SEED

np.random.seed(DEFAULT_SEED)
n = 200
close = np.random.uniform(100, 200, n)
high = close + np.random.uniform(0, 10, n)
low = close - np.random.uniform(0, 10, n)
volume = np.random.uniform(1000, 5000, n)
dates = pd.date_range("2023-01-01", periods=n, freq="D")
df = pd.DataFrame(
    {"high": high, "low": low, "close": close, "volume": volume}, index=dates
)

normalized_atr = compute_normalized_atr(df)
print(normalized_atr.head(20))
print(
    "min:",
    normalized_atr.min(),
    "max:",
    normalized_atr.max(),
    "mean:",
    normalized_atr.mean(),
)
