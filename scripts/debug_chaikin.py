import numpy as np
import pandas as pd

from ztb.features.volume.chaikin_ad_oscillator import compute_chaikin_ad_oscillator

np.random.seed(42)
n = 200
close = np.random.uniform(100, 200, n)
high = close + np.random.uniform(0, 10, n)
low = close - np.random.uniform(0, 10, n)
volume = np.random.uniform(1000, 5000, n)
dates = pd.date_range("2023-01-01", periods=n, freq="D")
df = pd.DataFrame(
    {"high": high, "low": low, "close": close, "volume": volume}, index=dates
)

chaikin_ad_osc = compute_chaikin_ad_oscillator(df)
print(chaikin_ad_osc.head(20))
print(
    "min:",
    chaikin_ad_osc.min(),
    "max:",
    chaikin_ad_osc.max(),
    "mean:",
    chaikin_ad_osc.mean(),
)
