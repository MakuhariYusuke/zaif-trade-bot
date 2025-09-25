import numpy as np
import pandas as pd
from ..base import BaseFeature

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
        else:
            ha_open = np.empty(0)
        ha_df["ha_open"] = ha_open
        ha_df["ha_high"] = np.maximum(df["high"].values, np.maximum(ha_df["ha_open"].values, ha_df["ha_close"].values))
        ha_df["ha_low"] = np.minimum(df["low"].values, np.minimum(ha_df["ha_open"].values, ha_df["ha_close"].values))

        return ha_df