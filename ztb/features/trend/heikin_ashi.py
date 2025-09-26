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
        Args:
            df (pd.DataFrame): Input DataFrame. Must include columns:
                - 'open' (float): Open price
                - 'high' (float): High price
                - 'low' (float): Low price
                - 'close' (float): Close price

        Example:
            pd.DataFrame({
                'open': [100.0, 101.0, ...],
                'high': [102.0, 103.0, ...],
                'low': [99.0, 100.5, ...],
                'close': [101.5, 102.0, ...]
            })

        Returns:
            pd.DataFrame: DataFrame with Heikin-Ashi OHLC columns:
                - ha_open
                - ha_high
                - ha_low
                - ha_close
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
        ha_df["ha_high"] = np.maximum.reduce([
            df["high"].to_numpy(),
            ha_df["ha_open"].to_numpy(),
            ha_df["ha_close"].to_numpy()
        ])
        ha_df["ha_low"] = np.minimum.reduce([
            df["low"].to_numpy(),
            ha_df["ha_open"].to_numpy(),
            ha_df["ha_close"].to_numpy()
        ])

        return ha_df