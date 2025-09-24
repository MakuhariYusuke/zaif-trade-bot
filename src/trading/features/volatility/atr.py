import numpy as np
import pandas as pd
from ..base import BaseFeature

class ATRSimplified(BaseFeature):
    """Simplified Average True Range"""

    def __init__(self, period: int = 14):
        super().__init__("ATR_simplified", deps=["high", "low", "close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)
        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            )
        )
        atr = pd.Series(tr).rolling(period).mean()
        return pd.DataFrame({'ATR_simplified': atr.fillna(0)})