import numpy as np
import pandas as pd
from ..base import BaseFeature

class RSI(BaseFeature):
    """Relative Strength Index"""

    def __init__(self, period: int = 14):
        super().__init__("RSI", deps=["return"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)
        delta = df['return']
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return pd.DataFrame({'RSI': rsi.fillna(50)})