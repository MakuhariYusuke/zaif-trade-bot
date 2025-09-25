import numpy as np
import pandas as pd
from ..base import BaseFeature

class ROC(BaseFeature):
    """Rate of Change"""

    def __init__(self, period: int = 10):
        super().__init__("ROC", deps=["close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)
        roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return pd.DataFrame({'ROC': roc})