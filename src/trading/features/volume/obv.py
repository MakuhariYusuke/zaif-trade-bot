import numpy as np
import pandas as pd
from ..base import BaseFeature

class OBV(BaseFeature):
    """
    OBV (On-Balance Volume) feature implementation.
    Measures buying and selling pressure as a cumulative indicator.
    Output columns:
      - obv
    """
    def __init__(self, **kwargs):
        super().__init__("OBV", deps=["close", "volume"])
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